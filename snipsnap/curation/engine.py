"""LLM curation engine for SnipSnap.

The :class:`CurationEngine` analyses a set of video transcriptions and
produces a :class:`~snipsnap.models.CutList` using a configurable LLM via
the OpenAI SDK (pointed at OpenRouter).

Strategy
--------
* **Single-pass** (small sets): If all transcriptions fit in a single LLM
  context window, the engine skips directly to the final cut-list generation
  pass (1 API call).

* **Multi-pass** (large sets): When transcriptions exceed the context budget:
    1. Pass 1 – summarise each transcription individually.
    2. Pass 2 – identify which videos are relevant to the user's prompt.
    3. Pass 3 – produce the precise cut list using only the relevant videos.

Error handling
--------------
All LLM errors are translated into typed :class:`CurationError` subclasses
with user-friendly messages that **never** expose the API key value.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from snipsnap.curation.chunking import (
    DEFAULT_MAX_TOKENS,
    fits_in_context,
    transcription_to_text,
)
from snipsnap.models import CutList, CutSegment, Transcription
from snipsnap.storage import save_cut_list

logger = logging.getLogger(__name__)

# Default LLM model (OpenRouter model identifier)
DEFAULT_MODEL: str = "google/gemini-2.5-flash-lite"

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class CurationError(Exception):
    """Base class for all curation engine errors."""


class MissingApiKeyError(CurationError):
    """Raised when the OpenRouter API key is not configured."""


class AuthenticationError(CurationError):
    """Raised when the API key is rejected (HTTP 401)."""


class RateLimitError(CurationError):
    """Raised when the API rate limit is exceeded (HTTP 429)."""


class NetworkError(CurationError):
    """Raised when a network error prevents reaching the API."""


class MalformedResponseError(CurationError):
    """Raised when the LLM response cannot be parsed into a valid cut list."""


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(response: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from an LLM response string.

    Handles three common LLM output patterns:
    1. Pure JSON (``{"key": ...}``)
    2. JSON wrapped in a fenced code block (triple-backtick json blocks)
    3. JSON embedded in surrounding prose

    Args:
        response: Raw text returned by the LLM.

    Returns:
        Parsed dictionary.

    Raises:
        MalformedResponseError: If no valid JSON object can be found.
    """
    # 1. Try code-block extraction first (```json ... ``` or ``` ... ```)
    match = _CODE_BLOCK_RE.search(response)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # 2. Try parsing the entire response as JSON
    stripped = response.strip()
    try:
        return json.loads(stripped)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # 3. Try to find any JSON-like object within the response
    obj_match = _JSON_OBJECT_RE.search(response)
    if obj_match:
        try:
            return json.loads(obj_match.group())  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    raise MalformedResponseError(
        f"Could not extract valid JSON from LLM response. "
        f"First 200 chars: {response[:200]!r}"
    )


def _parse_cut_list_response(
    data: Dict[str, Any],
    prompt: str,
) -> tuple[str, list[CutSegment], float]:
    """Validate and extract fields from the parsed cut-list dictionary.

    Args:
        data: Parsed JSON dictionary from the LLM.
        prompt: The original user prompt (stored in the CutList).

    Returns:
        A tuple of ``(theme, segments, total_duration)``.

    Raises:
        MalformedResponseError: If required fields are missing or malformed.
    """
    if "segments" not in data:
        raise MalformedResponseError(
            "LLM response JSON is missing required 'segments' key. "
            f"Keys present: {list(data.keys())}"
        )

    theme: str = str(data.get("theme", prompt))
    raw_segments = data["segments"]

    if not isinstance(raw_segments, list):
        raise MalformedResponseError(
            f"Expected 'segments' to be a list, got {type(raw_segments).__name__}"
        )

    segments: List[CutSegment] = []
    for idx, seg_data in enumerate(raw_segments):
        try:
            segments.append(
                CutSegment(
                    source_file=str(seg_data["source_file"]),
                    start=float(seg_data["start"]),
                    end=float(seg_data["end"]),
                    description=str(seg_data.get("description", "")),
                    order=int(seg_data.get("order", idx)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedResponseError(
                f"Segment {idx} in LLM response is malformed: {exc}"
            ) from exc

    total_duration: float = float(sum(max(0.0, s.end - s.start) for s in segments))
    return theme, segments, total_duration


# ---------------------------------------------------------------------------
# Pass 3 prompt construction
# ---------------------------------------------------------------------------

_PASS3_SYSTEM = (
    "You are a professional video editor assistant. "
    "Analyse the provided video transcriptions and create a precise cut list "
    "based on the user's request. "
    "Return ONLY a JSON object with this exact structure:\n"
    '{\n'
    '  "theme": "brief description of the overall theme",\n'
    '  "segments": [\n'
    '    {\n'
    '      "source_file": "/exact/path/from/transcription",\n'
    '      "start": <start_seconds_as_float>,\n'
    '      "end": <end_seconds_as_float>,\n'
    '      "description": "what happens in this clip",\n'
    '      "order": <integer starting at 0>\n'
    '    }\n'
    '  ]\n'
    "}\n"
    "If no segments match the request, return an empty segments array. "
    "Return ONLY the JSON — no other text."
)

_PASS1_SYSTEM = (
    "You are a video content analyst. "
    "Briefly summarise the following video transcription in 2–3 sentences. "
    "Focus on the main topics and any notable timestamps."
)

_PASS2_SYSTEM = (
    "You are a video editor assistant. "
    "Given the following video summaries and the user's request, identify which "
    "videos contain relevant content. "
    "Return a JSON object: "
    '{"relevant_files": ["/path/to/file1.mp4", "/path/to/file2.mp4"]}'
    "\nReturn ONLY the JSON."
)


# ---------------------------------------------------------------------------
# CurationEngine
# ---------------------------------------------------------------------------


class CurationEngine:
    """LLM-powered engine that turns transcriptions into cut lists.

    Args:
        client: A pre-configured OpenAI SDK client pointing at OpenRouter
            (``base_url='https://openrouter.ai/api/v1'``).
        api_key: The OpenRouter API key.  Checked for presence before any
            API call is made so errors surface early with a clear message.
    """

    def __init__(self, client: Any, api_key: str) -> None:
        self._client = client
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def curate(
        self,
        transcriptions: List[Transcription],
        prompt: str,
        model: str = DEFAULT_MODEL,
        data_dir: Optional[Path] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> CutList:
        """Analyse *transcriptions* and produce a :class:`~snipsnap.models.CutList`.

        Implements a multi-pass chunking strategy:

        * If all transcriptions fit within *max_tokens*, sends them directly
          to the LLM in a single Pass 3 call.
        * Otherwise executes Pass 1 (summarise) → Pass 2 (identify relevant
          files) → Pass 3 (generate cut list).

        Args:
            transcriptions: Transcriptions to analyse.  Must be non-empty.
            prompt: User's curation instruction (e.g. "find funny moments").
            model: OpenRouter model identifier.
            data_dir: Override the storage data directory.  Defaults to the
                value from :func:`snipsnap.config.get_config`.
            max_tokens: Token budget used to decide between single-pass and
                multi-pass.  Mainly useful for testing.

        Returns:
            A saved :class:`~snipsnap.models.CutList`.

        Raises:
            MissingApiKeyError: If no API key is configured.
            AuthenticationError: If the API key is invalid (HTTP 401).
            RateLimitError: If the API rate limit is hit (HTTP 429).
            NetworkError: If a network error prevents reaching the API.
            MalformedResponseError: If the LLM response cannot be parsed.
        """
        self._check_api_key()

        if fits_in_context(transcriptions, max_tokens=max_tokens):
            logger.info("Using single-pass strategy (%d transcription(s))", len(transcriptions))
            theme, segments, total_duration = self._pass3(
                transcriptions, prompt, model
            )
        else:
            logger.info(
                "Using multi-pass strategy (%d transcription(s) exceed context budget)",
                len(transcriptions),
            )
            theme, segments, total_duration = self._multi_pass(
                transcriptions, prompt, model
            )

        # ------------------------------------------------------------------
        # Validate segments against input transcriptions
        # ------------------------------------------------------------------
        if segments:
            original_count = len(segments)
            segments = self._validate_and_filter_segments(segments, transcriptions)
            if not segments:
                raise CurationError(
                    f"All {original_count} segment(s) returned by the LLM were invalid "
                    "(source files not found in transcriptions or timestamps out of bounds). "
                    "The LLM response cannot be used to build a cut list."
                )
            # Recalculate total_duration based on the validated (filtered) segments
            total_duration = float(sum(max(0.0, s.end - s.start) for s in segments))

        cut_list = CutList(
            id=str(uuid.uuid4()),
            prompt=prompt,
            theme=theme,
            created_at=datetime.now(timezone.utc),
            segments=segments,
            total_duration=total_duration,
        )

        save_cut_list(cut_list, data_dir)
        logger.info("Saved cut list %s with %d segment(s)", cut_list.id, len(segments))
        return cut_list

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_api_key(self) -> None:
        """Raise MissingApiKeyError if the API key is absent."""
        if not self._api_key or not self._api_key.strip():
            raise MissingApiKeyError(
                "OpenRouter API key is not configured. "
                "Set the OPENROUTER_API_KEY environment variable or add it to your .env file."
            )

    def _validate_and_filter_segments(
        self,
        segments: List[CutSegment],
        transcriptions: List[Transcription],
    ) -> List[CutSegment]:
        """Validate and filter *segments* against *transcriptions*.

        Each segment is checked for:
        * ``source_file`` existing in the set of input transcriptions.
        * Timestamps satisfying ``0 <= start < end <= transcription.duration``.

        Invalid segments are discarded with a warning log message.

        Args:
            segments: Candidate segments from the LLM response.
            transcriptions: Transcriptions that were provided to the LLM.

        Returns:
            List of segments that passed all validation checks.
        """
        transcription_map: dict[str, Transcription] = {
            t.source_file: t for t in transcriptions
        }

        valid: List[CutSegment] = []
        for seg in segments:
            # Validate source_file exists in input transcriptions
            if seg.source_file not in transcription_map:
                logger.warning(
                    "Discarding segment: source_file %r not found in input "
                    "transcriptions. Known files: %s",
                    seg.source_file,
                    list(transcription_map.keys()),
                )
                continue

            transcription = transcription_map[seg.source_file]

            # Validate timestamps: 0 <= start < end <= duration
            if not (seg.start >= 0 and seg.start < seg.end and seg.end <= transcription.duration):
                logger.warning(
                    "Discarding segment: invalid timestamps for %r "
                    "(start=%.3f, end=%.3f, duration=%.3f). "
                    "Required: 0 <= start < end <= duration.",
                    seg.source_file,
                    seg.start,
                    seg.end,
                    transcription.duration,
                )
                continue

            valid.append(seg)

        return valid

    def _call_llm(self, model: str, system: str, user: str) -> str:
        """Make a single chat-completion call and return the response text.

        Translates OpenAI SDK exceptions into typed CurationError subclasses.

        Args:
            model: OpenRouter model identifier.
            system: System prompt text.
            user: User message text.

        Returns:
            The LLM's response as a plain string.

        Raises:
            AuthenticationError: On HTTP 401.
            RateLimitError: On HTTP 429.
            NetworkError: On connection failure.
        """
        import openai  # Local import to keep module import lightweight

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return str(response.choices[0].message.content)

        except openai.AuthenticationError as exc:
            raise AuthenticationError(
                "Authentication failed: the OpenRouter API key is invalid or has been revoked. "
                "Please check your OPENROUTER_API_KEY configuration."
            ) from exc

        except openai.RateLimitError as exc:
            raise RateLimitError(
                "OpenRouter API rate limit exceeded. "
                "Please wait a moment before retrying, or check your usage limits."
            ) from exc

        except openai.APIConnectionError as exc:
            raise NetworkError(
                "Failed to connect to the OpenRouter API. "
                "Please check your network connection and try again."
            ) from exc

    def _pass3(
        self,
        transcriptions: List[Transcription],
        prompt: str,
        model: str,
    ) -> tuple[str, List[CutSegment], float]:
        """Execute Pass 3: generate a precise cut list from full transcripts.

        Args:
            transcriptions: The transcriptions to include (full text).
            prompt: User's curation instruction.
            model: OpenRouter model identifier.

        Returns:
            A ``(theme, segments, total_duration)`` tuple.

        Raises:
            MalformedResponseError: If the LLM response cannot be parsed.
        """
        transcripts_text = "\n\n---\n\n".join(
            transcription_to_text(t) for t in transcriptions
        )
        user_message = (
            f"VIDEO TRANSCRIPTIONS:\n\n{transcripts_text}\n\n"
            f"USER REQUEST: {prompt}"
        )
        raw = self._call_llm(model=model, system=_PASS3_SYSTEM, user=user_message)
        data = _extract_json(raw)
        return _parse_cut_list_response(data, prompt)

    def _summarise(self, transcription: Transcription, model: str) -> str:
        """Execute Pass 1 for a single transcription: return its summary."""
        user_message = transcription_to_text(transcription)
        return self._call_llm(model=model, system=_PASS1_SYSTEM, user=user_message)

    def _identify_relevant(
        self,
        transcriptions: List[Transcription],
        summaries: Dict[str, str],
        prompt: str,
        model: str,
    ) -> List[Transcription]:
        """Execute Pass 2: identify which videos are relevant to *prompt*.

        If the LLM response cannot be parsed, falls back to returning all
        *transcriptions* to ensure Pass 3 still runs.

        Args:
            transcriptions: All transcriptions (used as fallback).
            summaries: Mapping ``{source_file: summary_text}``.
            prompt: User's curation instruction.
            model: OpenRouter model identifier.

        Returns:
            Subset of *transcriptions* deemed relevant (or all of them on
            parse failure).
        """
        summaries_text = "\n\n".join(
            f"File: {sf}\nSummary: {summary}" for sf, summary in summaries.items()
        )
        user_message = (
            f"VIDEO SUMMARIES:\n\n{summaries_text}\n\n"
            f"USER REQUEST: {prompt}"
        )
        raw = self._call_llm(model=model, system=_PASS2_SYSTEM, user=user_message)

        # Try to parse the list of relevant files; fall back to all files on error
        try:
            data = _extract_json(raw)
            relevant_files = set(data.get("relevant_files", []))
            if relevant_files:
                filtered = [t for t in transcriptions if t.source_file in relevant_files]
                if filtered:
                    return filtered
        except (MalformedResponseError, KeyError, TypeError, AttributeError):
            logger.warning(
                "Could not parse relevant files from Pass 2 response; "
                "falling back to all transcriptions."
            )

        return transcriptions

    def _multi_pass(
        self,
        transcriptions: List[Transcription],
        prompt: str,
        model: str,
    ) -> tuple[str, List[CutSegment], float]:
        """Execute the full three-pass strategy.

        Pass 1: Summarise each transcription individually.
        Pass 2: Identify relevant files from the summaries.
        Pass 3: Generate precise cut list from relevant files.

        Args:
            transcriptions: All transcriptions to process.
            prompt: User's curation instruction.
            model: OpenRouter model identifier.

        Returns:
            A ``(theme, segments, total_duration)`` tuple.
        """
        # Pass 1: summarise each transcription
        summaries: Dict[str, str] = {}
        for transcription in transcriptions:
            logger.info("Pass 1: summarising %s", transcription.source_file)
            summaries[transcription.source_file] = self._summarise(transcription, model)

        # Pass 2: identify relevant files
        logger.info("Pass 2: identifying relevant files for prompt %r", prompt)
        relevant = self._identify_relevant(transcriptions, summaries, prompt, model)
        logger.info("Pass 2: %d/%d file(s) identified as relevant", len(relevant), len(transcriptions))

        # Pass 3: generate cut list from relevant files
        logger.info("Pass 3: generating cut list from %d relevant file(s)", len(relevant))
        return self._pass3(relevant, prompt, model)
