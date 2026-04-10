"""Utilities to split large transcriptions into LLM-friendly chunks.

When a transcript set is too large to fit in a single LLM context window
the curation engine uses these helpers to:
1. Estimate how many tokens a piece of text will consume (character heuristic).
2. Format a Transcription as a plain-text block for the LLM.
3. Check whether all transcriptions fit in a single context window.
4. Split transcriptions into groups (chunks) that each fit within the budget.

These utilities are intentionally separate from the engine so they can be
unit-tested without mocking any LLM client.
"""

from __future__ import annotations

import math
from typing import List

from snipsnap.models import Transcription

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rough heuristic: ~4 characters per token (conservative estimate)
_CHARS_PER_TOKEN: int = 4

# Default token budget for a single LLM context window (leaves room for
# output and system prompt).
DEFAULT_MAX_TOKENS: int = 80_000

# Fixed overhead reserved for the system prompt, user message wrapper,
# and response buffer.
_PROMPT_OVERHEAD_TOKENS: int = 4_000


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text* using a character heuristic.

    Uses the conservative rule of thumb: 1 token ≈ 4 characters.

    Args:
        text: The text to estimate token count for.

    Returns:
        Estimated token count (always >= 0).
    """
    if not text:
        return 0
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def transcription_to_text(transcription: Transcription) -> str:
    """Format a :class:`~snipsnap.models.Transcription` as plain text for LLM input.

    Produces a human-readable block that includes the source file path,
    total duration, and every segment with its timestamps and text.

    Args:
        transcription: The transcription to format.

    Returns:
        A multi-line string suitable for inclusion in an LLM prompt.
    """
    lines: List[str] = [
        f"File: {transcription.source_file}",
        f"Duration: {transcription.duration:.1f}s",
        "",
    ]
    for seg in transcription.segments:
        lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
    return "\n".join(lines)


def fits_in_context(
    transcriptions: List[Transcription],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> bool:
    """Return ``True`` if all transcriptions fit within a single LLM context.

    Args:
        transcriptions: List of transcriptions to check.
        max_tokens: Maximum token budget for the context window (including
            overhead for prompts and expected response).

    Returns:
        ``True`` if the estimated total token count is within the budget.
    """
    budget = max_tokens - _PROMPT_OVERHEAD_TOKENS
    if budget <= 0:
        return False
    total_tokens = sum(estimate_tokens(transcription_to_text(t)) for t in transcriptions)
    return total_tokens <= budget


def chunk_transcriptions(
    transcriptions: List[Transcription],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[List[Transcription]]:
    """Split *transcriptions* into groups that each fit within *max_tokens*.

    Transcriptions are packed greedily in order: each new transcription is
    appended to the current chunk if it fits; otherwise a new chunk is started.
    A single transcription that exceeds the token budget is placed in its own
    chunk (it cannot be split further at this level).

    Args:
        transcriptions: Ordered list of transcriptions to partition.
        max_tokens: Maximum token budget per chunk.

    Returns:
        A list of chunks, where each chunk is a (non-empty) list of
        transcriptions.  Returns an empty list if *transcriptions* is empty.
    """
    if not transcriptions:
        return []

    budget = max_tokens - _PROMPT_OVERHEAD_TOKENS

    chunks: List[List[Transcription]] = []
    current_chunk: List[Transcription] = []
    current_tokens: int = 0

    for transcription in transcriptions:
        t_tokens = estimate_tokens(transcription_to_text(transcription))
        if current_chunk and current_tokens + t_tokens > budget:
            # Current chunk is full — start a new one
            chunks.append(current_chunk)
            current_chunk = [transcription]
            current_tokens = t_tokens
        else:
            current_chunk.append(transcription)
            current_tokens += t_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
