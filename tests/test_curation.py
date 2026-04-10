"""Tests for the LLM curation engine and chunking utilities.

Tests are split into:
1. Chunking utilities – estimate_tokens, transcription_to_text, fits_in_context, chunk_transcriptions
2. CurationEngine error handling – missing API key, 401, 429, network, malformed response
3. Single-pass shortcut – small transcript sets go directly to Pass 3
4. Multi-pass strategy – large transcript sets use Pass 1 → Pass 2 → Pass 3
5. Storage – cut list saved with unique UUID
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from snipsnap.curation.chunking import (
    DEFAULT_MAX_TOKENS,
    chunk_transcriptions,
    estimate_tokens,
    fits_in_context,
    transcription_to_text,
)
from snipsnap.curation.engine import (
    AuthenticationError,
    CurationEngine,
    CurationError,
    MalformedResponseError,
    MissingApiKeyError,
    NetworkError,
    RateLimitError,
)
from snipsnap.models import CutList, Segment, Transcription

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(content: str) -> MagicMock:
    """Return a mock openai ChatCompletion response with *content*."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _cut_list_json(
    theme: str = "Test theme",
    segments: list | None = None,
) -> str:
    """Return a JSON string representing a cut list LLM response."""
    return json.dumps(
        {
            "theme": theme,
            "segments": segments or [],
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def transcription_a() -> Transcription:
    return Transcription(
        source_file="/videos/video_a.mp4",
        duration=30.0,
        language="en",
        model_used="small",
        segments=[
            Segment(start=0.0, end=5.0, text="This is the beginning of video A"),
            Segment(start=5.0, end=15.0, text="This is the middle section of video A"),
            Segment(start=15.0, end=30.0, text="This is the ending of video A"),
        ],
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture()
def transcription_b() -> Transcription:
    return Transcription(
        source_file="/videos/video_b.mp4",
        duration=20.0,
        language="en",
        model_used="small",
        segments=[
            Segment(start=0.0, end=8.0, text="Introduction of video B"),
            Segment(start=8.0, end=20.0, text="Main content of video B"),
        ],
        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )


@pytest.fixture()
def mock_client() -> MagicMock:
    """A mock OpenAI client."""
    return MagicMock()


# ---------------------------------------------------------------------------
# 1. Chunking utilities
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        result = estimate_tokens("")
        assert result >= 0

    def test_single_word(self) -> None:
        result = estimate_tokens("Hello")
        assert result >= 1

    def test_longer_text_has_more_tokens(self) -> None:
        short = estimate_tokens("Hi")
        long_text = estimate_tokens("Hello this is a much longer piece of text with many words")
        assert long_text > short

    def test_proportional_to_length(self) -> None:
        base = estimate_tokens("a" * 100)
        doubled = estimate_tokens("a" * 200)
        assert doubled > base


class TestTranscriptionToText:
    def test_includes_source_file(self, transcription_a: Transcription) -> None:
        text = transcription_to_text(transcription_a)
        assert "/videos/video_a.mp4" in text

    def test_includes_duration(self, transcription_a: Transcription) -> None:
        text = transcription_to_text(transcription_a)
        assert "30" in text

    def test_includes_segment_text(self, transcription_a: Transcription) -> None:
        text = transcription_to_text(transcription_a)
        assert "beginning of video A" in text
        assert "middle section" in text

    def test_includes_timestamps(self, transcription_a: Transcription) -> None:
        text = transcription_to_text(transcription_a)
        # Timestamps appear in some form
        assert "0.0" in text or "0s" in text or "0:" in text

    def test_empty_segments_no_crash(self) -> None:
        t = Transcription(
            source_file="video.mp4",
            duration=10.0,
            language="en",
            model_used="small",
            segments=[],
        )
        text = transcription_to_text(t)
        assert "video.mp4" in text

    def test_returns_string(self, transcription_a: Transcription) -> None:
        assert isinstance(transcription_to_text(transcription_a), str)


class TestFitsInContext:
    def test_small_transcriptions_fit(
        self, transcription_a: Transcription, transcription_b: Transcription
    ) -> None:
        result = fits_in_context([transcription_a, transcription_b])
        assert result is True

    def test_empty_list_fits(self) -> None:
        assert fits_in_context([]) is True

    def test_impossibly_low_limit_does_not_fit(
        self, transcription_a: Transcription
    ) -> None:
        result = fits_in_context([transcription_a], max_tokens=1)
        assert result is False

    def test_returns_bool(self, transcription_a: Transcription) -> None:
        result = fits_in_context([transcription_a])
        assert isinstance(result, bool)


class TestChunkTranscriptions:
    def test_small_set_returns_single_chunk(
        self, transcription_a: Transcription, transcription_b: Transcription
    ) -> None:
        chunks = chunk_transcriptions([transcription_a, transcription_b])
        assert len(chunks) == 1
        assert len(chunks[0]) == 2
        assert transcription_a in chunks[0]
        assert transcription_b in chunks[0]

    def test_empty_list_returns_empty(self) -> None:
        assert chunk_transcriptions([]) == []

    def test_low_limit_splits_into_multiple_chunks(
        self, transcription_a: Transcription, transcription_b: Transcription
    ) -> None:
        chunks = chunk_transcriptions(
            [transcription_a, transcription_b],
            max_tokens=100,
        )
        assert len(chunks) >= 2

    def test_all_transcriptions_preserved_across_chunks(
        self, transcription_a: Transcription, transcription_b: Transcription
    ) -> None:
        for limit in [50, 100, 500, DEFAULT_MAX_TOKENS]:
            chunks = chunk_transcriptions(
                [transcription_a, transcription_b],
                max_tokens=limit,
            )
            all_in_chunks = [t for chunk in chunks for t in chunk]
            assert len(all_in_chunks) == 2
            assert transcription_a in all_in_chunks
            assert transcription_b in all_in_chunks

    def test_single_transcription_always_in_one_chunk(
        self, transcription_a: Transcription
    ) -> None:
        # Even with tiny limit, one transcription cannot be split further
        chunks = chunk_transcriptions([transcription_a], max_tokens=1)
        assert len(chunks) == 1
        assert chunks[0] == [transcription_a]


# ---------------------------------------------------------------------------
# 2. CurationEngine — error handling
# ---------------------------------------------------------------------------


class TestMissingApiKey:
    def test_raises_missing_api_key_error(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        engine = CurationEngine(client=mock_client, api_key="")
        with pytest.raises(MissingApiKeyError):
            engine.curate(
                [transcription_a],
                prompt="find interesting moments",
                data_dir=tmp_data_dir,
            )

    def test_no_llm_call_when_key_missing(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        engine = CurationEngine(client=mock_client, api_key="")
        with pytest.raises(MissingApiKeyError):
            engine.curate(
                [transcription_a],
                prompt="find anything",
                data_dir=tmp_data_dir,
            )
        mock_client.chat.completions.create.assert_not_called()

    def test_missing_api_key_is_curation_error(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        engine = CurationEngine(client=mock_client, api_key="")
        with pytest.raises(CurationError):
            engine.curate([transcription_a], prompt="test", data_dir=tmp_data_dir)


class TestAuthenticationError:
    def test_raises_authentication_error_on_401(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        import openai

        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body={"error": {"message": "Invalid API key"}},
        )
        engine = CurationEngine(client=mock_client, api_key="invalid-key-value")
        with pytest.raises(AuthenticationError):
            engine.curate(
                [transcription_a],
                prompt="find interesting moments",
                data_dir=tmp_data_dir,
            )

    def test_authentication_error_does_not_expose_key(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        import openai

        secret_key = "sk-or-secret-12345"
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body={"error": {"message": "Invalid API key"}},
        )
        engine = CurationEngine(client=mock_client, api_key=secret_key)
        with pytest.raises(AuthenticationError) as exc_info:
            engine.curate(
                [transcription_a],
                prompt="find interesting moments",
                data_dir=tmp_data_dir,
            )
        assert secret_key not in str(exc_info.value)

    def test_authentication_error_is_curation_error(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        import openai

        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body={"error": {"message": "Invalid API key"}},
        )
        engine = CurationEngine(client=mock_client, api_key="bad-key")
        with pytest.raises(CurationError):
            engine.curate([transcription_a], prompt="test", data_dir=tmp_data_dir)

    def test_authentication_error_message_is_user_friendly(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        import openai

        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body={"error": {"message": "Invalid API key"}},
        )
        engine = CurationEngine(client=mock_client, api_key="bad-key")
        with pytest.raises(AuthenticationError) as exc_info:
            engine.curate([transcription_a], prompt="test", data_dir=tmp_data_dir)
        msg = str(exc_info.value).lower()
        assert "authentication" in msg or "api key" in msg or "invalid" in msg


class TestRateLimitError:
    def test_raises_rate_limit_error_on_429(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        import openai

        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        with pytest.raises(RateLimitError):
            engine.curate(
                [transcription_a],
                prompt="find interesting moments",
                data_dir=tmp_data_dir,
            )

    def test_rate_limit_error_message_suggests_retry(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        import openai

        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        with pytest.raises(RateLimitError) as exc_info:
            engine.curate([transcription_a], prompt="test", data_dir=tmp_data_dir)
        msg = str(exc_info.value).lower()
        assert "rate limit" in msg or "retry" in msg

    def test_rate_limit_error_is_curation_error(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        import openai

        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        with pytest.raises(CurationError):
            engine.curate([transcription_a], prompt="test", data_dir=tmp_data_dir)


class TestNetworkError:
    def test_raises_network_error_on_connection_failure(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        import openai

        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        with pytest.raises(NetworkError):
            engine.curate(
                [transcription_a],
                prompt="find interesting moments",
                data_dir=tmp_data_dir,
            )

    def test_network_error_is_curation_error(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        import openai

        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        with pytest.raises(CurationError):
            engine.curate([transcription_a], prompt="test", data_dir=tmp_data_dir)


class TestMalformedResponse:
    def test_raises_malformed_response_on_non_json(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "This is just plain text, not JSON at all."
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        with pytest.raises(MalformedResponseError):
            engine.curate(
                [transcription_a],
                prompt="find interesting moments",
                data_dir=tmp_data_dir,
            )

    def test_raises_malformed_response_on_missing_segments_key(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        # JSON, but missing required 'segments' key
        mock_client.chat.completions.create.return_value = _make_mock_response(
            json.dumps({"theme": "ok", "something_else": []})
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        with pytest.raises(MalformedResponseError):
            engine.curate(
                [transcription_a],
                prompt="find interesting moments",
                data_dir=tmp_data_dir,
            )

    def test_malformed_response_is_curation_error(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response("not json")
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        with pytest.raises(CurationError):
            engine.curate([transcription_a], prompt="test", data_dir=tmp_data_dir)

    def test_json_in_code_block_is_parsed(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        """LLM wraps JSON in ```json ... ``` — must still parse correctly."""
        content = '```json\n{"theme": "Wrapped", "segments": []}\n```'
        mock_client.chat.completions.create.return_value = _make_mock_response(content)
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="test", data_dir=tmp_data_dir
        )
        assert result.theme == "Wrapped"

    def test_json_in_backtick_block_without_language_tag(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        content = '```\n{"theme": "NoTag", "segments": []}\n```'
        mock_client.chat.completions.create.return_value = _make_mock_response(content)
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="test", data_dir=tmp_data_dir
        )
        assert result.theme == "NoTag"


# ---------------------------------------------------------------------------
# 3. Single-pass shortcut
# ---------------------------------------------------------------------------


class TestSinglePass:
    def test_produces_cut_list_instance(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="find moments", data_dir=tmp_data_dir
        )
        assert isinstance(result, CutList)

    def test_result_has_all_required_fields(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json(theme="My Theme")
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="find moments", data_dir=tmp_data_dir
        )
        assert result.id
        assert result.prompt == "find moments"
        assert result.theme == "My Theme"
        assert isinstance(result.total_duration, float)
        assert isinstance(result.segments, list)
        assert result.created_at is not None

    def test_cut_list_with_segments(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        segments_data = [
            {
                "source_file": "/videos/video_a.mp4",
                "start": 0.0,
                "end": 5.0,
                "description": "Beginning",
                "order": 0,
            }
        ]
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json(theme="Highlights", segments=segments_data)
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="find highlights", data_dir=tmp_data_dir
        )
        assert len(result.segments) == 1
        seg = result.segments[0]
        assert seg.source_file == "/videos/video_a.mp4"
        assert seg.start == 0.0
        assert seg.end == 5.0
        assert seg.description == "Beginning"
        assert seg.order == 0

    def test_total_duration_equals_sum_of_segment_durations(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        segments_data = [
            {
                "source_file": "/videos/video_a.mp4",
                "start": 0.0,
                "end": 5.0,
                "description": "First",
                "order": 0,
            },
            {
                "source_file": "/videos/video_a.mp4",
                "start": 10.0,
                "end": 20.0,
                "description": "Second",
                "order": 1,
            },
        ]
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json(segments=segments_data)
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="find moments", data_dir=tmp_data_dir
        )
        expected = (5.0 - 0.0) + (20.0 - 10.0)  # 5 + 10 = 15
        assert abs(result.total_duration - expected) < 0.01

    def test_empty_segments_total_duration_is_zero(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json(theme="Nothing", segments=[])
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="find nothing", data_dir=tmp_data_dir
        )
        assert result.segments == []
        assert result.total_duration == 0.0

    def test_only_one_llm_call_for_small_set(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        engine.curate([transcription_a], prompt="test", data_dir=tmp_data_dir)
        assert mock_client.chat.completions.create.call_count == 1

    def test_uses_specified_model(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        engine.curate(
            [transcription_a],
            prompt="test",
            model="google/gemini-2.5-pro",
            data_dir=tmp_data_dir,
        )
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "google/gemini-2.5-pro"

    def test_default_model_is_gemini_flash_lite(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        engine.curate([transcription_a], prompt="test", data_dir=tmp_data_dir)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "google/gemini-2.5-flash-lite"

    def test_prompt_stored_in_cut_list(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="my custom prompt", data_dir=tmp_data_dir
        )
        assert result.prompt == "my custom prompt"

    def test_created_at_is_timezone_aware(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="test", data_dir=tmp_data_dir
        )
        assert result.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# 4. Multi-pass strategy
# ---------------------------------------------------------------------------


class TestMultiPass:
    """Test multi-pass strategy triggered when max_tokens is set very low."""

    def _setup_multi_pass_mocks(
        self,
        mock_client: MagicMock,
        num_transcriptions: int = 2,
        cut_list_segments: list | None = None,
    ) -> None:
        """Configure mock_client with side_effects for a multi-pass run."""
        # Pass 1: one summary per transcription
        summaries = [
            _make_mock_response(f"Summary of video {i}.") for i in range(num_transcriptions)
        ]
        # Pass 2: identify relevant files
        relevant_response = _make_mock_response("Relevant videos identified.")
        # Pass 3: cut list
        cut_list_response = _make_mock_response(
            _cut_list_json(
                theme="Multi-pass theme",
                segments=cut_list_segments or [
                    {
                        "source_file": "/videos/video_a.mp4",
                        "start": 0.0,
                        "end": 5.0,
                        "description": "Key moment",
                        "order": 0,
                    }
                ],
            )
        )
        mock_client.chat.completions.create.side_effect = (
            summaries + [relevant_response, cut_list_response]
        )

    def test_multi_pass_triggered_with_low_max_tokens(
        self,
        transcription_a: Transcription,
        transcription_b: Transcription,
        mock_client: MagicMock,
        tmp_data_dir: Path,
    ) -> None:
        self._setup_multi_pass_mocks(mock_client, num_transcriptions=2)
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a, transcription_b],
            prompt="find interesting moments",
            max_tokens=1,  # Force multi-pass
            data_dir=tmp_data_dir,
        )
        assert isinstance(result, CutList)
        # Must use more than 1 API call
        assert mock_client.chat.completions.create.call_count > 1

    def test_multi_pass_produces_correct_cut_list(
        self,
        transcription_a: Transcription,
        transcription_b: Transcription,
        mock_client: MagicMock,
        tmp_data_dir: Path,
    ) -> None:
        segs = [
            {
                "source_file": "/videos/video_a.mp4",
                "start": 5.0,
                "end": 15.0,
                "description": "Key content",
                "order": 0,
            }
        ]
        self._setup_multi_pass_mocks(mock_client, num_transcriptions=2, cut_list_segments=segs)
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a, transcription_b],
            prompt="find key content",
            max_tokens=1,
            data_dir=tmp_data_dir,
        )
        assert result.theme == "Multi-pass theme"
        assert len(result.segments) == 1
        assert result.segments[0].source_file == "/videos/video_a.mp4"
        assert result.segments[0].start == 5.0
        assert result.segments[0].end == 15.0

    def test_multi_pass_total_duration_correct(
        self,
        transcription_a: Transcription,
        transcription_b: Transcription,
        mock_client: MagicMock,
        tmp_data_dir: Path,
    ) -> None:
        segs = [
            {
                "source_file": "/videos/video_a.mp4",
                "start": 0.0,
                "end": 10.0,
                "description": "First",
                "order": 0,
            },
            {
                "source_file": "/videos/video_b.mp4",
                "start": 5.0,
                "end": 15.0,
                "description": "Second",
                "order": 1,
            },
        ]
        self._setup_multi_pass_mocks(mock_client, num_transcriptions=2, cut_list_segments=segs)
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a, transcription_b],
            prompt="find moments",
            max_tokens=1,
            data_dir=tmp_data_dir,
        )
        expected_total = (10.0 - 0.0) + (15.0 - 5.0)  # 10 + 10 = 20
        assert abs(result.total_duration - expected_total) < 0.01

    def test_multi_pass_cut_list_has_unique_id(
        self,
        transcription_a: Transcription,
        transcription_b: Transcription,
        mock_client: MagicMock,
        tmp_data_dir: Path,
    ) -> None:
        self._setup_multi_pass_mocks(mock_client, num_transcriptions=2)
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a, transcription_b],
            prompt="find moments",
            max_tokens=1,
            data_dir=tmp_data_dir,
        )
        # Must be a valid UUID
        uuid.UUID(result.id)


# ---------------------------------------------------------------------------
# 5. Storage
# ---------------------------------------------------------------------------


class TestCutListStorage:
    def test_cut_list_saved_to_disk(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="test prompt", data_dir=tmp_data_dir
        )
        cut_list_path = tmp_data_dir / "cut_lists" / f"{result.id}.json"
        assert cut_list_path.exists()

    def test_saved_cut_list_is_valid_json(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="test", data_dir=tmp_data_dir
        )
        cut_list_path = tmp_data_dir / "cut_lists" / f"{result.id}.json"
        data = json.loads(cut_list_path.read_text())
        assert "id" in data
        assert "segments" in data
        assert "theme" in data
        assert "total_duration" in data

    def test_multiple_curations_produce_distinct_ids(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result1 = engine.curate(
            [transcription_a], prompt="prompt A", data_dir=tmp_data_dir
        )
        result2 = engine.curate(
            [transcription_a], prompt="prompt B", data_dir=tmp_data_dir
        )
        assert result1.id != result2.id

    def test_multiple_curations_both_saved_separately(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result1 = engine.curate(
            [transcription_a], prompt="prompt A", data_dir=tmp_data_dir
        )
        result2 = engine.curate(
            [transcription_a], prompt="prompt B", data_dir=tmp_data_dir
        )
        assert (tmp_data_dir / "cut_lists" / f"{result1.id}.json").exists()
        assert (tmp_data_dir / "cut_lists" / f"{result2.id}.json").exists()

    def test_ids_are_valid_uuids(
        self, transcription_a: Transcription, mock_client: MagicMock, tmp_data_dir: Path
    ) -> None:
        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        result = engine.curate(
            [transcription_a], prompt="test", data_dir=tmp_data_dir
        )
        # Will raise ValueError if not a valid UUID
        parsed = uuid.UUID(result.id)
        assert str(parsed) == result.id

    def test_transcription_files_not_mutated(
        self,
        transcription_a: Transcription,
        mock_client: MagicMock,
        tmp_data_dir: Path,
    ) -> None:
        """Running curate must not modify transcription files."""
        from snipsnap.storage import save_transcription

        # save_transcription now returns the actual path (includes hash suffix)
        trans_path = save_transcription(transcription_a, tmp_data_dir)
        original_mtime = trans_path.stat().st_mtime
        original_content = trans_path.read_bytes()

        mock_client.chat.completions.create.return_value = _make_mock_response(
            _cut_list_json()
        )
        engine = CurationEngine(client=mock_client, api_key="valid-key")
        engine.curate([transcription_a], prompt="test", data_dir=tmp_data_dir)

        assert trans_path.read_bytes() == original_content
        assert trans_path.stat().st_mtime == original_mtime
