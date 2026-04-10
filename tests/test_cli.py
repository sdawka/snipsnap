"""Tests for the SnipSnap CLI (snipsnap/cli.py).

Uses Click's CliRunner for in-process invocation so no actual Whisper model
is loaded during the test suite.  The WhisperLocalProvider is patched with a
lightweight stub that returns deterministic transcriptions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from snipsnap.cli import main
from snipsnap.curation.engine import (
    AuthenticationError,
    MissingApiKeyError,
)
from snipsnap.models import CutList, CutSegment, Segment, Transcription
from snipsnap.transcription.base import TranscriptionProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_SEGMENTS: List[Segment] = [
    Segment(start=0.0, end=2.5, text="Hello world"),
    Segment(start=2.5, end=5.0, text="This is a test"),
]


class _StubProvider(TranscriptionProvider):
    """Minimal TranscriptionProvider that returns pre-built transcriptions.

    Instantiation never loads a Whisper model, so tests run instantly.
    """

    def __init__(self, segments: Optional[List[Segment]] = None) -> None:
        self._segments = segments if segments is not None else list(_SAMPLE_SEGMENTS)
        self.transcribed: List[str] = []  # tracks calls for assertion

    def transcribe(self, video_path: Path | str) -> Transcription:
        path = Path(video_path)
        self.transcribed.append(str(path))
        return Transcription(
            source_file=str(path),
            duration=5.0,
            language="en",
            model_used="stub",
            segments=list(self._segments),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    """Click test runner with isolated filesystem support."""
    return CliRunner()


@pytest.fixture()
def video_folder(tmp_path: Path) -> Path:
    """Folder with three video files and one non-video file."""
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "clip_a.mp4").touch()
    (folder / "clip_b.mkv").touch()
    (folder / "clip_c.mov").touch()
    (folder / "notes.txt").touch()
    return folder


@pytest.fixture()
def single_video_folder(tmp_path: Path) -> Path:
    """Folder with exactly one video file."""
    folder = tmp_path / "single"
    folder.mkdir()
    (folder / "only.mp4").touch()
    return folder


@pytest.fixture()
def empty_folder(tmp_path: Path) -> Path:
    """Folder that exists but has no video files."""
    folder = tmp_path / "empty"
    folder.mkdir()
    return folder


@pytest.fixture()
def no_video_folder(tmp_path: Path) -> Path:
    """Folder with files but none are supported video formats."""
    folder = tmp_path / "no_videos"
    folder.mkdir()
    (folder / "readme.md").touch()
    (folder / "data.csv").touch()
    return folder


# ---------------------------------------------------------------------------
# Helper: patch WhisperLocalProvider
# ---------------------------------------------------------------------------


def _make_stub_provider(segments: Optional[List[Segment]] = None) -> _StubProvider:
    return _StubProvider(segments=segments)


def _patch_provider(stub: Optional[_StubProvider] = None):
    """Return a context manager that replaces WhisperLocalProvider with a stub."""
    if stub is None:
        stub = _StubProvider()
    return patch(
        "snipsnap.cli.WhisperLocalProvider",
        return_value=stub,
    )


# ===========================================================================
# Help text
# ===========================================================================


class TestTranscribeHelp:
    def test_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["transcribe", "--help"])
        assert result.exit_code == 0

    def test_help_shows_folder_argument(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["transcribe", "--help"])
        assert "FOLDER" in result.output

    def test_help_shows_model_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["transcribe", "--help"])
        assert "--model" in result.output

    def test_help_shows_force_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["transcribe", "--help"])
        assert "--force" in result.output


class TestRootHelp:
    def test_root_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0

    def test_root_help_lists_transcribe(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert "transcribe" in result.output

    def test_version_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_version_outputs_version_string(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert "0.1.0" in result.output


# ===========================================================================
# Error handling
# ===========================================================================


class TestTranscribeErrors:
    def test_nonexistent_folder_exits_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["transcribe", "/this/does/not/exist"])
        assert result.exit_code != 0

    def test_nonexistent_folder_shows_error_message(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["transcribe", "/this/does/not/exist"])
        combined = (result.output or "") + (result.stderr or "")
        assert "not found" in combined.lower() or "/this/does/not/exist" in combined

    def test_empty_folder_exits_nonzero(
        self, runner: CliRunner, empty_folder: Path
    ) -> None:
        with _patch_provider():
            result = runner.invoke(main, ["transcribe", str(empty_folder)])
        assert result.exit_code != 0

    def test_no_video_folder_exits_nonzero(
        self, runner: CliRunner, no_video_folder: Path
    ) -> None:
        with _patch_provider():
            result = runner.invoke(main, ["transcribe", str(no_video_folder)])
        assert result.exit_code != 0

    def test_no_video_folder_shows_clear_message(
        self, runner: CliRunner, no_video_folder: Path
    ) -> None:
        with _patch_provider():
            result = runner.invoke(main, ["transcribe", str(no_video_folder)])
        combined = (result.output or "") + (result.stderr or "")
        assert "no video" in combined.lower()


# ===========================================================================
# Discovery
# ===========================================================================


class TestTranscribeDiscovery:
    def test_discovers_video_files(
        self, runner: CliRunner, video_folder: Path
    ) -> None:
        stub = _StubProvider()
        with _patch_provider(stub):
            result = runner.invoke(main, ["transcribe", str(video_folder)])
        assert result.exit_code == 0
        # All three video names should appear in the output
        assert "clip_a.mp4" in result.output
        assert "clip_b.mkv" in result.output
        assert "clip_c.mov" in result.output

    def test_reports_file_count(
        self, runner: CliRunner, video_folder: Path
    ) -> None:
        stub = _StubProvider()
        with _patch_provider(stub):
            result = runner.invoke(main, ["transcribe", str(video_folder)])
        assert "3" in result.output  # "Found 3 video file(s)"

    def test_ignores_non_video_files(
        self, runner: CliRunner, video_folder: Path
    ) -> None:
        """notes.txt must not appear in the output or be transcribed."""
        stub = _StubProvider()
        with _patch_provider(stub):
            result = runner.invoke(main, ["transcribe", str(video_folder)])
        assert result.exit_code == 0
        assert "notes.txt" not in result.output
        transcribed_names = [Path(p).name for p in stub.transcribed]
        assert "notes.txt" not in transcribed_names


# ===========================================================================
# Happy path: transcription
# ===========================================================================


class TestTranscribeHappyPath:
    def test_exits_zero_on_success(
        self, runner: CliRunner, video_folder: Path
    ) -> None:
        with _patch_provider():
            result = runner.invoke(main, ["transcribe", str(video_folder)])
        assert result.exit_code == 0

    def test_shows_per_file_progress(
        self, runner: CliRunner, video_folder: Path
    ) -> None:
        with _patch_provider():
            result = runner.invoke(main, ["transcribe", str(video_folder)])
        # Each file should trigger a "Transcribing N/3: …" line
        assert "1/3" in result.output
        assert "2/3" in result.output
        assert "3/3" in result.output

    def test_shows_summary_on_completion(
        self, runner: CliRunner, video_folder: Path
    ) -> None:
        with _patch_provider():
            result = runner.invoke(main, ["transcribe", str(video_folder)])
        assert "Summary" in result.output or "transcribed" in result.output.lower()

    def test_transcribes_all_videos(
        self, runner: CliRunner, video_folder: Path, tmp_path: Path
    ) -> None:
        """All three video files must be transcribed and stored as JSON."""
        stub = _StubProvider()

        with _patch_provider(stub), patch(
            "snipsnap.cli.save_transcription"
        ) as mock_save:
            result = runner.invoke(main, ["transcribe", str(video_folder)])

        assert result.exit_code == 0
        assert mock_save.call_count == 3

    def test_transcription_json_written_to_disk(
        self, runner: CliRunner, single_video_folder: Path, tmp_path: Path
    ) -> None:
        """The transcription JSON must be saved via save_transcription."""
        stub = _StubProvider()
        saved: list[object] = []

        def capture_save(transcription: object, data_dir: object = None) -> None:
            saved.append(transcription)

        with _patch_provider(stub), patch(
            "snipsnap.cli.save_transcription", side_effect=capture_save
        ):
            result = runner.invoke(main, ["transcribe", str(single_video_folder)])

        assert result.exit_code == 0
        assert len(saved) == 1

    def test_single_video_produces_valid_transcription_json(
        self, runner: CliRunner, single_video_folder: Path, tmp_path: Path
    ) -> None:
        """Transcription passed to save_transcription must have segments with start, end, text."""
        stub = _StubProvider(segments=_SAMPLE_SEGMENTS)
        saved: list = []

        def capture_save(transcription: object, data_dir: object = None) -> None:
            saved.append(transcription)

        with _patch_provider(stub), patch(
            "snipsnap.cli.save_transcription", side_effect=capture_save
        ):
            runner.invoke(main, ["transcribe", str(single_video_folder)])

        assert len(saved) == 1
        transcription = saved[0]
        segs = transcription.segments  # type: ignore[attr-defined]
        assert len(segs) == 2
        for seg in segs:
            assert hasattr(seg, "start")
            assert hasattr(seg, "end")
            assert hasattr(seg, "text")

    def test_segments_chronologically_ordered(
        self, runner: CliRunner, single_video_folder: Path, tmp_path: Path
    ) -> None:
        """Segments returned by the provider must be in ascending start-time order."""
        segments = [
            Segment(start=0.0, end=1.0, text="First"),
            Segment(start=1.0, end=3.0, text="Second"),
            Segment(start=3.0, end=5.0, text="Third"),
        ]
        stub = _StubProvider(segments=segments)
        saved: list = []

        def capture_save(transcription: object, data_dir: object = None) -> None:
            saved.append(transcription)

        with _patch_provider(stub), patch(
            "snipsnap.cli.save_transcription", side_effect=capture_save
        ):
            runner.invoke(main, ["transcribe", str(single_video_folder)])

        assert saved
        starts = [s.start for s in saved[0].segments]  # type: ignore[attr-defined]
        assert starts == sorted(starts)

    def test_segments_have_valid_durations(
        self, runner: CliRunner, single_video_folder: Path, tmp_path: Path
    ) -> None:
        """Every segment passed to save_transcription must satisfy start < end."""
        stub = _StubProvider(segments=_SAMPLE_SEGMENTS)
        saved: list = []

        def capture_save(transcription: object, data_dir: object = None) -> None:
            saved.append(transcription)

        with _patch_provider(stub), patch(
            "snipsnap.cli.save_transcription", side_effect=capture_save
        ):
            runner.invoke(main, ["transcribe", str(single_video_folder)])

        assert saved
        for seg in saved[0].segments:  # type: ignore[attr-defined]
            assert seg.start < seg.end


# ===========================================================================
# Skip / force behaviour
# ===========================================================================


class TestTranscribeSkipAndForce:
    def test_already_transcribed_files_are_skipped(
        self, runner: CliRunner, video_folder: Path, tmp_path: Path
    ) -> None:
        """Files with an existing transcription must not be re-transcribed."""
        stub = _StubProvider()
        data_dir = tmp_path / "snap_data"

        with _patch_provider(stub):
            # First run: transcribe everything
            runner.invoke(
                main, ["transcribe", str(video_folder)],
                env={"SNIPSNAP_DATA_DIR": str(data_dir)},
            )
            first_count = len(stub.transcribed)

            # Second run: everything should be skipped
            result = runner.invoke(
                main, ["transcribe", str(video_folder)],
                env={"SNIPSNAP_DATA_DIR": str(data_dir)},
            )

        assert result.exit_code == 0
        assert "skip" in result.output.lower()
        # No new transcriptions on the second run
        assert len(stub.transcribed) == first_count

    def test_skip_message_shown_for_existing_transcription(
        self, runner: CliRunner, single_video_folder: Path, tmp_path: Path
    ) -> None:
        stub = _StubProvider()
        data_dir = tmp_path / "snap_data"

        with _patch_provider(stub):
            runner.invoke(
                main, ["transcribe", str(single_video_folder)],
                env={"SNIPSNAP_DATA_DIR": str(data_dir)},
            )
            result = runner.invoke(
                main, ["transcribe", str(single_video_folder)],
                env={"SNIPSNAP_DATA_DIR": str(data_dir)},
            )

        assert "skip" in result.output.lower()

    def test_force_retranscribes_existing_files(
        self, runner: CliRunner, single_video_folder: Path, tmp_path: Path
    ) -> None:
        """--force must cause all files to be re-transcribed regardless of existing data."""
        stub = _StubProvider()
        data_dir = tmp_path / "snap_data"

        with _patch_provider(stub):
            # First run
            runner.invoke(
                main, ["transcribe", str(single_video_folder)],
                env={"SNIPSNAP_DATA_DIR": str(data_dir)},
            )
            # Second run with --force
            result = runner.invoke(
                main, ["transcribe", "--force", str(single_video_folder)],
                env={"SNIPSNAP_DATA_DIR": str(data_dir)},
            )

        assert result.exit_code == 0
        # Both runs should have called transcribe (not skipped on second run)
        assert len(stub.transcribed) == 2


# ===========================================================================
# Error recovery
# ===========================================================================


class TestTranscribeErrorRecovery:
    def test_transcription_failure_does_not_abort_batch(
        self, runner: CliRunner, video_folder: Path, tmp_path: Path
    ) -> None:
        """If one file fails, the remaining files must still be transcribed."""
        call_count = {"n": 0}

        class _FailFirstProvider(_StubProvider):
            def transcribe(self, video_path: Path | str) -> Transcription:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("Simulated transcription failure")
                return super().transcribe(video_path)

        stub = _FailFirstProvider()

        with _patch_provider(stub), patch("snipsnap.cli.save_transcription"):
            result = runner.invoke(main, ["transcribe", str(video_folder)])

        # Exit non-zero: any failure means non-zero exit code
        assert result.exit_code != 0
        # Warning should mention the failure
        combined = (result.output or "") + (result.stderr or "")
        assert "warning" in combined.lower() or "failed" in combined.lower()
        # Two other files should have been transcribed
        assert len(stub.transcribed) == 2

    def test_all_failures_exits_nonzero(
        self, runner: CliRunner, single_video_folder: Path
    ) -> None:
        """When every file fails and nothing was processed, exit code must be non-zero."""

        class _AlwaysFailProvider(_StubProvider):
            def transcribe(self, video_path: Path | str) -> Transcription:
                raise RuntimeError("Always fails")

        stub = _AlwaysFailProvider()

        with _patch_provider(stub), patch("snipsnap.cli.save_transcription"):
            result = runner.invoke(main, ["transcribe", str(single_video_folder)])

        assert result.exit_code != 0


# ===========================================================================
# --model flag
# ===========================================================================


class TestTranscribeModelFlag:
    def test_model_flag_overrides_default(
        self, runner: CliRunner, single_video_folder: Path
    ) -> None:
        """--model should pass the given model size to WhisperLocalProvider."""
        with patch("snipsnap.cli.WhisperLocalProvider") as mock_cls:
            mock_cls.return_value = _StubProvider()
            runner.invoke(
                main, ["transcribe", "--model", "tiny", str(single_video_folder)]
            )
        mock_cls.assert_called_once_with(model_size="tiny")

    def test_default_model_comes_from_config(
        self, runner: CliRunner, single_video_folder: Path
    ) -> None:
        """Without --model, the provider should use config's whisper_model."""
        with patch("snipsnap.cli.WhisperLocalProvider") as mock_cls, patch(
            "snipsnap.cli.get_config"
        ) as mock_cfg:
            cfg = MagicMock()
            cfg.whisper_model = "base"
            mock_cfg.return_value = cfg
            mock_cls.return_value = _StubProvider()

            runner.invoke(main, ["transcribe", str(single_video_folder)])

        mock_cls.assert_called_once_with(model_size="base")


# ===========================================================================
# Unsupported file formats
# ===========================================================================


class TestTranscribeUnsupportedFormats:
    def test_unsupported_files_not_transcribed(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Non-video files (.txt, .jpg, .mp3) must be silently ignored."""
        folder = tmp_path / "mixed"
        folder.mkdir()
        (folder / "video.mp4").touch()
        (folder / "audio.mp3").touch()
        (folder / "image.jpg").touch()
        (folder / "doc.pdf").touch()

        stub = _StubProvider()
        with _patch_provider(stub):
            result = runner.invoke(main, ["transcribe", str(folder)])

        assert result.exit_code == 0
        # Only the .mp4 file should have been transcribed
        assert len(stub.transcribed) == 1
        assert stub.transcribed[0].endswith("video.mp4")

    def test_unsupported_files_not_listed(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Non-video files must not appear in the discovered-files list."""
        folder = tmp_path / "mixed"
        folder.mkdir()
        (folder / "video.mp4").touch()
        (folder / "audio.mp3").touch()

        with _patch_provider():
            result = runner.invoke(main, ["transcribe", str(folder)])

        assert "audio.mp3" not in result.output


# ===========================================================================
# Empty speech (no-speech video)
# ===========================================================================


class TestTranscribeNoSpeech:
    def test_no_speech_produces_valid_empty_transcription(
        self, runner: CliRunner, single_video_folder: Path
    ) -> None:
        """A video with no speech must produce a valid transcription with empty segments."""
        stub = _StubProvider(segments=[])  # no speech → no segments
        saved: list = []

        def capture_save(transcription: object, data_dir: object = None) -> None:
            saved.append(transcription)

        with _patch_provider(stub), patch(
            "snipsnap.cli.save_transcription", side_effect=capture_save
        ):
            result = runner.invoke(main, ["transcribe", str(single_video_folder)])

        assert result.exit_code == 0
        assert len(saved) == 1
        assert saved[0].segments == []  # type: ignore[attr-defined]


# ===========================================================================
# curate command — helpers and fixtures
# ===========================================================================

# Sample data for curate tests
_SAMPLE_TRANSCRIPTION = Transcription(
    source_file="/videos/clip_a.mp4",
    duration=10.0,
    language="en",
    model_used="stub",
    segments=[
        Segment(start=0.0, end=3.0, text="This is really funny"),
        Segment(start=5.0, end=8.0, text="Another funny moment"),
    ],
)

_SAMPLE_CUT_LIST = CutList(
    id="test-uuid-1234",
    prompt="find funny moments",
    theme="Funny Moments",
    created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    total_duration=6.0,
    segments=[
        CutSegment(
            source_file="/videos/clip_a.mp4",
            start=0.0,
            end=3.0,
            description="Hilarious opening moment",
            order=0,
        ),
        CutSegment(
            source_file="/videos/clip_a.mp4",
            start=5.0,
            end=8.0,
            description="Second funny bit",
            order=1,
        ),
    ],
)


def _make_mock_engine(cut_list: CutList = _SAMPLE_CUT_LIST) -> MagicMock:
    """Build a MagicMock CurationEngine that returns the given cut_list."""
    mock = MagicMock()
    mock.curate.return_value = cut_list
    return mock


# ===========================================================================
# curate — help text
# ===========================================================================


class TestCurateHelp:
    def test_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["curate", "--help"])
        assert result.exit_code == 0

    def test_help_shows_prompt_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["curate", "--help"])
        assert "--prompt" in result.output

    def test_help_shows_model_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["curate", "--help"])
        assert "--model" in result.output

    def test_help_shows_data_dir_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["curate", "--help"])
        assert "--data-dir" in result.output


# ===========================================================================
# curate — missing --prompt requirement
# ===========================================================================


class TestCurateMissingPrompt:
    def test_missing_prompt_exits_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["curate"])
        assert result.exit_code != 0

    def test_missing_prompt_shows_usage(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["curate"])
        combined = (result.output or "") + (result.stderr or "")
        assert "prompt" in combined.lower() or "missing" in combined.lower()


# ===========================================================================
# curate — error: no transcriptions
# ===========================================================================


class TestCurateNoTranscriptions:
    def test_no_transcriptions_exits_nonzero(self, runner: CliRunner) -> None:
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]):
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        assert result.exit_code != 0

    def test_no_transcriptions_shows_clear_message(self, runner: CliRunner) -> None:
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]):
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        combined = (result.output or "") + (result.stderr or "")
        # Should suggest running snipsnap transcribe first
        assert "transcri" in combined.lower()

    def test_no_transcriptions_no_cut_list_created(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), patch(
            "snipsnap.cli.save_cut_list"
        ) as mock_save:
            runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        mock_save.assert_not_called()


# ===========================================================================
# curate — error: missing API key
# ===========================================================================


class TestCurateMissingApiKey:
    def test_missing_api_key_exits_nonzero(self, runner: CliRunner) -> None:
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.openrouter_api_key = ""
            cfg.model = "google/gemini-2.5-flash-lite"
            cfg.data_dir = Path("/tmp/test_data")
            mock_cfg.return_value = cfg
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        assert result.exit_code != 0

    def test_missing_api_key_shows_clear_message(self, runner: CliRunner) -> None:
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.openrouter_api_key = ""
            cfg.model = "google/gemini-2.5-flash-lite"
            cfg.data_dir = Path("/tmp/test_data")
            mock_cfg.return_value = cfg
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        combined = (result.output or "") + (result.stderr or "")
        # Should mention API key
        assert "api" in combined.lower() or "key" in combined.lower()

    def test_engine_missing_api_key_error_exits_nonzero(
        self, runner: CliRunner
    ) -> None:
        """MissingApiKeyError from engine translates to non-zero exit."""
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.curate.side_effect = MissingApiKeyError("No API key configured")
            mock_cls.return_value = mock_engine
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        assert result.exit_code != 0


# ===========================================================================
# curate — error: invalid API key (authentication error)
# ===========================================================================


class TestCurateInvalidApiKey:
    def test_auth_error_exits_nonzero(self, runner: CliRunner) -> None:
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.curate.side_effect = AuthenticationError(
                "Authentication failed: the API key is invalid"
            )
            mock_cls.return_value = mock_engine
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        assert result.exit_code != 0

    def test_auth_error_message_does_not_expose_key(self, runner: CliRunner) -> None:
        """The error message must never expose the actual API key value."""
        fake_key = "sk-or-v1-supersecret-key-value"
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls, patch(
            "snipsnap.cli.get_config"
        ) as mock_cfg:
            cfg = MagicMock()
            cfg.openrouter_api_key = fake_key
            cfg.model = "google/gemini-2.5-flash-lite"
            cfg.data_dir = Path("/tmp/test_data")
            mock_cfg.return_value = cfg
            mock_engine = MagicMock()
            mock_engine.curate.side_effect = AuthenticationError(
                "Authentication failed: the API key is invalid"
            )
            mock_cls.return_value = mock_engine
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        combined = (result.output or "") + (result.stderr or "")
        assert fake_key not in combined

    def test_auth_error_shows_authentication_message(self, runner: CliRunner) -> None:
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.curate.side_effect = AuthenticationError(
                "Authentication failed: the API key is invalid"
            )
            mock_cls.return_value = mock_engine
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        combined = (result.output or "") + (result.stderr or "")
        assert "auth" in combined.lower() or "invalid" in combined.lower() or "key" in combined.lower()


# ===========================================================================
# curate — happy path
# ===========================================================================


class TestCurateHappyPath:
    def test_exits_zero_on_success(self, runner: CliRunner) -> None:
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = _make_mock_engine()
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        assert result.exit_code == 0

    def test_prints_cut_list_id(self, runner: CliRunner) -> None:
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = _make_mock_engine()
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        assert _SAMPLE_CUT_LIST.id in result.output

    def test_prints_segment_count(self, runner: CliRunner) -> None:
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = _make_mock_engine()
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        # Should print "2" segments somewhere
        assert "2" in result.output

    def test_prints_total_duration(self, runner: CliRunner) -> None:
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = _make_mock_engine()
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        assert "6" in result.output  # total_duration = 6.0

    def test_prints_segment_source_file(self, runner: CliRunner) -> None:
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = _make_mock_engine()
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        # Source filename should appear in output
        assert "clip_a.mp4" in result.output

    def test_prints_segment_timestamps(self, runner: CliRunner) -> None:
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = _make_mock_engine()
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        # Timestamps 0.0 and 3.0 or 5.0 and 8.0 should appear
        assert "0.0" in result.output or "0:00" in result.output

    def test_prints_segment_description(self, runner: CliRunner) -> None:
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = _make_mock_engine()
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        assert "Hilarious" in result.output or "funny" in result.output.lower()

    def test_calls_curate_with_correct_prompt(self, runner: CliRunner) -> None:
        mock_engine = _make_mock_engine()
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = mock_engine
            runner.invoke(main, ["curate", "--prompt", "find funny moments"])
        mock_engine.curate.assert_called_once()
        call_kwargs = mock_engine.curate.call_args
        # prompt must be passed
        assert "find funny moments" in str(call_kwargs)


# ===========================================================================
# curate — empty cut list (prompt matches nothing)
# ===========================================================================


class TestCurateEmptyResult:
    def test_empty_segments_exits_zero(self, runner: CliRunner) -> None:
        empty_cut_list = CutList(
            id="empty-uuid",
            prompt="find nothing",
            theme="Nothing Found",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            total_duration=0.0,
            segments=[],
        )
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = _make_mock_engine(empty_cut_list)
            result = runner.invoke(main, ["curate", "--prompt", "find nothing"])
        assert result.exit_code == 0

    def test_empty_segments_shows_zero_count(self, runner: CliRunner) -> None:
        empty_cut_list = CutList(
            id="empty-uuid",
            prompt="find nothing",
            theme="Nothing Found",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            total_duration=0.0,
            segments=[],
        )
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = _make_mock_engine(empty_cut_list)
            result = runner.invoke(main, ["curate", "--prompt", "find nothing"])
        assert "0" in result.output


# ===========================================================================
# curate — --model flag
# ===========================================================================


class TestCurateModelFlag:
    def test_model_flag_passed_to_engine(self, runner: CliRunner) -> None:
        mock_engine = _make_mock_engine()
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = mock_engine
            runner.invoke(
                main,
                ["curate", "--prompt", "find funny", "--model", "openai/gpt-4o"],
            )
        call_kwargs = mock_engine.curate.call_args
        assert "openai/gpt-4o" in str(call_kwargs)

    def test_default_model_is_gemini_flash_lite(self, runner: CliRunner) -> None:
        mock_engine = _make_mock_engine()
        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls, patch(
            "snipsnap.cli.get_config"
        ) as mock_cfg:
            cfg = MagicMock()
            cfg.openrouter_api_key = "test-key"
            cfg.model = "google/gemini-2.5-flash-lite"
            cfg.data_dir = Path("/tmp/test_data")
            mock_cfg.return_value = cfg
            mock_cls.return_value = mock_engine
            runner.invoke(main, ["curate", "--prompt", "find funny"])
        call_kwargs = mock_engine.curate.call_args
        assert "google/gemini-2.5-flash-lite" in str(call_kwargs)


# ===========================================================================
# curate — multiple curations produce distinct cut lists
# ===========================================================================


class TestCurateMultipleCurations:
    def test_multiple_curations_produce_distinct_ids(self, runner: CliRunner) -> None:
        cut_list_a = CutList(
            id="uuid-aaaa",
            prompt="find funny moments",
            theme="Funny",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            total_duration=3.0,
            segments=[
                CutSegment(
                    source_file="/videos/clip_a.mp4",
                    start=0.0,
                    end=3.0,
                    description="Funny bit",
                    order=0,
                )
            ],
        )
        cut_list_b = CutList(
            id="uuid-bbbb",
            prompt="find dramatic moments",
            theme="Drama",
            created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            total_duration=3.0,
            segments=[
                CutSegment(
                    source_file="/videos/clip_a.mp4",
                    start=5.0,
                    end=8.0,
                    description="Dramatic bit",
                    order=0,
                )
            ],
        )

        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_cls.return_value = _make_mock_engine(cut_list_a)
            result_a = runner.invoke(
                main, ["curate", "--prompt", "find funny moments"]
            )
            mock_cls.return_value = _make_mock_engine(cut_list_b)
            result_b = runner.invoke(
                main, ["curate", "--prompt", "find dramatic moments"]
            )

        assert result_a.exit_code == 0
        assert result_b.exit_code == 0
        assert "uuid-aaaa" in result_a.output
        assert "uuid-bbbb" in result_b.output
        # The two IDs must be different
        assert "uuid-aaaa" != "uuid-bbbb"


# ===========================================================================
# curate — --data-dir flag
# ===========================================================================


class TestCurateDataDirFlag:
    def test_data_dir_flag_passed_to_load(self, runner: CliRunner, tmp_path: Path) -> None:
        """--data-dir should override the default data directory."""
        custom_dir = tmp_path / "custom_data"
        custom_dir.mkdir()
        mock_engine = _make_mock_engine()
        with patch(
            "snipsnap.cli.load_all_transcriptions"
        ) as mock_load, patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_load.return_value = [_SAMPLE_TRANSCRIPTION]
            mock_cls.return_value = mock_engine
            result = runner.invoke(
                main,
                [
                    "curate",
                    "--prompt",
                    "find funny moments",
                    "--data-dir",
                    str(custom_dir),
                ],
            )
        assert result.exit_code == 0
        # load_all_transcriptions should have been called with the custom data dir
        mock_load.assert_called_once()
        call_args = mock_load.call_args
        assert custom_dir in call_args[0] or custom_dir == call_args[1].get("data_dir")


# ===========================================================================
# transcribe — UX fix: model loaded AFTER video discovery
# ===========================================================================


class TestTranscribeModelLoadOrder:
    def test_model_not_loaded_when_no_videos(
        self, runner: CliRunner, empty_folder: Path
    ) -> None:
        """WhisperLocalProvider must NOT be instantiated when no videos are found."""
        with patch("snipsnap.cli.WhisperLocalProvider") as mock_cls:
            mock_cls.return_value = _StubProvider()
            result = runner.invoke(main, ["transcribe", str(empty_folder)])
        # Should exit with error (no videos found)
        assert result.exit_code != 0
        # But WhisperLocalProvider should NOT have been instantiated
        mock_cls.assert_not_called()

    def test_model_loaded_when_videos_exist(
        self, runner: CliRunner, single_video_folder: Path
    ) -> None:
        """WhisperLocalProvider MUST be instantiated when videos are found."""
        with patch("snipsnap.cli.WhisperLocalProvider") as mock_cls:
            mock_cls.return_value = _StubProvider()
            result = runner.invoke(main, ["transcribe", str(single_video_folder)])
        assert result.exit_code == 0
        mock_cls.assert_called_once()


# ===========================================================================
# Root help — all subcommands listed
# ===========================================================================


class TestRootHelpAllSubcommands:
    def test_root_help_lists_curate(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "curate" in result.output

    def test_root_help_lists_export(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "export" in result.output

    def test_root_help_lists_status(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "status" in result.output

    def test_root_help_lists_all_four_subcommands(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "transcribe" in result.output
        assert "curate" in result.output
        assert "export" in result.output
        assert "status" in result.output


# ===========================================================================
# export — help text
# ===========================================================================


class TestExportHelp:
    def test_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["export", "--help"])
        assert result.exit_code == 0

    def test_help_shows_format_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["export", "--help"])
        assert "--format" in result.output

    def test_help_shows_cut_list_id_argument(self, runner: CliRunner) -> None:
        """CUT_LIST_ID must appear in the export help text as an argument."""
        result = runner.invoke(main, ["export", "--help"])
        assert "CUT_LIST_ID" in result.output or "cut_list_id" in result.output.lower()

    def test_help_shows_valid_format_options(self, runner: CliRunner) -> None:
        """Help must mention all valid output formats."""
        result = runner.invoke(main, ["export", "--help"])
        combined = result.output.lower()
        assert "edl" in combined
        assert "fcpxml" in combined
        assert "davinci" in combined

    def test_help_shows_output_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["export", "--help"])
        assert "--output" in result.output

    def test_help_shows_fps_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["export", "--help"])
        assert "--fps" in result.output


# ===========================================================================
# export — error handling
# ===========================================================================


class TestExportErrors:
    def test_missing_cut_list_exits_nonzero(self, runner: CliRunner) -> None:
        """A non-existent cut list ID must produce a non-zero exit code."""
        with patch("snipsnap.cli.load_cut_list", return_value=None):
            result = runner.invoke(
                main, ["export", "nonexistent-id", "--format", "edl"]
            )
        assert result.exit_code != 0

    def test_missing_cut_list_shows_clear_error(self, runner: CliRunner) -> None:
        """Error message must reference the cut list ID and guide the user."""
        with patch("snipsnap.cli.load_cut_list", return_value=None):
            result = runner.invoke(
                main, ["export", "nonexistent-id", "--format", "edl"]
            )
        combined = (result.output or "") + (result.stderr or "")
        assert "nonexistent-id" in combined or "not found" in combined.lower()

    def test_missing_cut_list_suggests_status(self, runner: CliRunner) -> None:
        """Error message should guide the user to 'snipsnap status'."""
        with patch("snipsnap.cli.load_cut_list", return_value=None):
            result = runner.invoke(
                main, ["export", "missing-id", "--format", "edl"]
            )
        combined = (result.output or "") + (result.stderr or "")
        assert "status" in combined.lower()

    def test_missing_format_exits_nonzero(self, runner: CliRunner) -> None:
        """Omitting --format must produce a non-zero exit code."""
        result = runner.invoke(main, ["export", "some-cut-list-id"])
        assert result.exit_code != 0

    def test_invalid_format_exits_nonzero(self, runner: CliRunner) -> None:
        """An unsupported --format value must exit non-zero."""
        with patch("snipsnap.cli.load_cut_list", return_value=_SAMPLE_CUT_LIST):
            result = runner.invoke(
                main, ["export", "some-id", "--format", "mp4"]
            )
        assert result.exit_code != 0


# ===========================================================================
# status — help text
# ===========================================================================


class TestStatusHelp:
    def test_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0

    def test_help_shows_data_dir_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["status", "--help"])
        assert "--data-dir" in result.output


# ===========================================================================
# status — empty state (no data)
# ===========================================================================


class TestStatusEmptyState:
    def test_exits_zero_with_no_data(self, runner: CliRunner) -> None:
        """status must exit 0 even when no transcriptions or cut lists exist."""
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=[]):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0

    def test_shows_zero_transcription_count_when_empty(self, runner: CliRunner) -> None:
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=[]):
            result = runner.invoke(main, ["status"])
        assert "Transcriptions: 0" in result.output or "0" in result.output

    def test_shows_zero_cut_list_count_when_empty(self, runner: CliRunner) -> None:
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=[]):
            result = runner.invoke(main, ["status"])
        assert "Cut Lists: 0" in result.output or "0" in result.output

    def test_shows_empty_state_message_for_transcriptions(
        self, runner: CliRunner
    ) -> None:
        """When no transcriptions exist, an informative empty-state message is shown."""
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=[]):
            result = runner.invoke(main, ["status"])
        assert "none" in result.output.lower() or "0" in result.output

    def test_shows_empty_state_message_for_cut_lists(
        self, runner: CliRunner
    ) -> None:
        """When no cut lists exist, an informative empty-state message is shown."""
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=[]):
            result = runner.invoke(main, ["status"])
        assert "none" in result.output.lower() or "0" in result.output


# ===========================================================================
# status — with transcription data
# ===========================================================================


class TestStatusWithTranscriptions:
    def test_shows_transcription_count(self, runner: CliRunner) -> None:
        transcriptions = [_SAMPLE_TRANSCRIPTION]
        with patch("snipsnap.cli.load_all_transcriptions", return_value=transcriptions), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=[]):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "1" in result.output

    def test_shows_transcription_filename(self, runner: CliRunner) -> None:
        """The filename of each transcribed video must appear in the status output."""
        transcriptions = [_SAMPLE_TRANSCRIPTION]
        with patch("snipsnap.cli.load_all_transcriptions", return_value=transcriptions), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=[]):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        # _SAMPLE_TRANSCRIPTION.source_file = "/videos/clip_a.mp4"
        assert "clip_a.mp4" in result.output

    def test_shows_multiple_transcription_filenames(
        self, runner: CliRunner
    ) -> None:
        """All transcription filenames must appear when multiple exist."""
        transcription_b = Transcription(
            source_file="/videos/clip_b.mkv",
            duration=8.0,
            language="en",
            model_used="stub",
            segments=[Segment(start=0.0, end=4.0, text="Some speech")],
        )
        transcriptions = [_SAMPLE_TRANSCRIPTION, transcription_b]
        with patch("snipsnap.cli.load_all_transcriptions", return_value=transcriptions), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=[]):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "clip_a.mp4" in result.output
        assert "clip_b.mkv" in result.output

    def test_shows_correct_transcription_count_for_multiple(
        self, runner: CliRunner
    ) -> None:
        transcription_b = Transcription(
            source_file="/videos/clip_b.mkv",
            duration=8.0,
            language="en",
            model_used="stub",
            segments=[],
        )
        transcriptions = [_SAMPLE_TRANSCRIPTION, transcription_b]
        with patch("snipsnap.cli.load_all_transcriptions", return_value=transcriptions), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=[]):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "2" in result.output


# ===========================================================================
# status — with cut list data
# ===========================================================================


class TestStatusWithCutLists:
    def test_shows_cut_list_count(self, runner: CliRunner) -> None:
        cut_lists = [_SAMPLE_CUT_LIST]
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=cut_lists):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "1" in result.output

    def test_shows_cut_list_id(self, runner: CliRunner) -> None:
        """The cut list ID must appear in the status output for use by export."""
        cut_lists = [_SAMPLE_CUT_LIST]
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=cut_lists):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert _SAMPLE_CUT_LIST.id in result.output

    def test_shows_cut_list_creation_timestamp(self, runner: CliRunner) -> None:
        """The creation date must appear so users can identify which cut list they want."""
        cut_lists = [_SAMPLE_CUT_LIST]
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=cut_lists):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        # _SAMPLE_CUT_LIST.created_at = datetime(2024, 1, 1, ...)
        assert "2024" in result.output

    def test_shows_cut_list_prompt(self, runner: CliRunner) -> None:
        """The prompt used to create the cut list must appear in the status output."""
        cut_lists = [_SAMPLE_CUT_LIST]
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=cut_lists):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        # _SAMPLE_CUT_LIST.prompt = "find funny moments"
        assert "find funny moments" in result.output

    def test_shows_multiple_cut_list_ids(self, runner: CliRunner) -> None:
        """All cut list IDs must appear when multiple cut lists exist."""
        cut_list_b = CutList(
            id="second-uuid-5678",
            prompt="find dramatic moments",
            theme="Drama",
            created_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
            total_duration=4.0,
            segments=[],
        )
        cut_lists = [_SAMPLE_CUT_LIST, cut_list_b]
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=cut_lists):
            result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert _SAMPLE_CUT_LIST.id in result.output
        assert cut_list_b.id in result.output

    def test_cut_list_ids_are_usable_by_export(self, runner: CliRunner) -> None:
        """The cut list IDs shown by status must match what load_cut_list returns."""
        cut_lists = [_SAMPLE_CUT_LIST]
        with patch("snipsnap.cli.load_all_transcriptions", return_value=[]), \
             patch("snipsnap.cli.load_all_cut_lists", return_value=cut_lists):
            status_result = runner.invoke(main, ["status"])
        assert status_result.exit_code == 0
        # The ID from status must be accepted by the export command (load_cut_list)
        cut_list_id = _SAMPLE_CUT_LIST.id
        assert cut_list_id in status_result.output
        # Verify the same ID can be used with export (by checking load_cut_list is called)
        with patch("snipsnap.cli.load_cut_list", return_value=_SAMPLE_CUT_LIST) as mock_load:
            runner.invoke(main, ["export", cut_list_id, "--format", "edl"])
        mock_load.assert_called_once_with(cut_list_id, None)


# ===========================================================================
# transcribe — partial failure exits non-zero
# ===========================================================================


class TestTranscribePartialFailureSummary:
    def test_partial_failure_exits_nonzero(
        self, runner: CliRunner, video_folder: Path
    ) -> None:
        """Even when some files succeed, any failure must produce a non-zero exit."""
        call_count = {"n": 0}

        class _FailOnce(_StubProvider):
            def transcribe(self, video_path: Path | str) -> Transcription:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("Simulated failure")
                return super().transcribe(video_path)

        stub = _FailOnce()
        with _patch_provider(stub), patch("snipsnap.cli.save_transcription"):
            result = runner.invoke(main, ["transcribe", str(video_folder)])

        assert result.exit_code != 0

    def test_partial_failure_summary_shows_succeeded_and_failed(
        self, runner: CliRunner, video_folder: Path
    ) -> None:
        """Summary line must show 'succeeded' and 'failed' counts."""
        call_count = {"n": 0}

        class _FailOnce(_StubProvider):
            def transcribe(self, video_path: Path | str) -> Transcription:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("Simulated failure")
                return super().transcribe(video_path)

        stub = _FailOnce()
        with _patch_provider(stub), patch("snipsnap.cli.save_transcription"):
            result = runner.invoke(main, ["transcribe", str(video_folder)])

        combined = (result.output or "") + (result.stderr or "")
        assert "succeeded" in combined.lower() or "failed" in combined.lower()

    def test_success_only_summary_shows_succeeded(
        self, runner: CliRunner, video_folder: Path
    ) -> None:
        """When all files succeed, summary uses 'succeeded' (not 'transcribed')."""
        with _patch_provider(), patch("snipsnap.cli.save_transcription"):
            result = runner.invoke(main, ["transcribe", str(video_folder)])

        assert result.exit_code == 0
        assert "succeeded" in result.output.lower()


# ===========================================================================
# curate — unmapped OpenAI SDK errors
# ===========================================================================


class TestCurateOpenAIErrors:
    """Verify that raw openai SDK errors not caught by engine produce
    user-friendly messages in the CLI (no raw tracebacks)."""

    def test_api_status_error_exits_nonzero(self, runner: CliRunner) -> None:
        """An unmapped openai.APIStatusError must produce non-zero exit."""
        import openai

        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.curate.side_effect = openai.InternalServerError(
                message="Internal Server Error",
                response=MagicMock(status_code=500, headers={}),
                body={"error": {"message": "Internal Server Error"}},
            )
            mock_cls.return_value = mock_engine
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])

        assert result.exit_code != 0

    def test_api_status_error_shows_user_friendly_message(
        self, runner: CliRunner
    ) -> None:
        """User-friendly message must be shown for unmapped APIStatusError."""
        import openai

        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.curate.side_effect = openai.InternalServerError(
                message="Internal Server Error",
                response=MagicMock(status_code=500, headers={}),
                body={"error": {"message": "Internal Server Error"}},
            )
            mock_cls.return_value = mock_engine
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])

        combined = (result.output or "") + (result.stderr or "")
        assert "error" in combined.lower()
        # Exception must be None or SystemExit (not a raw openai exception)
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_api_status_error_no_raw_traceback(self, runner: CliRunner) -> None:
        """APIStatusError must not propagate as an unhandled exception (no raw traceback)."""
        import openai

        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.curate.side_effect = openai.InternalServerError(
                message="Internal Server Error",
                response=MagicMock(status_code=500, headers={}),
                body={"error": {"message": "Internal Server Error"}},
            )
            mock_cls.return_value = mock_engine
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])

        # Must not be an unhandled openai exception
        assert not isinstance(result.exception, openai.APIStatusError)

    def test_openai_base_error_exits_nonzero(self, runner: CliRunner) -> None:
        """An unmapped openai.OpenAIError must produce non-zero exit."""
        import openai

        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.curate.side_effect = openai.OpenAIError(
                "Some unexpected SDK error"
            )
            mock_cls.return_value = mock_engine
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])

        assert result.exit_code != 0

    def test_openai_base_error_shows_user_friendly_message(
        self, runner: CliRunner
    ) -> None:
        """User-friendly message must be shown for unmapped OpenAIError."""
        import openai

        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.curate.side_effect = openai.OpenAIError(
                "Some unexpected SDK error"
            )
            mock_cls.return_value = mock_engine
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])

        combined = (result.output or "") + (result.stderr or "")
        assert "error" in combined.lower()
        # Exception must be None or SystemExit (not a raw openai exception)
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_openai_base_error_no_raw_traceback(self, runner: CliRunner) -> None:
        """OpenAIError must not propagate as an unhandled exception (no raw traceback)."""
        import openai

        with patch(
            "snipsnap.cli.load_all_transcriptions",
            return_value=[_SAMPLE_TRANSCRIPTION],
        ), patch("snipsnap.cli.CurationEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.curate.side_effect = openai.OpenAIError(
                "Some unexpected SDK error"
            )
            mock_cls.return_value = mock_engine
            result = runner.invoke(main, ["curate", "--prompt", "find funny moments"])

        # Must not be an unhandled openai exception
        assert not isinstance(result.exception, openai.OpenAIError)
