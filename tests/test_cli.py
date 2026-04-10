"""Tests for the SnipSnap CLI (snipsnap/cli.py).

Uses Click's CliRunner for in-process invocation so no actual Whisper model
is loaded during the test suite.  The WhisperLocalProvider is patched with a
lightweight stub that returns deterministic transcriptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from snipsnap.cli import main
from snipsnap.models import Segment, Transcription
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

        # Exit 0: partial success is still success
        assert result.exit_code == 0
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
