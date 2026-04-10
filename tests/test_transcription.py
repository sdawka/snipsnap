"""Comprehensive tests for the transcription engine.

Tests are split into three layers:

1. TranscriptionProvider base class – video discovery and batch processing
   logic, using a lightweight concrete subclass (no model loading required).
2. WhisperLocalProvider – unit tests that patch faster_whisper.WhisperModel
   so no model download is needed.
3. Integration test – uses a real ffmpeg-generated 5-second sine-tone video
   with the 'tiny' Whisper model (downloaded on first run, ~75 MB).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from snipsnap.models import Segment, Transcription
from snipsnap.transcription.base import SUPPORTED_EXTENSIONS, TranscriptionProvider
from snipsnap.transcription.whisper_local import WhisperLocalProvider

# ---------------------------------------------------------------------------
# Helpers / minimal concrete provider for base-class tests
# ---------------------------------------------------------------------------


class _DummyProvider(TranscriptionProvider):
    """Minimal TranscriptionProvider that returns a hard-coded Transcription."""

    def __init__(self, segments: Optional[List[Segment]] = None) -> None:
        self._segments = segments or [
            Segment(start=0.0, end=2.0, text="Hello"),
            Segment(start=2.0, end=4.0, text="World"),
        ]

    def transcribe(self, video_path: Path | str) -> Transcription:
        return Transcription(
            source_file=str(video_path),
            duration=5.0,
            language="en",
            model_used="dummy",
            segments=self._segments,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def video_folder(tmp_path: Path) -> Path:
    """Folder containing one file for each supported extension plus non-videos.

    Each video file has a unique stem so the storage layer writes distinct
    JSON files (storage uses the stem as the filename key).
    """
    folder = tmp_path / "videos"
    folder.mkdir()
    for ext in (".mp4", ".mkv", ".mov", ".avi", ".webm"):
        # Use the extension (without the dot) as the stem so each JSON is unique
        stem = ext.lstrip(".")
        (folder / f"{stem}{ext}").touch()
    # Non-video files that must NOT be discovered
    (folder / "readme.txt").touch()
    (folder / "thumbnail.jpg").touch()
    (folder / "audio.mp3").touch()
    return folder


@pytest.fixture()
def nested_video_folder(tmp_path: Path) -> Path:
    """Folder with videos spread across nested subdirectories."""
    root = tmp_path / "videos"
    root.mkdir()
    (root / "top.mp4").touch()
    sub = root / "subdir"
    sub.mkdir()
    (sub / "mid.mkv").touch()
    deep = sub / "deep"
    deep.mkdir()
    (deep / "bottom.webm").touch()
    return root


@pytest.fixture()
def empty_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "empty"
    folder.mkdir()
    return folder


@pytest.fixture()
def non_video_folder(tmp_path: Path) -> Path:
    """Folder that exists but contains no video files."""
    folder = tmp_path / "no_videos"
    folder.mkdir()
    (folder / "notes.txt").touch()
    (folder / "data.csv").touch()
    return folder


@pytest.fixture()
def dummy_video_file(tmp_path: Path) -> Path:
    """A file with an .mp4 extension that exists on disk.

    Used for unit tests that mock WhisperModel – they only need a path that
    passes the ``exists()`` check; actual video content is irrelevant.
    """
    video_path = tmp_path / "dummy.mp4"
    video_path.touch()
    return video_path


@pytest.fixture()
def test_video(tmp_path: Path) -> Path:
    """5-second video with a 440 Hz sine tone, created with ffmpeg.

    Used for integration tests that require real video content.
    Skips the test if ffmpeg is unavailable or fails.
    """
    video_path = tmp_path / "sine_test.mp4"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=5",
            "-f", "lavfi",
            "-i", "color=c=black:size=320x240:duration=5",
            "-shortest",
            "-c:v", "libx264",
            "-c:a", "aac",
            str(video_path),
        ],
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.skip(
            f"ffmpeg failed to create test video: {result.stderr.decode()[:200]}"
        )
    return video_path


@pytest.fixture()
def mock_whisper_model():
    """Patched WhisperModel that returns two pre-defined segments."""
    seg1 = MagicMock()
    seg1.start = 0.0
    seg1.end = 2.5
    seg1.text = " Hello world"

    seg2 = MagicMock()
    seg2.start = 2.5
    seg2.end = 5.0
    seg2.text = " Goodbye"

    info = MagicMock()
    info.duration = 5.0
    info.language = "en"

    model_instance = MagicMock()
    model_instance.transcribe.return_value = ([seg1, seg2], info)

    with patch("snipsnap.transcription.whisper_local.WhisperModel") as mock_cls:
        mock_cls.return_value = model_instance
        yield mock_cls, model_instance


# ===========================================================================
# Tests: SUPPORTED_EXTENSIONS constant
# ===========================================================================


class TestSupportedExtensions:
    def test_contains_required_formats(self) -> None:
        assert SUPPORTED_EXTENSIONS == {".mp4", ".mkv", ".mov", ".avi", ".webm"}

    def test_all_extensions_start_with_dot(self) -> None:
        for ext in SUPPORTED_EXTENSIONS:
            assert ext.startswith(".")


# ===========================================================================
# Tests: TranscriptionProvider.discover_videos
# ===========================================================================


class TestDiscoverVideos:
    def test_finds_all_supported_formats(self, video_folder: Path) -> None:
        provider = _DummyProvider()
        videos = provider.discover_videos(video_folder)
        extensions = {v.suffix for v in videos}
        assert extensions == SUPPORTED_EXTENSIONS

    def test_returns_paths_not_strings(self, video_folder: Path) -> None:
        provider = _DummyProvider()
        videos = provider.discover_videos(video_folder)
        for v in videos:
            assert isinstance(v, Path)

    def test_ignores_non_video_files(self, video_folder: Path) -> None:
        provider = _DummyProvider()
        videos = provider.discover_videos(video_folder)
        for v in videos:
            assert v.suffix in SUPPORTED_EXTENSIONS

    def test_returns_empty_list_for_empty_folder(self, empty_folder: Path) -> None:
        provider = _DummyProvider()
        assert provider.discover_videos(empty_folder) == []

    def test_returns_empty_list_for_no_video_files(self, non_video_folder: Path) -> None:
        provider = _DummyProvider()
        assert provider.discover_videos(non_video_folder) == []

    def test_discovers_recursively_in_subdirectories(
        self, nested_video_folder: Path
    ) -> None:
        provider = _DummyProvider()
        videos = provider.discover_videos(nested_video_folder)
        assert len(videos) == 3

    def test_results_are_sorted(self, video_folder: Path) -> None:
        provider = _DummyProvider()
        videos = provider.discover_videos(video_folder)
        assert videos == sorted(videos)

    def test_raises_for_nonexistent_folder(self, tmp_path: Path) -> None:
        provider = _DummyProvider()
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            provider.discover_videos(missing)

    def test_raises_for_path_that_is_a_file(self, tmp_path: Path) -> None:
        provider = _DummyProvider()
        file_path = tmp_path / "file.txt"
        file_path.touch()
        with pytest.raises(NotADirectoryError):
            provider.discover_videos(file_path)

    def test_accepts_string_path(self, video_folder: Path) -> None:
        provider = _DummyProvider()
        videos = provider.discover_videos(str(video_folder))
        assert len(videos) > 0


# ===========================================================================
# Tests: TranscriptionProvider.transcribe_batch
# ===========================================================================


class TestTranscribeBatch:
    def test_batch_processes_all_videos(
        self, video_folder: Path, tmp_data_dir: Path
    ) -> None:
        provider = _DummyProvider()
        results = provider.transcribe_batch(video_folder, data_dir=tmp_data_dir)
        assert len(results) == 5  # one per supported extension

    def test_batch_returns_transcription_objects(
        self, video_folder: Path, tmp_data_dir: Path
    ) -> None:
        provider = _DummyProvider()
        results = provider.transcribe_batch(video_folder, data_dir=tmp_data_dir)
        for t in results:
            assert isinstance(t, Transcription)

    def test_batch_saves_json_files(
        self, video_folder: Path, tmp_data_dir: Path
    ) -> None:
        provider = _DummyProvider()
        provider.transcribe_batch(video_folder, data_dir=tmp_data_dir)
        json_files = list((tmp_data_dir / "transcriptions").glob("*.json"))
        assert len(json_files) == 5

    def test_batch_json_is_valid(
        self, video_folder: Path, tmp_data_dir: Path
    ) -> None:
        provider = _DummyProvider()
        provider.transcribe_batch(video_folder, data_dir=tmp_data_dir)
        for json_file in (tmp_data_dir / "transcriptions").glob("*.json"):
            data = json.loads(json_file.read_text())
            assert "source_file" in data
            assert "segments" in data

    def test_batch_skips_already_transcribed_by_default(
        self, video_folder: Path, tmp_data_dir: Path
    ) -> None:
        provider = _DummyProvider()
        # First run – transcribe all
        provider.transcribe_batch(video_folder, data_dir=tmp_data_dir)

        call_count = [0]
        original_transcribe = provider.transcribe

        def counting_transcribe(video_path: Path | str) -> Transcription:
            call_count[0] += 1
            return original_transcribe(video_path)

        provider.transcribe = counting_transcribe  # type: ignore[method-assign]

        # Second run – all should be skipped
        provider.transcribe_batch(video_folder, data_dir=tmp_data_dir)
        assert call_count[0] == 0

    def test_batch_force_flag_retranscribes(
        self, video_folder: Path, tmp_data_dir: Path
    ) -> None:
        provider = _DummyProvider()
        provider.transcribe_batch(video_folder, data_dir=tmp_data_dir)

        call_count = [0]
        original_transcribe = provider.transcribe

        def counting_transcribe(video_path: Path | str) -> Transcription:
            call_count[0] += 1
            return original_transcribe(video_path)

        provider.transcribe = counting_transcribe  # type: ignore[method-assign]

        # Second run with force=True – all should be re-transcribed
        provider.transcribe_batch(video_folder, data_dir=tmp_data_dir, force=True)
        assert call_count[0] == 5

    def test_batch_calls_progress_callback(
        self, video_folder: Path, tmp_data_dir: Path
    ) -> None:
        provider = _DummyProvider()
        progress_calls: list[tuple[str, int, int]] = []

        def on_progress(filename: str, current: int, total: int) -> None:
            progress_calls.append((filename, current, total))

        provider.transcribe_batch(
            video_folder, on_progress=on_progress, data_dir=tmp_data_dir
        )
        assert len(progress_calls) == 5
        # All calls should have the same total
        for _, _, total in progress_calls:
            assert total == 5
        # current indices are 1-based and monotonically increasing
        currents = [c for _, c, _ in progress_calls]
        assert currents == sorted(currents)
        assert currents[0] == 1

    def test_batch_calls_progress_for_skipped_files(
        self, video_folder: Path, tmp_data_dir: Path
    ) -> None:
        provider = _DummyProvider()
        provider.transcribe_batch(video_folder, data_dir=tmp_data_dir)

        progress_calls: list[tuple[str, int, int]] = []
        provider.transcribe_batch(
            video_folder,
            on_progress=lambda f, c, t: progress_calls.append((f, c, t)),
            data_dir=tmp_data_dir,
        )
        # Even skipped files should trigger the callback
        assert len(progress_calls) == 5

    def test_batch_continues_on_transcription_error(
        self, tmp_path: Path, tmp_data_dir: Path
    ) -> None:
        """A failing transcribe() should log a warning and continue."""
        folder = tmp_path / "vids"
        folder.mkdir()
        (folder / "good.mp4").touch()
        (folder / "bad.mkv").touch()

        class _ErrorProvider(TranscriptionProvider):
            def transcribe(self, video_path: Path | str) -> Transcription:
                if Path(video_path).suffix == ".mkv":
                    raise RuntimeError("Simulated decode error")
                return Transcription(
                    source_file=str(video_path),
                    duration=3.0,
                    language="en",
                    model_used="dummy",
                    segments=[Segment(start=0.0, end=1.0, text="ok")],
                )

        provider = _ErrorProvider()
        results = provider.transcribe_batch(folder, data_dir=tmp_data_dir)
        # Only the good file should be in results
        assert len(results) == 1
        assert results[0].source_file.endswith("good.mp4")

    def test_batch_returns_empty_list_for_no_videos(
        self, empty_folder: Path, tmp_data_dir: Path
    ) -> None:
        provider = _DummyProvider()
        results = provider.transcribe_batch(empty_folder, data_dir=tmp_data_dir)
        assert results == []


# ===========================================================================
# Tests: WhisperLocalProvider (mocked faster_whisper.WhisperModel)
# ===========================================================================


class TestWhisperLocalProvider:
    def test_init_passes_correct_params_to_model(self) -> None:
        with patch("snipsnap.transcription.whisper_local.WhisperModel") as mock_cls:
            mock_cls.return_value = MagicMock()
            mock_cls.return_value.transcribe.return_value = ([], MagicMock(duration=0.0, language="en"))
            WhisperLocalProvider(model_size="tiny", device="cpu", compute_type="int8")
            mock_cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")

    def test_transcribe_returns_transcription_object(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        assert isinstance(result, Transcription)

    def test_transcribe_segments_have_required_fields(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        for seg in result.segments:
            assert isinstance(seg.start, float)
            assert isinstance(seg.end, float)
            assert isinstance(seg.text, str)

    def test_transcribe_sets_source_file(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        assert result.source_file == str(dummy_video_file)

    def test_transcribe_sets_model_used(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        assert result.model_used == "tiny"

    def test_transcribe_sets_language_from_info(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        _, model_instance = mock_whisper_model
        info = model_instance.transcribe.return_value[1]
        info.language = "fr"
        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        assert result.language == "fr"

    def test_transcribe_sets_duration_from_info(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        _, model_instance = mock_whisper_model
        info = model_instance.transcribe.return_value[1]
        info.duration = 5.0
        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        assert result.duration == pytest.approx(5.0)

    def test_transcribe_segments_chronologically_ordered(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        """Segments must be sorted by start time."""
        _, model_instance = mock_whisper_model
        # Inject out-of-order segments
        seg_b = MagicMock()
        seg_b.start = 3.0
        seg_b.end = 5.0
        seg_b.text = " second"
        seg_a = MagicMock()
        seg_a.start = 0.0
        seg_a.end = 3.0
        seg_a.text = " first"
        info = MagicMock()
        info.duration = 5.0
        info.language = "en"
        model_instance.transcribe.return_value = ([seg_b, seg_a], info)

        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        starts = [s.start for s in result.segments]
        assert starts == sorted(starts)

    def test_transcribe_filters_zero_length_segments(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        _, model_instance = mock_whisper_model
        # Add a zero-length and a negative-length segment
        seg_zero = MagicMock()
        seg_zero.start = 1.0
        seg_zero.end = 1.0
        seg_zero.text = " empty"
        seg_neg = MagicMock()
        seg_neg.start = 2.0
        seg_neg.end = 1.5
        seg_neg.text = " negative"
        seg_valid = MagicMock()
        seg_valid.start = 0.0
        seg_valid.end = 1.0
        seg_valid.text = " valid"
        info = MagicMock()
        info.duration = 5.0
        info.language = "en"
        model_instance.transcribe.return_value = ([seg_zero, seg_neg, seg_valid], info)

        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        assert len(result.segments) == 1
        assert result.segments[0].text == "valid"

    def test_transcribe_strips_leading_trailing_whitespace_from_text(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        _, model_instance = mock_whisper_model
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 2.0
        seg.text = "  Hello world  "
        info = MagicMock()
        info.duration = 2.0
        info.language = "en"
        model_instance.transcribe.return_value = ([seg], info)

        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        assert result.segments[0].text == "Hello world"

    def test_transcribe_uses_vad_filter(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        _, model_instance = mock_whisper_model
        provider = WhisperLocalProvider(model_size="tiny")
        provider.transcribe(dummy_video_file)
        _, kwargs = model_instance.transcribe.call_args
        assert kwargs.get("vad_filter") is True

    def test_transcribe_raises_for_nonexistent_file(
        self, mock_whisper_model: tuple, tmp_path: Path
    ) -> None:
        provider = WhisperLocalProvider(model_size="tiny")
        with pytest.raises(FileNotFoundError):
            provider.transcribe(tmp_path / "missing.mp4")

    def test_transcribe_video_without_speech_returns_empty_segments(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        """VAD filter may yield no segments for silent/music-only content."""
        _, model_instance = mock_whisper_model
        info = MagicMock()
        info.duration = 5.0
        info.language = "en"
        model_instance.transcribe.return_value = ([], info)

        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        assert result.segments == []
        assert isinstance(result, Transcription)


# ===========================================================================
# Tests: Segment validity invariants
# ===========================================================================


class TestSegmentInvariants:
    """Verify the key invariants stated in the feature requirements."""

    def test_each_segment_start_less_than_end(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        for seg in result.segments:
            assert seg.start < seg.end, (
                f"Segment has start >= end: start={seg.start}, end={seg.end}"
            )

    def test_consecutive_segments_start_nondecreasing(
        self, mock_whisper_model: tuple, dummy_video_file: Path
    ) -> None:
        provider = WhisperLocalProvider(model_size="tiny")
        result = provider.transcribe(dummy_video_file)
        for i in range(len(result.segments) - 1):
            assert result.segments[i].start <= result.segments[i + 1].start

    def test_transcription_persisted_as_valid_json(
        self, mock_whisper_model: tuple, dummy_video_file: Path, tmp_data_dir: Path
    ) -> None:
        from snipsnap.storage import save_transcription

        provider = WhisperLocalProvider(model_size="tiny")
        transcription = provider.transcribe(dummy_video_file)
        json_path = save_transcription(transcription, tmp_data_dir)

        assert json_path.exists()
        loaded = json.loads(json_path.read_text())
        assert "source_file" in loaded
        assert "segments" in loaded
        assert "language" in loaded
        assert "duration" in loaded

    def test_transcription_json_naming_convention(
        self, mock_whisper_model: tuple, dummy_video_file: Path, tmp_data_dir: Path
    ) -> None:
        """JSON filename must be derived from the source video stem."""
        from snipsnap.storage import save_transcription

        provider = WhisperLocalProvider(model_size="tiny")
        transcription = provider.transcribe(dummy_video_file)
        json_path = save_transcription(transcription, tmp_data_dir)

        # The JSON stem must match the video stem
        assert json_path.stem == dummy_video_file.stem


# ===========================================================================
# Integration test: real faster-whisper 'tiny' model + real ffmpeg video
# ===========================================================================


@pytest.mark.integration
class TestIntegrationRealTranscription:
    """End-to-end test using the actual faster-whisper 'tiny' model.

    Downloads ~75 MB on first run; cached in ~/.cache/huggingface/hub/.
    """

    @pytest.fixture(scope="class")
    def tiny_provider(self) -> WhisperLocalProvider:
        """Create a WhisperLocalProvider with the 'tiny' model."""
        try:
            return WhisperLocalProvider(model_size="tiny", device="cpu", compute_type="int8")
        except Exception as exc:
            pytest.skip(f"Could not load tiny Whisper model: {exc}")

    def test_transcribe_returns_valid_transcription(
        self,
        tiny_provider: WhisperLocalProvider,
        test_video: Path,
    ) -> None:
        result = tiny_provider.transcribe(test_video)
        assert isinstance(result, Transcription)
        assert result.duration > 0
        assert result.language  # non-empty language code

    def test_segments_have_valid_timestamps(
        self,
        tiny_provider: WhisperLocalProvider,
        test_video: Path,
    ) -> None:
        result = tiny_provider.transcribe(test_video)
        for seg in result.segments:
            assert seg.start >= 0.0
            assert seg.end > seg.start

    def test_segments_are_chronologically_ordered(
        self,
        tiny_provider: WhisperLocalProvider,
        test_video: Path,
    ) -> None:
        result = tiny_provider.transcribe(test_video)
        starts = [s.start for s in result.segments]
        assert starts == sorted(starts)

    def test_batch_skips_existing_transcription(
        self,
        tiny_provider: WhisperLocalProvider,
        test_video: Path,
        tmp_data_dir: Path,
    ) -> None:
        folder = test_video.parent
        # First pass
        tiny_provider.transcribe_batch(folder, data_dir=tmp_data_dir)
        json_files_after_first = list((tmp_data_dir / "transcriptions").glob("*.json"))
        assert len(json_files_after_first) >= 1

        # Record mtime of first transcription file
        first_json = json_files_after_first[0]
        mtime_before = first_json.stat().st_mtime

        # Second pass – should skip without modifying the file
        tiny_provider.transcribe_batch(folder, data_dir=tmp_data_dir)
        mtime_after = first_json.stat().st_mtime
        assert mtime_before == mtime_after, "Existing transcription JSON was modified"

    def test_transcription_json_is_valid_and_has_required_fields(
        self,
        tiny_provider: WhisperLocalProvider,
        test_video: Path,
        tmp_data_dir: Path,
    ) -> None:
        from snipsnap.storage import save_transcription

        result = tiny_provider.transcribe(test_video)
        json_path = save_transcription(result, tmp_data_dir)

        data = json.loads(json_path.read_text())
        assert "source_file" in data
        assert "duration" in data
        assert "language" in data
        assert "model_used" in data
        assert "segments" in data
        assert isinstance(data["segments"], list)
