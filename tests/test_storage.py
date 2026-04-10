"""Unit tests for snipsnap.storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from snipsnap.models import CutList, Segment, Transcription
from snipsnap.storage import (
    load_all_cut_lists,
    load_all_transcriptions,
    load_cut_list,
    load_transcription,
    save_cut_list,
    save_transcription,
    transcription_exists,
)


class TestSaveLoadTranscription:
    def test_save_transcription_creates_file(
        self, tmp_data_dir: Path, sample_transcription: Transcription
    ) -> None:
        path = save_transcription(sample_transcription, data_dir=tmp_data_dir)
        assert path.exists()

    def test_save_transcription_returns_correct_path(
        self, tmp_data_dir: Path, sample_transcription: Transcription
    ) -> None:
        path = save_transcription(sample_transcription, data_dir=tmp_data_dir)
        assert path.parent == tmp_data_dir / "transcriptions"
        # Filename is <stem>_<hash8>.json — starts with the video stem
        assert path.name.startswith("test_video_")
        assert path.suffix == ".json"

    def test_save_transcription_is_valid_json(
        self, tmp_data_dir: Path, sample_transcription: Transcription
    ) -> None:
        path = save_transcription(sample_transcription, data_dir=tmp_data_dir)
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_save_transcription_json_has_required_fields(
        self, tmp_data_dir: Path, sample_transcription: Transcription
    ) -> None:
        path = save_transcription(sample_transcription, data_dir=tmp_data_dir)
        data = json.loads(path.read_text())
        assert "source_file" in data
        assert "duration" in data
        assert "language" in data
        assert "model_used" in data
        assert "segments" in data
        assert "created_at" in data

    def test_save_transcription_creates_directory_if_missing(
        self, tmp_path: Path, sample_transcription: Transcription
    ) -> None:
        data_dir = tmp_path / "new_data_dir" / "nested"
        save_transcription(sample_transcription, data_dir=data_dir)
        assert (data_dir / "transcriptions").exists()

    def test_load_transcription_round_trip(
        self, tmp_data_dir: Path, sample_transcription: Transcription
    ) -> None:
        save_transcription(sample_transcription, data_dir=tmp_data_dir)
        loaded = load_transcription(
            sample_transcription.source_file, data_dir=tmp_data_dir
        )
        assert loaded is not None
        assert loaded.source_file == sample_transcription.source_file
        assert loaded.duration == sample_transcription.duration
        assert loaded.language == sample_transcription.language
        assert loaded.model_used == sample_transcription.model_used
        assert len(loaded.segments) == len(sample_transcription.segments)

    def test_load_transcription_segments_match(
        self, tmp_data_dir: Path, sample_transcription: Transcription
    ) -> None:
        save_transcription(sample_transcription, data_dir=tmp_data_dir)
        loaded = load_transcription(
            sample_transcription.source_file, data_dir=tmp_data_dir
        )
        assert loaded is not None
        for orig, loaded_seg in zip(sample_transcription.segments, loaded.segments):
            assert loaded_seg.start == orig.start
            assert loaded_seg.end == orig.end
            assert loaded_seg.text == orig.text

    def test_load_transcription_created_at_preserved(
        self, tmp_data_dir: Path, sample_transcription: Transcription
    ) -> None:
        save_transcription(sample_transcription, data_dir=tmp_data_dir)
        loaded = load_transcription(
            sample_transcription.source_file, data_dir=tmp_data_dir
        )
        assert loaded is not None
        assert loaded.created_at == sample_transcription.created_at

    def test_load_transcription_returns_none_if_missing(
        self, tmp_data_dir: Path
    ) -> None:
        result = load_transcription("nonexistent.mp4", data_dir=tmp_data_dir)
        assert result is None

    def test_transcription_exists_returns_true_after_save(
        self, tmp_data_dir: Path, sample_transcription: Transcription
    ) -> None:
        save_transcription(sample_transcription, data_dir=tmp_data_dir)
        assert transcription_exists(
            sample_transcription.source_file, data_dir=tmp_data_dir
        )

    def test_transcription_exists_returns_false_before_save(
        self, tmp_data_dir: Path
    ) -> None:
        assert not transcription_exists("not_yet.mp4", data_dir=tmp_data_dir)

    def test_transcription_empty_segments(self, tmp_data_dir: Path) -> None:
        t = Transcription(
            source_file="silence.mp4",
            duration=60.0,
            language="en",
            model_used="small",
            segments=[],
        )
        save_transcription(t, data_dir=tmp_data_dir)
        loaded = load_transcription("silence.mp4", data_dir=tmp_data_dir)
        assert loaded is not None
        assert loaded.segments == []


class TestLoadAllTranscriptions:
    def test_load_all_returns_empty_list_when_no_files(
        self, tmp_data_dir: Path
    ) -> None:
        result = load_all_transcriptions(data_dir=tmp_data_dir)
        assert result == []

    def test_load_all_returns_empty_list_when_dir_missing(
        self, tmp_path: Path
    ) -> None:
        data_dir = tmp_path / "empty_data"
        result = load_all_transcriptions(data_dir=data_dir)
        assert result == []

    def test_load_all_returns_all_saved_transcriptions(
        self, tmp_data_dir: Path, sample_segments: list[Segment]
    ) -> None:
        t1 = Transcription(
            source_file="video1.mp4",
            duration=10.0,
            language="en",
            model_used="small",
            segments=sample_segments[:1],
        )
        t2 = Transcription(
            source_file="video2.mp4",
            duration=20.0,
            language="fr",
            model_used="base",
            segments=sample_segments[:2],
        )
        save_transcription(t1, data_dir=tmp_data_dir)
        save_transcription(t2, data_dir=tmp_data_dir)

        result = load_all_transcriptions(data_dir=tmp_data_dir)
        assert len(result) == 2
        source_files = {t.source_file for t in result}
        assert "video1.mp4" in source_files
        assert "video2.mp4" in source_files

    def test_load_all_skips_malformed_json(self, tmp_data_dir: Path) -> None:
        trans_dir = tmp_data_dir / "transcriptions"
        trans_dir.mkdir(parents=True, exist_ok=True)
        (trans_dir / "bad.json").write_text("not valid json{{{")

        result = load_all_transcriptions(data_dir=tmp_data_dir)
        assert result == []


class TestSaveLoadCutList:
    def test_save_cut_list_creates_file(
        self, tmp_data_dir: Path, sample_cut_list: CutList
    ) -> None:
        path = save_cut_list(sample_cut_list, data_dir=tmp_data_dir)
        assert path.exists()

    def test_save_cut_list_filename_is_uuid(
        self, tmp_data_dir: Path, sample_cut_list: CutList
    ) -> None:
        path = save_cut_list(sample_cut_list, data_dir=tmp_data_dir)
        assert path.stem == sample_cut_list.id

    def test_save_cut_list_is_valid_json(
        self, tmp_data_dir: Path, sample_cut_list: CutList
    ) -> None:
        path = save_cut_list(sample_cut_list, data_dir=tmp_data_dir)
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_save_cut_list_json_has_required_fields(
        self, tmp_data_dir: Path, sample_cut_list: CutList
    ) -> None:
        path = save_cut_list(sample_cut_list, data_dir=tmp_data_dir)
        data = json.loads(path.read_text())
        assert "id" in data
        assert "prompt" in data
        assert "theme" in data
        assert "created_at" in data
        assert "segments" in data
        assert "total_duration" in data

    def test_save_cut_list_creates_directory_if_missing(
        self, tmp_path: Path, sample_cut_list: CutList
    ) -> None:
        data_dir = tmp_path / "new_data"
        save_cut_list(sample_cut_list, data_dir=data_dir)
        assert (data_dir / "cut_lists").exists()

    def test_load_cut_list_round_trip(
        self, tmp_data_dir: Path, sample_cut_list: CutList
    ) -> None:
        save_cut_list(sample_cut_list, data_dir=tmp_data_dir)
        loaded = load_cut_list(sample_cut_list.id, data_dir=tmp_data_dir)
        assert loaded is not None
        assert loaded.id == sample_cut_list.id
        assert loaded.prompt == sample_cut_list.prompt
        assert loaded.theme == sample_cut_list.theme
        assert loaded.total_duration == sample_cut_list.total_duration
        assert len(loaded.segments) == len(sample_cut_list.segments)

    def test_load_cut_list_segments_match(
        self, tmp_data_dir: Path, sample_cut_list: CutList
    ) -> None:
        save_cut_list(sample_cut_list, data_dir=tmp_data_dir)
        loaded = load_cut_list(sample_cut_list.id, data_dir=tmp_data_dir)
        assert loaded is not None
        for orig, loaded_seg in zip(sample_cut_list.segments, loaded.segments):
            assert loaded_seg.source_file == orig.source_file
            assert loaded_seg.start == orig.start
            assert loaded_seg.end == orig.end
            assert loaded_seg.description == orig.description
            assert loaded_seg.order == orig.order

    def test_load_cut_list_returns_none_if_missing(
        self, tmp_data_dir: Path
    ) -> None:
        result = load_cut_list("nonexistent-uuid", data_dir=tmp_data_dir)
        assert result is None

    def test_load_cut_list_empty_segments(self, tmp_data_dir: Path) -> None:
        cl = CutList(
            id="empty-uuid",
            prompt="nothing",
            theme="Empty",
            created_at=datetime.now(timezone.utc),
            segments=[],
            total_duration=0.0,
        )
        save_cut_list(cl, data_dir=tmp_data_dir)
        loaded = load_cut_list("empty-uuid", data_dir=tmp_data_dir)
        assert loaded is not None
        assert loaded.segments == []
        assert loaded.total_duration == 0.0


class TestLoadAllCutLists:
    def test_load_all_cut_lists_empty_when_no_files(
        self, tmp_data_dir: Path
    ) -> None:
        result = load_all_cut_lists(data_dir=tmp_data_dir)
        assert result == []

    def test_load_all_cut_lists_empty_when_dir_missing(
        self, tmp_path: Path
    ) -> None:
        data_dir = tmp_path / "no_data"
        result = load_all_cut_lists(data_dir=data_dir)
        assert result == []

    def test_load_all_cut_lists_returns_all_saved(
        self, tmp_data_dir: Path
    ) -> None:
        cl1 = CutList(
            id="uuid-1",
            prompt="prompt A",
            theme="Theme A",
            created_at=datetime.now(timezone.utc),
            segments=[],
            total_duration=0.0,
        )
        cl2 = CutList(
            id="uuid-2",
            prompt="prompt B",
            theme="Theme B",
            created_at=datetime.now(timezone.utc),
            segments=[],
            total_duration=0.0,
        )
        save_cut_list(cl1, data_dir=tmp_data_dir)
        save_cut_list(cl2, data_dir=tmp_data_dir)

        result = load_all_cut_lists(data_dir=tmp_data_dir)
        assert len(result) == 2
        ids = {cl.id for cl in result}
        assert "uuid-1" in ids
        assert "uuid-2" in ids

    def test_load_all_cut_lists_skips_malformed_json(
        self, tmp_data_dir: Path
    ) -> None:
        cut_dir = tmp_data_dir / "cut_lists"
        cut_dir.mkdir(parents=True, exist_ok=True)
        (cut_dir / "bad.json").write_text("{broken}")

        result = load_all_cut_lists(data_dir=tmp_data_dir)
        assert result == []


class TestDataDirectoryCreation:
    def test_save_transcription_auto_creates_nested_dirs(
        self, tmp_path: Path, sample_transcription: Transcription
    ) -> None:
        """Data directory (including transcriptions subdir) is created automatically."""
        data_dir = tmp_path / "auto_created" / "deep" / "path"
        assert not data_dir.exists()
        save_transcription(sample_transcription, data_dir=data_dir)
        assert (data_dir / "transcriptions").exists()

    def test_save_cut_list_auto_creates_nested_dirs(
        self, tmp_path: Path, sample_cut_list: CutList
    ) -> None:
        """Data directory (including cut_lists subdir) is created automatically."""
        data_dir = tmp_path / "auto_created_cut" / "deep"
        assert not data_dir.exists()
        save_cut_list(sample_cut_list, data_dir=data_dir)
        assert (data_dir / "cut_lists").exists()
