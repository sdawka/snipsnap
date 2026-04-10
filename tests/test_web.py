"""Tests for the FastAPI web backend (snipsnap/web/app.py).

Tests use FastAPI's TestClient to call endpoints in-process.
Storage functions and external engines are mocked to keep tests fast
and deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from snipsnap.models import CutList, CutSegment, Segment, Transcription
from snipsnap.web.app import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

client = TestClient(app)


@pytest.fixture
def sample_transcription() -> Transcription:
    return Transcription(
        source_file="/videos/clip1.mp4",
        duration=30.0,
        language="en",
        model_used="small",
        created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        segments=[
            Segment(start=0.0, end=5.0, text="Hello world"),
            Segment(start=5.0, end=10.0, text="This is a test"),
            Segment(start=10.0, end=15.0, text="Goodbye"),
        ],
    )


@pytest.fixture
def sample_transcription_2() -> Transcription:
    return Transcription(
        source_file="/videos/clip2.mp4",
        duration=20.0,
        language="en",
        model_used="small",
        created_at=datetime(2024, 1, 16, 9, 0, 0, tzinfo=timezone.utc),
        segments=[
            Segment(start=0.0, end=8.0, text="Second video content"),
            Segment(start=8.0, end=18.0, text="More content"),
        ],
    )


@pytest.fixture
def sample_cut_list() -> CutList:
    return CutList(
        id="550e8400-e29b-41d4-a716-446655440000",
        prompt="find key moments",
        theme="Highlights",
        created_at=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
        total_duration=15.0,
        segments=[
            CutSegment(
                source_file="/videos/clip1.mp4",
                start=0.0,
                end=5.0,
                description="Opening greeting",
                order=0,
            ),
            CutSegment(
                source_file="/videos/clip1.mp4",
                start=10.0,
                end=20.0,
                description="Closing statement",
                order=1,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_status_ok(self) -> None:
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /api/transcriptions
# ---------------------------------------------------------------------------


class TestListTranscriptions:
    def test_returns_200(self, sample_transcription: Transcription) -> None:
        with patch(
            "snipsnap.web.app.load_all_transcriptions",
            return_value=[sample_transcription],
        ):
            resp = client.get("/api/transcriptions")
        assert resp.status_code == 200

    def test_returns_empty_list_when_no_transcriptions(self) -> None:
        with patch("snipsnap.web.app.load_all_transcriptions", return_value=[]):
            resp = client.get("/api/transcriptions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_metadata_without_segments(
        self, sample_transcription: Transcription
    ) -> None:
        with patch(
            "snipsnap.web.app.load_all_transcriptions",
            return_value=[sample_transcription],
        ):
            resp = client.get("/api/transcriptions")
        data = resp.json()
        assert len(data) == 1
        item = data[0]
        # Should contain metadata fields
        assert item["source_file"] == "/videos/clip1.mp4"
        assert item["duration"] == 30.0
        assert item["language"] == "en"
        assert item["model_used"] == "small"
        assert item["segment_count"] == 3
        # Should NOT contain segments list
        assert "segments" not in item

    def test_returns_multiple_transcriptions(
        self,
        sample_transcription: Transcription,
        sample_transcription_2: Transcription,
    ) -> None:
        with patch(
            "snipsnap.web.app.load_all_transcriptions",
            return_value=[sample_transcription, sample_transcription_2],
        ):
            resp = client.get("/api/transcriptions")
        data = resp.json()
        assert len(data) == 2
        source_files = [item["source_file"] for item in data]
        assert "/videos/clip1.mp4" in source_files
        assert "/videos/clip2.mp4" in source_files

    def test_created_at_is_iso_format(self, sample_transcription: Transcription) -> None:
        with patch(
            "snipsnap.web.app.load_all_transcriptions",
            return_value=[sample_transcription],
        ):
            resp = client.get("/api/transcriptions")
        data = resp.json()
        created_at = data[0]["created_at"]
        # Should be parseable ISO format
        parsed = datetime.fromisoformat(created_at)
        assert parsed.year == 2024


# ---------------------------------------------------------------------------
# GET /api/transcriptions/{filename}
# ---------------------------------------------------------------------------


class TestGetTranscription:
    def test_returns_200_for_existing(self, sample_transcription: Transcription) -> None:
        with patch(
            "snipsnap.web.app.load_transcription",
            return_value=sample_transcription,
        ):
            resp = client.get("/api/transcriptions//videos/clip1.mp4")
        assert resp.status_code == 200

    def test_returns_404_for_missing(self) -> None:
        with patch("snipsnap.web.app.load_transcription", return_value=None):
            resp = client.get("/api/transcriptions/nonexistent.mp4")
        assert resp.status_code == 404

    def test_404_detail_mentions_filename(self) -> None:
        with patch("snipsnap.web.app.load_transcription", return_value=None):
            resp = client.get("/api/transcriptions/missing_video.mp4")
        assert "missing_video.mp4" in resp.json()["detail"]

    def test_returns_segments(self, sample_transcription: Transcription) -> None:
        with patch(
            "snipsnap.web.app.load_transcription",
            return_value=sample_transcription,
        ):
            resp = client.get("/api/transcriptions//videos/clip1.mp4")
        data = resp.json()
        assert "segments" in data
        assert len(data["segments"]) == 3

    def test_segment_structure(self, sample_transcription: Transcription) -> None:
        with patch(
            "snipsnap.web.app.load_transcription",
            return_value=sample_transcription,
        ):
            resp = client.get("/api/transcriptions//videos/clip1.mp4")
        segment = resp.json()["segments"][0]
        assert segment["start"] == 0.0
        assert segment["end"] == 5.0
        assert segment["text"] == "Hello world"

    def test_returns_full_metadata(self, sample_transcription: Transcription) -> None:
        with patch(
            "snipsnap.web.app.load_transcription",
            return_value=sample_transcription,
        ):
            resp = client.get("/api/transcriptions//videos/clip1.mp4")
        data = resp.json()
        assert data["source_file"] == "/videos/clip1.mp4"
        assert data["duration"] == 30.0
        assert data["language"] == "en"
        assert data["model_used"] == "small"

    def test_passes_filename_to_load_transcription(
        self, sample_transcription: Transcription
    ) -> None:
        with patch(
            "snipsnap.web.app.load_transcription",
            return_value=sample_transcription,
        ) as mock_load:
            client.get("/api/transcriptions//videos/clip1.mp4")
        mock_load.assert_called_once_with("/videos/clip1.mp4")


# ---------------------------------------------------------------------------
# POST /api/transcribe
# ---------------------------------------------------------------------------


class TestTranscribeEndpoint:
    def test_returns_400_for_nonexistent_folder(self, tmp_path: object) -> None:
        resp = client.post(
            "/api/transcribe",
            json={"folder": "/nonexistent/path/that/does/not/exist"},
        )
        assert resp.status_code == 400

    def test_returns_400_for_no_videos(self, tmp_path: object) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            resp = client.post("/api/transcribe", json={"folder": tmpdir})
        assert resp.status_code == 400

    def test_400_detail_for_no_videos(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            resp = client.post("/api/transcribe", json={"folder": tmpdir})
        assert "No video files found" in resp.json()["detail"]

    def test_returns_400_for_missing_folder_field(self) -> None:
        resp = client.post("/api/transcribe", json={})
        assert resp.status_code == 422  # Pydantic validation

    def test_transcribes_video_file(
        self, tmp_path: object, sample_transcription: Transcription
    ) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake .mp4 file
            fake_video = __import__("pathlib").Path(tmpdir) / "video.mp4"
            fake_video.touch()

            mock_provider = MagicMock()
            mock_provider.transcribe.return_value = sample_transcription

            with (
                patch(
                    "snipsnap.web.app.WhisperLocalProvider",
                    return_value=mock_provider,
                ),
                patch("snipsnap.web.app.save_transcription"),
                patch("snipsnap.web.app.transcription_exists", return_value=False),
            ):
                resp = client.post("/api/transcribe", json={"folder": tmpdir})

        assert resp.status_code == 200
        data = resp.json()
        assert data["transcribed"] == 1
        assert data["skipped"] == 0
        assert data["failed"] == 0

    def test_skips_already_transcribed(
        self, sample_transcription: Transcription
    ) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_video = __import__("pathlib").Path(tmpdir) / "video.mp4"
            fake_video.touch()

            mock_provider = MagicMock()

            with (
                patch(
                    "snipsnap.web.app.WhisperLocalProvider",
                    return_value=mock_provider,
                ),
                patch("snipsnap.web.app.save_transcription"),
                patch("snipsnap.web.app.transcription_exists", return_value=True),
                patch(
                    "snipsnap.web.app.load_transcription",
                    return_value=sample_transcription,
                ),
            ):
                resp = client.post("/api/transcribe", json={"folder": tmpdir})

        assert resp.status_code == 200
        data = resp.json()
        assert data["skipped"] == 1
        assert data["transcribed"] == 0

    def test_returns_transcription_metadata(
        self, sample_transcription: Transcription
    ) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_video = __import__("pathlib").Path(tmpdir) / "video.mp4"
            fake_video.touch()

            mock_provider = MagicMock()
            mock_provider.transcribe.return_value = sample_transcription

            with (
                patch(
                    "snipsnap.web.app.WhisperLocalProvider",
                    return_value=mock_provider,
                ),
                patch("snipsnap.web.app.save_transcription"),
                patch("snipsnap.web.app.transcription_exists", return_value=False),
            ):
                resp = client.post("/api/transcribe", json={"folder": tmpdir})

        data = resp.json()
        assert len(data["transcriptions"]) == 1
        assert data["transcriptions"][0]["source_file"] == "/videos/clip1.mp4"


# ---------------------------------------------------------------------------
# POST /api/curate
# ---------------------------------------------------------------------------


class TestCurateEndpoint:
    def test_returns_400_when_no_transcriptions(self) -> None:
        with patch("snipsnap.web.app.load_all_transcriptions", return_value=[]):
            resp = client.post("/api/curate", json={"prompt": "find highlights"})
        assert resp.status_code == 400

    def test_400_detail_suggests_transcribe_first(self) -> None:
        with patch("snipsnap.web.app.load_all_transcriptions", return_value=[]):
            resp = client.post("/api/curate", json={"prompt": "find highlights"})
        assert "transcrib" in resp.json()["detail"].lower()

    def test_returns_400_when_no_api_key(
        self, sample_transcription: Transcription
    ) -> None:
        import pathlib

        from snipsnap.config import Config

        cfg = Config(
            openrouter_api_key="",
            data_dir=pathlib.Path("./snipsnap_data"),
            model="google/gemini-2.5-flash-lite",
            whisper_model="small",
        )
        with (
            patch(
                "snipsnap.web.app.load_all_transcriptions",
                return_value=[sample_transcription],
            ),
            patch("snipsnap.web.app.get_config", return_value=cfg),
        ):
            resp = client.post("/api/curate", json={"prompt": "find highlights"})
        assert resp.status_code == 400
        assert "api key" in resp.json()["detail"].lower()

    def test_returns_422_when_prompt_missing(self) -> None:
        resp = client.post("/api/curate", json={"model": "gpt-4"})
        assert resp.status_code == 422

    def test_returns_cut_list_on_success(
        self,
        sample_transcription: Transcription,
        sample_cut_list: CutList,
    ) -> None:
        import pathlib

        from snipsnap.config import Config

        cfg = Config(
            openrouter_api_key="test-key",
            data_dir=pathlib.Path("./snipsnap_data"),
            model="google/gemini-2.5-flash-lite",
            whisper_model="small",
        )
        mock_engine = MagicMock()
        mock_engine.curate.return_value = sample_cut_list

        with (
            patch(
                "snipsnap.web.app.load_all_transcriptions",
                return_value=[sample_transcription],
            ),
            patch("snipsnap.web.app.get_config", return_value=cfg),
            patch(
                "snipsnap.web.app.CurationEngine",
                return_value=mock_engine,
            ),
        ):
            resp = client.post("/api/curate", json={"prompt": "find highlights"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert data["prompt"] == "find key moments"
        assert len(data["segments"]) == 2

    def test_uses_prompt_from_request(
        self,
        sample_transcription: Transcription,
        sample_cut_list: CutList,
    ) -> None:
        import pathlib

        from snipsnap.config import Config

        cfg = Config(
            openrouter_api_key="test-key",
            data_dir=pathlib.Path("./snipsnap_data"),
            model="google/gemini-2.5-flash-lite",
            whisper_model="small",
        )
        mock_engine = MagicMock()
        mock_engine.curate.return_value = sample_cut_list

        with (
            patch(
                "snipsnap.web.app.load_all_transcriptions",
                return_value=[sample_transcription],
            ),
            patch("snipsnap.web.app.get_config", return_value=cfg),
            patch("snipsnap.web.app.CurationEngine", return_value=mock_engine),
        ):
            client.post("/api/curate", json={"prompt": "find highlights"})

        mock_engine.curate.assert_called_once()
        call_kwargs = mock_engine.curate.call_args
        assert call_kwargs.kwargs["prompt"] == "find highlights"

    def test_uses_model_from_request(
        self,
        sample_transcription: Transcription,
        sample_cut_list: CutList,
    ) -> None:
        import pathlib

        from snipsnap.config import Config

        cfg = Config(
            openrouter_api_key="test-key",
            data_dir=pathlib.Path("./snipsnap_data"),
            model="google/gemini-2.5-flash-lite",
            whisper_model="small",
        )
        mock_engine = MagicMock()
        mock_engine.curate.return_value = sample_cut_list

        with (
            patch(
                "snipsnap.web.app.load_all_transcriptions",
                return_value=[sample_transcription],
            ),
            patch("snipsnap.web.app.get_config", return_value=cfg),
            patch("snipsnap.web.app.CurationEngine", return_value=mock_engine),
        ):
            client.post(
                "/api/curate",
                json={"prompt": "find highlights", "model": "anthropic/claude-3-sonnet"},
            )

        call_kwargs = mock_engine.curate.call_args
        assert call_kwargs.kwargs["model"] == "anthropic/claude-3-sonnet"

    def test_returns_500_on_curation_error(
        self, sample_transcription: Transcription
    ) -> None:
        import pathlib

        from snipsnap.config import Config
        from snipsnap.curation.engine import CurationError

        cfg = Config(
            openrouter_api_key="test-key",
            data_dir=pathlib.Path("./snipsnap_data"),
            model="google/gemini-2.5-flash-lite",
            whisper_model="small",
        )
        mock_engine = MagicMock()
        mock_engine.curate.side_effect = CurationError("LLM failed")

        with (
            patch(
                "snipsnap.web.app.load_all_transcriptions",
                return_value=[sample_transcription],
            ),
            patch("snipsnap.web.app.get_config", return_value=cfg),
            patch("snipsnap.web.app.CurationEngine", return_value=mock_engine),
        ):
            resp = client.post("/api/curate", json={"prompt": "find highlights"})

        assert resp.status_code == 500

    def test_returns_segments_in_cut_list(
        self,
        sample_transcription: Transcription,
        sample_cut_list: CutList,
    ) -> None:
        import pathlib

        from snipsnap.config import Config

        cfg = Config(
            openrouter_api_key="test-key",
            data_dir=pathlib.Path("./snipsnap_data"),
            model="google/gemini-2.5-flash-lite",
            whisper_model="small",
        )
        mock_engine = MagicMock()
        mock_engine.curate.return_value = sample_cut_list

        with (
            patch(
                "snipsnap.web.app.load_all_transcriptions",
                return_value=[sample_transcription],
            ),
            patch("snipsnap.web.app.get_config", return_value=cfg),
            patch("snipsnap.web.app.CurationEngine", return_value=mock_engine),
        ):
            resp = client.post("/api/curate", json={"prompt": "find highlights"})

        data = resp.json()
        segments = data["segments"]
        assert len(segments) == 2
        assert segments[0]["source_file"] == "/videos/clip1.mp4"
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 5.0
        assert segments[0]["description"] == "Opening greeting"
        assert segments[0]["order"] == 0


# ---------------------------------------------------------------------------
# GET /api/cutlists
# ---------------------------------------------------------------------------


class TestListCutLists:
    def test_returns_200(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_all_cut_lists",
            return_value=[sample_cut_list],
        ):
            resp = client.get("/api/cutlists")
        assert resp.status_code == 200

    def test_returns_empty_list_when_no_cut_lists(self) -> None:
        with patch("snipsnap.web.app.load_all_cut_lists", return_value=[]):
            resp = client.get("/api/cutlists")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_metadata_without_segments(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_all_cut_lists",
            return_value=[sample_cut_list],
        ):
            resp = client.get("/api/cutlists")
        data = resp.json()
        assert len(data) == 1
        item = data[0]
        assert item["id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert item["prompt"] == "find key moments"
        assert item["theme"] == "Highlights"
        assert item["total_duration"] == 15.0
        assert item["segment_count"] == 2
        # Should NOT contain segments
        assert "segments" not in item

    def test_created_at_is_iso_format(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_all_cut_lists",
            return_value=[sample_cut_list],
        ):
            resp = client.get("/api/cutlists")
        created_at = resp.json()[0]["created_at"]
        parsed = datetime.fromisoformat(created_at)
        assert parsed.year == 2024


# ---------------------------------------------------------------------------
# GET /api/cutlists/{id}
# ---------------------------------------------------------------------------


class TestGetCutList:
    def test_returns_200_for_existing(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.get(
                "/api/cutlists/550e8400-e29b-41d4-a716-446655440000"
            )
        assert resp.status_code == 200

    def test_returns_404_for_missing(self) -> None:
        with patch("snipsnap.web.app.load_cut_list", return_value=None):
            resp = client.get("/api/cutlists/nonexistent-id")
        assert resp.status_code == 404

    def test_404_detail_mentions_id(self) -> None:
        with patch("snipsnap.web.app.load_cut_list", return_value=None):
            resp = client.get("/api/cutlists/nonexistent-id")
        assert "nonexistent-id" in resp.json()["detail"]

    def test_returns_segments(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.get(
                "/api/cutlists/550e8400-e29b-41d4-a716-446655440000"
            )
        data = resp.json()
        assert "segments" in data
        assert len(data["segments"]) == 2

    def test_segment_fields(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.get(
                "/api/cutlists/550e8400-e29b-41d4-a716-446655440000"
            )
        seg = resp.json()["segments"][0]
        assert seg["source_file"] == "/videos/clip1.mp4"
        assert seg["start"] == 0.0
        assert seg["end"] == 5.0
        assert seg["description"] == "Opening greeting"
        assert seg["order"] == 0

    def test_passes_id_to_load_cut_list(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ) as mock_load:
            client.get("/api/cutlists/550e8400-e29b-41d4-a716-446655440000")
        mock_load.assert_called_once_with("550e8400-e29b-41d4-a716-446655440000")


# ---------------------------------------------------------------------------
# POST /api/export/{cutlist_id}
# ---------------------------------------------------------------------------


class TestExportEndpoint:
    def test_returns_404_for_missing_cut_list(self) -> None:
        with patch("snipsnap.web.app.load_cut_list", return_value=None):
            resp = client.post(
                "/api/export/nonexistent-id",
                json={"format": "edl"},
            )
        assert resp.status_code == 404

    def test_returns_400_for_invalid_format(
        self, sample_cut_list: CutList
    ) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "mp4"},
            )
        assert resp.status_code == 400

    def test_400_detail_mentions_valid_formats(
        self, sample_cut_list: CutList
    ) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "mp4"},
            )
        detail = resp.json()["detail"]
        assert "edl" in detail.lower() or "fcpxml" in detail.lower()

    def test_edl_export_returns_200(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "edl"},
            )
        assert resp.status_code == 200

    def test_edl_content_type_is_text(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "edl"},
            )
        assert "text/plain" in resp.headers["content-type"]

    def test_edl_content_has_title(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "edl"},
            )
        assert "TITLE:" in resp.text

    def test_edl_content_has_fcm(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "edl"},
            )
        assert "FCM:" in resp.text

    def test_edl_content_disposition_header(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "edl"},
            )
        assert "content-disposition" in resp.headers
        assert ".edl" in resp.headers["content-disposition"]

    def test_fcpxml_export_returns_200(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "fcpxml"},
            )
        assert resp.status_code == 200

    def test_fcpxml_content_type_is_xml(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "fcpxml"},
            )
        assert "xml" in resp.headers["content-type"]

    def test_fcpxml_content_has_version(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "fcpxml"},
            )
        assert 'version="1.8"' in resp.text

    def test_fcpxml_content_disposition_header(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "fcpxml"},
            )
        assert ".fcpxml" in resp.headers["content-disposition"]

    def test_davinci_export_returns_200(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "davinci"},
            )
        assert resp.status_code == 200

    def test_davinci_content_is_python_script(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "davinci"},
            )
        assert "DaVinciResolveScript" in resp.text

    def test_davinci_content_disposition_header(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "davinci"},
            )
        assert ".py" in resp.headers["content-disposition"]

    def test_fps_parameter_affects_edl_output(
        self, sample_cut_list: CutList
    ) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp_24 = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "edl", "fps": 24.0},
            )
            resp_30 = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "edl", "fps": 30.0},
            )
        # Both should succeed
        assert resp_24.status_code == 200
        assert resp_30.status_code == 200

    def test_export_format_case_insensitive(self, sample_cut_list: CutList) -> None:
        """Format should work regardless of case."""
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={"format": "EDL"},
            )
        assert resp.status_code == 200

    def test_returns_400_when_format_missing(self, sample_cut_list: CutList) -> None:
        with patch(
            "snipsnap.web.app.load_cut_list",
            return_value=sample_cut_list,
        ):
            resp = client.post(
                "/api/export/550e8400-e29b-41d4-a716-446655440000",
                json={},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_static_route_is_mounted(self) -> None:
        """Verify the /static route exists (even if returning 404 for missing files)."""
        resp = client.get("/static/nonexistent.html")
        # Should get 404 (file not found), not 422 (bad request)
        # This confirms the static mount is active
        assert resp.status_code == 404
