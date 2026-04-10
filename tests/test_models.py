"""Unit tests for snipsnap.models."""

from __future__ import annotations

from datetime import datetime, timezone

from snipsnap.models import CutList, CutSegment, Segment, Transcription


class TestSegment:
    def test_create_segment(self) -> None:
        seg = Segment(start=1.5, end=4.2, text="Hello")
        assert seg.start == 1.5
        assert seg.end == 4.2
        assert seg.text == "Hello"

    def test_segment_float_fields(self) -> None:
        seg = Segment(start=0.0, end=100.0, text="Long segment")
        assert isinstance(seg.start, float)
        assert isinstance(seg.end, float)

    def test_segment_empty_text(self) -> None:
        # Empty text is allowed at the model level
        seg = Segment(start=0.0, end=1.0, text="")
        assert seg.text == ""


class TestTranscription:
    def test_create_transcription(self, sample_transcription: Transcription) -> None:
        t = sample_transcription
        assert t.source_file == "/videos/test_video.mp4"
        assert t.duration == 12.0
        assert t.language == "en"
        assert t.model_used == "small"
        assert len(t.segments) == 3

    def test_transcription_default_created_at(self) -> None:
        t = Transcription(
            source_file="video.mp4",
            duration=10.0,
            language="en",
            model_used="tiny",
            segments=[],
        )
        assert t.created_at is not None
        assert isinstance(t.created_at, datetime)

    def test_transcription_created_at_is_timezone_aware(self) -> None:
        t = Transcription(
            source_file="video.mp4",
            duration=10.0,
            language="en",
            model_used="tiny",
            segments=[],
        )
        assert t.created_at.tzinfo is not None

    def test_transcription_with_explicit_created_at(self) -> None:
        dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        t = Transcription(
            source_file="video.mp4",
            duration=5.0,
            language="fr",
            model_used="base",
            segments=[],
            created_at=dt,
        )
        assert t.created_at == dt

    def test_transcription_empty_segments(self) -> None:
        t = Transcription(
            source_file="silence.mp4",
            duration=30.0,
            language="en",
            model_used="small",
            segments=[],
        )
        assert t.segments == []

    def test_transcription_segments_are_segments(
        self, sample_transcription: Transcription
    ) -> None:
        for seg in sample_transcription.segments:
            assert isinstance(seg, Segment)


class TestCutSegment:
    def test_create_cut_segment(self) -> None:
        cs = CutSegment(
            source_file="/videos/clip.mp4",
            start=10.0,
            end=25.5,
            description="Interesting moment",
            order=0,
        )
        assert cs.source_file == "/videos/clip.mp4"
        assert cs.start == 10.0
        assert cs.end == 25.5
        assert cs.description == "Interesting moment"
        assert cs.order == 0

    def test_cut_segment_order_is_int(self) -> None:
        cs = CutSegment(
            source_file="video.mp4",
            start=0.0,
            end=5.0,
            description="Intro",
            order=3,
        )
        assert isinstance(cs.order, int)


class TestCutList:
    def test_create_cut_list(self, sample_cut_list: CutList) -> None:
        cl = sample_cut_list
        assert cl.id == "550e8400-e29b-41d4-a716-446655440000"
        assert cl.prompt == "find key moments"
        assert cl.theme == "Highlights"
        assert len(cl.segments) == 2
        assert cl.total_duration == 8.3

    def test_cut_list_segments_are_cut_segments(
        self, sample_cut_list: CutList
    ) -> None:
        for seg in sample_cut_list.segments:
            assert isinstance(seg, CutSegment)

    def test_cut_list_total_duration_type(self, sample_cut_list: CutList) -> None:
        assert isinstance(sample_cut_list.total_duration, float)

    def test_cut_list_empty_segments(self) -> None:
        cl = CutList(
            id="test-uuid",
            prompt="empty prompt",
            theme="Empty",
            created_at=datetime.now(timezone.utc),
            segments=[],
            total_duration=0.0,
        )
        assert cl.segments == []
        assert cl.total_duration == 0.0

    def test_cut_list_segment_ordering(
        self, sample_cut_segments: list[CutSegment]
    ) -> None:
        # Segments should preserve their order values
        assert sample_cut_segments[0].order == 0
        assert sample_cut_segments[1].order == 1

    def test_models_importable_from_package(self) -> None:
        """Ensure all models are importable from snipsnap.models."""
        from snipsnap.models import (  # noqa: F401
            CutList,
            CutSegment,
            Segment,
            Transcription,
        )
