"""Shared pytest fixtures for SnipSnap tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from snipsnap.models import CutList, CutSegment, Segment, Transcription


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a temporary data directory for storage tests."""
    data_dir = tmp_path / "snipsnap_data"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def sample_segments() -> list[Segment]:
    return [
        Segment(start=0.0, end=3.5, text="Hello world"),
        Segment(start=3.5, end=7.2, text="This is a test"),
        Segment(start=7.2, end=12.0, text="Goodbye"),
    ]


@pytest.fixture
def sample_transcription(sample_segments: list[Segment]) -> Transcription:
    return Transcription(
        source_file="/videos/test_video.mp4",
        duration=12.0,
        language="en",
        model_used="small",
        segments=sample_segments,
        created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_cut_segments() -> list[CutSegment]:
    return [
        CutSegment(
            source_file="/videos/test_video.mp4",
            start=0.0,
            end=3.5,
            description="Opening greeting",
            order=0,
        ),
        CutSegment(
            source_file="/videos/test_video.mp4",
            start=7.2,
            end=12.0,
            description="Closing statement",
            order=1,
        ),
    ]


@pytest.fixture
def sample_cut_list(sample_cut_segments: list[CutSegment]) -> CutList:
    return CutList(
        id="550e8400-e29b-41d4-a716-446655440000",
        prompt="find key moments",
        theme="Highlights",
        created_at=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
        segments=sample_cut_segments,
        total_duration=8.3,
    )
