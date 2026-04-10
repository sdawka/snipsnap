"""Data models for SnipSnap pipeline.

All timestamps are in seconds (float). Conversion to format-specific time
representations (SMPTE timecode, rational time, etc.) happens only at
the export boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List


@dataclass
class Segment:
    """A single transcribed utterance within a video."""

    start: float
    end: float
    text: str


@dataclass
class Transcription:
    """Full transcription of a single video file."""

    source_file: str
    duration: float
    language: str
    model_used: str
    segments: List[Segment]
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class CutSegment:
    """A single segment selected for inclusion in the final cut."""

    source_file: str
    start: float
    end: float
    description: str
    order: int


@dataclass
class CutList:
    """A curated selection of video segments produced by the curation engine."""

    id: str
    prompt: str
    theme: str
    created_at: datetime
    segments: List[CutSegment]
    total_duration: float
