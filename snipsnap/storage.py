"""JSON file storage layer for SnipSnap.

Provides save/load functions for Transcription and CutList objects.
All data is persisted as JSON files under the configured data directory.

Directory layout:
    <data_dir>/
    ├── transcriptions/
    │   └── <stem>.json        (one file per transcribed video)
    └── cut_lists/
        └── <uuid>.json        (one file per cut list)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from snipsnap.config import get_config
from snipsnap.models import CutList, CutSegment, Segment, Transcription

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> Path:
    """Create *path* (and any parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_data_dir(data_dir: Optional[Path]) -> Path:
    """Return *data_dir* if provided, otherwise the configured default."""
    if data_dir is not None:
        return data_dir
    return get_config().data_dir


def _transcription_path(source_file: str, data_dir: Path) -> Path:
    """Return the expected JSON path for a transcription of *source_file*."""
    stem = Path(source_file).stem
    return data_dir / "transcriptions" / f"{stem}.json"


def _transcription_from_dict(data: dict) -> Transcription:
    return Transcription(
        source_file=data["source_file"],
        duration=float(data["duration"]),
        language=data["language"],
        model_used=data["model_used"],
        created_at=datetime.fromisoformat(data["created_at"]),
        segments=[
            Segment(start=float(s["start"]), end=float(s["end"]), text=s["text"])
            for s in data.get("segments", [])
        ],
    )


def _transcription_to_dict(t: Transcription) -> dict:
    return {
        "source_file": t.source_file,
        "duration": t.duration,
        "language": t.language,
        "model_used": t.model_used,
        "created_at": t.created_at.isoformat(),
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text} for s in t.segments
        ],
    }


def _cut_list_from_dict(data: dict) -> CutList:
    return CutList(
        id=data["id"],
        prompt=data["prompt"],
        theme=data["theme"],
        created_at=datetime.fromisoformat(data["created_at"]),
        total_duration=float(data["total_duration"]),
        segments=[
            CutSegment(
                source_file=seg["source_file"],
                start=float(seg["start"]),
                end=float(seg["end"]),
                description=seg["description"],
                order=int(seg["order"]),
            )
            for seg in data.get("segments", [])
        ],
    )


def _cut_list_to_dict(cl: CutList) -> dict:
    return {
        "id": cl.id,
        "prompt": cl.prompt,
        "theme": cl.theme,
        "created_at": cl.created_at.isoformat(),
        "total_duration": cl.total_duration,
        "segments": [
            {
                "source_file": s.source_file,
                "start": s.start,
                "end": s.end,
                "description": s.description,
                "order": s.order,
            }
            for s in cl.segments
        ],
    }


# ---------------------------------------------------------------------------
# Public API — Transcriptions
# ---------------------------------------------------------------------------


def save_transcription(
    transcription: Transcription,
    data_dir: Optional[Path] = None,
) -> Path:
    """Persist *transcription* as JSON and return the file path."""
    resolved = _resolve_data_dir(data_dir)
    _ensure_dir(resolved / "transcriptions")
    path = _transcription_path(transcription.source_file, resolved)
    path.write_text(json.dumps(_transcription_to_dict(transcription), indent=2))
    return path


def load_transcription(
    source_file: str,
    data_dir: Optional[Path] = None,
) -> Optional[Transcription]:
    """Load the transcription for *source_file*, or ``None`` if not found."""
    resolved = _resolve_data_dir(data_dir)
    path = _transcription_path(source_file, resolved)
    if not path.exists():
        return None
    return _transcription_from_dict(json.loads(path.read_text()))


def transcription_exists(
    source_file: str,
    data_dir: Optional[Path] = None,
) -> bool:
    """Return True if a transcription JSON exists for *source_file*."""
    resolved = _resolve_data_dir(data_dir)
    return _transcription_path(source_file, resolved).exists()


def load_all_transcriptions(
    data_dir: Optional[Path] = None,
) -> List[Transcription]:
    """Load all transcriptions from the data directory."""
    resolved = _resolve_data_dir(data_dir)
    trans_dir = resolved / "transcriptions"
    if not trans_dir.exists():
        return []
    transcriptions = []
    for path in sorted(trans_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            transcriptions.append(_transcription_from_dict(data))
        except (json.JSONDecodeError, KeyError):
            # Skip malformed files rather than crashing
            continue
    return transcriptions


# ---------------------------------------------------------------------------
# Public API — Cut Lists
# ---------------------------------------------------------------------------


def save_cut_list(
    cut_list: CutList,
    data_dir: Optional[Path] = None,
) -> Path:
    """Persist *cut_list* as JSON and return the file path."""
    resolved = _resolve_data_dir(data_dir)
    _ensure_dir(resolved / "cut_lists")
    path = resolved / "cut_lists" / f"{cut_list.id}.json"
    path.write_text(json.dumps(_cut_list_to_dict(cut_list), indent=2))
    return path


def load_cut_list(
    cut_list_id: str,
    data_dir: Optional[Path] = None,
) -> Optional[CutList]:
    """Load a cut list by its UUID, or ``None`` if not found."""
    resolved = _resolve_data_dir(data_dir)
    path = resolved / "cut_lists" / f"{cut_list_id}.json"
    if not path.exists():
        return None
    return _cut_list_from_dict(json.loads(path.read_text()))


def load_all_cut_lists(
    data_dir: Optional[Path] = None,
) -> List[CutList]:
    """Load all cut lists from the data directory."""
    resolved = _resolve_data_dir(data_dir)
    cut_dir = resolved / "cut_lists"
    if not cut_dir.exists():
        return []
    cut_lists = []
    for path in sorted(cut_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            cut_lists.append(_cut_list_from_dict(data))
        except (json.JSONDecodeError, KeyError):
            continue
    return cut_lists
