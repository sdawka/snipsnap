"""CMX 3600 EDL (Edit Decision List) generator.

Converts a CutList into a standard CMX 3600 EDL file, compatible with
DaVinci Resolve, Premiere Pro, Final Cut Pro, and other NLEs.
"""

from __future__ import annotations

from pathlib import Path

from snipsnap.models import CutList


def seconds_to_smpte(seconds: float, fps: int = 24) -> str:
    """Convert seconds to SMPTE timecode string HH:MM:SS:FF.

    Args:
        seconds: Time in seconds (non-negative).
        fps: Frame rate (frames per second). Default 24.

    Returns:
        SMPTE timecode string in HH:MM:SS:FF format.
    """
    total_frames = round(seconds * fps)
    frames = total_frames % fps
    total_seconds = total_frames // fps
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{frames:02d}"


def _reel_name(source_file: str) -> str:
    """Derive an 8-character EDL reel name from a source file path.

    CMX 3600 limits reel names to 8 characters. The full filename is
    included in a ``* FROM CLIP NAME:`` comment line.
    """
    stem = Path(source_file).stem
    # Replace spaces/special chars with underscores
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return safe[:8]


def generate_edl(cut_list: CutList, frame_rate: int = 24, title: str = "") -> str:
    """Generate a CMX 3600 EDL string from a CutList.

    Args:
        cut_list: The CutList containing ordered CutSegments.
        frame_rate: Frame rate for SMPTE timecode conversion. Default 24.
        title: Optional title for the EDL. Defaults to the cut list theme.

    Returns:
        EDL file content as a string (UTF-8 text).
    """
    edl_title = title if title else cut_list.theme or "SnipSnap Cut"

    lines: list[str] = []
    lines.append(f"TITLE: {edl_title}")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")

    # Record timecode starts at 01:00:00:00 by convention
    rec_frames = 1 * 3600 * frame_rate  # 01:00:00:00 in frames

    # Sort segments by order to ensure correct sequencing
    segments = sorted(cut_list.segments, key=lambda s: s.order)

    for i, seg in enumerate(segments, start=1):
        event_num = f"{i:03d}"
        reel = _reel_name(seg.source_file)

        src_in = seconds_to_smpte(seg.start, frame_rate)
        src_out = seconds_to_smpte(seg.end, frame_rate)

        rec_in = _frames_to_smpte(rec_frames, frame_rate)
        seg_frames = round(seg.end * frame_rate) - round(seg.start * frame_rate)
        rec_frames_out = rec_frames + seg_frames
        rec_out = _frames_to_smpte(rec_frames_out, frame_rate)

        # CMX 3600 event line format:
        # NNN  REEL     TRACK TRANS    SRC_IN     SRC_OUT    REC_IN     REC_OUT
        lines.append(
            f"{event_num}  {reel:<8} V     C        "
            f"{src_in} {src_out} {rec_in} {rec_out}"
        )

        # Comment: source filename
        source_name = Path(seg.source_file).name
        lines.append(f"* FROM CLIP NAME: {source_name}")

        # Comment: segment description
        if seg.description:
            lines.append(f"* {seg.description}")

        rec_frames = rec_frames_out

    return "\n".join(lines) + "\n"


def _frames_to_smpte(total_frames: int, fps: int) -> str:
    """Convert an absolute frame count to SMPTE timecode."""
    frames = total_frames % fps
    total_seconds = total_frames // fps
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{frames:02d}"
