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


def _get_fcm(frame_rate: float) -> str:
    """Return the FCM (Frame Code Mode) string for the given frame rate.

    Drop frame timecode is used for 29.97fps (NTSC broadcast standard).
    All other standard frame rates (24, 25, 30fps) use non-drop frame.

    Args:
        frame_rate: Frame rate in frames per second.

    Returns:
        ``"DROP FRAME"`` for 29.97fps, ``"NON-DROP FRAME"`` otherwise.
    """
    if abs(frame_rate - 29.97) < 0.01:
        return "DROP FRAME"
    return "NON-DROP FRAME"


def _normalize_fps(frame_rate: float) -> int:
    """Normalize a frame rate float to the integer fps used for timecode math.

    29.97fps uses 30 as the integer frame count per second (drop frame
    convention: frame numbers 0–29 are used, with periodic frame-number
    skipping to maintain wall-clock accuracy).

    Args:
        frame_rate: Frame rate in frames per second.

    Returns:
        Integer fps for use in SMPTE timecode calculations.
    """
    if abs(frame_rate - 29.97) < 0.01:
        return 30
    return round(frame_rate)


def generate_edl(cut_list: CutList, frame_rate: float = 24, title: str = "") -> str:
    """Generate a CMX 3600 EDL string from a CutList.

    Args:
        cut_list: The CutList containing ordered CutSegments.
        frame_rate: Frame rate for SMPTE timecode conversion. Default 24.
            Use 29.97 for NTSC drop-frame (emits ``FCM: DROP FRAME``).
        title: Optional title for the EDL. Defaults to the cut list theme.

    Returns:
        EDL file content as a string (UTF-8 text).
    """
    edl_title = title if title else cut_list.theme or "SnipSnap Cut"
    fcm = _get_fcm(frame_rate)
    fps = _normalize_fps(frame_rate)

    lines: list[str] = []
    lines.append(f"TITLE: {edl_title}")
    lines.append(f"FCM: {fcm}")
    lines.append("")

    # Record timecode starts at 01:00:00:00 by convention
    rec_frames = 1 * 3600 * fps  # 01:00:00:00 in frames

    # Sort segments by order to ensure correct sequencing
    segments = sorted(cut_list.segments, key=lambda s: s.order)

    for i, seg in enumerate(segments, start=1):
        event_num = f"{i:03d}"
        reel = _reel_name(seg.source_file)

        src_in = seconds_to_smpte(seg.start, fps)
        src_out = seconds_to_smpte(seg.end, fps)

        rec_in = _frames_to_smpte(rec_frames, fps)
        seg_frames = round(seg.end * fps) - round(seg.start * fps)
        rec_frames_out = rec_frames + seg_frames
        rec_out = _frames_to_smpte(rec_frames_out, fps)

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
