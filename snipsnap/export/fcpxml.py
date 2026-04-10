"""FCPXML 1.8 generator.

Generates Final Cut Pro XML (FCPXML) version 1.8 from a CutList.
FCPXML is supported by DaVinci Resolve (1.3–1.10), Final Cut Pro, and
can be imported into other professional NLEs.

All times are expressed as rational fractions of seconds (e.g., "120/24s"
for 5 seconds at 24fps).
"""

from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from urllib.parse import quote

from snipsnap.models import CutList


def seconds_to_rational(seconds: float, fps: int) -> str:
    """Convert seconds to FCPXML rational time string.

    Args:
        seconds: Time in seconds (non-negative).
        fps: Frame rate.

    Returns:
        Rational time string like "120/24s" or "0/1s" for zero.
    """
    if seconds == 0.0:
        return "0/1s"
    frames = round(seconds * fps)
    return f"{frames}/{fps}s"


def _frame_duration(fps: int) -> str:
    """Return the frameDuration attribute string for the given frame rate."""
    return f"1/{fps}s"


def _format_name(fps: int) -> str:
    """Return a descriptive format name string for the given frame rate."""
    return f"FFVideoFormat1080p{fps}"


def _asset_src(source_file: str) -> str:
    """Convert a file path to a file:/// URI with proper encoding."""
    path = Path(source_file)
    # Encode each path component (preserving drive/root on all platforms)
    encoded_parts = [quote(part, safe="") for part in path.parts[1:]]
    return "file:///" + "/".join(encoded_parts)


def generate_fcpxml(cut_list: CutList, frame_rate: int = 24) -> str:
    """Generate FCPXML 1.8 content from a CutList.

    Args:
        cut_list: The CutList containing ordered CutSegments.
        frame_rate: Frame rate for rational time conversion. Default 24.

    Returns:
        FCPXML file content as a UTF-8 string.
    """
    # Build root element
    root = ET.Element("fcpxml", version="1.8")

    # --- resources ---
    resources = ET.SubElement(root, "resources")

    # Format element
    fmt_id = "r1"
    ET.SubElement(
        resources,
        "format",
        id=fmt_id,
        name=_format_name(frame_rate),
        frameDuration=_frame_duration(frame_rate),
        width="1920",
        height="1080",
    )

    # Collect unique source files (in appearance order for stable IDs)
    seen: dict[str, str] = {}  # source_file -> asset_id
    asset_counter = 2  # Start at r2 (r1 is the format)

    segments = sorted(cut_list.segments, key=lambda s: s.order)

    for seg in segments:
        if seg.source_file not in seen:
            asset_id = f"r{asset_counter}"
            seen[seg.source_file] = asset_id
            asset_counter += 1

            name = Path(seg.source_file).stem
            # Generate a stable UID from the source file path
            asset_uid = (
                str(uuid.uuid5(uuid.NAMESPACE_URL, seg.source_file))
                .upper()
                .replace("-", "")[:16]
            )

            ET.SubElement(
                resources,
                "asset",
                id=asset_id,
                name=name,
                uid=asset_uid,
                src=_asset_src(seg.source_file),
                start="0/1s",
                duration="0/1s",  # placeholder; actual duration unknown
                hasVideo="1",
                format=fmt_id,
                hasAudio="1",
                audioSources="1",
                audioChannels="2",
                audioRate="48000",
            )

    # --- library / event / project / sequence / spine ---
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="SnipSnap")
    project_elem = ET.SubElement(event, "project", name=cut_list.theme or "SnipSnap Cut")

    # Compute total timeline duration
    total_frames = sum(
        round(seg.end * frame_rate) - round(seg.start * frame_rate)
        for seg in segments
    )
    total_duration_rational = f"{total_frames}/{frame_rate}s" if total_frames > 0 else "0/1s"

    sequence = ET.SubElement(
        project_elem,
        "sequence",
        format=fmt_id,
        duration=total_duration_rational,
        tcStart="0/1s",
        tcFormat="NDF",
        audioLayout="stereo",
        audioRate="48k",
    )
    spine = ET.SubElement(sequence, "spine")

    # Add asset-clips in cut list order
    current_offset_frames = 0

    for seg in segments:
        asset_id = seen[seg.source_file]
        name = Path(seg.source_file).stem

        seg_frames = round(seg.end * frame_rate) - round(seg.start * frame_rate)
        start_rational = seconds_to_rational(seg.start, frame_rate)
        duration_rational = f"{seg_frames}/{frame_rate}s" if seg_frames > 0 else "0/1s"
        offset_rational = (
            f"{current_offset_frames}/{frame_rate}s"
            if current_offset_frames > 0
            else "0/1s"
        )

        clip_elem = ET.SubElement(
            spine,
            "asset-clip",
            ref=asset_id,
            name=name,
            offset=offset_rational,
            start=start_rational,
            duration=duration_rational,
            format=fmt_id,
            audioRole="dialogue",
        )

        # Add note child element for description
        if seg.description:
            note = ET.SubElement(clip_elem, "note")
            note.text = seg.description

        current_offset_frames += seg_frames

    # Serialize with XML declaration and DOCTYPE
    ET.indent(root, space="    ")
    xml_body = ET.tostring(root, encoding="unicode", xml_declaration=False)

    buf = StringIO()
    buf.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    buf.write("<!DOCTYPE fcpxml>\n")
    buf.write(xml_body)
    buf.write("\n")
    return buf.getvalue()
