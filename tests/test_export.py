"""Comprehensive tests for export format generators.

Tests cover:
- EDL (CMX 3600) generator
- FCPXML 1.8 generator
- DaVinci Resolve script generator

Including structure validation, timecode accuracy, edge cases, and
multi-source cut lists.
"""

from __future__ import annotations

import py_compile
import re
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pytest

from snipsnap.export.davinci import generate_davinci_script
from snipsnap.export.edl import generate_edl, seconds_to_smpte
from snipsnap.export.fcpxml import _asset_src, generate_fcpxml, seconds_to_rational
from snipsnap.models import CutList, CutSegment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_segment_cut_list() -> CutList:
    """A CutList with exactly one segment."""
    return CutList(
        id="test-id-001",
        prompt="find opening",
        theme="Opening",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        segments=[
            CutSegment(
                source_file="/videos/interview.mp4",
                start=5.0,
                end=10.0,
                description="Opening greeting",
                order=0,
            )
        ],
        total_duration=5.0,
    )


@pytest.fixture
def multi_segment_cut_list() -> CutList:
    """A CutList with multiple segments from the same source file."""
    return CutList(
        id="test-id-002",
        prompt="key moments",
        theme="Highlights",
        created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        segments=[
            CutSegment(
                source_file="/videos/interview.mp4",
                start=0.0,
                end=3.5,
                description="Opening greeting",
                order=0,
            ),
            CutSegment(
                source_file="/videos/interview.mp4",
                start=7.2,
                end=12.0,
                description="Main point",
                order=1,
            ),
            CutSegment(
                source_file="/videos/interview.mp4",
                start=20.0,
                end=25.0,
                description="Closing statement",
                order=2,
            ),
        ],
        total_duration=13.3,
    )


@pytest.fixture
def multi_source_cut_list() -> CutList:
    """A CutList with segments from multiple source files."""
    return CutList(
        id="test-id-003",
        prompt="best clips",
        theme="Multi-Source Cut",
        created_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        segments=[
            CutSegment(
                source_file="/videos/interview_wide.mp4",
                start=5.0,
                end=10.0,
                description="Wide shot intro",
                order=0,
            ),
            CutSegment(
                source_file="/videos/broll_sunset.mp4",
                start=2.5,
                end=8.0,
                description="Sunset broll",
                order=1,
            ),
            CutSegment(
                source_file="/videos/interview_closeup.mp4",
                start=0.0,
                end=3.75,
                description="Closeup reaction",
                order=2,
            ),
            CutSegment(
                source_file="/videos/interview_wide.mp4",
                start=15.0,
                end=22.0,
                description="Wide shot conclusion",
                order=3,
            ),
        ],
        total_duration=20.25,
    )


@pytest.fixture
def empty_cut_list() -> CutList:
    """A CutList with zero segments."""
    return CutList(
        id="test-id-empty",
        prompt="nothing",
        theme="Empty Cut",
        created_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
        segments=[],
        total_duration=0.0,
    )


@pytest.fixture
def relative_path_cut_list() -> CutList:
    """A CutList with a relative source file path (multi-component)."""
    return CutList(
        id="test-id-rel",
        prompt="relative path test",
        theme="Relative Path Test",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        segments=[
            CutSegment(
                source_file="projects/video/interview.mp4",
                start=0.0,
                end=5.0,
                description="Relative path segment",
                order=0,
            )
        ],
        total_duration=5.0,
    )


# ===========================================================================
# FCPXML _asset_src Path Fidelity Regression Tests
# ===========================================================================


class TestAssetSrcPathFidelity:
    """Regression tests ensuring _asset_src() preserves full path components.

    Bug: the original implementation used path.parts[1:] unconditionally,
    which drops the first component of relative paths.
    """

    def test_absolute_path_produces_correct_file_uri(self) -> None:
        """Absolute path must map to a correct file:/// URI."""
        result = _asset_src("/videos/foo/bar.mp4")
        assert result == "file:///videos/foo/bar.mp4"

    def test_relative_path_preserves_all_components(self) -> None:
        """Relative paths must NOT drop the first path component."""
        result = _asset_src("some/path/to/video.mp4")
        assert result == "file:///some/path/to/video.mp4"

    def test_relative_path_single_directory_component_preserved(self) -> None:
        """Single-directory relative paths must not become bare filenames."""
        result = _asset_src("myfolder/video.mp4")
        assert result == "file:///myfolder/video.mp4"

    def test_relative_path_bare_filename_preserved(self) -> None:
        """A bare filename (no directory) must remain accessible at file:///."""
        result = _asset_src("video.mp4")
        assert result == "file:///video.mp4"

    def test_absolute_path_with_spaces_encoded(self) -> None:
        """Spaces in absolute path components must be percent-encoded."""
        result = _asset_src("/my videos/my file.mp4")
        assert result.startswith("file:///")
        assert "my%20videos" in result
        assert "my%20file.mp4" in result

    def test_relative_path_with_spaces_all_components_preserved(self) -> None:
        """Relative paths with spaces: all components must be preserved and encoded."""
        result = _asset_src("my videos/my file.mp4")
        assert result.startswith("file:///")
        assert "my%20videos" in result
        assert "my%20file.mp4" in result

    def test_fcpxml_asset_src_for_relative_path_in_full_document(
        self, relative_path_cut_list: CutList
    ) -> None:
        """Integration: FCPXML generated with a relative source_file must preserve
        all path components in the asset src attribute."""
        content = generate_fcpxml(relative_path_cut_list)
        # Strip DOCTYPE for ET parser
        lines = [ln for ln in content.splitlines() if not ln.startswith("<!DOCTYPE")]
        root = ET.fromstring("\n".join(lines))
        asset = root.find("resources/asset")
        assert asset is not None
        src = asset.attrib.get("src", "")
        assert src.startswith("file:///"), f"src should start with file:///: {src}"
        # All three path components must be present
        assert "projects" in src, f"First path component 'projects' missing from src: {src}"
        assert "video" in src, f"Second path component 'video' missing from src: {src}"
        assert "interview.mp4" in src, f"Filename missing from src: {src}"


# ===========================================================================
# EDL Tests
# ===========================================================================


class TestSecondsToSmpte:
    """Tests for the SMPTE timecode conversion utility."""

    def test_zero_seconds(self) -> None:
        assert seconds_to_smpte(0.0, fps=24) == "00:00:00:00"

    def test_one_second_24fps(self) -> None:
        assert seconds_to_smpte(1.0, fps=24) == "00:00:01:00"

    def test_5_seconds_24fps(self) -> None:
        assert seconds_to_smpte(5.0, fps=24) == "00:00:05:00"

    def test_one_minute_24fps(self) -> None:
        assert seconds_to_smpte(60.0, fps=24) == "00:01:00:00"

    def test_one_hour_24fps(self) -> None:
        assert seconds_to_smpte(3600.0, fps=24) == "01:00:00:00"

    def test_one_frame_24fps(self) -> None:
        # 1/24 second = 1 frame
        assert seconds_to_smpte(1 / 24, fps=24) == "00:00:00:01"

    def test_half_second_24fps(self) -> None:
        # 0.5s = 12 frames at 24fps
        assert seconds_to_smpte(0.5, fps=24) == "00:00:00:12"

    def test_25fps(self) -> None:
        # 0.5s = 12.5 frames => rounds to 13 frames at 25fps
        result = seconds_to_smpte(0.5, fps=25)
        hh, mm, ss, ff = (int(x) for x in result.split(":"))
        assert hh == 0 and mm == 0 and ss == 0
        assert 0 <= ff < 25

    def test_30fps(self) -> None:
        # 0.5s = 15 frames at 30fps
        assert seconds_to_smpte(0.5, fps=30) == "00:00:00:15"

    def test_frames_within_range_24fps(self) -> None:
        """Frame component must be 0–23 for 24fps."""
        for s in [0.0, 0.5, 1.0, 3.14, 60.0, 3600.0]:
            result = seconds_to_smpte(s, fps=24)
            ff = int(result.split(":")[-1])
            assert 0 <= ff < 24, f"Frame {ff} out of range for {s}s at 24fps"

    def test_frames_within_range_25fps(self) -> None:
        """Frame component must be 0–24 for 25fps."""
        for s in [0.0, 0.04, 0.5, 1.0, 59.99]:
            result = seconds_to_smpte(s, fps=25)
            ff = int(result.split(":")[-1])
            assert 0 <= ff < 25, f"Frame {ff} out of range for {s}s at 25fps"

    def test_frames_within_range_30fps(self) -> None:
        """Frame component must be 0–29 for 30fps."""
        for s in [0.0, 0.1, 1.0, 30.0]:
            result = seconds_to_smpte(s, fps=30)
            ff = int(result.split(":")[-1])
            assert 0 <= ff < 30, f"Frame {ff} out of range for {s}s at 30fps"

    def test_smpte_format_regex(self) -> None:
        """All outputs match HH:MM:SS:FF pattern."""
        pattern = re.compile(r"^\d{2}:\d{2}:\d{2}:\d{2}$")
        for seconds in [0.0, 1.0, 59.99, 3600.0, 7265.5]:
            result = seconds_to_smpte(seconds, fps=24)
            assert pattern.match(result), f"Bad format for {seconds}s: {result}"

    def test_complex_timecode_24fps(self) -> None:
        """1 hour 2 minutes 3 seconds 4 frames = 3723 + 4/24 seconds."""
        total_frames = 1 * 3600 * 24 + 2 * 60 * 24 + 3 * 24 + 4
        seconds = total_frames / 24.0
        assert seconds_to_smpte(seconds, fps=24) == "01:02:03:04"


class TestGenerateEdl:
    """Tests for the CMX 3600 EDL generator."""

    def test_edl_starts_with_title(self, multi_segment_cut_list: CutList) -> None:
        edl = generate_edl(multi_segment_cut_list)
        assert edl.startswith("TITLE: ")

    def test_edl_title_uses_cut_list_theme(self, multi_segment_cut_list: CutList) -> None:
        edl = generate_edl(multi_segment_cut_list)
        assert "TITLE: Highlights" in edl

    def test_edl_custom_title(self, multi_segment_cut_list: CutList) -> None:
        edl = generate_edl(multi_segment_cut_list, title="My Custom Title")
        assert "TITLE: My Custom Title" in edl

    def test_edl_contains_fcm_line(self, multi_segment_cut_list: CutList) -> None:
        edl = generate_edl(multi_segment_cut_list)
        assert "FCM: NON-DROP FRAME" in edl

    def test_edl_fcm_before_events(self, multi_segment_cut_list: CutList) -> None:
        edl = generate_edl(multi_segment_cut_list)
        lines = edl.splitlines()
        title_idx = next(i for i, ln in enumerate(lines) if ln.startswith("TITLE:"))
        fcm_idx = next(i for i, ln in enumerate(lines) if ln.startswith("FCM:"))
        assert fcm_idx > title_idx

    def test_edl_sequential_event_numbers(self, multi_segment_cut_list: CutList) -> None:
        edl = generate_edl(multi_segment_cut_list)
        lines = edl.splitlines()
        events = [ln for ln in lines if re.match(r"^\d{3}\s", ln)]
        assert len(events) == 3
        for i, event_line in enumerate(events, start=1):
            assert event_line.startswith(f"{i:03d}")

    def test_edl_single_segment_event_numbered_001(
        self, single_segment_cut_list: CutList
    ) -> None:
        edl = generate_edl(single_segment_cut_list)
        assert "001  " in edl
        lines = edl.splitlines()
        events = [ln for ln in lines if re.match(r"^\d{3}\s", ln)]
        assert len(events) == 1

    def test_edl_timecodes_match_smpte_format(self, multi_segment_cut_list: CutList) -> None:
        edl = generate_edl(multi_segment_cut_list)
        tc_pattern = re.compile(r"\d{2}:\d{2}:\d{2}:\d{2}")
        lines = edl.splitlines()
        event_lines = [ln for ln in lines if re.match(r"^\d{3}\s", ln)]
        for line in event_lines:
            tcs = tc_pattern.findall(line)
            assert len(tcs) == 4, f"Expected 4 timecodes in: {line}"

    def test_edl_record_timecodes_contiguous(
        self, multi_segment_cut_list: CutList
    ) -> None:
        """Record-out of event N must equal record-in of event N+1."""
        edl = generate_edl(multi_segment_cut_list)
        lines = edl.splitlines()
        event_lines = [ln for ln in lines if re.match(r"^\d{3}\s", ln)]
        tc_pattern = re.compile(r"\d{2}:\d{2}:\d{2}:\d{2}")
        rec_ins = []
        rec_outs = []
        for line in event_lines:
            tcs = tc_pattern.findall(line)
            # Format: NNN  REEL     V     C        SRC_IN SRC_OUT REC_IN REC_OUT
            rec_ins.append(tcs[2])
            rec_outs.append(tcs[3])
        for i in range(len(rec_outs) - 1):
            assert rec_outs[i] == rec_ins[i + 1], (
                f"Gap between events {i+1} and {i+2}: "
                f"rec_out={rec_outs[i]}, next rec_in={rec_ins[i+1]}"
            )

    def test_edl_record_starts_at_01_00_00_00(
        self, single_segment_cut_list: CutList
    ) -> None:
        """Record in of first event should be 01:00:00:00."""
        edl = generate_edl(single_segment_cut_list)
        lines = edl.splitlines()
        event_lines = [ln for ln in lines if re.match(r"^\d{3}\s", ln)]
        tc_pattern = re.compile(r"\d{2}:\d{2}:\d{2}:\d{2}")
        first_event = event_lines[0]
        tcs = tc_pattern.findall(first_event)
        # Record In is the 3rd timecode
        assert tcs[2] == "01:00:00:00"

    def test_edl_source_timecodes_match_segment_times(
        self, single_segment_cut_list: CutList
    ) -> None:
        """Source in/out should match segment start/end."""
        edl = generate_edl(single_segment_cut_list)
        lines = edl.splitlines()
        event_lines = [ln for ln in lines if re.match(r"^\d{3}\s", ln)]
        tc_pattern = re.compile(r"\d{2}:\d{2}:\d{2}:\d{2}")
        first_event = event_lines[0]
        tcs = tc_pattern.findall(first_event)
        seg = single_segment_cut_list.segments[0]
        expected_src_in = seconds_to_smpte(seg.start)
        expected_src_out = seconds_to_smpte(seg.end)
        assert tcs[0] == expected_src_in
        assert tcs[1] == expected_src_out

    def test_edl_includes_from_clip_name_comment(
        self, multi_segment_cut_list: CutList
    ) -> None:
        edl = generate_edl(multi_segment_cut_list)
        assert "* FROM CLIP NAME: interview.mp4" in edl

    def test_edl_includes_description_comment(
        self, multi_segment_cut_list: CutList
    ) -> None:
        edl = generate_edl(multi_segment_cut_list)
        assert "* Opening greeting" in edl
        assert "* Main point" in edl
        assert "* Closing statement" in edl

    def test_edl_multi_source_includes_all_clip_names(
        self, multi_source_cut_list: CutList
    ) -> None:
        edl = generate_edl(multi_source_cut_list)
        assert "interview_wide.mp4" in edl
        assert "broll_sunset.mp4" in edl
        assert "interview_closeup.mp4" in edl

    def test_edl_multi_source_event_count(
        self, multi_source_cut_list: CutList
    ) -> None:
        edl = generate_edl(multi_source_cut_list)
        lines = edl.splitlines()
        events = [ln for ln in lines if re.match(r"^\d{3}\s", ln)]
        assert len(events) == 4

    def test_edl_empty_cut_list_has_no_events(self, empty_cut_list: CutList) -> None:
        edl = generate_edl(empty_cut_list)
        lines = edl.splitlines()
        events = [ln for ln in lines if re.match(r"^\d{3}\s", ln)]
        assert len(events) == 0
        # Should still have header
        assert "TITLE:" in edl
        assert "FCM:" in edl

    def test_edl_25fps_frame_values_in_range(
        self, multi_segment_cut_list: CutList
    ) -> None:
        edl = generate_edl(multi_segment_cut_list, frame_rate=25)
        tc_pattern = re.compile(r"\d{2}:\d{2}:\d{2}:\d{2}")
        tcs = tc_pattern.findall(edl)
        for tc in tcs:
            ff = int(tc.split(":")[-1])
            assert 0 <= ff < 25, f"Frame {ff} out of range at 25fps in {tc}"

    def test_edl_30fps_frame_values_in_range(
        self, multi_segment_cut_list: CutList
    ) -> None:
        edl = generate_edl(multi_segment_cut_list, frame_rate=30)
        tc_pattern = re.compile(r"\d{2}:\d{2}:\d{2}:\d{2}")
        tcs = tc_pattern.findall(edl)
        for tc in tcs:
            ff = int(tc.split(":")[-1])
            assert 0 <= ff < 30, f"Frame {ff} out of range at 30fps in {tc}"

    def test_edl_preserves_segment_order(self, multi_source_cut_list: CutList) -> None:
        """Events should appear in cut list order."""
        edl = generate_edl(multi_source_cut_list)
        lines = edl.splitlines()
        comment_lines = [ln for ln in lines if ln.startswith("* FROM CLIP NAME:")]
        expected_names = [
            "interview_wide.mp4",
            "broll_sunset.mp4",
            "interview_closeup.mp4",
            "interview_wide.mp4",
        ]
        actual_names = [ln.split(": ", 1)[1] for ln in comment_lines]
        assert actual_names == expected_names

    def test_edl_returns_string(self, single_segment_cut_list: CutList) -> None:
        result = generate_edl(single_segment_cut_list)
        assert isinstance(result, str)

    def test_edl_ends_with_newline(self, single_segment_cut_list: CutList) -> None:
        result = generate_edl(single_segment_cut_list)
        assert result.endswith("\n")

    def test_edl_reel_name_max_8_chars(self, multi_source_cut_list: CutList) -> None:
        """Reel names embedded in event lines must be at most 8 characters."""
        edl = generate_edl(multi_source_cut_list)
        lines = edl.splitlines()
        event_lines = [ln for ln in lines if re.match(r"^\d{3}\s", ln)]
        for line in event_lines:
            # Format: NNN  REEL...
            parts = line.split()
            # reel name is the second token
            reel = parts[1]
            assert len(reel) <= 8, f"Reel name too long: '{reel}'"


# ===========================================================================
# EDL FCM Header Regression Tests
# ===========================================================================


class TestEdlFcmHeader:
    """Regression tests for FCM (Frame Code Mode) header correctness.

    Validates that the FCM line reflects the configured frame rate:
    - NON-DROP FRAME for 24, 25, 30 fps
    - DROP FRAME for 29.97 fps
    """

    def test_edl_fcm_non_drop_frame_24fps(
        self, single_segment_cut_list: CutList
    ) -> None:
        """24fps must emit FCM: NON-DROP FRAME."""
        edl = generate_edl(single_segment_cut_list, frame_rate=24)
        assert "FCM: NON-DROP FRAME" in edl

    def test_edl_fcm_non_drop_frame_25fps(
        self, single_segment_cut_list: CutList
    ) -> None:
        """25fps must emit FCM: NON-DROP FRAME."""
        edl = generate_edl(single_segment_cut_list, frame_rate=25)
        assert "FCM: NON-DROP FRAME" in edl

    def test_edl_fcm_non_drop_frame_30fps(
        self, single_segment_cut_list: CutList
    ) -> None:
        """30fps must emit FCM: NON-DROP FRAME."""
        edl = generate_edl(single_segment_cut_list, frame_rate=30)
        assert "FCM: NON-DROP FRAME" in edl

    def test_edl_fcm_drop_frame_29_97fps(
        self, single_segment_cut_list: CutList
    ) -> None:
        """29.97fps must emit FCM: DROP FRAME."""
        edl = generate_edl(single_segment_cut_list, frame_rate=29.97)
        assert "FCM: DROP FRAME" in edl

    def test_edl_fcm_not_drop_frame_for_24fps(
        self, single_segment_cut_list: CutList
    ) -> None:
        """24fps must not emit FCM: DROP FRAME."""
        edl = generate_edl(single_segment_cut_list, frame_rate=24)
        assert "FCM: DROP FRAME" not in edl

    def test_edl_fcm_not_drop_frame_for_25fps(
        self, single_segment_cut_list: CutList
    ) -> None:
        """25fps must not emit FCM: DROP FRAME."""
        edl = generate_edl(single_segment_cut_list, frame_rate=25)
        assert "FCM: DROP FRAME" not in edl

    def test_edl_fcm_not_drop_frame_for_30fps(
        self, single_segment_cut_list: CutList
    ) -> None:
        """30fps must not emit FCM: DROP FRAME."""
        edl = generate_edl(single_segment_cut_list, frame_rate=30)
        assert "FCM: DROP FRAME" not in edl

    def test_edl_fcm_not_non_drop_frame_for_29_97fps(
        self, single_segment_cut_list: CutList
    ) -> None:
        """29.97fps must not emit FCM: NON-DROP FRAME."""
        edl = generate_edl(single_segment_cut_list, frame_rate=29.97)
        assert "FCM: NON-DROP FRAME" not in edl

    def test_edl_fcm_line_at_second_position(
        self, single_segment_cut_list: CutList
    ) -> None:
        """FCM line must immediately follow the TITLE line."""
        for fps in [24, 25, 30, 29.97]:
            edl = generate_edl(single_segment_cut_list, frame_rate=fps)
            lines = edl.splitlines()
            assert lines[1].startswith("FCM:"), (
                f"Expected FCM on line 2 at {fps}fps, got: {lines[1]!r}"
            )

    def test_edl_29_97fps_timecode_frames_in_range(
        self, single_segment_cut_list: CutList
    ) -> None:
        """For 29.97fps, frame component must be in 0–29 range (30fps display)."""
        edl = generate_edl(single_segment_cut_list, frame_rate=29.97)
        tc_pattern = re.compile(r"\d{2}:\d{2}:\d{2}:\d{2}")
        for tc in tc_pattern.findall(edl):
            ff = int(tc.split(":")[-1])
            assert 0 <= ff < 30, f"Frame {ff} out of range for 29.97fps in {tc}"

    def test_edl_fcm_default_fps_is_non_drop_frame(
        self, single_segment_cut_list: CutList
    ) -> None:
        """Default frame rate (24fps) must produce NON-DROP FRAME."""
        edl = generate_edl(single_segment_cut_list)
        assert "FCM: NON-DROP FRAME" in edl


# ===========================================================================
# FCPXML Tests
# ===========================================================================


class TestSecondsToRational:
    """Tests for FCPXML rational time conversion."""

    def test_zero_seconds(self) -> None:
        assert seconds_to_rational(0.0, fps=24) == "0/1s"

    def test_one_second_24fps(self) -> None:
        assert seconds_to_rational(1.0, fps=24) == "24/24s"

    def test_5_seconds_24fps(self) -> None:
        assert seconds_to_rational(5.0, fps=24) == "120/24s"

    def test_one_second_25fps(self) -> None:
        assert seconds_to_rational(1.0, fps=25) == "25/25s"

    def test_one_second_30fps(self) -> None:
        assert seconds_to_rational(1.0, fps=30) == "30/30s"

    def test_half_second_24fps(self) -> None:
        # 0.5 * 24 = 12 frames
        assert seconds_to_rational(0.5, fps=24) == "12/24s"

    def test_rational_ends_with_s(self) -> None:
        for fps in [24, 25, 30]:
            result = seconds_to_rational(3.0, fps=fps)
            assert result.endswith("s"), f"Expected trailing 's' but got: {result}"


class TestGenerateFcpxml:
    """Tests for the FCPXML 1.8 generator."""

    def _parse_xml(self, content: str) -> ET.Element:
        """Helper: parse FCPXML content, skipping DOCTYPE."""
        # Remove DOCTYPE line which ET doesn't support
        lines = [ln for ln in content.splitlines() if not ln.startswith("<!DOCTYPE")]
        return ET.fromstring("\n".join(lines))

    def test_fcpxml_is_valid_xml(self, multi_segment_cut_list: CutList) -> None:
        content = generate_fcpxml(multi_segment_cut_list)
        # Should not raise
        root = self._parse_xml(content)
        assert root is not None

    def test_fcpxml_root_version_1_8(self, multi_segment_cut_list: CutList) -> None:
        content = generate_fcpxml(multi_segment_cut_list)
        root = self._parse_xml(content)
        assert root.tag == "fcpxml"
        assert root.attrib.get("version") == "1.8"

    def test_fcpxml_xml_declaration(self, multi_segment_cut_list: CutList) -> None:
        content = generate_fcpxml(multi_segment_cut_list)
        assert content.startswith('<?xml version="1.0" encoding="UTF-8"?>')

    def test_fcpxml_doctype_present(self, multi_segment_cut_list: CutList) -> None:
        content = generate_fcpxml(multi_segment_cut_list)
        assert "<!DOCTYPE fcpxml>" in content

    def test_fcpxml_has_resources_element(self, multi_segment_cut_list: CutList) -> None:
        content = generate_fcpxml(multi_segment_cut_list)
        root = self._parse_xml(content)
        resources = root.find("resources")
        assert resources is not None

    def test_fcpxml_has_format_element(self, multi_segment_cut_list: CutList) -> None:
        content = generate_fcpxml(multi_segment_cut_list)
        root = self._parse_xml(content)
        fmt = root.find("resources/format")
        assert fmt is not None
        assert fmt.attrib.get("id") is not None

    def test_fcpxml_has_asset_for_each_unique_source(
        self, multi_source_cut_list: CutList
    ) -> None:
        content = generate_fcpxml(multi_source_cut_list)
        root = self._parse_xml(content)
        assets = root.findall("resources/asset")
        # 3 unique sources
        assert len(assets) == 3

    def test_fcpxml_asset_refs_match_clip_refs(
        self, multi_source_cut_list: CutList
    ) -> None:
        """Every asset-clip ref must match an asset id in resources."""
        content = generate_fcpxml(multi_source_cut_list)
        root = self._parse_xml(content)
        asset_ids = {a.attrib["id"] for a in root.findall("resources/asset")}
        clip_refs = {
            c.attrib["ref"] for c in root.findall(".//spine/asset-clip")
        }
        assert clip_refs.issubset(asset_ids), (
            f"Dangling refs: {clip_refs - asset_ids}"
        )

    def test_fcpxml_library_event_project_sequence_spine(
        self, multi_segment_cut_list: CutList
    ) -> None:
        content = generate_fcpxml(multi_segment_cut_list)
        root = self._parse_xml(content)
        assert root.find("library") is not None
        assert root.find("library/event") is not None
        assert root.find("library/event/project") is not None
        assert root.find("library/event/project/sequence") is not None
        assert root.find("library/event/project/sequence/spine") is not None

    def test_fcpxml_spine_has_correct_asset_clip_count(
        self, multi_segment_cut_list: CutList
    ) -> None:
        content = generate_fcpxml(multi_segment_cut_list)
        root = self._parse_xml(content)
        spine = root.find(".//spine")
        assert spine is not None
        clips = spine.findall("asset-clip")
        assert len(clips) == 3

    def test_fcpxml_single_segment(self, single_segment_cut_list: CutList) -> None:
        content = generate_fcpxml(single_segment_cut_list)
        root = self._parse_xml(content)
        clips = root.findall(".//spine/asset-clip")
        assert len(clips) == 1

    def test_fcpxml_empty_cut_list_spine_is_empty(
        self, empty_cut_list: CutList
    ) -> None:
        content = generate_fcpxml(empty_cut_list)
        root = self._parse_xml(content)
        spine = root.find(".//spine")
        assert spine is not None
        assert len(list(spine)) == 0

    def test_fcpxml_asset_clip_has_required_attributes(
        self, single_segment_cut_list: CutList
    ) -> None:
        content = generate_fcpxml(single_segment_cut_list)
        root = self._parse_xml(content)
        clip = root.find(".//spine/asset-clip")
        assert clip is not None
        for attr in ["ref", "offset", "start", "duration", "name"]:
            assert attr in clip.attrib, f"Missing attribute: {attr}"

    def test_fcpxml_asset_clip_start_matches_segment_start(
        self, single_segment_cut_list: CutList
    ) -> None:
        content = generate_fcpxml(single_segment_cut_list, frame_rate=24)
        root = self._parse_xml(content)
        clip = root.find(".//spine/asset-clip")
        assert clip is not None
        seg = single_segment_cut_list.segments[0]
        expected_start = seconds_to_rational(seg.start, fps=24)
        assert clip.attrib["start"] == expected_start

    def test_fcpxml_first_clip_offset_zero(
        self, multi_segment_cut_list: CutList
    ) -> None:
        """First clip's offset should be 0."""
        content = generate_fcpxml(multi_segment_cut_list)
        root = self._parse_xml(content)
        clips = root.findall(".//spine/asset-clip")
        assert clips[0].attrib["offset"] == "0/1s"

    def test_fcpxml_clip_offsets_are_contiguous(
        self, multi_segment_cut_list: CutList
    ) -> None:
        """Verify clips are placed contiguously (no gaps)."""
        content = generate_fcpxml(multi_segment_cut_list, frame_rate=24)
        root = self._parse_xml(content)
        clips = root.findall(".//spine/asset-clip")

        def parse_rational(s: str) -> float:
            """Parse 'N/Ds' or '0/1s' to float."""
            s = s.rstrip("s")
            num, den = s.split("/")
            return int(num) / int(den)

        cumulative = 0.0
        for clip in clips:
            offset = parse_rational(clip.attrib["offset"])
            duration = parse_rational(clip.attrib["duration"])
            assert abs(offset - cumulative) < 1e-9, (
                f"Gap at clip {clip.attrib['name']}: expected offset {cumulative}, got {offset}"
            )
            cumulative += duration

    def test_fcpxml_rational_time_format(self, single_segment_cut_list: CutList) -> None:
        """All time values should end with 's' and contain '/'."""
        content = generate_fcpxml(single_segment_cut_list)
        root = self._parse_xml(content)
        clip = root.find(".//spine/asset-clip")
        assert clip is not None
        for attr in ["start", "duration", "offset"]:
            val = clip.attrib[attr]
            assert "/" in val and val.endswith("s"), (
                f"Attribute {attr}='{val}' is not rational time format"
            )

    def test_fcpxml_preserves_segment_order(
        self, multi_source_cut_list: CutList
    ) -> None:
        content = generate_fcpxml(multi_source_cut_list)
        root = self._parse_xml(content)
        clips = root.findall(".//spine/asset-clip")
        assert len(clips) == 4
        expected_names = [
            "interview_wide",
            "broll_sunset",
            "interview_closeup",
            "interview_wide",
        ]
        actual_names = [c.attrib["name"] for c in clips]
        assert actual_names == expected_names

    def test_fcpxml_25fps_frame_duration(
        self, single_segment_cut_list: CutList
    ) -> None:
        content = generate_fcpxml(single_segment_cut_list, frame_rate=25)
        root = self._parse_xml(content)
        fmt = root.find("resources/format")
        assert fmt is not None
        assert fmt.attrib["frameDuration"] == "1/25s"

    def test_fcpxml_30fps_frame_duration(
        self, single_segment_cut_list: CutList
    ) -> None:
        content = generate_fcpxml(single_segment_cut_list, frame_rate=30)
        root = self._parse_xml(content)
        fmt = root.find("resources/format")
        assert fmt is not None
        assert fmt.attrib["frameDuration"] == "1/30s"

    def test_fcpxml_asset_has_src_with_file_uri(
        self, single_segment_cut_list: CutList
    ) -> None:
        content = generate_fcpxml(single_segment_cut_list)
        root = self._parse_xml(content)
        asset = root.find("resources/asset")
        assert asset is not None
        src = asset.attrib.get("src", "")
        assert src.startswith("file:///"), f"src should be file URI: {src}"

    def test_fcpxml_returns_string(self, single_segment_cut_list: CutList) -> None:
        result = generate_fcpxml(single_segment_cut_list)
        assert isinstance(result, str)

    def test_fcpxml_single_source_one_asset(
        self, multi_segment_cut_list: CutList
    ) -> None:
        """Multiple clips from same source should produce exactly 1 asset."""
        content = generate_fcpxml(multi_segment_cut_list)
        root = self._parse_xml(content)
        assets = root.findall("resources/asset")
        assert len(assets) == 1


# ===========================================================================
# DaVinci Resolve Script Tests
# ===========================================================================


class TestGenerateDavinciScript:
    """Tests for the DaVinci Resolve Python script generator."""

    def test_davinci_script_is_valid_python(
        self, multi_segment_cut_list: CutList
    ) -> None:
        """Generated script must compile without syntax errors."""
        script = generate_davinci_script(multi_segment_cut_list)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp_path = f.name
        try:
            py_compile.compile(tmp_path, doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Generated DaVinci script has syntax errors: {e}")

    def test_davinci_script_imports_dvr_module(
        self, multi_segment_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(multi_segment_cut_list)
        assert "import DaVinciResolveScript" in script

    def test_davinci_script_calls_scriptapp(
        self, multi_segment_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(multi_segment_cut_list)
        assert 'scriptapp("Resolve")' in script

    def test_davinci_script_calls_create_project(
        self, multi_segment_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(multi_segment_cut_list)
        assert "CreateProject" in script

    def test_davinci_script_calls_import_media(
        self, multi_segment_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(multi_segment_cut_list)
        assert "ImportMedia" in script

    def test_davinci_script_calls_create_timeline(
        self, multi_segment_cut_list: CutList
    ) -> None:
        """Script must call CreateTimelineFromClips or CreateEmptyTimeline."""
        script = generate_davinci_script(multi_segment_cut_list)
        assert "CreateTimelineFromClips" in script or "CreateEmptyTimeline" in script

    def test_davinci_script_contains_media_paths(
        self, multi_segment_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(multi_segment_cut_list)
        assert "/videos/interview.mp4" in script

    def test_davinci_script_multi_source_contains_all_paths(
        self, multi_source_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(multi_source_cut_list)
        assert "/videos/interview_wide.mp4" in script
        assert "/videos/broll_sunset.mp4" in script
        assert "/videos/interview_closeup.mp4" in script

    def test_davinci_script_multi_source_unique_paths_only(
        self, multi_source_cut_list: CutList
    ) -> None:
        """interview_wide.mp4 appears twice in cut list but once in MEDIA_FILES."""
        script = generate_davinci_script(multi_source_cut_list)
        # Count occurrences in MEDIA_FILES block
        media_files_block = _extract_media_files_block(script)
        assert media_files_block.count("interview_wide.mp4") == 1
        assert media_files_block.count("broll_sunset.mp4") == 1
        assert media_files_block.count("interview_closeup.mp4") == 1

    def test_davinci_script_contains_frame_values(
        self, single_segment_cut_list: CutList
    ) -> None:
        """Frame values must appear in the CLIPS list."""
        script = generate_davinci_script(single_segment_cut_list, frame_rate=24)
        seg = single_segment_cut_list.segments[0]
        expected_start = round(seg.start * 24)
        expected_end = round(seg.end * 24)
        assert f'"start_frame": {expected_start}' in script
        assert f'"end_frame": {expected_end}' in script

    def test_davinci_script_24fps_frame_values(
        self, single_segment_cut_list: CutList
    ) -> None:
        """5.0s at 24fps = 120 frames, 10.0s = 240 frames."""
        script = generate_davinci_script(single_segment_cut_list, frame_rate=24)
        assert '"start_frame": 120' in script
        assert '"end_frame": 240' in script

    def test_davinci_script_25fps_frame_values(
        self, single_segment_cut_list: CutList
    ) -> None:
        """5.0s at 25fps = 125 frames, 10.0s = 250 frames."""
        script = generate_davinci_script(single_segment_cut_list, frame_rate=25)
        assert '"start_frame": 125' in script
        assert '"end_frame": 250' in script

    def test_davinci_script_30fps_frame_values(
        self, single_segment_cut_list: CutList
    ) -> None:
        """5.0s at 30fps = 150 frames, 10.0s = 300 frames."""
        script = generate_davinci_script(single_segment_cut_list, frame_rate=30)
        assert '"start_frame": 150' in script
        assert '"end_frame": 300' in script

    def test_davinci_script_includes_descriptions(
        self, multi_segment_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(multi_segment_cut_list)
        assert "Opening greeting" in script
        assert "Main point" in script
        assert "Closing statement" in script

    def test_davinci_script_contains_frame_rate(
        self, single_segment_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(single_segment_cut_list, frame_rate=25)
        assert "FRAME_RATE = 25" in script

    def test_davinci_script_empty_cut_list_uses_create_empty_timeline(
        self, empty_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(empty_cut_list)
        assert "CreateEmptyTimeline" in script

    def test_davinci_script_empty_cut_list_is_valid_python(
        self, empty_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(empty_cut_list)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp_path = f.name
        try:
            py_compile.compile(tmp_path, doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Empty cut list DaVinci script has syntax errors: {e}")

    def test_davinci_script_multi_source_is_valid_python(
        self, multi_source_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(multi_source_cut_list)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp_path = f.name
        try:
            py_compile.compile(tmp_path, doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Multi-source DaVinci script has syntax errors: {e}")

    def test_davinci_script_returns_string(
        self, single_segment_cut_list: CutList
    ) -> None:
        result = generate_davinci_script(single_segment_cut_list)
        assert isinstance(result, str)

    def test_davinci_script_contains_set_frame_rate(
        self, single_segment_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(single_segment_cut_list)
        assert "SetSetting" in script
        assert "timelineFrameRate" in script

    def test_davinci_script_media_index_correct_for_multi_source(
        self, multi_source_cut_list: CutList
    ) -> None:
        """Clip media_index must reference correct source file index."""
        script = generate_davinci_script(multi_source_cut_list)
        # interview_wide.mp4 is index 0, broll_sunset.mp4 is index 1,
        # interview_closeup.mp4 is index 2
        # There are 4 clips: 0, 1, 2, 0 (interview_wide appears at positions 0 and 3)
        clips_block = _extract_clips_block(script)
        # Count occurrences of media_index: 0 (should be 2 for interview_wide)
        assert clips_block.count('"media_index": 0') == 2
        assert clips_block.count('"media_index": 1') == 1
        assert clips_block.count('"media_index": 2') == 1

    def test_davinci_script_saves_project(
        self, single_segment_cut_list: CutList
    ) -> None:
        script = generate_davinci_script(single_segment_cut_list)
        assert "SaveProject" in script


# ===========================================================================
# Cross-Format Tests
# ===========================================================================


class TestCrossFormatConsistency:
    """Tests verifying consistent behavior across all three generators."""

    def test_all_generators_handle_empty_cut_list(
        self, empty_cut_list: CutList
    ) -> None:
        """All generators must accept an empty cut list without raising."""
        edl = generate_edl(empty_cut_list)
        fcpxml = generate_fcpxml(empty_cut_list)
        davinci = generate_davinci_script(empty_cut_list)
        assert isinstance(edl, str)
        assert isinstance(fcpxml, str)
        assert isinstance(davinci, str)

    def test_all_generators_handle_single_segment(
        self, single_segment_cut_list: CutList
    ) -> None:
        edl = generate_edl(single_segment_cut_list)
        fcpxml = generate_fcpxml(single_segment_cut_list)
        davinci = generate_davinci_script(single_segment_cut_list)
        # EDL: 1 event
        lines = edl.splitlines()
        events = [ln for ln in lines if re.match(r"^\d{3}\s", ln)]
        assert len(events) == 1
        # FCPXML: 1 asset-clip
        root = ET.fromstring(
            "\n".join(
                ln for ln in fcpxml.splitlines() if not ln.startswith("<!DOCTYPE")
            )
        )
        assert len(root.findall(".//spine/asset-clip")) == 1
        # DaVinci: valid Python
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(davinci)
            tmp_path = f.name
        py_compile.compile(tmp_path, doraise=True)

    def test_all_generators_handle_multi_source(
        self, multi_source_cut_list: CutList
    ) -> None:
        edl = generate_edl(multi_source_cut_list)
        fcpxml = generate_fcpxml(multi_source_cut_list)
        davinci = generate_davinci_script(multi_source_cut_list)
        # All three reference broll_sunset
        assert "broll_sunset" in edl
        assert "broll_sunset" in fcpxml
        assert "broll_sunset" in davinci

    def test_all_generators_accept_different_frame_rates(
        self, single_segment_cut_list: CutList
    ) -> None:
        for fps in [24, 25, 30]:
            edl = generate_edl(single_segment_cut_list, frame_rate=fps)
            fcpxml = generate_fcpxml(single_segment_cut_list, frame_rate=fps)
            davinci = generate_davinci_script(single_segment_cut_list, frame_rate=fps)
            assert isinstance(edl, str)
            assert isinstance(fcpxml, str)
            assert isinstance(davinci, str)


# ===========================================================================
# Helpers
# ===========================================================================


def _extract_media_files_block(script: str) -> str:
    """Extract the MEDIA_FILES list from the script."""
    start = script.find("MEDIA_FILES = [")
    end = script.find("]", start) + 1
    return script[start:end]


def _extract_clips_block(script: str) -> str:
    """Extract the CLIPS list from the script."""
    start = script.find("CLIPS = [")
    # Find matching closing bracket
    depth = 0
    for i, ch in enumerate(script[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return script[start : i + 1]
    return script[start:]
