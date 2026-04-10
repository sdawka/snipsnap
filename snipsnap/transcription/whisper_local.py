"""Local Whisper transcription provider using faster-whisper.

Uses the faster-whisper library which provides a CTranslate2-based backend
for efficient CPU and GPU inference.  faster-whisper uses PyAV internally to
extract audio from video files, so video files can be passed directly without
a separate audio-extraction step.
"""

from __future__ import annotations

import logging
from pathlib import Path

from faster_whisper import WhisperModel

from snipsnap.models import Segment, Transcription
from snipsnap.transcription.base import TranscriptionProvider

logger = logging.getLogger(__name__)


class WhisperLocalProvider(TranscriptionProvider):
    """Transcription provider backed by faster-whisper (local CPU inference).

    The model is loaded once during construction and reused across all calls
    to :meth:`transcribe`.

    Args:
        model_size: Whisper model size identifier.  One of ``tiny``, ``base``,
            ``small`` (default), ``medium``, ``large``, or a path to a local
            model directory.
        device: Compute device.  ``"cpu"`` (default) or ``"cuda"``.
        compute_type: Quantisation type.  ``"int8"`` (default) is recommended
            for CPU inference; ``"float16"`` for GPU.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model_size = model_size
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Loaded Whisper model '%s' on %s (%s)", model_size, device, compute_type)

    def transcribe(self, video_path: Path | str) -> Transcription:
        """Transcribe *video_path* and return a :class:`~snipsnap.models.Transcription`.

        faster-whisper handles audio extraction from container formats (MP4,
        MKV, MOV, AVI, WebM) via PyAV, so no separate ffmpeg step is needed.

        Args:
            video_path: Path to the video file to transcribe.

        Returns:
            :class:`~snipsnap.models.Transcription` with segments ordered by
            start time.  Zero-length segments are dropped.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info("Transcribing: %s", video_path.name)

        segments_iter, info = self._model.transcribe(
            str(video_path),
            vad_filter=True,
        )

        segments: list[Segment] = []
        for seg in segments_iter:
            # Filter out zero-length or negative-duration segments
            if seg.end > seg.start:
                segments.append(
                    Segment(
                        start=float(seg.start),
                        end=float(seg.end),
                        text=seg.text.strip(),
                    )
                )

        # Guarantee chronological ordering (faster-whisper should already do
        # this, but be defensive)
        segments.sort(key=lambda s: s.start)

        return Transcription(
            source_file=str(video_path),
            duration=float(info.duration),
            language=info.language,
            model_used=self._model_size,
            segments=segments,
        )
