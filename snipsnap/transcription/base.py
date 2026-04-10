"""Abstract base class for transcription providers.

Defines the TranscriptionProvider interface that all backends must implement.
The base class also provides a default implementation of discover_videos()
and transcribe_batch() so concrete providers only need to implement transcribe().
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional

from snipsnap.models import Transcription

logger = logging.getLogger(__name__)

# Video file extensions supported by the pipeline
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".mov", ".avi", ".webm"}
)


class TranscriptionProvider(ABC):
    """Abstract base class for transcription backends.

    Concrete providers must implement :meth:`transcribe`.
    The :meth:`discover_videos` and :meth:`transcribe_batch` methods are
    provided as a default template implementation.
    """

    def discover_videos(self, folder_path: Path | str) -> List[Path]:
        """Recursively discover all supported video files under *folder_path*.

        Args:
            folder_path: Root directory to search. Must exist.

        Returns:
            Sorted list of :class:`~pathlib.Path` objects for each video file
            found.  Returns an empty list if the folder contains no supported
            video files.

        Raises:
            FileNotFoundError: If *folder_path* does not exist.
            NotADirectoryError: If *folder_path* is not a directory.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")

        videos: List[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            videos.extend(folder.rglob(f"*{ext}"))

        return sorted(videos)

    @abstractmethod
    def transcribe(self, video_path: Path | str) -> Transcription:
        """Transcribe a single video file.

        Args:
            video_path: Path to the video file.

        Returns:
            :class:`~snipsnap.models.Transcription` with segment-level
            timestamps ordered chronologically by start time.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            Exception: Any provider-specific error during transcription.
        """
        ...

    def transcribe_batch(
        self,
        folder_path: Path | str,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        force: bool = False,
        data_dir: Optional[Path] = None,
    ) -> List[Transcription]:
        """Transcribe all supported videos in *folder_path*.

        Already-transcribed files are skipped by default; pass ``force=True``
        to re-transcribe them.  Errors on individual files are logged as
        warnings and the batch continues with the remaining files.

        Args:
            folder_path: Directory containing video files.
            on_progress: Optional callback invoked for every file (including
                skipped ones) with signature ``(filename, current, total)``.
                *current* is 1-based.
            force: When *True*, re-transcribe files that already have a
                persisted transcription JSON on disk.
            data_dir: Override the storage data directory.  Defaults to the
                value from :func:`snipsnap.config.get_config`.

        Returns:
            List of :class:`~snipsnap.models.Transcription` objects, one for
            each successfully processed (or loaded) video file.
        """
        # Import here to avoid circular imports at module load time
        from snipsnap.storage import load_transcription, save_transcription, transcription_exists

        videos = self.discover_videos(folder_path)
        total = len(videos)
        results: List[Transcription] = []

        for idx, video_path in enumerate(videos, start=1):
            filename = video_path.name

            # Check whether a transcription already exists for this file
            if not force and transcription_exists(str(video_path), data_dir):
                logger.info("Skipping already-transcribed file: %s", filename)
                existing = load_transcription(str(video_path), data_dir)
                if existing is not None:
                    results.append(existing)
                if on_progress is not None:
                    on_progress(filename, idx, total)
                continue

            try:
                transcription = self.transcribe(video_path)
                save_transcription(transcription, data_dir)
                results.append(transcription)
                logger.info("Transcribed %s (%d/%d)", filename, idx, total)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to transcribe %s: %s", filename, exc)

            if on_progress is not None:
                on_progress(filename, idx, total)

        return results
