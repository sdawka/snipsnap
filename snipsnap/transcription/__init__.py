"""Transcription provider abstraction and implementations."""

from snipsnap.transcription.base import SUPPORTED_EXTENSIONS, TranscriptionProvider
from snipsnap.transcription.whisper_local import WhisperLocalProvider

__all__ = ["TranscriptionProvider", "WhisperLocalProvider", "SUPPORTED_EXTENSIONS"]
