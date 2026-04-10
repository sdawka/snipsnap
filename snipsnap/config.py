"""Configuration loading for SnipSnap.

Reads environment variables from .env via python-dotenv.
All settings have sensible defaults so the tool can be used without
a fully populated .env file (except OPENROUTER_API_KEY for curation).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import time (idempotent; doesn't overwrite existing vars)
load_dotenv()


@dataclass
class Config:
    """Typed configuration for SnipSnap."""

    openrouter_api_key: str
    data_dir: Path
    model: str
    whisper_model: str

    def __repr__(self) -> str:
        # Never expose the API key in repr
        key_hint = "***" if self.openrouter_api_key else "<not set>"
        return (
            f"Config("
            f"openrouter_api_key={key_hint}, "
            f"data_dir={self.data_dir!r}, "
            f"model={self.model!r}, "
            f"whisper_model={self.whisper_model!r})"
        )


def get_config() -> Config:
    """Build and return a Config from the current environment."""
    data_dir_str = os.getenv("SNIPSNAP_DATA_DIR", "./snipsnap_data")
    return Config(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        data_dir=Path(data_dir_str),
        model=os.getenv("SNIPSNAP_MODEL", "google/gemini-2.5-flash-lite"),
        whisper_model=os.getenv("SNIPSNAP_WHISPER_MODEL", "small"),
    )
