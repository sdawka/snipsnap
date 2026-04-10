"""CLI entry point for SnipSnap.

Exposes pipeline operations as Click commands:

    snipsnap transcribe <folder>   – transcribe all videos in a folder
    snipsnap curate --prompt "…"   – curate transcriptions with an LLM prompt
    snipsnap export --format …     – export a cut list to EDL / FCPXML / DaVinci
    snipsnap status                – show pipeline state
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from snipsnap import __version__
from snipsnap.config import get_config
from snipsnap.storage import save_transcription, transcription_exists
from snipsnap.transcription.whisper_local import WhisperLocalProvider

# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=__version__, prog_name="snipsnap")
def main() -> None:
    """SnipSnap – multi-stage video curation pipeline."""


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


@main.command()
@click.argument("folder", type=click.Path())
@click.option(
    "--model",
    default=None,
    metavar="WHISPER_MODEL",
    help=(
        "Whisper model size to use for transcription "
        "(tiny, base, small, medium, large). "
        "Defaults to the value of SNIPSNAP_WHISPER_MODEL or 'small'."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-transcribe files that already have a transcription on disk.",
)
def transcribe(folder: str, model: Optional[str], force: bool) -> None:
    """Transcribe all video files found in FOLDER.

    Supported formats: .mp4, .mkv, .mov, .avi, .webm

    Already-transcribed files are skipped unless --force is given.
    Transcription failures for individual files are reported and the batch
    continues with the remaining files.
    """
    folder_path = Path(folder)

    # ------------------------------------------------------------------
    # Validate input folder
    # ------------------------------------------------------------------
    if not folder_path.exists():
        click.echo(f"Error: Folder not found: {folder}", err=True)
        sys.exit(1)

    if not folder_path.is_dir():
        click.echo(f"Error: Not a directory: {folder}", err=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve configuration
    # ------------------------------------------------------------------
    config = get_config()
    whisper_model = model or config.whisper_model

    # ------------------------------------------------------------------
    # Load transcription provider
    # ------------------------------------------------------------------
    click.echo(f"Loading Whisper model '{whisper_model}'…")
    try:
        provider = WhisperLocalProvider(model_size=whisper_model)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Error: Failed to load Whisper model '{whisper_model}': {exc}", err=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Discover videos
    # ------------------------------------------------------------------
    videos = provider.discover_videos(folder_path)

    if not videos:
        click.echo(f"No video files found in: {folder}", err=True)
        sys.exit(1)

    total = len(videos)
    click.echo(f"Found {total} video file(s):")
    for video in videos:
        click.echo(f"  {video.name}")
    click.echo()

    # ------------------------------------------------------------------
    # Transcribe each file
    # ------------------------------------------------------------------
    completed = 0
    skipped = 0
    failed = 0

    for idx, video_path in enumerate(videos, start=1):
        filename = video_path.name

        if not force and transcription_exists(str(video_path)):
            click.echo(f"Skipping  {idx}/{total}: {filename} (already transcribed)")
            skipped += 1
            continue

        click.echo(f"Transcribing {idx}/{total}: {filename}…")
        try:
            transcription = provider.transcribe(video_path)
            save_transcription(transcription)
            completed += 1
        except Exception as exc:  # noqa: BLE001
            click.echo(f"Warning: Failed to transcribe {filename}: {exc}", err=True)
            failed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    click.echo()
    parts = [f"{completed} transcribed"]
    if skipped:
        parts.append(f"{skipped} skipped")
    if failed:
        parts.append(f"{failed} failed")
    click.echo("Summary: " + ", ".join(parts) + ".")

    # Exit with non-zero if every file failed (nothing was processed).
    if failed > 0 and completed == 0 and skipped == 0:
        sys.exit(1)
