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
from snipsnap.curation.engine import (
    AuthenticationError,
    CurationEngine,
    CurationError,
    MissingApiKeyError,
)
from snipsnap.storage import (
    load_all_cut_lists,
    load_all_transcriptions,
    load_cut_list,
    save_cut_list,  # noqa: F401 – imported for test patching
    save_transcription,
    transcription_exists,
)
from snipsnap.transcription.base import SUPPORTED_EXTENSIONS
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
    # Discover videos (before loading the model to fail fast on empty folders)
    # ------------------------------------------------------------------
    videos: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        videos.extend(folder_path.rglob(f"*{ext}"))
    videos.sort()

    if not videos:
        click.echo(f"No video files found in: {folder}", err=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load transcription provider (only after confirming there are videos)
    # ------------------------------------------------------------------
    click.echo(f"Loading Whisper model '{whisper_model}'…")
    try:
        provider = WhisperLocalProvider(model_size=whisper_model)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Error: Failed to load Whisper model '{whisper_model}': {exc}", err=True)
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
    parts = [f"{completed} succeeded"]
    if skipped:
        parts.append(f"{skipped} skipped")
    if failed:
        parts.append(f"{failed} failed")
    click.echo("Summary: " + ", ".join(parts) + ".")

    # Exit with non-zero if any file failed.
    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# curate
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--prompt",
    required=True,
    metavar="PROMPT",
    help="Natural-language curation prompt (e.g. 'find funny moments').",
)
@click.option(
    "--model",
    default=None,
    metavar="LLM_MODEL",
    help=(
        "LLM model identifier to use (OpenRouter model string). "
        "Defaults to SNIPSNAP_MODEL or 'google/gemini-2.5-flash-lite'."
    ),
)
@click.option(
    "--data-dir",
    default=None,
    metavar="DIR",
    type=click.Path(),
    help="Override the data directory (default: configured data directory).",
)
def curate(prompt: str, model: Optional[str], data_dir: Optional[str]) -> None:
    """Curate transcriptions with an LLM prompt and produce a cut list.

    Loads all existing transcriptions from storage, passes them to the
    curation engine together with PROMPT, and writes a new cut list to disk.
    Prints the cut list ID and a formatted summary of selected segments.

    Run 'snipsnap transcribe <folder>' first if no transcriptions exist.
    """
    # ------------------------------------------------------------------
    # Resolve data directory override
    # ------------------------------------------------------------------
    data_path: Optional[Path] = Path(data_dir) if data_dir else None

    # ------------------------------------------------------------------
    # Load transcriptions
    # ------------------------------------------------------------------
    transcriptions = load_all_transcriptions(data_path)

    if not transcriptions:
        click.echo(
            "Error: No transcriptions found. "
            "Run 'snipsnap transcribe <folder>' first to generate transcriptions.",
            err=True,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve configuration and API key
    # ------------------------------------------------------------------
    config = get_config()
    api_key = config.openrouter_api_key

    if not api_key or not api_key.strip():
        click.echo(
            "Error: OpenRouter API key is not configured. "
            "Set the OPENROUTER_API_KEY environment variable or add it to your .env file.",
            err=True,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve LLM model
    # ------------------------------------------------------------------
    llm_model = model or config.model

    # ------------------------------------------------------------------
    # Build OpenAI client (pointing at OpenRouter)
    # ------------------------------------------------------------------
    import openai  # local import – keeps module-level imports lightweight

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # ------------------------------------------------------------------
    # Run curation engine
    # ------------------------------------------------------------------
    engine = CurationEngine(client=client, api_key=api_key)

    click.echo(
        f"Curating {len(transcriptions)} transcription(s) with prompt: {prompt!r}"
    )
    click.echo(f"Using model: {llm_model}")
    click.echo()

    try:
        cut_list = engine.curate(
            transcriptions=transcriptions,
            prompt=prompt,
            model=llm_model,
            data_dir=data_path,
        )
    except MissingApiKeyError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except AuthenticationError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except CurationError as exc:
        click.echo(f"Error: Curation failed: {exc}", err=True)
        sys.exit(1)
    except openai.APIStatusError as exc:
        click.echo(
            f"Error: The API returned an error (HTTP {exc.status_code}). "
            "Please check your configuration and try again.",
            err=True,
        )
        sys.exit(1)
    except openai.OpenAIError:
        click.echo(
            "Error: An unexpected error occurred while communicating with the API. "
            "Please check your configuration and try again.",
            err=True,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    click.echo(f"Cut list ID: {cut_list.id}")
    click.echo(f"Segments:    {len(cut_list.segments)}")
    click.echo(f"Duration:    {cut_list.total_duration:.1f}s")
    click.echo()

    if cut_list.segments:
        click.echo("Segments:")
        for seg in sorted(cut_list.segments, key=lambda s: s.order):
            source_name = Path(seg.source_file).name
            click.echo(
                f"  [{seg.order + 1}] {source_name}  "
                f"{seg.start:.1f}s–{seg.end:.1f}s  {seg.description}"
            )
    else:
        click.echo("No segments matched the prompt.")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--data-dir",
    default=None,
    metavar="DIR",
    type=click.Path(),
    help="Override the data directory (default: configured data directory).",
)
def status(data_dir: Optional[str]) -> None:
    """Show the current pipeline state.

    Displays the number of transcribed videos with their filenames, and all
    available cut lists with their IDs, creation timestamps, and prompts.

    Use cut list IDs shown here with 'snipsnap export <cut-list-id>'.
    """
    data_path: Optional[Path] = Path(data_dir) if data_dir else None

    # ------------------------------------------------------------------
    # Transcriptions
    # ------------------------------------------------------------------
    transcriptions = load_all_transcriptions(data_path)

    click.echo(f"Transcriptions: {len(transcriptions)}")
    if transcriptions:
        for t in sorted(transcriptions, key=lambda x: Path(x.source_file).name):
            click.echo(f"  {Path(t.source_file).name}")
    else:
        click.echo("  (none)")

    click.echo()

    # ------------------------------------------------------------------
    # Cut lists
    # ------------------------------------------------------------------
    cut_lists = load_all_cut_lists(data_path)

    click.echo(f"Cut Lists: {len(cut_lists)}")
    if cut_lists:
        for cl in sorted(cut_lists, key=lambda c: c.created_at):
            ts = cl.created_at.strftime("%Y-%m-%d %H:%M:%S")
            click.echo(f"  {cl.id}  [{ts}]  {cl.prompt!r}")
    else:
        click.echo("  (none)")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@main.command()
@click.argument("cut_list_id")
@click.option(
    "--format",
    "output_format",
    required=True,
    type=click.Choice(["edl", "fcpxml", "davinci"], case_sensitive=False),
    metavar="FORMAT",
    help=(
        "Output format: edl (CMX 3600 Edit Decision List), "
        "fcpxml (Final Cut Pro XML 1.8), or davinci (DaVinci Resolve script)."
    ),
)
@click.option(
    "--output",
    default=None,
    metavar="FILE",
    type=click.Path(),
    help="Output file path (default: <cut-list-id>.<format-extension>).",
)
@click.option(
    "--fps",
    default=24.0,
    metavar="FPS",
    type=float,
    show_default=True,
    help="Frame rate for EDL timecodes.",
)
@click.option(
    "--data-dir",
    default=None,
    metavar="DIR",
    type=click.Path(),
    help="Override the data directory (default: configured data directory).",
)
def export(
    cut_list_id: str,
    output_format: str,
    output: Optional[str],
    fps: float,
    data_dir: Optional[str],
) -> None:
    """Export a cut list to an NLE-compatible format.

    CUT_LIST_ID is the unique identifier of the cut list to export.
    Use 'snipsnap status' to discover available cut list IDs.

    \b
    Supported formats:
      edl     – CMX 3600 Edit Decision List
      fcpxml  – Final Cut Pro XML (version 1.8)
      davinci – DaVinci Resolve Python automation script
    """
    data_path: Optional[Path] = Path(data_dir) if data_dir else None

    # ------------------------------------------------------------------
    # Load the cut list
    # ------------------------------------------------------------------
    cut_list = load_cut_list(cut_list_id, data_path)

    if cut_list is None:
        click.echo(
            f"Error: Cut list '{cut_list_id}' not found. "
            "Use 'snipsnap status' to see available cut list IDs.",
            err=True,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Determine output path
    # ------------------------------------------------------------------
    ext_map = {"edl": ".edl", "fcpxml": ".fcpxml", "davinci": ".py"}
    ext = ext_map[output_format.lower()]

    if output:
        output_path = Path(output)
    else:
        # Default: <data_dir>/exports/<cut-list-id><ext>
        effective_data_dir = data_path if data_path is not None else get_config().data_dir
        exports_dir = effective_data_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        output_path = exports_dir / f"{cut_list_id}{ext}"

    # ------------------------------------------------------------------
    # Generate export (lazy imports keep module startup fast)
    # ------------------------------------------------------------------
    click.echo(f"Exporting cut list '{cut_list_id}' as {output_format.upper()}…")

    fmt = output_format.lower()
    try:
        if fmt == "edl":
            from snipsnap.export.edl import generate_edl

            content = generate_edl(cut_list, frame_rate=int(fps))
        elif fmt == "fcpxml":
            from snipsnap.export.fcpxml import generate_fcpxml

            content = generate_fcpxml(cut_list, frame_rate=int(fps))
        elif fmt == "davinci":
            from snipsnap.export.davinci import generate_davinci_script

            content = generate_davinci_script(cut_list, frame_rate=int(fps))
        else:
            click.echo(f"Error: Unsupported format: {fmt}", err=True)
            sys.exit(1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    except ImportError as exc:
        click.echo(f"Error: Export module not available: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Error: Export failed: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Exported to: {output_path}")
