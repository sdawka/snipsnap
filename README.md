# SnipSnap

Multi-stage video curation pipeline — transcribe raw video, curate segments with an LLM, and export edit decision lists for professional NLEs.

## What It Does

SnipSnap processes video in three stages:

1. **Transcribe** — Extracts speech from video files using faster-whisper (local, offline).
2. **Curate** — Sends transcriptions to an LLM (via OpenRouter) with a natural-language prompt to identify relevant segments. Uses a multi-pass chunking strategy to handle hours of source material.
3. **Export** — Converts the curated cut list into EDL (CMX 3600), FCPXML 1.8, or DaVinci Resolve script format.

Both the CLI and web dashboard are thin interfaces over the same pipeline and shared JSON storage.

## Quick Start

```bash
git clone <repo-url> && cd snipsnap
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the setup wizard:

```bash
snipsnap init
```

Run the full pipeline in one command:

```bash
snipsnap run ./raw-footage --prompt "find the funniest moments"
```

This transcribes all videos, curates with your prompt, and exports an EDL file.

## CLI Commands

### `snipsnap run <folder> --prompt "..."`

Run the full pipeline in one command: transcribe → curate → export.

```bash
snipsnap run ./footage --prompt "find the best moments"
snipsnap run ./footage --prompt "key arguments" --format fcpxml
```

| Flag | Description |
|------|-------------|
| `--prompt PROMPT` | What to curate (required) |
| `--format FORMAT` | Export format: `edl`, `fcpxml`, `davinci` (default: edl) |
| `--output FILE` | Output file path (default: auto-generated) |
| `--fps FPS` | Frame rate (default: 24) |

### `snipsnap init`

Interactive setup wizard that creates your `.env` configuration file.

| Flag | Description |
|------|-------------|
| `--api-key KEY` | Provide API key directly (skips prompt) |
| `--model MODEL` | LLM model to use |
| `--data-dir DIR` | Data directory path |
| `--non-interactive` | Skip prompts, use defaults (requires `--api-key`) |

### `snipsnap transcribe <folder>`

Transcribe all video files (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`) in a folder.

| Flag | Description |
|------|-------------|
| `--model MODEL` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`) |
| `--force` | Re-transcribe files that already have a transcription on disk |

### `snipsnap curate --prompt "..."`

Curate transcriptions with an LLM prompt and produce a cut list.

| Flag | Description |
|------|-------------|
| `--prompt PROMPT` | Natural-language curation prompt (required) |
| `--model MODEL` | Override the LLM model (default: configured model) |
| `--data-dir DIR` | Override the data directory |

### `snipsnap export <cut-list-id>`

Export a cut list to a professional NLE format.

| Flag | Description |
|------|-------------|
| `--format FORMAT` | Output format: `edl`, `fcpxml`, or `davinci` (required) |
| `--output PATH` | Output file path (default: auto-generated in exports dir) |
| `--fps FPS` | Frame rate for timecode conversion (default: 24.0) |
| `--data-dir DIR` | Override the data directory |

### `snipsnap status`

Show the current pipeline state — number of transcriptions and available cut lists.

| Flag | Description |
|------|-------------|
| `--data-dir DIR` | Override the data directory |

## Web Dashboard

```bash
uvicorn snipsnap.web.app:app --port 3200
```

Open [http://localhost:3200](http://localhost:3200). The dashboard provides the same transcribe → curate → export workflow through a browser UI built with Alpine.js and UnoCSS.

## Configuration

Environment variables (set in `.env`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | — | API key for LLM curation via OpenRouter |
| `SNIPSNAP_DATA_DIR` | No | `./snipsnap_data/` | Directory for transcriptions and cut lists |
| `SNIPSNAP_MODEL` | No | `google/gemini-2.5-flash-lite` | LLM model used for curation |
| `SNIPSNAP_WHISPER_MODEL` | No | `small` | Whisper model size for transcription |

## Development

```bash
pytest              # run tests
mypy snipsnap       # type checking
ruff check snipsnap # linting
```

Requires Python ≥ 3.11.

## Troubleshooting

**"OpenRouter API key is not configured"**
Run `snipsnap init` to set up your configuration, or manually add `OPENROUTER_API_KEY` to your `.env` file.

**"Your OpenRouter API key was rejected"**
Get a new API key from https://openrouter.ai/keys and run `snipsnap init` again.

**"Rate limit exceeded"**
Wait a few minutes and try again. Check your usage at https://openrouter.ai/activity.

**"No transcriptions found"**
Run `snipsnap transcribe ./your-folder` first to process your video files.

**"Cut list not found"**
Run `snipsnap status` to see available cut list IDs.
