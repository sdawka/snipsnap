# SnipSnap Architecture

SnipSnap is a multi-stage video curation pipeline. Users provide raw video files and natural-language prompts; the system transcribes audio, uses an LLM to identify relevant segments, and exports edit decision lists in professional NLE formats.

## Package Structure

```
snipsnap/
├── __init__.py
├── cli.py              # Click CLI entry point
├── config.py           # Configuration (env vars, defaults)
├── models.py           # Data models (Transcription, Segment, CutList, CutSegment)
├── storage.py          # JSON file storage layer
├── transcription/
│   ├── __init__.py
│   ├── base.py         # TranscriptionProvider abstract base class
│   └── whisper_local.py # faster-whisper implementation
├── curation/
│   ├── __init__.py
│   ├── engine.py       # LLM curation engine
│   └── chunking.py     # Multi-pass chunking strategy
├── export/
│   ├── __init__.py
│   ├── edl.py          # EDL (CMX 3600) generator
│   ├── fcpxml.py       # FCPXML 1.8 generator
│   └── davinci.py      # DaVinci Resolve script generator
└── web/
    ├── __init__.py
    ├── app.py          # FastAPI app + routes
    └── static/         # HTML, JS, CSS (Alpine.js + UnoCSS via CDN)
```

## Pipeline Overview

The system operates as a three-stage pipeline with a shared storage layer:

```
Video files
    │
    ▼
┌──────────────┐
│ Transcription │  (faster-whisper, provider pattern)
└──────┬───────┘
       │  Transcription JSON
       ▼
┌──────────────┐
│   Storage    │  (JSON file I/O)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Curation   │  (OpenRouter via OpenAI SDK, multi-pass chunking)
└──────┬───────┘
       │  CutList JSON
       ▼
┌──────────────┐
│    Export    │  (EDL, FCPXML, DaVinci Resolve script)
└──────────────┘
       │
       ▼
  Output files
```

Both the CLI (Click) and the web dashboard (FastAPI + Alpine.js + UnoCSS on port 3200) invoke the same pipeline stages and share the same storage layer. They are thin interfaces over common engine code.

## Components and Relationships

### Transcription Layer

The transcription layer uses a **provider pattern**. An abstract base class (`TranscriptionProvider`) defines the interface; concrete implementations (e.g., `whisper_local` using faster-whisper) fulfill it. This makes it straightforward to add new transcription backends without modifying the rest of the pipeline.

### Storage Layer

A dedicated storage module abstracts all JSON file I/O. Transcription results and cut lists are persisted as JSON files. Both CLI and web dashboard read and write through this single layer, ensuring consistency regardless of which interface is used.

### Curation Engine

The curation engine sends transcription data and a user-supplied prompt to an LLM (accessed via OpenRouter through the OpenAI SDK). It uses a **multi-pass chunking strategy** to stay within context limits and produce precise results (see below).

### Export Generators

Export modules convert a CutList into professional NLE formats:

- **EDL (CMX 3600):** Industry-standard edit decision list with SMPTE timecode.
- **FCPXML 1.8:** Apple Final Cut Pro XML interchange format using rational time.
- **DaVinci Resolve script:** Automation script for Resolve's scripting API.

Each generator converts from the internal seconds-based timestamps to the format's native time representation.

### CLI and Web Dashboard

- The **CLI** (`cli.py`) uses Click to expose pipeline operations as commands.
- The **web dashboard** (`web/app.py`) uses FastAPI for the backend and Alpine.js + UnoCSS (loaded via CDN) for the frontend, served on port 3200.

Both interfaces call into the same transcription, curation, and export modules. Neither contains business logic of its own.

## Data Models

### Transcription

Represents the full transcription of a single video file.

- `source_file` — path to the original video
- `duration` — total duration of the video
- `language` — detected or specified language
- `model` — transcription model used
- `segments[]` — ordered list of Segment objects

### Segment

A single transcribed utterance within a video.

- `start` — start time in seconds (float)
- `end` — end time in seconds (float)
- `text` — transcribed text

### CutList

A curated selection of video segments produced by the curation engine.

- `id` — unique identifier (UUID)
- `prompt` — the user's natural-language curation prompt
- `theme` — high-level theme or title for the cut
- `created_at` — creation timestamp
- `segments[]` — ordered list of CutSegment objects
- `total_duration` — sum of all segment durations

### CutSegment

A single segment selected for inclusion in the final cut.

- `source_file` — reference to the originating video's transcription
- `start` — start time in seconds (float)
- `end` — end time in seconds (float)
- `description` — LLM-generated description of why this segment was selected
- `order` — position in the final sequence

## Multi-Pass Chunking Strategy

The curation engine processes transcriptions in three passes to balance token efficiency with timestamp precision:

1. **Pass 1 — Summarize:** Each video's transcription is individually summarized by the LLM, reducing token count while preserving topical content.
2. **Pass 2 — Identify:** All summaries are sent together with the user's prompt. The LLM identifies which videos and approximate time ranges are relevant.
3. **Pass 3 — Extract:** Only the relevant full transcripts (narrowed by Pass 2) are sent to the LLM. It extracts precise start/end timestamps and generates the final cut list.

This three-pass approach allows the system to scale across many hours of source video without exceeding LLM context windows.

## Key Invariants

- **Timestamps are always in seconds (float).** All internal data models and storage use seconds. Conversion to SMPTE timecode, rational time, or frame numbers happens only at the export boundary.
- **Transcription segments are ordered by start time.** Consumers may rely on this ordering.
- **Cut list segments reference existing transcriptions.** Every `source_file` in a CutSegment must correspond to a transcription that exists in storage.
- **CLI and web dashboard share storage.** Both interfaces read and write the same JSON files on disk. There is no separate database or state store.
- **Export is a pure transformation.** Export generators read a CutList and produce an output file. They do not modify stored data.

## Export Path and Artifact Notes

- `source_file` values can be relative paths depending on how transcription discovery is invoked. Exporters that construct file URIs (especially FCPXML) must preserve the full relative path components rather than dropping intermediate directories.
- CLI export defaults to writing files under `<data_dir>/exports/`. Validation/test flows should prefer sandboxed `--data-dir` values to avoid leaving runtime export artifacts in the repository root.
