# User Testing

Testing surface, required testing skills/tools, resource cost classification per surface.

---

## Validation Surface

### CLI Surface
- **Tool:** Direct CLI execution via subprocess
- **Entry point:** `snipsnap` command (installed via `pip install -e .`)
- **Commands:** `snipsnap transcribe`, `snipsnap curate`, `snipsnap export`, `snipsnap status`
- **Assertions:** VAL-TRANS-*, VAL-CUR-*, VAL-EXP-*, VAL-CLI-*, VAL-CONFIG-*, VAL-CROSS-001-003/005-006/011

### Web Dashboard Surface
- **Tool:** agent-browser
- **URL:** http://localhost:3200
- **Backend:** FastAPI + uvicorn on port 3200
- **Frontend:** Alpine.js + UnoCSS (static HTML/JS, no build step)
- **Assertions:** VAL-WEB-*, VAL-CROSS-004/007-010

## Validation Concurrency

### CLI Surface
- **Max concurrent:** 3
- **Rationale:** CLI tests are lightweight (no browser). Each test spawns a Python process. Limit to 3 to avoid ffmpeg/whisper resource contention on 8 cores.

### Web Dashboard Surface (agent-browser)
- **Max concurrent:** 2
- **Rationale:** 16GB RAM, 8 cores, ~43% memory free (~6.9GB). Each agent-browser instance ~300MB + dev server ~200MB. 2 concurrent instances = ~800MB total, well within budget. Conservative due to potential ffmpeg/whisper background load.

## Test Data

- Sample video files needed for transcription testing
- Workers should create short (5-10 second) test videos using ffmpeg for testing purposes
- Example: `ffmpeg -f lavfi -i testsrc=duration=5:size=320x240:rate=25 -f lavfi -i sine=frequency=440:duration=5 -c:v libx264 -c:a aac -shortest test_video.mp4`

## Flow Validator Guidance: CLI

- Each flow validator must stay inside its assigned sandbox root only.
- Do not read/write another validator's sandbox, evidence, or data directory.
- Set `SNIPSNAP_DATA_DIR` to the assigned sandbox data directory for every CLI command.
- Use dedicated fixture/video directories within the sandbox for file creation.
- Avoid mutating repository source files; CLI validation should only create runtime artifacts in sandbox/evidence paths.
- If external LLM access is unavailable, record assertions as blocked with exact command output/evidence rather than forcing mocked application internals.
- Prefer `./.venv/bin/snipsnap` (or sandbox-local venv `bin/snipsnap`) instead of relying on system PATH.
- `transcribe` uses `SNIPSNAP_DATA_DIR` env for isolation (it does not expose `--data-dir`).

## Known External Dependency Blocker

- Successful curation assertions require a valid `OPENROUTER_API_KEY`.
- If `.env` contains a placeholder/revoked key, curation-dependent assertions should be marked blocked with captured auth-error evidence.
- Core-pipeline rerun (round 2) confirmed curation works in this environment, but export-dependent assertions are currently blocked by missing modules:
  - `snipsnap.export.edl`
  - `snipsnap.export.fcpxml`
  - `snipsnap.export.davinci`
  This indicates export validation should be deferred to the `export` milestone unless those modules are present.
