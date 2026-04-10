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

## Flow Validator Guidance: Web Dashboard Surface

- Use `agent-browser` with a non-default session and close the session after checks complete.
- Verify service availability with `curl -sf http://localhost:3200/health` before launching browser actions.
- Keep web evidence deterministic: startup/import errors, healthcheck output, and browser connection results.
- Do not rely solely on noisy browser console output when availability checks already prove surface is unreachable.
- If `agent-browser` request capture is empty, capture API evidence via `curl`, uvicorn request logs, or in-page fetch instrumentation.

## Known External Dependency Blocker

- Successful curation assertions require a valid `OPENROUTER_API_KEY`.
- If `.env` contains a placeholder/revoked key, curation-dependent assertions should be marked blocked with captured auth-error evidence.

## Known Validation Findings (Latest)

- Export user-testing round 1 validated export module availability and passed most export assertions.
- `VAL-CONFIG-002` currently fails: `--fps` changes EDL timecodes, but `FCM:` text remains unchanged (`NON-DROP FRAME`), so frame-rate signaling in header does not reflect configured fps.
- `VAL-CROSS-003` and `VAL-CROSS-006` are blocked in export milestone context because web-ui features (`web-backend`, `web-frontend`) are pending and `snipsnap.web.app` is unavailable.
