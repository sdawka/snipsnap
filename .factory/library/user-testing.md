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
