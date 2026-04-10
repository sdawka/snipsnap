---
name: web-worker
description: Implements FastAPI backend and Alpine.js + UnoCSS frontend for the web dashboard
---

# Web Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for features involving:
- FastAPI backend routes and API endpoints
- Alpine.js frontend components and interactivity
- UnoCSS styling
- Web dashboard pages (transcription browser, curation prompt, cut list viewer, timeline)
- Static file serving

## Required Skills

- `agent-browser` — MUST be invoked to verify all web UI features. After implementing each page/component, use agent-browser to navigate to the page, interact with it, and verify correct rendering and behavior.

## Work Procedure

1. **Read context**: Read `.factory/library/architecture.md` for system design. Read `.factory/services.yaml` for how to start the web server. Check the feature's preconditions — the core engines (transcription, curation, export) should already be implemented.

2. **Write API tests first (RED)**: Create test files in `tests/` for FastAPI endpoints using TestClient:
   - Test each endpoint returns correct status codes and JSON structure
   - Test error responses (404 for missing resources, 400 for invalid input)
   - Test that endpoints correctly call the underlying engine functions
   Run tests: `source .venv/bin/activate && python -m pytest tests/ -x -v --tb=short`

3. **Implement backend (GREEN)**: Build FastAPI routes in `snipsnap/web/app.py`:
   - Mount static files directory
   - REST endpoints: GET /api/transcriptions, GET /api/transcriptions/{id}, POST /api/curate, GET /api/cutlists, GET /api/cutlists/{id}, POST /api/export
   - Health endpoint: GET /health
   - All endpoints use the same storage/engine layer as CLI
   - Return proper JSON responses with appropriate status codes

4. **Implement frontend**: Create HTML files in `snipsnap/web/static/`:
   - Load Alpine.js and UnoCSS via CDN (no build step)
   - `index.html` — main dashboard with navigation between sections
   - Transcription browser: list videos, view segments, search text
   - Curation prompt: form with textarea, submit button, progress indicator
   - Cut list viewer: segment list with timeline visualization
   - Export: buttons for each format (EDL, FCPXML, DaVinci)
   - Handle empty states (no transcriptions, no cut lists)
   - Handle loading states (spinners during API calls)
   - Handle error states (API errors shown to user)

5. **Run validators**:
   ```
   source .venv/bin/activate && python -m pytest tests/ -x -v --tb=short
   source .venv/bin/activate && python -m mypy snipsnap/web/ --ignore-missing-imports
   source .venv/bin/activate && python -m ruff check snipsnap/web/ tests/
   ```

6. **Manual verification with agent-browser**: Start the web server and use agent-browser to verify EVERY page and interaction:
   - Start server: `source .venv/bin/activate && PORT=3200 python -m uvicorn snipsnap.web.app:app --host 0.0.0.0 --port 3200 &`
   - Navigate to http://localhost:3200
   - Verify dashboard loads, navigation works
   - Test each section: transcription browser, curation prompt, cut list viewer, export
   - Test empty states (when no data exists)
   - Check for console errors
   - Stop server when done: `lsof -ti :3200 | xargs kill`
   Record each check with what you did and what you observed.

7. **Update shared state**: Add any web-specific patterns or gotchas to `.factory/library/`.

## Example Handoff

```json
{
  "salientSummary": "Built FastAPI backend with 6 API endpoints and Alpine.js dashboard with transcription browser, curation prompt form, cut list timeline viewer, and export buttons. All 15 API tests pass. Verified all pages via agent-browser — dashboard loads, transcription list populates, curation form submits and shows progress, timeline renders segments correctly, export triggers downloads.",
  "whatWasImplemented": "FastAPI app in snipsnap/web/app.py with endpoints: GET /health, GET /api/transcriptions, GET /api/transcriptions/{id}, POST /api/curate, GET /api/cutlists, GET /api/cutlists/{id}, POST /api/export/{cutlist_id}. Frontend in snipsnap/web/static/index.html using Alpine.js for state management and UnoCSS for styling. Includes transcription browser with text search, curation prompt form with loading spinner, cut list viewer with SVG timeline visualization, and export buttons for EDL/FCPXML/DaVinci formats.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "source .venv/bin/activate && python -m pytest tests/test_web.py -v", "exitCode": 0, "observation": "15 tests passed covering all API endpoints"},
      {"command": "source .venv/bin/activate && python -m mypy snipsnap/web/ --ignore-missing-imports", "exitCode": 0, "observation": "No type errors"},
      {"command": "source .venv/bin/activate && python -m ruff check snipsnap/web/", "exitCode": 0, "observation": "No lint issues"}
    ],
    "interactiveChecks": [
      {"action": "Opened http://localhost:3200 in agent-browser", "observed": "Dashboard loads with navigation tabs for Transcriptions, Curate, Cut Lists. UnoCSS styles applied correctly."},
      {"action": "Clicked Transcriptions tab with existing transcription data", "observed": "List shows 3 transcribed videos with filenames and durations. Clicking a video shows segments with timestamps."},
      {"action": "Typed search term in transcription search box", "observed": "Segments filtered to show only matching text. Clearing search restores full list."},
      {"action": "Submitted curation prompt 'find highlights'", "observed": "Submit button disabled, spinner shown. After ~3 seconds, cut list appears with 5 segments."},
      {"action": "Viewed cut list timeline", "observed": "SVG timeline shows 5 colored blocks at correct proportional positions. Different colors for different source files."},
      {"action": "Clicked Export EDL button", "observed": "File download initiated. EDL content valid with 5 events."},
      {"action": "Checked browser console for errors", "observed": "No JavaScript errors or failed network requests."}
    ]
  },
  "tests": {
    "added": [
      {"file": "tests/test_web.py", "cases": [
        {"name": "test_health_endpoint", "verifies": "GET /health returns 200"},
        {"name": "test_list_transcriptions", "verifies": "GET /api/transcriptions returns list"},
        {"name": "test_get_transcription_detail", "verifies": "GET /api/transcriptions/{id} returns segments"},
        {"name": "test_curate_endpoint", "verifies": "POST /api/curate accepts prompt and returns cut list"},
        {"name": "test_export_endpoint", "verifies": "POST /api/export returns file in requested format"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Core engine modules (transcription, curation, export) are missing or broken
- Port 3200 is in use and cannot be freed
- Alpine.js or UnoCSS CDN is unreachable (consider fallback approach)
- agent-browser cannot connect to the local server
- Frontend requires functionality not exposed by the existing backend engines
