---
name: backend-worker
description: Implements Python backend features - models, engines, CLI commands, export generators
---

# Backend Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for features involving:
- Python package setup and data models
- Transcription engine and provider implementations
- LLM curation engine and chunking logic
- Export format generators (EDL, FCPXML, DaVinci)
- CLI commands and integration
- Core business logic and storage layer

## Required Skills

None.

## Work Procedure

1. **Read context**: Read `.factory/library/architecture.md` to understand how your feature fits the system. Read `.factory/services.yaml` for commands. Check feature preconditions.

2. **Write tests first (RED)**: Create test files in `tests/` mirroring the source structure. Write failing tests covering:
   - Happy path behavior
   - Edge cases mentioned in the feature's `expectedBehavior`
   - Error handling paths
   Run tests to confirm they fail: `source .venv/bin/activate && python -m pytest tests/ -x -v --tb=short`

3. **Implement (GREEN)**: Write the implementation code to make tests pass. Follow these patterns:
   - Use type hints on all functions and class methods
   - Use absolute imports: `from snipsnap.models import Transcription`
   - Provider pattern for transcription backends (abstract base class)
   - OpenAI SDK with `base_url="https://openrouter.ai/api/v1"` for LLM
   - Click for CLI commands
   - JSON file storage via the storage layer
   - All timestamps in seconds (float)

4. **Run all validators**:
   ```
   source .venv/bin/activate && python -m pytest tests/ -x -v --tb=short
   source .venv/bin/activate && python -m mypy snipsnap/ --ignore-missing-imports
   source .venv/bin/activate && python -m ruff check snipsnap/ tests/
   ```
   Fix any failures before proceeding.

5. **Manual verification**: Test the feature manually via CLI or Python REPL:
   - For CLI features: run the actual `snipsnap` commands with test data
   - For engine features: create a small test script that exercises the core logic
   - For export features: generate sample output and inspect the content
   - Record what you tested and what you observed

6. **Update shared state**: If you discover patterns, gotchas, or knowledge that future workers need, add it to the appropriate `.factory/library/` file.

## Example Handoff

```json
{
  "salientSummary": "Implemented transcription engine with provider abstraction and faster-whisper local provider. Wrote 12 tests covering video discovery, segment extraction, batch processing, and error handling. All pass. Manually verified transcription of a 5-second test video produces correct segment timestamps.",
  "whatWasImplemented": "TranscriptionProvider ABC in snipsnap/transcription/base.py with discover_videos(), transcribe(), and transcribe_batch() methods. WhisperLocalProvider in whisper_local.py using faster-whisper with configurable model size and int8 compute type. JSON storage for transcription results in storage.py. Video file discovery supporting .mp4, .mkv, .mov, .avi, .webm extensions.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "source .venv/bin/activate && python -m pytest tests/test_transcription.py -v", "exitCode": 0, "observation": "12 tests passed, including edge cases for empty folders and unsupported formats"},
      {"command": "source .venv/bin/activate && python -m mypy snipsnap/transcription/ --ignore-missing-imports", "exitCode": 0, "observation": "No type errors"},
      {"command": "source .venv/bin/activate && python -m ruff check snipsnap/transcription/", "exitCode": 0, "observation": "No lint issues"}
    ],
    "interactiveChecks": [
      {"action": "Created a 5-second test video with ffmpeg and ran WhisperLocalProvider.transcribe() on it", "observed": "Produced 1 segment with start=0.0, end=4.8, text containing expected speech. JSON file written correctly."},
      {"action": "Tested discover_videos() on a folder with mixed files", "observed": "Only .mp4 and .mkv files discovered, .txt and .jpg ignored as expected"}
    ]
  },
  "tests": {
    "added": [
      {"file": "tests/test_transcription.py", "cases": [
        {"name": "test_discover_videos_finds_supported_formats", "verifies": "Video discovery returns only supported extensions"},
        {"name": "test_discover_videos_empty_folder", "verifies": "Empty folder returns empty list"},
        {"name": "test_transcribe_produces_segments_with_timestamps", "verifies": "Transcription output has segments with start, end, text"},
        {"name": "test_segments_chronologically_ordered", "verifies": "Segments are sorted by start time"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Feature depends on a module or model that doesn't exist yet and isn't part of this feature
- The OpenRouter API returns unexpected errors that suggest a configuration problem
- faster-whisper fails to install or load models on this platform
- Test data (sample videos) cannot be generated with ffmpeg
- Requirements in the feature description are ambiguous or contradictory
