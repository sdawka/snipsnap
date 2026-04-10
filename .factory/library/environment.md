# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Required Environment Variables

- `OPENROUTER_API_KEY` - API key for OpenRouter (used for LLM curation via OpenAI SDK)
- `SNIPSNAP_DATA_DIR` - Optional: override default data directory (default: `./snipsnap_data/`)
- `SNIPSNAP_MODEL` - Optional: override default LLM model (default: `google/gemini-2.5-flash-lite`)
- `SNIPSNAP_WHISPER_MODEL` - Optional: override Whisper model size (default: `small`)

## Platform Notes

- macOS Apple Silicon (arm64), Python 3.14
- ffmpeg 8.1 installed via Homebrew at /opt/homebrew/bin/ffmpeg
- faster-whisper runs on CPU with int8 quantization (no CUDA on macOS)
- Whisper model files are downloaded on first use (~500MB for small model)
