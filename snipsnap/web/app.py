"""FastAPI web dashboard backend for SnipSnap.

Exposes the same pipeline operations as the CLI as REST API endpoints:

    GET  /health                         – health check
    GET  /api/transcriptions             – list all transcriptions (metadata)
    GET  /api/transcriptions/{filename}  – get transcription with segments
    POST /api/transcribe                 – trigger transcription for a folder
    POST /api/curate                     – run curation with a prompt
    GET  /api/cutlists                   – list all cut lists (metadata)
    GET  /api/cutlists/{id}              – get full cut list with segments
    POST /api/export/{cutlist_id}        – export a cut list to EDL/FCPXML/DaVinci

All endpoints use the same storage layer and engine modules as the CLI.
Static files are served from the ``static/`` subdirectory of this package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from snipsnap.config import get_config
from snipsnap.curation.engine import (
    AuthenticationError,
    CurationEngine,
    CurationError,
    MissingApiKeyError,
    NetworkError,
    RateLimitError,
)
from snipsnap.export.davinci import generate_davinci_script
from snipsnap.export.edl import generate_edl
from snipsnap.export.fcpxml import generate_fcpxml
from snipsnap.storage import (
    load_all_cut_lists,
    load_all_transcriptions,
    load_cut_list,
    load_transcription,
    save_transcription,
    transcription_exists,
)
from snipsnap.transcription.base import SUPPORTED_EXTENSIONS
from snipsnap.transcription.whisper_local import WhisperLocalProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(title="SnipSnap", version="0.1.0")

# Mount static files from the package's static/ subdirectory
_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return HTTP 400 instead of the default 422 for request validation errors.

    FastAPI raises RequestValidationError for missing or invalid request fields.
    This handler normalizes those to HTTP 400 Bad Request so clients receive a
    consistent error status for all bad-input scenarios.

    Args:
        request: The incoming HTTP request.
        exc: The validation error raised by FastAPI/Pydantic.

    Returns:
        A JSON response with status 400 and the validation error details.
    """
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class SegmentOut(BaseModel):
    """A single transcribed segment."""

    start: float
    end: float
    text: str


class TranscriptionMeta(BaseModel):
    """Transcription summary returned in list responses."""

    source_file: str
    duration: float
    language: str
    model_used: str
    created_at: str
    segment_count: int


class TranscriptionDetail(BaseModel):
    """Full transcription with all segments."""

    source_file: str
    duration: float
    language: str
    model_used: str
    created_at: str
    segments: List[SegmentOut]


class TranscribeRequest(BaseModel):
    """Request body for POST /api/transcribe."""

    folder: str


class TranscribeResult(BaseModel):
    """Response body for POST /api/transcribe."""

    transcribed: int
    skipped: int
    failed: int
    transcriptions: List[TranscriptionMeta]


class CurateRequest(BaseModel):
    """Request body for POST /api/curate."""

    prompt: str
    model: Optional[str] = None


class CutSegmentOut(BaseModel):
    """A single segment from a cut list."""

    source_file: str
    start: float
    end: float
    description: str
    order: int


class CutListMeta(BaseModel):
    """Cut list summary returned in list responses."""

    id: str
    prompt: str
    theme: str
    created_at: str
    total_duration: float
    segment_count: int


class CutListDetail(BaseModel):
    """Full cut list with all segments."""

    id: str
    prompt: str
    theme: str
    created_at: str
    total_duration: float
    segments: List[CutSegmentOut]


class ExportRequest(BaseModel):
    """Request body for POST /api/export/{cutlist_id}."""

    format: str  # edl | fcpxml | davinci
    fps: Optional[float] = 24.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_EXPORT_FORMATS = {"edl", "fcpxml", "davinci"}


def _transcription_meta(t: object) -> TranscriptionMeta:
    """Convert a Transcription model instance to a TranscriptionMeta response."""
    from snipsnap.models import Transcription

    assert isinstance(t, Transcription)
    return TranscriptionMeta(
        source_file=t.source_file,
        duration=t.duration,
        language=t.language,
        model_used=t.model_used,
        created_at=t.created_at.isoformat(),
        segment_count=len(t.segments),
    )


def _transcription_detail(t: object) -> TranscriptionDetail:
    """Convert a Transcription model instance to a TranscriptionDetail response."""
    from snipsnap.models import Transcription

    assert isinstance(t, Transcription)
    return TranscriptionDetail(
        source_file=t.source_file,
        duration=t.duration,
        language=t.language,
        model_used=t.model_used,
        created_at=t.created_at.isoformat(),
        segments=[SegmentOut(start=s.start, end=s.end, text=s.text) for s in t.segments],
    )


def _cut_list_meta(cl: object) -> CutListMeta:
    """Convert a CutList model instance to a CutListMeta response."""
    from snipsnap.models import CutList

    assert isinstance(cl, CutList)
    return CutListMeta(
        id=cl.id,
        prompt=cl.prompt,
        theme=cl.theme,
        created_at=cl.created_at.isoformat(),
        total_duration=cl.total_duration,
        segment_count=len(cl.segments),
    )


def _cut_list_detail(cl: object) -> CutListDetail:
    """Convert a CutList model instance to a CutListDetail response."""
    from snipsnap.models import CutList

    assert isinstance(cl, CutList)
    return CutListDetail(
        id=cl.id,
        prompt=cl.prompt,
        theme=cl.theme,
        created_at=cl.created_at.isoformat(),
        total_duration=cl.total_duration,
        segments=[
            CutSegmentOut(
                source_file=s.source_file,
                start=s.start,
                end=s.end,
                description=s.description,
                order=s.order,
            )
            for s in cl.segments
        ],
    )


# ---------------------------------------------------------------------------
# Root — serve the SPA frontend
# ---------------------------------------------------------------------------


@app.get("/")
def serve_frontend() -> FileResponse:
    """Serve the single-page application frontend.

    Returns:
        The ``index.html`` file from the static directory.
    """
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Health check endpoint.

    Returns:
        ``{"status": "ok"}`` when the server is up.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Transcription endpoints
# ---------------------------------------------------------------------------


@app.get("/api/transcriptions", response_model=List[TranscriptionMeta])
def list_transcriptions() -> List[TranscriptionMeta]:
    """List all transcriptions with metadata.

    Returns:
        List of transcription metadata objects (no segments).
    """
    transcriptions = load_all_transcriptions()
    return [_transcription_meta(t) for t in transcriptions]


@app.get("/api/transcriptions/{filename:path}", response_model=TranscriptionDetail)
def get_transcription(filename: str) -> TranscriptionDetail:
    """Get a specific transcription including all its segments.

    The ``filename`` path parameter should be the ``source_file`` value
    from the list response.  Slashes in the path are supported.

    Args:
        filename: The ``source_file`` identifier of the transcription.

    Returns:
        Full transcription with all segments.

    Raises:
        HTTPException(404): If no transcription exists for *filename*.
    """
    t = load_transcription(filename)
    if t is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transcription not found: {filename}",
        )
    return _transcription_detail(t)


@app.post("/api/transcribe", response_model=TranscribeResult)
def transcribe(request: TranscribeRequest) -> TranscribeResult:
    """Trigger transcription for all video files in a folder.

    Discovers supported video files in the given folder, transcribes each
    one using the configured Whisper model, and saves results to storage.
    Already-transcribed files are skipped.

    Args:
        request: ``{folder: "/path/to/videos"}``

    Returns:
        Summary with counts and metadata for each transcription.

    Raises:
        HTTPException(400): If the folder does not exist or is not a directory.
        HTTPException(400): If no video files are found in the folder.
        HTTPException(500): If the transcription engine fails to load.
    """
    folder_path = Path(request.folder)

    if not folder_path.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {request.folder}")

    if not folder_path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Not a directory: {request.folder}"
        )

    # Discover video files
    videos: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        videos.extend(folder_path.rglob(f"*{ext}"))
    videos.sort()

    if not videos:
        raise HTTPException(
            status_code=400,
            detail=f"No video files found in: {request.folder}",
        )

    config = get_config()

    try:
        provider = WhisperLocalProvider(model_size=config.whisper_model)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load Whisper model: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load transcription model: {exc}",
        ) from exc

    transcribed_metas: list[TranscriptionMeta] = []
    transcribed = 0
    skipped = 0
    failed = 0

    for video_path in videos:
        if transcription_exists(str(video_path)):
            existing = load_transcription(str(video_path))
            if existing is not None:
                transcribed_metas.append(_transcription_meta(existing))
            skipped += 1
            continue

        try:
            result = provider.transcribe(video_path)
            save_transcription(result)
            transcribed_metas.append(_transcription_meta(result))
            transcribed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to transcribe %s: %s", video_path.name, exc)
            failed += 1

    return TranscribeResult(
        transcribed=transcribed,
        skipped=skipped,
        failed=failed,
        transcriptions=transcribed_metas,
    )


# ---------------------------------------------------------------------------
# Curation endpoint
# ---------------------------------------------------------------------------


@app.post("/api/curate", response_model=CutListDetail)
def curate(request: CurateRequest) -> CutListDetail:
    """Run LLM curation on all existing transcriptions.

    Loads all transcriptions from storage and passes them to the curation
    engine together with the provided prompt.  Returns the resulting cut list.

    Args:
        request: ``{prompt: "find funny moments", model?: "..."}``

    Returns:
        The newly generated cut list with all segments.

    Raises:
        HTTPException(400): If no transcriptions exist in storage.
        HTTPException(400): If the OpenRouter API key is not configured.
        HTTPException(500): If the curation engine raises an error.
    """
    transcriptions = load_all_transcriptions()
    if not transcriptions:
        raise HTTPException(
            status_code=400,
            detail=(
                "No transcriptions found. "
                "Run POST /api/transcribe first to generate transcriptions."
            ),
        )

    config = get_config()
    api_key = config.openrouter_api_key

    if not api_key or not api_key.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "OpenRouter API key is not configured. "
                "Set OPENROUTER_API_KEY in your .env file."
            ),
        )

    llm_model = request.model or config.model

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    engine = CurationEngine(client=client, api_key=api_key)

    try:
        cut_list = engine.curate(
            transcriptions=transcriptions,
            prompt=request.prompt,
            model=llm_model,
        )
    except MissingApiKeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=400,
            detail="Authentication failed: the OpenRouter API key is invalid.",
        ) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail="OpenRouter API rate limit exceeded. Please retry later.",
        ) from exc
    except NetworkError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Network error while contacting OpenRouter: {exc}",
        ) from exc
    except CurationError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Curation failed: {exc}",
        ) from exc
    except openai.APIError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LLM API error: {exc}",
        ) from exc

    return _cut_list_detail(cut_list)


# ---------------------------------------------------------------------------
# Cut list endpoints
# ---------------------------------------------------------------------------


@app.get("/api/cutlists", response_model=List[CutListMeta])
def list_cut_lists() -> List[CutListMeta]:
    """List all cut lists with metadata.

    Returns:
        List of cut list metadata objects (no segments).
    """
    cut_lists = load_all_cut_lists()
    return [_cut_list_meta(cl) for cl in cut_lists]


@app.get("/api/cutlists/{cutlist_id}", response_model=CutListDetail)
def get_cut_list(cutlist_id: str) -> CutListDetail:
    """Get a specific cut list including all its segments.

    Args:
        cutlist_id: UUID identifier of the cut list.

    Returns:
        Full cut list with all segments.

    Raises:
        HTTPException(404): If no cut list with the given ID exists.
    """
    cl = load_cut_list(cutlist_id)
    if cl is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cut list not found: {cutlist_id}",
        )
    return _cut_list_detail(cl)


# ---------------------------------------------------------------------------
# Export endpoint
# ---------------------------------------------------------------------------


@app.post("/api/export/{cutlist_id}")
def export_cut_list(cutlist_id: str, request: ExportRequest) -> Response:
    """Export a cut list to EDL, FCPXML, or DaVinci Resolve script.

    Args:
        cutlist_id: UUID identifier of the cut list.
        request: ``{format: "edl"|"fcpxml"|"davinci", fps?: 24.0}``

    Returns:
        File content with appropriate Content-Type and Content-Disposition headers.

    Raises:
        HTTPException(400): If the requested format is invalid.
        HTTPException(404): If no cut list with the given ID exists.
        HTTPException(500): If export generation fails.
    """
    fmt = request.format.lower().strip()
    if fmt not in _VALID_EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid export format: {request.format!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_EXPORT_FORMATS))}"
            ),
        )

    cl = load_cut_list(cutlist_id)
    if cl is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cut list not found: {cutlist_id}",
        )

    fps = request.fps if request.fps is not None else 24.0

    if fps <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid fps value: {fps!r}. fps must be a positive number greater than 0.",
        )

    try:
        if fmt == "edl":
            content = generate_edl(cl, frame_rate=fps)
            media_type = "text/plain"
            filename = f"{cutlist_id}.edl"
        elif fmt == "fcpxml":
            content = generate_fcpxml(cl, frame_rate=int(fps))
            media_type = "application/xml"
            filename = f"{cutlist_id}.fcpxml"
        else:  # davinci
            content = generate_davinci_script(cl, frame_rate=int(fps))
            media_type = "text/plain"
            filename = f"{cutlist_id}.py"
    except Exception as exc:  # noqa: BLE001
        logger.error("Export generation failed for %s (%s): %s", cutlist_id, fmt, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Export generation failed: {exc}",
        ) from exc

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
