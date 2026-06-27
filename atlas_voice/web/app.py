from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from atlas_voice.config import Settings
from atlas_voice.database import Database, row_to_dict
from atlas_voice.exporter import export_payload, export_recording
from atlas_voice.merge import format_seconds
from atlas_voice.pipeline import PipelineProcessor
from atlas_voice.storage import safe_filename


settings = Settings.from_env()
db = Database(settings.db_path)
processor = PipelineProcessor(settings, db)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_directories()
    db.initialize()
    yield


app = FastAPI(
    title="Atlas Voice",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)

templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(templates_dir))
templates.env.filters["timecode"] = format_seconds

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> Response:
    recordings = [dict(row) for row in db.list_recordings()]
    return templates.TemplateResponse(
        request,
        "index.html",
        {"recordings": recordings},
    )


@app.post("/upload")
def upload_audio(file: UploadFile = File(...)) -> Response:
    settings.ensure_directories()
    upload_dir = settings.inbox_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(file.filename or "upload")
    destination = upload_dir / f"{uuid.uuid4().hex}-{filename}"
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    recording_id = processor.enqueue_source(destination)
    return RedirectResponse(f"/recordings/{recording_id}", status_code=303)


@app.get("/recordings/{recording_id}", response_class=HTMLResponse)
def recording_detail(request: Request, recording_id: str) -> Response:
    recording = row_to_dict(db.get_recording(recording_id))
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return templates.TemplateResponse(
        request,
        "recording.html",
        {
            "recording": recording,
            "jobs": [dict(row) for row in db.jobs_for_recording(recording_id)],
            "segments": db.get_segments(recording_id),
            "summary": db.get_summary(recording_id),
        },
    )


@app.post("/recordings/{recording_id}/retry")
def retry_recording(recording_id: str) -> Response:
    if db.get_recording(recording_id) is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    db.retry_recording(recording_id)
    return RedirectResponse(f"/recordings/{recording_id}", status_code=303)


@app.get("/search", response_class=HTMLResponse)
def search(request: Request, q: str = "") -> Response:
    results = db.search(q) if q.strip() else []
    return templates.TemplateResponse(
        request,
        "search.html",
        {"query": q, "results": results},
    )


@app.get("/api/recordings/{recording_id}")
def api_recording(recording_id: str) -> JSONResponse:
    try:
        return JSONResponse(export_payload(db, recording_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/recordings/{recording_id}/export")
def api_export(recording_id: str, format: str = "json") -> Response:
    try:
        content = export_recording(db, recording_id, format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    media_type = {
        "json": "application/json",
        "md": "text/markdown; charset=utf-8",
        "txt": "text/plain; charset=utf-8",
    }[format]
    return Response(content=content, media_type=media_type)


@app.get("/media/recordings/{recording_id}/normalized.wav")
def normalized_audio(recording_id: str) -> FileResponse:
    recording = db.get_recording(recording_id)
    if recording is None or not recording["normalized_path"]:
        raise HTTPException(status_code=404, detail="Audio not found")
    path = Path(recording["normalized_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav", filename=f"{recording_id}.wav")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
