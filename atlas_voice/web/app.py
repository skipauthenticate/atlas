from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from atlas_voice.assistant_config import load_assistant_config
from atlas_voice.anythingllm import (
    AnythingLLMConfigError,
    AnythingLLMError,
    sync_recording_to_anythingllm,
)
from atlas_voice.config import Settings, dotenv_path
from atlas_voice.database import Database, row_to_dict
from atlas_voice.exporter import export_payload, export_recording
from atlas_voice.merge import format_seconds
from atlas_voice.pipeline import PipelineProcessor
from atlas_voice.privacy import privacy_summary
from atlas_voice.realtime import (
    REALTIME_SYSTEM_PROMPT,
    audio_delta_payload,
    chunk_text,
    decode_audio_delta,
    extract_text_input,
    generate_realtime_reply,
    is_tts_sidecar_provider,
    normalize_tts_provider,
    synthesize_with_piper,
    synthesize_with_tts_sidecar,
    transcribe_realtime_audio,
)
from atlas_voice.status import assistant_health as collect_assistant_health
from atlas_voice.status import runtime_status as collect_runtime_status
from atlas_voice.storage import safe_filename
from atlas_voice.summarizer import get_template, list_templates, summary_to_sections
from atlas_voice.turn_state import RealtimeTurnState


settings = Settings.from_env()
assistant_config = load_assistant_config(settings.assistant_config_path)
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

ASR_PROVIDERS = ("whisperx", "parakeet", "canary", "vibevoice")
DIARIZATION_PROVIDERS = ("pyannote", "transcript", "none")
PROVIDER_DEFAULT_MODELS = {
    "whisperx": "large-v3-turbo",
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "canary": "nvidia/canary-1b-v2",
    "vibevoice": "microsoft/VibeVoice-ASR",
}
COMMON_ASR_MODELS = (
    "large-v3-turbo",
    "large-v3",
    "medium",
    "small",
    "base",
    "tiny.en",
    "nvidia/parakeet-tdt-0.6b-v3",
    "nvidia/canary-1b-v2",
    "microsoft/VibeVoice-ASR",
)
RUNTIME_ENV_KEYS = {
    "ATLAS_VOICE_ASR_PROVIDER",
    "ATLAS_VOICE_ASR_MODEL",
    "ATLAS_VOICE_DIARIZATION_PROVIDER",
    "ATLAS_VOICE_VIBEVOICE_MODEL",
    "WHISPERX_MODEL",
}


def runtime_info(current_settings: Settings) -> dict[str, Any]:
    device = current_settings.whisperx_device
    accelerator = "GPU" if device != "cpu" else "CPU"
    return {
        "accelerator": accelerator,
        "device": device,
        "asr_provider": current_settings.asr_provider,
        "asr_model": _effective_asr_model(current_settings),
        "asr_model_value": _model_form_value(current_settings),
        "asr_providers": ASR_PROVIDERS,
        "asr_model_options": COMMON_ASR_MODELS,
        "whisperx_model": current_settings.whisperx_model,
        "compute_type": current_settings.whisperx_compute_type,
        "diarization_provider": current_settings.diarization_provider,
        "diarization_providers": DIARIZATION_PROVIDERS,
        "pyannote_model": current_settings.pyannote_model,
    }


def _effective_asr_model(current_settings: Settings) -> str:
    provider = current_settings.asr_provider
    if provider == "whisperx":
        return current_settings.whisperx_model
    if current_settings.asr_model:
        return current_settings.asr_model
    if provider == "vibevoice":
        return current_settings.vibevoice_model
    return PROVIDER_DEFAULT_MODELS.get(provider, current_settings.whisperx_model)


def _model_form_value(current_settings: Settings) -> str:
    return _effective_asr_model(current_settings)


def _runtime_env_values(provider: str, model: str, diarization: str) -> dict[str, str]:
    values = {
        "ATLAS_VOICE_ASR_PROVIDER": provider,
        "ATLAS_VOICE_DIARIZATION_PROVIDER": diarization,
    }
    if provider == "whisperx":
        values["WHISPERX_MODEL"] = model or PROVIDER_DEFAULT_MODELS["whisperx"]
        values["ATLAS_VOICE_ASR_MODEL"] = ""
    elif provider == "vibevoice":
        values["ATLAS_VOICE_VIBEVOICE_MODEL"] = model or PROVIDER_DEFAULT_MODELS["vibevoice"]
        values["ATLAS_VOICE_ASR_MODEL"] = ""
    else:
        default_model = PROVIDER_DEFAULT_MODELS[provider]
        values["ATLAS_VOICE_ASR_MODEL"] = "" if model in {"", default_model} else model
    return values


def _write_dotenv_values(values: dict[str, str]) -> None:
    unsafe = set(values) - RUNTIME_ENV_KEYS
    if unsafe:
        raise ValueError(f"Unsupported runtime setting keys: {sorted(unsafe)}")

    path = dotenv_path()
    lines = path.read_text().splitlines() if path.exists() else []
    remaining = dict(values)
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            updated.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            updated.append(f"{key}={_clean_env_value(remaining.pop(key))}")
        else:
            updated.append(line)

    if remaining and updated and updated[-1].strip():
        updated.append("")
    for key, value in remaining.items():
        updated.append(f"{key}={_clean_env_value(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(updated).rstrip() + "\n")


def _clean_env_value(value: str) -> str:
    return value.replace("\n", " ").replace("\r", " ").strip()


def _refresh_runtime_settings() -> None:
    global settings, assistant_config, processor
    settings = Settings.from_env()
    assistant_config = load_assistant_config(settings.assistant_config_path)
    settings.ensure_directories()
    processor = PipelineProcessor(settings, db)


def _restart_worker_if_idle() -> str:
    if _running_job_count() > 0:
        return "pending"
    if shutil.which("systemctl") is None:
        return "manual"
    try:
        result = subprocess.run(
            ["systemctl", "--user", "restart", "atlas-voice-worker.service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except Exception:
        return "manual"
    return "restarted" if result.returncode == 0 else "manual"


def _running_job_count() -> int:
    with db.connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM jobs WHERE status = 'running'").fetchone()
    return int(row["count"] if row else 0)


def _runtime_settings_message(status: str | None, worker: str | None) -> str | None:
    if status != "saved":
        return None
    if worker == "restarted":
        return "Runtime settings saved. Worker restarted."
    if worker == "pending":
        return "Runtime settings saved. Worker restart deferred because a job is running."
    return "Runtime settings saved. Restart the worker to apply them."


def _anythingllm_message(status: str | None, error: str | None) -> dict[str, str] | None:
    if status == "synced":
        return {"kind": "ok", "text": "Synced to AnythingLLM."}
    if status == "not_configured":
        return {
            "kind": "error",
            "text": "AnythingLLM sync is not configured. Set the API key and workspace slug.",
        }
    if status == "error":
        safe_error = (error or "request failed").replace("\n", " ").replace("\r", " ")[:240]
        return {"kind": "error", "text": f"AnythingLLM sync failed: {safe_error}"}
    return None


def job_views(rows: list[Any]) -> list[dict[str, Any]]:
    return [_job_view(dict(row)) for row in rows]


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    started_at = _parse_timestamp(job.get("started_at"))
    finished_at = _parse_timestamp(job.get("finished_at"))
    duration_seconds: float | None = None
    if started_at and finished_at:
        duration_seconds = (finished_at - started_at).total_seconds()
    elif started_at and job.get("status") == "running":
        duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()

    if duration_seconds is None:
        duration_label = "not started"
    elif job.get("status") == "running" and not finished_at:
        duration_label = f"elapsed {format_seconds(duration_seconds)}"
    else:
        duration_label = format_seconds(duration_seconds)

    return {
        **job,
        "duration_seconds": duration_seconds,
        "duration_label": duration_label,
        "started_at_display": _timestamp_label(started_at),
        "finished_at_display": _timestamp_label(finished_at),
    }


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_label(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")


@app.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request, runtime_settings: str | None = None, worker: str | None = None
) -> Response:
    recordings = [dict(row) for row in db.list_recordings()]
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "recordings": recordings,
            "runtime": runtime_info(settings),
            "status": collect_runtime_status(settings, db, assistant_config),
            "runtime_settings_message": _runtime_settings_message(runtime_settings, worker),
        },
    )


@app.get("/voice", response_class=HTMLResponse)
def voice_console(request: Request) -> Response:
    status = collect_runtime_status(settings, db, assistant_config)
    sessions = _voice_session_views(db.list_ambient_sessions(limit=8, mode="direct_voice"))
    ambient_timeline = _ambient_timeline_views(db.list_ambient_sessions(limit=12, mode=None))
    return templates.TemplateResponse(
        request,
        "voice.html",
        {
            "runtime": runtime_info(settings),
            "status": status,
            "assistant_enabled": settings.assistant_enabled,
            "sessions": sessions,
            "ambient_timeline": ambient_timeline,
            "voice_settings": _voice_settings_view(),
        },
    )


def _voice_settings_view() -> dict[str, Any]:
    tts_provider = _realtime_tts_provider()
    return {
        "realtime_host": f"{settings.host}:{settings.port}",
        "websocket_path": "/v1/realtime",
        "sample_rate": settings.realtime_audio_sample_rate,
        "channels": settings.realtime_audio_channels,
        "tts_provider": tts_provider,
        "tts_model": _realtime_tts_model(tts_provider) if tts_provider != "none" else "none",
        "tts_base_url": settings.tts_base_url if is_tts_sidecar_provider(tts_provider) else None,
        "llm_model": settings.llm_model,
    }


def _voice_session_views(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for session in sessions:
        utterances = db.list_utterances(str(session["id"]))
        turns = db.list_assistant_turns(str(session["id"]))
        views.append(
            {
                **session,
                "started_at_display": _timestamp_label(_parse_timestamp(session.get("started_at"))),
                "last_utterance": utterances[-1]["text"] if utterances else "",
                "last_reply": turns[-1]["text"] if turns else "",
            }
        )
    return views


def _ambient_timeline_views(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for session in sessions:
        if session.get("mode") == "direct_voice":
            continue
        utterances = db.list_utterances(str(session["id"]))
        last_utterance = utterances[-1] if utterances else None
        views.append(
            {
                **session,
                "started_at_display": _timestamp_label(_parse_timestamp(session.get("started_at"))),
                "last_utterance": last_utterance["text"] if last_utterance else "",
                "last_speaker": last_utterance["speaker"] if last_utterance else "",
            }
        )
        if len(views) >= 8:
            break
    return views


@app.post("/api/voice/playground/model")
def api_voice_playground_model(payload: dict[str, Any]) -> JSONResponse:
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if not settings.assistant_enabled:
        raise HTTPException(status_code=403, detail="Assistant runtime is disabled")

    settings.ensure_directories()
    db.initialize()
    provider = "stub" if settings.stub_mode else "openai-compatible"
    try:
        reply = generate_realtime_reply(
            text,
            settings,
            instructions=_realtime_instructions(),
        )
    except Exception as exc:  # noqa: BLE001 - playground errors should surface cleanly.
        db.log_model_run(
            provider=provider,
            model=settings.llm_model,
            task="voice_playground_model",
            input_ref="voice_playground:text",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=502, detail=_safe_realtime_error(exc)) from exc

    db.log_model_run(
        provider=provider,
        model=settings.llm_model,
        task="voice_playground_model",
        input_ref="voice_playground:text",
        latency_ms=reply.latency_ms,
        tokens_in=reply.tokens_in,
        tokens_out=reply.tokens_out,
    )
    return JSONResponse(
        {
            "status": "ok",
            "provider": provider,
            "model": settings.llm_model,
            "text": reply.text,
            "latency_ms": reply.latency_ms,
            "tokens_in": reply.tokens_in,
            "tokens_out": reply.tokens_out,
        }
    )


@app.post("/api/voice/playground/stt")
async def api_voice_playground_stt(file: UploadFile = File(...)) -> JSONResponse:
    if not settings.assistant_enabled:
        raise HTTPException(status_code=403, detail="Assistant runtime is disabled")
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Audio file is required")

    settings.ensure_directories()
    db.initialize()
    output_dir = settings.artifacts_dir / "voice-playground"
    provider = settings.asr_provider
    model = _effective_asr_model(settings)
    started = time.perf_counter()
    try:
        text, audio_path = transcribe_realtime_audio(audio, settings, output_dir)
    except Exception as exc:  # noqa: BLE001 - playground errors should surface cleanly.
        latency_ms = int((time.perf_counter() - started) * 1000)
        db.log_model_run(
            provider=provider,
            model=model,
            task="voice_playground_stt",
            input_ref=file.filename or "voice_playground:audio",
            latency_ms=latency_ms,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=502, detail=_safe_realtime_error(exc)) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    db.log_model_run(
        provider=provider,
        model=model,
        task="voice_playground_stt",
        input_ref=file.filename or "voice_playground:audio",
        output_ref=str(audio_path),
        latency_ms=latency_ms,
    )
    return JSONResponse(
        {
            "status": "ok",
            "provider": provider,
            "model": model,
            "text": text,
            "audio_path": str(audio_path),
            "latency_ms": latency_ms,
        }
    )


@app.post("/api/voice/playground/tts")
def api_voice_playground_tts(payload: dict[str, Any]) -> JSONResponse:
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if not settings.assistant_enabled:
        raise HTTPException(status_code=403, detail="Assistant runtime is disabled")

    tts_provider = _realtime_tts_provider()
    if not _tts_outputs_audio(tts_provider):
        raise HTTPException(status_code=400, detail="TTS provider is disabled")

    output_dir = settings.artifacts_dir / "voice-playground"
    settings.ensure_directories()
    db.initialize()
    model = _realtime_tts_model(tts_provider)
    try:
        if tts_provider == "piper":
            audio = synthesize_with_piper(text, settings, output_dir)
        else:
            audio = synthesize_with_tts_sidecar(text, settings, output_dir)
    except Exception as exc:  # noqa: BLE001 - playground errors should surface cleanly.
        db.log_model_run(
            provider=tts_provider,
            model=model,
            task="voice_playground_tts",
            input_ref="voice_playground:text",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise HTTPException(status_code=502, detail=_safe_realtime_error(exc)) from exc

    db.log_model_run(
        provider=tts_provider,
        model=model,
        task="voice_playground_tts",
        input_ref="voice_playground:text",
        output_ref=str(audio.path),
        latency_ms=audio.latency_ms,
    )
    return JSONResponse(
        {
            "status": "ok",
            "provider": tts_provider,
            "model": model,
            "audio_path": str(audio.path),
            "media_type": audio.media_type,
            "latency_ms": audio.latency_ms,
        }
    )


@app.post("/settings/runtime")
def update_runtime_settings(
    asr_provider: str = Form(...),
    asr_model: str = Form(""),
    diarization_provider: str = Form(...),
) -> Response:
    provider = asr_provider.strip().lower()
    diarization = diarization_provider.strip().lower()
    if provider not in ASR_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown ASR provider")
    if diarization not in DIARIZATION_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown diarization provider")

    model = asr_model.strip()
    if provider != settings.asr_provider and model == _effective_asr_model(settings):
        model = ""
    provider_changed = provider != settings.asr_provider
    if (
        provider == "vibevoice"
        and provider_changed
        and diarization == settings.diarization_provider
    ):
        diarization = "transcript"
    if (
        provider != "vibevoice"
        and provider_changed
        and diarization == "transcript"
        and diarization == settings.diarization_provider
    ):
        diarization = "pyannote"

    values = _runtime_env_values(provider, model, diarization)
    _write_dotenv_values(values)
    os.environ.update(values)
    _refresh_runtime_settings()
    worker_state = _restart_worker_if_idle()
    return RedirectResponse(
        f"/?runtime_settings=saved&worker={worker_state}", status_code=303
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
def recording_detail(
    request: Request,
    recording_id: str,
    anythingllm: str | None = None,
    anythingllm_error: str | None = None,
) -> Response:
    recording = row_to_dict(db.get_recording(recording_id))
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    summary = db.get_summary(recording_id)
    preferred_template_id = db.get_recording_template(recording_id)
    summary_template_id = summary.get("template_id") if summary else None
    selected_template_id = preferred_template_id or summary_template_id or "meeting"
    preferred_template = get_template(selected_template_id) or get_template("meeting")

    # Check if a summary is being regenerated (old template in use, new template set)
    # If so, keep the cached summary but note the pending change
    cached_summary = summary  # existing summary (may be None)
    is_resummarizing = (
        summary is not None
        and preferred_template_id
        and preferred_template is not None
        and summary.get("template_id") != preferred_template_id
    )

    return templates.TemplateResponse(
        request,
        "recording.html",
        {
            "recording": recording,
            "runtime": runtime_info(settings),
            "jobs": job_views(db.jobs_for_recording(recording_id)),
            "segments": db.get_segments(recording_id),
            "summary": summary,
            "summary_sections": summary_to_sections(summary["text"]) if summary else [],
            "cached_summary": cached_summary,
            "cached_summary_sections": (
                summary_to_sections(cached_summary["text"]) if cached_summary else []
            ),
            "is_resummarizing": is_resummarizing,
            "all_templates": {tid: tpl.to_dict() for tid, tpl in list_templates().items()},
            "preferred_template_id": selected_template_id,
            "preferred_template": preferred_template.to_dict() if preferred_template else None,
            "anythingllm_workspace_slug": settings.anythingllm_workspace_slug,
            "anythingllm_message": _anythingllm_message(anythingllm, anythingllm_error),
        },
    )


@app.post("/recordings/{recording_id}/sync-anythingllm")
def sync_anythingllm(recording_id: str) -> Response:
    if db.get_recording(recording_id) is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    try:
        sync_recording_to_anythingllm(db, recording_id, settings)
    except AnythingLLMConfigError:
        return RedirectResponse(
            f"/recordings/{recording_id}?anythingllm=not_configured",
            status_code=303,
        )
    except (ValueError, AnythingLLMError) as exc:
        message = quote(str(exc)[:240])
        return RedirectResponse(
            f"/recordings/{recording_id}?anythingllm=error&anythingllm_error={message}",
            status_code=303,
        )
    return RedirectResponse(f"/recordings/{recording_id}?anythingllm=synced", status_code=303)


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
        {"query": q, "results": results, "runtime": runtime_info(settings)},
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


@app.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    if not settings.assistant_enabled:
        await _send_realtime_error(
            websocket,
            "Realtime assistant is disabled; set ATLAS_ASSISTANT_ENABLED=true to enable /v1/realtime",
        )
        await websocket.close(code=1008)
        return
    settings.ensure_directories()
    db.initialize()
    session_id = db.create_ambient_session(
        mode="direct_voice",
        source="websocket",
        title="Realtime voice session",
    )
    instructions = _realtime_instructions()
    tts_provider = _realtime_tts_provider()
    state = RealtimeTurnState(
        sample_rate=settings.realtime_audio_sample_rate,
        channels=settings.realtime_audio_channels,
    )
    active_response_task: asyncio.Task[None] | None = None

    await _send_realtime_event(
        websocket,
        "session.created",
        session={
            "id": session_id,
            "object": "realtime.session",
            "model": settings.llm_model,
            "modalities": ["text", "audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "wav" if _tts_outputs_audio(tts_provider) else "none",
        },
    )

    async def cancel_active_response(reason: str) -> bool:
        nonlocal active_response_task
        if not active_response_task or active_response_task.done():
            return False
        active_response_task.cancel()
        with suppress(asyncio.CancelledError):
            await active_response_task
        active_response_task = None
        await _handle_realtime_response_cancel(
            websocket,
            state=state,
            event={"reason": reason},
        )
        return True

    try:
        while True:
            event = await websocket.receive_json()
            event_type = str(event.get("type") or "")
            if event_type == "session.update":
                instructions = _updated_realtime_instructions(event, instructions)
                state.update_audio_format(event)
                await _send_realtime_event(
                    websocket,
                    "session.updated",
                    session={"id": session_id, "instructions": instructions},
                )
                continue

            if event_type == "input_audio_buffer.append":
                try:
                    buffered_bytes = state.append_audio(decode_audio_delta(event))
                except ValueError as exc:
                    await _send_realtime_error(websocket, str(exc), event_type=event_type)
                    continue
                await _send_realtime_event(
                    websocket,
                    "input_audio_buffer.appended",
                    byte_count=buffered_bytes,
                )
                continue

            if event_type == "input_audio_buffer.clear":
                state.clear_audio()
                await _send_realtime_event(websocket, "input_audio_buffer.cleared")
                continue

            if event_type in {"response.cancel", "response.interrupt", "input_audio_buffer.interrupt"}:
                if active_response_task and not active_response_task.done():
                    await cancel_active_response(str(event.get("reason") or "client_cancelled"))
                else:
                    await _handle_realtime_interrupt(websocket, state=state, event=event)
                continue

            if event_type == "input_audio_buffer.commit":
                committed = state.commit_audio()
                await _send_realtime_event(
                    websocket,
                    "input_audio_buffer.committed",
                    byte_count=len(committed),
                )
                text = extract_text_input(event)
                if not text and committed:
                    try:
                        text, _audio_path = await asyncio.to_thread(
                            transcribe_realtime_audio,
                            committed,
                            settings,
                            _realtime_artifact_dir(session_id),
                            sample_rate=state.sample_rate,
                            channels=state.channels,
                        )
                    except Exception as exc:  # noqa: BLE001 - send realtime errors to client.
                        await _send_realtime_error(
                            websocket,
                            f"Audio transcription failed: {_safe_realtime_error(exc)}",
                            event_type=event_type,
                        )
                        continue
                if not text:
                    await _send_realtime_error(
                        websocket,
                        "input_audio_buffer.commit requires audio or transcript text",
                        event_type=event_type,
                    )
                    continue
                await cancel_active_response("barge_in")
                active_response_task = asyncio.create_task(
                    _handle_realtime_user_text(
                        websocket,
                        session_id=session_id,
                        text=text,
                        source_provider=settings.asr_provider if committed else "text",
                        transcript_prefix="conversation.item.input_audio_transcription",
                        instructions=instructions,
                        tts_provider=tts_provider,
                        state=state,
                    )
                )
                continue

            if event_type in {"input_text", "input.text", "message"}:
                text = extract_text_input(event)
                if not text:
                    await _send_realtime_error(
                        websocket,
                        "text event requires a non-empty text field",
                        event_type=event_type,
                    )
                    continue
                await cancel_active_response("barge_in")
                active_response_task = asyncio.create_task(
                    _handle_realtime_user_text(
                        websocket,
                        session_id=session_id,
                        text=text,
                        source_provider="text",
                        transcript_prefix="conversation.item.input_text",
                        instructions=instructions,
                        tts_provider=tts_provider,
                        state=state,
                    )
                )
                continue

            if event_type == "conversation.item.create":
                state.set_pending_text(extract_text_input(event))
                await _send_realtime_event(
                    websocket,
                    "conversation.item.created",
                    item={"id": f"item_{uuid.uuid4().hex}", "type": "message"},
                )
                continue

            if event_type == "response.create":
                text = state.pop_pending_text()
                if not text:
                    await _send_realtime_error(
                        websocket,
                        "response.create has no pending user text",
                        event_type=event_type,
                    )
                    continue
                await cancel_active_response("barge_in")
                active_response_task = asyncio.create_task(
                    _handle_realtime_user_text(
                        websocket,
                        session_id=session_id,
                        text=text,
                        source_provider="text",
                        transcript_prefix="conversation.item.input_text",
                        instructions=instructions,
                        tts_provider=tts_provider,
                        state=state,
                    )
                )
                continue

            await _send_realtime_error(
                websocket,
                f"Unsupported realtime event type: {event_type or '<missing>'}",
                event_type=event_type or None,
            )
    except WebSocketDisconnect:
        pass
    finally:
        if active_response_task and not active_response_task.done():
            active_response_task.cancel()
            with suppress(asyncio.CancelledError):
                await active_response_task
        db.end_ambient_session(session_id)


async def _handle_realtime_response_cancel(
    websocket: WebSocket,
    *,
    state: RealtimeTurnState,
    event: dict[str, Any],
) -> None:
    response_id = state.active_response_id
    state.cancel_response()
    response: dict[str, Any] = {"status": "cancelled"}
    if response_id:
        response["id"] = response_id
    state.complete_response()
    reason = str(event.get("reason") or "client_cancelled").strip() or "client_cancelled"
    await _send_realtime_event(
        websocket,
        "response.cancelled",
        response=response,
        reason=reason[:120],
    )


async def _handle_realtime_interrupt(
    websocket: WebSocket,
    *,
    state: RealtimeTurnState,
    event: dict[str, Any],
) -> None:
    response_id = state.active_response_id
    response_was_active = state.response_in_progress
    cleared_audio_bytes = state.buffered_audio_bytes
    cleared_pending_text = state.pop_pending_text() is not None
    state.clear_audio()
    state.cancel_response()
    if response_was_active:
        state.complete_response()

    reason = str(event.get("reason") or "client_interrupt").strip() or "client_interrupt"
    response: dict[str, Any] = {"status": "interrupted"}
    if response_id:
        response["id"] = response_id
    await _send_realtime_event(
        websocket,
        "response.interrupted",
        response=response,
        reason=reason[:120],
        cleared_audio_bytes=cleared_audio_bytes,
        cleared_pending_text=cleared_pending_text,
        active_response=response_was_active,
    )


async def _handle_realtime_user_text(
    websocket: WebSocket,
    *,
    session_id: str,
    text: str,
    source_provider: str,
    transcript_prefix: str,
    instructions: str,
    tts_provider: str,
    state: RealtimeTurnState,
) -> None:
    utterance_id = db.add_utterance(
        session_id=session_id,
        text=text,
        source_provider=source_provider,
    )
    item_id = f"utt_{utterance_id}"
    await _send_realtime_event(
        websocket,
        f"{transcript_prefix}.delta",
        item_id=item_id,
        delta=text,
    )
    await _send_realtime_event(
        websocket,
        f"{transcript_prefix}.done",
        item_id=item_id,
        text=text,
    )

    response_id = f"resp_{uuid.uuid4().hex}"
    state.begin_response(response_id)
    await _send_realtime_event(
        websocket,
        "response.created",
        response={"id": response_id, "status": "in_progress"},
    )

    provider = "stub" if settings.stub_mode else "openai-compatible"
    try:
        reply = await asyncio.to_thread(
            generate_realtime_reply,
            text,
            settings,
            instructions=instructions,
        )
    except Exception as exc:  # noqa: BLE001 - send realtime errors to client.
        db.log_model_run(
            provider=provider,
            model=settings.llm_model,
            task="realtime_chat",
            input_ref=f"utterance:{utterance_id}",
            error=f"{type(exc).__name__}: {exc}",
        )
        await _send_realtime_event(
            websocket,
            "response.failed",
            response={"id": response_id, "status": "failed"},
            error={"message": _safe_realtime_error(exc)},
        )
        state.complete_response()
        return

    for delta in chunk_text(reply.text):
        await _send_realtime_event(
            websocket,
            "response.text.delta",
            response_id=response_id,
            delta=delta,
        )
    await _send_realtime_event(
        websocket,
        "response.text.done",
        response_id=response_id,
        text=reply.text,
    )

    audio_path: str | None = None
    if _tts_outputs_audio(tts_provider):
        try:
            if tts_provider == "piper":
                audio = await asyncio.to_thread(
                    synthesize_with_piper,
                    reply.text,
                    settings,
                    _realtime_artifact_dir(session_id),
                )
            else:
                audio = await asyncio.to_thread(
                    synthesize_with_tts_sidecar,
                    reply.text,
                    settings,
                    _realtime_artifact_dir(session_id),
                )
            audio_path = str(audio.path)
            db.log_model_run(
                provider=tts_provider,
                model=_realtime_tts_model(tts_provider),
                task="realtime_tts",
                input_ref=f"response:{response_id}",
                output_ref=audio_path,
                latency_ms=audio.latency_ms,
            )
            await _send_realtime_event(
                websocket,
                "response.audio.delta",
                response_id=response_id,
                delta=audio_delta_payload(audio.payload),
                media_type=audio.media_type,
            )
            await _send_realtime_event(
                websocket,
                "response.audio.done",
                response_id=response_id,
                audio_path=audio_path,
            )
        except Exception as exc:  # noqa: BLE001 - TTS failure should not drop text response.
            db.log_model_run(
                provider=tts_provider,
                model=_realtime_tts_model(tts_provider),
                task="realtime_tts",
                input_ref=f"response:{response_id}",
                error=f"{type(exc).__name__}: {exc}",
            )
            await _send_realtime_event(
                websocket,
                "response.audio.failed",
                response_id=response_id,
                error={"message": _safe_realtime_error(exc)},
            )
    else:
        await _send_realtime_event(
            websocket,
            "response.audio.done",
            response_id=response_id,
            status="skipped",
        )

    turn_id = db.add_assistant_turn(
        session_id=session_id,
        user_utterance_id=utterance_id,
        text=reply.text,
        audio_path=audio_path,
        model=settings.llm_model,
        latency_ms=reply.latency_ms,
    )
    db.log_model_run(
        provider=provider,
        model=settings.llm_model,
        task="realtime_chat",
        input_ref=f"utterance:{utterance_id}",
        output_ref=f"assistant_turn:{turn_id}",
        latency_ms=reply.latency_ms,
        tokens_in=reply.tokens_in,
        tokens_out=reply.tokens_out,
    )
    await _send_realtime_event(
        websocket,
        "response.done",
        response={
            "id": response_id,
            "status": "completed",
            "output": [{"type": "message", "text": reply.text}],
            "turn_id": turn_id,
        },
    )
    state.complete_response()


async def _send_realtime_event(websocket: WebSocket, event_type: str, **payload: Any) -> None:
    await websocket.send_json({"event_id": f"evt_{uuid.uuid4().hex}", "type": event_type, **payload})


async def _send_realtime_error(
    websocket: WebSocket,
    message: str,
    *,
    event_type: str | None = None,
) -> None:
    payload: dict[str, Any] = {"error": {"message": message}}
    if event_type:
        payload["error"]["event_type"] = event_type
    await _send_realtime_event(websocket, "error", **payload)


def _realtime_instructions() -> str:
    profile = assistant_config.profiles.get("direct_voice", {})
    instructions = profile.get("instructions") if isinstance(profile, dict) else None
    return str(instructions or REALTIME_SYSTEM_PROMPT)


def _updated_realtime_instructions(event: dict[str, Any], current: str) -> str:
    session = event.get("session")
    if isinstance(session, dict) and session.get("instructions"):
        return str(session["instructions"])
    if event.get("instructions"):
        return str(event["instructions"])
    return current


def _realtime_tts_provider() -> str:
    provider = normalize_tts_provider(settings.tts_provider)
    if provider != "none":
        return provider
    profile = assistant_config.profiles.get("direct_voice", {})
    if isinstance(profile, dict) and profile.get("enabled") is True:
        return normalize_tts_provider(str(profile.get("tts_provider") or "none"))
    return "none"


def _tts_outputs_audio(tts_provider: str) -> bool:
    return tts_provider == "piper" or is_tts_sidecar_provider(tts_provider)


def _realtime_tts_model(tts_provider: str) -> str:
    if tts_provider == "piper":
        return settings.piper_voice or "piper"
    if is_tts_sidecar_provider(tts_provider):
        return settings.tts_model
    return tts_provider


def _realtime_artifact_dir(session_id: str) -> Path:
    return settings.artifacts_dir / "realtime" / session_id


def _safe_realtime_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ").replace("\r", " ")[:500]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse(collect_runtime_status(settings, db, assistant_config))


@app.get("/api/assistant/health")
def api_assistant_health() -> JSONResponse:
    return JSONResponse(collect_assistant_health(settings, db, assistant_config))


@app.get("/api/assistant/sessions")
def api_assistant_sessions(
    limit: int = 20,
    mode: str = "direct_voice",
    status: str | None = None,
) -> JSONResponse:
    safe_limit = min(max(limit, 1), 100)
    normalized_mode = mode.strip().lower() if mode else "direct_voice"
    session_mode = None if normalized_mode == "all" else normalized_mode
    sessions = db.list_ambient_sessions(safe_limit, status=status, mode=session_mode)
    return JSONResponse(
        {
            "sessions": sessions,
            "mode": normalized_mode,
            "limit": safe_limit,
            "status": status,
        }
    )


@app.get("/api/assistant/privacy")
def api_assistant_privacy() -> JSONResponse:
    summary = privacy_summary(settings, assistant_config)
    return JSONResponse(
        {
            **summary,
            "local_only": summary["status"] == "ok",
            "controls": [
                "pause",
                "private_mode",
                "audit_egress",
                "transcript_only_retention",
            ],
        }
    )


@app.get("/api/ambient/sessions")
def api_ambient_sessions(limit: int = 20) -> JSONResponse:
    safe_limit = min(max(limit, 1), 100)
    return JSONResponse({"sessions": db.list_ambient_sessions(safe_limit)})


# ---------------------------------------------------------------------------
# Template management endpoints
# ---------------------------------------------------------------------------


@app.get("/api/templates")
def api_list_templates() -> JSONResponse:
    """Return all available summary templates."""
    templates = {tid: tpl.to_dict() for tid, tpl in list_templates().items()}
    return JSONResponse({"templates": templates})


@app.post("/recordings/{recording_id}/template")
async def api_set_template(recording_id: str, request: Request) -> JSONResponse:
    """Set the summary template for a recording and re-run only summarization."""
    recording = db.get_recording(recording_id)
    if recording is None:
        return JSONResponse(
            {"error": "Recording not found"}, status_code=404
        )

    body = {}
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body"}, status_code=400
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "JSON body must be an object"}, status_code=400
        )

    template_id = body.get("template_id", "")
    if not template_id:
        return JSONResponse(
            {"error": "template_id is required"}, status_code=400
        )

    template = get_template(template_id)
    if template is None:
        return JSONResponse(
            {"error": f"Unknown template: {template_id}"}, status_code=400
        )

    db.set_recording_template(recording_id, template_id)

    queued = db.reset_summary_job(recording_id)
    if not queued and recording["status"] == "done":
        db.enqueue_job(recording_id, "summarize")
        db.update_recording(recording_id, status="queued", error=None)
        queued = True

    if queued:
        message = f"Template changed to {template.name}; re-summarizing."
    else:
        message = f"Template changed to {template.name}; it will be used when summarization runs."

    return JSONResponse({
        "status": "ok",
        "template_id": template_id,
        "queued": queued,
        "message": message,
    })
