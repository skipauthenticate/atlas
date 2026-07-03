from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx

from atlas_voice.config import Settings

DEFAULT_MODEL = "hyprwhspr-local"


def hyprwhspr_available(settings: Settings) -> bool:
    if _endpoint(settings):
        return True
    return _resolve_cli(_cli(settings)) is not None


def transcribe_hyprwhspr(audio_path: Path, settings: Settings) -> dict[str, Any]:
    endpoint = _endpoint(settings)
    if endpoint:
        return _transcribe_endpoint(audio_path, settings, endpoint)
    return _transcribe_cli(audio_path, settings)


def _transcribe_endpoint(
    audio_path: Path,
    settings: Settings,
    endpoint: str,
) -> dict[str, Any]:
    timeout = _timeout(settings)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            endpoint,
            files={"file": (audio_path.name, audio_path.read_bytes(), _content_type(audio_path))},
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            payload: Any = response.json()
        else:
            payload = response.text
    return _normalize_result(payload, settings)


def _transcribe_cli(audio_path: Path, settings: Settings) -> dict[str, Any]:
    cli = _resolve_cli(_cli(settings))
    if cli is None:
        raise RuntimeError(
            "Hyprwhspr is not available. Set ATLAS_VOICE_HYPRWHSPR_ENDPOINT or "
            "install/configure ATLAS_VOICE_HYPRWHSPR_CLI."
        )
    completed = subprocess.run(
        [cli, str(audio_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=_timeout(settings),
    )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("Hyprwhspr CLI returned no transcript output")
    try:
        payload: Any = json.loads(stdout)
    except json.JSONDecodeError:
        payload = stdout
    return _normalize_result(payload, settings)


def _normalize_result(payload: Any, settings: Settings) -> dict[str, Any]:
    model = getattr(settings, "asr_model", None) or DEFAULT_MODEL
    if isinstance(payload, dict):
        transcript = dict(payload)
        transcript.setdefault("provider", "hyprwhspr")
        transcript.setdefault("model", model)
        if "text" not in transcript and "transcript" in transcript:
            transcript["text"] = transcript["transcript"]
        if not str(transcript.get("text") or "").strip():
            text = _segments_text(transcript.get("segments"))
            if text:
                transcript["text"] = text
        return transcript
    text = str(payload).strip()
    return {
        "provider": "hyprwhspr",
        "model": model,
        "text": text,
        "segments": [{"start": 0.0, "end": 0.0, "text": text}] if text else [],
    }


def _segments_text(segments: Any) -> str:
    if not isinstance(segments, list):
        return ""
    parts = []
    for segment in segments:
        if isinstance(segment, dict):
            text = str(segment.get("text") or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts).strip()


def _endpoint(settings: Settings) -> str | None:
    endpoint = getattr(settings, "hyprwhspr_endpoint", None)
    if not endpoint:
        return None
    return str(endpoint).strip().rstrip("/") or None


def _cli(settings: Settings) -> str:
    return str(getattr(settings, "hyprwhspr_cli", "hyprwhspr") or "hyprwhspr")


def _timeout(settings: Settings) -> float:
    return float(getattr(settings, "hyprwhspr_timeout", 10.0) or 10.0)


def _resolve_cli(command: str) -> str | None:
    command = command.strip()
    if not command:
        return None
    if os.sep in command or (os.altsep and os.altsep in command):
        path = Path(command).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
        return None
    return shutil.which(command)


def _content_type(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
    }.get(suffix, "application/octet-stream")
