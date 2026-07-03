from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings


REALTIME_SYSTEM_PROMPT = (
    "You are Atlas Voice, a private local realtime assistant. "
    "Answer conversationally, stay concise, and do not claim to access cloud services. "
    "Use only local context provided in the session."
)


@dataclass(frozen=True)
class RealtimeReply:
    text: str
    latency_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RealtimeAudio:
    path: Path
    payload: bytes
    media_type: str = "audio/wav"
    latency_ms: int | None = None


TTS_SIDECAR_PROVIDER = "faster-qwen3-tts"
TTS_SIDECAR_ALIASES = {
    "faster-qwen3",
    "faster-qwen3-tts-sidecar",
    "http",
    "qwen3",
    "qwen3-tts",
    "sidecar",
}


def extract_text_input(event: dict[str, Any]) -> str | None:
    event_type = str(event.get("type") or "")
    if event_type in {"input_text", "input.text", "message"}:
        return _clean_text(event.get("text") or event.get("input_text"))
    if event_type == "input_audio_buffer.commit":
        return _clean_text(event.get("text") or event.get("transcript"))
    if event_type != "conversation.item.create":
        return None

    item = event.get("item") if isinstance(event.get("item"), dict) else event
    direct = _clean_text(item.get("text") or item.get("input_text"))
    if direct:
        return direct

    content = item.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        if part_type in {"input_text", "text"}:
            text = _clean_text(part.get("text"))
            if text:
                parts.append(text)
    return _clean_text("\n".join(parts))


def decode_audio_delta(event: dict[str, Any]) -> bytes:
    value = event.get("audio") or event.get("delta")
    if not isinstance(value, str) or not value:
        raise ValueError("input_audio_buffer.append requires base64 audio in audio or delta")
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("audio payload is not valid base64") from exc


def write_realtime_audio(
    audio: bytes,
    output_dir: Path,
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    media_type: str | None = None,
) -> Path:
    if not audio:
        raise ValueError("audio buffer is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_id = uuid.uuid4().hex
    container_extension = _input_audio_extension(audio, media_type)
    if container_extension:
        output_path = output_dir / f"input-{audio_id}.{container_extension}"
        output_path.write_bytes(audio)
        return output_path
    if audio.startswith(b"RIFF"):
        output_path = output_dir / f"input-{audio_id}.wav"
        output_path.write_bytes(audio)
        return output_path

    output_path = output_dir / f"input-{audio_id}.wav"
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)
    return output_path


def transcribe_realtime_audio(
    audio: bytes,
    settings: Settings,
    output_dir: Path,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
    media_type: str | None = None,
) -> tuple[str, Path]:
    audio_path = write_realtime_audio(
        audio,
        output_dir,
        sample_rate=sample_rate or settings.realtime_audio_sample_rate,
        channels=channels or settings.realtime_audio_channels,
        media_type=media_type,
    )
    if settings.stub_mode:
        return "Audio input received.", audio_path

    from .providers.asr import transcribe_audio

    payload = transcribe_audio(audio_path, settings, realtime=True)
    text = transcript_text(payload)
    if not text:
        raise RuntimeError("ASR returned no transcript text")
    return text, audio_path


def transcript_text(payload: dict[str, Any]) -> str:
    direct = _clean_text(payload.get("text") or payload.get("transcript"))
    if direct:
        return direct
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return ""
    parts = []
    for segment in segments:
        if isinstance(segment, dict):
            text = _clean_text(segment.get("text"))
            if text:
                parts.append(text)
    return _clean_text(" ".join(parts)) or ""


def generate_realtime_reply(
    text: str,
    settings: Settings,
    *,
    instructions: str | None = None,
    timeout_seconds: float = 120.0,
) -> RealtimeReply:
    started = time.perf_counter()
    if settings.stub_mode:
        return RealtimeReply(
            text=f"Atlas heard: {text}",
            latency_ms=_elapsed_ms(started),
            tokens_in=len(text.split()),
            tokens_out=len(text.split()) + 2,
        )

    import httpx

    system_prompt = instructions or REALTIME_SYSTEM_PROMPT
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            settings.llm_base_url,
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "temperature": settings.llm_temperature,
                "max_tokens": min(settings.llm_max_tokens, 600),
                "stream": False,
            },
        )
        response.raise_for_status()
        payload = response.json()

    message = payload["choices"][0]["message"]
    reply = _clean_text(message.get("content")) or ""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    return RealtimeReply(
        text=reply,
        latency_ms=_elapsed_ms(started),
        tokens_in=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        tokens_out=usage.get("completion_tokens") if isinstance(usage, dict) else None,
        tool_calls=_extract_tool_calls(message.get("tool_calls")),
    )


def synthesize_with_piper(text: str, settings: Settings, output_dir: Path) -> RealtimeAudio:
    started = time.perf_counter()
    voice = settings.piper_voice
    if not voice:
        raise RuntimeError("ATLAS_VOICE_PIPER_VOICE is required when Piper TTS is enabled")
    executable = settings.piper_executable
    resolved = executable if Path(executable).exists() else shutil.which(executable)
    if not resolved:
        raise RuntimeError(f"Piper executable not found: {executable}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"assistant-{uuid.uuid4().hex}.wav"
    result = subprocess.run(
        [resolved, "--model", voice, "--output_file", str(output_path)],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Piper failed: {stderr or result.returncode}")
    if not output_path.exists():
        raise RuntimeError("Piper did not create an audio file")
    return RealtimeAudio(
        path=output_path,
        payload=output_path.read_bytes(),
        latency_ms=_elapsed_ms(started),
    )


def synthesize_with_tts_sidecar(
    text: str,
    settings: Settings,
    output_dir: Path,
    *,
    timeout_seconds: float | None = None,
) -> RealtimeAudio:
    cleaned = _clean_text(text)
    if not cleaned:
        raise RuntimeError("TTS sidecar requires non-empty input text")

    import httpx

    started = time.perf_counter()
    payload = {
        "model": settings.tts_model,
        "input": cleaned,
        "voice": settings.tts_voice,
        "response_format": settings.tts_response_format or "wav",
    }
    with httpx.Client(timeout=timeout_seconds or settings.tts_timeout) as client:
        response = client.post(settings.tts_base_url, json=payload)
        response.raise_for_status()
        audio = response.content

    if not audio:
        raise RuntimeError("TTS sidecar returned empty audio")

    media_type = _clean_media_type(response.headers.get("content-type"))
    extension = _audio_extension(media_type, settings.tts_response_format)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"assistant-{uuid.uuid4().hex}.{extension}"
    output_path.write_bytes(audio)
    return RealtimeAudio(
        path=output_path,
        payload=audio,
        media_type=media_type,
        latency_ms=_elapsed_ms(started),
    )


def check_tts_sidecar_health(
    settings: Settings,
    *,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    import httpx

    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_seconds or settings.tts_health_timeout) as client:
            response = client.get(settings.tts_health_url)
    except Exception as exc:  # noqa: BLE001 - health checks should report, not raise.
        return {
            "status": "error",
            "detail": f"{type(exc).__name__}: {exc}",
            "url": settings.tts_health_url,
            "latency_ms": _elapsed_ms(started),
        }

    detail = _response_detail(response)
    if 200 <= response.status_code < 300:
        return {
            "status": "ok",
            "detail": detail or "reachable",
            "url": settings.tts_health_url,
            "latency_ms": _elapsed_ms(started),
        }
    return {
        "status": "error",
        "detail": detail or f"HTTP {response.status_code}",
        "url": settings.tts_health_url,
        "latency_ms": _elapsed_ms(started),
    }


def normalize_tts_provider(provider: str | None) -> str:
    normalized = (provider or "none").strip().lower().replace("_", "-")
    if normalized in TTS_SIDECAR_ALIASES:
        return TTS_SIDECAR_PROVIDER
    return normalized or "none"


def is_tts_sidecar_provider(provider: str | None) -> bool:
    return normalize_tts_provider(provider) == TTS_SIDECAR_PROVIDER


def chunk_text(text: str, *, chunk_size: int = 48) -> list[str]:
    clean = text.strip()
    if not clean:
        return []
    words = clean.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) > chunk_size and current:
            chunks.append(current + " ")
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def audio_delta_payload(audio: bytes) -> str:
    return base64.b64encode(audio).decode("ascii")


def _extract_tool_calls(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    calls: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = _clean_text(function.get("name") or item.get("name"))
        if not name:
            continue
        arguments = function.get("arguments", item.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                parsed_arguments = {"raw": arguments}
        elif isinstance(arguments, dict):
            parsed_arguments = arguments
        else:
            parsed_arguments = {}
        calls.append(
            {
                "id": str(item.get("id") or f"call_{uuid.uuid4().hex}"),
                "name": name,
                "arguments": parsed_arguments,
                "mutating": bool(item.get("mutating", False)),
            }
        )
    return calls


def _clean_media_type(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "audio/wav"
    return value.split(";", 1)[0].strip().lower() or "audio/wav"


def _input_audio_extension(audio: bytes, media_type: str | None) -> str | None:
    clean_type = _clean_media_type(media_type) if media_type else ""
    if clean_type in {"audio/webm", "video/webm"} or audio.startswith(b"\x1aE\xdf\xa3"):
        return "webm"
    if clean_type == "audio/ogg" or audio.startswith(b"OggS"):
        return "ogg"
    if clean_type in {"audio/mp4", "audio/m4a", "audio/x-m4a"}:
        return "m4a"
    if clean_type == "audio/mpeg" or audio.startswith(b"ID3"):
        return "mp3"
    return None


def _audio_extension(media_type: str, response_format: str | None) -> str:
    if media_type == "audio/mpeg":
        return "mp3"
    if media_type == "audio/ogg":
        return "ogg"
    if media_type == "audio/flac":
        return "flac"
    if media_type == "audio/aac":
        return "aac"
    if media_type in {"audio/wav", "audio/wave", "audio/x-wav"}:
        return "wav"
    clean_format = (response_format or "wav").strip().lower().lstrip(".")
    return clean_format if clean_format in {"aac", "flac", "mp3", "ogg", "opus", "pcm", "wav"} else "wav"


def _response_detail(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - health bodies are best-effort.
        text = getattr(response, "text", "") or ""
        return text.replace("\n", " ").replace("\r", " ").strip()[:240]
    if isinstance(payload, dict):
        for key in ("status", "detail", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:240]
        return ", ".join(sorted(str(key) for key in payload.keys()))[:240]
    return str(payload)[:240]


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _elapsed_ms(started: float) -> int:
    return max(int((time.perf_counter() - started) * 1000), 0)
