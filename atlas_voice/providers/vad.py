from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from atlas_voice.config import Settings


@dataclass(frozen=True)
class VadSegment:
    start: float
    end: float
    confidence: float | None = None
    provider: str = "hyprwhspr"


def ambient_vad_provider_chain(settings: Settings) -> list[str]:
    provider = _normalize_provider(getattr(settings, "ambient_vad_provider", "auto"))
    fallback = _normalize_provider(getattr(settings, "ambient_vad_fallback_provider", "energy"))

    providers: list[str] = []
    if provider == "auto":
        if hyprwhspr_vad_reliable(settings):
            providers.append("hyprwhspr")
        providers.append(fallback or "energy")
    elif provider == "hyprwhspr":
        if hyprwhspr_vad_reliable(settings):
            providers.append("hyprwhspr")
        providers.append(fallback or "energy")
    else:
        providers.append(provider or "energy")

    if "energy" not in providers:
        providers.append("energy")
    return _dedupe([candidate for candidate in providers if candidate != "none"])


def hyprwhspr_vad_reliable(settings: Settings) -> bool:
    endpoint = _vad_endpoint(settings)
    if not endpoint:
        return False
    health_url = _vad_health_url(settings, endpoint)
    try:
        response = httpx.get(health_url, timeout=min(_timeout(settings), 2.0))
    except Exception:  # noqa: BLE001 - health probes are best-effort gates.
        return False
    return 200 <= int(getattr(response, "status_code", 0)) < 400


def detect_hyprwhspr_vad(audio_path: Path, settings: Settings) -> list[VadSegment]:
    endpoint = _vad_endpoint(settings)
    if not endpoint:
        raise RuntimeError("Hyprwhspr VAD endpoint is not configured.")
    with httpx.Client(timeout=_timeout(settings)) as client:
        response = client.post(
            endpoint,
            files={"file": (audio_path.name, audio_path.read_bytes(), _content_type(audio_path))},
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            payload: Any = response.json()
        else:
            raise RuntimeError("Hyprwhspr VAD endpoint returned non-JSON response")
    return _normalize_segments(payload)


def _normalize_segments(payload: Any) -> list[VadSegment]:
    if isinstance(payload, dict):
        raw_segments = (
            payload.get("segments")
            or payload.get("speech_segments")
            or payload.get("vad_segments")
            or []
        )
    else:
        raw_segments = payload
    if not isinstance(raw_segments, list):
        return []

    segments: list[VadSegment] = []
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        start = _optional_float(_first_present(segment, "start", "start_sec"))
        end = _optional_float(_first_present(segment, "end", "end_sec"))
        if start is None or end is None or end <= start:
            continue
        confidence = _optional_float(
            _first_present(segment, "confidence", "probability", "score")
        )
        segments.append(VadSegment(start=start, end=end, confidence=confidence))
    return segments


def _vad_endpoint(settings: Settings) -> str | None:
    explicit = getattr(settings, "hyprwhspr_vad_endpoint", None)
    if explicit:
        return str(explicit).strip().rstrip("/") or None
    endpoint = getattr(settings, "hyprwhspr_endpoint", None)
    if not endpoint:
        return None
    base = str(endpoint).strip().rstrip("/")
    if not base:
        return None
    if base.endswith("/transcribe"):
        return f"{base.rsplit('/', 1)[0]}/vad"
    return f"{base}/vad"


def _vad_health_url(settings: Settings, endpoint: str) -> str:
    health_url = getattr(settings, "hyprwhspr_health_url", None)
    if health_url:
        return str(health_url).strip()
    return f"{endpoint.rsplit('/', 1)[0]}/health"


def _timeout(settings: Settings) -> float:
    return float(getattr(settings, "hyprwhspr_timeout", 10.0) or 10.0)


def _content_type(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
    }.get(suffix, "application/octet-stream")


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_provider(provider: str) -> str:
    return str(provider or "").strip().lower().replace("_", "-")


def _dedupe(providers: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for provider in providers:
        if provider and provider not in seen:
            result.append(provider)
            seen.add(provider)
    return result
