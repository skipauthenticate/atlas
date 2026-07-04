from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .realtime import (
    check_tts_sidecar_health,
    is_tts_sidecar_provider,
    normalize_tts_provider,
    synthesize_with_tts_sidecar,
)

DEFAULT_TTS_VALIDATION_TEXT = (
    "Atlas Voice faster qwen TTS validation. "
    "This local sidecar should return clear speech without leaving the machine."
)


@dataclass(frozen=True)
class TTSValidationResult:
    status: str
    provider: str
    model: str
    health_url: str
    synthesis_url: str
    audio_path: Path
    media_type: str
    audio_bytes: int
    health_latency_ms: int | None
    synthesis_latency_ms: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "health_url": self.health_url,
            "synthesis_url": self.synthesis_url,
            "audio_path": str(self.audio_path),
            "media_type": self.media_type,
            "audio_bytes": self.audio_bytes,
            "health_latency_ms": self.health_latency_ms,
            "synthesis_latency_ms": self.synthesis_latency_ms,
        }


def validate_tts_sidecar(
    settings: Settings,
    *,
    text: str = DEFAULT_TTS_VALIDATION_TEXT,
    output_dir: Path | None = None,
    health_timeout_seconds: float | None = None,
    synthesis_timeout_seconds: float | None = None,
) -> TTSValidationResult:
    provider = normalize_tts_provider(getattr(settings, "tts_provider", None))
    if not is_tts_sidecar_provider(provider):
        raise RuntimeError(
            f"TTS sidecar provider is not configured; current provider is {provider!r}."
        )

    if health_timeout_seconds is None:
        health = check_tts_sidecar_health(settings)
    else:
        health = check_tts_sidecar_health(
            settings, timeout_seconds=health_timeout_seconds
        )
    if health.get("status") != "ok":
        detail = str(health.get("detail") or "unhealthy")
        raise RuntimeError(f"TTS sidecar health check failed: {detail}")

    target_dir = output_dir or settings.artifacts_dir / "tts-validation"
    if synthesis_timeout_seconds is None:
        audio = synthesize_with_tts_sidecar(text, settings, target_dir)
    else:
        audio = synthesize_with_tts_sidecar(
            text,
            settings,
            target_dir,
            timeout_seconds=synthesis_timeout_seconds,
        )
    return TTSValidationResult(
        status="ok",
        provider=provider,
        model=str(settings.tts_model),
        health_url=str(settings.tts_health_url),
        synthesis_url=str(settings.tts_base_url),
        audio_path=audio.path,
        media_type=audio.media_type,
        audio_bytes=len(audio.payload),
        health_latency_ms=_optional_int(health.get("latency_ms")),
        synthesis_latency_ms=audio.latency_ms,
    )


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None
