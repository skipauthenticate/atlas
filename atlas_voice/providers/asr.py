from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from atlas_voice.config import Settings
from atlas_voice.providers.faster_whisper_provider import transcribe_faster_whisper
from atlas_voice.providers.hyprwhspr_provider import (
    hyprwhspr_reliable,
    transcribe_hyprwhspr,
)
from atlas_voice.providers.whisperx_provider import transcribe_audio as transcribe_whisperx


def realtime_asr_provider_chain(settings: Settings) -> list[str]:
    configured_provider = _normalize_provider(settings.asr_provider)
    providers: list[str] = []

    prefer_hyprwhspr = getattr(settings, "realtime_asr_prefer_hyprwhspr", True)
    if prefer_hyprwhspr and hyprwhspr_reliable(settings):
        providers.append("hyprwhspr")

    fallback_provider = _normalize_provider(
        getattr(settings, "realtime_asr_fallback_provider", "faster-whisper")
    )
    if fallback_provider and fallback_provider != "none":
        providers.append(fallback_provider)
    if configured_provider and configured_provider != "none":
        providers.append(configured_provider)
    return _dedupe(providers)


def select_realtime_asr_provider(settings: Settings) -> str:
    return realtime_asr_provider_chain(settings)[0]


def transcribe_audio(
    audio_path: Path,
    settings: Settings,
    *,
    realtime: bool = False,
) -> dict[str, Any]:
    configured_provider = _normalize_provider(settings.asr_provider)
    providers = realtime_asr_provider_chain(settings) if realtime else [configured_provider]
    errors: list[str] = []
    empty_result = False
    for provider in providers:
        try:
            payload = _transcribe_with_provider(
                audio_path,
                _settings_for_provider(settings, provider),
                provider,
            )
            if realtime and not _payload_has_text(payload):
                empty_result = True
                errors.append(f"{provider}: returned no transcript text")
                continue
            return payload
        except Exception as exc:  # noqa: BLE001 - realtime ASR should advance through fallbacks.
            if not realtime:
                raise
            errors.append(f"{provider}: {type(exc).__name__}: {exc}")
    if empty_result:
        raise RuntimeError(
            "No speech could be recognized. Please repeat that and speak for a little longer."
        )
    raise RuntimeError("Realtime ASR providers failed: " + "; ".join(errors))


def _transcribe_with_provider(
    audio_path: Path,
    settings: Settings,
    provider: str,
) -> dict[str, Any]:
    if provider == "hyprwhspr":
        return transcribe_hyprwhspr(audio_path, settings)
    if provider == "faster-whisper":
        return transcribe_faster_whisper(audio_path, settings)
    if provider == "whisperx":
        return transcribe_whisperx(audio_path, settings)
    if provider == "parakeet":
        from atlas_voice.providers.nemo_provider import transcribe_parakeet

        return transcribe_parakeet(audio_path, settings)
    if provider == "canary":
        from atlas_voice.providers.nemo_provider import transcribe_canary

        return transcribe_canary(audio_path, settings)
    if provider == "vibevoice":
        from atlas_voice.providers.vibevoice_provider import transcribe_vibevoice

        return transcribe_vibevoice(audio_path, settings)
    raise RuntimeError(
        f"Unknown ATLAS_VOICE_ASR_PROVIDER={settings.asr_provider!r}. "
        "Use whisperx, faster-whisper, hyprwhspr, parakeet, canary, or vibevoice."
    )


def _settings_for_provider(settings: Settings, provider: str) -> Settings:
    try:
        return replace(settings, asr_provider=provider)
    except TypeError:
        return settings


def _normalize_provider(provider: str) -> str:
    return str(provider or "").strip().lower().replace("_", "-")


def _dedupe(providers: list[str]) -> list[str]:
    result = []
    seen = set()
    for provider in providers:
        if provider and provider not in seen:
            result.append(provider)
            seen.add(provider)
    return result


def _payload_has_text(payload: dict[str, Any]) -> bool:
    if str(payload.get("text") or payload.get("transcript") or "").strip():
        return True
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return False
    return any(
        isinstance(segment, dict) and str(segment.get("text") or "").strip()
        for segment in segments
    )
