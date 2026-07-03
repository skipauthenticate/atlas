from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from atlas_voice.config import Settings
from atlas_voice.providers.hyprwhspr_provider import (
    hyprwhspr_available,
    transcribe_hyprwhspr,
)
from atlas_voice.providers.whisperx_provider import transcribe_audio as transcribe_whisperx


def select_realtime_asr_provider(settings: Settings) -> str:
    provider = settings.asr_provider.lower()
    if provider == "hyprwhspr":
        return provider
    if getattr(settings, "realtime_asr_prefer_hyprwhspr", True) and hyprwhspr_available(settings):
        return "hyprwhspr"
    return provider


def transcribe_audio(
    audio_path: Path,
    settings: Settings,
    *,
    realtime: bool = False,
) -> dict[str, Any]:
    configured_provider = settings.asr_provider.lower()
    provider = select_realtime_asr_provider(settings) if realtime else configured_provider
    try:
        return _transcribe_with_provider(audio_path, settings, provider)
    except Exception:  # noqa: BLE001 - realtime ASR should fall back if preferred Hyprwhspr fails.
        if realtime and provider == "hyprwhspr" and configured_provider != "hyprwhspr":
            fallback_settings = replace(settings, asr_provider=configured_provider)
            return _transcribe_with_provider(audio_path, fallback_settings, configured_provider)
        raise


def _transcribe_with_provider(
    audio_path: Path,
    settings: Settings,
    provider: str,
) -> dict[str, Any]:
    if provider == "hyprwhspr":
        return transcribe_hyprwhspr(audio_path, settings)
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
        "Use whisperx, hyprwhspr, parakeet, canary, or vibevoice."
    )
