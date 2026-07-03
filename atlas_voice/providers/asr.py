from __future__ import annotations

from pathlib import Path
from typing import Any

from atlas_voice.config import Settings
from atlas_voice.providers.whisperx_provider import transcribe_audio as transcribe_whisperx


def transcribe_audio(audio_path: Path, settings: Settings) -> dict[str, Any]:
    provider = settings.asr_provider.lower()
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
        "Use whisperx, parakeet, canary, or vibevoice."
    )
