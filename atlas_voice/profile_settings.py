from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from .assistant_config import AssistantConfig
from .config import Settings

PROFILE_SETTING_KEYS = {
    "stt_provider": "asr_provider",
    "asr_provider": "asr_provider",
    "stt_model": "asr_model",
    "asr_model": "asr_model",
    "diarization_provider": "diarization_provider",
    "llm_model": "llm_model",
    "llm_temperature": "llm_temperature",
    "llm_max_tokens": "llm_max_tokens",
    "tts_provider": "tts_provider",
    "tts_base_url": "tts_base_url",
    "tts_health_url": "tts_health_url",
    "tts_model": "tts_model",
    "tts_voice": "tts_voice",
    "tts_response_format": "tts_response_format",
    "tts_timeout": "tts_timeout",
    "tts_health_timeout": "tts_health_timeout",
    "piper_executable": "piper_executable",
    "piper_voice": "piper_voice",
    "source": "ambient_source",
    "ambient_source": "ambient_source",
    "mode": "ambient_mode",
    "ambient_mode": "ambient_mode",
    "chunk_seconds": "ambient_chunk_seconds",
    "ambient_chunk_seconds": "ambient_chunk_seconds",
    "poll_seconds": "ambient_poll_seconds",
    "ambient_poll_seconds": "ambient_poll_seconds",
    "mic_device": "ambient_mic_device",
    "ambient_mic_device": "ambient_mic_device",
    "vad_threshold": "ambient_vad_threshold",
    "ambient_vad_threshold": "ambient_vad_threshold",
    "min_speech_seconds": "ambient_min_speech_seconds",
    "ambient_min_speech_seconds": "ambient_min_speech_seconds",
    "retain_audio": "ambient_retain_audio",
    "ambient_retain_audio": "ambient_retain_audio",
    "raw_audio_retention_days": "ambient_raw_audio_retention_days",
    "ambient_raw_audio_retention_days": "ambient_raw_audio_retention_days",
    "transcript_retention_days": "ambient_transcript_retention_days",
    "ambient_transcript_retention_days": "ambient_transcript_retention_days",
    "audio_sample_rate": "realtime_audio_sample_rate",
    "realtime_audio_sample_rate": "realtime_audio_sample_rate",
    "audio_channels": "realtime_audio_channels",
    "realtime_audio_channels": "realtime_audio_channels",
    "vad_enabled": "realtime_vad_enabled",
    "realtime_vad_enabled": "realtime_vad_enabled",
    "realtime_vad_threshold": "realtime_vad_threshold",
    "realtime_vad_min_speech_ms": "realtime_vad_min_speech_ms",
    "realtime_vad_silence_ms": "realtime_vad_silence_ms",
}

_LOWERCASE_SETTINGS = {
    "asr_provider",
    "diarization_provider",
    "tts_provider",
    "tts_response_format",
    "ambient_mode",
}


def settings_for_profile(
    settings: Settings,
    assistant_config: AssistantConfig,
    profile_name: str,
) -> Settings:
    overrides = assistant_config.profile_overrides(profile_name)

    valid_settings = {field.name for field in fields(Settings)}
    updates: dict[str, Any] = {}
    _apply_llm_profile_updates(
        updates,
        settings,
        assistant_config,
        _llm_profile_name(assistant_config, profile_name, overrides),
    )
    for profile_key, value in overrides.items():
        setting_name = PROFILE_SETTING_KEYS.get(str(profile_key))
        if setting_name not in valid_settings:
            continue
        updates[setting_name] = _coerce_profile_value(
            setting_name, value, getattr(settings, setting_name)
        )

    return replace(settings, **updates) if updates else settings


def _llm_profile_name(
    assistant_config: AssistantConfig,
    profile_name: str,
    overrides: dict[str, Any],
) -> str | None:
    if "llm_profile" in overrides:
        value = overrides.get("llm_profile")
    else:
        value = assistant_config.profiles.get(profile_name, {}).get("llm_profile")
    return str(value).strip() if value else None


def _apply_llm_profile_updates(
    updates: dict[str, Any],
    settings: Settings,
    assistant_config: AssistantConfig,
    profile_name: str | None,
) -> None:
    if not profile_name:
        return
    profile = assistant_config.llm_profiles.get(profile_name)
    if not profile:
        return
    mapping = {
        "model": "llm_model",
        "base_url": "llm_base_url",
        "temperature": "llm_temperature",
        "max_tokens": "llm_max_tokens",
    }
    for profile_key, setting_name in mapping.items():
        if profile_key in profile:
            updates[setting_name] = _coerce_profile_value(
                setting_name, profile[profile_key], getattr(settings, setting_name)
            )


def _coerce_profile_value(setting_name: str, value: Any, current: Any) -> Any:
    if value is None and (isinstance(current, str) or setting_name in _LOWERCASE_SETTINGS):
        value = "none"
    if isinstance(current, bool):
        return _truthy(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, Path):
        return Path(str(value)).expanduser().resolve()
    if setting_name in _LOWERCASE_SETTINGS:
        return str(value).strip().lower()
    if current is None:
        return value
    if isinstance(current, str):
        return str(value)
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
