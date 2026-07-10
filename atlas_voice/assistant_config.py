from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .voice_profiles import VOICE_PROFILE_ORDER, default_voice_profile_config


class AssistantConfigError(ValueError):
    pass


DEFAULT_ASSISTANT_CONFIG: dict[str, Any] = {
    "profiles": {
        "ambient": {
            "enabled": False,
            "stt_provider": "hyprwhspr",
            "vad_provider": "auto",
            "vad_fallback_provider": "energy",
            "llm_profile": "small-classifier",
            "tts_provider": "none",
            "store_raw_audio_seconds": 30,
        },
        "direct_voice": {
            "enabled": False,
            "realtime_backend": "atlas-native",
            "stt_provider": "whisperx",
            "llm_profile": "qwen-voice",
            "voice_profile": "torch",
            "tts_provider": "faster-qwen3-tts",
            "tts_model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "tts_voice": "Aiden",
            "tts_base_url": "http://127.0.0.1:8008/v1/audio/speech",
            "tts_health_url": "http://127.0.0.1:8008/health",
            "require_tool_confirmation": True,
        },
        "reflection": {
            "enabled": False,
            "asr_provider": "whisperx",
            "diarization_provider": "pyannote",
            "llm_profile": "qwen-deep",
            "schedule": "manual",
        },
        "summarization": {
            "llm_profile": "qwen-summary",
        },
    },
    "voice_profiles": default_voice_profile_config(),
    "llm_profiles": {
        "small-classifier": {
            "provider": "openai-compatible",
            "model": "qwen2.5-0.5b-instruct",
            "temperature": 0.0,
            "max_tokens": 256,
            "load_policy": "hot_optional",
            "roles": [
                "intent_classification",
                "sensitivity_routing",
                "low_stakes_routing",
            ],
        },
        "qwen-voice-light": {
            "provider": "openai-compatible",
            "model": "qwen3.5-2b",
            "temperature": 0.2,
            "max_tokens": 600,
            "load_policy": "warm_optional",
            "roles": [
                "fast_voice_replies",
                "low_stakes_routing",
            ],
        },
        "qwen-voice": {
            "provider": "openai-compatible",
            "model": "qwen3.5-9b",
            "temperature": 0.3,
            "max_tokens": 800,
            "load_policy": "warm_optional",
            "roles": [
                "fast_voice_replies",
                "low_stakes_routing",
                "brief_tool_planning",
            ],
        },
        "qwen-voice-fire": {
            "provider": "openai-compatible",
            "model": "qwen3.6-35b-a3b",
            "temperature": 0.2,
            "max_tokens": 800,
            "load_policy": "on_demand",
            "roles": [
                "fast_voice_replies",
                "complex_reasoning",
                "brief_tool_planning",
            ],
        },
        "qwen-summary": {
            "provider": "openai-compatible",
            "model": "qwen-27b-instruct",
            "load_policy": "on_demand",
            "roles": [
                "recording_summarization",
                "session_summaries",
                "template_summaries",
            ],
        },
        "qwen-deep": {
            "provider": "openai-compatible",
            "model": "qwen-27b-instruct",
            "load_policy": "on_demand",
            "roles": [
                "deep_coaching",
                "complex_reasoning",
                "weekly_reviews",
                "direct_questions",
            ],
        },
    },
    "privacy": {
        "local_only": True,
        "telemetry": False,
        "allowed_hosts": ["127.0.0.1", "localhost", "llm"],
        "raw_audio_retention_seconds": 30,
    },
}


@dataclass(frozen=True)
class AssistantConfig:
    path: Path
    loaded: bool
    raw: dict[str, Any]
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def profiles(self) -> dict[str, dict[str, Any]]:
        profiles = self.raw.get("profiles")
        if isinstance(profiles, dict):
            return {
                str(name): dict(value)
                for name, value in profiles.items()
                if isinstance(value, dict)
            }
        return {}

    @property
    def privacy(self) -> dict[str, Any]:
        privacy = self.raw.get("privacy")
        return dict(privacy) if isinstance(privacy, dict) else {}

    @property
    def llm_profiles(self) -> dict[str, dict[str, Any]]:
        profiles = self.raw.get("llm_profiles")
        if isinstance(profiles, dict):
            return {
                str(name): dict(value)
                for name, value in profiles.items()
                if isinstance(value, dict)
            }
        return {}

    @property
    def voice_profiles(self) -> dict[str, dict[str, Any]]:
        """Return only the three product-approved realtime voice profiles."""

        profiles = self.raw.get("voice_profiles")
        if not isinstance(profiles, dict):
            return {}
        return {
            profile_id: dict(profiles[profile_id])
            for profile_id in VOICE_PROFILE_ORDER
            if isinstance(profiles.get(profile_id), dict)
        }

    def enabled_profiles(self) -> list[str]:
        return [
            name
            for name, profile in self.profiles.items()
            if _truthy(profile.get("enabled", False))
        ]

    def profile_overrides(self, name: str) -> dict[str, Any]:
        profiles = self.overrides.get("profiles")
        if not isinstance(profiles, dict):
            return {}
        profile = profiles.get(name)
        return dict(profile) if isinstance(profile, dict) else {}


def assistant_config_path(path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    configured = os.environ.get("ATLAS_VOICE_ASSISTANT_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd() / "config" / "atlas.assistant.yaml"


def load_assistant_config(path: Path | None = None) -> AssistantConfig:
    config_path = assistant_config_path(path)
    if not config_path.exists():
        return AssistantConfig(
            path=config_path,
            loaded=False,
            raw=_deep_merge(DEFAULT_ASSISTANT_CONFIG, {}),
            overrides={},
        )

    try:
        parsed = _load_mapping(config_path)
    except Exception as exc:  # noqa: BLE001 - normalize parser failures for callers.
        raise AssistantConfigError(f"Could not load assistant config {config_path}: {exc}") from exc

    return AssistantConfig(
        path=config_path,
        loaded=True,
        raw=_deep_merge(DEFAULT_ASSISTANT_CONFIG, parsed),
        overrides=parsed,
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text()
    stripped = text.lstrip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        parsed = json.loads(text)
    else:
        parsed = _parse_simple_yaml(text)
    if not isinstance(parsed, dict):
        raise AssistantConfigError("assistant config must be a mapping")
    return parsed


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line[: len(raw_line) - len(raw_line.lstrip())].replace(" ", ""):
            raise AssistantConfigError(f"line {line_no}: tabs are not supported")

        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        item = line.strip()
        if ":" not in item:
            raise AssistantConfigError(f"line {line_no}: expected key: value")
        key, raw_value = item.split(":", 1)
        key = key.strip()
        if not key:
            raise AssistantConfigError(f"line {line_no}: missing key")

        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = raw_value.strip()
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "#" and quote is None:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(item.strip()) for item in body.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = _deep_merge(value, {})
        elif isinstance(value, list):
            merged[key] = list(value)
        else:
            merged[key] = value
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
