from __future__ import annotations

from pathlib import Path
from typing import Any

from .assistant_config import AssistantConfig
from .config import Settings
from .database import Database
from .privacy import privacy_summary
from .realtime import check_tts_sidecar_health, is_tts_sidecar_provider, normalize_tts_provider

PROVIDER_DEFAULT_MODELS = {
    "whisperx": "large-v3-turbo",
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "canary": "nvidia/canary-1b-v2",
    "vibevoice": "microsoft/VibeVoice-ASR",
}


def runtime_status(
    settings: Settings,
    db: Database,
    assistant_config: AssistantConfig,
) -> dict[str, Any]:
    return {
        "system": system_status(),
        "services": service_health(settings, db, assistant_config),
        "privacy": privacy_summary(settings, assistant_config),
        "active_models": active_models(settings, assistant_config),
        "active_listeners": active_listeners(assistant_config, db),
    }


def assistant_health(
    settings: Settings,
    db: Database,
    assistant_config: AssistantConfig,
) -> dict[str, Any]:
    status = runtime_status(settings, db, assistant_config)
    components = {str(item["name"]): dict(item) for item in status["services"]}
    privacy = dict(status["privacy"])
    components["privacy"] = {
        "name": "privacy",
        "status": privacy["status"],
        "detail": f"{privacy['issue_count']} local-only issues",
    }

    issues = _assistant_health_issues(components, privacy)
    return {
        "status": _assistant_health_status(components),
        "local_only": privacy["status"] == "ok",
        "components": components,
        "issues": issues,
        "privacy": privacy,
        "active_listener_count": len(status["active_listeners"]),
        "active_listeners": status["active_listeners"],
        "active_models": status["active_models"],
        "realtime": {
            "websocket_path": "/v1/realtime",
            "transport": "websocket",
            "host": settings.host,
            "port": settings.port,
            "audio_sample_rate": settings.realtime_audio_sample_rate,
            "audio_channels": settings.realtime_audio_channels,
        },
    }


def system_status() -> dict[str, Any]:
    meminfo = _read_meminfo()
    memory_total = meminfo.get("MemTotal")
    memory_available = meminfo.get("MemAvailable")
    swap_total = meminfo.get("SwapTotal")
    swap_free = meminfo.get("SwapFree")
    return {
        "ram": {
            "total_bytes": memory_total,
            "available_bytes": memory_available,
            "used_percent": _used_percent(memory_total, memory_available),
            "label": _capacity_label(memory_available, memory_total),
        },
        "swap": {
            "total_bytes": swap_total,
            "free_bytes": swap_free,
            "used_percent": _used_percent(swap_total, swap_free),
            "label": _capacity_label(swap_free, swap_total),
        },
        "gpu": gpu_status(),
    }


def service_health(
    settings: Settings,
    db: Database,
    assistant_config: AssistantConfig,
) -> list[dict[str, Any]]:
    services: list[dict[str, Any]] = []
    try:
        with db.connect() as conn:
            conn.execute("SELECT 1").fetchone()
        db_status = "ok"
        db_message = "SQLite reachable"
    except Exception as exc:  # noqa: BLE001 - health checks should report, not raise.
        db_status = "error"
        db_message = str(exc)

    services.append({"name": "database", "status": db_status, "detail": db_message})
    services.append(
        {
            "name": "web",
            "status": "ok",
            "detail": f"{settings.host}:{settings.port}",
        }
    )
    services.append(
        {
            "name": "llm",
            "status": "configured",
            "detail": f"{settings.llm_model} at {settings.llm_base_url}",
        }
    )
    services.append(
        {
            "name": "asr",
            "status": "configured",
            "detail": settings.asr_provider,
        }
    )
    services.append(tts_service_health(settings, assistant_config))
    enabled_profiles = assistant_config.enabled_profiles()
    services.append(
        {
            "name": "assistant_profiles",
            "status": "configured" if enabled_profiles else "idle",
            "detail": ", ".join(enabled_profiles) if enabled_profiles else "no active listeners",
        }
    )
    active_ambient = db.count_ambient_sessions(status="active")
    services.append(
        {
            "name": "ambient",
            "status": "active" if active_ambient else "idle",
            "detail": f"{active_ambient} active sessions",
        }
    )
    return services


def tts_service_health(
    settings: Settings,
    assistant_config: AssistantConfig,
) -> dict[str, Any]:
    provider = _effective_tts_provider(settings, assistant_config)
    if provider == "none":
        return {"name": "tts", "status": "idle", "detail": "disabled"}
    if provider == "piper":
        detail = settings.piper_voice or "voice not configured"
        status = "configured" if settings.piper_voice else "warning"
        return {"name": "tts", "status": status, "detail": f"Piper: {detail}"}
    if is_tts_sidecar_provider(provider):
        health = check_tts_sidecar_health(settings)
        detail = f"{settings.tts_model} at {settings.tts_base_url}: {health['detail']}"
        return {"name": "tts", "status": str(health["status"]), "detail": detail}
    return {"name": "tts", "status": "configured", "detail": provider}


def active_models(
    settings: Settings,
    assistant_config: AssistantConfig,
) -> list[dict[str, str]]:
    asr_model = _effective_asr_model(settings)
    models = [
        {"role": "ASR", "provider": settings.asr_provider, "model": asr_model},
        {
            "role": "Diarization",
            "provider": settings.diarization_provider,
            "model": (
                settings.pyannote_model
                if settings.diarization_provider == "pyannote"
                else settings.diarization_provider
            ),
        },
        {"role": "LLM", "provider": "openai-compatible", "model": settings.llm_model},
    ]
    tts_provider = _effective_tts_provider(settings, assistant_config)
    if tts_provider != "none":
        models.append(
            {"role": "TTS", "provider": tts_provider, "model": _effective_tts_model(settings, tts_provider)}
        )
    return models


def _effective_asr_model(settings: Settings) -> str:
    if settings.asr_provider == "whisperx":
        return settings.whisperx_model
    if settings.asr_model:
        return settings.asr_model
    if settings.asr_provider == "vibevoice":
        return settings.vibevoice_model
    return PROVIDER_DEFAULT_MODELS.get(settings.asr_provider, settings.whisperx_model)


def _effective_tts_provider(
    settings: Settings,
    assistant_config: AssistantConfig,
) -> str:
    provider = normalize_tts_provider(settings.tts_provider)
    if provider != "none":
        return provider
    profile = assistant_config.profiles.get("direct_voice", {})
    if isinstance(profile, dict) and _truthy(profile.get("enabled", False)):
        return normalize_tts_provider(str(profile.get("tts_provider") or "none"))
    return "none"


def _effective_tts_model(settings: Settings, provider: str) -> str:
    if provider == "piper":
        return settings.piper_voice or "piper"
    if is_tts_sidecar_provider(provider):
        return settings.tts_model
    return provider


def active_listeners(
    assistant_config: AssistantConfig,
    db: Database,
) -> list[dict[str, str]]:
    listeners: list[dict[str, str]] = []
    for name, profile in assistant_config.profiles.items():
        if not _truthy(profile.get("enabled", False)):
            continue
        listeners.append(
            {
                "profile": name,
                "backend": str(profile.get("realtime_backend") or profile.get("stt_provider") or "-"),
                "tts": str(profile.get("tts_provider") or "none"),
            }
        )
    for session in db.list_ambient_sessions(status="active"):
        listeners.append(
            {
                "profile": str(session.get("mode") or "ambient"),
                "backend": str(session.get("source") or "ambient"),
                "tts": "none",
            }
        )
    return listeners


def _assistant_health_status(components: dict[str, dict[str, Any]]) -> str:
    critical = {"database", "privacy"}
    for name in critical:
        if components.get(name, {}).get("status") == "error":
            return "error"
    if any(_component_problem(component) for component in components.values()):
        return "degraded"
    return "ok"


def _assistant_health_issues(
    components: dict[str, dict[str, Any]],
    privacy: dict[str, Any],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for name, component in components.items():
        if not _component_problem(component):
            continue
        issues.append(
            {
                "component": name,
                "status": str(component.get("status") or "unknown"),
                "message": str(component.get("detail") or name),
            }
        )
    for issue in privacy.get("issues") or []:
        if isinstance(issue, dict):
            issues.append(
                {
                    "component": "privacy",
                    "status": str(issue.get("severity") or "error"),
                    "message": str(issue.get("message") or issue.get("check") or "privacy issue"),
                }
            )
    return issues


def _component_problem(component: dict[str, Any]) -> bool:
    return str(component.get("status") or "").lower() in {"error", "warn", "warning"}


def gpu_status() -> dict[str, Any]:
    if Path("/proc/driver/nvidia/version").exists():
        return {"available": True, "label": "NVIDIA driver present"}
    model_path = Path("/proc/device-tree/model")
    try:
        model = model_path.read_text(errors="ignore").strip("\x00\n ")
    except OSError:
        model = ""
    if "NVIDIA" in model or "Jetson" in model:
        return {"available": True, "label": model}
    return {"available": False, "label": "not detected"}


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0]) * 1024
        except ValueError:
            continue
    return values


def _used_percent(total: int | None, available: int | None) -> float | None:
    if not total:
        return None
    free = available or 0
    used = max(total - free, 0)
    return round((used / total) * 100, 1)


def _capacity_label(available: int | None, total: int | None) -> str:
    if total is None:
        return "unknown"
    if total == 0:
        return "not configured"
    if available is None:
        return _format_bytes(total)
    return f"{_format_bytes(available)} free of {_format_bytes(total)}"


def _format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{amount:.1f} TB"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
