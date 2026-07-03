from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .assistant_config import AssistantConfig
from .config import Settings


LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "llm"}


@dataclass(frozen=True)
class PrivacyIssue:
    severity: str
    check: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "check": self.check,
            "message": self.message,
        }


def validate_local_only(
    settings: Settings,
    assistant_config: AssistantConfig,
) -> list[PrivacyIssue]:
    issues: list[PrivacyIssue] = []
    allowed_hosts = _allowed_hosts(assistant_config)

    if not _is_local_host(settings.host, allowed_hosts):
        issues.append(
            PrivacyIssue(
                severity="error",
                check="web_bind",
                message=(
                    f"ATLAS_VOICE_HOST is {settings.host!r}; bind to 127.0.0.1 "
                    "unless a local LAN listener is explicitly enabled."
                ),
            )
        )

    _validate_url(
        settings.llm_base_url,
        "llm_base_url",
        "LLM_BASE_URL points outside the local allow-list.",
        allowed_hosts,
        issues,
    )

    if (
        settings.anythingllm_auto_sync
        or settings.anythingllm_api_key
        or settings.anythingllm_workspace_slug
    ):
        _validate_url(
            settings.anythingllm_base_url,
            "anythingllm_base_url",
            "ANYTHINGLLM_BASE_URL points outside the local allow-list.",
            allowed_hosts,
            issues,
        )

    privacy = assistant_config.privacy
    if privacy.get("local_only") is False:
        issues.append(
            PrivacyIssue(
                severity="error",
                check="assistant_local_only",
                message="assistant config privacy.local_only is false.",
            )
        )
    if privacy.get("telemetry") is True:
        issues.append(
            PrivacyIssue(
                severity="error",
                check="telemetry",
                message="assistant config privacy.telemetry is enabled.",
            )
        )

    for name, profile in assistant_config.profiles.items():
        if not _truthy(profile.get("enabled", False)):
            continue
        bind_host = profile.get("host") or profile.get("bind") or profile.get("listen_host")
        if bind_host and not _is_local_host(str(bind_host), allowed_hosts):
            issues.append(
                PrivacyIssue(
                    severity="error",
                    check=f"profile.{name}.bind",
                    message=f"assistant profile {name!r} listens on non-local host {bind_host!r}.",
                )
            )

    for key, value in _iter_url_values(assistant_config.raw):
        _validate_url(
            value,
            f"assistant_config.{key}",
            f"assistant config URL {key!r} points outside the local allow-list.",
            allowed_hosts,
            issues,
        )

    return issues


def privacy_summary(
    settings: Settings,
    assistant_config: AssistantConfig,
) -> dict[str, Any]:
    issues = validate_local_only(settings, assistant_config)
    status = "ok"
    if any(issue.severity == "error" for issue in issues):
        status = "error"
    elif issues:
        status = "warn"
    return {
        "status": status,
        "issue_count": len(issues),
        "issues": [issue.to_dict() for issue in issues],
        "assistant_config_path": str(assistant_config.path),
        "assistant_config_loaded": assistant_config.loaded,
    }


def _allowed_hosts(assistant_config: AssistantConfig) -> set[str]:
    allowed = set(LOCAL_HOSTS)
    configured = assistant_config.privacy.get("allowed_hosts", [])
    if isinstance(configured, str):
        configured = [configured]
    if isinstance(configured, list):
        allowed.update(str(host).strip().lower() for host in configured if str(host).strip())
    return allowed


def _validate_url(
    value: str,
    check: str,
    message: str,
    allowed_hosts: set[str],
    issues: list[PrivacyIssue],
) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        return
    host = parsed.hostname
    if host and not _is_local_host(host, allowed_hosts):
        issues.append(
            PrivacyIssue(
                severity="error",
                check=check,
                message=f"{message} Host: {host}.",
            )
        )


def _is_local_host(host: str, allowed_hosts: set[str]) -> bool:
    normalized = host.strip().lower().strip("[]")
    if normalized in allowed_hosts:
        return True
    if normalized.startswith("127."):
        return True
    return False


def _iter_url_values(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, str) and _looks_like_url_key(str(key)):
                urls.append((child_prefix, child))
            else:
                urls.extend(_iter_url_values(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            urls.extend(_iter_url_values(child, f"{prefix}[{index}]"))
    return urls


def _looks_like_url_key(key: str) -> bool:
    normalized = key.lower()
    return normalized.endswith("_url") or normalized in {"url", "endpoint", "base_url"}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
