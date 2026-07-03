from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assistant_config import _load_mapping


class ToolRegistryError(ValueError):
    pass


PERMISSIONS = {"allow", "confirm", "deny"}


@dataclass(frozen=True)
class ToolDefinition:
    id: str
    name: str
    description: str
    handler: str
    permission: str
    mutating: bool
    parameters: dict[str, str]
    source: Path | None = None

    @property
    def requires_confirmation(self) -> bool:
        return self.permission == "confirm" or self.mutating

    @property
    def allowed(self) -> bool:
        return self.permission != "deny"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "handler": self.handler,
            "permission": self.permission,
            "mutating": self.mutating,
            "requires_confirmation": self.requires_confirmation,
            "allowed": self.allowed,
            "parameters": dict(self.parameters),
            "source": str(self.source) if self.source else None,
        }


class ToolRegistry:
    def __init__(self, tools: dict[str, ToolDefinition]):
        self._tools = dict(tools)

    def ids(self) -> list[str]:
        return sorted(self._tools)

    def get(self, tool_id: str) -> ToolDefinition:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise ToolRegistryError(f"Unknown tool definition: {tool_id}") from exc

    def allowed_tools(self) -> list[ToolDefinition]:
        return [tool for tool in sorted(self._tools.values(), key=lambda item: item.id) if tool.allowed]

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {tool_id: tool.to_dict() for tool_id, tool in sorted(self._tools.items())}


def tool_config_dir(path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    configured = os.environ.get("ATLAS_VOICE_TOOLS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd() / "config" / "tools"


def load_tool_registry(path: Path | None = None) -> ToolRegistry:
    tools: dict[str, ToolDefinition] = {}
    for tool_id, payload in _builtin_tool_payloads().items():
        tools[tool_id] = _tool_from_payload(tool_id, payload, source=None)

    directory = tool_config_dir(path)
    if directory.exists():
        if not directory.is_dir():
            raise ToolRegistryError(f"Tool config path is not a directory: {directory}")
        for file_path in sorted(directory.glob("*.yaml")):
            for tool in _load_tool_file(file_path):
                tools[tool.id] = tool
    return ToolRegistry(tools)


def _load_tool_file(path: Path) -> list[ToolDefinition]:
    try:
        parsed = _load_mapping(path)
    except Exception as exc:  # noqa: BLE001 - normalize parser errors for callers.
        raise ToolRegistryError(f"Could not load tool config {path}: {exc}") from exc
    raw_tools = parsed.get("tools")
    if not isinstance(raw_tools, dict):
        raise ToolRegistryError(f"Tool config {path} must define a tools mapping")
    tools: list[ToolDefinition] = []
    for tool_id, payload in raw_tools.items():
        if not isinstance(payload, dict):
            raise ToolRegistryError(f"Tool {tool_id} in {path} must be a mapping")
        tools.append(_tool_from_payload(str(tool_id), payload, source=path))
    return tools


def _tool_from_payload(
    tool_id: str,
    payload: dict[str, Any],
    *,
    source: Path | None,
) -> ToolDefinition:
    name = _required_text(payload, "name", tool_id, source)
    description = _required_text(payload, "description", tool_id, source)
    handler = _required_text(payload, "handler", tool_id, source)
    permission = str(payload.get("permission", "confirm")).strip().lower()
    if permission not in PERMISSIONS:
        raise ToolRegistryError(
            _error_prefix(tool_id, source) + f"permission must be one of {sorted(PERMISSIONS)}"
        )
    raw_parameters = payload.get("parameters") or {}
    if not isinstance(raw_parameters, dict):
        raise ToolRegistryError(_error_prefix(tool_id, source) + "parameters must be a mapping")
    return ToolDefinition(
        id=tool_id,
        name=name,
        description=description,
        handler=handler,
        permission=permission,
        mutating=_truthy(payload.get("mutating", False)),
        parameters={str(key): str(value) for key, value in raw_parameters.items()},
        source=source,
    )


def _required_text(
    payload: dict[str, Any],
    key: str,
    tool_id: str,
    source: Path | None,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolRegistryError(_error_prefix(tool_id, source) + f"missing required text field: {key}")
    return value.strip()


def _error_prefix(tool_id: str, source: Path | None) -> str:
    location = f" in {source}" if source else ""
    return f"Tool {tool_id}{location}: "


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _builtin_tool_payloads() -> dict[str, dict[str, Any]]:
    return {
        "search_recordings": {
            "name": "Search Recordings",
            "description": "Search local transcripts and summaries.",
            "handler": "atlas_voice.tools.search_recordings",
            "permission": "allow",
            "mutating": False,
            "parameters": {
                "query": "Search query for local recordings.",
                "limit": "Maximum result count.",
            },
        },
        "summarize_recent_day": {
            "name": "Summarize Recent Day",
            "description": "Draft a local daily summary from recent sessions.",
            "handler": "atlas_voice.tools.summarize_recent_day",
            "permission": "allow",
            "mutating": False,
            "parameters": {
                "date": "Optional YYYY-MM-DD date to summarize.",
            },
        },
        "privacy_purge": {
            "name": "Privacy Purge",
            "description": "Delete local privacy-scoped sessions after confirmation.",
            "handler": "atlas_voice.tools.privacy_purge",
            "permission": "confirm",
            "mutating": True,
            "parameters": {
                "session_id": "Exact session id to purge.",
                "keyword": "Optional keyword filter.",
            },
        },
    }
