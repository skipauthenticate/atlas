from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assistant_config import _load_mapping


class PromptRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    name: str
    domain: str
    version: str
    system: str
    user: str
    rubric: dict[str, str]
    source: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "version": self.version,
            "system": self.system,
            "user": self.user,
            "rubric": dict(self.rubric),
            "source": str(self.source) if self.source else None,
        }


class PromptRegistry:
    def __init__(self, prompts: dict[str, PromptTemplate]):
        self._prompts = dict(prompts)

    def ids(self) -> list[str]:
        return sorted(self._prompts)

    def get(self, prompt_id: str) -> PromptTemplate:
        try:
            return self._prompts[prompt_id]
        except KeyError as exc:
            raise PromptRegistryError(f"Unknown prompt template: {prompt_id}") from exc

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {prompt_id: prompt.to_dict() for prompt_id, prompt in sorted(self._prompts.items())}


def prompt_config_dir(path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    configured = os.environ.get("ATLAS_VOICE_PROMPTS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd() / "config" / "prompts"


def load_prompt_registry(path: Path | None = None) -> PromptRegistry:
    prompts: dict[str, PromptTemplate] = {}
    for prompt_id, payload in _builtin_prompt_payloads().items():
        prompts[prompt_id] = _prompt_from_payload(prompt_id, payload, source=None)

    directory = prompt_config_dir(path)
    if directory.exists():
        if not directory.is_dir():
            raise PromptRegistryError(f"Prompt config path is not a directory: {directory}")
        for file_path in sorted(directory.glob("*.yaml")):
            for prompt in _load_prompt_file(file_path):
                prompts[prompt.id] = prompt
    return PromptRegistry(prompts)


def _load_prompt_file(path: Path) -> list[PromptTemplate]:
    try:
        parsed = _load_mapping(path)
    except Exception as exc:  # noqa: BLE001 - normalize prompt parser errors for callers.
        raise PromptRegistryError(f"Could not load prompt config {path}: {exc}") from exc
    raw_prompts = parsed.get("prompts")
    if not isinstance(raw_prompts, dict):
        raise PromptRegistryError(f"Prompt config {path} must define a prompts mapping")
    prompts: list[PromptTemplate] = []
    for prompt_id, payload in raw_prompts.items():
        if not isinstance(payload, dict):
            raise PromptRegistryError(f"Prompt {prompt_id} in {path} must be a mapping")
        prompts.append(_prompt_from_payload(str(prompt_id), payload, source=path))
    return prompts


def _prompt_from_payload(
    prompt_id: str,
    payload: dict[str, Any],
    *,
    source: Path | None,
) -> PromptTemplate:
    name = _required_text(payload, "name", prompt_id, source)
    domain = _required_text(payload, "domain", prompt_id, source)
    system = _required_text(payload, "system", prompt_id, source)
    user = _required_text(payload, "user", prompt_id, source)
    version = str(payload.get("version", "1"))
    raw_rubric = payload.get("rubric") or {}
    if not isinstance(raw_rubric, dict):
        raise PromptRegistryError(_error_prefix(prompt_id, source) + "rubric must be a mapping")
    rubric = {str(key): str(value) for key, value in raw_rubric.items()}
    return PromptTemplate(
        id=prompt_id,
        name=name,
        domain=domain,
        version=version,
        system=system,
        user=user,
        rubric=rubric,
        source=source,
    )


def _required_text(
    payload: dict[str, Any],
    key: str,
    prompt_id: str,
    source: Path | None,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PromptRegistryError(_error_prefix(prompt_id, source) + f"missing required text field: {key}")
    return value.strip()


def _error_prefix(prompt_id: str, source: Path | None) -> str:
    location = f" in {source}" if source else ""
    return f"Prompt {prompt_id}{location}: "


def _builtin_prompt_payloads() -> dict[str, dict[str, Any]]:
    return {
        "direct_voice_assistant": {
            "name": "Direct Voice Assistant",
            "domain": "voice",
            "version": "1",
            "system": (
                "You are Atlas Voice, a private local realtime assistant. Answer "
                "conversationally, stay concise, and use only local context provided "
                "in the session."
            ),
            "user": "Respond to the user's latest local voice request.",
            "rubric": {
                "privacy": "Do not claim cloud access or send data outside local services.",
                "concision": "Prefer short, useful replies unless more detail is requested.",
                "actionability": "Surface the next useful action when one is clear.",
            },
        },
        "coaching_reflection": {
            "name": "Coaching Reflection",
            "domain": "coaching",
            "version": "1",
            "system": "You are a local coaching reviewer focused on practical behavior change.",
            "user": "Review the provided local evidence and produce one concrete next action.",
            "rubric": {
                "clarity": "Prefer specific observations over vague judgment.",
                "autonomy": "Frame suggestions as user-controlled choices.",
                "follow_through": "Connect feedback to commitments and next actions.",
            },
        },
    }
