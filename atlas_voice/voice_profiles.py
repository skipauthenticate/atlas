from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .quality import MODE_ICON_FILENAMES

if TYPE_CHECKING:
    from .assistant_config import AssistantConfig


VOICE_PROFILE_ORDER = ("light", "torch", "fire")
DEFAULT_VOICE_PROFILE_ID = "torch"

_MIN_CONTEXT_BUDGET_CHARS = 256
_MAX_CONTEXT_BUDGET_CHARS = 32_000

_VOICE_PROFILE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "light": {
        "name": "Light",
        "rank": 1,
        "promise": "Fastest replies",
        "description": "Uses the smallest realtime language model and the least context.",
        "icon": MODE_ICON_FILENAMES["light"],
        "llm_profile": "qwen-voice-light",
        "context_budget_chars": 3_000,
    },
    "torch": {
        "name": "Torch",
        "rank": 2,
        "promise": "Balanced",
        "description": "The recommended balance of conversational quality and response time.",
        "icon": MODE_ICON_FILENAMES["torch"],
        "llm_profile": "qwen-voice",
        "context_budget_chars": 6_000,
    },
    "fire": {
        "name": "Fire",
        "rank": 3,
        "promise": "Strongest reasoning",
        "description": "Uses the strongest configured local model and a larger evidence budget.",
        "icon": MODE_ICON_FILENAMES["fire"],
        "llm_profile": "qwen-voice-fire",
        "context_budget_chars": 12_000,
    },
}


class VoiceProfileError(ValueError):
    pass


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    name: str
    rank: int
    promise: str
    description: str
    icon: str
    llm_profile: str
    context_budget_chars: int

    def to_public_dict(self) -> dict[str, Any]:
        """Return UI-safe metadata without model endpoints or internal profile names."""

        return {
            "id": self.id,
            "name": self.name,
            "rank": self.rank,
            "promise": self.promise,
            "description": self.description,
            "icon": self.icon,
            "context_budget_chars": self.context_budget_chars,
        }


def default_voice_profile_config() -> dict[str, dict[str, Any]]:
    """Return mutable config defaults for the three allowlisted profiles."""

    return {
        profile_id: {
            "llm_profile": str(definition["llm_profile"]),
            "context_budget_chars": int(definition["context_budget_chars"]),
        }
        for profile_id, definition in _VOICE_PROFILE_DEFINITIONS.items()
    }


def normalize_voice_profile_id(
    value: object,
    *,
    default: str | None = None,
) -> str:
    profile_id = str(value or "").strip().lower()
    if not profile_id and default is not None:
        profile_id = str(default).strip().lower()
    if profile_id not in VOICE_PROFILE_ORDER:
        allowed = ", ".join(VOICE_PROFILE_ORDER)
        raise VoiceProfileError(
            f"Unknown voice profile {profile_id or '<empty>'!r}. Use one of: {allowed}."
        )
    return profile_id


def default_voice_profile_id(assistant_config: AssistantConfig) -> str:
    direct_voice = assistant_config.profiles.get("direct_voice", {})
    configured = direct_voice.get("voice_profile", direct_voice.get("voice_tier"))
    return normalize_voice_profile_id(configured, default=DEFAULT_VOICE_PROFILE_ID)


def voice_profiles(assistant_config: AssistantConfig) -> dict[str, VoiceProfile]:
    configured_profiles = assistant_config.voice_profiles
    known_llm_profiles = assistant_config.llm_profiles
    resolved: dict[str, VoiceProfile] = {}

    for profile_id in VOICE_PROFILE_ORDER:
        definition = _VOICE_PROFILE_DEFINITIONS[profile_id]
        configured = configured_profiles.get(profile_id, {})
        llm_profile = str(
            configured.get("llm_profile") or definition["llm_profile"]
        ).strip()
        if not llm_profile:
            raise VoiceProfileError(f"Voice profile {profile_id!r} requires llm_profile.")
        if llm_profile not in known_llm_profiles:
            raise VoiceProfileError(
                f"Voice profile {profile_id!r} references unknown LLM profile "
                f"{llm_profile!r}."
            )

        resolved[profile_id] = VoiceProfile(
            id=profile_id,
            name=str(definition["name"]),
            rank=int(definition["rank"]),
            promise=str(definition["promise"]),
            description=str(definition["description"]),
            icon=str(definition["icon"]),
            llm_profile=llm_profile,
            context_budget_chars=_context_budget_chars(
                configured.get("context_budget_chars"),
                default=int(definition["context_budget_chars"]),
                profile_id=profile_id,
            ),
        )
    return resolved


def voice_profile(
    assistant_config: AssistantConfig,
    value: object | None = None,
) -> VoiceProfile:
    profile_id = normalize_voice_profile_id(
        value,
        default=default_voice_profile_id(assistant_config),
    )
    return voice_profiles(assistant_config)[profile_id]


def public_voice_profiles(assistant_config: AssistantConfig) -> list[dict[str, Any]]:
    profiles = voice_profiles(assistant_config)
    return [profiles[profile_id].to_public_dict() for profile_id in VOICE_PROFILE_ORDER]


def _context_budget_chars(value: object, *, default: int, profile_id: str) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise VoiceProfileError(
            f"Voice profile {profile_id!r} context_budget_chars must be an integer."
        )
    try:
        budget = int(value)
    except (TypeError, ValueError) as exc:
        raise VoiceProfileError(
            f"Voice profile {profile_id!r} context_budget_chars must be an integer."
        ) from exc
    if not _MIN_CONTEXT_BUDGET_CHARS <= budget <= _MAX_CONTEXT_BUDGET_CHARS:
        raise VoiceProfileError(
            f"Voice profile {profile_id!r} context_budget_chars must be between "
            f"{_MIN_CONTEXT_BUDGET_CHARS} and {_MAX_CONTEXT_BUDGET_CHARS}."
        )
    return budget
