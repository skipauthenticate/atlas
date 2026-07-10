from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .config import Settings


QUALITY_TIER_ORDER = ("light", "torch", "fire")
MODE_ICON_FILENAMES = {
    "light": "feather.svg",
    "torch": "flame.svg",
    "fire": "flame-kindling.svg",
}


@dataclass(frozen=True)
class QualityProfile:
    id: str
    name: str
    rank: int
    promise: str
    description: str
    icon: str
    asr_provider: str
    asr_model: str
    beam_size: int
    audio_cleanup: str
    alignment: bool

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "rank": self.rank,
            "promise": self.promise,
            "description": self.description,
            "icon": self.icon,
        }


def quality_profiles(settings: Settings | None = None) -> dict[str, QualityProfile]:
    """Return the three stable product-facing pipeline profiles.

    The names and ordering are a product contract. Provider and model values are
    deliberately kept behind that contract so Atlas can promote a better model
    after it wins the device benchmark without changing the user experience.
    """

    return {
        "light": QualityProfile(
            id="light",
            name="Light",
            rank=1,
            promise="Good accuracy",
            description="Finishes sooner and uses the least device memory.",
            icon=MODE_ICON_FILENAMES["light"],
            asr_provider=_setting(settings, "quality_light_asr_provider", "faster-whisper"),
            asr_model=_setting(settings, "quality_light_asr_model", "tiny.en"),
            beam_size=1,
            audio_cleanup="none",
            alignment=False,
        ),
        "torch": QualityProfile(
            id="torch",
            name="Torch",
            rank=2,
            promise="More accurate",
            description="The recommended balance for everyday recordings.",
            icon=MODE_ICON_FILENAMES["torch"],
            asr_provider=_setting(settings, "quality_torch_asr_provider", "faster-whisper"),
            asr_model=_setting(settings, "quality_torch_asr_model", "large-v3-turbo"),
            beam_size=3,
            audio_cleanup="adaptive",
            alignment=False,
        ),
        "fire": QualityProfile(
            id="fire",
            name="Fire",
            rank=3,
            promise="Most accurate",
            description="Runs the strongest checks and may take considerably longer.",
            icon=MODE_ICON_FILENAMES["fire"],
            asr_provider=_setting(settings, "quality_fire_asr_provider", "whisperx"),
            asr_model=_setting(settings, "quality_fire_asr_model", "large-v3"),
            beam_size=5,
            audio_cleanup="enhanced",
            alignment=True,
        ),
    }


def normalize_quality_tier(value: object, *, default: str = "torch") -> str:
    tier = str(value or "").strip().lower()
    if tier in QUALITY_TIER_ORDER:
        return tier
    fallback = str(default or "torch").strip().lower()
    return fallback if fallback in QUALITY_TIER_ORDER else "torch"


def quality_profile(value: object, settings: Settings | None = None) -> QualityProfile:
    tier = normalize_quality_tier(
        value,
        default=getattr(settings, "default_quality_tier", "torch") if settings else "torch",
    )
    return quality_profiles(settings)[tier]


def settings_for_quality(settings: Settings, value: object) -> Settings:
    profile = quality_profile(value, settings)
    updates: dict[str, Any] = {
        "asr_provider": profile.asr_provider,
        "asr_model": profile.asr_model,
        "asr_beam_size": profile.beam_size,
        "audio_cleanup": profile.audio_cleanup,
    }
    if profile.asr_provider == "whisperx":
        updates["whisperx_model"] = profile.asr_model
    if profile.asr_provider == "faster-whisper":
        updates["faster_whisper_model"] = profile.asr_model
    return replace(settings, **updates)


def _setting(settings: Settings | None, name: str, default: str) -> str:
    value = getattr(settings, name, None) if settings is not None else None
    return str(value or default).strip()
