from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .quality import normalize_quality_tier


_MIB = 1024 * 1024
_MIN_AVAILABLE_MIB = {
    "light": {"transcribe": 1400, "diarize": 2200},
    "torch": {"transcribe": 3600, "diarize": 2800},
    "fire": {"transcribe": 9000, "diarize": 4200},
}


@dataclass(frozen=True)
class ResourceSnapshot:
    available_memory_mib: int
    free_swap_mib: int


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    tier: str
    step: str
    required_memory_mib: int
    snapshot: ResourceSnapshot
    message: str


def resource_snapshot(meminfo_path: Path = Path("/proc/meminfo")) -> ResourceSnapshot:
    values: dict[str, int] = {}
    try:
        for line in meminfo_path.read_text().splitlines():
            key, separator, raw = line.partition(":")
            if not separator:
                continue
            token = raw.strip().split()[0]
            if token.isdigit():
                values[key] = int(token) * 1024
    except OSError:
        return ResourceSnapshot(available_memory_mib=0, free_swap_mib=0)
    return ResourceSnapshot(
        available_memory_mib=int(values.get("MemAvailable", 0) / _MIB),
        free_swap_mib=int(values.get("SwapFree", 0) / _MIB),
    )


def admit_pipeline_step(
    tier: object,
    step: str,
    *,
    snapshot: ResourceSnapshot | None = None,
) -> AdmissionDecision:
    selected_tier = normalize_quality_tier(tier)
    selected_step = str(step or "").strip().lower()
    requirement = _MIN_AVAILABLE_MIB.get(selected_tier, {}).get(selected_step, 0)
    current = snapshot or resource_snapshot()
    allowed = requirement == 0 or current.available_memory_mib >= requirement
    if selected_tier == "fire" and selected_step == "transcribe":
        allowed = allowed and (
            current.free_swap_mib >= 256 or current.available_memory_mib >= requirement + 2048
        )
    if allowed:
        message = "Device resources are ready."
    else:
        message = (
            f"{selected_tier.title()} processing is waiting for device memory. "
            f"It needs about {requirement / 1024:.1f} GB free; "
            f"{current.available_memory_mib / 1024:.1f} GB is available."
        )
    return AdmissionDecision(
        allowed=allowed,
        tier=selected_tier,
        step=selected_step,
        required_memory_mib=requirement,
        snapshot=current,
        message=message,
    )
