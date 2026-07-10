from __future__ import annotations

import re

from .config import Settings
from .summarizer import summary_to_sections


_TITLE_PREFIX_RE = re.compile(
    r"^(?:this\s+(?:recording|meeting|conversation)|the\s+(?:recording|meeting|conversation))\s+"
    r"(?:is\s+about|covers|discusses|focused\s+on|focuses\s+on|summarizes)\s+",
    flags=re.IGNORECASE,
)


def generate_recording_title(
    summary: str,
    settings: Settings,
    *,
    timeout_seconds: float = 60.0,
) -> str:
    """Derive a short semantic title from the summary without another model pass.

    The summary has already paid the local LLM inference cost. Reusing its opening
    snapshot keeps the pipeline responsive on shared Jetson memory and avoids a
    second request delaying an otherwise complete recording.
    """
    del settings, timeout_seconds
    return fallback_recording_title(summary)


def fallback_recording_title(summary: str) -> str:
    """Derive a useful title when the title model is unavailable."""
    sections = summary_to_sections(summary)
    candidates: list[str] = []
    for section in sections:
        title = str(section.get("title") or "").casefold()
        if title in {"snapshot", "overview", "summary", "topic overview", "call context"}:
            candidates.extend(str(item) for item in section.get("paragraphs") or [])
            candidates.extend(str(item.get("text") or "") for item in section.get("items") or [])
    if not candidates:
        for section in sections:
            candidates.extend(str(item) for item in section.get("paragraphs") or [])
            candidates.extend(str(item.get("text") or "") for item in section.get("items") or [])
            if candidates:
                break
    candidate = next((value for value in candidates if value.strip()), "Untitled recording")
    candidate = re.split(r"(?<=[.!?])\s+", candidate.strip(), maxsplit=1)[0]
    candidate = _TITLE_PREFIX_RE.sub("", candidate).strip()
    return clean_recording_title(candidate) or "Untitled recording"


def clean_recording_title(value: object) -> str:
    raw = re.sub(
        r"<think>.*?</think>",
        "",
        str(value or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    title = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    for _ in range(2):
        title = re.sub(
            r"^(?:#{1,6}\s*|title\s*:\s*)",
            "",
            title,
            flags=re.IGNORECASE,
        )
    title = title.strip(" \t\"'`*_#")
    title = re.sub(r"\.(?:wav|mp3|m4a|mp4|webm|ogg|flac)$", "", title, flags=re.IGNORECASE)
    title = " ".join(title.split())
    words = title.split()
    if len(words) > 9:
        title = " ".join(words[:9])
    title = title.rstrip(" .,:;!-_")
    return title[:80].strip()
