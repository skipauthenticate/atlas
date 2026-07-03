from __future__ import annotations

import re
from dataclasses import dataclass

from .database import Database


AMBIENT_MEMORY_MODES = {"ambient", "meeting"}
DIRECT_VOICE_MEMORY_MODES = {"direct_voice"}


@dataclass(frozen=True)
class MemoryCandidate:
    kind: str
    title: str
    text: str
    source_type: str
    source_id: str
    importance: float
    confidence: float


@dataclass(frozen=True)
class MemoryExtractionResult:
    session_id: str
    status: str
    dry_run: bool
    candidates: tuple[MemoryCandidate, ...]
    created_ids: tuple[int, ...] = ()
    skipped_duplicates: int = 0


def extract_memories_from_ambient_session(
    db: Database,
    session_id: str,
    *,
    dry_run: bool = True,
    max_items: int = 20,
) -> MemoryExtractionResult:
    return _extract_memories_from_session(
        db,
        session_id,
        eligible_modes=AMBIENT_MEMORY_MODES,
        source_type="ambient_session",
        dry_run=dry_run,
        max_items=max_items,
    )


def extract_memories_from_direct_voice_session(
    db: Database,
    session_id: str,
    *,
    dry_run: bool = True,
    max_items: int = 20,
) -> MemoryExtractionResult:
    return _extract_memories_from_session(
        db,
        session_id,
        eligible_modes=DIRECT_VOICE_MEMORY_MODES,
        source_type="direct_voice_session",
        dry_run=dry_run,
        max_items=max_items,
    )


def _extract_memories_from_session(
    db: Database,
    session_id: str,
    *,
    eligible_modes: set[str],
    source_type: str,
    dry_run: bool,
    max_items: int,
) -> MemoryExtractionResult:
    session = db.get_ambient_session(session_id)
    if session is None:
        raise ValueError(f"Unknown ambient session: {session_id}")
    if str(session.get("mode") or "") not in eligible_modes:
        return MemoryExtractionResult(
            session_id=session_id,
            status="skipped",
            dry_run=dry_run,
            candidates=(),
        )

    candidates = tuple(_candidate_memories(db.list_utterances(session_id), session_id, source_type))[
        :max_items
    ]
    existing = _existing_memory_texts(db, session_id, source_type)
    fresh_candidates = tuple(
        candidate for candidate in candidates if _dedupe_key(candidate.text) not in existing
    )
    skipped_duplicates = len(candidates) - len(fresh_candidates)
    if dry_run:
        return MemoryExtractionResult(
            session_id=session_id,
            status="ok",
            dry_run=True,
            candidates=fresh_candidates,
            skipped_duplicates=skipped_duplicates,
        )

    created_ids: list[int] = []
    for candidate in fresh_candidates:
        created_ids.append(
            db.create_memory_item(
                kind=candidate.kind,
                title=candidate.title,
                text=candidate.text,
                source_type=candidate.source_type,
                source_id=candidate.source_id,
                importance=candidate.importance,
                confidence=candidate.confidence,
            )
        )

    return MemoryExtractionResult(
        session_id=session_id,
        status="ok",
        dry_run=False,
        candidates=fresh_candidates,
        created_ids=tuple(created_ids),
        skipped_duplicates=skipped_duplicates,
    )


def _candidate_memories(
    utterances: list[dict[str, object]],
    session_id: str,
    source_type: str,
) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    for utterance in utterances:
        text = _clean_text(str(utterance.get("text") or ""))
        if not text:
            continue
        candidates.extend(_remember_candidates(text, session_id, source_type))
        candidates.extend(_preference_candidates(text, session_id, source_type))
    return candidates


def _remember_candidates(text: str, session_id: str, source_type: str) -> list[MemoryCandidate]:
    pattern = re.compile(
        r"(?:^|\b)remember(?:\s+(?:that|this))?[:\s]+(?P<value>[^.?!]+[.?!]?)",
        flags=re.I,
    )
    return [
        MemoryCandidate(
            kind="fact",
            title=_title_for("Memory", value),
            text=_sentence_case(value),
            source_type=source_type,
            source_id=session_id,
            importance=0.7,
            confidence=0.75,
        )
        for value in (_clean_text(match.group("value")) for match in pattern.finditer(text))
        if value
    ]


def _preference_candidates(text: str, session_id: str, source_type: str) -> list[MemoryCandidate]:
    pattern = re.compile(
        r"(?:^|\b)(?:i|we)\s+prefer\s+(?P<value>[^.?!]+[.?!]?)",
        flags=re.I,
    )
    candidates: list[MemoryCandidate] = []
    for match in pattern.finditer(text):
        value = _clean_text(match.group("value"))
        if not value:
            continue
        memory_text = f"Prefer {value[0].lower()}{value[1:]}" if value else value
        candidates.append(
            MemoryCandidate(
                kind="preference",
                title=_title_for("Preference", value),
                text=_sentence_case(memory_text),
                source_type=source_type,
                source_id=session_id,
                importance=0.8,
                confidence=0.8,
            )
        )
    return candidates


def _existing_memory_texts(db: Database, session_id: str, source_type: str) -> set[str]:
    memories = db.list_memory_items(source_type=source_type, limit=500)
    return {
        _dedupe_key(str(memory.get("text") or ""))
        for memory in memories
        if memory.get("source_id") == session_id
    }


def _title_for(prefix: str, text: str) -> str:
    words = re.findall(r"[A-Za-z0-9:_,-]+", text)[:6]
    return f"{prefix}: {' '.join(words)}" if words else prefix


def _sentence_case(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    if cleaned[-1] not in ".?!":
        cleaned += "."
    return cleaned[0].upper() + cleaned[1:]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,;:\t\n")


def _dedupe_key(text: str) -> str:
    return re.sub(r"\W+", " ", text).strip().lower()
