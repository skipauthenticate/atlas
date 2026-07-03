from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from .database import Database


@dataclass(frozen=True)
class DailyCoachingSummaryResult:
    day: str
    status: str
    dry_run: bool
    message: str
    session_count: int
    utterance_count: int
    assistant_turn_count: int
    question_count: int
    commitment_count: int
    event_id: int | None = None


def generate_daily_coaching_summary(
    db: Database,
    day: str | date | None = None,
    *,
    dry_run: bool = True,
) -> DailyCoachingSummaryResult:
    day_value = _day_string(day)
    existing = _existing_daily_summary(db, day_value)
    if existing is not None:
        return DailyCoachingSummaryResult(
            day=day_value,
            status="existing",
            dry_run=dry_run,
            message=str(existing["message"]),
            session_count=int(existing["metadata"].get("session_count", 0)),
            utterance_count=int(existing["metadata"].get("utterance_count", 0)),
            assistant_turn_count=int(existing["metadata"].get("assistant_turn_count", 0)),
            question_count=int(existing["metadata"].get("question_count", 0)),
            commitment_count=int(existing["metadata"].get("commitment_count", 0)),
            event_id=int(existing["id"]),
        )

    sessions = db.find_privacy_purge_sessions(started_on=day_value, limit=1000)
    details = [_session_detail(db, session) for session in sessions]
    utterance_count = sum(len(item["utterances"]) for item in details)
    assistant_turn_count = sum(len(item["turns"]) for item in details)
    question_count = sum(_question_count(item["utterances"]) for item in details)
    commitment_count = sum(_commitment_count(item["utterances"]) for item in details)
    message = _summary_message(
        day_value,
        details,
        utterance_count=utterance_count,
        assistant_turn_count=assistant_turn_count,
        question_count=question_count,
        commitment_count=commitment_count,
    )

    event_id: int | None = None
    if not dry_run:
        event_id = db.log_feedback_event(
            event_type="coaching.daily_summary",
            category="daily_summary",
            message=message,
            metadata={
                "day": day_value,
                "session_count": len(details),
                "utterance_count": utterance_count,
                "assistant_turn_count": assistant_turn_count,
                "question_count": question_count,
                "commitment_count": commitment_count,
            },
        )

    return DailyCoachingSummaryResult(
        day=day_value,
        status="ok",
        dry_run=dry_run,
        message=message,
        session_count=len(details),
        utterance_count=utterance_count,
        assistant_turn_count=assistant_turn_count,
        question_count=question_count,
        commitment_count=commitment_count,
        event_id=event_id,
    )


def _existing_daily_summary(db: Database, day: str) -> dict[str, Any] | None:
    for event in db.list_feedback_events(category="daily_summary", limit=500):
        if event.get("event_type") == "coaching.daily_summary" and event.get("metadata", {}).get("day") == day:
            return event
    return None


def _session_detail(db: Database, session: dict[str, Any]) -> dict[str, Any]:
    session_id = str(session["id"])
    return {
        "session": session,
        "utterances": db.list_utterances(session_id),
        "turns": db.list_assistant_turns(session_id),
    }


def _summary_message(
    day: str,
    details: list[dict[str, Any]],
    *,
    utterance_count: int,
    assistant_turn_count: int,
    question_count: int,
    commitment_count: int,
) -> str:
    session_count = len(details)
    question_ratio = question_count / utterance_count if utterance_count else 0.0
    lines = [
        f"Daily Coaching Summary - {day}",
        "",
        f"Sessions: {session_count}",
        f"User utterances: {utterance_count}",
        f"Assistant turns: {assistant_turn_count}",
        f"Question ratio: {question_ratio:.0%} ({question_count}/{utterance_count or 0})",
        f"Commitments: {commitment_count}",
    ]
    if details:
        lines.extend(["", "Session notes:"])
        for item in details[:8]:
            session = item["session"]
            title = session.get("title") or session.get("id")
            utterances = item["utterances"]
            preview = _preview(utterances)
            lines.append(f"- {title}: {preview}")
    lines.extend(["", "Suggested focus:", _suggested_focus(question_count, commitment_count, utterance_count)])
    return "\n".join(lines)


def _preview(utterances: list[dict[str, Any]]) -> str:
    texts = [str(item.get("text") or "").strip() for item in utterances if item.get("text")]
    if not texts:
        return "No user utterances captured."
    joined = " ".join(texts)
    return joined[:220] + ("..." if len(joined) > 220 else "")


def _suggested_focus(question_count: int, commitment_count: int, utterance_count: int) -> str:
    if utterance_count == 0:
        return "Capture one useful conversation before generating deeper coaching."
    if question_count == 0:
        return "Ask one concrete follow-up question before giving advice."
    if commitment_count > 0:
        return "Review commitments and choose the next visible follow-through step."
    return "Keep questions specific and summarize the next action before ending conversations."


def _question_count(utterances: list[dict[str, Any]]) -> int:
    return sum(1 for item in utterances if "?" in str(item.get("text") or ""))


def _commitment_count(utterances: list[dict[str, Any]]) -> int:
    pattern = re.compile(r"\b(i|we)('ll| will| need to| should)\b", re.I)
    return sum(1 for item in utterances if pattern.search(str(item.get("text") or "")))


def _day_string(value: str | date | None) -> str:
    if value is None:
        return datetime.now(UTC).date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    parsed = datetime.fromisoformat(value.strip()) if "T" in value else None
    if parsed is not None:
        return parsed.date().isoformat()
    date.fromisoformat(value.strip())
    return value.strip()
