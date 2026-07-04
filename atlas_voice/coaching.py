from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from .database import Database


@dataclass(frozen=True)
class ConversationSignalResult:
    session_id: str
    status: str
    dry_run: bool
    message: str
    metrics: dict[str, float | int | None]
    event_id: int | None = None


@dataclass(frozen=True)
class WritingSignalResult:
    text_hash: str
    label: str | None
    status: str
    dry_run: bool
    message: str
    metrics: dict[str, float | int]
    event_id: int | None = None


@dataclass(frozen=True)
class WeeklyCoachingSummaryResult:
    week_start: str
    week_end: str
    status: str
    dry_run: bool
    message: str
    session_count: int
    utterance_count: int
    assistant_turn_count: int
    question_count: int
    commitment_count: int
    event_id: int | None = None


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


def track_conversation_signals(
    db: Database,
    session_id: str,
    *,
    dry_run: bool = True,
) -> ConversationSignalResult:
    session = db.get_ambient_session(session_id)
    if session is None:
        raise ValueError(f"Unknown ambient session: {session_id}")
    existing = _existing_conversation_signals(db, session_id)
    if existing is not None:
        return ConversationSignalResult(
            session_id=session_id,
            status="existing",
            dry_run=dry_run,
            message=str(existing["message"]),
            metrics=dict(existing["metadata"].get("signals") or {}),
            event_id=int(existing["id"]),
        )

    utterances = db.list_utterances(session_id)
    turns = db.list_assistant_turns(session_id)
    metrics = _conversation_signal_metrics(utterances, turns)
    message = _conversation_signal_message(session, metrics)
    event_id: int | None = None
    if not dry_run:
        event_id = db.log_feedback_event(
            event_type="coaching.conversation_signals",
            category="conversation_signals",
            session_id=session_id,
            message=message,
            score=float(metrics.get("clarity") or 0.0),
            metadata={"signals": metrics},
        )

    return ConversationSignalResult(
        session_id=session_id,
        status="ok",
        dry_run=dry_run,
        message=message,
        metrics=metrics,
        event_id=event_id,
    )


def track_writing_signals(
    db: Database,
    text: str,
    *,
    label: str | None = None,
    dry_run: bool = True,
) -> WritingSignalResult:
    normalized_text = _normalize_writing_text(text)
    if not normalized_text:
        raise ValueError("Writing text is empty")
    normalized_label = label.strip() if label and label.strip() else None
    text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    existing = _existing_writing_signals(db, text_hash, normalized_label)
    if existing is not None:
        return WritingSignalResult(
            text_hash=text_hash,
            label=normalized_label,
            status="existing",
            dry_run=dry_run,
            message=str(existing["message"]),
            metrics=dict(existing["metadata"].get("signals") or {}),
            event_id=int(existing["id"]),
        )

    metrics = _writing_signal_metrics(normalized_text)
    message = _writing_signal_message(normalized_label, metrics)
    event_id: int | None = None
    if not dry_run:
        event_id = db.log_feedback_event(
            event_type="coaching.writing_signals",
            category="writing_signals",
            message=message,
            score=float(metrics.get("clarity") or 0.0),
            metadata={
                "text_hash": text_hash,
                "label": normalized_label,
                "signals": metrics,
            },
        )

    return WritingSignalResult(
        text_hash=text_hash,
        label=normalized_label,
        status="ok",
        dry_run=dry_run,
        message=message,
        metrics=metrics,
        event_id=event_id,
    )


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


def generate_weekly_coaching_summary(
    db: Database,
    week_start: str | date | None = None,
    *,
    dry_run: bool = True,
) -> WeeklyCoachingSummaryResult:
    start_date = _week_start_date(week_start)
    end_date = start_date + timedelta(days=6)
    week_start_value = start_date.isoformat()
    week_end_value = end_date.isoformat()
    existing = _existing_weekly_summary(db, week_start_value)
    if existing is not None:
        return WeeklyCoachingSummaryResult(
            week_start=week_start_value,
            week_end=str(existing["metadata"].get("week_end") or week_end_value),
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

    sessions = db.find_privacy_purge_sessions(
        after=week_start_value,
        before=(end_date + timedelta(days=1)).isoformat(),
        limit=1000,
    )
    details = [_session_detail(db, session) for session in sessions]
    utterance_count = sum(len(item["utterances"]) for item in details)
    assistant_turn_count = sum(len(item["turns"]) for item in details)
    question_count = sum(_question_count(item["utterances"]) for item in details)
    commitment_count = sum(_commitment_count(item["utterances"]) for item in details)
    message = _weekly_summary_message(
        week_start_value,
        week_end_value,
        details,
        utterance_count=utterance_count,
        assistant_turn_count=assistant_turn_count,
        question_count=question_count,
        commitment_count=commitment_count,
    )

    event_id: int | None = None
    if not dry_run:
        event_id = db.log_feedback_event(
            event_type="coaching.weekly_summary",
            category="weekly_summary",
            message=message,
            metadata={
                "week_start": week_start_value,
                "week_end": week_end_value,
                "session_count": len(details),
                "utterance_count": utterance_count,
                "assistant_turn_count": assistant_turn_count,
                "question_count": question_count,
                "commitment_count": commitment_count,
            },
        )

    return WeeklyCoachingSummaryResult(
        week_start=week_start_value,
        week_end=week_end_value,
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


def _existing_conversation_signals(db: Database, session_id: str) -> dict[str, Any] | None:
    for event in db.list_feedback_events(session_id=session_id, category="conversation_signals", limit=100):
        if event.get("event_type") == "coaching.conversation_signals":
            return event
    return None


def _existing_writing_signals(db: Database, text_hash: str, label: str | None) -> dict[str, Any] | None:
    for event in db.list_feedback_events(category="writing_signals", limit=500):
        metadata = event.get("metadata") or {}
        if (
            event.get("event_type") == "coaching.writing_signals"
            and metadata.get("text_hash") == text_hash
            and metadata.get("label") == label
        ):
            return event
    return None


def _existing_daily_summary(db: Database, day: str) -> dict[str, Any] | None:
    for event in db.list_feedback_events(category="daily_summary", limit=500):
        if event.get("event_type") == "coaching.daily_summary" and event.get("metadata", {}).get("day") == day:
            return event
    return None


def _existing_weekly_summary(db: Database, week_start: str) -> dict[str, Any] | None:
    for event in db.list_feedback_events(category="weekly_summary", limit=500):
        if (
            event.get("event_type") == "coaching.weekly_summary"
            and event.get("metadata", {}).get("week_start") == week_start
        ):
            return event
    return None


def _session_detail(db: Database, session: dict[str, Any]) -> dict[str, Any]:
    session_id = str(session["id"])
    return {
        "session": session,
        "utterances": db.list_utterances(session_id),
        "turns": db.list_assistant_turns(session_id),
    }


def _conversation_signal_metrics(
    utterances: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    utterance_count = len(utterances)
    question_count = _question_count(utterances)
    open_question_count = _open_question_count(utterances)
    closed_question_count = max(question_count - open_question_count, 0)
    affirmation_count = _affirmation_count(utterances)
    reflection_count = _reflection_count(utterances)
    summary_count = _summary_count(utterances)
    commitment_count = _commitment_count(utterances)
    actionable_count = _actionable_next_step_count(utterances)
    interruption_count = _interruption_count(utterances)
    word_counts = [_word_count(str(item.get("text") or "")) for item in utterances]
    avg_words = sum(word_counts) / utterance_count if utterance_count else 0.0
    concise_utterances = sum(1 for count in word_counts if 0 < count <= 24)
    clear_utterances = sum(1 for item in utterances if _is_clear_text(str(item.get("text") or "")))
    return {
        "utterance_count": utterance_count,
        "assistant_turn_count": len(turns),
        "question_count": question_count,
        "question_ratio": round(question_count / utterance_count, 3) if utterance_count else 0.0,
        "open_question_count": open_question_count,
        "closed_question_count": closed_question_count,
        "open_question_ratio": round(open_question_count / question_count, 3) if question_count else 0.0,
        "affirmation_count": affirmation_count,
        "affirmation_ratio": round(affirmation_count / utterance_count, 3) if utterance_count else 0.0,
        "reflection_count": reflection_count,
        "reflection_ratio": round(reflection_count / utterance_count, 3) if utterance_count else 0.0,
        "summary_count": summary_count,
        "summary_ratio": round(summary_count / utterance_count, 3) if utterance_count else 0.0,
        "commitment_count": commitment_count,
        "follow_through": round(commitment_count / utterance_count, 3) if utterance_count else 0.0,
        "actionable_next_steps": actionable_count,
        "interruption_count": interruption_count if interruption_count else None,
        "concision": round(concise_utterances / utterance_count, 3) if utterance_count else 0.0,
        "clarity": round(clear_utterances / utterance_count, 3) if utterance_count else 0.0,
        "average_words": round(avg_words, 1),
    }


def _conversation_signal_message(
    session: dict[str, Any],
    metrics: dict[str, float | int | None],
) -> str:
    title = session.get("title") or session.get("id")
    lines = [
        f"Conversation Signals - {title}",
        "",
        f"Clarity: {metrics['clarity']}",
        f"Concision: {metrics['concision']}",
        f"Question ratio: {metrics['question_ratio']}",
        f"Open questions: {metrics['open_question_count']}/{metrics['question_count']}",
        f"Affirmations: {metrics['affirmation_count']}",
        f"Reflections: {metrics['reflection_count']}",
        f"Summaries: {metrics['summary_count']}",
        f"Follow-through: {metrics['follow_through']}",
        f"Commitments: {metrics['commitment_count']}",
        f"Actionable next steps: {metrics['actionable_next_steps']}",
    ]
    if metrics.get("interruption_count") is not None:
        lines.append(f"Interruptions: {metrics['interruption_count']}")
    else:
        lines.append("Interruptions: unavailable")
    return "\n".join(lines)


def _writing_signal_metrics(text: str) -> dict[str, float | int]:
    words = re.findall(r"[A-Za-z0-9']+", text)
    word_count = len(words)
    sentences = _writing_sentences(text)
    sentence_count = len(sentences)
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    paragraph_count = len(paragraphs)
    avg_sentence_words = word_count / sentence_count if sentence_count else 0.0
    hedging_count = _pattern_count(text, r"\b(maybe|perhaps|possibly|kind of|sort of|i think|i guess|might|could)\b")
    action_count = _pattern_count(text, r"\b(please|need|approve|review|send|schedule|decide|owner|by \w+)\b")
    specificity_count = _pattern_count(text, r"\b(by \w+|today|tomorrow|monday|tuesday|wednesday|thursday|friday|owner|alice|bob|[0-9]+)\b")
    audience_count = _pattern_count(text, r"\b(hi|hello|team|you|your|please|thanks|thank you)\b")
    tone_count = _pattern_count(text, r"\b(please|thanks|thank you|could you|would you|appreciate)\b")
    repeated_phrases = _repeated_phrase_count(words)
    concise_sentences = sum(1 for sentence in sentences if 0 < _word_count(sentence) <= 24)
    clear_sentences = sum(1 for sentence in sentences if _is_clear_text(sentence))
    structured_units = min(paragraph_count, 3)
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "average_sentence_words": round(avg_sentence_words, 1),
        "clarity": round(clear_sentences / sentence_count, 3) if sentence_count else 0.0,
        "concision": round(concise_sentences / sentence_count, 3) if sentence_count else 0.0,
        "structure": round(structured_units / 3, 3) if paragraph_count else 0.0,
        "specificity": round(min(specificity_count / max(sentence_count, 1), 1.0), 3),
        "audience_fit": round(min(audience_count / max(paragraph_count, 1), 1.0), 3),
        "ask_action_clarity": round(min(action_count / max(sentence_count, 1), 1.0), 3),
        "tone": round(min(tone_count / max(sentence_count, 1), 1.0), 3),
        "hedging_count": hedging_count,
        "repeated_phrasing": repeated_phrases,
    }


def _writing_signal_message(label: str | None, metrics: dict[str, float | int]) -> str:
    title = label or "untitled writing"
    return "\n".join(
        [
            f"Writing Signals - {title}",
            "",
            f"Clarity: {metrics['clarity']}",
            f"Concision: {metrics['concision']}",
            f"Structure: {metrics['structure']}",
            f"Specificity: {metrics['specificity']}",
            f"Audience fit: {metrics['audience_fit']}",
            f"Ask/action clarity: {metrics['ask_action_clarity']}",
            f"Tone: {metrics['tone']}",
            f"Hedging: {metrics['hedging_count']}",
            f"Repeated phrasing: {metrics['repeated_phrasing']}",
        ]
    )


def _normalize_writing_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    normalized_lines: list[str] = []
    blank_pending = False
    for line in lines:
        if line:
            if blank_pending and normalized_lines:
                normalized_lines.append("")
            normalized_lines.append(line)
            blank_pending = False
        else:
            blank_pending = True
    return "\n".join(normalized_lines).strip()


def _writing_sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]


def _pattern_count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, re.I))


def _repeated_phrase_count(words: list[str]) -> int:
    lowered = [word.lower() for word in words]
    phrases = [tuple(lowered[index : index + 3]) for index in range(max(len(lowered) - 2, 0))]
    counts: dict[tuple[str, str, str], int] = {}
    for phrase in phrases:
        counts[phrase] = counts.get(phrase, 0) + 1
    return sum(1 for count in counts.values() if count > 1)


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


def _weekly_summary_message(
    week_start: str,
    week_end: str,
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
        f"Weekly Coaching Summary - {week_start} to {week_end}",
        "",
        f"Sessions: {session_count}",
        f"User utterances: {utterance_count}",
        f"Assistant turns: {assistant_turn_count}",
        f"Question ratio: {question_ratio:.0%} ({question_count}/{utterance_count or 0})",
        f"Commitments: {commitment_count}",
    ]
    if details:
        lines.extend(["", "Week highlights:"])
        for item in details[:10]:
            session = item["session"]
            title = session.get("title") or session.get("id")
            day = str(session.get("started_at") or "")[:10] or "unknown date"
            lines.append(f"- {day} {title}: {_preview(item['utterances'])}")
    lines.extend(["", "Suggested weekly focus:", _suggested_focus(question_count, commitment_count, utterance_count)])
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


def _open_question_count(utterances: list[dict[str, Any]]) -> int:
    return sum(1 for item in utterances if _is_open_question(str(item.get("text") or "")))


def _is_open_question(text: str) -> bool:
    question_parts = [part.strip().lower() for part in re.split(r"[?]+", text) if part.strip()]
    openers = (
        "what ",
        "how ",
        "why ",
        "when ",
        "where ",
        "who ",
        "which ",
        "tell me",
        "describe ",
        "walk me through",
        "in what way",
    )
    return any(part.startswith(openers) for part in question_parts)


def _affirmation_count(utterances: list[dict[str, Any]]) -> int:
    return sum(1 for item in utterances if _is_affirmation(str(item.get("text") or "")))


def _is_affirmation(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    if not normalized or "?" in normalized:
        return False
    patterns = (
        r"\b(i|we)\s+(appreciate|value|respect|admire)\b.+\b(you|your|how)\b",
        r"\b(i'?m|i am|we are)\s+impressed\b.+\b(you|your|how)\b",
        r"\b(great|good|nice|strong)\s+(job|work|point|catch)\b",
        r"\bwell done\b",
        r"\byou\s+(handled|showed|demonstrated|did|were|are)\b.+\b(clear|clearly|thoughtful|strong|careful|brave|patient|focused|prepared)\b",
        r"\bthat shows\b.+\b(strength|care|commitment|progress|thoughtfulness)\b",
    )
    return any(re.search(pattern, normalized, re.I) for pattern in patterns)


def _reflection_count(utterances: list[dict[str, Any]]) -> int:
    return sum(1 for item in utterances if _is_reflection(str(item.get("text") or "")))


def _is_reflection(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    if not normalized or "?" in normalized:
        return False
    patterns = (
        r"\b(it sounds like|sounds like|it seems like|seems like|it looks like|looks like)\b",
        r"\b(what i'?m hearing is|what i am hearing is|i hear that)\b",
        r"\b(you'?re|you are)\s+(feeling|trying|weighing|noticing|wondering|hoping|concerned|frustrated|excited|stuck)\b",
        r"\b(you want|you need|you care about|you value)\b",
        r"\b(on one hand|part of you)\b.+\b(on the other hand|another part)\b",
        r"\b(the tradeoff is|the tension is|the pattern is)\b",
    )
    return any(re.search(pattern, normalized, re.I) for pattern in patterns)


def _summary_count(utterances: list[dict[str, Any]]) -> int:
    return sum(1 for item in utterances if _is_summary_statement(str(item.get("text") or "")))


def _is_summary_statement(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    if not normalized or "?" in normalized:
        return False
    patterns = (
        r"\b(to summarize|in summary|quick summary|brief summary|let me summarize)\b",
        r"\b(the summary is|the gist is|the key points are|the main points are)\b",
        r"\b(so far|what we have so far|where we landed)\b",
        r"\b(we covered|we agreed|we decided)\b.+\b(and|,)\b",
        r"\b(here'?s what i heard|here is what i heard|here'?s what i captured)\b",
    )
    return any(re.search(pattern, normalized, re.I) for pattern in patterns)


def _commitment_count(utterances: list[dict[str, Any]]) -> int:
    pattern = re.compile(r"\b(i|we)('ll| will| need to| should)\b", re.I)
    return sum(1 for item in utterances if pattern.search(str(item.get("text") or "")))


def _actionable_next_step_count(utterances: list[dict[str, Any]]) -> int:
    pattern = re.compile(r"\b(next step|follow up|send|schedule|document|owner|tomorrow|by \w+)\b", re.I)
    return sum(1 for item in utterances if pattern.search(str(item.get("text") or "")))


def _interruption_count(utterances: list[dict[str, Any]]) -> int:
    return sum(
        1
        for item in utterances
        if "interrupt" in str(item.get("text") or "").lower()
        or str(item.get("source_provider") or "").lower() == "interruption"
    )


def _is_clear_text(text: str) -> bool:
    words = _word_count(text)
    if words == 0:
        return False
    has_specific_signal = bool(re.search(r"\b(who|what|when|where|why|how|owner|by|will|should|need)\b", text, re.I))
    return words <= 32 and (has_specific_signal or text.strip().endswith("?"))


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def _week_start_date(value: str | date | None) -> date:
    if value is None:
        current = datetime.now(UTC).date()
    elif isinstance(value, date):
        current = value
    else:
        current = date.fromisoformat(_day_string(value))
    return current - timedelta(days=current.weekday())


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
