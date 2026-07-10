from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_CONTEXT_MAX_CHARS = 6000
DEFAULT_CONTEXT_MAX_TOKENS = 1600
DEFAULT_MAX_EVIDENCE_WINDOWS = 3
MAX_COALESCED_TURN_CHARS = 800
MAX_COALESCED_TURN_SECONDS = 45.0
MAX_COALESCED_TURN_SEGMENTS = 8

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_PREFIX_RE = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)")
_FOLLOW_UP_WORDS = frozenset(
    {
        "he",
        "her",
        "hers",
        "him",
        "his",
        "it",
        "its",
        "she",
        "that",
        "their",
        "theirs",
        "them",
        "they",
        "this",
        "those",
        "why",
    }
)
_STOPWORDS = frozenset(
    """
    a about after again all am an and any are as at be because been before being by can could
    did do does doing for from give had has have he her hers him his how i if in into is it its
    me more most my of on or our ours please recording said say says she should so some summarize
    summary tell than that the their theirs them then there these they this those through to us
    was we were what when where which who why will with would you your yours
    """.split()
)
_GIST_PATTERNS = (
    re.compile(r"\b(?:gist|overview|recap|summary|summarize|key takeaways?|main topics?)\b", re.I),
    re.compile(r"\bwhat (?:was|is) (?:this|the) (?:conversation|recording) about\b", re.I),
    re.compile(r"\b(?:whole|entire|full) (?:conversation|recording|discussion)\b", re.I),
)


@dataclass(frozen=True)
class EvidenceWindow:
    """A complete, timestamped span of transcript turns used as answer evidence."""

    key: str
    start_segment_index: int
    end_segment_index: int
    start_seconds: float
    end_seconds: float
    speakers: tuple[str, ...]
    transcript: str
    score: float
    reason: str


@dataclass(frozen=True)
class ContextResult:
    """Structured focused-recording context ready for realtime prompt injection."""

    context: str
    evidence: tuple[EvidenceWindow, ...]
    resolved_query: str
    mode: str
    used_full_transcript: bool
    estimated_tokens: int


@dataclass(frozen=True)
class _Segment:
    position: int
    index: int
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class _Turn:
    start_position: int
    end_position: int
    start_segment_index: int
    end_segment_index: int
    start: float
    end: float
    speaker: str
    text: str
    segment_count: int

    @property
    def rendered(self) -> str:
        return (
            f"[{_format_seconds(self.start)}-{_format_seconds(self.end)}] "
            f"{self.speaker}: {self.text}"
        )


def build_focused_recording_context(
    recording: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    segments: Sequence[Mapping[str, Any]],
    query: str,
    *,
    recent_user_turns: Sequence[str] = (),
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    max_tokens: int | None = DEFAULT_CONTEXT_MAX_TOKENS,
    max_evidence_windows: int = DEFAULT_MAX_EVIDENCE_WINDOWS,
    target_window_chars: int = 900,
    max_window_turns: int = 6,
) -> ContextResult:
    """Build bounded, source-grounded context for chat about one recording.

    The transcript is always budgeted before derived summaries. A short transcript is
    included in full. Longer transcripts use timestamped, complete-turn windows selected
    either across the whole timeline for overview questions or by speaker-aware lexical
    relevance for specific questions. Underspecified follow-ups are expanded only from
    ``recent_user_turns``; assistant replies are deliberately not accepted by this API.
    """

    _validate_budget(max_chars, max_tokens, max_evidence_windows, target_window_chars)
    clean_query = _clean_text(query)
    query_tokens, expanded = _resolve_query_tokens(clean_query, recent_user_turns)
    resolved_query = " ".join(query_tokens)
    source_segments = _normalize_segments(segments)
    turns = _coalesce_turns(source_segments)
    recording_key = _clean_text(recording.get("id")) or "selected-recording"
    title = (_clean_text(recording.get("title")) or "Untitled recording")[:160]
    header = (
        f"Exclusive selected recording: {title}\n"
        "The timestamped transcript is the source of truth. Derived summaries are only "
        "navigation aids; resolve conflicts in favor of the transcript."
    )

    if not turns:
        blocks = [header]
        for block in _derived_summary_blocks(summary, query_tokens, overview=True):
            if _blocks_fit(blocks + [block], max_chars=max_chars, max_tokens=max_tokens):
                blocks.append(block)
        context = "\n\n".join(blocks)
        return ContextResult(
            context=context,
            evidence=(),
            resolved_query=resolved_query,
            mode="summary_only",
            used_full_transcript=False,
            estimated_tokens=estimate_tokens(context),
        )

    full_window = _window_from_turn_range(
        turns,
        0,
        len(turns) - 1,
        recording_key=recording_key,
        score=0.0,
        reason="complete_transcript",
    )
    full_block = _evidence_block(full_window, 1, complete=True)
    if _blocks_fit([header, full_block], max_chars=max_chars, max_tokens=max_tokens):
        evidence = [full_window]
        evidence_blocks = [full_block]
        mode = "full_transcript"
        used_full_transcript = True
    else:
        overview = _is_overview_query(clean_query) or not query_tokens
        if overview:
            candidates = _overview_windows(
                turns,
                recording_key=recording_key,
                max_windows=max_evidence_windows,
                target_window_chars=target_window_chars,
                max_window_turns=max_window_turns,
            )
            mode = "overview"
        else:
            candidates = _lexical_windows(
                turns,
                query_tokens,
                clean_query,
                recording_key=recording_key,
                max_windows=max_evidence_windows,
                target_window_chars=target_window_chars,
                max_window_turns=max_window_turns,
            )
            if not candidates:
                candidates = _overview_windows(
                    turns,
                    recording_key=recording_key,
                    max_windows=max_evidence_windows,
                    target_window_chars=target_window_chars,
                    max_window_turns=max_window_turns,
                )
                mode = "overview_fallback"
            else:
                mode = "focused"
        evidence, evidence_blocks = _pack_evidence(
            header,
            candidates,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )
        used_full_transcript = False

    # Evidence has already claimed its budget. Derived material may use only what remains.
    overview_for_summary = mode in {"overview", "overview_fallback", "summary_only"}
    summary_blocks: list[str] = []
    for block in _derived_summary_blocks(summary, query_tokens, overview=overview_for_summary):
        proposed = [header, *summary_blocks, block, *evidence_blocks]
        if _blocks_fit(proposed, max_chars=max_chars, max_tokens=max_tokens):
            summary_blocks.append(block)

    blocks = [header, *summary_blocks, *evidence_blocks]
    context = "\n\n".join(blocks)
    if expanded and resolved_query:
        # The resolved query is returned structurally, not inserted into untrusted evidence.
        mode += "_followup"
    return ContextResult(
        context=context,
        evidence=tuple(evidence),
        resolved_query=resolved_query,
        mode=mode,
        used_full_transcript=used_full_transcript,
        estimated_tokens=estimate_tokens(context),
    )


def estimate_tokens(text: str) -> int:
    """Conservative dependency-free token estimate for budget enforcement.

    ASCII text is approximated at four non-whitespace characters per token. Non-ASCII
    characters count individually so CJK transcripts are not dramatically under-budgeted.
    """

    ascii_non_space = 0
    non_ascii_non_space = 0
    for character in text:
        if character.isspace():
            continue
        if character.isascii():
            ascii_non_space += 1
        else:
            non_ascii_non_space += 1
    return math.ceil(ascii_non_space / 4) + non_ascii_non_space


def _validate_budget(
    max_chars: int,
    max_tokens: int | None,
    max_evidence_windows: int,
    target_window_chars: int,
) -> None:
    if max_chars < 256:
        raise ValueError("Focused recording context max_chars must be at least 256")
    if max_tokens is not None and max_tokens < 64:
        raise ValueError("Focused recording context max_tokens must be at least 64")
    if max_evidence_windows < 1:
        raise ValueError("max_evidence_windows must be at least 1")
    if target_window_chars < 80:
        raise ValueError("target_window_chars must be at least 80")


def _normalize_segments(segments: Sequence[Mapping[str, Any]]) -> list[_Segment]:
    normalized: list[_Segment] = []
    for position, segment in enumerate(segments):
        text = _clean_text(segment.get("text"))
        if not text:
            continue
        start = _safe_float(segment.get("start"), default=0.0)
        end = max(_safe_float(segment.get("end"), default=start), start)
        speaker = (
            _clean_text(segment.get("speaker"))
            or _clean_text(segment.get("display_name"))
            or _clean_text(segment.get("speaker_label"))
            or "Speaker"
        )
        normalized.append(
            _Segment(
                position=position,
                index=_safe_int(segment.get("idx"), default=position),
                start=start,
                end=end,
                speaker=speaker,
                text=text,
            )
        )
    return normalized


def _coalesce_turns(segments: Sequence[_Segment]) -> list[_Turn]:
    turns: list[_Turn] = []
    for segment in segments:
        if (
            turns
            and turns[-1].speaker.casefold() == segment.speaker.casefold()
            and segment.start - turns[-1].end <= 2.0
            and len(turns[-1].text) + 1 + len(segment.text) <= MAX_COALESCED_TURN_CHARS
            and max(turns[-1].end, segment.end) - turns[-1].start
            <= MAX_COALESCED_TURN_SECONDS
            and turns[-1].segment_count < MAX_COALESCED_TURN_SEGMENTS
        ):
            previous = turns[-1]
            turns[-1] = _Turn(
                start_position=previous.start_position,
                end_position=segment.position,
                start_segment_index=previous.start_segment_index,
                end_segment_index=segment.index,
                start=previous.start,
                end=max(previous.end, segment.end),
                speaker=previous.speaker,
                text=f"{previous.text} {segment.text}",
                segment_count=previous.segment_count + 1,
            )
            continue
        turns.append(
            _Turn(
                start_position=segment.position,
                end_position=segment.position,
                start_segment_index=segment.index,
                end_segment_index=segment.index,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                text=segment.text,
                segment_count=1,
            )
        )
    return turns


def _resolve_query_tokens(query: str, recent_user_turns: Sequence[str]) -> tuple[list[str], bool]:
    current_words = _tokens(query)
    meaningful = _meaningful_tokens(query)
    is_follow_up = (
        len(meaningful) < 3
        or any(word in _FOLLOW_UP_WORDS for word in current_words)
        or query.casefold().startswith(("and ", "but ", "how about", "what else"))
    )
    resolved = list(meaningful)
    expanded = False
    if is_follow_up:
        for prior_turn in reversed(tuple(recent_user_turns)[-2:]):
            for token in _meaningful_tokens(prior_turn):
                if token not in resolved:
                    resolved.append(token)
                    expanded = True
                if len(resolved) >= 16:
                    break
            if len(resolved) >= 16:
                break
    return resolved[:16], expanded


def _is_overview_query(query: str) -> bool:
    return any(pattern.search(query) for pattern in _GIST_PATTERNS)


def _overview_windows(
    turns: Sequence[_Turn],
    *,
    recording_key: str,
    max_windows: int,
    target_window_chars: int,
    max_window_turns: int,
) -> list[EvidenceWindow]:
    if not turns:
        return []
    anchors = _overview_anchors(len(turns), max_windows)
    windows: list[EvidenceWindow] = []
    seen_ranges: set[tuple[int, int]] = set()
    for anchor in anchors:
        start, end = _expanded_turn_range(
            turns,
            anchor,
            target_window_chars=target_window_chars,
            max_window_turns=max_window_turns,
            include_adjacent=False,
        )
        key = (start, end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        if anchor == 0:
            reason = "timeline_start"
        elif anchor == len(turns) - 1:
            reason = "timeline_end"
        else:
            reason = "timeline_middle"
        windows.append(
            _window_from_turn_range(
                turns,
                start,
                end,
                recording_key=recording_key,
                score=0.0,
                reason=reason,
            )
        )
    return windows


def _overview_anchors(turn_count: int, max_windows: int) -> list[int]:
    """Prioritize timeline coverage before adding finer-grained evidence windows."""

    if max_windows == 1:
        return [turn_count // 2]
    if max_windows == 2:
        return [0, turn_count - 1]

    last = turn_count - 1
    anchors = [0, turn_count // 2, last]
    evenly_spaced = [
        round(position * last / (max_windows - 1))
        for position in range(max_windows)
    ]
    for anchor in evenly_spaced:
        if anchor not in anchors:
            anchors.append(anchor)
        if len(anchors) >= max_windows:
            break
    return anchors


def _lexical_windows(
    turns: Sequence[_Turn],
    query_tokens: Sequence[str],
    raw_query: str,
    *,
    recording_key: str,
    max_windows: int,
    target_window_chars: int,
    max_window_turns: int,
) -> list[EvidenceWindow]:
    if not turns or not query_tokens:
        return []
    tokenized_turns = [_tokens(f"{turn.speaker} {turn.text}") for turn in turns]
    document_frequency = {
        term: sum(_term_frequency(term, tokens) > 0 for tokens in tokenized_turns)
        for term in query_tokens
    }
    average_length = sum(len(tokens) for tokens in tokenized_turns) / max(len(turns), 1)
    raw_phrase = " ".join(_meaningful_tokens(raw_query))
    scored: list[tuple[float, int]] = []
    for index, (turn, tokens) in enumerate(zip(turns, tokenized_turns)):
        score = _bm25_score(
            tokens,
            query_tokens,
            document_frequency,
            document_count=len(turns),
            average_length=average_length,
        )
        speaker_tokens = _tokens(turn.speaker)
        for term in query_tokens:
            if _term_frequency(term, speaker_tokens):
                score += 2.25
        normalized_turn = " ".join(_tokens(turn.text))
        if raw_phrase and len(raw_phrase.split()) > 1 and raw_phrase in normalized_turn:
            score += 3.0
        if score > 0:
            scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))

    windows: list[EvidenceWindow] = []
    covered_anchors: set[int] = set()
    for score, anchor in scored:
        if anchor in covered_anchors:
            continue
        start, end = _expanded_turn_range(
            turns,
            anchor,
            target_window_chars=target_window_chars,
            max_window_turns=max_window_turns,
        )
        windows.append(
            _window_from_turn_range(
                turns,
                start,
                end,
                recording_key=recording_key,
                score=round(score, 6),
                reason="lexical_match",
            )
        )
        covered_anchors.update(range(start, end + 1))
        if len(windows) >= max_windows:
            break
    return windows


def _bm25_score(
    tokens: Sequence[str],
    query_tokens: Sequence[str],
    document_frequency: Mapping[str, int],
    *,
    document_count: int,
    average_length: float,
) -> float:
    if not tokens:
        return 0.0
    score = 0.0
    k1 = 1.2
    b = 0.75
    for term in query_tokens:
        frequency = _term_frequency(term, tokens)
        if frequency <= 0:
            continue
        frequency_in_documents = document_frequency.get(term, 0)
        inverse_frequency = math.log(
            1.0 + (document_count - frequency_in_documents + 0.5) / (frequency_in_documents + 0.5)
        )
        denominator = frequency + k1 * (
            1.0 - b + b * len(tokens) / max(average_length, 1.0)
        )
        score += inverse_frequency * (frequency * (k1 + 1.0)) / denominator
    return score


def _term_frequency(term: str, tokens: Sequence[str]) -> int:
    return sum(
        token == term
        or (len(term) >= 4 and token.startswith(term))
        or (len(token) >= 4 and term.startswith(token))
        for token in tokens
    )


def _expanded_turn_range(
    turns: Sequence[_Turn],
    anchor: int,
    *,
    target_window_chars: int,
    max_window_turns: int,
    include_adjacent: bool = True,
) -> tuple[int, int]:
    if include_adjacent:
        # Lexical hits retain setup and reaction instead of returning an isolated line.
        start = max(anchor - 1, 0)
        end = min(anchor + 1, len(turns) - 1)
    else:
        # Overview anchors stay compact so start, middle, and end can share one budget.
        start = anchor
        end = anchor
    while end - start + 1 < max_window_turns:
        current_size = sum(len(turn.rendered) + 1 for turn in turns[start : end + 1])
        if current_size >= target_window_chars:
            break
        can_left = start > 0
        can_right = end + 1 < len(turns)
        if not can_left and not can_right:
            break
        left_size = len(turns[start - 1].rendered) if can_left else math.inf
        right_size = len(turns[end + 1].rendered) if can_right else math.inf
        if not include_adjacent and current_size + min(left_size, right_size) > target_window_chars:
            break

        if left_size <= right_size:
            start -= 1
        else:
            end += 1
    return start, end


def _window_from_turn_range(
    turns: Sequence[_Turn],
    start: int,
    end: int,
    *,
    recording_key: str,
    score: float,
    reason: str,
) -> EvidenceWindow:
    selected = turns[start : end + 1]
    speakers = tuple(dict.fromkeys(turn.speaker for turn in selected))
    first = selected[0]
    last = selected[-1]
    return EvidenceWindow(
        key=f"{recording_key}:{first.start_segment_index}-{last.end_segment_index}",
        start_segment_index=first.start_segment_index,
        end_segment_index=last.end_segment_index,
        start_seconds=first.start,
        end_seconds=last.end,
        speakers=speakers,
        transcript="\n".join(turn.rendered for turn in selected),
        score=score,
        reason=reason,
    )


def _pack_evidence(
    header: str,
    candidates: Sequence[EvidenceWindow],
    *,
    max_chars: int,
    max_tokens: int | None,
) -> tuple[list[EvidenceWindow], list[str]]:
    selected: list[EvidenceWindow] = []
    blocks: list[str] = []
    for candidate in candidates:
        block = _evidence_block(candidate, len(selected) + 1, complete=False)
        if _blocks_fit([header, *blocks, block], max_chars=max_chars, max_tokens=max_tokens):
            selected.append(candidate)
            blocks.append(block)
    return selected, blocks


def _evidence_block(window: EvidenceWindow, number: int, *, complete: bool) -> str:
    label = "Complete timestamped transcript" if complete else f"Transcript evidence E{number}"
    time_range = f"{_format_seconds(window.start_seconds)}-{_format_seconds(window.end_seconds)}"
    segment_range = (
        f"segments {window.start_segment_index}-{window.end_segment_index}"
        if window.start_segment_index != window.end_segment_index
        else f"segment {window.start_segment_index}"
    )
    return f"{label} [{time_range}; {segment_range}]:\n{window.transcript}"


def _derived_summary_blocks(
    summary: Mapping[str, Any] | None,
    query_tokens: Sequence[str],
    *,
    overview: bool,
) -> list[str]:
    if not summary:
        return []
    blocks: list[str] = []
    overview_text = _compact_derived_text(summary.get("text"), max_chars=520)
    if overview_text:
        blocks.append(f"Derived recording overview:\n{overview_text}")

    chunks = [chunk for chunk in summary.get("chunks") or [] if isinstance(chunk, Mapping)]
    if not chunks:
        return blocks
    if overview:
        positions = sorted({0, len(chunks) // 2, len(chunks) - 1})
    else:
        positions = sorted(
            range(len(chunks)),
            key=lambda index: (
                -sum(
                    _term_frequency(term, _tokens(str(chunks[index].get("text") or "")))
                    for term in query_tokens
                ),
                index,
            ),
        )[:1]
    for position in positions:
        chunk = chunks[position]
        text = _compact_derived_text(chunk.get("text"), max_chars=360)
        if not text or text.startswith("[Chunk "):
            continue
        chunk_index = chunk.get("chunk_index") or position + 1
        blocks.append(f"Derived chunk summary {chunk_index}:\n{text}")
    return blocks


def _compact_derived_text(value: object, *, max_chars: int) -> str:
    units: list[str] = []
    for raw_line in str(value or "").splitlines():
        line = _MARKDOWN_PREFIX_RE.sub("", raw_line).strip().strip("*_`").strip()
        if not line:
            continue
        for sentence in _SENTENCE_BOUNDARY_RE.split(line):
            clean_sentence = _clean_text(sentence)
            if clean_sentence and len(clean_sentence) <= max_chars:
                units.append(clean_sentence)
    selected: list[str] = []
    used = 0
    seen: set[str] = set()
    for unit in units:
        key = re.sub(r"\W+", " ", unit.casefold()).strip()
        if not key or key in seen:
            continue
        separator = 1 if selected else 0
        if used + separator + len(unit) > max_chars:
            continue
        selected.append(unit)
        seen.add(key)
        used += separator + len(unit)
    return " ".join(selected)


def _blocks_fit(blocks: Sequence[str], *, max_chars: int, max_tokens: int | None) -> bool:
    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        return False
    return max_tokens is None or estimate_tokens(text) <= max_tokens


def _meaningful_tokens(value: object) -> list[str]:
    return [token for token in _tokens(value) if token not in _STOPWORDS]


def _tokens(value: object) -> list[str]:
    return [token.casefold() for token in _WORD_RE.findall(_clean_text(value))]


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _safe_float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_seconds(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
