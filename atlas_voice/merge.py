from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any


UNKNOWN_SPEAKER = "SPEAKER_UNKNOWN"


class _DiarizationIndex:
    """Sorted interval index for repeated word-to-speaker lookups."""

    def __init__(self, diarization: list[dict[str, Any]]) -> None:
        speaker_rank: dict[Any, int] = {}
        turns: list[tuple[float, float, Any, int]] = []
        for original_index, turn in enumerate(diarization):
            speaker = turn.get("speaker") or UNKNOWN_SPEAKER
            speaker_rank.setdefault(speaker, len(speaker_rank))
            turns.append(
                (
                    float(turn["start"]),
                    float(turn["end"]),
                    speaker,
                    original_index,
                )
            )

        turns.sort(key=lambda item: (item[0], item[1], item[3]))
        self.turns = turns
        self.starts = [turn[0] for turn in turns]
        self.speaker_rank = speaker_rank
        self.prefix_max_ends: list[float] = []
        maximum_end = float("-inf")
        for _start, end, _speaker, _original_index in turns:
            maximum_end = max(maximum_end, end)
            self.prefix_max_ends.append(maximum_end)

    def speaker_for_interval(self, start: float, end: float) -> str:
        start = float(start)
        end = float(end)
        if not self.turns or end <= start:
            return UNKNOWN_SPEAKER

        # Only turns starting before the word ends can overlap it. The prefix
        # maximum lets us skip every earlier turn that has already ended.
        right = bisect_left(self.starts, end)
        left = bisect_right(self.prefix_max_ends, start, 0, right)
        scores: dict[Any, float] = {}
        for turn_start, turn_end, speaker, _original_index in self.turns[left:right]:
            seconds = overlap_seconds(start, end, turn_start, turn_end)
            if seconds <= 0:
                continue
            scores[speaker] = scores.get(speaker, 0.0) + seconds

        if not scores:
            return UNKNOWN_SPEAKER
        return max(
            scores,
            key=lambda speaker: (
                scores[speaker],
                -self.speaker_rank[speaker],
            ),
        )


def overlap_seconds(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def speaker_for_interval(start: float, end: float, diarization: list[dict[str, Any]]) -> str:
    return _DiarizationIndex(diarization).speaker_for_interval(start, end)


def merge_transcript_with_diarization(
    transcript: dict[str, Any], diarization: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    diarization_index = _DiarizationIndex(diarization)

    for segment in transcript.get("segments", []):
        words = _normal_words(segment)
        if not words:
            speaker = diarization_index.speaker_for_interval(
                float(segment.get("start", 0.0)),
                float(segment.get("end", 0.0)),
            )
            merged_segment = {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "speaker": speaker,
                "text": (segment.get("text") or "").strip(),
                "words": [],
            }
            for confidence_key in (
                "avg_logprob",
                "confidence",
                "score",
                "probability",
                "no_speech_prob",
            ):
                if confidence_key in segment:
                    merged_segment[confidence_key] = segment[confidence_key]
            merged.append(merged_segment)
            continue

        current: dict[str, Any] | None = None
        for word in words:
            speaker = diarization_index.speaker_for_interval(word["start"], word["end"])
            word = {**word, "speaker": speaker}
            if current is None or current["speaker"] != speaker:
                if current is not None:
                    _finish_run(current, merged)
                current = {
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": speaker,
                    "words": [word],
                }
            else:
                current["end"] = word["end"]
                current["words"].append(word)
        if current is not None:
            _finish_run(current, merged)

    for idx, segment in enumerate(merged):
        segment["idx"] = idx
    return merged


def _normal_words(segment: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for raw in segment.get("words") or []:
        if "start" not in raw or "end" not in raw:
            continue
        token = raw.get("word") or raw.get("text") or ""
        token = str(token).strip()
        if not token:
            continue
        word = {
            "word": token,
            "start": float(raw["start"]),
            "end": float(raw["end"]),
        }
        for confidence_key in ("score", "probability", "confidence"):
            if confidence_key in raw:
                word[confidence_key] = raw[confidence_key]
        words.append(word)
    return words


def _finish_run(current: dict[str, Any], merged: list[dict[str, Any]]) -> None:
    text = " ".join(word["word"].strip() for word in current["words"]).strip()
    merged.append(
        {
            "start": current["start"],
            "end": current["end"],
            "speaker": current["speaker"],
            "text": text,
            "words": current["words"],
        }
    )


def format_diarized_lines(segments: list[dict[str, Any]]) -> list[str]:
    return [
        f"[{format_seconds(segment['start'])} - {format_seconds(segment['end'])}] "
        f"{segment['speaker']}: {segment['text']}"
        for segment in segments
    ]


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
