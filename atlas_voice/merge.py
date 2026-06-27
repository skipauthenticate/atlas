from __future__ import annotations

from collections import defaultdict
from typing import Any


UNKNOWN_SPEAKER = "SPEAKER_UNKNOWN"


def overlap_seconds(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def speaker_for_interval(
    start: float, end: float, diarization: list[dict[str, Any]]
) -> str:
    scores: dict[str, float] = defaultdict(float)
    for turn in diarization:
        speaker = turn.get("speaker") or UNKNOWN_SPEAKER
        scores[speaker] += overlap_seconds(
            start,
            end,
            float(turn["start"]),
            float(turn["end"]),
        )
    if not scores:
        return UNKNOWN_SPEAKER
    speaker, score = max(scores.items(), key=lambda item: item[1])
    return speaker if score > 0 else UNKNOWN_SPEAKER


def merge_transcript_with_diarization(
    transcript: dict[str, Any], diarization: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []

    for segment in transcript.get("segments", []):
        words = _normal_words(segment)
        if not words:
            speaker = speaker_for_interval(
                float(segment.get("start", 0.0)),
                float(segment.get("end", 0.0)),
                diarization,
            )
            merged.append(
                {
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", 0.0)),
                    "speaker": speaker,
                    "text": (segment.get("text") or "").strip(),
                    "words": [],
                }
            )
            continue

        current: dict[str, Any] | None = None
        for word in words:
            speaker = speaker_for_interval(word["start"], word["end"], diarization)
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
        words.append(
            {
                "word": token,
                "start": float(raw["start"]),
                "end": float(raw["end"]),
            }
        )
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
