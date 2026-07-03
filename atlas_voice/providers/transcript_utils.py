from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Any


def audio_duration_seconds(audio_path: Path, default: float = 0.0) -> float:
    try:
        with wave.open(str(audio_path), "rb") as audio:
            frame_rate = audio.getframerate()
            if frame_rate > 0:
                return audio.getnframes() / frame_rate
    except Exception:
        pass

    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        if info.samplerate > 0:
            return info.frames / info.samplerate
    except Exception:
        pass

    try:
        import torchaudio

        info = torchaudio.info(str(audio_path))
        if info.sample_rate > 0:
            return info.num_frames / info.sample_rate
    except Exception:
        pass

    return default


def transcript_from_text(
    text: str,
    audio_path: Path,
    *,
    language: str | None = None,
    provider: str,
    model: str,
) -> dict[str, Any]:
    duration = audio_duration_seconds(audio_path, default=0.0)
    clean_text = text.strip()
    return {
        "language": language or "unknown",
        "provider": provider,
        "model": model,
        "segments": [
            {
                "start": 0.0,
                "end": duration,
                "text": clean_text,
                "words": [],
            }
        ] if clean_text else [],
    }


def transcript_from_timestamped_output(
    output: Any,
    audio_path: Path,
    *,
    language: str | None = None,
    provider: str,
    model: str,
) -> dict[str, Any]:
    text = str(getattr(output, "text", output) or "").strip()
    timestamp = getattr(output, "timestamp", None) or {}
    segment_stamps = timestamp.get("segment") or []
    word_stamps = timestamp.get("word") or []

    if segment_stamps:
        segments = [_segment_from_stamp(stamp, word_stamps) for stamp in segment_stamps]
        segments = [segment for segment in segments if segment["text"] or segment["words"]]
    elif word_stamps:
        words = [_word_from_stamp(stamp) for stamp in word_stamps]
        words = [word for word in words if word["word"]]
        segments = []
        if words:
            segments.append(
                {
                    "start": words[0]["start"],
                    "end": words[-1]["end"],
                    "text": " ".join(word["word"] for word in words),
                    "words": words,
                }
            )
    else:
        return transcript_from_text(
            text, audio_path, language=language, provider=provider, model=model
        )

    if not segments and text:
        return transcript_from_text(
            text, audio_path, language=language, provider=provider, model=model
        )

    return {
        "language": language or "unknown",
        "provider": provider,
        "model": model,
        "segments": segments,
    }


def transcript_from_vibevoice_result(
    result: dict[str, Any],
    audio_path: Path,
    *,
    provider: str,
    model: str,
) -> dict[str, Any]:
    raw_segments = result.get("segments") or []
    segments: list[dict[str, Any]] = []
    diarization: list[dict[str, Any]] = []

    for raw in raw_segments:
        start = _parse_time(raw.get("start_time") or raw.get("start") or 0.0)
        end = _parse_time(raw.get("end_time") or raw.get("end") or start)
        if end < start:
            end = start
        speaker = _speaker_label(raw.get("speaker_id") or raw.get("speaker"))
        text = str(raw.get("text") or "").strip()
        segment = {
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": text,
            "words": [],
        }
        segments.append(segment)
        diarization.append({"start": start, "end": end, "speaker": speaker})

    if not segments:
        transcript = transcript_from_text(
            str(result.get("raw_text") or ""),
            audio_path,
            provider=provider,
            model=model,
        )
        transcript["diarization"] = []
        return transcript

    return {
        "language": "unknown",
        "provider": provider,
        "model": model,
        "raw_text": result.get("raw_text") or "",
        "segments": segments,
        "diarization": diarization,
    }


def diarization_from_transcript(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    if transcript.get("diarization"):
        return [dict(turn) for turn in transcript["diarization"]]

    turns: list[dict[str, Any]] = []
    for segment in transcript.get("segments") or []:
        speaker = segment.get("speaker")
        if not speaker:
            continue
        turns.append(
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "speaker": str(speaker),
            }
        )
    return turns


def _segment_from_stamp(stamp: dict[str, Any], word_stamps: list[dict[str, Any]]) -> dict[str, Any]:
    start = float(stamp.get("start", 0.0))
    end = float(stamp.get("end", start))
    text = str(stamp.get("segment") or stamp.get("text") or "").strip()
    words = [
        word
        for word in (_word_from_stamp(raw) for raw in word_stamps)
        if word["word"] and word["start"] >= start and word["end"] <= end
    ]
    return {"start": start, "end": end, "text": text, "words": words}


def _word_from_stamp(stamp: dict[str, Any]) -> dict[str, Any]:
    return {
        "word": str(stamp.get("word") or stamp.get("text") or "").strip(),
        "start": float(stamp.get("start", 0.0)),
        "end": float(stamp.get("end", stamp.get("start", 0.0))),
    }


def _parse_time(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return 0.0
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def _speaker_label(value: Any) -> str:
    raw = str(value or "0").strip()
    if raw.upper().startswith("SPEAKER_"):
        return raw.upper()
    match = re.search(r"\d+", raw)
    if match:
        return f"SPEAKER_{int(match.group(0)):02d}"
    return raw or "SPEAKER_00"
