from __future__ import annotations

from pathlib import Path
from typing import Any

from atlas_voice.config import Settings
from atlas_voice.providers.whisperx_provider import stub_transcript


def transcribe_faster_whisper(audio_path: Path, settings: Settings) -> dict[str, Any]:
    if settings.stub_mode:
        transcript = stub_transcript()
        transcript["provider"] = "faster-whisper"
        transcript["model"] = _model_name(settings)
        return transcript

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install worker extras or run "
            "scripts/install-experimental-asr.sh faster-whisper, then retry."
        ) from exc

    model = WhisperModel(
        _model_name(settings),
        device=settings.whisperx_device,
        compute_type=settings.whisperx_compute_type,
        download_root=str(settings.models_dir / "asr"),
    )
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=1,
        vad_filter=True,
        word_timestamps=True,
    )
    normalized_segments = [_segment_to_dict(segment) for segment in segments]
    text = " ".join(segment["text"] for segment in normalized_segments if segment["text"]).strip()
    return {
        "provider": "faster-whisper",
        "model": _model_name(settings),
        "language": _value(info, "language"),
        "language_probability": _value(info, "language_probability"),
        "text": text,
        "segments": normalized_segments,
    }


def _model_name(settings: Settings) -> str:
    return (
        getattr(settings, "asr_model", None)
        or getattr(settings, "faster_whisper_model", None)
        or settings.whisperx_model
    )


def _segment_to_dict(segment: Any) -> dict[str, Any]:
    words = _value(segment, "words") or []
    return {
        "start": float(_value(segment, "start") or 0.0),
        "end": float(_value(segment, "end") or 0.0),
        "text": str(_value(segment, "text") or "").strip(),
        "words": [_word_to_dict(word) for word in words],
    }


def _word_to_dict(word: Any) -> dict[str, Any]:
    payload = {
        "word": str(_value(word, "word") or "").strip(),
        "start": float(_value(word, "start") or 0.0),
        "end": float(_value(word, "end") or 0.0),
    }
    probability = _value(word, "probability")
    if probability is not None:
        payload["probability"] = float(probability)
    return payload


def _value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
