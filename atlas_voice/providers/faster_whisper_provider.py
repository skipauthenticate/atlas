from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from atlas_voice.config import Settings
from atlas_voice.providers.whisperx_provider import stub_transcript


_model_cache_lock = threading.Lock()
_model_cache: dict[tuple[object, ...], Any] = {}
_inference_locks: dict[tuple[object, ...], threading.Lock] = {}


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

    model, inference_lock = _warm_model(settings, WhisperModel)
    with inference_lock:
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=max(int(getattr(settings, "asr_beam_size", 1)), 1),
            vad_filter=True,
            word_timestamps=True,
        )
        normalized_segments = [_segment_to_dict(segment) for segment in segments]
        if not _segments_text(normalized_segments):
            retry_options: dict[str, Any] = {
                "beam_size": 5,
                "vad_filter": False,
                "word_timestamps": True,
                "condition_on_previous_text": False,
            }
            if _model_name(settings).casefold().endswith(".en"):
                retry_options["language"] = "en"
            segments, info = model.transcribe(str(audio_path), **retry_options)
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


def _warm_model(settings: Settings, model_class: Any) -> tuple[Any, threading.Lock]:
    key: tuple[object, ...] = (
        model_class,
        _model_name(settings),
        settings.whisperx_device,
        settings.whisperx_compute_type,
        str(settings.models_dir / "asr"),
    )
    with _model_cache_lock:
        model = _model_cache.get(key)
        if model is None:
            model = model_class(
                _model_name(settings),
                device=settings.whisperx_device,
                compute_type=settings.whisperx_compute_type,
                download_root=str(settings.models_dir / "asr"),
            )
            _model_cache[key] = model
            _inference_locks[key] = threading.Lock()
        return model, _inference_locks[key]


def _clear_model_cache() -> None:
    """Clear warmed faster-whisper models for tests and bounded benchmarks."""

    with _model_cache_lock:
        _model_cache.clear()
        _inference_locks.clear()


def _segments_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(str(segment.get("text") or "").strip() for segment in segments).strip()


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
