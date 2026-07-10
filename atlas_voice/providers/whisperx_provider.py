from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from atlas_voice.config import Settings


_model_cache_lock = threading.Lock()
_asr_model_cache: dict[tuple[object, ...], Any] = {}
_asr_inference_locks: dict[tuple[object, ...], threading.Lock] = {}
_alignment_cache: dict[tuple[object, ...], tuple[Any, Any]] = {}
_alignment_inference_locks: dict[tuple[object, ...], threading.Lock] = {}


def transcribe_audio(audio_path: Path, settings: Settings) -> dict[str, Any]:
    if settings.stub_mode:
        return stub_transcript()

    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX is not installed. Install the worker extras or run through Docker."
        ) from exc

    if settings.whisperx_device != "cpu":
        _ensure_ctranslate2_cuda(settings.whisperx_device)

    model, inference_lock = _warm_asr_model(settings, whisperx)
    with inference_lock:
        result = model.transcribe(str(audio_path), batch_size=16)
    language = result.get("language")
    if language:
        (align_model, metadata), alignment_lock = _warm_alignment_model(
            settings,
            whisperx,
            str(language),
        )
        with alignment_lock:
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                str(audio_path),
                settings.whisperx_device,
                return_char_alignments=False,
            )
        result["language"] = language
    return result


def _warm_asr_model(settings: Settings, whisperx: Any) -> tuple[Any, threading.Lock]:
    key: tuple[object, ...] = (
        whisperx,
        settings.whisperx_model,
        settings.whisperx_device,
        settings.whisperx_compute_type,
        str(settings.models_dir / "asr"),
    )
    with _model_cache_lock:
        model = _asr_model_cache.get(key)
        if model is None:
            model = whisperx.load_model(
                settings.whisperx_model,
                settings.whisperx_device,
                compute_type=settings.whisperx_compute_type,
                download_root=str(settings.models_dir / "asr"),
            )
            _asr_model_cache[key] = model
            _asr_inference_locks[key] = threading.Lock()
        return model, _asr_inference_locks[key]


def _warm_alignment_model(
    settings: Settings,
    whisperx: Any,
    language: str,
) -> tuple[tuple[Any, Any], threading.Lock]:
    key: tuple[object, ...] = (
        whisperx,
        language,
        settings.whisperx_device,
        str(settings.models_dir / "asr"),
    )
    with _model_cache_lock:
        alignment = _alignment_cache.get(key)
        if alignment is None:
            alignment = whisperx.load_align_model(
                language_code=language,
                device=settings.whisperx_device,
                model_dir=str(settings.models_dir / "asr"),
            )
            _alignment_cache[key] = alignment
            _alignment_inference_locks[key] = threading.Lock()
        return alignment, _alignment_inference_locks[key]


def _clear_model_caches() -> None:
    """Clear process-local model caches. Intended for tests and controlled reloads."""

    with _model_cache_lock:
        _asr_model_cache.clear()
        _asr_inference_locks.clear()
        _alignment_cache.clear()
        _alignment_inference_locks.clear()


def stub_transcript() -> dict[str, Any]:
    return {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 4.0,
                "text": "This is a local test recording for Atlas Voice.",
                "words": [
                    {"word": "This", "start": 0.0, "end": 0.4},
                    {"word": "is", "start": 0.4, "end": 0.7},
                    {"word": "a", "start": 0.7, "end": 0.9},
                    {"word": "local", "start": 0.9, "end": 1.4},
                    {"word": "test", "start": 1.4, "end": 1.9},
                    {"word": "recording", "start": 1.9, "end": 2.8},
                    {"word": "for", "start": 2.8, "end": 3.1},
                    {"word": "Atlas", "start": 3.1, "end": 3.5},
                    {"word": "Voice.", "start": 3.5, "end": 4.0},
                ],
            }
        ],
    }


def _ensure_ctranslate2_cuda(device: str) -> None:
    try:
        import ctranslate2
    except ImportError as exc:
        raise RuntimeError("CTranslate2 is required for WhisperX GPU transcription.") from exc

    if ctranslate2.get_cuda_device_count() < 1:
        raise RuntimeError(
            f"CTranslate2 CUDA is not available, but WHISPERX_DEVICE={device!r}. "
            "Install or build CTranslate2 with CUDA support for this host."
        )
