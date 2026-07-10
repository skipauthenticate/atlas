from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from atlas_voice.config import Settings
from atlas_voice.providers.transcript_utils import transcript_from_timestamped_output

PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
CANARY_MODEL = "nvidia/canary-1b-v2"
_model_cache_lock = threading.Lock()
_model_cache: dict[tuple[object, ...], Any] = {}
_model_inference_locks: dict[tuple[object, ...], threading.Lock] = {}


def transcribe_parakeet(audio_path: Path, settings: Settings) -> dict[str, Any]:
    model_name = settings.asr_model or PARAKEET_MODEL
    output = _transcribe_nemo(
        audio_path,
        model_name=model_name,
        settings=settings,
        timestamps=True,
    )
    return transcript_from_timestamped_output(
        output,
        audio_path,
        language=None,
        provider="parakeet",
        model=model_name,
    )


def transcribe_canary(audio_path: Path, settings: Settings) -> dict[str, Any]:
    model_name = settings.asr_model or CANARY_MODEL
    output = _transcribe_nemo(
        audio_path,
        model_name=model_name,
        settings=settings,
        timestamps=True,
        source_lang=settings.nemo_source_lang,
        target_lang=settings.nemo_target_lang,
    )
    return transcript_from_timestamped_output(
        output,
        audio_path,
        language=settings.nemo_target_lang,
        provider="canary",
        model=model_name,
    )


def _transcribe_nemo(
    audio_path: Path,
    *,
    model_name: str,
    settings: Settings,
    timestamps: bool,
    source_lang: str | None = None,
    target_lang: str | None = None,
) -> Any:
    try:
        from nemo.collections.asr.models import ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "NeMo ASR is required for Parakeet/Canary providers. Run "
            "scripts/install-experimental-asr.sh nemo, then retry."
        ) from exc

    model, inference_lock = _warm_nemo_model(
        ASRModel,
        model_name=model_name,
        device=settings.whisperx_device,
    )

    kwargs: dict[str, Any] = {"timestamps": timestamps}
    if source_lang is not None:
        kwargs["source_lang"] = source_lang
    if target_lang is not None:
        kwargs["target_lang"] = target_lang

    with inference_lock:
        try:
            outputs = model.transcribe([str(audio_path)], **kwargs)
        except TypeError:
            kwargs.pop("timestamps", None)
            outputs = model.transcribe([str(audio_path)], **kwargs)

    if not outputs:
        raise RuntimeError(f"{model_name} returned no transcription output")
    return outputs[0]


def _warm_nemo_model(
    model_class: Any,
    *,
    model_name: str,
    device: str,
) -> tuple[Any, threading.Lock]:
    key: tuple[object, ...] = (model_class, model_name, device)
    with _model_cache_lock:
        model = _model_cache.get(key)
        if model is None:
            model = model_class.from_pretrained(model_name=model_name)
            if device != "cpu":
                try:
                    model = model.to(device)
                except Exception:
                    pass
            _disable_cuda_graphs(model)
            _model_cache[key] = model
            _model_inference_locks[key] = threading.Lock()
        return model, _model_inference_locks[key]


def _clear_model_cache() -> None:
    """Clear warmed NeMo models for tests and controlled runtime reloads."""

    with _model_cache_lock:
        _model_cache.clear()
        _model_inference_locks.clear()


def _disable_cuda_graphs(model: Any) -> None:
    decoding_cfg = getattr(getattr(model, "cfg", None), "decoding", None)
    if decoding_cfg is not None and _set_cuda_graph_config(decoding_cfg):
        change_strategy = getattr(model, "change_decoding_strategy", None)
        if callable(change_strategy):
            try:
                change_strategy(decoding_cfg, verbose=False)
            except TypeError:
                change_strategy(decoding_cfg)
            except Exception:
                pass

    decoding = getattr(model, "decoding", None)
    inner_decoding = getattr(decoding, "decoding", None)
    candidates = [
        decoding,
        inner_decoding,
        getattr(decoding, "decoding_computer", None),
        getattr(inner_decoding, "decoding_computer", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        force_mode = getattr(candidate, "force_cuda_graphs_mode", None)
        if callable(force_mode):
            try:
                force_mode("no_graphs")
                return
            except Exception:
                pass
        disable = getattr(candidate, "disable_cuda_graphs", None)
        if callable(disable):
            try:
                disable()
                return
            except Exception:
                pass


def _set_cuda_graph_config(decoding_cfg: Any) -> bool:
    try:
        greedy_cfg = decoding_cfg.get("greedy")
    except Exception:
        greedy_cfg = getattr(decoding_cfg, "greedy", None)
    if greedy_cfg is None:
        return False

    try:
        from omegaconf import open_dict

        with open_dict(greedy_cfg):
            greedy_cfg.use_cuda_graph_decoder = False
        return True
    except ImportError:
        pass
    except Exception:
        pass

    try:
        greedy_cfg["use_cuda_graph_decoder"] = False
        return True
    except Exception:
        try:
            setattr(greedy_cfg, "use_cuda_graph_decoder", False)
            return True
        except Exception:
            return False
