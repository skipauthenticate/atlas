from __future__ import annotations

from pathlib import Path
from typing import Any

from atlas_voice.config import Settings


def diarize_audio(audio_path: Path, settings: Settings) -> list[dict[str, Any]]:
    if settings.stub_mode or settings.allow_single_speaker_fallback:
        return [{"start": 0.0, "end": 3600.0, "speaker": "SPEAKER_00"}]

    if not settings.hf_token:
        raise RuntimeError(
            "HF_TOKEN is required for pyannote diarization. Run ./setup.sh or set HF_TOKEN. "
            "Set ATLAS_VOICE_ALLOW_SINGLE_SPEAKER_FALLBACK=true to process audio "
            "temporarily without speaker diarization."
        )
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Install the worker extras or run through Docker."
        ) from exc

    try:
        pipeline = Pipeline.from_pretrained(settings.pyannote_model, token=settings.hf_token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            settings.pyannote_model, use_auth_token=settings.hf_token
        )
    try:
        import torch

        if settings.whisperx_device != "cpu":
            pipeline.to(torch.device(settings.whisperx_device))
    except Exception:
        pass

    diarization = _diarization_annotation(pipeline(_pipeline_audio_input(audio_path)))
    turns: list[dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    return turns


def _diarization_annotation(diarization: Any) -> Any:
    return (
        getattr(diarization, "exclusive_speaker_diarization", None)
        or getattr(diarization, "speaker_diarization", None)
        or diarization
    )


def _pipeline_audio_input(audio_path: Path) -> str | dict[str, Any]:
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception:
        pass

    try:
        import soundfile as sf
        import torch

        audio, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
        waveform = torch.from_numpy(audio.T)
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception:
        return str(audio_path)
