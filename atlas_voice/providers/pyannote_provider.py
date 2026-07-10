from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import threading
from typing import Any
import wave

from atlas_voice.config import Settings


DIARIZATION_CHUNK_SECONDS = 20 * 60.0
DIARIZATION_CHUNK_OVERLAP_SECONDS = 30.0
MIN_DIARIZATION_TURN_SECONDS = 0.05
MERGE_ADJACENT_TURN_GAP_SECONDS = 0.25
_pipeline_cache_lock = threading.Lock()
_pipeline_cache: dict[tuple[object, ...], Any] = {}
_pipeline_inference_locks: dict[tuple[object, ...], threading.Lock] = {}


def diarize_audio(
    audio_path: Path,
    settings: Settings,
    *,
    expected_speakers: int | None = None,
) -> list[dict[str, Any]]:
    speaker_kwargs = _speaker_hint_kwargs(expected_speakers)
    if settings.stub_mode:
        return _single_speaker_turns(audio_path)

    try:
        return _diarize_with_pyannote(
            audio_path,
            settings,
            speaker_kwargs=speaker_kwargs,
        )
    except Exception:
        if settings.allow_single_speaker_fallback:
            return _single_speaker_turns(audio_path)
        raise


def _single_speaker_turns(audio_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "start": 0.0,
            "end": _audio_duration_seconds(audio_path, default=3600.0),
            "speaker": "SPEAKER_00",
        }
    ]


def _diarize_with_pyannote(
    audio_path: Path,
    settings: Settings,
    *,
    speaker_kwargs: dict[str, int],
) -> list[dict[str, Any]]:
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

    pipeline, inference_lock = _warm_pipeline(settings, Pipeline)
    duration = _audio_duration_seconds(audio_path, default=0.0)
    with inference_lock:
        if duration > DIARIZATION_CHUNK_SECONDS:
            return _diarize_audio_chunks(
                pipeline,
                audio_path,
                duration,
                speaker_kwargs=speaker_kwargs,
            )

        diarization = _diarization_annotation(
            _apply_pipeline(
                pipeline,
                _pipeline_audio_input(audio_path),
                speaker_kwargs,
            )
        )
        return _annotation_turns(diarization)


def _warm_pipeline(settings: Settings, pipeline_class: Any) -> tuple[Any, threading.Lock]:
    key: tuple[object, ...] = (
        pipeline_class,
        settings.pyannote_model,
        settings.whisperx_device,
    )
    with _pipeline_cache_lock:
        pipeline = _pipeline_cache.get(key)
        if pipeline is None:
            try:
                pipeline = pipeline_class.from_pretrained(
                    settings.pyannote_model,
                    token=settings.hf_token,
                )
            except TypeError:
                pipeline = pipeline_class.from_pretrained(
                    settings.pyannote_model,
                    use_auth_token=settings.hf_token,
                )
            if settings.whisperx_device != "cpu":
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "PyTorch CUDA is not available, but "
                        f"WHISPERX_DEVICE={settings.whisperx_device!r}."
                    )
                pipeline.to(torch.device(settings.whisperx_device))
            _pipeline_cache[key] = pipeline
            _pipeline_inference_locks[key] = threading.Lock()
        return pipeline, _pipeline_inference_locks[key]


def _speaker_hint_kwargs(expected_speakers: int | None) -> dict[str, int]:
    if expected_speakers is None:
        return {}
    if isinstance(expected_speakers, bool) or not isinstance(expected_speakers, int):
        raise ValueError("expected_speakers must be a positive integer")
    if expected_speakers < 1:
        raise ValueError("expected_speakers must be a positive integer")

    # This is a hint about the planned main voices, not a hard promise. Leave
    # room for a missed attendee or incidental voice so diarization can still
    # surface additional speakers instead of forcing them into a known person.
    extra_headroom = max(4, expected_speakers)
    return {
        "min_speakers": max(1, expected_speakers - 1),
        "max_speakers": expected_speakers + extra_headroom,
    }


def _apply_pipeline(
    pipeline: Any,
    audio_input: str | dict[str, Any],
    speaker_kwargs: dict[str, int],
) -> Any:
    if speaker_kwargs:
        return pipeline(audio_input, **speaker_kwargs)
    return pipeline(audio_input)


def _clear_pipeline_cache() -> None:
    """Clear process-local pipeline caches. Intended for tests and controlled reloads."""

    with _pipeline_cache_lock:
        _pipeline_cache.clear()
        _pipeline_inference_locks.clear()


def _diarize_audio_chunks(
    pipeline: Any,
    audio_path: Path,
    duration: float,
    *,
    speaker_kwargs: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    speaker_kwargs = speaker_kwargs or {}
    turns: list[dict[str, Any]] = []
    next_speaker_index = 0
    keep_start = 0.0

    while keep_start < duration:
        keep_end = min(duration, keep_start + DIARIZATION_CHUNK_SECONDS)
        source_start = max(0.0, keep_start - DIARIZATION_CHUNK_OVERLAP_SECONDS)
        source_end = min(duration, keep_end + DIARIZATION_CHUNK_OVERLAP_SECONDS)

        audio_input = _pipeline_audio_input(audio_path, start=source_start, end=source_end)
        diarization = _diarization_annotation(
            _apply_pipeline(pipeline, audio_input, speaker_kwargs)
        )
        chunk_turns = _annotation_turns(diarization, offset=source_start)
        speaker_map, next_speaker_index = _map_chunk_speakers(
            chunk_turns,
            turns,
            next_speaker_index,
            overlap_start=source_start,
            overlap_end=keep_start,
        )

        for turn in chunk_turns:
            start = max(float(turn["start"]), keep_start)
            end = min(float(turn["end"]), keep_end)
            if end - start < MIN_DIARIZATION_TURN_SECONDS:
                continue
            turns.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker_map[turn["speaker"]],
                }
            )

        keep_start = keep_end

    return _merge_adjacent_turns(turns)


def _map_chunk_speakers(
    chunk_turns: list[dict[str, Any]],
    reference_turns: list[dict[str, Any]],
    next_speaker_index: int,
    *,
    overlap_start: float,
    overlap_end: float,
) -> tuple[dict[str, str], int]:
    local_speakers = _speaker_order(chunk_turns)
    scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    if overlap_end > overlap_start:
        for chunk_turn in chunk_turns:
            chunk_start = max(float(chunk_turn["start"]), overlap_start)
            chunk_end = min(float(chunk_turn["end"]), overlap_end)
            if chunk_end <= chunk_start:
                continue
            for reference_turn in reference_turns:
                reference_start = max(float(reference_turn["start"]), overlap_start)
                reference_end = min(float(reference_turn["end"]), overlap_end)
                seconds = _overlap_seconds(
                    chunk_start,
                    chunk_end,
                    reference_start,
                    reference_end,
                )
                if seconds > 0:
                    scores[chunk_turn["speaker"]][reference_turn["speaker"]] += seconds

    mapping: dict[str, str] = {}
    used_global_speakers: set[str] = set()
    candidates: list[tuple[float, str, str]] = []
    for local_speaker, global_scores in scores.items():
        for global_speaker, seconds in global_scores.items():
            candidates.append((seconds, local_speaker, global_speaker))

    for _seconds, local_speaker, global_speaker in sorted(candidates, reverse=True):
        if local_speaker in mapping or global_speaker in used_global_speakers:
            continue
        mapping[local_speaker] = global_speaker
        used_global_speakers.add(global_speaker)

    for local_speaker in local_speakers:
        if local_speaker in mapping:
            continue
        global_speaker = _speaker_label(next_speaker_index)
        next_speaker_index += 1
        mapping[local_speaker] = global_speaker

    return mapping, next_speaker_index


def _speaker_order(turns: list[dict[str, Any]]) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        speaker = str(turn["speaker"])
        if speaker not in seen:
            seen.add(speaker)
            speakers.append(speaker)
    return speakers


def _speaker_label(index: int) -> str:
    return f"SPEAKER_{index:02d}"


def _overlap_seconds(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _merge_adjacent_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for turn in sorted(turns, key=lambda item: (float(item["start"]), float(item["end"]))):
        if not merged:
            merged.append(dict(turn))
            continue
        previous = merged[-1]
        if (
            previous["speaker"] == turn["speaker"]
            and float(turn["start"]) <= float(previous["end"]) + MERGE_ADJACENT_TURN_GAP_SECONDS
        ):
            previous["end"] = max(float(previous["end"]), float(turn["end"]))
            continue
        merged.append(dict(turn))
    return merged


def _diarization_annotation(diarization: Any) -> Any:
    return (
        getattr(diarization, "exclusive_speaker_diarization", None)
        or getattr(diarization, "speaker_diarization", None)
        or diarization
    )


def _annotation_turns(diarization: Any, offset: float = 0.0) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = float(turn.start) + offset
        end = float(turn.end) + offset
        if end - start < MIN_DIARIZATION_TURN_SECONDS:
            continue
        turns.append(
            {
                "start": start,
                "end": end,
                "speaker": str(speaker),
            }
        )
    return turns


def _audio_duration_seconds(audio_path: Path, default: float) -> float:
    try:
        with wave.open(str(audio_path), "rb") as audio:
            frame_rate = audio.getframerate()
            if frame_rate <= 0:
                return default
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


def _pipeline_audio_input(
    audio_path: Path, start: float | None = None, end: float | None = None
) -> str | dict[str, Any]:
    try:
        import torchaudio

        if start is None and end is None:
            waveform, sample_rate = torchaudio.load(str(audio_path))
        else:
            info = torchaudio.info(str(audio_path))
            sample_rate = int(info.sample_rate)
            frame_offset = max(0, int((start or 0.0) * sample_rate))
            num_frames = -1
            if end is not None:
                frame_end = max(frame_offset + 1, int(end * sample_rate))
                num_frames = frame_end - frame_offset
            waveform, sample_rate = torchaudio.load(
                str(audio_path),
                frame_offset=frame_offset,
                num_frames=num_frames,
            )
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception:
        pass

    try:
        import soundfile as sf
        import torch

        if start is None and end is None:
            audio, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
        else:
            with sf.SoundFile(str(audio_path)) as audio_file:
                sample_rate = audio_file.samplerate
                frame_offset = max(0, int((start or 0.0) * sample_rate))
                audio_file.seek(frame_offset)
                frames = -1
                if end is not None:
                    frame_end = max(frame_offset + 1, int(end * sample_rate))
                    frames = frame_end - frame_offset
                audio = audio_file.read(frames=frames, always_2d=True, dtype="float32")
        waveform = torch.from_numpy(audio.T)
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception:
        if start is not None or end is not None:
            raise RuntimeError(f"Unable to load audio chunk from {audio_path}")
        return str(audio_path)
