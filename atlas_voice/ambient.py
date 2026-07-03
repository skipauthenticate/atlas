from __future__ import annotations

import math
import shutil
import subprocess
import sys
import time
import wave
from array import array
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from .audio import normalize_audio
from .config import Settings
from .database import Database
from .realtime import transcript_text
from .storage import is_audio_file, safe_filename


AMBIENT_MODES = {"ambient", "meeting", "direct", "private", "paused"}


@dataclass(frozen=True)
class VoiceSegment:
    start: float
    end: float
    peak_rms: float


@dataclass(frozen=True)
class AmbientResult:
    session_id: str | None
    mode: str
    source: str
    status: str
    segment_count: int = 0
    utterance_count: int = 0


def process_ambient_file(
    source_path: Path,
    settings: Settings,
    db: Database,
    *,
    mode: str = "ambient",
    retain_audio: bool | None = None,
    vad_threshold: float | None = None,
    min_speech_seconds: float | None = None,
) -> AmbientResult:
    mode = _clean_mode(mode)
    if mode in {"private", "paused"}:
        db.log_privacy_event(
            "ambient.skipped",
            f"Ambient mode {mode} skipped audio processing.",
            metadata={"source": str(source_path), "mode": mode},
        )
        return AmbientResult(None, mode, str(source_path), mode)

    source_path = source_path.expanduser().resolve()
    if not is_audio_file(source_path):
        raise ValueError(f"Unsupported ambient audio source: {source_path}")

    retain = settings.ambient_retain_audio if retain_audio is None else retain_audio
    session_id = db.create_ambient_session(
        mode=mode,
        source=str(source_path),
        retention_policy="retain_audio" if retain else "transcript_only",
        title=source_path.stem,
    )
    artifact_dir = settings.artifacts_dir / "ambient" / session_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = artifact_dir / f"{safe_filename(source_path.stem)}.normalized.wav"

    try:
        _prepare_normalized_audio(source_path, normalized_path)
        segments = vad_segments(
            normalized_path,
            energy_threshold=vad_threshold or settings.ambient_vad_threshold,
            min_speech_seconds=min_speech_seconds or settings.ambient_min_speech_seconds,
        )
        utterance_count = 0
        for index, segment in enumerate(segments, start=1):
            segment_path = artifact_dir / f"segment-{index:04d}.wav"
            write_wav_segment(normalized_path, segment_path, segment.start, segment.end)
            text = _transcribe_ambient_segment(segment_path, settings, db, session_id, index)
            if not text:
                continue
            db.add_utterance(
                session_id=session_id,
                text=text,
                start=segment.start,
                end=segment.end,
                source_provider="stub" if settings.stub_mode else settings.asr_provider,
                sensitivity=_sensitivity_for_mode(mode),
            )
            utterance_count += 1
        db.end_ambient_session(session_id, status="done")
        return AmbientResult(
            session_id,
            mode,
            str(source_path),
            "done",
            segment_count=len(segments),
            utterance_count=utterance_count,
        )
    except Exception:
        db.end_ambient_session(session_id, status="failed")
        raise
    finally:
        if not retain:
            with suppress(FileNotFoundError):
                shutil.rmtree(artifact_dir)


def process_ambient_path(
    source_path: Path,
    settings: Settings,
    db: Database,
    *,
    mode: str = "ambient",
    retain_audio: bool | None = None,
    vad_threshold: float | None = None,
    min_speech_seconds: float | None = None,
) -> list[AmbientResult]:
    source_path = source_path.expanduser().resolve()
    if source_path.is_dir():
        results: list[AmbientResult] = []
        for path in sorted(source_path.iterdir()):
            if is_audio_file(path):
                results.append(
                    process_ambient_file(
                        path,
                        settings,
                        db,
                        mode=mode,
                        retain_audio=retain_audio,
                        vad_threshold=vad_threshold,
                        min_speech_seconds=min_speech_seconds,
                    )
                )
        return results
    return [
        process_ambient_file(
            source_path,
            settings,
            db,
            mode=mode,
            retain_audio=retain_audio,
            vad_threshold=vad_threshold,
            min_speech_seconds=min_speech_seconds,
        )
    ]


def capture_microphone_chunk(
    output_path: Path,
    *,
    device: str = "default",
    seconds: float = 15.0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "alsa",
        "-i",
        device,
        "-t",
        str(seconds),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def process_microphone_once(
    settings: Settings,
    db: Database,
    *,
    mode: str = "ambient",
    seconds: float | None = None,
    device: str | None = None,
    retain_audio: bool | None = None,
) -> AmbientResult:
    capture_dir = settings.artifacts_dir / "ambient-capture"
    capture_path = capture_dir / f"mic-{int(time.time())}.wav"
    capture_microphone_chunk(
        capture_path,
        device=device or settings.ambient_mic_device,
        seconds=seconds or settings.ambient_chunk_seconds,
    )
    try:
        return process_ambient_file(
            capture_path,
            settings,
            db,
            mode=mode,
            retain_audio=retain_audio,
        )
    finally:
        if not (settings.ambient_retain_audio if retain_audio is None else retain_audio):
            with suppress(FileNotFoundError):
                capture_path.unlink()


def vad_segments(
    wav_path: Path,
    *,
    energy_threshold: float = 500.0,
    frame_ms: int = 30,
    min_speech_seconds: float = 0.4,
    merge_gap_seconds: float = 0.25,
    padding_seconds: float = 0.1,
) -> list[VoiceSegment]:
    with wave.open(str(wav_path), "rb") as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        frame_count = audio.getnframes()
        frames = audio.readframes(frame_count)

    if channels != 1 or sample_width != 2:
        raise ValueError("VAD requires 16-bit mono WAV audio")

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()

    frame_size = max(int(sample_rate * frame_ms / 1000), 1)
    voiced: list[tuple[float, float, float]] = []
    for start_sample in range(0, len(samples), frame_size):
        frame = samples[start_sample : start_sample + frame_size]
        if not frame:
            continue
        rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame))
        if rms >= energy_threshold:
            start = start_sample / sample_rate
            end = min((start_sample + len(frame)) / sample_rate, len(samples) / sample_rate)
            voiced.append((start, end, rms))

    if not voiced:
        return []

    merged: list[VoiceSegment] = []
    current_start, current_end, current_peak = voiced[0]
    for start, end, rms in voiced[1:]:
        if start - current_end <= merge_gap_seconds:
            current_end = end
            current_peak = max(current_peak, rms)
            continue
        _append_segment(
            merged,
            current_start,
            current_end,
            current_peak,
            total_seconds=len(samples) / sample_rate,
            padding_seconds=padding_seconds,
            min_speech_seconds=min_speech_seconds,
        )
        current_start, current_end, current_peak = start, end, rms
    _append_segment(
        merged,
        current_start,
        current_end,
        current_peak,
        total_seconds=len(samples) / sample_rate,
        padding_seconds=padding_seconds,
        min_speech_seconds=min_speech_seconds,
    )
    return merged


def write_wav_segment(source_path: Path, output_path: Path, start: float, end: float) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source_path), "rb") as source:
        params = source.getparams()
        sample_rate = source.getframerate()
        start_frame = max(int(start * sample_rate), 0)
        end_frame = min(int(end * sample_rate), source.getnframes())
        source.setpos(start_frame)
        data = source.readframes(max(end_frame - start_frame, 0))

    with wave.open(str(output_path), "wb") as output:
        output.setparams(params)
        output.writeframes(data)
    return output_path


def _append_segment(
    segments: list[VoiceSegment],
    start: float,
    end: float,
    peak: float,
    *,
    total_seconds: float,
    padding_seconds: float,
    min_speech_seconds: float,
) -> None:
    if end - start < min_speech_seconds:
        return
    segments.append(
        VoiceSegment(
            start=max(start - padding_seconds, 0.0),
            end=min(end + padding_seconds, total_seconds),
            peak_rms=peak,
        )
    )


def _prepare_normalized_audio(source_path: Path, output_path: Path) -> None:
    if _is_normalized_wav(source_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
        return
    normalize_audio(source_path, output_path)


def _is_normalized_wav(path: Path) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "rb") as audio:
            return (
                audio.getnchannels() == 1
                and audio.getsampwidth() == 2
                and audio.getframerate() == 16000
            )
    except wave.Error:
        return False


def _transcribe_ambient_segment(
    segment_path: Path,
    settings: Settings,
    db: Database,
    session_id: str,
    index: int,
) -> str:
    started = time.perf_counter()
    provider = "stub" if settings.stub_mode else settings.asr_provider
    model = _asr_model(settings)
    input_ref = f"ambient_segment:{session_id}:{index}"
    try:
        if settings.stub_mode:
            text = f"Ambient segment {index} captured."
        else:
            from .providers.asr import transcribe_audio

            text = transcript_text(transcribe_audio(segment_path, settings))
        db.log_model_run(
            provider=provider,
            model=model,
            task="ambient_transcribe",
            input_ref=input_ref,
            latency_ms=_elapsed_ms(started),
        )
        return text
    except Exception as exc:
        db.log_model_run(
            provider=provider,
            model=model,
            task="ambient_transcribe",
            input_ref=input_ref,
            latency_ms=_elapsed_ms(started),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def _asr_model(settings: Settings) -> str:
    if settings.asr_provider == "whisperx":
        return settings.whisperx_model
    if settings.asr_model:
        return settings.asr_model
    if settings.asr_provider == "vibevoice":
        return settings.vibevoice_model
    defaults = {
        "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
        "canary": "nvidia/canary-1b-v2",
    }
    return defaults.get(settings.asr_provider, settings.whisperx_model)


def _clean_mode(mode: str) -> str:
    cleaned = mode.strip().lower()
    if cleaned not in AMBIENT_MODES:
        raise ValueError(f"Unknown ambient mode: {mode}")
    return cleaned


def _sensitivity_for_mode(mode: str) -> str | None:
    if mode == "meeting":
        return "shared_meeting"
    if mode == "direct":
        return "directed_to_assistant"
    return None


def _elapsed_ms(started: float) -> int:
    return max(int((time.perf_counter() - started) * 1000), 0)
