from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
import time
import wave
from array import array
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import normalize_audio
from .config import Settings
from .database import Database
from .realtime import transcript_text
from .storage import is_audio_file, safe_filename


AMBIENT_MODES = {"ambient", "meeting", "direct", "private", "paused"}

_ASSISTANT_WAKE_RE = re.compile(r"\b(?:atlas|hey atlas|assistant)\b", re.IGNORECASE)
_ASSISTANT_COMMAND_RE = re.compile(
    r"\b(?:remind me|set (?:a )?reminder|take (?:a )?note|note this|"
    r"summarize this|help me|what should i|how do i|schedule|add (?:a )?task)\b",
    re.IGNORECASE,
)
_PRIVATE_SENSITIVE_RE = re.compile(
    r"\b(?:password|passcode|social security|ssn|credit card|bank account|"
    r"api key|secret key|access token|private key|medical record|diagnosis|"
    r"therapy|salary|confidential|attorney|legal advice)\b",
    re.IGNORECASE,
)
_PERSONAL_RE = re.compile(
    r"\b(?:remind me|my|personal|doctor|appointment|family|home|todo|"
    r"follow up|remember to|i need to)\b",
    re.IGNORECASE,
)


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


@dataclass(frozen=True)
class AmbientUtteranceClassification:
    is_directed_to_assistant: bool
    sensitivity: str | None
    confidence: float
    reason: str


@dataclass(frozen=True)
class MicrophoneAsrValidationResult:
    status: str
    device: str
    audio_path: Path
    provider: str
    transcript: str
    audio_seconds: float


@dataclass(frozen=True)
class AmbientTranscriptResult:
    text: str
    speaker: str | None = None
    confidence: float | None = None
    provider: str | None = None
    model: str | None = None


def classify_ambient_utterance(text: str, *, mode: str = "ambient") -> AmbientUtteranceClassification:
    """Classify ambient text with a tiny deterministic first pass.

    The always-on path must stay cheap on Jetson, so this intentionally avoids an
    LLM call. The result maps onto existing utterance columns: directed intent is
    stored as ``is_directed_to_assistant`` and privacy routing as ``sensitivity``.
    """

    mode = _clean_mode(mode)
    normalized = " ".join(text.split())
    if not normalized:
        return AmbientUtteranceClassification(
            is_directed_to_assistant=mode == "direct",
            sensitivity=_sensitivity_for_mode(mode),
            confidence=0.4,
            reason="empty",
        )

    has_wake_word = bool(_ASSISTANT_WAKE_RE.search(normalized))
    has_command = bool(_ASSISTANT_COMMAND_RE.search(normalized))
    is_directed = mode == "direct" or has_wake_word or has_command

    sensitivity = _sensitivity_for_mode(mode)
    reason = "mode" if sensitivity else "default"
    confidence = 0.65

    if _PRIVATE_SENSITIVE_RE.search(normalized):
        sensitivity = "private_sensitive"
        reason = "private_sensitive_terms"
        confidence = 0.9
    elif sensitivity is None and _PERSONAL_RE.search(normalized):
        sensitivity = "personal"
        reason = "personal_terms"
        confidence = 0.85
    elif sensitivity is None and is_directed:
        sensitivity = "directed_to_assistant"
        reason = "assistant_intent"
        confidence = 0.8

    if has_wake_word or has_command:
        confidence = max(confidence, 0.8)
        if reason == "default":
            reason = "assistant_intent"

    return AmbientUtteranceClassification(
        is_directed_to_assistant=is_directed,
        sensitivity=sensitivity,
        confidence=confidence,
        reason=reason,
    )


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

    retain = _retain_ambient_audio(settings, retain_audio)
    session_id = db.create_ambient_session(
        mode=mode,
        source=str(source_path),
        retention_policy=_ambient_retention_policy(settings, retain_audio, retain),
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
            transcript = _transcribe_ambient_segment(segment_path, settings, db, session_id, index)
            if not transcript.text:
                continue
            classification = classify_ambient_utterance(transcript.text, mode=mode)
            db.add_utterance(
                session_id=session_id,
                text=transcript.text,
                speaker=transcript.speaker or "user",
                start=segment.start,
                end=segment.end,
                confidence=transcript.confidence,
                source_provider=transcript.provider or ("stub" if settings.stub_mode else settings.asr_provider),
                is_directed_to_assistant=classification.is_directed_to_assistant,
                sensitivity=classification.sensitivity,
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


def _retain_ambient_audio(settings: Settings, retain_audio: bool | None) -> bool:
    if retain_audio is not None:
        return retain_audio
    return settings.ambient_retain_audio or settings.ambient_raw_audio_retention_days > 0


def _ambient_retention_policy(
    settings: Settings,
    retain_audio: bool | None,
    retain: bool,
) -> str:
    if not retain:
        return "transcript_only"
    if retain_audio is None and not settings.ambient_retain_audio:
        return "retain_audio_window"
    return "retain_audio"


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


def validate_microphone_asr(
    settings: Settings,
    *,
    device: str = "plughw:2,0",
    seconds: float = 5.0,
    output_path: Path | None = None,
    allow_stub: bool = False,
    min_transcript_chars: int = 1,
) -> MicrophoneAsrValidationResult:
    if settings.stub_mode and not allow_stub:
        raise RuntimeError(
            "BRIO validation requires real ASR; disable ATLAS_VOICE_STUB_MODE or pass allow_stub."
        )
    settings.ensure_directories()
    capture_path = output_path or settings.artifacts_dir / "mic-validation" / "brio-validation.wav"
    capture_microphone_chunk(capture_path, device=device, seconds=seconds)

    from atlas_voice.providers.asr import transcribe_audio

    transcript = transcribe_audio(capture_path, settings)
    text = transcript_text(transcript)
    if len(text.strip()) < min_transcript_chars:
        raise RuntimeError(
            f"BRIO validation captured audio from {device}, but ASR returned no transcript text."
        )
    return MicrophoneAsrValidationResult(
        status="ok",
        device=device,
        audio_path=capture_path,
        provider=settings.asr_provider,
        transcript=text,
        audio_seconds=seconds,
    )


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
) -> AmbientTranscriptResult:
    started = time.perf_counter()
    provider = "stub" if settings.stub_mode else settings.asr_provider
    model = _asr_model(settings)
    input_ref = f"ambient_segment:{session_id}:{index}"
    try:
        if settings.stub_mode:
            transcript = AmbientTranscriptResult(
                text=f"Ambient segment {index} captured.",
                provider="stub",
                model=model,
            )
        else:
            from .providers.asr import transcribe_audio

            transcript = _ambient_transcript_result(transcribe_audio(segment_path, settings, realtime=True))
        db.log_model_run(
            provider=transcript.provider or provider,
            model=transcript.model or model,
            task="ambient_transcribe",
            input_ref=input_ref,
            latency_ms=_elapsed_ms(started),
        )
        return transcript
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


def _ambient_transcript_result(payload: dict[str, Any]) -> AmbientTranscriptResult:
    text = transcript_text(payload)
    segment = _first_text_segment(payload)
    speaker = _ambient_speaker(payload, segment)
    confidence = _optional_float(
        segment.get("confidence") or segment.get("score") or segment.get("probability")
        if segment
        else None
    )
    provider = _clean_optional_text(payload.get("provider"))
    model = _clean_optional_text(payload.get("model"))
    return AmbientTranscriptResult(
        text=text,
        speaker=speaker,
        confidence=confidence,
        provider=provider,
        model=model,
    )


def _ambient_speaker(payload: dict[str, Any], segment: dict[str, Any] | None) -> str | None:
    speaker = _clean_optional_text(segment.get("speaker") if segment else None)
    if speaker:
        return speaker
    if not segment:
        return None
    return _diarization_speaker_for_segment(
        payload.get("diarization"),
        _optional_float(segment.get("start")),
        _optional_float(segment.get("end")),
    )


def _diarization_speaker_for_segment(
    diarization: Any,
    start: float | None,
    end: float | None,
) -> str | None:
    if not isinstance(diarization, list) or start is None or end is None:
        return None
    best_speaker = None
    best_overlap = 0.0
    for turn in diarization:
        if not isinstance(turn, dict):
            continue
        turn_start = _optional_float(turn.get("start"))
        turn_end = _optional_float(turn.get("end"))
        if turn_start is None or turn_end is None:
            continue
        overlap = max(min(end, turn_end) - max(start, turn_start), 0.0)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = _clean_optional_text(turn.get("speaker"))
    return best_speaker


def _first_text_segment(payload: dict[str, Any]) -> dict[str, Any] | None:
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return None
    for segment in segments:
        if isinstance(segment, dict) and str(segment.get("text") or "").strip():
            return segment
    return None


def _clean_optional_text(value: Any) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _asr_model(settings: Settings) -> str:
    if settings.asr_provider == "whisperx":
        return settings.whisperx_model
    if settings.asr_model:
        return settings.asr_model
    if settings.asr_provider == "vibevoice":
        return settings.vibevoice_model
    defaults = {
        "faster-whisper": "large-v3-turbo",
        "hyprwhspr": "hyprwhspr-local",
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
