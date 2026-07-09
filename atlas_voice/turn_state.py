from __future__ import annotations

import math
import sys
from array import array
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RealtimeVadResult:
    analyzed: bool
    speech_started: bool = False
    end_of_turn: bool = False
    speech_ms: int = 0
    silence_ms: int = 0
    rms: float = 0.0


@dataclass
class RealtimeTurnState:
    sample_rate: int
    channels: int
    active_response_id: str | None = None
    response_in_progress: bool = False
    cancel_requested: bool = False
    audio_media_type: str | None = None
    vad_speech_started: bool = False
    vad_speech_ms: int = 0
    vad_silence_ms: int = 0
    _audio_buffer: bytearray = field(default_factory=bytearray)
    _pending_text: str | None = None
    _streaming_transcript: str | None = None

    @property
    def buffered_audio_bytes(self) -> int:
        return len(self._audio_buffer)

    @property
    def has_pending_text(self) -> bool:
        return bool(self._pending_text)

    @property
    def pending_streaming_transcript(self) -> str | None:
        return self._streaming_transcript

    def append_audio(self, payload: bytes, *, media_type: str | None = None) -> int:
        if media_type:
            self.audio_media_type = _clean_media_type(media_type)
        self._audio_buffer.extend(payload)
        return len(self._audio_buffer)

    def retain_recent_pcm(self, duration_ms: int) -> int:
        if duration_ms < 0 or not _is_pcm_audio(self.audio_media_type):
            return len(self._audio_buffer)
        bytes_per_second = max(self.sample_rate, 1) * max(self.channels, 1) * 2
        max_bytes = int(bytes_per_second * duration_ms / 1000)
        if len(self._audio_buffer) > max_bytes:
            if max_bytes:
                del self._audio_buffer[:-max_bytes]
            else:
                self._audio_buffer.clear()
        return len(self._audio_buffer)

    def clear_audio(self) -> None:
        self._audio_buffer.clear()
        self.audio_media_type = None
        self._streaming_transcript = None
        self.reset_realtime_vad()

    def commit_audio(self) -> bytes:
        payload, _media_type = self.commit_audio_with_media_type()
        return payload

    def commit_audio_with_media_type(self) -> tuple[bytes, str | None]:
        payload = bytes(self._audio_buffer)
        media_type = self.audio_media_type
        self._audio_buffer.clear()
        self.audio_media_type = None
        self.reset_realtime_vad()
        return payload, media_type

    def append_streaming_transcript(self, delta: str | None, *, final: bool = False) -> str | None:
        cleaned_delta = str(delta or "")
        if not cleaned_delta.strip():
            return self._streaming_transcript
        current = self._streaming_transcript or ""
        self._streaming_transcript = " ".join(f"{current}{cleaned_delta}".split())
        return self._streaming_transcript

    def pop_streaming_transcript(self) -> str | None:
        text = self._streaming_transcript
        self._streaming_transcript = None
        return text

    def update_realtime_vad(
        self,
        payload: bytes,
        *,
        enabled: bool = True,
        media_type: str | None = None,
        energy_threshold: float = 500.0,
        min_speech_ms: int = 200,
        silence_duration_ms: int = 600,
    ) -> RealtimeVadResult:
        if not enabled or not payload:
            return RealtimeVadResult(analyzed=False)
        if not _is_pcm_audio(media_type):
            return RealtimeVadResult(analyzed=False)
        if self.channels != 1:
            return RealtimeVadResult(analyzed=False)

        samples = array("h")
        sample_bytes = payload[: len(payload) - (len(payload) % 2)]
        if not sample_bytes:
            return RealtimeVadResult(analyzed=False)
        samples.frombytes(sample_bytes)
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return RealtimeVadResult(analyzed=False)

        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        duration_ms = int(round(len(samples) / max(self.sample_rate, 1) * 1000))
        speech_started = False
        end_of_turn = False
        if rms >= energy_threshold:
            self.vad_speech_ms += duration_ms
            self.vad_silence_ms = 0
            if not self.vad_speech_started and self.vad_speech_ms >= min_speech_ms:
                self.vad_speech_started = True
                speech_started = True
        else:
            if self.vad_speech_started:
                self.vad_silence_ms += duration_ms
                if self.vad_silence_ms >= silence_duration_ms:
                    end_of_turn = True

        return RealtimeVadResult(
            analyzed=True,
            speech_started=speech_started,
            end_of_turn=end_of_turn,
            speech_ms=self.vad_speech_ms,
            silence_ms=self.vad_silence_ms,
            rms=rms,
        )

    def reset_realtime_vad(self) -> None:
        self.vad_speech_started = False
        self.vad_speech_ms = 0
        self.vad_silence_ms = 0

    def update_audio_format(self, event: dict[str, Any]) -> None:
        sample_rate = event.get("input_audio_sample_rate") or event.get("sample_rate")
        channels = event.get("channels")
        media_type = event.get("media_type") or event.get("mime_type")
        if sample_rate:
            self.sample_rate = max(int(sample_rate), 1)
        if channels:
            self.channels = max(int(channels), 1)
        if media_type:
            self.audio_media_type = _clean_media_type(str(media_type))

    def set_pending_text(self, text: str | None) -> None:
        normalized = (text or "").strip()
        self._pending_text = normalized or None

    def pop_pending_text(self) -> str | None:
        text = self._pending_text
        self._pending_text = None
        return text

    def begin_response(self, response_id: str) -> None:
        self.active_response_id = response_id
        self.response_in_progress = True
        self.cancel_requested = False

    def cancel_response(self) -> None:
        if self.response_in_progress:
            self.cancel_requested = True

    def complete_response(self) -> None:
        self.active_response_id = None
        self.response_in_progress = False
        self.cancel_requested = False


def _clean_media_type(value: str) -> str:
    return value.split(";", 1)[0].strip().lower() or "application/octet-stream"


def _is_pcm_audio(media_type: str | None) -> bool:
    if not media_type:
        return True
    cleaned = _clean_media_type(str(media_type))
    return cleaned in {"audio/pcm", "audio/raw", "audio/wav", "audio/x-wav"}
