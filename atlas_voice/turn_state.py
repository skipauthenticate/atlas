from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RealtimeTurnState:
    sample_rate: int
    channels: int
    active_response_id: str | None = None
    response_in_progress: bool = False
    cancel_requested: bool = False
    _audio_buffer: bytearray = field(default_factory=bytearray)
    _pending_text: str | None = None

    @property
    def buffered_audio_bytes(self) -> int:
        return len(self._audio_buffer)

    @property
    def has_pending_text(self) -> bool:
        return bool(self._pending_text)

    def append_audio(self, payload: bytes) -> int:
        self._audio_buffer.extend(payload)
        return len(self._audio_buffer)

    def clear_audio(self) -> None:
        self._audio_buffer.clear()

    def commit_audio(self) -> bytes:
        payload = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        return payload

    def update_audio_format(self, event: dict[str, Any]) -> None:
        sample_rate = event.get("input_audio_sample_rate") or event.get("sample_rate")
        channels = event.get("channels")
        if sample_rate:
            self.sample_rate = max(int(sample_rate), 1)
        if channels:
            self.channels = max(int(channels), 1)

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
