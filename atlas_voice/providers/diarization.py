from __future__ import annotations

from pathlib import Path
from typing import Any

from atlas_voice.config import Settings
from atlas_voice.providers.pyannote_provider import diarize_audio as diarize_pyannote
from atlas_voice.providers.transcript_utils import (
    audio_duration_seconds,
    diarization_from_transcript,
)


def diarize_audio(
    audio_path: Path, settings: Settings, *, transcript: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    provider = settings.diarization_provider.lower()
    if provider == "pyannote":
        return diarize_pyannote(audio_path, settings)
    if provider == "transcript":
        if transcript is None:
            raise RuntimeError("Transcript diarization requires transcript metadata.")
        turns = diarization_from_transcript(transcript)
        if turns:
            return turns
        raise RuntimeError(
            "Transcript does not contain speaker turns. Use pyannote diarization or a "
            "combined ASR provider such as vibevoice."
        )
    if provider in {"none", "single", "single-speaker"}:
        return [
            {
                "start": 0.0,
                "end": audio_duration_seconds(audio_path, default=0.0),
                "speaker": "SPEAKER_00",
            }
        ]
    raise RuntimeError(
        f"Unknown ATLAS_VOICE_DIARIZATION_PROVIDER={settings.diarization_provider!r}. "
        "Use pyannote, transcript, or none."
    )
