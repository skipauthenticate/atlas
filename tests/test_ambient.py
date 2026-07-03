from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from atlas_voice.ambient import process_ambient_file, vad_segments
from atlas_voice.config import Settings
from atlas_voice.database import Database


class AmbientTests(unittest.TestCase):
    def test_vad_segments_detect_speech_region(self) -> None:
        with TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "speech.wav"
            _write_test_wav(audio_path)

            segments = vad_segments(
                audio_path,
                energy_threshold=1000,
                min_speech_seconds=0.2,
            )

        self.assertEqual(len(segments), 1)
        self.assertLess(segments[0].start, 0.25)
        self.assertGreater(segments[0].end, 0.9)

    def test_process_ambient_file_stores_session_utterance_and_model_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "meeting.wav"
            _write_test_wav(audio_path)
            settings = _settings(root)
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()

            result = process_ambient_file(
                audio_path,
                settings,
                db,
                mode="meeting",
                retain_audio=False,
                vad_threshold=1000,
                min_speech_seconds=0.2,
            )

            session = db.get_ambient_session(result.session_id)
            utterances = db.list_utterances(result.session_id)
            model_run = db.list_model_runs()[0]

            self.assertEqual(result.status, "done")
            self.assertEqual(result.segment_count, 1)
            self.assertEqual(result.utterance_count, 1)
            self.assertEqual(session["mode"], "meeting")
            self.assertEqual(session["retention_policy"], "transcript_only")
            self.assertEqual(utterances[0]["text"], "Ambient segment 1 captured.")
            self.assertEqual(utterances[0]["sensitivity"], "shared_meeting")
            self.assertEqual(model_run["task"], "ambient_transcribe")
            self.assertFalse((settings.artifacts_dir / "ambient" / result.session_id).exists())

    def test_private_mode_skips_audio_and_logs_privacy_event(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "private.wav"
            _write_test_wav(audio_path)
            settings = _settings(root)
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()

            result = process_ambient_file(audio_path, settings, db, mode="private")

            self.assertIsNone(result.session_id)
            self.assertEqual(result.status, "private")
            self.assertEqual(db.list_privacy_events()[0]["event_type"], "ambient.skipped")


def _write_test_wav(path: Path) -> None:
    sample_rate = 16000
    samples = array("h")
    samples.extend([0] * int(sample_rate * 0.2))
    for index in range(int(sample_rate * 0.8)):
        value = int(6000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        samples.append(value)
    samples.extend([0] * int(sample_rate * 0.2))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())


def _settings(root: Path) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8787,
        data_dir=root / "data",
        models_dir=root / "models",
        hf_cache_dir=root / "cache" / "huggingface",
        whisperx_model="tiny.en",
        whisperx_device="cpu",
        whisperx_compute_type="int8",
        pyannote_model="pyannote/speaker-diarization-community-1",
        hf_token=None,
        llm_base_url="http://127.0.0.1:8080/v1/chat/completions",
        llm_model="qwen-local",
        llm_temperature=0.2,
        llm_max_tokens=1200,
        stub_mode=True,
    )


if __name__ == "__main__":
    unittest.main()
