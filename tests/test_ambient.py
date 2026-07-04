from __future__ import annotations

import math
import wave
from array import array
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from atlas_voice.ambient import (
    classify_ambient_utterance,
    process_ambient_file,
    validate_microphone_asr,
    vad_segments,
)
from atlas_voice.cli import build_parser
from atlas_voice.config import Settings
from atlas_voice.database import Database


class AmbientClassificationTests(unittest.TestCase):
    def test_rule_based_classifier_detects_assistant_intent_without_model_call(self) -> None:
        classification = classify_ambient_utterance(
            "Atlas, remind me to send the launch note tomorrow.",
            mode="ambient",
        )

        self.assertTrue(classification.is_directed_to_assistant)
        self.assertEqual(classification.sensitivity, "personal")
        self.assertGreaterEqual(classification.confidence, 0.8)

    def test_rule_based_classifier_keeps_meeting_chatter_out_of_direct_voice_path(self) -> None:
        classification = classify_ambient_utterance(
            "The roadmap risk is GPU memory pressure during demos.",
            mode="meeting",
        )

        self.assertFalse(classification.is_directed_to_assistant)
        self.assertEqual(classification.sensitivity, "shared_meeting")

    def test_rule_based_classifier_escalates_private_sensitive_content(self) -> None:
        classification = classify_ambient_utterance(
            "My password is hunter two and my social security number is in the file.",
            mode="ambient",
        )

        self.assertFalse(classification.is_directed_to_assistant)
        self.assertEqual(classification.sensitivity, "private_sensitive")


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
            self.assertEqual(utterances[0]["is_directed_to_assistant"], 0)
            self.assertEqual(utterances[0]["sensitivity"], "shared_meeting")
            self.assertEqual(model_run["task"], "ambient_transcribe")
            self.assertFalse((settings.artifacts_dir / "ambient" / result.session_id).exists())

    def test_process_ambient_file_retains_audio_for_configured_window(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "meeting.wav"
            _write_test_wav(audio_path)
            settings = replace(_settings(root), ambient_raw_audio_retention_days=1)
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()

            result = process_ambient_file(
                audio_path,
                settings,
                db,
                mode="meeting",
                retain_audio=None,
                vad_threshold=1000,
                min_speech_seconds=0.2,
            )

            session = db.get_ambient_session(result.session_id)

            self.assertEqual(session["retention_policy"], "retain_audio_window")
            self.assertTrue((settings.artifacts_dir / "ambient" / result.session_id).exists())

    def test_process_ambient_file_uses_low_latency_asr_fallback_provider(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "meeting.wav"
            _write_test_wav(audio_path)
            settings = replace(
                _settings(root),
                stub_mode=False,
                asr_provider="hyprwhspr",
                hyprwhspr_cli=str(root / "missing-hyprwhspr"),
                hyprwhspr_endpoint=None,
                realtime_asr_fallback_provider="faster-whisper",
            )
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()

            with mock.patch(
                "atlas_voice.providers.asr.transcribe_faster_whisper",
                return_value={
                    "provider": "faster-whisper",
                    "model": "large-v3-turbo",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 0.8,
                            "text": "fallback transcript",
                        }
                    ],
                },
            ) as fallback_mock:
                result = process_ambient_file(
                    audio_path,
                    settings,
                    db,
                    mode="meeting",
                    retain_audio=False,
                    vad_threshold=1000,
                    min_speech_seconds=0.2,
                )

            utterance = db.list_utterances(result.session_id)[0]
            model_run = db.list_model_runs()[0]
            self.assertEqual(utterance["text"], "fallback transcript")
            self.assertEqual(utterance["source_provider"], "faster-whisper")
            self.assertEqual(model_run["provider"], "faster-whisper")
            self.assertEqual(model_run["model"], "large-v3-turbo")
            fallback_mock.assert_called_once()


    def test_process_ambient_file_stores_speaker_from_speaker_aware_asr(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "meeting.wav"
            _write_test_wav(audio_path)
            settings = replace(_settings(root), stub_mode=False, asr_provider="vibevoice")
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()

            with mock.patch(
                "atlas_voice.providers.asr.transcribe_audio",
                return_value={
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 0.8,
                            "speaker": "SPEAKER_02",
                            "text": "speaker aware transcript",
                        }
                    ]
                },
            ):
                result = process_ambient_file(
                    audio_path,
                    settings,
                    db,
                    mode="meeting",
                    retain_audio=False,
                    vad_threshold=1000,
                    min_speech_seconds=0.2,
                )

            utterance = db.list_utterances(result.session_id)[0]
            self.assertEqual(utterance["text"], "speaker aware transcript")
            self.assertEqual(utterance["speaker"], "SPEAKER_02")
            self.assertFalse(utterance["is_directed_to_assistant"])

    def test_process_ambient_file_stores_speaker_from_transcript_diarization(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "meeting.wav"
            _write_test_wav(audio_path)
            settings = replace(_settings(root), stub_mode=False, asr_provider="vibevoice")
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()

            with mock.patch(
                "atlas_voice.providers.asr.transcribe_audio",
                return_value={
                    "segments": [{"start": 0.0, "end": 0.8, "text": "diarized transcript"}],
                    "diarization": [{"start": 0.0, "end": 0.8, "speaker": "SPEAKER_03"}],
                },
            ):
                result = process_ambient_file(
                    audio_path,
                    settings,
                    db,
                    mode="meeting",
                    retain_audio=False,
                    vad_threshold=1000,
                    min_speech_seconds=0.2,
                )

            utterance = db.list_utterances(result.session_id)[0]
            self.assertEqual(utterance["text"], "diarized transcript")
            self.assertEqual(utterance["speaker"], "SPEAKER_03")

    def test_validate_microphone_asr_captures_brio_and_returns_transcript(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = replace(_settings(root), stub_mode=False)
            audio_path = root / "data" / "artifacts" / "mic-validation" / "brio-validation.wav"

            def fake_capture(path: Path, *, device: str, seconds: float) -> Path:
                self.assertEqual(device, "plughw:2,0")
                self.assertEqual(seconds, 2.0)
                _write_test_wav(path)
                return path

            with (
                mock.patch("atlas_voice.ambient.capture_microphone_chunk", side_effect=fake_capture),
                mock.patch(
                    "atlas_voice.providers.asr.transcribe_audio",
                    return_value={"text": "brio validation transcript", "segments": []},
                ) as transcribe_mock,
            ):
                result = validate_microphone_asr(
                    settings,
                    device="plughw:2,0",
                    seconds=2.0,
                    output_path=audio_path,
                )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.device, "plughw:2,0")
        self.assertEqual(result.transcript, "brio validation transcript")
        self.assertEqual(result.provider, "whisperx")
        self.assertEqual(result.audio_path, audio_path)
        transcribe_mock.assert_called_once()

    def test_validate_microphone_asr_requires_real_asr_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(Path(tmp))

            with self.assertRaisesRegex(RuntimeError, "real ASR"):
                validate_microphone_asr(settings, device="plughw:2,0")

    def test_validate_brio_cli_exposes_device_and_real_asr_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args([
            "validate-brio",
            "--device",
            "plughw:2,0",
            "--seconds",
            "2",
            "--min-transcript-chars",
            "4",
        ])

        self.assertEqual(args.device, "plughw:2,0")
        self.assertEqual(args.seconds, 2)
        self.assertEqual(args.min_transcript_chars, 4)
        self.assertFalse(args.allow_stub)

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
