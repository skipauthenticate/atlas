from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from atlas_voice.config import Settings
from atlas_voice.database import Database
from atlas_voice.quality import (
    QUALITY_TIER_ORDER,
    normalize_quality_tier,
    quality_profile,
    quality_profiles,
    settings_for_quality,
)
from atlas_voice.resources import ResourceSnapshot, admit_pipeline_step
from atlas_voice.worker import Worker


def _settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "host": "127.0.0.1",
        "port": 8787,
        "data_dir": root / "data",
        "models_dir": root / "models",
        "hf_cache_dir": root / "cache" / "huggingface",
        "whisperx_model": "base-whisperx",
        "whisperx_device": "cpu",
        "whisperx_compute_type": "int8",
        "pyannote_model": "pyannote/test",
        "hf_token": None,
        "llm_base_url": "http://127.0.0.1:8080/v1/chat/completions",
        "llm_model": "test-llm",
        "llm_temperature": 0.0,
        "llm_max_tokens": 100,
        "stub_mode": True,
    }
    values.update(overrides)
    return Settings(**values)


class QualityProfileTests(unittest.TestCase):
    def test_quality_contract_is_ordered_and_defaults_to_torch(self) -> None:
        self.assertEqual(QUALITY_TIER_ORDER, ("light", "torch", "fire"))
        profiles = quality_profiles()

        self.assertEqual(tuple(profiles), QUALITY_TIER_ORDER)
        self.assertEqual([profile.rank for profile in profiles.values()], [1, 2, 3])
        self.assertEqual(
            [profile.promise for profile in profiles.values()],
            ["Good accuracy", "More accurate", "Most accurate"],
        )
        self.assertEqual(normalize_quality_tier(None), "torch")
        self.assertEqual(normalize_quality_tier("unknown"), "torch")
        self.assertEqual(normalize_quality_tier("unknown", default="fire"), "fire")

    def test_profile_and_derived_settings_honor_configured_overrides(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(
                Path(tmp),
                default_quality_tier="fire",
                quality_light_asr_provider="faster-whisper",
                quality_light_asr_model="small-local",
                quality_torch_asr_provider="faster-whisper",
                quality_torch_asr_model="medium-local",
                quality_fire_asr_provider="whisperx",
                quality_fire_asr_model="large-local",
            )

            self.assertEqual(quality_profile(None, settings).id, "fire")
            self.assertEqual(quality_profile("invalid", settings).id, "fire")
            self.assertEqual(quality_profiles(settings)["light"].asr_model, "small-local")

            light_settings = settings_for_quality(settings, "light")
            fire_settings = settings_for_quality(settings, "fire")

        self.assertEqual(light_settings.asr_provider, "faster-whisper")
        self.assertEqual(light_settings.asr_model, "small-local")
        self.assertEqual(light_settings.faster_whisper_model, "small-local")
        self.assertEqual(light_settings.asr_beam_size, 1)
        self.assertEqual(light_settings.audio_cleanup, "none")
        self.assertEqual(fire_settings.asr_provider, "whisperx")
        self.assertEqual(fire_settings.asr_model, "large-local")
        self.assertEqual(fire_settings.whisperx_model, "large-local")
        self.assertEqual(fire_settings.asr_beam_size, 5)
        self.assertEqual(fire_settings.audio_cleanup, "enhanced")


class DatabaseQualityFeatureTests(unittest.TestCase):
    def test_processing_options_round_trip_without_overwriting_template(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "atlas.sqlite")
            db.initialize()
            recording_id = db.create_recording(root / "meeting.wav")

            self.assertEqual(
                db.get_recording_processing_options(recording_id),
                {"expected_main_speakers": None, "quality_tier": "torch"},
            )
            db.set_recording_template(recording_id, "standup")
            db.set_recording_processing_options(
                recording_id,
                expected_main_speakers=3,
                quality_tier="light",
            )

            self.assertEqual(
                db.get_recording_processing_options(recording_id),
                {"expected_main_speakers": 3, "quality_tier": "light"},
            )
            self.assertEqual(db.get_recording_template(recording_id), "standup")

            db.set_recording_processing_options(
                recording_id,
                expected_main_speakers=None,
                quality_tier="fire",
            )
            self.assertEqual(
                db.get_recording_processing_options(recording_id),
                {"expected_main_speakers": None, "quality_tier": "fire"},
            )

            with self.assertRaisesRegex(ValueError, "between 1 and 20"):
                db.set_recording_processing_options(
                    recording_id,
                    expected_main_speakers=21,
                    quality_tier="torch",
                )
            with self.assertRaisesRegex(ValueError, "light, torch, or fire"):
                db.set_recording_processing_options(
                    recording_id,
                    expected_main_speakers=2,
                    quality_tier="spark",
                )

    def test_speaker_aliases_update_segments_and_the_speaker_search_index(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "atlas.sqlite")
            db.initialize()
            recording_id = db.create_recording(root / "interview.wav", title="Interview")
            db.replace_segments(
                recording_id,
                [
                    {
                        "start": 0.0,
                        "end": 3.0,
                        "speaker": "SPEAKER_00",
                        "text": "Welcome to the product review.",
                    },
                    {
                        "start": 3.0,
                        "end": 5.0,
                        "speaker": "SPEAKER_01",
                        "text": "Thanks for inviting me.",
                    },
                ],
            )

            db.save_recording_speaker_names(
                recording_id,
                {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
            )
            segments = db.get_segments(recording_id)

            self.assertEqual(
                [(segment["speaker_label"], segment["speaker"]) for segment in segments],
                [("SPEAKER_00", "Alice"), ("SPEAKER_01", "Bob")],
            )
            alice_results = db.search("Alice", kind="speaker")
            self.assertEqual(len(alice_results), 1)
            self.assertEqual(alice_results[0]["recording_id"], recording_id)
            self.assertEqual(alice_results[0]["speaker"], "Alice")

            db.save_recording_speaker_names(
                recording_id,
                {"SPEAKER_00": "Alicia", "SPEAKER_01": "Bob"},
            )

            self.assertEqual(db.search("Alice", kind="speaker"), [])
            renamed_results = db.search("Alicia", kind="speaker")
            self.assertEqual(len(renamed_results), 1)
            self.assertEqual(renamed_results[0]["speaker"], "Alicia")
            self.assertEqual(db.get_segments(recording_id)[0]["speaker"], "Alicia")

    def test_defer_and_recovery_do_not_consume_retry_attempts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "atlas.sqlite")
            db.initialize()
            recording_id = db.create_recording(root / "meeting.wav")
            db.enqueue_job(recording_id, "transcribe", max_attempts=2)

            claimed = db.claim_next_job()
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["attempts"], 1)

            before_defer = datetime.now(timezone.utc)
            deferred = db.defer_job(
                claimed["id"],
                "Waiting for device memory.",
                delay_seconds=60,
            )

            self.assertEqual(deferred["status"], "queued")
            self.assertEqual(deferred["attempts"], 0)
            self.assertEqual(deferred["error"], "Waiting for device memory.")
            self.assertGreater(datetime.fromisoformat(deferred["available_at"]), before_defer)
            self.assertIsNone(db.claim_next_job())

            with db.connect() as conn:
                conn.execute(
                    "UPDATE jobs SET available_at = '2000-01-01T00:00:00+00:00' WHERE id = ?",
                    (claimed["id"],),
                )
            reclaimed = db.claim_next_job()
            self.assertIsNotNone(reclaimed)
            self.assertEqual(reclaimed["attempts"], 1)

            db.update_recording(recording_id, status="transcribe_running", error="stale")
            self.assertEqual(db.recover_running_jobs(), 1)
            recovered = db.jobs_for_recording(recording_id)[0]
            recording = db.get_recording(recording_id)

            self.assertEqual(recovered["status"], "queued")
            self.assertEqual(recovered["attempts"], 0)
            self.assertEqual(recovered["error"], "Recovered after worker restart.")
            self.assertIsNone(recovered["available_at"])
            self.assertEqual(recording["status"], "queued")
            self.assertIsNone(recording["error"])
            self.assertEqual(db.recover_running_jobs(), 0)


class WorkerStartupTests(unittest.TestCase):
    def test_schema_is_initialized_before_job_recovery(self) -> None:
        events: list[str] = []

        class FakeSettings:
            inbox_dir = Path("/tmp/atlas-voice-worker-test")

            def ensure_directories(self) -> None:
                events.append("directories")

        class FakeDatabase:
            initialized = False

            def initialize(self) -> None:
                self.initialized = True
                events.append("initialize")

            def recover_running_jobs(self) -> int:
                self_test.assertTrue(self.initialized)
                events.append("recover")
                return 0

        self_test = self
        worker = Worker(FakeSettings(), FakeDatabase())

        def stop_before_loop() -> None:
            events.append("signals")
            worker._stop = True

        worker.install_signal_handlers = stop_before_loop
        worker.run_forever()

        self.assertEqual(events, ["directories", "initialize", "recover", "signals"])


class ResourceAdmissionTests(unittest.TestCase):
    def test_memory_thresholds_are_tier_and_step_specific(self) -> None:
        cases = [
            ("light", "transcribe", 1399, 1024, False, 1400),
            ("light", "transcribe", 1400, 0, True, 1400),
            ("torch", "diarize", 2799, 1024, False, 2800),
            ("torch", "diarize", 2800, 0, True, 2800),
            ("fire", "diarize", 4199, 1024, False, 4200),
            ("fire", "diarize", 4200, 0, True, 4200),
        ]
        for tier, step, memory, swap, allowed, required in cases:
            with self.subTest(tier=tier, step=step, memory=memory, swap=swap):
                decision = admit_pipeline_step(
                    tier,
                    step,
                    snapshot=ResourceSnapshot(memory, swap),
                )
                self.assertEqual(decision.allowed, allowed)
                self.assertEqual(decision.required_memory_mib, required)
                self.assertEqual(decision.tier, tier)
                self.assertEqual(decision.step, step)

    def test_fire_transcription_requires_swap_or_extra_memory(self) -> None:
        low_swap = admit_pipeline_step(
            "fire",
            "transcribe",
            snapshot=ResourceSnapshot(9000, 255),
        )
        enough_swap = admit_pipeline_step(
            "fire",
            "transcribe",
            snapshot=ResourceSnapshot(9000, 256),
        )
        extra_memory = admit_pipeline_step(
            "fire",
            "transcribe",
            snapshot=ResourceSnapshot(11048, 0),
        )

        self.assertFalse(low_swap.allowed)
        self.assertIn("waiting for device memory", low_swap.message)
        self.assertTrue(enough_swap.allowed)
        self.assertTrue(extra_memory.allowed)
        self.assertEqual(enough_swap.message, "Device resources are ready.")

    def test_unknown_step_is_admitted_without_a_memory_requirement(self) -> None:
        decision = admit_pipeline_step(
            "not-a-tier",
            "summarize",
            snapshot=ResourceSnapshot(0, 0),
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.tier, "torch")
        self.assertEqual(decision.required_memory_mib, 0)


if __name__ == "__main__":
    unittest.main()
