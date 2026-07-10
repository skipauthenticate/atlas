from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from atlas_voice.config import Settings
from atlas_voice.database import Database
from atlas_voice.pipeline import PipelineProcessor


class PipelineTests(unittest.TestCase):
    def test_summarize_auto_syncs_to_anythingllm_when_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root, anythingllm_auto_sync=True)
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()
            recording_id = db.create_recording(root / "meeting.wav", title="Meeting")
            db.replace_segments(
                recording_id,
                [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hello."}],
            )
            db.enqueue_job(recording_id, "summarize")
            job = db.claim_next_job()

            processor = PipelineProcessor(settings, db)
            with patch(
                "atlas_voice.pipeline.sync_recording_to_anythingllm",
                return_value={"success": True, "documents": []},
            ) as sync_mock:
                processor.process_job(dict(job))

            self.assertEqual(db.get_recording(recording_id)["status"], "done")
            self.assertIsNotNone(db.get_summary(recording_id))
            model_run = db.list_model_runs()[0]
            self.assertEqual(model_run["task"], "summarize")
            self.assertEqual(model_run["provider"], "stub")
            self.assertEqual(model_run["input_ref"], f"recording:{recording_id}")
            sync_mock.assert_called_once_with(db, recording_id, settings)

    def test_summarize_does_not_auto_sync_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()
            recording_id = db.create_recording(root / "meeting.wav", title="Meeting")
            db.replace_segments(
                recording_id,
                [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hello."}],
            )
            db.enqueue_job(recording_id, "summarize")
            job = db.claim_next_job()

            processor = PipelineProcessor(settings, db)
            with patch("atlas_voice.pipeline.sync_recording_to_anythingllm") as sync_mock:
                processor.process_job(dict(job))

            self.assertEqual(db.get_recording(recording_id)["status"], "done")
            sync_mock.assert_not_called()

    def test_auto_sync_failure_keeps_recording_done_and_stores_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root, anythingllm_auto_sync=True)
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()
            recording_id = db.create_recording(root / "meeting.wav", title="Meeting")
            db.replace_segments(
                recording_id,
                [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hello."}],
            )
            db.enqueue_job(recording_id, "summarize")
            job = db.claim_next_job()

            processor = PipelineProcessor(settings, db)
            with patch(
                "atlas_voice.pipeline.sync_recording_to_anythingllm",
                side_effect=ValueError("no transcript or summary"),
            ):
                processor.process_job(dict(job))

            recording = db.get_recording(recording_id)
            self.assertEqual(recording["status"], "done")
            self.assertIn("AnythingLLM auto-sync failed", recording["error"])

    def test_first_summary_generates_a_semantic_title_for_filename_titles(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()
            recording_id = db.create_recording(root / "2026-07-09-audio.wav")
            db.replace_segments(
                recording_id,
                [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hello."}],
            )
            db.enqueue_job(recording_id, "summarize")

            PipelineProcessor(settings, db).process_job(dict(db.claim_next_job()))

            recording = db.get_recording(recording_id)
            self.assertEqual(recording["status"], "done")
            self.assertEqual(recording["title_origin"], "generated")
            self.assertEqual(recording["title"], "Atlas Voice processed a local test recording")

    def test_summary_never_overwrites_a_manual_title(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()
            recording_id = db.create_recording(root / "audio.wav", title="My chosen title")
            db.replace_segments(
                recording_id,
                [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hello."}],
            )
            db.enqueue_job(recording_id, "summarize")

            PipelineProcessor(settings, db).process_job(dict(db.claim_next_job()))

            recording = db.get_recording(recording_id)
            self.assertEqual(recording["title"], "My chosen title")
            self.assertEqual(recording["title_origin"], "manual")

    def test_title_generation_failure_does_not_fail_the_recording(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()
            recording_id = db.create_recording(root / "audio.wav")
            db.replace_segments(
                recording_id,
                [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hello."}],
            )
            db.enqueue_job(recording_id, "summarize")

            with patch(
                "atlas_voice.pipeline.generate_recording_title",
                side_effect=RuntimeError("title model unavailable"),
            ):
                PipelineProcessor(settings, db).process_job(dict(db.claim_next_job()))

            recording = db.get_recording(recording_id)
            self.assertEqual(recording["status"], "done")
            self.assertEqual(recording["title_origin"], "filename")
            title_runs = [run for run in db.list_model_runs() if run["task"] == "generate_title"]
            self.assertEqual(len(title_runs), 1)
            self.assertIn("title model unavailable", title_runs[0]["error"])


def _settings(root: Path, **overrides: object) -> Settings:
    values = {
        "host": "127.0.0.1",
        "port": 8787,
        "data_dir": root / "data",
        "models_dir": root / "models",
        "hf_cache_dir": root / "cache" / "huggingface",
        "whisperx_model": "tiny.en",
        "whisperx_device": "cpu",
        "whisperx_compute_type": "int8",
        "pyannote_model": "pyannote/speaker-diarization-community-1",
        "hf_token": None,
        "llm_base_url": "http://127.0.0.1:8080/v1/chat/completions",
        "llm_model": "qwen-local",
        "llm_temperature": 0.2,
        "llm_max_tokens": 1200,
        "stub_mode": True,
        "allow_single_speaker_fallback": True,
        "asr_provider": "whisperx",
        "asr_model": None,
        "diarization_provider": "pyannote",
        "nemo_source_lang": "en",
        "nemo_target_lang": "en",
        "vibevoice_model": "microsoft/VibeVoice-ASR",
        "vibevoice_max_new_tokens": 32768,
        "anythingllm_base_url": "http://127.0.0.1:3001/api",
        "anythingllm_api_key": "test-key",
        "anythingllm_workspace_slug": "notes",
        "anythingllm_timeout": 10.0,
        "anythingllm_auto_sync": False,
    }
    values.update(overrides)
    return Settings(**values)


if __name__ == "__main__":
    unittest.main()
