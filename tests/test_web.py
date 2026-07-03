from pathlib import Path
from tempfile import TemporaryDirectory
import importlib
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


class WebTests(unittest.TestCase):
    def test_html_pages_render_with_current_starlette_signature(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            recording_id = web_app.db.create_recording(root / "audio.wav", title="Test Audio")
            web_app.db.update_recording(recording_id, status="done")
            web_app.db.enqueue_job(recording_id, "ingest")
            job = web_app.db.claim_next_job()
            with web_app.db.connect() as conn:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'done',
                        attempts = 1,
                        started_at = '2026-06-27T08:00:00+00:00',
                        finished_at = '2026-06-27T08:00:05+00:00'
                    WHERE id = ?
                    """,
                    (job["id"],),
                )
            web_app.db.replace_segments(
                recording_id,
                [
                    {
                        "start": 0,
                        "end": 1,
                        "speaker": "SPEAKER_00",
                        "text": "Atlas Voice transcript",
                    }
                ],
            )
            web_app.db.save_summary(
                recording_id,
                "**Overview**\n"
                "Atlas Voice summary\n\n"
                "**Key Points**\n"
                "- **Result:** readable summary",
                model="test",
            )

            client = TestClient(web_app.app)

            dashboard = client.get("/")
            detail = client.get(f"/recordings/{recording_id}")
            search = client.get("/search", params={"q": "Atlas"})

            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("Test Audio", dashboard.text)
            self.assertIn("CPU", dashboard.text)
            self.assertIn("tiny.en", dashboard.text)
            self.assertEqual(detail.status_code, 200)
            self.assertIn("SPEAKER_00", detail.text)
            self.assertIn("summary-section-overview", detail.text)
            self.assertIn("summary-section-key-points", detail.text)
            self.assertIn("<strong>Result</strong>", detail.text)
            self.assertIn("Duration", detail.text)
            self.assertIn("00:05", detail.text)
            self.assertIn("int8", detail.text)
            self.assertEqual(search.status_code, 200)
            self.assertIn("[Atlas]", search.text)
            self.assertNotIn("&lt;mark&gt;", search.text)

    def test_runtime_settings_form_persists_provider_config(self) -> None:
        keys = [
            "ATLAS_VOICE_ENV_FILE",
            "ATLAS_VOICE_DATA_DIR",
            "ATLAS_VOICE_MODELS_DIR",
            "ATLAS_VOICE_HF_CACHE",
            "ATLAS_VOICE_STUB_MODE",
            "ATLAS_VOICE_ASR_PROVIDER",
            "ATLAS_VOICE_ASR_MODEL",
            "ATLAS_VOICE_DIARIZATION_PROVIDER",
            "WHISPERX_DEVICE",
            "WHISPERX_MODEL",
            "WHISPERX_COMPUTE_TYPE",
        ]
        old_env = {key: os.environ.get(key) for key in keys}
        try:
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                env_file = root / ".env"
                env_file.write_text("ATLAS_VOICE_ASR_PROVIDER=whisperx\nWHISPERX_MODEL=tiny.en\n")
                os.environ["ATLAS_VOICE_ENV_FILE"] = str(env_file)
                os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
                os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
                os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
                os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
                os.environ["ATLAS_VOICE_ASR_PROVIDER"] = "whisperx"
                os.environ["ATLAS_VOICE_ASR_MODEL"] = ""
                os.environ["ATLAS_VOICE_DIARIZATION_PROVIDER"] = "pyannote"
                os.environ["WHISPERX_DEVICE"] = "cpu"
                os.environ["WHISPERX_MODEL"] = "tiny.en"
                os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

                import atlas_voice.web.app as web_app

                web_app = importlib.reload(web_app)
                web_app.settings.ensure_directories()
                web_app.db.initialize()
                web_app._restart_worker_if_idle = lambda: "restarted"
                client = TestClient(web_app.app)

                response = client.post(
                    "/settings/runtime",
                    data={
                        "asr_provider": "canary",
                        "asr_model": "nvidia/canary-1b-v2",
                        "diarization_provider": "pyannote",
                    },
                )

                self.assertEqual(response.status_code, 200)
                self.assertIn("Runtime settings saved", response.text)
                self.assertEqual(web_app.settings.asr_provider, "canary")
                self.assertEqual(web_app.settings.asr_model, None)
                saved_env = env_file.read_text()
                self.assertIn("ATLAS_VOICE_ASR_PROVIDER=canary", saved_env)
                self.assertIn("ATLAS_VOICE_ASR_MODEL=", saved_env)
                self.assertIn("ATLAS_VOICE_DIARIZATION_PROVIDER=pyannote", saved_env)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_template_change_queues_resummary_and_preserves_cached_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            recording_id = web_app.db.create_recording(root / "audio.wav", title="Test Audio")
            web_app.db.update_recording(recording_id, status="done")
            web_app.db.enqueue_job(recording_id, "summarize")
            job = web_app.db.claim_next_job()
            web_app.db.complete_job(job["id"])
            web_app.db.save_summary(
                recording_id,
                "**Overview**\nCached summary",
                model="test",
                template_id="meeting",
            )

            client = TestClient(web_app.app)
            response = client.post(
                f"/recordings/{recording_id}/template",
                json={"template_id": "personal"},
            )
            detail = client.get(f"/recordings/{recording_id}")

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["queued"])
            self.assertEqual(
                web_app.db.get_summary(recording_id)["text"], "**Overview**\nCached summary"
            )
            self.assertEqual(web_app.db.get_recording(recording_id)["status"], "queued")
            self.assertIn("Switching to", detail.text)
            self.assertIn("Cached summary", detail.text)

    def test_recording_detail_can_sync_to_anythingllm(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"
            os.environ["ANYTHINGLLM_BASE_URL"] = "http://127.0.0.1:3001/api"
            os.environ["ANYTHINGLLM_API_KEY"] = "test-key"
            os.environ["ANYTHINGLLM_WORKSPACE_SLUG"] = "notes"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            recording_id = web_app.db.create_recording(root / "audio.wav", title="Test Audio")
            web_app.db.update_recording(recording_id, status="done")
            web_app.db.replace_segments(
                recording_id,
                [
                    {
                        "start": 0,
                        "end": 1,
                        "speaker": "SPEAKER_00",
                        "text": "Atlas Voice transcript",
                    }
                ],
            )
            web_app.db.save_summary(recording_id, "Atlas Voice summary", model="test")

            client = TestClient(web_app.app)
            with patch.object(
                web_app,
                "sync_recording_to_anythingllm",
                return_value={"success": True, "documents": []},
            ) as sync_mock:
                response = client.post(f"/recordings/{recording_id}/sync-anythingllm")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Synced to AnythingLLM", response.text)
            sync_mock.assert_called_once_with(web_app.db, recording_id, web_app.settings)


if __name__ == "__main__":
    unittest.main()
