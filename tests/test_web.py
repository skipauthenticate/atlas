from pathlib import Path
from array import array
import base64
import math
from tempfile import TemporaryDirectory
import importlib
import os
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from atlas_voice.realtime import RealtimeAudio, RealtimeReply


class WebTests(unittest.TestCase):
    def test_html_pages_render_with_current_starlette_signature(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
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
            ambient_id = web_app.db.create_ambient_session(
                mode="ambient",
                source="test",
                title="Ambient Test",
            )
            web_app.db.add_utterance(
                session_id=ambient_id,
                text="ambient transcript",
                source_provider="stub",
            )
            web_app.db.end_ambient_session(ambient_id, status="done")

            client = TestClient(web_app.app)

            dashboard = client.get("/")
            detail = client.get(f"/recordings/{recording_id}")
            search = client.get("/search", params={"q": "Atlas"})
            status = client.get("/api/status")
            ambient_sessions = client.get("/api/ambient/sessions")

            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("Local workspace", dashboard.text)
            self.assertIn('action="/upload"', dashboard.text)
            self.assertIn('enctype="multipart/form-data"', dashboard.text)
            self.assertIn("data-chat-file-input", dashboard.text)
            self.assertIn("data-chat-attachments", dashboard.text)
            self.assertIn("data-chat-source", dashboard.text)
            self.assertIn("All sources", dashboard.text)
            self.assertIn('data-upload-endpoint="/upload"', dashboard.text)
            self.assertIn("data-chat-voice-toggle", dashboard.text)
            self.assertIn('id="recordings"', dashboard.text)
            self.assertIn("data-dashboard-recording-list", dashboard.text)
            self.assertIn("Recordings", dashboard.text)
            self.assertIn("Test Audio", dashboard.text)
            self.assertIn(f"/recordings/{recording_id}", dashboard.text)
            self.assertIn("CPU", dashboard.text)
            self.assertIn("tiny.en", dashboard.text)
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["privacy"]["status"], "ok")
            self.assertEqual(ambient_sessions.status_code, 200)
            self.assertEqual(ambient_sessions.json()["sessions"][0]["id"], ambient_id)
            self.assertEqual(ambient_sessions.json()["sessions"][0]["utterance_count"], 1)
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

    def test_chat_upload_endpoint_can_enqueue_file_without_navigation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.post(
                "/upload",
                files={"file": ("chat-input.wav", b"RIFFtest", "audio/wav")},
                headers={"accept": "application/json"},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "queued")
            self.assertIn("recording_id", payload)
            self.assertEqual(payload["url"], f"/recordings/{payload['recording_id']}")
            self.assertIn("chat-input.wav", payload["title"])
            recording = dict(web_app.db.get_recording(payload["recording_id"]))
            self.assertEqual(recording["status"], "queued")
            self.assertTrue(Path(recording["source_path"]).exists())
            jobs = web_app.db.jobs_for_recording(payload["recording_id"])
            self.assertEqual(jobs[0]["step"], "ingest")

    def test_voice_console_uses_direct_voice_profile_settings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "atlas.assistant.yaml").write_text(
                "profiles:\n"
                "  direct_voice:\n"
                "    enabled: true\n"
                "    stt_provider: parakeet\n"
                "    asr_model: nvidia/parakeet-tdt-0.6b-v3\n"
                "    tts_provider: faster-qwen3-tts\n"
                "    tts_model: qwen-profile\n"
                "  reflection:\n"
                "    asr_provider: canary\n"
                "    diarization_provider: none\n"
            )
            env = {
                "ATLAS_VOICE_DATA_DIR": str(root / "data"),
                "ATLAS_VOICE_MODELS_DIR": str(root / "models"),
                "ATLAS_VOICE_HF_CACHE": str(root / "cache" / "huggingface"),
                "ATLAS_VOICE_ASSISTANT_CONFIG": str(config_dir / "atlas.assistant.yaml"),
                "ATLAS_ASSISTANT_ENABLED": "true",
                "ATLAS_VOICE_ASR_PROVIDER": "whisperx",
                "ATLAS_VOICE_ASR_MODEL": "env-model",
                "ATLAS_VOICE_TTS_PROVIDER": "none",
                "ATLAS_VOICE_STUB_MODE": "true",
                "WHISPERX_DEVICE": "cpu",
                "WHISPERX_MODEL": "tiny.en",
                "WHISPERX_COMPUTE_TYPE": "int8",
            }
            with patch.dict(os.environ, env, clear=True):
                import atlas_voice.web.app as web_app

                web_app = importlib.reload(web_app)
                web_app.settings.ensure_directories()
                web_app.db.initialize()
                client = TestClient(web_app.app)

                response = client.get("/voice")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(web_app.processor.settings.asr_provider, "canary")
        self.assertEqual(web_app.processor.settings.diarization_provider, "none")
        self.assertIn("parakeet", response.text)
        self.assertIn("nvidia/parakeet-tdt-0.6b-v3", response.text)
        self.assertIn("qwen-profile", response.text)

    def test_assistant_health_api_reports_local_components(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "false"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.get("/api/assistant/health")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "ok")
            self.assertTrue(payload["local_only"])
            self.assertEqual(payload["realtime"]["websocket_path"], "/v1/realtime")
            self.assertFalse(payload["realtime"]["enabled"])
            self.assertEqual(payload["components"]["assistant_runtime"]["status"], "disabled")
            self.assertEqual(payload["components"]["database"]["status"], "ok")
            self.assertEqual(payload["components"]["privacy"]["status"], "ok")
            self.assertEqual(payload["components"]["tts"]["status"], "idle")
            self.assertEqual(payload["active_listener_count"], 0)

    def test_assistant_health_api_reports_tts_sidecar_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "faster-qwen3-tts"
            os.environ["ATLAS_TTS_MODEL"] = "faster-qwen3-tts-0.6b"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.status as status_module
            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            with patch.object(
                status_module,
                "check_tts_sidecar_health",
                return_value={
                    "status": "error",
                    "detail": "ConnectError: refused",
                    "url": "http://127.0.0.1:8008/health",
                    "latency_ms": 3,
                },
            ):
                response = client.get("/api/assistant/health")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "degraded")
            self.assertEqual(payload["components"]["tts"]["status"], "error")
            self.assertIn("faster-qwen3-tts-0.6b", payload["components"]["tts"]["detail"])
            self.assertIn("tts", payload["issues"][0]["component"])

    def test_assistant_privacy_api_reports_local_only_status(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.get("/api/assistant/privacy")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "ok")
            self.assertTrue(payload["local_only"])
            self.assertEqual(payload["issue_count"], 0)
            self.assertIn("pause", payload["controls"])
            self.assertIn("audit_egress", payload["controls"])

    def test_dashboard_pause_private_mode_control_persists_and_logs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / ".env"
            env_file.write_text("ATLAS_VOICE_AMBIENT_MODE=ambient\n")
            env = {
                "ATLAS_VOICE_ENV_FILE": str(env_file),
                "ATLAS_VOICE_DATA_DIR": str(root / "data"),
                "ATLAS_VOICE_MODELS_DIR": str(root / "models"),
                "ATLAS_VOICE_HF_CACHE": str(root / "cache" / "huggingface"),
                "ATLAS_VOICE_ASSISTANT_CONFIG": str(root / "config" / "atlas.assistant.yaml"),
                "ATLAS_ASSISTANT_ENABLED": "true",
                "ATLAS_VOICE_TTS_PROVIDER": "none",
                "ATLAS_VOICE_STUB_MODE": "true",
                "LLM_BASE_URL": "http://127.0.0.1:8080/v1/chat/completions",
                "WHISPERX_DEVICE": "cpu",
                "WHISPERX_MODEL": "tiny.en",
                "WHISPERX_COMPUTE_TYPE": "int8",
            }
            with patch.dict(os.environ, env, clear=True):
                import atlas_voice.web.app as web_app

                web_app = importlib.reload(web_app)
                web_app.settings.ensure_directories()
                web_app.db.initialize()
                client = TestClient(web_app.app)

                dashboard = client.get("/")
                response = client.post("/settings/assistant-mode", data={"mode": "private"})

                self.assertEqual(dashboard.status_code, 200)
                self.assertIn('action="/settings/assistant-mode"', dashboard.text)
                self.assertIn('name="mode" value="paused"', dashboard.text)
                self.assertIn('name="mode" value="private"', dashboard.text)
                self.assertEqual(response.status_code, 200)
                self.assertIn("Assistant mode saved", response.text)
                self.assertEqual(web_app.settings.ambient_mode, "private")
                self.assertIn("ATLAS_VOICE_AMBIENT_MODE=private", env_file.read_text())
                event = web_app.db.list_privacy_events()[0]
                self.assertEqual(event["event_type"], "assistant.mode")
                self.assertEqual(event["metadata"]["mode"], "private")
                refreshed = client.get("/")
                self.assertIn('data-mode="private" aria-pressed="true"', refreshed.text)

    def test_assistant_mode_redirects_to_safe_requested_path(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / ".env"
            env_file.write_text("ATLAS_VOICE_AMBIENT_MODE=ambient\n")
            env = {
                "ATLAS_VOICE_ENV_FILE": str(env_file),
                "ATLAS_VOICE_DATA_DIR": str(root / "data"),
                "ATLAS_VOICE_MODELS_DIR": str(root / "models"),
                "ATLAS_VOICE_HF_CACHE": str(root / "cache" / "huggingface"),
                "ATLAS_VOICE_ASSISTANT_CONFIG": str(root / "config" / "atlas.assistant.yaml"),
                "ATLAS_ASSISTANT_ENABLED": "true",
                "ATLAS_VOICE_TTS_PROVIDER": "none",
                "ATLAS_VOICE_STUB_MODE": "true",
                "LLM_BASE_URL": "http://127.0.0.1:8080/v1/chat/completions",
                "WHISPERX_DEVICE": "cpu",
                "WHISPERX_MODEL": "tiny.en",
                "WHISPERX_COMPUTE_TYPE": "int8",
            }
            with patch.dict(os.environ, env, clear=True):
                import atlas_voice.web.app as web_app

                web_app = importlib.reload(web_app)
                web_app.settings.ensure_directories()
                web_app.db.initialize()
                client = TestClient(web_app.app, follow_redirects=False)

                voice_response = client.post(
                    "/settings/assistant-mode",
                    data={"mode": "paused", "redirect_to": "/voice"},
                )
                external_response = client.post(
                    "/settings/assistant-mode",
                    data={"mode": "private", "redirect_to": "//example.com"},
                )

                self.assertEqual(voice_response.status_code, 303)
                self.assertEqual(voice_response.headers["location"], "/voice?assistant_mode=saved")
                self.assertEqual(external_response.status_code, 303)
                self.assertEqual(external_response.headers["location"], "/?assistant_mode=saved")

    def test_assistant_privacy_api_surfaces_external_endpoint_errors(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "https://api.example.com/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.get("/api/assistant/privacy")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "error")
            self.assertFalse(payload["local_only"])
            self.assertIn("llm_base_url", {issue["check"] for issue in payload["issues"]})

    def test_voice_console_renders_workbench_with_recent_sessions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            session_id = web_app.db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Realtime coaching check-in",
            )
            utterance_id = web_app.db.add_utterance(
                session_id=session_id,
                text="How direct was my update?",
                source_provider="text",
            )
            web_app.db.add_assistant_turn(
                session_id=session_id,
                user_utterance_id=utterance_id,
                text="Your update was direct and specific.",
                model="qwen-local",
                latency_ms=11,
            )
            web_app.db.end_ambient_session(session_id)
            client = TestClient(web_app.app)

            response = client.get("/voice")

            self.assertEqual(response.status_code, 200)
            self.assertIn('class="voice-workbench"', response.text)
            self.assertIn("Voice", response.text)
            self.assertIn("Ambient", response.text)
            self.assertIn("Memory", response.text)
            self.assertIn("Coaching", response.text)
            self.assertIn("Realtime coaching check-in", response.text)
            self.assertIn("How direct was my update?", response.text)
            self.assertIn("Your update was direct and specific.", response.text)
            self.assertIn("Call controls", response.text)
            self.assertIn("Start call", response.text)
            self.assertIn("End call", response.text)
            self.assertIn("Privacy", response.text)
            self.assertIn("Models", response.text)

    def test_voice_console_shows_ambient_session_timeline(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            ambient_id = web_app.db.create_ambient_session(
                mode="ambient",
                source="mic",
                title="Kitchen capture",
            )
            meeting_id = web_app.db.create_ambient_session(
                mode="meeting",
                source="file",
                title="Planning meeting",
            )
            web_app.db.add_utterance(
                session_id=ambient_id,
                text="Remember to follow up on the launch note.",
                source_provider="stub",
                is_directed_to_assistant=True,
                sensitivity="personal",
            )
            web_app.db.add_utterance(
                session_id=meeting_id,
                text="We discussed the roadmap risks.",
                source_provider="stub",
            )
            web_app.db.end_ambient_session(ambient_id)
            web_app.db.end_ambient_session(meeting_id)
            client = TestClient(web_app.app)

            response = client.get("/voice")

            self.assertEqual(response.status_code, 200)
            self.assertIn("data-ambient-timeline", response.text)
            self.assertIn("Ambient Timeline", response.text)
            self.assertIn("Kitchen capture", response.text)
            self.assertIn("Planning meeting", response.text)
            self.assertIn("Remember to follow up on the launch note.", response.text)
            self.assertIn("We discussed the roadmap risks.", response.text)
            self.assertIn("mic", response.text)
            self.assertIn("file", response.text)
            self.assertIn("assistant-directed", response.text)
            self.assertIn("personal", response.text)

    def test_voice_console_shows_recent_privacy_events(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            purge_id = web_app.db.log_privacy_event(
                "privacy.purge",
                "Purged 2 privacy session(s).",
                severity="warning",
                metadata={"session_count": 2, "artifact_count": 1},
            )
            audit_id = web_app.db.log_privacy_event(
                "privacy.audit_egress",
                "Local-only status: ok",
                metadata={"issue_count": 0},
            )
            with web_app.db.connect() as conn:
                conn.execute(
                    "UPDATE privacy_events SET created_at = ? WHERE id = ?",
                    ("2026-07-01T10:00:00+00:00", audit_id),
                )
                conn.execute(
                    "UPDATE privacy_events SET created_at = ? WHERE id = ?",
                    ("2026-07-02T10:00:00+00:00", purge_id),
                )
            client = TestClient(web_app.app)

            api_response = client.get("/api/privacy/events")
            voice_response = client.get("/voice")

            self.assertEqual(api_response.status_code, 200)
            payload = api_response.json()
            self.assertEqual(payload["event_count"], 2)
            self.assertEqual(payload["events"][0]["event_type"], "privacy.purge")
            self.assertEqual(payload["events"][0]["severity"], "warning")
            self.assertEqual(
                payload["events"][0]["metadata_summary"], "artifact_count=1, session_count=2"
            )
            self.assertEqual(payload["severity_counts"]["warning"], 1)
            self.assertEqual(payload["severity_counts"]["info"], 1)
            self.assertEqual(voice_response.status_code, 200)
            self.assertIn("data-privacy-events", voice_response.text)
            self.assertIn("Privacy Events", voice_response.text)
            self.assertIn("privacy.purge", voice_response.text)
            self.assertIn("Purged 2 privacy session(s).", voice_response.text)
            self.assertIn("warning", voice_response.text)
            self.assertIn("artifact_count=1, session_count=2", voice_response.text)

    def test_voice_console_shows_coaching_progress_dashboard(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            session_id = web_app.db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Coaching progress session",
            )
            conversation_event = web_app.db.log_feedback_event(
                event_type="coaching.conversation_signals",
                category="conversation_signals",
                session_id=session_id,
                message="Conversation Signals - Coaching progress session",
                score=0.75,
                metadata={
                    "signals": {
                        "clarity": 0.75,
                        "concision": 0.5,
                        "question_ratio": 0.25,
                        "talk_listen_ratio": 0.42,
                        "open_question_ratio": 0.5,
                        "affirmation_ratio": 0.4,
                        "reflection_ratio": 0.3,
                        "summary_ratio": 0.2,
                        "change_talk_ratio": 0.35,
                        "sustain_talk_ratio": 0.15,
                        "autonomy_support_ratio": 0.67,
                        "interruption_overlap_ratio": 0.12,
                        "hedging_ratio": 0.22,
                        "follow_through": 0.5,
                    }
                },
            )
            writing_event = web_app.db.log_feedback_event(
                event_type="coaching.writing_signals",
                category="writing_signals",
                message="Writing Signals - Launch note",
                score=0.9,
                metadata={
                    "label": "Launch note",
                    "signals": {
                        "clarity": 0.9,
                        "concision": 0.8,
                        "ask_action_clarity": 1.0,
                        "tone": 1.0,
                    },
                },
            )
            with web_app.db.connect() as conn:
                conn.execute(
                    "UPDATE feedback_events SET created_at = ? WHERE id = ?",
                    ("2026-07-01T10:00:00+00:00", conversation_event),
                )
                conn.execute(
                    "UPDATE feedback_events SET created_at = ? WHERE id = ?",
                    ("2026-07-02T10:00:00+00:00", writing_event),
                )
            client = TestClient(web_app.app)

            api_response = client.get("/api/coaching/progress")
            voice_response = client.get("/voice")

            self.assertEqual(api_response.status_code, 200)
            payload = api_response.json()
            self.assertEqual(payload["event_count"], 2)
            self.assertEqual(payload["latest_events"][0]["title"], "Writing Signals - Launch note")
            self.assertEqual(payload["averages"]["clarity"], 0.825)
            self.assertEqual(payload["averages"]["question_ratio"], 0.25)
            self.assertEqual(payload["averages"]["open_question_ratio"], 0.5)
            self.assertIn(
                {"label": "Question ratio", "value": 0.25, "display": "25%"},
                payload["headline_metrics"],
            )
            self.assertIn(
                {"label": "Talk/listen ratio", "value": 0.42, "display": "0.42:1"},
                payload["headline_metrics"],
            )
            self.assertIn(
                {"label": "Open questions", "value": 0.5, "display": "50%"},
                payload["headline_metrics"],
            )
            self.assertIn(
                {"label": "Affirmations", "value": 0.4, "display": "40%"},
                payload["headline_metrics"],
            )
            self.assertEqual(payload["averages"]["reflection_ratio"], 0.3)
            self.assertIn(
                {"label": "Reflection ratio", "value": 0.3, "display": "30%"},
                payload["headline_metrics"],
            )
            self.assertIn(
                {"label": "Summaries", "value": 0.2, "display": "20%"},
                payload["headline_metrics"],
            )
            self.assertIn(
                {"label": "Change talk", "value": 0.35, "display": "35%"},
                payload["headline_metrics"],
            )
            self.assertIn(
                {"label": "Sustain talk", "value": 0.15, "display": "15%"},
                payload["headline_metrics"],
            )
            self.assertIn(
                {"label": "Autonomy support", "value": 0.67, "display": "67%"},
                payload["headline_metrics"],
            )
            self.assertIn(
                {"label": "Interruptions/overlap", "value": 0.12, "display": "12%"},
                payload["headline_metrics"],
            )
            self.assertIn(
                {"label": "Hedging", "value": 0.22, "display": "22%"},
                payload["headline_metrics"],
            )
            self.assertEqual(payload["categories"]["conversation_signals"], 1)
            self.assertEqual(payload["categories"]["writing_signals"], 1)
            self.assertEqual(voice_response.status_code, 200)
            self.assertIn("data-coaching-progress", voice_response.text)
            self.assertIn("Progress Dashboard", voice_response.text)
            self.assertIn("Clarity", voice_response.text)
            self.assertIn("Question ratio", voice_response.text)
            self.assertIn("Talk/listen ratio", voice_response.text)
            self.assertIn("Open questions", voice_response.text)
            self.assertIn("Affirmations", voice_response.text)
            self.assertIn("Reflection ratio", voice_response.text)
            self.assertIn("Summaries", voice_response.text)
            self.assertIn("Change talk", voice_response.text)
            self.assertIn("Sustain talk", voice_response.text)
            self.assertIn("Autonomy support", voice_response.text)
            self.assertIn("Interruptions/overlap", voice_response.text)
            self.assertIn("Hedging", voice_response.text)
            self.assertIn("82%", voice_response.text)
            self.assertIn("0.42:1", voice_response.text)
            self.assertIn("25%", voice_response.text)
            self.assertIn("50%", voice_response.text)
            self.assertIn("40%", voice_response.text)
            self.assertIn("30%", voice_response.text)
            self.assertIn("20%", voice_response.text)
            self.assertIn("35%", voice_response.text)
            self.assertIn("15%", voice_response.text)
            self.assertIn("67%", voice_response.text)
            self.assertIn("12%", voice_response.text)
            self.assertIn("22%", voice_response.text)
            self.assertIn("Conversation Signals - Coaching progress session", voice_response.text)
            self.assertIn("Writing Signals - Launch note", voice_response.text)

    def test_voice_console_coaching_goals_dashboard_can_create_and_complete_goals(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            goal_id = web_app.db.create_coaching_goal(
                title="Ask better follow-up questions",
                description="Practice one reflective question per conversation.",
                target_date="2026-08-01",
                metric="question_ratio",
                metadata={"next_action": "Review the next daily summary"},
            )
            web_app.db.create_coaching_goal(title="Finished goal", status="completed")
            web_app.db.log_feedback_event(
                event_type="coaching.conversation_signals",
                category="conversation_signals",
                goal_id=goal_id,
                message="Conversation Signals - Practice",
                score=0.6,
                metadata={"signals": {"question_ratio": 0.6, "clarity": 0.8}},
            )
            client = TestClient(web_app.app)

            api_response = client.get("/api/coaching/goals", params={"status": "all"})
            voice_response = client.get("/voice")
            created = client.post(
                "/coaching/goals",
                data={
                    "title": "Reduce hedging",
                    "description": "Use direct asks in launch notes.",
                    "target_date": "2026-09-01",
                    "metric": "hedging_count",
                    "next_action": "Review writing signals weekly",
                },
                follow_redirects=False,
            )
            completed = client.post(
                f"/coaching/goals/{goal_id}/status",
                data={"status": "completed"},
                follow_redirects=False,
            )

            self.assertEqual(api_response.status_code, 200)
            payload = api_response.json()
            self.assertEqual(payload["active_count"], 1)
            self.assertEqual(payload["completed_count"], 1)
            self.assertEqual(payload["goals"][0]["title"], "Ask better follow-up questions")
            self.assertEqual(payload["goals"][0]["latest_score_display"], "60%")
            self.assertEqual(payload["goals"][0]["feedback_count"], 1)
            self.assertEqual(voice_response.status_code, 200)
            self.assertIn("data-coaching-goals", voice_response.text)
            self.assertIn("Coaching Goals", voice_response.text)
            self.assertIn("Ask better follow-up questions", voice_response.text)
            self.assertIn("Review the next daily summary", voice_response.text)
            self.assertIn("question_ratio", voice_response.text)
            self.assertIn("60%", voice_response.text)
            self.assertEqual(created.status_code, 303)
            created_goal = web_app.db.list_coaching_goals(status="active")[0]
            self.assertEqual(created_goal["title"], "Reduce hedging")
            self.assertEqual(
                created_goal["metadata"], {"next_action": "Review writing signals weekly"}
            )
            self.assertEqual(completed.status_code, 303)
            self.assertEqual(web_app.db.get_coaching_goal(goal_id)["status"], "completed")

    def test_voice_console_memory_ui_can_edit_and_delete_items(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            memory_id = web_app.db.create_memory_item(
                kind="preference",
                title="Reply preference",
                text="The user prefers concise answers first.",
                source_type="direct_voice_session",
                source_id="session-1",
                importance=0.8,
                confidence=0.9,
            )
            client = TestClient(web_app.app)

            response = client.get("/voice")

            self.assertEqual(response.status_code, 200)
            self.assertIn("data-memory-list", response.text)
            self.assertIn("Reply preference", response.text)
            self.assertIn("The user prefers concise answers first.", response.text)
            self.assertIn(f"/memory/{memory_id}/edit", response.text)
            self.assertIn(f"/memory/{memory_id}/delete", response.text)

            edited = client.post(
                f"/memory/{memory_id}/edit",
                data={
                    "kind": "preference",
                    "title": "Response style",
                    "text": "The user prefers concise answers with rationale second.",
                    "importance": "0.7",
                    "confidence": "0.85",
                    "valid_until": "",
                },
                follow_redirects=False,
            )

            self.assertEqual(edited.status_code, 303)
            updated = web_app.db.get_memory_item(memory_id)
            self.assertEqual(updated["title"], "Response style")
            self.assertIn("rationale second", updated["text"])
            self.assertEqual(web_app.db.search_memory_items("rationale")[0]["id"], memory_id)

            deleted = client.post(f"/memory/{memory_id}/delete", follow_redirects=False)

            self.assertEqual(deleted.status_code, 303)
            self.assertIsNone(web_app.db.get_memory_item(memory_id))
            self.assertEqual(web_app.db.search_memory_items("rationale"), [])

    def test_voice_console_inspector_shows_voice_settings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_REALTIME_HOST"] = "127.0.0.1"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "piper"
            os.environ["ATLAS_VOICE_PIPER_VOICE"] = "/models/voice.onnx"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.get("/voice")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Voice Settings", response.text)
            self.assertIn("127.0.0.1:8787", response.text)
            self.assertIn("/v1/realtime", response.text)
            self.assertIn("piper", response.text)
            self.assertIn("/models/voice.onnx", response.text)
            self.assertIn("24000 Hz", response.text)

    def test_voice_console_center_workbench_is_realtime_ready(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.get("/voice")

            self.assertEqual(response.status_code, 200)
            self.assertIn('data-voice-console="true"', response.text)
            self.assertIn('data-websocket-path="/v1/realtime"', response.text)
            self.assertIn('data-assistant-enabled="true"', response.text)
            self.assertIn("data-transcript-stream", response.text)
            self.assertIn('id="voice-prompt-input"', response.text)
            self.assertIn('id="voice-prompt-submit"', response.text)
            self.assertNotIn('id="voice-prompt-input" disabled', response.text)
            self.assertIn("/static/voice.js", response.text)

    def test_voice_console_exposes_tts_playground(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "piper"
            os.environ["ATLAS_VOICE_PIPER_VOICE"] = "/models/voice.onnx"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.get("/voice")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Voice Playground", response.text)
            self.assertIn("Text to Speech", response.text)
            self.assertIn("data-tts-playground", response.text)
            self.assertIn("/api/voice/playground/tts", response.text)

    def test_voice_tts_playground_api_synthesizes_and_logs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "piper"
            os.environ["ATLAS_VOICE_PIPER_VOICE"] = "/models/voice.onnx"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)
            audio_path = root / "tts.wav"

            with patch.object(
                web_app,
                "synthesize_with_piper",
                return_value=RealtimeAudio(
                    path=audio_path,
                    payload=b"RIFFtest",
                    media_type="audio/wav",
                    latency_ms=5,
                ),
            ) as synth_mock:
                response = client.post(
                    "/api/voice/playground/tts",
                    json={"text": "Hello from the playground"},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["provider"], "piper")
            self.assertEqual(payload["model"], "/models/voice.onnx")
            self.assertEqual(payload["latency_ms"], 5)
            self.assertEqual(payload["media_type"], "audio/wav")
            synth_mock.assert_called_once()
            tts_runs = [
                run for run in web_app.db.list_model_runs() if run["task"] == "voice_playground_tts"
            ]
            self.assertEqual(len(tts_runs), 1)
            self.assertEqual(tts_runs[0]["provider"], "piper")
            self.assertEqual(tts_runs[0]["latency_ms"], 5)

    def test_voice_console_transport_controls_are_wired(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.get("/voice")

            self.assertEqual(response.status_code, 200)
            self.assertIn("data-voice-transport", response.text)
            for action in (
                "start-call",
                "end-call",
                "mic",
                "pause",
                "private",
                "interrupt",
                "play",
            ):
                self.assertIn(f'data-transport-action="{action}"', response.text)
            self.assertIn("Start call", response.text)
            self.assertIn("End call", response.text)
            self.assertIn("data-call-title", response.text)
            self.assertIn("data-call-state", response.text)
            self.assertIn("data-voice-timer", response.text)
            self.assertIn("data-volume-output", response.text)
            self.assertIn('aria-pressed="false"', response.text)
            self.assertNotIn(
                'data-transport-action="start-call" aria-pressed="false" disabled', response.text
            )
            self.assertIn(
                'data-transport-action="end-call" aria-pressed="false" disabled', response.text
            )
            self.assertRegex(
                response.text,
                r'data-transport-action="mic"[^>]+aria-pressed="false"[^>]+disabled',
            )

    def test_voice_console_exposes_browser_playback_ui(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.get("/voice")

            self.assertEqual(response.status_code, 200)
            self.assertIn("data-response-audio", response.text)
            self.assertIn('aria-label="Assistant audio playback"', response.text)
            self.assertIn('preload="none"', response.text)
            self.assertIn("data-playback-status", response.text)

    def test_voice_static_script_streams_browser_mic_audio(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()

        self.assertIn("navigator.mediaDevices.getUserMedia", script)
        self.assertIn("AudioWorkletNode", script)
        self.assertIn("downsampleToPCM16", script)
        self.assertIn("input_audio_buffer.append", script)
        self.assertIn("input_audio_buffer.commit", script)
        self.assertIn("media_type", script)
        self.assertIn("audio/pcm", script)
        self.assertIn("echoCancellation: true", script)
        self.assertNotIn("new MediaRecorder", script)

    def test_voice_static_script_handles_transport_state(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()

        self.assertIn("data-voice-transport", script)
        self.assertIn("data-transport-action", script)
        self.assertIn("startCall", script)
        self.assertIn("endCall", script)
        self.assertIn("callActive", script)
        self.assertIn("data-volume-output", script)
        self.assertIn("data-voice-timer", script)
        self.assertIn("response.cancel", script)
        self.assertIn("setInterval", script)

    def test_voice_static_script_handles_browser_audio_playback(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()

        self.assertIn("data-response-audio", script)
        self.assertIn("data-playback-status", script)
        self.assertIn("response.audio.delta", script)
        self.assertIn("atob", script)
        self.assertIn("Blob", script)
        self.assertIn("URL.createObjectURL", script)
        self.assertIn("URL.revokeObjectURL", script)
        self.assertIn("audio.play", script)
        self.assertIn("playbackQueue", script)
        self.assertIn("audio.volume =", script)
        self.assertIn("response.audio.done", script)
        self.assertIn("response.interrupted", script)
        self.assertIn("response.cancelled", script)

    def test_voice_static_script_surfaces_tool_call_events(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()

        self.assertIn("response.tool_call.created", script)
        self.assertIn("response.tool_call.requires_confirmation", script)
        self.assertIn("Local action:", script)
        self.assertIn("Confirmation needed:", script)

    def test_voice_console_exposes_stt_playground(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.get("/voice")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Speech to Text", response.text)
            self.assertIn("data-stt-playground", response.text)
            self.assertIn("/api/voice/playground/stt", response.text)
            self.assertIn('name="stt_audio"', response.text)

    def test_voice_stt_playground_api_transcribes_and_logs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["ATLAS_VOICE_ASR_PROVIDER"] = "whisperx"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)
            audio_path = root / "input.wav"

            with patch.object(
                web_app,
                "transcribe_realtime_audio",
                return_value=("playground transcript", audio_path),
            ) as transcribe_mock:
                response = client.post(
                    "/api/voice/playground/stt",
                    files={"file": ("input.wav", b"RIFFtest", "audio/wav")},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["text"], "playground transcript")
            self.assertEqual(payload["provider"], "whisperx")
            self.assertEqual(payload["model"], "tiny.en")
            self.assertEqual(payload["audio_path"], str(audio_path))
            transcribe_mock.assert_called_once()
            stt_runs = [
                run for run in web_app.db.list_model_runs() if run["task"] == "voice_playground_stt"
            ]
            self.assertEqual(len(stt_runs), 1)
            self.assertEqual(stt_runs[0]["provider"], "whisperx")
            self.assertIsNotNone(stt_runs[0]["latency_ms"])

    def test_voice_console_exposes_model_playground(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.get("/voice")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Model Response", response.text)
            self.assertIn("data-model-playground", response.text)
            self.assertIn("/api/voice/playground/model", response.text)
            self.assertIn('name="model_text"', response.text)

    def test_voice_model_playground_api_replies_and_logs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.post(
                "/api/voice/playground/model",
                json={"text": "Summarize this update"},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["provider"], "stub")
            self.assertEqual(payload["model"], "qwen2.5-7b-instruct")
            self.assertEqual(payload["text"], "Atlas heard: Summarize this update")
            self.assertIsNotNone(payload["latency_ms"])
            model_runs = [
                run
                for run in web_app.db.list_model_runs()
                if run["task"] == "voice_playground_model"
            ]
            self.assertEqual(len(model_runs), 1)
            self.assertEqual(model_runs[0]["provider"], "stub")
            self.assertEqual(model_runs[0]["model"], "qwen2.5-7b-instruct")

    def test_voice_playground_outputs_have_latency_display_hooks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "piper"
            os.environ["ATLAS_VOICE_PIPER_VOICE"] = "/models/voice.onnx"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.get("/voice")

            self.assertEqual(response.status_code, 200)
            self.assertIn('data-playground-latency="tts"', response.text)
            self.assertIn('data-playground-latency="stt"', response.text)
            self.assertIn('data-playground-latency="model"', response.text)
            script = Path("atlas_voice/web/static/voice.js").read_text()
            self.assertGreaterEqual(script.count("payload.latency_ms"), 3)

    def test_assistant_sessions_api_lists_direct_voice_sessions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "false"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            ambient_id = web_app.db.create_ambient_session(
                mode="ambient",
                source="file",
                title="Ambient",
            )
            direct_id = web_app.db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Realtime",
            )
            utterance_id = web_app.db.add_utterance(
                session_id=direct_id,
                text="Hello Atlas",
                source_provider="text",
            )
            web_app.db.add_assistant_turn(
                session_id=direct_id,
                user_utterance_id=utterance_id,
                text="Hello back",
                model="qwen-local",
                latency_ms=9,
            )
            web_app.db.end_ambient_session(ambient_id)
            web_app.db.end_ambient_session(direct_id)
            client = TestClient(web_app.app)

            response = client.get("/api/assistant/sessions")
            all_response = client.get("/api/assistant/sessions", params={"mode": "all"})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["mode"], "direct_voice")
            self.assertEqual([session["id"] for session in payload["sessions"]], [direct_id])
            self.assertEqual(payload["sessions"][0]["utterance_count"], 1)
            self.assertEqual(payload["sessions"][0]["assistant_turn_count"], 1)
            self.assertEqual(all_response.status_code, 200)
            self.assertEqual(
                {session["id"] for session in all_response.json()["sessions"]},
                {ambient_id, direct_id},
            )

    def test_ambient_sessions_api_can_search_and_delete_sessions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            matching_id = web_app.db.create_ambient_session(
                mode="ambient",
                source="mic",
                title="Launch notes",
            )
            other_id = web_app.db.create_ambient_session(
                mode="ambient",
                source="mic",
                title="Kitchen notes",
            )
            web_app.db.add_utterance(
                session_id=matching_id,
                text="Launch blocker is resolved",
                source_provider="text",
            )
            web_app.db.add_utterance(
                session_id=other_id,
                text="Dinner planning",
                source_provider="text",
            )
            web_app.db.end_ambient_session(matching_id)
            web_app.db.end_ambient_session(other_id)
            client = TestClient(web_app.app)

            search = client.get("/api/ambient/sessions", params={"q": "blocker"})
            deleted = client.delete(f"/api/ambient/sessions/{matching_id}")
            remaining = client.get("/api/ambient/sessions")

            self.assertEqual(search.status_code, 200)
            self.assertEqual(
                [session["id"] for session in search.json()["sessions"]], [matching_id]
            )
            self.assertEqual(search.json()["query"], "blocker")
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(deleted.json()["session_count"], 1)
            self.assertEqual(deleted.json()["utterance_count"], 1)
            self.assertIsNone(web_app.db.get_ambient_session(matching_id))
            self.assertEqual(
                [session["id"] for session in remaining.json()["sessions"]], [other_id]
            )
            self.assertEqual(web_app.db.list_privacy_events()[0]["event_type"], "ambient.delete")

    def test_ambient_sessions_delete_returns_404_for_missing_session(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            response = client.delete("/api/ambient/sessions/missing")

            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["detail"], "Ambient session not found")

    def test_realtime_websocket_requires_assistant_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "false"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            with client.websocket_connect("/v1/realtime") as websocket:
                event = websocket.receive_json()

            self.assertEqual(event["type"], "error")
            self.assertIn("ATLAS_ASSISTANT_ENABLED", event["error"]["message"])
            self.assertEqual(web_app.db.list_ambient_sessions(10, mode="direct_voice"), [])

    def test_realtime_websocket_handles_text_and_persists_turn(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)

            with client.websocket_connect("/v1/realtime") as websocket:
                created = websocket.receive_json()
                session_id = created["session"]["id"]
                websocket.send_json({"type": "input_text", "text": "Hello Atlas"})
                events = _receive_until(websocket, "response.done")

            event_types = [event["type"] for event in events]
            self.assertEqual(created["type"], "session.created")
            self.assertIn("conversation.item.input_text.delta", event_types)
            self.assertIn("response.text.delta", event_types)
            self.assertIn("response.audio.done", event_types)
            self.assertEqual(web_app.db.get_ambient_session(session_id)["status"], "ended")
            self.assertEqual(web_app.db.list_utterances(session_id)[0]["text"], "Hello Atlas")
            self.assertEqual(
                web_app.db.list_assistant_turns(session_id)[0]["text"],
                "Atlas heard: Hello Atlas",
            )
            self.assertEqual(web_app.db.list_model_runs()[0]["task"], "realtime_chat")

    def test_realtime_websocket_accepts_interrupt_event_and_clears_pending_input(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)
            audio = base64.b64encode(b"\0\0" * 12).decode("ascii")

            with client.websocket_connect("/v1/realtime") as websocket:
                created = websocket.receive_json()
                websocket.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "content": [{"type": "input_text", "text": "Never send"}],
                        },
                    }
                )
                item_created = websocket.receive_json()
                websocket.send_json({"type": "input_audio_buffer.append", "audio": audio})
                appended = websocket.receive_json()
                websocket.send_json({"type": "response.cancel", "reason": "user_interrupt"})
                interrupted = websocket.receive_json()
                websocket.send_json({"type": "response.create"})
                error = websocket.receive_json()

            self.assertEqual(created["type"], "session.created")
            self.assertEqual(item_created["type"], "conversation.item.created")
            self.assertEqual(appended["type"], "input_audio_buffer.appended")
            self.assertEqual(interrupted["type"], "response.interrupted")
            self.assertEqual(interrupted["reason"], "user_interrupt")
            self.assertEqual(interrupted["cleared_audio_bytes"], 24)
            self.assertTrue(interrupted["cleared_pending_text"])
            self.assertEqual(interrupted["response"]["status"], "interrupted")
            self.assertEqual(error["type"], "error")
            self.assertIn("no pending user text", error["error"]["message"])
            self.assertEqual(web_app.db.list_utterances(created["session"]["id"]), [])

    def test_realtime_websocket_cancels_in_progress_response(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)
            started = threading.Event()
            release = threading.Event()

            def slow_reply(*_args, **_kwargs) -> RealtimeReply:
                started.set()
                release.wait(2.0)
                return RealtimeReply(text="Too late", latency_ms=200, tokens_in=1, tokens_out=2)

            with patch.object(web_app, "generate_realtime_reply", side_effect=slow_reply):
                with client.websocket_connect("/v1/realtime") as websocket:
                    created = websocket.receive_json()
                    session_id = created["session"]["id"]
                    websocket.send_json({"type": "input_text", "text": "Cancel this"})
                    events = _receive_until(websocket, "response.created")
                    self.assertTrue(started.wait(1.0))
                    response_id = events[-1]["response"]["id"]
                    websocket.send_json({"type": "response.cancel", "response_id": response_id})
                    cancelled = websocket.receive_json()
                    release.set()

            event_types = [event["type"] for event in events]
            self.assertIn("conversation.item.input_text.done", event_types)
            self.assertEqual(cancelled["type"], "response.cancelled")
            self.assertEqual(cancelled["response"]["id"], response_id)
            self.assertEqual(cancelled["response"]["status"], "cancelled")
            self.assertEqual(cancelled["reason"], "client_cancelled")
            self.assertEqual(web_app.db.list_utterances(session_id)[0]["text"], "Cancel this")
            self.assertEqual(web_app.db.list_assistant_turns(session_id), [])
            self.assertEqual(web_app.db.list_model_runs(), [])

    def test_realtime_websocket_barge_in_cancels_active_response_and_starts_new_turn(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)
            first_started = threading.Event()
            release_first = threading.Event()

            def reply_for_barge_in(text: str, *_args, **_kwargs) -> RealtimeReply:
                if text == "First request":
                    first_started.set()
                    release_first.wait(2.0)
                    return RealtimeReply(
                        text="Stale reply", latency_ms=200, tokens_in=2, tokens_out=2
                    )
                return RealtimeReply(
                    text=f"Fresh reply: {text}", latency_ms=5, tokens_in=2, tokens_out=3
                )

            with patch.object(web_app, "generate_realtime_reply", side_effect=reply_for_barge_in):
                with client.websocket_connect("/v1/realtime") as websocket:
                    created = websocket.receive_json()
                    session_id = created["session"]["id"]
                    websocket.send_json({"type": "input_text", "text": "First request"})
                    first_events = _receive_until(websocket, "response.created")
                    first_response_id = first_events[-1]["response"]["id"]
                    self.assertTrue(first_started.wait(1.0))
                    websocket.send_json({"type": "input_text", "text": "Second request"})
                    barge_events = _receive_until(websocket, "response.done")
                    release_first.set()

            event_types = [event["type"] for event in barge_events]
            cancelled = next(
                event for event in barge_events if event["type"] == "response.cancelled"
            )
            done = barge_events[-1]
            utterances = web_app.db.list_utterances(session_id)
            assistant_turns = web_app.db.list_assistant_turns(session_id)
            model_runs = web_app.db.list_model_runs()

            self.assertEqual(cancelled["response"]["id"], first_response_id)
            self.assertEqual(cancelled["reason"], "barge_in")
            self.assertIn("conversation.item.input_text.done", event_types)
            self.assertEqual(done["type"], "response.done")
            self.assertEqual(done["response"]["output"][0]["text"], "Fresh reply: Second request")
            self.assertEqual(
                [utterance["text"] for utterance in utterances], ["First request", "Second request"]
            )
            self.assertEqual(
                [turn["text"] for turn in assistant_turns], ["Fresh reply: Second request"]
            )
            self.assertEqual(len(model_runs), 1)
            self.assertEqual(model_runs[0]["input_ref"], f"utterance:{utterances[1]['id']}")

    def test_realtime_websocket_vad_speech_start_barges_in_before_audio_commit(self) -> None:
        keys = [
            "ATLAS_VOICE_DATA_DIR",
            "ATLAS_VOICE_MODELS_DIR",
            "ATLAS_VOICE_HF_CACHE",
            "ATLAS_VOICE_ASSISTANT_CONFIG",
            "ATLAS_ASSISTANT_ENABLED",
            "ATLAS_VOICE_TTS_PROVIDER",
            "ATLAS_VOICE_STUB_MODE",
            "ATLAS_VOICE_REALTIME_AUDIO_SAMPLE_RATE",
            "ATLAS_VOICE_REALTIME_AUDIO_CHANNELS",
            "ATLAS_VOICE_REALTIME_VAD_ENABLED",
            "ATLAS_VOICE_REALTIME_VAD_THRESHOLD",
            "ATLAS_VOICE_REALTIME_VAD_MIN_SPEECH_MS",
            "ATLAS_VOICE_REALTIME_VAD_SILENCE_MS",
            "WHISPERX_DEVICE",
            "WHISPERX_MODEL",
            "WHISPERX_COMPUTE_TYPE",
        ]
        old_env = {key: os.environ.get(key) for key in keys}
        try:
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
                os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
                os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
                os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                    root / "config" / "atlas.assistant.yaml"
                )
                os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
                os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
                os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
                os.environ["ATLAS_VOICE_REALTIME_AUDIO_SAMPLE_RATE"] = "16000"
                os.environ["ATLAS_VOICE_REALTIME_AUDIO_CHANNELS"] = "1"
                os.environ["ATLAS_VOICE_REALTIME_VAD_ENABLED"] = "true"
                os.environ["ATLAS_VOICE_REALTIME_VAD_THRESHOLD"] = "1000"
                os.environ["ATLAS_VOICE_REALTIME_VAD_MIN_SPEECH_MS"] = "100"
                os.environ["ATLAS_VOICE_REALTIME_VAD_SILENCE_MS"] = "100"
                os.environ["WHISPERX_DEVICE"] = "cpu"
                os.environ["WHISPERX_MODEL"] = "tiny.en"
                os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

                import atlas_voice.web.app as web_app

                web_app = importlib.reload(web_app)
                web_app.settings.ensure_directories()
                web_app.db.initialize()
                client = TestClient(web_app.app)
                started = threading.Event()
                release = threading.Event()

                def slow_reply(*_args, **_kwargs) -> RealtimeReply:
                    started.set()
                    release.wait(2.0)
                    return RealtimeReply(text="Too late", latency_ms=200, tokens_in=1, tokens_out=2)

                speech = base64.b64encode(_pcm_tone(16000, 0.2, amplitude=7000)).decode("ascii")

                with patch.object(web_app, "generate_realtime_reply", side_effect=slow_reply):
                    with client.websocket_connect("/v1/realtime") as websocket:
                        created = websocket.receive_json()
                        session_id = created["session"]["id"]
                        websocket.send_json({"type": "input_text", "text": "Keep talking"})
                        first_events = _receive_until(websocket, "response.created")
                        first_response_id = first_events[-1]["response"]["id"]
                        self.assertTrue(started.wait(1.0))
                        websocket.send_json({"type": "input_audio_buffer.append", "audio": speech})
                        appended = websocket.receive_json()
                        speech_started = websocket.receive_json()
                        cancelled = websocket.receive_json()
                        release.set()

                self.assertEqual(appended["type"], "input_audio_buffer.appended")
                self.assertEqual(speech_started["type"], "input_audio_buffer.speech_started")
                self.assertEqual(cancelled["type"], "response.cancelled")
                self.assertEqual(cancelled["response"]["id"], first_response_id)
                self.assertEqual(cancelled["reason"], "barge_in")
                self.assertEqual(web_app.db.list_utterances(session_id)[0]["text"], "Keep talking")
                self.assertEqual(web_app.db.list_assistant_turns(session_id), [])
                self.assertEqual(web_app.db.list_model_runs(), [])
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_realtime_websocket_emits_confirmation_gated_tool_call_events(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)
            reply = RealtimeReply(
                text="I can search locally.",
                latency_ms=8,
                tokens_in=2,
                tokens_out=4,
                tool_calls=[
                    {
                        "id": "call_search",
                        "name": "search_recordings",
                        "arguments": {"query": "Atlas"},
                        "mutating": False,
                    }
                ],
            )

            with patch.object(web_app, "generate_realtime_reply", return_value=reply):
                with client.websocket_connect("/v1/realtime") as websocket:
                    created = websocket.receive_json()
                    session_id = created["session"]["id"]
                    websocket.send_json({"type": "input_text", "text": "Find Atlas"})
                    events = _receive_until(websocket, "response.done")

            event_types = [event["type"] for event in events]
            created_call = next(
                event for event in events if event["type"] == "response.tool_call.created"
            )
            confirmation = next(
                event
                for event in events
                if event["type"] == "response.tool_call.requires_confirmation"
            )
            turn = web_app.db.list_assistant_turns(session_id)[0]

            self.assertIn("response.tool_call.created", event_types)
            self.assertIn("response.tool_call.requires_confirmation", event_types)
            self.assertEqual(created_call["tool_call"]["id"], "call_search")
            self.assertEqual(created_call["tool_call"]["name"], "search_recordings")
            self.assertEqual(confirmation["tool_call"]["status"], "requires_confirmation")
            self.assertEqual(turn["tool_calls"][0]["status"], "requires_confirmation")
            self.assertEqual(turn["tool_calls"][0]["arguments"], {"query": "Atlas"})

    def test_realtime_websocket_applies_tool_registry_permission_gates(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            assistant_config = root / "config" / "atlas.assistant.yaml"
            assistant_config.parent.mkdir(parents=True)
            assistant_config.write_text(
                "profiles:\n  direct_voice:\n    require_tool_confirmation: false\n"
            )
            tools_dir = root / "tools"
            tools_dir.mkdir()
            (tools_dir / "custom.yaml").write_text(
                "tools:\n"
                "  external_shell:\n"
                "    name: External Shell\n"
                "    description: Block external shell execution.\n"
                "    handler: atlas_voice.tools.external_shell\n"
                "    permission: deny\n"
                "    mutating: true\n"
            )
            env = {
                "ATLAS_VOICE_DATA_DIR": str(root / "data"),
                "ATLAS_VOICE_MODELS_DIR": str(root / "models"),
                "ATLAS_VOICE_HF_CACHE": str(root / "cache" / "huggingface"),
                "ATLAS_VOICE_ASSISTANT_CONFIG": str(assistant_config),
                "ATLAS_VOICE_TOOLS_DIR": str(tools_dir),
                "ATLAS_ASSISTANT_ENABLED": "true",
                "ATLAS_VOICE_TTS_PROVIDER": "none",
                "ATLAS_VOICE_STUB_MODE": "true",
                "WHISPERX_DEVICE": "cpu",
                "WHISPERX_MODEL": "tiny.en",
                "WHISPERX_COMPUTE_TYPE": "int8",
            }
            with patch.dict(os.environ, env, clear=False):
                import atlas_voice.web.app as web_app

                web_app = importlib.reload(web_app)
                web_app.settings.ensure_directories()
                web_app.db.initialize()
                client = TestClient(web_app.app)
                reply = RealtimeReply(
                    text="I can call local tools.",
                    latency_ms=8,
                    tokens_in=2,
                    tokens_out=4,
                    tool_calls=[
                        {
                            "id": "call_search",
                            "name": "search_recordings",
                            "arguments": {"query": "Atlas"},
                        },
                        {
                            "id": "call_purge",
                            "name": "privacy_purge",
                            "arguments": {"keyword": "secret"},
                        },
                        {
                            "id": "call_shell",
                            "name": "external_shell",
                            "arguments": {"cmd": "rm -rf /"},
                        },
                        {"id": "call_missing", "name": "missing_tool", "arguments": {}},
                    ],
                )

                with patch.object(web_app, "generate_realtime_reply", return_value=reply):
                    with client.websocket_connect("/v1/realtime") as websocket:
                        created = websocket.receive_json()
                        session_id = created["session"]["id"]
                        websocket.send_json({"type": "input_text", "text": "Use tools"})
                        events = _receive_until(websocket, "response.done")

                by_type = {}
                for event in events:
                    by_type.setdefault(event["type"], []).append(event)
                stored_calls = web_app.db.list_assistant_turns(session_id)[0]["tool_calls"]
                stored_by_id = {call["id"]: call for call in stored_calls}

                ready = by_type["response.tool_call.ready"][0]["tool_call"]
                confirmation = by_type["response.tool_call.requires_confirmation"][0]["tool_call"]
                denied = [event["tool_call"] for event in by_type["response.tool_call.denied"]]

                self.assertEqual(ready["id"], "call_search")
                self.assertEqual(ready["status"], "ready")
                self.assertEqual(confirmation["id"], "call_purge")
                self.assertEqual(confirmation["status"], "requires_confirmation")
                self.assertEqual({call["id"] for call in denied}, {"call_shell", "call_missing"})
                self.assertEqual(stored_by_id["call_search"]["status"], "ready")
                self.assertEqual(stored_by_id["call_purge"]["status"], "requires_confirmation")
                self.assertEqual(stored_by_id["call_shell"]["status"], "denied")
                self.assertEqual(stored_by_id["call_shell"]["reason"], "permission_denied")
                self.assertEqual(stored_by_id["call_missing"]["status"], "denied")
                self.assertEqual(stored_by_id["call_missing"]["reason"], "unknown_tool")

    def test_realtime_websocket_uses_tts_sidecar_and_logs_model_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "faster-qwen3-tts"
            os.environ["ATLAS_TTS_MODEL"] = "faster-qwen3-tts-0.6b"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)
            audio_path = root / "assistant.wav"

            with patch.object(
                web_app,
                "synthesize_with_tts_sidecar",
                return_value=RealtimeAudio(
                    path=audio_path,
                    payload=b"RIFFtest",
                    media_type="audio/wav",
                    latency_ms=7,
                ),
            ) as synth_mock:
                with client.websocket_connect("/v1/realtime") as websocket:
                    created = websocket.receive_json()
                    websocket.send_json({"type": "input_text", "text": "Hello Atlas"})
                    events = _receive_until(websocket, "response.done")

            event_types = [event["type"] for event in events]
            self.assertEqual(created["session"]["output_audio_format"], "wav")
            self.assertIn("response.audio.delta", event_types)
            self.assertIn("response.audio.done", event_types)
            synth_mock.assert_called_once()
            tts_runs = [
                run for run in web_app.db.list_model_runs() if run["task"] == "realtime_tts"
            ]
            self.assertEqual(len(tts_runs), 1)
            self.assertEqual(tts_runs[0]["provider"], "faster-qwen3-tts")
            self.assertEqual(tts_runs[0]["model"], "faster-qwen3-tts-0.6b")
            self.assertEqual(tts_runs[0]["latency_ms"], 7)

    def test_realtime_websocket_uses_tts_sidecar_by_default_profile(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ.pop("ATLAS_VOICE_TTS_PROVIDER", None)
            os.environ.pop("ATLAS_TTS_MODEL", None)
            os.environ["ATLAS_VOICE_ASR_PROVIDER"] = "whisperx"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            # Point env file at a non-existent path so load_dotenv() is a no-op.
            # This prevents the workspace .env from polluting the test fixture.
            os.environ["ATLAS_VOICE_ENV_FILE"] = str(root / ".env.nonexistent")
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)
            audio_path = root / "assistant.wav"

            with patch.object(
                web_app,
                "synthesize_with_tts_sidecar",
                return_value=RealtimeAudio(
                    path=audio_path,
                    payload=b"RIFFtest",
                    media_type="audio/wav",
                    latency_ms=7,
                ),
            ) as synth_mock:
                with client.websocket_connect("/v1/realtime") as websocket:
                    created = websocket.receive_json()
                    websocket.send_json({"type": "input_text", "text": "Hello Atlas"})
                    events = _receive_until(websocket, "response.done")

            event_types = [event["type"] for event in events]
            self.assertEqual(created["session"]["output_audio_format"], "wav")
            self.assertIn("response.audio.delta", event_types)
            self.assertIn("response.audio.done", event_types)
            synth_mock.assert_called_once()
            tts_runs = [
                run for run in web_app.db.list_model_runs() if run["task"] == "realtime_tts"
            ]
            self.assertEqual(len(tts_runs), 1)
            self.assertEqual(tts_runs[0]["provider"], "faster-qwen3-tts")
            self.assertEqual(tts_runs[0]["model"], "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
            self.assertEqual(tts_runs[0]["latency_ms"], 7)

    def test_realtime_websocket_accepts_audio_buffer_with_transcript(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["ATLAS_VOICE_ASR_PROVIDER"] = "whisperx"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)
            audio = base64.b64encode(b"\0\0" * 12).decode("ascii")

            with client.websocket_connect("/v1/realtime") as websocket:
                created = websocket.receive_json()
                session_id = created["session"]["id"]
                websocket.send_json({"type": "input_audio_buffer.append", "audio": audio})
                appended = websocket.receive_json()
                websocket.send_json(
                    {
                        "type": "input_audio_buffer.commit",
                        "transcript": "Audio hello",
                    }
                )
                events = _receive_until(websocket, "response.done")

            event_types = [event["type"] for event in events]
            self.assertEqual(appended["type"], "input_audio_buffer.appended")
            self.assertIn("conversation.item.input_audio_transcription.delta", event_types)
            utterance = web_app.db.list_utterances(session_id)[0]
            self.assertEqual(utterance["text"], "Audio hello")
            self.assertEqual(utterance["source_provider"], "whisperx")

    def test_realtime_websocket_streams_input_audio_transcript_deltas_before_commit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
            os.environ["WHISPERX_DEVICE"] = "cpu"
            os.environ["WHISPERX_MODEL"] = "tiny.en"
            os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            client = TestClient(web_app.app)
            audio = base64.b64encode(b"\0\0" * 12).decode("ascii")

            with patch.object(
                web_app,
                "transcribe_realtime_audio",
                side_effect=AssertionError("commit should reuse streaming transcript"),
            ):
                with client.websocket_connect("/v1/realtime") as websocket:
                    created = websocket.receive_json()
                    session_id = created["session"]["id"]
                    websocket.send_json(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": audio,
                            "transcript_delta": "Hello",
                        }
                    )
                    first = websocket.receive_json()
                    first_delta = websocket.receive_json()
                    websocket.send_json(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": audio,
                            "transcript_delta": " Atlas",
                            "transcript_final": True,
                        }
                    )
                    second = websocket.receive_json()
                    second_delta = websocket.receive_json()
                    websocket.send_json({"type": "input_audio_buffer.commit"})
                    events = _receive_until(websocket, "response.done")

            event_types = [event["type"] for event in events]
            self.assertEqual(first["type"], "input_audio_buffer.appended")
            self.assertEqual(second["type"], "input_audio_buffer.appended")
            self.assertEqual(
                first_delta["type"], "conversation.item.input_audio_transcription.delta"
            )
            self.assertEqual(first_delta["delta"], "Hello")
            self.assertFalse(first_delta["final"])
            self.assertEqual(
                second_delta["type"], "conversation.item.input_audio_transcription.delta"
            )
            self.assertEqual(second_delta["delta"], " Atlas")
            self.assertTrue(second_delta["final"])
            self.assertIn("input_audio_buffer.committed", event_types)
            self.assertIn("conversation.item.input_audio_transcription.done", event_types)
            utterance = web_app.db.list_utterances(session_id)[0]
            self.assertEqual(utterance["text"], "Hello Atlas")
            self.assertEqual(utterance["source_provider"], "streaming_stt")

    def test_realtime_websocket_auto_commits_audio_after_vad_silence(self) -> None:
        keys = [
            "ATLAS_VOICE_DATA_DIR",
            "ATLAS_VOICE_MODELS_DIR",
            "ATLAS_VOICE_HF_CACHE",
            "ATLAS_VOICE_ASSISTANT_CONFIG",
            "ATLAS_ASSISTANT_ENABLED",
            "ATLAS_VOICE_TTS_PROVIDER",
            "ATLAS_VOICE_STUB_MODE",
            "ATLAS_VOICE_REALTIME_AUDIO_SAMPLE_RATE",
            "ATLAS_VOICE_REALTIME_AUDIO_CHANNELS",
            "ATLAS_VOICE_REALTIME_VAD_ENABLED",
            "ATLAS_VOICE_REALTIME_VAD_THRESHOLD",
            "ATLAS_VOICE_REALTIME_VAD_MIN_SPEECH_MS",
            "ATLAS_VOICE_REALTIME_VAD_SILENCE_MS",
            "WHISPERX_DEVICE",
            "WHISPERX_MODEL",
            "WHISPERX_COMPUTE_TYPE",
        ]
        old_env = {key: os.environ.get(key) for key in keys}
        try:
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
                os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
                os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
                os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                    root / "config" / "atlas.assistant.yaml"
                )
                os.environ["ATLAS_ASSISTANT_ENABLED"] = "true"
                os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
                os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
                os.environ["ATLAS_VOICE_REALTIME_AUDIO_SAMPLE_RATE"] = "16000"
                os.environ["ATLAS_VOICE_REALTIME_AUDIO_CHANNELS"] = "1"
                os.environ["ATLAS_VOICE_REALTIME_VAD_ENABLED"] = "true"
                os.environ["ATLAS_VOICE_REALTIME_VAD_THRESHOLD"] = "1000"
                os.environ["ATLAS_VOICE_REALTIME_VAD_MIN_SPEECH_MS"] = "100"
                os.environ["ATLAS_VOICE_REALTIME_VAD_SILENCE_MS"] = "100"
                os.environ["WHISPERX_DEVICE"] = "cpu"
                os.environ["WHISPERX_MODEL"] = "tiny.en"
                os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

                import atlas_voice.web.app as web_app

                web_app = importlib.reload(web_app)
                web_app.settings.ensure_directories()
                web_app.db.initialize()
                client = TestClient(web_app.app)
                speech = base64.b64encode(_pcm_tone(16000, 0.2, amplitude=7000)).decode("ascii")
                silence = base64.b64encode(_pcm_silence(16000, 0.15)).decode("ascii")

                with client.websocket_connect("/v1/realtime") as websocket:
                    created = websocket.receive_json()
                    session_id = created["session"]["id"]
                    websocket.send_json({"type": "input_audio_buffer.append", "audio": speech})
                    first_append = websocket.receive_json()
                    speech_started = websocket.receive_json()
                    websocket.send_json({"type": "input_audio_buffer.append", "audio": silence})
                    events = _receive_until(websocket, "response.done")

                event_types = [event["type"] for event in events]
                self.assertEqual(first_append["type"], "input_audio_buffer.appended")
                self.assertEqual(speech_started["type"], "input_audio_buffer.speech_started")
                self.assertIn("input_audio_buffer.speech_stopped", event_types)
                self.assertIn("input_audio_buffer.committed", event_types)
                self.assertIn("conversation.item.input_audio_transcription.delta", event_types)
                utterance = web_app.db.list_utterances(session_id)[0]
                self.assertEqual(utterance["text"], "Audio input received.")
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_runtime_settings_form_persists_provider_config(self) -> None:
        keys = [
            "ATLAS_VOICE_ENV_FILE",
            "ATLAS_VOICE_DATA_DIR",
            "ATLAS_VOICE_MODELS_DIR",
            "ATLAS_VOICE_HF_CACHE",
            "ATLAS_VOICE_ASSISTANT_CONFIG",
            "ATLAS_VOICE_TTS_PROVIDER",
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
                os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                    root / "config" / "atlas.assistant.yaml"
                )
                os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
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

    def test_runtime_settings_form_persists_summary_model_config(self) -> None:
        keys = [
            "ATLAS_VOICE_ENV_FILE",
            "ATLAS_VOICE_DATA_DIR",
            "ATLAS_VOICE_MODELS_DIR",
            "ATLAS_VOICE_HF_CACHE",
            "ATLAS_VOICE_ASSISTANT_CONFIG",
            "ATLAS_VOICE_TTS_PROVIDER",
            "ATLAS_VOICE_STUB_MODE",
            "ATLAS_VOICE_ASR_PROVIDER",
            "ATLAS_VOICE_ASR_MODEL",
            "ATLAS_VOICE_DIARIZATION_PROVIDER",
            "LLM_MODEL",
            "LLM_BASE_URL",
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
                assistant_path = root / "config" / "atlas.assistant.yaml"
                os.environ["ATLAS_VOICE_ENV_FILE"] = str(env_file)
                os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
                os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
                os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
                os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(assistant_path)
                os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
                os.environ["ATLAS_VOICE_STUB_MODE"] = "true"
                os.environ["ATLAS_VOICE_ASR_PROVIDER"] = "whisperx"
                os.environ["ATLAS_VOICE_ASR_MODEL"] = ""
                os.environ["ATLAS_VOICE_DIARIZATION_PROVIDER"] = "pyannote"
                os.environ["LLM_MODEL"] = "qwen-local"
                os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8080/v1/chat/completions"
                os.environ["WHISPERX_DEVICE"] = "cpu"
                os.environ["WHISPERX_MODEL"] = "tiny.en"
                os.environ["WHISPERX_COMPUTE_TYPE"] = "int8"

                import atlas_voice.web.app as web_app

                web_app = importlib.reload(web_app)
                web_app.settings.ensure_directories()
                web_app.db.initialize()
                web_app._restart_worker_if_idle = lambda: "restarted"
                client = TestClient(web_app.app)

                dashboard = client.get("/")
                response = client.post(
                    "/settings/runtime",
                    data={
                        "asr_provider": "whisperx",
                        "asr_model": "tiny.en",
                        "diarization_provider": "pyannote",
                        "summary_llm_model": "qwen-27b-custom",
                        "summary_llm_base_url": "http://127.0.0.1:8088/v1/chat/completions",
                    },
                )

                self.assertEqual(dashboard.status_code, 200)
                self.assertIn('name="summary_llm_model"', dashboard.text)
                self.assertIn('value="qwen-27b-instruct"', dashboard.text)
                self.assertEqual(response.status_code, 200)
                self.assertIn("Runtime settings saved", response.text)
                self.assertEqual(web_app.processor.settings.llm_model, "qwen-27b-custom")
                self.assertEqual(
                    web_app.processor.settings.llm_base_url,
                    "http://127.0.0.1:8088/v1/chat/completions",
                )
                saved_config = assistant_path.read_text()
                self.assertIn("llm_profiles:", saved_config)
                self.assertIn("summarization:", saved_config)
                self.assertIn("qwen-summary:", saved_config)
                self.assertIn("model: qwen-27b-custom", saved_config)
                self.assertIn("base_url: http://127.0.0.1:8088/v1/chat/completions", saved_config)
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
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
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
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(
                root / "config" / "atlas.assistant.yaml"
            )
            os.environ["ATLAS_VOICE_TTS_PROVIDER"] = "none"
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


def _pcm_tone(sample_rate: int, seconds: float, *, amplitude: int) -> bytes:
    samples = array("h")
    for index in range(int(sample_rate * seconds)):
        value = int(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate))
        samples.append(value)
    return samples.tobytes()


def _pcm_silence(sample_rate: int, seconds: float) -> bytes:
    return array("h", [0] * int(sample_rate * seconds)).tobytes()


def _receive_until(websocket, event_type: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    while True:
        event = websocket.receive_json()
        events.append(event)
        if event.get("type") == event_type:
            return events


if __name__ == "__main__":
    unittest.main()
