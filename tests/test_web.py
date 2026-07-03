from pathlib import Path
import base64
from tempfile import TemporaryDirectory
import importlib
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from atlas_voice.realtime import RealtimeAudio


class WebTests(unittest.TestCase):
    def test_html_pages_render_with_current_starlette_signature(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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
            self.assertIn("Test Audio", dashboard.text)
            self.assertIn("CPU", dashboard.text)
            self.assertIn("tiny.en", dashboard.text)
            self.assertIn("Privacy", dashboard.text)
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


    def test_assistant_health_api_reports_local_components(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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

    def test_assistant_privacy_api_surfaces_external_endpoint_errors(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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
            self.assertIn("Transport", response.text)
            self.assertIn("Privacy", response.text)
            self.assertIn("Local Models", response.text)

    def test_voice_console_inspector_shows_voice_settings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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
            self.assertIn('data-transcript-stream', response.text)
            self.assertIn('id="voice-prompt-input"', response.text)
            self.assertIn('id="voice-prompt-submit"', response.text)
            self.assertNotIn('id="voice-prompt-input" disabled', response.text)
            self.assertIn('/static/voice.js', response.text)

    def test_assistant_sessions_api_lists_direct_voice_sessions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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

    def test_realtime_websocket_requires_assistant_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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


    def test_realtime_websocket_uses_tts_sidecar_and_logs_model_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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
            tts_runs = [run for run in web_app.db.list_model_runs() if run["task"] == "realtime_tts"]
            self.assertEqual(len(tts_runs), 1)
            self.assertEqual(tts_runs[0]["provider"], "faster-qwen3-tts")
            self.assertEqual(tts_runs[0]["model"], "faster-qwen3-tts-0.6b")
            self.assertEqual(tts_runs[0]["latency_ms"], 7)

    def test_realtime_websocket_accepts_audio_buffer_with_transcript(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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
                session_id = created["session"]["id"]
                websocket.send_json({"type": "input_audio_buffer.append", "audio": audio})
                appended = websocket.receive_json()
                websocket.send_json({
                    "type": "input_audio_buffer.commit",
                    "transcript": "Audio hello",
                })
                events = _receive_until(websocket, "response.done")

            event_types = [event["type"] for event in events]
            self.assertEqual(appended["type"], "input_audio_buffer.appended")
            self.assertIn("conversation.item.input_audio_transcription.delta", event_types)
            utterance = web_app.db.list_utterances(session_id)[0]
            self.assertEqual(utterance["text"], "Audio hello")
            self.assertEqual(utterance["source_provider"], "whisperx")

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
                os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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

    def test_template_change_queues_resummary_and_preserves_cached_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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
            os.environ["ATLAS_VOICE_ASSISTANT_CONFIG"] = str(root / "config" / "atlas.assistant.yaml")
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


def _receive_until(websocket, event_type: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    while True:
        event = websocket.receive_json()
        events.append(event)
        if event.get("type") == event_type:
            return events


if __name__ == "__main__":
    unittest.main()
