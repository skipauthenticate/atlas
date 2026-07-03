from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from atlas_voice.anythingllm import (
    AnythingLLMConfigError,
    anythingllm_api_url,
    build_anythingllm_document,
    sync_recording_to_anythingllm,
)
from atlas_voice.config import Settings
from atlas_voice.database import Database


class AnythingLLMTests(unittest.TestCase):
    def test_build_document_includes_summary_transcript_and_no_local_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "db.sqlite")
            db.initialize()
            recording_id = db.create_recording(root / "private" / "meeting.wav", title="Meeting")
            db.update_recording(recording_id, status="done", sha256="abc123")
            db.enqueue_job(recording_id, "summarize")
            db.replace_segments(
                recording_id,
                [
                    {
                        "start": 0,
                        "end": 2,
                        "speaker": "SPEAKER_00",
                        "text": "Remember the launch checklist.",
                    }
                ],
            )
            db.save_summary(
                recording_id,
                "Overview\nLaunch checklist was discussed.",
                model="qwen-local",
                template_id="meeting",
                chunks=[{"chunk_index": 1, "text": "Checklist details."}],
            )

            document = build_anythingllm_document(db, recording_id, workspace_slug="notes")

        self.assertEqual(document["addToWorkspaces"], "notes")
        self.assertEqual(document["metadata"]["title"], "Meeting")
        text = document["textContent"]
        self.assertIn("## Summary", text)
        self.assertIn("Launch checklist was discussed.", text)
        self.assertIn("## Summary Chunks", text)
        self.assertIn("[00:00-00:02] SPEAKER_00: Remember the launch checklist.", text)
        self.assertNotIn(str(root), text)
        self.assertNotIn("source_path", text)
        self.assertNotIn("abc123", text)

    def test_sync_posts_raw_text_document_with_bearer_token(self) -> None:
        requests: list[dict[str, object]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": json.loads(body.decode("utf-8")),
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    b'{"success":true,"documents":[{"location":"custom-documents/meeting.json"}]}'
                )

            def log_message(self, _format: str, *_args: object) -> None:
                return

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "db.sqlite")
            db.initialize()
            recording_id = db.create_recording(root / "meeting.wav", title="Meeting")
            db.replace_segments(
                recording_id,
                [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hello."}],
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                settings = _settings(
                    root,
                    anythingllm_base_url=f"http://127.0.0.1:{server.server_port}/api",
                    anythingllm_api_key="test-key",
                    anythingllm_workspace_slug="notes",
                )
                result = sync_recording_to_anythingllm(db, recording_id, settings)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertTrue(result["success"])
        self.assertEqual(requests[0]["path"], "/api/v1/document/raw-text")
        self.assertEqual(requests[0]["authorization"], "Bearer test-key")
        body = requests[0]["body"]
        self.assertEqual(body["addToWorkspaces"], "notes")
        self.assertIn("Hello.", body["textContent"])

    def test_sync_requires_api_key_and_workspace(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "db.sqlite")
            db.initialize()
            recording_id = db.create_recording(root / "meeting.wav")
            db.replace_segments(
                recording_id,
                [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "Hello."}],
            )
            settings = _settings(root, anythingllm_api_key=None, anythingllm_workspace_slug=None)

            with self.assertRaises(AnythingLLMConfigError):
                sync_recording_to_anythingllm(db, recording_id, settings)

    def test_api_url_accepts_host_api_or_api_v1_base(self) -> None:
        self.assertEqual(
            anythingllm_api_url("http://127.0.0.1:3001", "/v1/document/raw-text"),
            "http://127.0.0.1:3001/api/v1/document/raw-text",
        )
        self.assertEqual(
            anythingllm_api_url("http://127.0.0.1:3001/api", "/v1/document/raw-text"),
            "http://127.0.0.1:3001/api/v1/document/raw-text",
        )
        self.assertEqual(
            anythingllm_api_url("http://127.0.0.1:3001/api/v1", "/v1/document/raw-text"),
            "http://127.0.0.1:3001/api/v1/document/raw-text",
        )


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
