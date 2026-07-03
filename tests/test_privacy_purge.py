from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

from atlas_voice.cli import main
from atlas_voice.config import Settings
from atlas_voice.database import Database


class PrivacyPurgeCliTests(unittest.TestCase):
    def test_privacy_purge_dry_run_and_confirmed_session_delete(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "ATLAS_VOICE_DATA_DIR": str(root / "data"),
                "ATLAS_VOICE_MODELS_DIR": str(root / "models"),
                "ATLAS_VOICE_HF_CACHE": str(root / "cache" / "huggingface"),
                "ATLAS_VOICE_ASSISTANT_CONFIG": str(root / "config" / "atlas.assistant.yaml"),
                "ATLAS_VOICE_TTS_PROVIDER": "none",
                "ATLAS_VOICE_STUB_MODE": "true",
                "LLM_BASE_URL": "http://127.0.0.1:8080/v1/chat/completions",
                "WHISPERX_DEVICE": "cpu",
                "WHISPERX_MODEL": "tiny.en",
                "WHISPERX_COMPUTE_TYPE": "int8",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = Settings.from_env()
                settings.ensure_directories()
                db = Database(settings.db_path)
                db.initialize()
                session_id = db.create_ambient_session(
                    mode="direct_voice",
                    source="websocket",
                    title="Private session",
                )
                db.add_utterance(
                    session_id=session_id,
                    text="delete me",
                    source_provider="text",
                )
                db.end_ambient_session(session_id)

                dry_run = _run_cli(["privacy", "purge", "--session", session_id])

                self.assertEqual(dry_run["code"], 0)
                self.assertIn("dry run", dry_run["stdout"])
                self.assertIsNotNone(db.get_ambient_session(session_id))

                confirmed = _run_cli(["privacy", "purge", "--session", session_id, "--yes"])
                self.assertEqual(confirmed["code"], 0)
                self.assertIn("purged 1 session", confirmed["stdout"])
                self.assertIsNone(db.get_ambient_session(session_id))
                self.assertEqual(db.list_privacy_events()[0]["event_type"], "privacy.purge")

    def test_privacy_purge_requires_at_least_one_filter(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "ATLAS_VOICE_DATA_DIR": str(root / "data"),
                "ATLAS_VOICE_MODELS_DIR": str(root / "models"),
                "ATLAS_VOICE_HF_CACHE": str(root / "cache" / "huggingface"),
            }
            with patch.dict(os.environ, env, clear=True):
                result = _run_cli(["privacy", "purge", "--yes"])

        self.assertEqual(result["code"], 1)
        self.assertIn("at least one purge filter", result["stderr"])


def _run_cli(argv: list[str]) -> dict[str, object]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return {"code": code, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}


if __name__ == "__main__":
    unittest.main()
