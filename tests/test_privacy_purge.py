from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
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

    def test_privacy_retention_dry_run_and_confirmed_apply(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "ATLAS_VOICE_DATA_DIR": str(root / "data"),
                "ATLAS_VOICE_MODELS_DIR": str(root / "models"),
                "ATLAS_VOICE_HF_CACHE": str(root / "cache" / "huggingface"),
                "ATLAS_VOICE_AMBIENT_RAW_AUDIO_RETENTION_DAYS": "1",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = Settings.from_env()
                settings.ensure_directories()
                db = Database(settings.db_path)
                db.initialize()
                session_id = db.create_ambient_session(
                    mode="ambient",
                    source="mic",
                    retention_policy="retain_audio",
                    title="Expired audio",
                )
                db.end_ambient_session(session_id)
                old_time = (datetime.now(UTC) - timedelta(days=3)).isoformat()
                with db.connect() as conn:
                    conn.execute(
                        """
                        UPDATE ambient_sessions
                        SET started_at = ?, ended_at = ?, created_at = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (old_time, old_time, old_time, old_time, session_id),
                    )
                artifact_dir = settings.artifacts_dir / "ambient" / session_id
                artifact_dir.mkdir(parents=True)
                (artifact_dir / "segment.wav").write_text("audio")

                dry_run = _run_cli(["privacy", "retention"])

                self.assertEqual(dry_run["code"], 0)
                self.assertIn("privacy retention dry run", dry_run["stdout"])
                self.assertTrue(artifact_dir.exists())

                confirmed = _run_cli(["privacy", "retention", "--yes"])

                self.assertEqual(confirmed["code"], 0)
                self.assertIn("privacy retention applied", confirmed["stdout"])
                self.assertFalse(artifact_dir.exists())
                self.assertEqual(db.list_privacy_events()[0]["event_type"], "privacy.retention")

    def test_memory_extract_ambient_dry_run_and_confirmed_apply(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "ATLAS_VOICE_DATA_DIR": str(root / "data"),
                "ATLAS_VOICE_MODELS_DIR": str(root / "models"),
                "ATLAS_VOICE_HF_CACHE": str(root / "cache" / "huggingface"),
            }
            with patch.dict(os.environ, env, clear=True):
                settings = Settings.from_env()
                settings.ensure_directories()
                db = Database(settings.db_path)
                db.initialize()
                session_id = db.create_ambient_session(
                    mode="ambient",
                    source="mic",
                    title="Ambient",
                )
                db.add_utterance(
                    session_id=session_id,
                    text="Remember that the desk mic is quieter than the BRIO.",
                    source_provider="text",
                )
                db.end_ambient_session(session_id)

                dry_run = _run_cli(["memory", "extract-ambient", "--session", session_id])

                self.assertEqual(dry_run["code"], 0)
                self.assertIn("memory extraction dry run", dry_run["stdout"])
                self.assertEqual(db.list_memory_items(), [])

                confirmed = _run_cli(["memory", "extract-ambient", "--session", session_id, "--yes"])

                self.assertEqual(confirmed["code"], 0)
                self.assertIn("memory extraction applied", confirmed["stdout"])
                self.assertEqual(len(db.list_memory_items()), 1)

    def test_memory_extract_direct_dry_run_and_confirmed_apply(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "ATLAS_VOICE_DATA_DIR": str(root / "data"),
                "ATLAS_VOICE_MODELS_DIR": str(root / "models"),
                "ATLAS_VOICE_HF_CACHE": str(root / "cache" / "huggingface"),
            }
            with patch.dict(os.environ, env, clear=True):
                settings = Settings.from_env()
                settings.ensure_directories()
                db = Database(settings.db_path)
                db.initialize()
                session_id = db.create_ambient_session(
                    mode="direct_voice",
                    source="websocket",
                    title="Direct",
                )
                db.add_utterance(
                    session_id=session_id,
                    text="I prefer direct answers before detailed rationale.",
                    source_provider="text",
                )
                db.end_ambient_session(session_id)

                dry_run = _run_cli(["memory", "extract-direct", "--session", session_id])

                self.assertEqual(dry_run["code"], 0)
                self.assertIn("memory extraction dry run", dry_run["stdout"])
                self.assertEqual(db.list_memory_items(), [])

                confirmed = _run_cli(["memory", "extract-direct", "--session", session_id, "--yes"])

                self.assertEqual(confirmed["code"], 0)
                self.assertIn("memory extraction applied", confirmed["stdout"])
                self.assertEqual(len(db.list_memory_items()), 1)

    def test_coaching_daily_summary_dry_run_and_confirmed_apply(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "ATLAS_VOICE_DATA_DIR": str(root / "data"),
                "ATLAS_VOICE_MODELS_DIR": str(root / "models"),
                "ATLAS_VOICE_HF_CACHE": str(root / "cache" / "huggingface"),
            }
            with patch.dict(os.environ, env, clear=True):
                settings = Settings.from_env()
                settings.ensure_directories()
                db = Database(settings.db_path)
                db.initialize()
                session_id = db.create_ambient_session(
                    mode="direct_voice",
                    source="websocket",
                    title="Daily voice check",
                )
                db.add_utterance(
                    session_id=session_id,
                    text="What should I improve tomorrow?",
                    source_provider="text",
                )
                db.end_ambient_session(session_id)

                dry_run = _run_cli(["coaching", "daily", "--date", datetime.now(UTC).date().isoformat()])

                self.assertEqual(dry_run["code"], 0)
                self.assertIn("coaching daily dry run", dry_run["stdout"])
                self.assertEqual(db.list_feedback_events(category="daily_summary"), [])

                confirmed = _run_cli([
                    "coaching",
                    "daily",
                    "--date",
                    datetime.now(UTC).date().isoformat(),
                    "--yes",
                ])

                self.assertEqual(confirmed["code"], 0)
                self.assertIn("coaching daily stored", confirmed["stdout"])
                self.assertEqual(len(db.list_feedback_events(category="daily_summary")), 1)

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
