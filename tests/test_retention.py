from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from atlas_voice.config import Settings
from atlas_voice.database import Database
from atlas_voice.retention import apply_ambient_retention


class AmbientRetentionTests(unittest.TestCase):
    def test_apply_retention_dry_run_then_deletes_expired_audio_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = replace(
                _settings(root),
                ambient_raw_audio_retention_days=1,
                ambient_transcript_retention_days=None,
            )
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()
            now = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
            session_id = _ambient_session(db, now - timedelta(days=3))
            artifact_dir = _artifact_dir(settings, session_id)
            (artifact_dir / "segment.wav").write_text("audio")

            dry_run = apply_ambient_retention(settings, db, now=now, dry_run=True)

            self.assertEqual(dry_run.audio_artifact_count, 1)
            self.assertEqual(dry_run.session_count, 0)
            self.assertTrue(artifact_dir.exists())
            self.assertIsNotNone(db.get_ambient_session(session_id))

            applied = apply_ambient_retention(settings, db, now=now, dry_run=False)

            self.assertEqual(applied.audio_artifact_count, 1)
            self.assertEqual(applied.session_count, 0)
            self.assertFalse(artifact_dir.exists())
            self.assertIsNotNone(db.get_ambient_session(session_id))

    def test_apply_retention_deletes_expired_transcripts_and_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = replace(
                _settings(root),
                ambient_raw_audio_retention_days=30,
                ambient_transcript_retention_days=7,
            )
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()
            now = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
            session_id = _ambient_session(db, now - timedelta(days=10))
            db.add_utterance(session_id=session_id, text="old transcript")
            artifact_dir = _artifact_dir(settings, session_id)
            (artifact_dir / "segment.wav").write_text("audio")

            result = apply_ambient_retention(settings, db, now=now, dry_run=False)

            self.assertEqual(result.session_count, 1)
            self.assertEqual(result.audio_artifact_count, 1)
            self.assertIsNone(db.get_ambient_session(session_id))
            self.assertFalse(artifact_dir.exists())

    def test_apply_retention_skips_active_sessions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = replace(
                _settings(root),
                ambient_raw_audio_retention_days=1,
                ambient_transcript_retention_days=1,
            )
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()
            now = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
            session_id = db.create_ambient_session(
                mode="ambient",
                source="mic",
                retention_policy="retain_audio",
                title="active",
            )
            _set_session_time(db, session_id, now - timedelta(days=10), status="active")
            artifact_dir = _artifact_dir(settings, session_id)
            (artifact_dir / "segment.wav").write_text("audio")

            result = apply_ambient_retention(settings, db, now=now, dry_run=False)

            self.assertEqual(result.session_count, 0)
            self.assertEqual(result.audio_artifact_count, 0)
            self.assertIsNotNone(db.get_ambient_session(session_id))
            self.assertTrue(artifact_dir.exists())


def _ambient_session(db: Database, started_at: datetime) -> str:
    session_id = db.create_ambient_session(
        mode="ambient",
        source="mic",
        retention_policy="retain_audio",
        title="retained",
    )
    db.end_ambient_session(session_id)
    _set_session_time(db, session_id, started_at, status="done")
    return session_id


def _set_session_time(
    db: Database,
    session_id: str,
    started_at: datetime,
    *,
    status: str,
) -> None:
    value = started_at.isoformat()
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE ambient_sessions
            SET started_at = ?, ended_at = ?, created_at = ?, updated_at = ?, status = ?
            WHERE id = ?
            """,
            (value, value, value, value, status, session_id),
        )


def _artifact_dir(settings: Settings, session_id: str) -> Path:
    artifact_dir = settings.artifacts_dir / "ambient" / session_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


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
