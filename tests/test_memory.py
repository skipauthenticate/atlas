from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from atlas_voice.database import Database
from atlas_voice.memory import extract_memories_from_ambient_session


class MemoryExtractionTests(unittest.TestCase):
    def test_extracts_explicit_memories_from_ambient_session_with_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            session_id = db.create_ambient_session(
                mode="ambient",
                source="mic",
                title="Ambient",
            )
            db.add_utterance(
                session_id=session_id,
                speaker="Alice",
                text="Remember that the Logitech BRIO uses plughw:2,0.",
                source_provider="text",
            )
            db.add_utterance(
                session_id=session_id,
                speaker="Alice",
                text="I prefer concise meeting summaries with action items.",
                source_provider="text",
            )
            db.end_ambient_session(session_id)

            dry_run = extract_memories_from_ambient_session(db, session_id, dry_run=True)

            self.assertEqual(dry_run.status, "ok")
            self.assertEqual(dry_run.created_ids, ())
            self.assertEqual(len(dry_run.candidates), 2)
            self.assertEqual([candidate.kind for candidate in dry_run.candidates], ["fact", "preference"])
            self.assertEqual(db.list_memory_items(), [])

            applied = extract_memories_from_ambient_session(db, session_id, dry_run=False)

            memories = db.list_memory_items(source_type="ambient_session")
            self.assertEqual(set(applied.created_ids), {item["id"] for item in memories})
            self.assertEqual([item["source_id"] for item in memories], [session_id, session_id])
            self.assertEqual([item["kind"] for item in memories], ["preference", "fact"])
            self.assertIn("BRIO uses plughw:2,0", memories[1]["text"])
            self.assertIn("concise meeting summaries", memories[0]["text"])

    def test_ambient_memory_extraction_is_idempotent_and_skips_direct_voice(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            session_id = db.create_ambient_session(
                mode="meeting",
                source="file",
                title="Meeting",
            )
            db.add_utterance(
                session_id=session_id,
                text="Remember: the project codename is Atlas.",
                source_provider="text",
            )
            db.end_ambient_session(session_id)
            direct_id = db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Direct",
            )
            db.add_utterance(
                session_id=direct_id,
                text="Remember that direct voice extraction is a separate item.",
                source_provider="text",
            )
            db.end_ambient_session(direct_id)

            first = extract_memories_from_ambient_session(db, session_id, dry_run=False)
            second = extract_memories_from_ambient_session(db, session_id, dry_run=False)
            skipped = extract_memories_from_ambient_session(db, direct_id, dry_run=False)

            self.assertEqual(len(first.created_ids), 1)
            self.assertEqual(second.created_ids, ())
            self.assertEqual(second.skipped_duplicates, 1)
            self.assertEqual(skipped.status, "skipped")
            self.assertEqual(len(db.list_memory_items()), 1)


if __name__ == "__main__":
    unittest.main()
