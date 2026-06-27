from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from atlas_voice.database import Database


class DatabaseTests(unittest.TestCase):
    def test_queue_retries_until_max_attempts(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            recording_id = db.create_recording(Path(tmp) / "a.wav")
            db.enqueue_job(recording_id, "summarize", max_attempts=2)

            first = db.claim_next_job()
            self.assertIsNotNone(first)
            failed = db.fail_job(first["id"], "llm unavailable")
            self.assertEqual(failed["status"], "queued")

            second = db.claim_next_job()
            self.assertEqual(second["attempts"], 2)
            failed = db.fail_job(second["id"], "llm unavailable")
            self.assertEqual(failed["status"], "failed")

    def test_search_finds_segments_and_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            recording_id = db.create_recording(Path(tmp) / "a.wav")
            db.replace_segments(
                recording_id,
                [
                    {
                        "start": 0,
                        "end": 1,
                        "speaker": "SPEAKER_00",
                        "text": "Quarterly planning discussion",
                    }
                ],
            )
            db.save_summary(
                recording_id,
                "The team discussed launch readiness.",
                model="test",
            )

            transcript_results = db.search("quarterly")
            summary_results = db.search("launch")

            self.assertEqual(transcript_results[0]["recording_id"], recording_id)
            self.assertEqual(transcript_results[0]["kind"], "transcript")
            self.assertEqual(summary_results[0]["kind"], "summary")


if __name__ == "__main__":
    unittest.main()
