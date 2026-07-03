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

    def test_reset_summary_job_preserves_cached_summary_and_search(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            recording_id = db.create_recording(Path(tmp) / "a.wav")
            db.enqueue_job(recording_id, "summarize")
            job = db.claim_next_job()
            self.assertIsNotNone(job)
            db.complete_job(job["id"])
            db.update_recording(recording_id, status="done")
            db.save_summary(
                recording_id,
                "Cached launch summary.",
                model="test",
                template_id="meeting",
            )

            self.assertTrue(db.reset_summary_job(recording_id))

            summary = db.get_summary(recording_id)
            self.assertIsNotNone(summary)
            self.assertEqual(summary["text"], "Cached launch summary.")
            self.assertEqual(db.search("launch")[0]["kind"], "summary")
            self.assertEqual(db.get_recording(recording_id)["status"], "queued")

            summarize_job = dict(db.jobs_for_recording(recording_id)[0])
            self.assertEqual(summarize_job["status"], "queued")
            self.assertEqual(summarize_job["attempts"], 0)
            self.assertIsNone(summarize_job["started_at"])
            self.assertIsNone(summarize_job["finished_at"])

    def test_realtime_session_persists_utterances_and_turns(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()

            session_id = db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Realtime",
            )
            utterance_id = db.add_utterance(
                session_id=session_id,
                text="Hello Atlas",
                source_provider="text",
            )
            turn_id = db.add_assistant_turn(
                session_id=session_id,
                user_utterance_id=utterance_id,
                text="Hello back",
                model="qwen-local",
                latency_ms=12,
            )
            db.end_ambient_session(session_id)

            session = db.get_ambient_session(session_id)
            utterances = db.list_utterances(session_id)
            turns = db.list_assistant_turns(session_id)

            self.assertEqual(session["status"], "ended")
            self.assertEqual(db.count_ambient_sessions(), 1)
            self.assertEqual(db.count_ambient_sessions(status="ended"), 1)
            self.assertEqual(db.list_ambient_sessions()[0]["utterance_count"], 1)
            self.assertEqual(utterances[0]["id"], utterance_id)
            self.assertEqual(utterances[0]["idx"], 0)
            self.assertEqual(utterances[0]["text"], "Hello Atlas")
            self.assertEqual(turns[0]["id"], turn_id)
            self.assertEqual(turns[0]["user_utterance_id"], utterance_id)
            self.assertEqual(turns[0]["tool_calls"], [])

    def test_audit_tables_store_model_runs_and_privacy_events(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()

            run_id = db.log_model_run(
                provider="openai-compatible",
                model="qwen-local",
                task="summarize",
                input_ref="recording:abc",
                output_ref="summary:abc",
                latency_ms=42,
            )
            event_id = db.log_privacy_event(
                "privacy.audit_egress",
                "Local-only status: ok",
                metadata={"issue_count": 0},
            )

            self.assertGreater(run_id, 0)
            self.assertGreater(event_id, 0)
            model_run = db.list_model_runs()[0]
            privacy_event = db.list_privacy_events()[0]
            self.assertEqual(model_run["task"], "summarize")
            self.assertEqual(model_run["latency_ms"], 42)
            self.assertEqual(privacy_event["event_type"], "privacy.audit_egress")
            self.assertEqual(privacy_event["metadata"], {"issue_count": 0})


if __name__ == "__main__":
    unittest.main()
