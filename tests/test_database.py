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

    def test_list_ambient_sessions_can_filter_by_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            ambient_id = db.create_ambient_session(
                mode="ambient",
                source="file",
                title="Ambient",
            )
            direct_id = db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Realtime",
            )
            db.add_utterance(
                session_id=direct_id,
                text="Hello Atlas",
                source_provider="text",
            )
            db.end_ambient_session(ambient_id)
            db.end_ambient_session(direct_id)

            direct_sessions = db.list_ambient_sessions(mode="direct_voice")
            all_sessions = db.list_ambient_sessions(mode=None)

        self.assertEqual([session["id"] for session in direct_sessions], [direct_id])
        self.assertEqual(direct_sessions[0]["utterance_count"], 1)
        self.assertEqual({session["id"] for session in all_sessions}, {ambient_id, direct_id})

    def test_privacy_purge_finds_and_deletes_matching_sessions(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            alice_id = db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Call with Alice",
            )
            bob_id = db.create_ambient_session(
                mode="ambient",
                source="file",
                title="Project standup",
            )
            alice_utterance_id = db.add_utterance(
                session_id=alice_id,
                speaker="Alice",
                text="We discussed the secret launch date.",
                source_provider="text",
            )
            db.add_assistant_turn(
                session_id=alice_id,
                user_utterance_id=alice_utterance_id,
                text="Noted privately.",
                model="qwen-local",
            )
            db.add_utterance(
                session_id=bob_id,
                speaker="Bob",
                text="General planning update.",
                source_provider="text",
            )
            db.end_ambient_session(alice_id)
            db.end_ambient_session(bob_id)

            keyword_matches = db.find_privacy_purge_sessions(keyword="secret")
            person_matches = db.find_privacy_purge_sessions(person="Alice")
            date_matches = db.find_privacy_purge_sessions(started_on=keyword_matches[0]["started_at"][:10])
            result = db.purge_privacy_sessions([alice_id])

            self.assertEqual([session["id"] for session in keyword_matches], [alice_id])
            self.assertEqual([session["id"] for session in person_matches], [alice_id])
            self.assertIn(alice_id, {session["id"] for session in date_matches})
            self.assertEqual(result["session_count"], 1)
            self.assertEqual(result["utterance_count"], 1)
            self.assertEqual(result["assistant_turn_count"], 1)
            self.assertIsNone(db.get_ambient_session(alice_id))
            self.assertIsNotNone(db.get_ambient_session(bob_id))

    def test_memory_items_can_be_created_retrieved_and_listed(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()

            memory_id = db.create_memory_item(
                kind="preference",
                title="Meeting preference",
                text="The user prefers concise meeting summaries with action items.",
                source_type="manual",
                source_id="seed",
                importance=0.8,
                confidence=0.9,
            )

            memory = db.get_memory_item(memory_id)
            memories = db.list_memory_items()

            self.assertEqual(memory["id"], memory_id)
            self.assertEqual(memory["kind"], "preference")
            self.assertEqual(memory["title"], "Meeting preference")
            self.assertEqual(memory["source_type"], "manual")
            self.assertEqual(memory["source_id"], "seed")
            self.assertEqual(memory["importance"], 0.8)
            self.assertEqual(memory["confidence"], 0.9)
            self.assertIsNone(memory["valid_until"])
            self.assertEqual([item["id"] for item in memories], [memory_id])

    def test_coaching_goals_can_be_created_listed_and_completed(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()

            goal_id = db.create_coaching_goal(
                title="Ask better follow-up questions",
                description="Practice one reflective question per conversation.",
                target_date="2026-08-01",
                metric="reflection_ratio",
                metadata={"value": "curiosity", "next_action": "review daily summary"},
            )
            archived_id = db.create_coaching_goal(
                title="Archived goal",
                status="archived",
            )

            active_goals = db.list_coaching_goals(status="active")
            all_goals = db.list_coaching_goals(status=None)
            goal = db.get_coaching_goal(goal_id)

            self.assertEqual([item["id"] for item in active_goals], [goal_id])
            self.assertEqual({item["id"] for item in all_goals}, {goal_id, archived_id})
            self.assertEqual(goal["title"], "Ask better follow-up questions")
            self.assertEqual(goal["metric"], "reflection_ratio")
            self.assertEqual(goal["metadata"], {"value": "curiosity", "next_action": "review daily summary"})
            self.assertIsNone(goal["completed_at"])

            db.update_coaching_goal(goal_id, status="completed")
            completed = db.get_coaching_goal(goal_id)

            self.assertEqual(completed["status"], "completed")
            self.assertIsNotNone(completed["completed_at"])

    def test_feedback_events_can_be_logged_and_filtered(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            goal_id = db.create_coaching_goal(title="Improve clarity")
            session_id = db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Coaching conversation",
            )

            event_id = db.log_feedback_event(
                event_type="coaching.signal",
                category="conversation",
                message="Asked a concrete follow-up question.",
                goal_id=goal_id,
                session_id=session_id,
                score=0.75,
                evidence_ref="utterance:1",
                metadata={"signal": "question_ratio", "direction": "positive"},
            )
            db.log_feedback_event(
                event_type="writing.signal",
                category="writing",
                message="Draft has a clear ask.",
            )

            event = db.get_feedback_event(event_id)
            goal_events = db.list_feedback_events(goal_id=goal_id)
            session_events = db.list_feedback_events(session_id=session_id)
            conversation_events = db.list_feedback_events(category="conversation")

            self.assertEqual(event["message"], "Asked a concrete follow-up question.")
            self.assertEqual(event["goal_id"], goal_id)
            self.assertEqual(event["session_id"], session_id)
            self.assertEqual(event["score"], 0.75)
            self.assertEqual(event["metadata"], {"signal": "question_ratio", "direction": "positive"})
            self.assertEqual([item["id"] for item in goal_events], [event_id])
            self.assertEqual([item["id"] for item in session_events], [event_id])
            self.assertEqual([item["id"] for item in conversation_events], [event_id])

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
