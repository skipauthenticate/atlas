from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from atlas_voice.coaching import (
    generate_daily_coaching_summary,
    generate_weekly_coaching_summary,
    track_conversation_signals,
)
from atlas_voice.database import Database


class CoachingSummaryTests(unittest.TestCase):
    def test_daily_coaching_summary_dry_run_then_store(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            direct_id = _session(db, "direct_voice", "Morning check-in", "2026-07-03T09:00:00+00:00")
            ambient_id = _session(db, "meeting", "Planning meeting", "2026-07-03T14:00:00+00:00")
            db.add_utterance(
                session_id=direct_id,
                text="What should I clarify before the launch review?",
                source_provider="text",
            )
            db.add_utterance(
                session_id=direct_id,
                text="I will send the follow-up note today.",
                source_provider="text",
            )
            db.add_utterance(
                session_id=ambient_id,
                text="We need a clearer owner for the demo.",
                source_provider="text",
            )

            dry_run = generate_daily_coaching_summary(db, "2026-07-03", dry_run=True)

            self.assertEqual(dry_run.status, "ok")
            self.assertIsNone(dry_run.event_id)
            self.assertEqual(dry_run.session_count, 2)
            self.assertEqual(dry_run.utterance_count, 3)
            self.assertIn("Daily Coaching Summary - 2026-07-03", dry_run.message)
            self.assertIn("Question ratio", dry_run.message)
            self.assertIn("Morning check-in", dry_run.message)
            self.assertEqual(db.list_feedback_events(category="daily_summary"), [])

            stored = generate_daily_coaching_summary(db, "2026-07-03", dry_run=False)

            events = db.list_feedback_events(category="daily_summary")
            self.assertEqual(stored.event_id, events[0]["id"])
            self.assertEqual(events[0]["event_type"], "coaching.daily_summary")
            self.assertEqual(events[0]["metadata"]["day"], "2026-07-03")
            self.assertEqual(events[0]["metadata"]["session_count"], 2)
            self.assertIn("follow-up note", events[0]["message"])

    def test_daily_coaching_summary_is_idempotent_for_day(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            session_id = _session(db, "direct_voice", "Daily check-in", "2026-07-03T10:00:00+00:00")
            db.add_utterance(
                session_id=session_id,
                text="I prefer asking one question before advice.",
                source_provider="text",
            )

            first = generate_daily_coaching_summary(db, "2026-07-03", dry_run=False)
            second = generate_daily_coaching_summary(db, "2026-07-03", dry_run=False)

            events = db.list_feedback_events(category="daily_summary")
            self.assertEqual(first.event_id, second.event_id)
            self.assertEqual(second.status, "existing")
            self.assertEqual(len(events), 1)

    def test_weekly_coaching_summary_dry_run_then_store(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            monday = _session(db, "direct_voice", "Monday check-in", "2026-07-06T09:00:00+00:00")
            friday = _session(db, "meeting", "Friday retro", "2026-07-10T16:00:00+00:00")
            outside = _session(db, "direct_voice", "Outside week", "2026-07-13T09:00:00+00:00")
            db.add_utterance(
                session_id=monday,
                text="What should I clarify before the roadmap review?",
                source_provider="text",
            )
            db.add_utterance(
                session_id=friday,
                text="We should document the launch owner and follow through next week.",
                source_provider="text",
            )
            db.add_utterance(
                session_id=outside,
                text="This should not be included in this weekly summary.",
                source_provider="text",
            )

            dry_run = generate_weekly_coaching_summary(db, "2026-07-06", dry_run=True)

            self.assertEqual(dry_run.status, "ok")
            self.assertIsNone(dry_run.event_id)
            self.assertEqual(dry_run.week_start, "2026-07-06")
            self.assertEqual(dry_run.week_end, "2026-07-12")
            self.assertEqual(dry_run.session_count, 2)
            self.assertEqual(dry_run.utterance_count, 2)
            self.assertIn("Weekly Coaching Summary - 2026-07-06 to 2026-07-12", dry_run.message)
            self.assertIn("Monday check-in", dry_run.message)
            self.assertNotIn("Outside week", dry_run.message)
            self.assertEqual(db.list_feedback_events(category="weekly_summary"), [])

            stored = generate_weekly_coaching_summary(db, "2026-07-06", dry_run=False)

            events = db.list_feedback_events(category="weekly_summary")
            self.assertEqual(stored.event_id, events[0]["id"])
            self.assertEqual(events[0]["event_type"], "coaching.weekly_summary")
            self.assertEqual(events[0]["metadata"]["week_start"], "2026-07-06")
            self.assertEqual(events[0]["metadata"]["week_end"], "2026-07-12")
            self.assertEqual(events[0]["metadata"]["session_count"], 2)

    def test_weekly_coaching_summary_is_idempotent_for_week(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            session_id = _session(db, "direct_voice", "Weekly check-in", "2026-07-08T10:00:00+00:00")
            db.add_utterance(
                session_id=session_id,
                text="I will ask clearer questions this week.",
                source_provider="text",
            )

            first = generate_weekly_coaching_summary(db, "2026-07-06", dry_run=False)
            second = generate_weekly_coaching_summary(db, "2026-07-06", dry_run=False)

            events = db.list_feedback_events(category="weekly_summary")
            self.assertEqual(first.event_id, second.event_id)
            self.assertEqual(second.status, "existing")
            self.assertEqual(len(events), 1)

    def test_tracks_conversation_signals_dry_run_then_store(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            session_id = _session(db, "direct_voice", "Coaching signal session", "2026-07-08T10:00:00+00:00")
            db.add_utterance(
                session_id=session_id,
                text="What is the clearest next step for launch?",
                source_provider="text",
            )
            db.add_utterance(
                session_id=session_id,
                text="I will send Alice the launch checklist tomorrow.",
                source_provider="text",
            )
            db.add_utterance(
                session_id=session_id,
                text="Need owner for demo.",
                source_provider="text",
            )

            dry_run = track_conversation_signals(db, session_id, dry_run=True)

            self.assertEqual(dry_run.status, "ok")
            self.assertIsNone(dry_run.event_id)
            self.assertEqual(dry_run.metrics["utterance_count"], 3)
            self.assertEqual(dry_run.metrics["question_count"], 1)
            self.assertEqual(dry_run.metrics["commitment_count"], 1)
            self.assertGreater(dry_run.metrics["clarity"], 0)
            self.assertGreater(dry_run.metrics["concision"], 0)
            self.assertGreater(dry_run.metrics["actionable_next_steps"], 0)
            self.assertEqual(db.list_feedback_events(category="conversation_signals"), [])

            stored = track_conversation_signals(db, session_id, dry_run=False)

            events = db.list_feedback_events(category="conversation_signals")
            self.assertEqual(stored.event_id, events[0]["id"])
            self.assertEqual(events[0]["event_type"], "coaching.conversation_signals")
            self.assertEqual(events[0]["session_id"], session_id)
            self.assertEqual(events[0]["metadata"]["signals"]["question_ratio"], stored.metrics["question_ratio"])
            self.assertIn("Conversation Signals", events[0]["message"])

    def test_conversation_signals_are_idempotent_per_session(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            session_id = _session(db, "meeting", "Planning", "2026-07-08T10:00:00+00:00")
            db.add_utterance(
                session_id=session_id,
                text="We should clarify the launch owner.",
                source_provider="text",
            )

            first = track_conversation_signals(db, session_id, dry_run=False)
            second = track_conversation_signals(db, session_id, dry_run=False)

            events = db.list_feedback_events(category="conversation_signals")
            self.assertEqual(first.event_id, second.event_id)
            self.assertEqual(second.status, "existing")
            self.assertEqual(len(events), 1)


def _session(db: Database, mode: str, title: str, started_at: str) -> str:
    session_id = db.create_ambient_session(mode=mode, source="test", title=title)
    db.end_ambient_session(session_id)
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE ambient_sessions
            SET started_at = ?, ended_at = ?, created_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (started_at, started_at, started_at, started_at, session_id),
        )
    return session_id


if __name__ == "__main__":
    unittest.main()
