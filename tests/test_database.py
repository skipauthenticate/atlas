from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from atlas_voice.database import Database, RECORDING_LIBRARY_SUMMARY_EXCERPT_CHARS


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

    def test_recording_segment_stats_avoid_loading_full_transcript(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "db.sqlite")
            db.initialize()
            recording_id = db.create_recording(root / "a.wav")

            self.assertEqual(
                db.get_recording_segment_stats(recording_id),
                {"segment_count": 0, "duration_seconds": None},
            )
            db.replace_segments(
                recording_id,
                [
                    {"start": 0, "end": 2.5, "speaker": "A", "text": "First"},
                    {"start": 2.5, "end": 7, "speaker": "B", "text": "Second"},
                ],
            )

            self.assertEqual(
                db.get_recording_segment_stats(recording_id),
                {"segment_count": 2, "duration_seconds": 7.0},
            )

    def test_search_recording_never_returns_another_recording(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "db.sqlite")
            db.initialize()
            selected_id = db.create_recording(root / "selected.wav", title="Selected")
            other_id = db.create_recording(root / "other.wav", title="Other")
            db.replace_segments(
                selected_id,
                [
                    {
                        "start": 0,
                        "end": 1,
                        "speaker": "SPEAKER_00",
                        "text": "Roadmap details for project Atlas",
                    }
                ],
            )
            db.replace_segments(
                other_id,
                [
                    {
                        "start": 0,
                        "end": 1,
                        "speaker": "SPEAKER_01",
                        "text": "Roadmap secret for project Zephyr",
                    }
                ],
            )
            db.save_summary(selected_id, "Atlas launch roadmap", model="test")
            db.save_summary(other_id, "Zephyr launch roadmap", model="test")

            selected_results = db.search_recording(selected_id, "roadmap")

            self.assertGreaterEqual(len(selected_results), 2)
            self.assertEqual(
                {result["recording_id"] for result in selected_results},
                {selected_id},
            )
            self.assertEqual(db.search_recording(selected_id, "Zephyr"), [])
            self.assertEqual(db.search_recording(selected_id, "roadmap", limit=0), [])

    def test_recording_folders_and_joined_library_filters(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "db.sqlite")
            db.initialize()
            projects_id = db.create_recording_folder("  Projects  ")
            archive_id = db.create_recording_folder("Archive")
            filed_id = db.create_recording(root / "filed.wav")
            unfiled_id = db.create_recording(root / "unfiled.wav")
            db.update_recording(filed_id, status="done")
            db.replace_segments(
                filed_id,
                [
                    {
                        "start": 0,
                        "end": 65.5,
                        "speaker": "SPEAKER_00",
                        "text": "Project planning notes",
                    }
                ],
            )
            db.save_summary(filed_id, "A concise project summary.", model="test")

            self.assertTrue(db.set_recording_folder(filed_id, projects_id))
            with self.assertRaisesRegex(ValueError, "Unknown recording folder"):
                db.set_recording_folder(unfiled_id, "missing-folder")
            with self.assertRaisesRegex(ValueError, "already exists"):
                db.create_recording_folder("projects")

            folders = db.list_recording_folders()
            counts = {folder["id"]: folder["recording_count"] for folder in folders}
            self.assertEqual(counts, {archive_id: 0, projects_id: 1})
            self.assertEqual(db.get_recording_folder(projects_id)["name"], "Projects")

            library = db.list_recording_library()
            by_id = {recording["id"]: recording for recording in library}
            self.assertEqual(by_id[filed_id]["folder_name"], "Projects")
            self.assertEqual(by_id[filed_id]["summary"], "A concise project summary.")
            self.assertEqual(by_id[filed_id]["duration_seconds"], 65.5)
            self.assertIsNone(by_id[unfiled_id]["folder_name"])
            self.assertIsNone(by_id[unfiled_id]["summary"])
            self.assertIsNone(by_id[unfiled_id]["duration_seconds"])
            self.assertEqual(
                [recording["id"] for recording in db.list_recording_library(projects_id)],
                [filed_id],
            )
            self.assertEqual(
                [recording["id"] for recording in db.list_recording_library("")],
                [unfiled_id],
            )
            self.assertEqual(
                [recording["id"] for recording in db.list_recording_library(status="done")],
                [filed_id],
            )
            self.assertEqual(len(db.list_recordings(1)), 1)

            self.assertTrue(db.rename_recording_folder(projects_id, "Active Projects"))
            self.assertEqual(
                db.list_recording_library(projects_id)[0]["folder_name"],
                "Active Projects",
            )
            self.assertTrue(db.delete_recording_folder(projects_id))
            self.assertIsNone(db.get_recording(filed_id)["folder_id"])
            self.assertFalse(db.delete_recording_folder(projects_id))

    def test_recording_library_pagination_projection_processing_and_counts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "db.sqlite")
            db.initialize()
            active_folder = db.create_recording_folder("Active")
            archive_folder = db.create_recording_folder("Archive")
            empty_folder = db.create_recording_folder("Empty")
            done_id = db.create_recording(root / "done.wav")
            queued_id = db.create_recording(root / "queued.wav")
            running_id = db.create_recording(root / "running.wav")
            failed_id = db.create_recording(root / "failed.wav")
            duplicate_id = db.create_recording(root / "duplicate.wav")
            ordered = [done_id, queued_id, running_id, failed_id, duplicate_id]
            statuses = ["done", "queued", "transcribe_running", "failed", "duplicate"]
            for day, (recording_id, status) in enumerate(
                zip(reversed(ordered), reversed(statuses)),
                start=1,
            ):
                db.update_recording(
                    recording_id,
                    status=status,
                    created_at=f"2026-07-{day:02d}T00:00:00+00:00",
                )
            db.set_recording_folder(done_id, active_folder)
            db.set_recording_folder(running_id, active_folder)
            db.set_recording_folder(failed_id, archive_folder)
            long_summary = "S" * (RECORDING_LIBRARY_SUMMARY_EXCERPT_CHARS + 100)
            db.save_summary(queued_id, long_summary, model="test")
            db.replace_segments(
                queued_id,
                [
                    {
                        "start": 0,
                        "end": 12,
                        "speaker": "SPEAKER_00",
                        "text": "Opening",
                    },
                    {
                        "start": 12,
                        "end": 90.25,
                        "speaker": "SPEAKER_01",
                        "text": "Closing",
                    },
                ],
            )

            page = db.list_recording_library(limit=2, offset=1)
            processing = db.list_recording_library(status="processing")
            counts = db.recording_library_folder_counts()

            self.assertEqual([item["id"] for item in page], [queued_id, running_id])
            self.assertEqual(
                set(page[0]),
                {
                    "id",
                    "title",
                    "title_origin",
                    "status",
                    "folder_id",
                    "created_at",
                    "updated_at",
                    "folder_name",
                    "summary",
                    "duration_seconds",
                },
            )
            self.assertNotIn("source_path", page[0])
            self.assertNotIn("original_path", page[0])
            self.assertNotIn("normalized_path", page[0])
            self.assertEqual(
                page[0]["summary"],
                long_summary[:RECORDING_LIBRARY_SUMMARY_EXCERPT_CHARS],
            )
            self.assertEqual(page[0]["duration_seconds"], 90.25)
            self.assertEqual(
                [item["id"] for item in processing],
                [queued_id, running_id],
            )
            self.assertEqual(db.count_recording_library(), 5)
            self.assertEqual(db.count_recording_library(""), 2)
            self.assertEqual(db.count_recording_library(active_folder), 2)
            self.assertEqual(db.count_recording_library(status="processing"), 2)
            self.assertEqual(
                db.count_recording_library(active_folder, "processing"),
                1,
            )
            self.assertEqual(
                counts,
                {
                    "all": 5,
                    "unfiled": 2,
                    "folders": {
                        active_folder: 2,
                        archive_folder: 1,
                        empty_folder: 0,
                    },
                },
            )
            self.assertEqual(db.list_recording_library(limit=0, offset=10), [])

    def test_recording_title_origins_and_guarded_updates(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "db.sqlite")
            db.initialize()
            filename_id = db.create_recording(root / "raw-audio.wav")
            manual_id = db.create_recording(root / "meeting.wav", title="  Team   Weekly  ")

            self.assertEqual(db.get_recording(filename_id)["title_origin"], "filename")
            self.assertEqual(db.get_recording(manual_id)["title"], "Team Weekly")
            self.assertEqual(db.get_recording(manual_id)["title_origin"], "manual")
            self.assertTrue(
                db.update_recording_title(
                    filename_id,
                    "Atlas Launch Planning",
                    origin="generated",
                    expected_origin="filename",
                )
            )
            self.assertFalse(
                db.update_recording_title(
                    manual_id,
                    "Generated Replacement",
                    origin="generated",
                    expected_origin="filename",
                )
            )
            self.assertEqual(db.get_recording(manual_id)["title"], "Team Weekly")
            self.assertEqual(db.get_recording(filename_id)["title_origin"], "generated")
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                db.update_recording_title(filename_id, "  ")
            with self.assertRaisesRegex(ValueError, "160 characters"):
                db.update_recording_title(filename_id, "x" * 161)

    def test_recording_library_migration_is_idempotent_and_preserves_legacy_title(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.sqlite"
            with sqlite3.connect(path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE recordings (
                        id TEXT PRIMARY KEY,
                        source_path TEXT NOT NULL,
                        title TEXT NOT NULL,
                        status TEXT NOT NULL,
                        sha256 TEXT,
                        duplicate_of TEXT,
                        original_path TEXT,
                        normalized_path TEXT,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    INSERT INTO recordings (
                        id, source_path, title, status, created_at, updated_at
                    ) VALUES (
                        'legacy-recording', '/private/legacy.wav',
                        'Do Not Replace This Title', 'done',
                        '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
                    );
                    """
                )

            db = Database(path)
            db.initialize()
            db.initialize()

            recording = db.get_recording("legacy-recording")
            self.assertEqual(recording["title"], "Do Not Replace This Title")
            self.assertEqual(recording["title_origin"], "legacy")
            self.assertIsNone(recording["folder_id"])
            with sqlite3.connect(path) as conn:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(recordings)").fetchall()
                }
                folder_foreign_keys = [
                    row
                    for row in conn.execute("PRAGMA foreign_key_list(recordings)").fetchall()
                    if row[3] == "folder_id"
                ]
                folder_index_count = conn.execute(
                    """
                    SELECT COUNT(*) FROM sqlite_master
                    WHERE type = 'index' AND name = 'idx_recordings_folder'
                    """
                ).fetchone()[0]

            self.assertIn("folder_id", columns)
            self.assertIn("title_origin", columns)
            self.assertEqual(folder_foreign_keys[0][6].upper(), "SET NULL")
            self.assertEqual(folder_index_count, 1)

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

    def test_recent_history_helpers_return_titles_summaries_and_distinct_sessions(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = Database(root / "db.sqlite")
            db.initialize()
            recording_id = db.create_recording(
                root / "Project-Akshaya.wav",
                title="Project Akshaya planning",
            )
            db.update_recording(recording_id, status="done")
            db.save_summary(
                recording_id,
                "Akshaya pricing and launch strategy.",
                model="test",
            )
            past_id = db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Travel planning",
            )
            db.add_utterance(
                session_id=past_id,
                text="First older detail",
                source_provider="faster-whisper",
            )
            second_id = db.add_utterance(
                session_id=past_id,
                text="Vietnam and Bali travel plans",
                source_provider="faster-whisper",
            )
            third_id = db.add_utterance(
                session_id=past_id,
                text="A wedding trip in September",
                source_provider="faster-whisper",
            )
            excluded_id = db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Current session",
            )
            db.add_utterance(
                session_id=excluded_id,
                text="Current private detail",
                source_provider="text",
            )

            summaries = db.list_recent_recording_summaries()
            title_matches = db.search_recording_titles("Project Akshaya")
            excerpts = db.list_recent_conversation_excerpts(
                session_limit=4,
                utterances_per_session=2,
                mode="direct_voice",
                exclude_session_id=excluded_id,
            )

        self.assertEqual(summaries[0]["recording_id"], recording_id)
        self.assertEqual(title_matches[0]["summary"], "Akshaya pricing and launch strategy.")
        self.assertEqual([item["session_id"] for item in excerpts], [past_id])
        self.assertEqual(
            [item["id"] for item in excerpts[0]["utterances"]],
            [second_id, third_id],
        )
        self.assertTrue(
            all(
                item["source_provider"] == "faster-whisper"
                for item in excerpts[0]["utterances"]
            )
        )

    def test_conversation_search_indexes_new_utterances_and_turns(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            excluded_id = db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Current voice chat",
            )
            db.add_utterance(
                session_id=excluded_id,
                text="Retrieval from the current session",
                source_provider="text",
            )
            matching_id = db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Jetson architecture",
            )
            utterance_id = db.add_utterance(
                session_id=matching_id,
                speaker="Alice",
                text="TensorRT retrieval pipeline",
                source_provider="text",
            )
            turn_id = db.add_assistant_turn(
                session_id=matching_id,
                user_utterance_id=utterance_id,
                text="The retrieval cache is warmed",
                model="qwen-local",
            )
            other_mode_id = db.create_ambient_session(
                mode="meeting",
                source="file",
                title="Meeting",
            )
            db.add_utterance(
                session_id=other_mode_id,
                text="Meeting retrieval notes",
                source_provider="text",
            )

            results = db.search_conversations(
                "retriev",
                mode="direct_voice",
                exclude_session_id=excluded_id,
            )

            self.assertEqual(
                {(item["kind"], item["item_id"]) for item in results},
                {("utterance", utterance_id), ("assistant", turn_id)},
            )
            self.assertTrue(all(item["session_id"] == matching_id for item in results))
            self.assertTrue(all(item["mode"] == "direct_voice" for item in results))
            self.assertTrue(all("[retrieval]" in item["snippet"].lower() for item in results))
            self.assertEqual(
                set(results[0]),
                {
                    "session_id",
                    "item_id",
                    "kind",
                    "speaker",
                    "snippet",
                    "title",
                    "mode",
                    "started_at",
                    "rank",
                },
            )
            self.assertEqual(db.search_conversations("retrieval", limit=0), [])

            unicode_id = db.create_ambient_session(
                mode="direct_voice",
                source="websocket",
                title="Unicode voice chat",
            )
            db.add_utterance(
                session_id=unicode_id,
                text="火星计划将在明天继续讨论",
                source_provider="text",
            )
            self.assertEqual(
                db.search_conversations("火星计划")[0]["session_id"],
                unicode_id,
            )

    def test_conversation_search_backfills_existing_database_once(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "db.sqlite"
            with sqlite3.connect(path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE ambient_sessions (
                        id TEXT PRIMARY KEY,
                        mode TEXT NOT NULL,
                        source TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        ended_at TEXT,
                        status TEXT NOT NULL,
                        retention_policy TEXT NOT NULL DEFAULT 'ephemeral',
                        title TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE utterances (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        start REAL,
                        end REAL,
                        speaker TEXT NOT NULL,
                        text TEXT NOT NULL,
                        confidence REAL,
                        source_provider TEXT NOT NULL,
                        is_directed_to_assistant INTEGER NOT NULL DEFAULT 1,
                        sensitivity TEXT,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE assistant_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        user_utterance_id INTEGER,
                        text TEXT NOT NULL,
                        audio_path TEXT,
                        model TEXT NOT NULL,
                        latency_ms INTEGER,
                        tool_calls_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL
                    );
                    INSERT INTO ambient_sessions (
                        id, mode, source, started_at, status, title, created_at, updated_at
                    ) VALUES (
                        'legacy-session', 'direct_voice', 'websocket',
                        '2026-07-01T00:00:00+00:00', 'ended', 'Legacy chat',
                        '2026-07-01T00:00:00+00:00', '2026-07-01T00:00:00+00:00'
                    );
                    INSERT INTO utterances (
                        session_id, idx, speaker, text, source_provider, created_at
                    ) VALUES (
                        'legacy-session', 0, 'user', 'Legacy thermal discussion',
                        'text', '2026-07-01T00:00:00+00:00'
                    );
                    INSERT INTO assistant_turns (
                        session_id, user_utterance_id, text, model, created_at
                    ) VALUES (
                        'legacy-session', 1, 'Legacy fan recommendation', 'qwen-local',
                        '2026-07-01T00:00:01+00:00'
                    );
                    """
                )

            db = Database(path)
            db.initialize()
            db.initialize()

            results = db.search_conversations("legacy")
            with db.connect() as conn:
                indexed_rows = conn.execute(
                    "SELECT COUNT(*) AS count FROM conversation_fts"
                ).fetchone()["count"]
                migrations = conn.execute(
                    """
                    SELECT COUNT(*) AS count FROM schema_migrations
                    WHERE name = 'conversation_fts_v1'
                    """
                ).fetchone()["count"]

            self.assertEqual({item["kind"] for item in results}, {"utterance", "assistant"})
            self.assertEqual(indexed_rows, 2)
            self.assertEqual(migrations, 1)

    def test_ambient_sessions_can_be_searched_and_deleted(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            keep_id = db.create_ambient_session(
                mode="ambient",
                source="mic",
                title="Kitchen planning",
            )
            delete_id = db.create_ambient_session(
                mode="meeting",
                source="file",
                title="Launch review",
            )
            keep_utterance = db.add_utterance(
                session_id=keep_id,
                text="Discuss groceries and dinner",
                source_provider="text",
            )
            db.add_utterance(
                session_id=delete_id,
                text="The launch blocker is fixed",
                source_provider="text",
            )
            db.add_assistant_turn(
                session_id=delete_id,
                user_utterance_id=keep_utterance,
                text="Review launch checklist",
                model="qwen-local",
            )
            db.end_ambient_session(keep_id)
            db.end_ambient_session(delete_id)

            title_matches = db.list_ambient_sessions(query="kitchen")
            text_matches = db.list_ambient_sessions(query="blocker")
            result = db.delete_ambient_session(delete_id)

            self.assertEqual([session["id"] for session in title_matches], [keep_id])
            self.assertEqual([session["id"] for session in text_matches], [delete_id])
            self.assertEqual(result["session_count"], 1)
            self.assertEqual(result["utterance_count"], 1)
            self.assertEqual(result["assistant_turn_count"], 1)
            self.assertIsNone(db.get_ambient_session(delete_id))
            self.assertIsNotNone(db.get_ambient_session(keep_id))
            self.assertEqual(db.search_conversations("launch"), [])
            self.assertEqual(
                db.search_conversations("groceries")[0]["session_id"],
                keep_id,
            )
            with db.connect() as conn:
                deleted_index_rows = conn.execute(
                    "SELECT COUNT(*) AS count FROM conversation_fts WHERE session_id = ?",
                    (delete_id,),
                ).fetchone()["count"]
            self.assertEqual(deleted_index_rows, 0)

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
            alice_turn_id = db.add_assistant_turn(
                session_id=alice_id,
                user_utterance_id=alice_utterance_id,
                text="Noted privately.",
                model="qwen-local",
            )
            db.log_model_run(
                provider="web:searxng",
                model="search",
                task="web_search",
                input_ref=f"utterance:{alice_utterance_id}",
                error="redacted provider error",
            )
            db.log_model_run(
                provider="openai-compatible",
                model="qwen-local",
                task="realtime_chat",
                input_ref=f"utterance:{alice_utterance_id}",
                output_ref=f"assistant_turn:{alice_turn_id}",
            )
            db.log_model_run(
                provider="local",
                model="keep",
                task="unrelated",
                input_ref="recording:keep",
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
            self.assertEqual(result["model_run_count"], 2)
            self.assertIsNone(db.get_ambient_session(alice_id))
            self.assertIsNotNone(db.get_ambient_session(bob_id))
            self.assertEqual(
                [run["task"] for run in db.list_model_runs()],
                ["unrelated"],
            )

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

    def test_memory_items_can_be_updated_deleted_and_reindexed(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            memory_id = db.create_memory_item(
                kind="preference",
                title="Original title",
                text="The user prefers long answers.",
                source_type="manual",
            )

            self.assertTrue(
                db.update_memory_item(
                    memory_id,
                    title="Reply preference",
                    text="The user prefers concise answers first.",
                    kind="preference",
                )
            )

            updated = db.get_memory_item(memory_id)
            self.assertEqual(updated["title"], "Reply preference")
            self.assertEqual(updated["text"], "The user prefers concise answers first.")
            self.assertEqual(db.search_memory_items("concise")[0]["id"], memory_id)
            self.assertEqual(db.search_memory_items("long"), [])

            self.assertTrue(db.delete_memory_item(memory_id))
            self.assertIsNone(db.get_memory_item(memory_id))
            self.assertEqual(db.search_memory_items("concise"), [])

    def test_search_memory_items_uses_fts_and_ignores_expired_items(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            matching_id = db.create_memory_item(
                kind="preference",
                title="Summary preference",
                text="The user prefers concise launch summaries with clear action items.",
                source_type="manual",
                source_id="seed",
                importance=0.8,
                confidence=0.9,
            )
            db.create_memory_item(
                kind="hardware",
                title="BRIO note",
                text="The Logitech BRIO uses ALSA device plughw:2,0.",
                source_type="manual",
            )
            db.create_memory_item(
                kind="preference",
                title="Expired preference",
                text="Expired memory about concise summaries should not be returned.",
                source_type="manual",
                valid_until="2020-01-01T00:00:00+00:00",
            )

            results = db.search_memory_items("concise summary", limit=5)

            self.assertEqual([item["id"] for item in results], [matching_id])
            self.assertEqual(results[0]["kind"], "preference")
            self.assertEqual(results[0]["title"], "Summary preference")
            self.assertIn("[concise]", results[0]["snippet"].lower())

    def test_memory_vectors_can_be_searched_and_follow_memory_lifecycle(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            close_id = db.create_memory_item(
                kind="preference",
                title="Concise replies",
                text="Prefer concise replies with actions first.",
                source_type="manual",
            )
            far_id = db.create_memory_item(
                kind="fact",
                title="Office color",
                text="The office wall is blue.",
                source_type="manual",
            )
            expired_id = db.create_memory_item(
                kind="fact",
                title="Expired concise memory",
                text="Expired memory about concise answers.",
                source_type="manual",
                valid_until="2020-01-01T00:00:00+00:00",
            )

            db.upsert_memory_vector(close_id, [1.0, 0.0, 0.0], model="test-3d")
            db.upsert_memory_vector(far_id, [0.0, 1.0, 0.0], model="test-3d")
            db.upsert_memory_vector(expired_id, [0.99, 0.0, 0.0], model="test-3d")

            results = db.search_memory_items_by_vector([0.9, 0.1, 0.0], model="test-3d", limit=5)

            self.assertEqual([item["id"] for item in results], [close_id, far_id])
            self.assertGreater(results[0]["vector_score"], results[1]["vector_score"])
            self.assertEqual(results[0]["vector_model"], "test-3d")

            self.assertTrue(db.delete_memory_item(close_id))

            remaining = db.search_memory_items_by_vector([1.0, 0.0, 0.0], model="test-3d", limit=5)
            self.assertNotIn(close_id, [item["id"] for item in remaining])


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

    def test_skill_scores_can_be_recorded_and_filtered(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "db.sqlite")
            db.initialize()
            goal_id = db.create_coaching_goal(
                title="Improve clarity",
                metric="clarity",
            )

            clarity_id = db.record_skill_score(
                goal_id=goal_id,
                domain="conversation",
                metric="clarity",
                value=0.82,
                evidence_count=4,
                period_start="2026-07-01",
                period_end="2026-07-07",
            )
            db.record_skill_score(
                goal_id=goal_id,
                domain="writing",
                metric="concision",
                value=0.6,
                evidence_count=2,
                period_start="2026-07-08",
                period_end="2026-07-14",
            )

            score = db.get_skill_score(clarity_id)
            goal_scores = db.list_skill_scores(goal_id=goal_id)
            conversation_scores = db.list_skill_scores(domain="conversation")
            clarity_scores = db.list_skill_scores(metric="clarity")

            self.assertEqual(score["goal_id"], goal_id)
            self.assertEqual(score["domain"], "conversation")
            self.assertEqual(score["metric"], "clarity")
            self.assertEqual(score["value"], 0.82)
            self.assertEqual(score["evidence_count"], 4)
            self.assertEqual(score["period_start"], "2026-07-01")
            self.assertEqual(score["period_end"], "2026-07-07")
            self.assertEqual([item["id"] for item in conversation_scores], [clarity_id])
            self.assertEqual([item["id"] for item in clarity_scores], [clarity_id])
            self.assertEqual(len(goal_scores), 2)


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
