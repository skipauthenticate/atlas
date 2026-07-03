from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterable[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS recordings (
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
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (duplicate_of) REFERENCES recordings(id)
                );

                CREATE INDEX IF NOT EXISTS idx_recordings_sha
                    ON recordings(sha256);
                CREATE INDEX IF NOT EXISTS idx_recordings_source
                    ON recordings(source_path);
                CREATE INDEX IF NOT EXISTS idx_recordings_status
                    ON recordings(status);

                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    step TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    UNIQUE(recording_id, step),
                    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status
                    ON jobs(status, created_at, id);
                CREATE INDEX IF NOT EXISTS idx_jobs_recording
                    ON jobs(recording_id, step);

                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    speaker TEXT NOT NULL,
                    text TEXT NOT NULL,
                    words_json TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_segments_recording
                    ON segments(recording_id, idx);

                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    template_id TEXT,
                    chunks_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
                    recording_id UNINDEXED,
                    kind UNINDEXED,
                    speaker UNINDEXED,
                    text
                );

                CREATE TABLE IF NOT EXISTS recording_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL UNIQUE,
                    summary_template TEXT NOT NULL DEFAULT 'meeting',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ambient_sessions (
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

                CREATE INDEX IF NOT EXISTS idx_ambient_sessions_status
                    ON ambient_sessions(status, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ambient_sessions_mode
                    ON ambient_sessions(mode, started_at DESC);

                CREATE TABLE IF NOT EXISTS utterances (
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
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES ambient_sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_utterances_session
                    ON utterances(session_id, idx);

                CREATE TABLE IF NOT EXISTS assistant_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_utterance_id INTEGER,
                    text TEXT NOT NULL,
                    audio_path TEXT,
                    model TEXT NOT NULL,
                    latency_ms INTEGER,
                    tool_calls_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES ambient_sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_utterance_id) REFERENCES utterances(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_assistant_turns_session
                    ON assistant_turns(session_id, id);

                CREATE TABLE IF NOT EXISTS model_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task TEXT NOT NULL,
                    input_ref TEXT,
                    output_ref TEXT,
                    latency_ms INTEGER,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    error TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_model_runs_created
                    ON model_runs(created_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_model_runs_task
                    ON model_runs(task, created_at DESC);

                CREATE TABLE IF NOT EXISTS privacy_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'info',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_privacy_events_created
                    ON privacy_events(created_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_privacy_events_type
                    ON privacy_events(event_type, created_at DESC);

                CREATE TABLE IF NOT EXISTS coaching_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    target_date TEXT,
                    metric TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_coaching_goals_status
                    ON coaching_goals(status, target_date, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_coaching_goals_updated
                    ON coaching_goals(updated_at DESC, id DESC);

                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id INTEGER,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    message TEXT NOT NULL,
                    score REAL,
                    evidence_ref TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES coaching_goals(id) ON DELETE SET NULL,
                    FOREIGN KEY (session_id) REFERENCES ambient_sessions(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_events_created
                    ON feedback_events(created_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_events_goal
                    ON feedback_events(goal_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_events_session
                    ON feedback_events(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_events_category
                    ON feedback_events(category, created_at DESC);

                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT,
                    importance REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    valid_from TEXT,
                    valid_until TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memory_items_kind
                    ON memory_items(kind, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_items_source
                    ON memory_items(source_type, source_id);
                CREATE INDEX IF NOT EXISTS idx_memory_items_valid_until
                    ON memory_items(valid_until);

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    memory_id UNINDEXED,
                    kind UNINDEXED,
                    title,
                    text
                );
                """
            )

            # --- Migration helpers for existing databases ---
            existing_tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            if "recording_settings" not in existing_tables:
                conn.execute(
                    """
                    CREATE TABLE recording_settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        recording_id TEXT NOT NULL UNIQUE,
                        summary_template TEXT NOT NULL DEFAULT 'meeting',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
                    )
                    """
                )

            if "summaries" in existing_tables:
                # Add template_id column if it doesn't exist
                summary_cols = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(summaries)").fetchall()
                }
                if "template_id" not in summary_cols:
                    conn.execute(
                        "ALTER TABLE summaries ADD COLUMN template_id TEXT"
                    )

            conn.execute(
                """
                INSERT INTO memory_fts (memory_id, kind, title, text)
                SELECT m.id, m.kind, m.title, m.text
                FROM memory_items m
                WHERE NOT EXISTS (
                    SELECT 1 FROM memory_fts f WHERE f.memory_id = m.id
                )
                """
            )

    def create_recording(self, source_path: Path | str, title: str | None = None) -> str:
        recording_id = uuid.uuid4().hex
        now = utc_now()
        path = str(Path(source_path).expanduser().resolve())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO recordings (
                    id, source_path, title, status, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?)
                """,
                (recording_id, path, title or Path(path).name, now, now),
            )
        return recording_id

    def get_recording(self, recording_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM recordings WHERE id = ?", (recording_id,)
            ).fetchone()

    def get_recording_by_source_path(self, source_path: Path | str) -> sqlite3.Row | None:
        path = str(Path(source_path).expanduser().resolve())
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM recordings WHERE source_path = ? ORDER BY created_at DESC LIMIT 1",
                (path,),
            ).fetchone()

    def get_recording_by_sha(self, sha256: str, exclude_id: str | None = None) -> sqlite3.Row | None:
        query = "SELECT * FROM recordings WHERE sha256 = ?"
        params: list[Any] = [sha256]
        if exclude_id:
            query += " AND id != ?"
            params.append(exclude_id)
        query += " ORDER BY created_at ASC LIMIT 1"
        with self.connect() as conn:
            return conn.execute(query, params).fetchone()

    def list_recordings(self, limit: int = 100) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM recordings ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            )

    def update_recording(self, recording_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [recording_id]
        with self.connect() as conn:
            conn.execute(
                f"UPDATE recordings SET {assignments} WHERE id = ?",
                values,
            )

    def enqueue_job(
        self,
        recording_id: str,
        step: str,
        *,
        max_attempts: int = 3,
        reset: bool = False,
    ) -> None:
        now = utc_now()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE recording_id = ? AND step = ?",
                (recording_id, step),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        recording_id, step, status, max_attempts, created_at, updated_at
                    ) VALUES (?, ?, 'queued', ?, ?, ?)
                    """,
                    (recording_id, step, max_attempts, now, now),
                )
                return
            if reset or existing["status"] != "done":
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued',
                        error = NULL,
                        started_at = NULL,
                        finished_at = NULL,
                        updated_at = ?
                    WHERE recording_id = ? AND step = ?
                    """,
                    (now, recording_id, step),
                )

    def claim_next_job(self) -> sqlite3.Row | None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """
            ).fetchone()
            if job is None:
                return None
            now = utc_now()
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    attempts = attempts + 1,
                    started_at = ?,
                    updated_at = ?,
                    error = NULL
                WHERE id = ?
                """,
                (now, now, job["id"]),
            )
            return conn.execute("SELECT * FROM jobs WHERE id = ?", (job["id"],)).fetchone()

    def complete_job(self, job_id: int) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'done',
                    error = NULL,
                    finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, job_id),
            )

    def fail_job(self, job_id: int, error: str) -> sqlite3.Row:
        now = utc_now()
        with self.connect() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise ValueError(f"Unknown job id: {job_id}")
            next_status = "failed" if job["attempts"] >= job["max_attempts"] else "queued"
            conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    error = ?,
                    finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (next_status, error[:4000], now, now, job_id),
            )
            return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def jobs_for_recording(self, recording_id: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE recording_id = ?
                    ORDER BY id ASC
                    """,
                    (recording_id,),
                ).fetchall()
            )

    def retry_recording(self, recording_id: str) -> None:
        now = utc_now()
        with self.connect() as conn:
            failed_jobs = conn.execute(
                """
                SELECT * FROM jobs
                WHERE recording_id = ? AND status = 'failed'
                ORDER BY id ASC
                """,
                (recording_id,),
            ).fetchall()
            for job in failed_jobs:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued',
                        error = NULL,
                        started_at = NULL,
                        finished_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, job["id"]),
                )
            conn.execute(
                """
                UPDATE recordings
                SET status = 'queued',
                    error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, recording_id),
            )

    def reset_summary_job(self, recording_id: str) -> bool:
        """Reset the summarize job so it will be re-run next tick.

        Returns True if a summarize job was found and reset.
        """
        now = utc_now()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE recording_id = ? AND step = 'summarize'",
                (recording_id,),
            ).fetchone()
            if existing is None:
                return False
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    attempts = 0,
                    error = NULL,
                    started_at = NULL,
                    finished_at = NULL,
                    updated_at = ?
                WHERE recording_id = ? AND step = 'summarize'
                """,
                (now, recording_id),
            )
            conn.execute(
                """
                UPDATE recordings
                SET status = 'queued',
                    error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, recording_id),
            )
            return True

    def replace_segments(self, recording_id: str, segments: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM segments WHERE recording_id = ?", (recording_id,))
            conn.execute(
                "DELETE FROM search_fts WHERE recording_id = ? AND kind = 'transcript'",
                (recording_id,),
            )
            for idx, segment in enumerate(segments):
                conn.execute(
                    """
                    INSERT INTO segments (
                        recording_id, idx, start, end, speaker, text, words_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        recording_id,
                        idx,
                        float(segment["start"]),
                        float(segment["end"]),
                        segment.get("speaker") or "SPEAKER_UNKNOWN",
                        segment.get("text") or "",
                        json.dumps(segment.get("words") or []),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO search_fts (recording_id, kind, speaker, text)
                    VALUES (?, 'transcript', ?, ?)
                    """,
                    (
                        recording_id,
                        segment.get("speaker") or "SPEAKER_UNKNOWN",
                        segment.get("text") or "",
                    ),
                )

    def get_segments(self, recording_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM segments
                WHERE recording_id = ?
                ORDER BY idx ASC
                """,
                (recording_id,),
            ).fetchall()
        return [
            {
                "idx": row["idx"],
                "start": row["start"],
                "end": row["end"],
                "speaker": row["speaker"],
                "text": row["text"],
                "words": json.loads(row["words_json"] or "[]"),
            }
            for row in rows
        ]

    def save_summary(
        self,
        recording_id: str,
        text: str,
        *,
        model: str,
        template_id: str | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        now = utc_now()
        chunks_json = json.dumps(chunks or [])
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO summaries (recording_id, text, model, template_id, chunks_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(recording_id) DO UPDATE SET
                    text = excluded.text,
                    model = excluded.model,
                    template_id = excluded.template_id,
                    chunks_json = excluded.chunks_json,
                    created_at = excluded.created_at
                """,
                (recording_id, text, model, template_id, chunks_json, now),
            )
            conn.execute(
                "DELETE FROM search_fts WHERE recording_id = ? AND kind = 'summary'",
                (recording_id,),
            )
            conn.execute(
                """
                INSERT INTO search_fts (recording_id, kind, speaker, text)
                VALUES (?, 'summary', '', ?)
                """,
                (recording_id, text),
            )

    def get_summary(self, recording_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM summaries WHERE recording_id = ?", (recording_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "text": row["text"],
            "model": row["model"],
            "template_id": row["template_id"],
            "chunks": json.loads(row["chunks_json"] or "[]"),
            "created_at": row["created_at"],
        }

    def set_recording_template(
        self, recording_id: str, template_id: str
    ) -> None:
        """Set or update the preferred summary template for a recording."""
        now = utc_now()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT * FROM recording_settings WHERE recording_id = ?",
                (recording_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO recording_settings
                        (recording_id, summary_template, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (recording_id, template_id, now, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE recording_settings
                    SET summary_template = ?, updated_at = ?
                    WHERE recording_id = ?
                    """,
                    (template_id, now, recording_id),
                )

    def get_recording_template(self, recording_id: str) -> str | None:
        """Return the preferred template id for a recording, or None."""
        try:
            with self.connect() as conn:
                row = conn.execute(
                    "SELECT summary_template FROM recording_settings WHERE recording_id = ?",
                    (recording_id,),
                ).fetchone()
            return row["summary_template"] if row else None
        except Exception:
            return None

    def list_ambient_sessions(
        self,
        limit: int = 20,
        *,
        status: str | None = None,
        mode: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                s.*,
                COUNT(DISTINCT u.id) AS utterance_count,
                COUNT(DISTINCT t.id) AS assistant_turn_count
            FROM ambient_sessions s
            LEFT JOIN utterances u ON u.session_id = s.id
            LEFT JOIN assistant_turns t ON t.session_id = s.id
        """
        params: list[Any] = []
        filters: list[str] = []
        if status:
            filters.append("s.status = ?")
            params.append(status)
        if mode:
            filters.append("s.mode = ?")
            params.append(mode)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " GROUP BY s.id ORDER BY s.started_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def count_ambient_sessions(self, *, status: str | None = None) -> int:
        query = "SELECT COUNT(*) AS count FROM ambient_sessions"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        with self.connect() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row["count"] if row else 0)

    def find_privacy_purge_sessions(
        self,
        *,
        session_id: str | None = None,
        keyword: str | None = None,
        person: str | None = None,
        started_on: str | None = None,
        before: str | None = None,
        after: str | None = None,
        mode: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                s.*,
                (SELECT COUNT(*) FROM utterances u WHERE u.session_id = s.id) AS utterance_count,
                (SELECT COUNT(*) FROM assistant_turns t WHERE t.session_id = s.id)
                    AS assistant_turn_count
            FROM ambient_sessions s
        """
        filters: list[str] = []
        params: list[Any] = []
        if session_id:
            filters.append("s.id = ?")
            params.append(session_id)
        if mode:
            filters.append("s.mode = ?")
            params.append(mode)
        if started_on:
            filters.append("substr(s.started_at, 1, 10) = ?")
            params.append(started_on)
        if before:
            filters.append("substr(s.started_at, 1, 10) < ?")
            params.append(before)
        if after:
            filters.append("substr(s.started_at, 1, 10) >= ?")
            params.append(after)
        if keyword:
            pattern = f"%{keyword.strip().lower()}%"
            filters.append(
                "("
                "lower(COALESCE(s.title, '')) LIKE ? OR "
                "EXISTS (SELECT 1 FROM utterances u "
                "WHERE u.session_id = s.id AND lower(u.text) LIKE ?) OR "
                "EXISTS (SELECT 1 FROM assistant_turns t "
                "WHERE t.session_id = s.id AND lower(t.text) LIKE ?)"
                ")"
            )
            params.extend([pattern, pattern, pattern])
        if person:
            pattern = f"%{person.strip().lower()}%"
            filters.append(
                "("
                "lower(COALESCE(s.title, '')) LIKE ? OR "
                "EXISTS (SELECT 1 FROM utterances u WHERE u.session_id = s.id "
                "AND (lower(u.speaker) LIKE ? OR lower(u.text) LIKE ?))"
                ")"
            )
            params.extend([pattern, pattern, pattern])
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY s.started_at DESC LIMIT ?"
        params.append(max(min(limit, 1000), 1))
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def purge_privacy_sessions(self, session_ids: list[str]) -> dict[str, Any]:
        unique_ids = list(dict.fromkeys(session_ids))
        if not unique_ids:
            return {
                "session_count": 0,
                "utterance_count": 0,
                "assistant_turn_count": 0,
                "session_ids": [],
            }
        placeholders = ", ".join("?" for _ in unique_ids)
        with self.connect() as conn:
            session_count = int(
                conn.execute(
                    f"SELECT COUNT(*) AS count FROM ambient_sessions WHERE id IN ({placeholders})",
                    unique_ids,
                ).fetchone()["count"]
            )
            utterance_count = int(
                conn.execute(
                    f"SELECT COUNT(*) AS count FROM utterances WHERE session_id IN ({placeholders})",
                    unique_ids,
                ).fetchone()["count"]
            )
            assistant_turn_count = int(
                conn.execute(
                    f"SELECT COUNT(*) AS count FROM assistant_turns WHERE session_id IN ({placeholders})",
                    unique_ids,
                ).fetchone()["count"]
            )
            conn.execute(f"DELETE FROM ambient_sessions WHERE id IN ({placeholders})", unique_ids)
        return {
            "session_count": session_count,
            "utterance_count": utterance_count,
            "assistant_turn_count": assistant_turn_count,
            "session_ids": unique_ids,
        }

    def create_ambient_session(
        self,
        *,
        mode: str,
        source: str,
        retention_policy: str = "ephemeral",
        title: str | None = None,
        status: str = "active",
    ) -> str:
        session_id = uuid.uuid4().hex
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ambient_sessions (
                    id, mode, source, started_at, status, retention_policy,
                    title, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, mode, source, now, status, retention_policy, title, now, now),
            )
        return session_id

    def end_ambient_session(self, session_id: str, *, status: str = "ended") -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE ambient_sessions
                SET status = ?, ended_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, now, now, session_id),
            )

    def get_ambient_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM ambient_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def add_utterance(
        self,
        *,
        session_id: str,
        text: str,
        speaker: str = "user",
        start: float | None = None,
        end: float | None = None,
        confidence: float | None = None,
        source_provider: str = "text",
        is_directed_to_assistant: bool = True,
        sensitivity: str | None = None,
    ) -> int:
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(idx) + 1, 0) AS next_idx FROM utterances WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            idx = int(row["next_idx"] if row else 0)
            cursor = conn.execute(
                """
                INSERT INTO utterances (
                    session_id, idx, start, end, speaker, text, confidence,
                    source_provider, is_directed_to_assistant, sensitivity, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    idx,
                    start,
                    end,
                    speaker,
                    text,
                    confidence,
                    source_provider,
                    1 if is_directed_to_assistant else 0,
                    sensitivity,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def list_utterances(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM utterances
                WHERE session_id = ?
                ORDER BY idx ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_assistant_turn(
        self,
        *,
        session_id: str,
        text: str,
        model: str,
        user_utterance_id: int | None = None,
        audio_path: str | None = None,
        latency_ms: int | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> int:
        now = utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO assistant_turns (
                    session_id, user_utterance_id, text, audio_path, model,
                    latency_ms, tool_calls_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_utterance_id,
                    text,
                    audio_path,
                    model,
                    latency_ms,
                    json.dumps(tool_calls or []),
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def list_assistant_turns(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM assistant_turns
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        turns: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["tool_calls"] = json.loads(item.pop("tool_calls_json") or "[]")
            except json.JSONDecodeError:
                item["tool_calls"] = []
            turns.append(item)
        return turns

    def create_memory_item(
        self,
        *,
        kind: str,
        title: str,
        text: str,
        source_type: str,
        source_id: str | None = None,
        importance: float = 0.0,
        confidence: float = 0.0,
        valid_from: str | None = None,
        valid_until: str | None = None,
    ) -> int:
        now = utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memory_items (
                    kind, title, text, source_type, source_id, importance, confidence,
                    valid_from, valid_until, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    title,
                    text,
                    source_type,
                    source_id,
                    importance,
                    confidence,
                    valid_from,
                    valid_until,
                    now,
                    now,
                ),
            )
            memory_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO memory_fts (memory_id, kind, title, text)
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, kind, title, text),
            )
            return memory_id

    def get_memory_item(self, memory_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_items WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_memory_items(
        self,
        *,
        kind: str | None = None,
        source_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM memory_items"
        params: list[Any] = []
        filters: list[str] = []
        if kind:
            filters.append("kind = ?")
            params.append(kind)
        if source_type:
            filters.append("source_type = ?")
            params.append(source_type)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(max(min(limit, 500), 1))
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_memory_item(self, memory_id: int, **fields: Any) -> bool:
        allowed = {
            "kind",
            "title",
            "text",
            "importance",
            "confidence",
            "valid_from",
            "valid_until",
        }
        updates: dict[str, Any] = {}
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"Unsupported memory item field: {key}")
            updates[key] = value
        if not updates:
            return False
        updates["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [memory_id]
        with self.connect() as conn:
            cursor = conn.execute(
                f"UPDATE memory_items SET {assignments} WHERE id = ?",
                values,
            )
            if cursor.rowcount <= 0:
                return False
            row = conn.execute(
                "SELECT kind, title, text FROM memory_items WHERE id = ?",
                (memory_id,),
            ).fetchone()
            conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            if row is not None:
                conn.execute(
                    """
                    INSERT INTO memory_fts (memory_id, kind, title, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (memory_id, row["kind"], row["title"], row["text"]),
                )
            return True

    def delete_memory_item(self, memory_id: int) -> bool:
        with self.connect() as conn:
            conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            cursor = conn.execute("DELETE FROM memory_items WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    def search_memory_items(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        match = self._fts_query(query)
        if not match:
            return []
        now = utc_now()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    m.*,
                    snippet(memory_fts, 3, '[', ']', '...', 20) AS snippet
                FROM memory_fts f
                JOIN memory_items m ON m.id = f.memory_id
                WHERE memory_fts MATCH ?
                    AND (m.valid_from IS NULL OR m.valid_from <= ?)
                    AND (m.valid_until IS NULL OR m.valid_until >= ?)
                ORDER BY rank, m.importance DESC, m.updated_at DESC
                LIMIT ?
                """,
                (match, now, now, max(min(limit, 100), 1)),
            ).fetchall()
        return [dict(row) for row in rows]


    def create_coaching_goal(
        self,
        *,
        title: str,
        description: str | None = None,
        status: str = "active",
        target_date: str | None = None,
        metric: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO coaching_goals (
                    title, description, status, target_date, metric, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title,
                    description,
                    status,
                    target_date,
                    metric,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def update_coaching_goal(self, goal_id: int, **fields: Any) -> bool:
        allowed = {
            "title": "title",
            "description": "description",
            "status": "status",
            "target_date": "target_date",
            "metric": "metric",
            "completed_at": "completed_at",
            "metadata": "metadata_json",
        }
        updates: dict[str, Any] = {}
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"Unsupported coaching goal field: {key}")
            updates[allowed[key]] = json.dumps(value or {}, sort_keys=True) if key == "metadata" else value
        if not updates:
            return False
        now = utc_now()
        if updates.get("status") == "completed" and "completed_at" not in updates:
            updates["completed_at"] = now
        updates["updated_at"] = now
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [goal_id]
        with self.connect() as conn:
            cursor = conn.execute(
                f"UPDATE coaching_goals SET {assignments} WHERE id = ?",
                values,
            )
            return cursor.rowcount > 0

    def get_coaching_goal(self, goal_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM coaching_goals WHERE id = ?",
                (goal_id,),
            ).fetchone()
        return self._coaching_goal_from_row(row)

    def list_coaching_goals(
        self,
        *,
        status: str | None = "active",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM coaching_goals"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(max(min(limit, 500), 1))
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._coaching_goal_from_row(row) for row in rows]

    @staticmethod
    def _coaching_goal_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            item["metadata"] = {}
        return item

    def log_feedback_event(
        self,
        *,
        event_type: str,
        category: str,
        message: str,
        goal_id: int | None = None,
        session_id: str | None = None,
        score: float | None = None,
        evidence_ref: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback_events (
                    goal_id, session_id, event_type, category, message, score,
                    evidence_ref, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal_id,
                    session_id,
                    event_type,
                    category,
                    message[:4000],
                    score,
                    evidence_ref,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def get_feedback_event(self, event_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM feedback_events WHERE id = ?",
                (event_id,),
            ).fetchone()
        return self._feedback_event_from_row(row)

    def list_feedback_events(
        self,
        *,
        goal_id: int | None = None,
        session_id: str | None = None,
        category: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM feedback_events"
        params: list[Any] = []
        filters: list[str] = []
        if goal_id is not None:
            filters.append("goal_id = ?")
            params.append(goal_id)
        if session_id is not None:
            filters.append("session_id = ?")
            params.append(session_id)
        if category is not None:
            filters.append("category = ?")
            params.append(category)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(min(limit, 500), 1))
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._feedback_event_from_row(row) for row in rows]

    @staticmethod
    def _feedback_event_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            item["metadata"] = {}
        return item

    def log_model_run(
        self,
        *,
        provider: str,
        model: str,
        task: str,
        input_ref: str | None = None,
        output_ref: str | None = None,
        latency_ms: int | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        error: str | None = None,
    ) -> int:
        now = utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO model_runs (
                    provider, model, task, input_ref, output_ref, latency_ms,
                    tokens_in, tokens_out, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider,
                    model,
                    task,
                    input_ref,
                    output_ref,
                    latency_ms,
                    tokens_in,
                    tokens_out,
                    error[:4000] if error else None,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def list_model_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM model_runs
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def log_privacy_event(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO privacy_events (
                    event_type, message, severity, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    message[:4000],
                    severity,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def list_privacy_events(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM privacy_events
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            except json.JSONDecodeError:
                item["metadata"] = {}
            events.append(item)
        return events

    def search(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        match = self._fts_query(query)
        if not match:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    f.recording_id,
                    f.kind,
                    f.speaker,
                    snippet(search_fts, 3, '[', ']', '...', 20) AS snippet,
                    r.title,
                    r.status
                FROM search_fts f
                JOIN recordings r ON r.id = f.recording_id
                WHERE search_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        return " OR ".join(f"{token}*" for token in tokens)


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}
