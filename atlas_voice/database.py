from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


RECORDING_LIBRARY_SUMMARY_EXCERPT_CHARS = 600


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
                CREATE TABLE IF NOT EXISTS recording_folders (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recordings (
                    id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    title_origin TEXT NOT NULL DEFAULT 'legacy',
                    status TEXT NOT NULL,
                    sha256 TEXT,
                    duplicate_of TEXT,
                    folder_id TEXT,
                    original_path TEXT,
                    normalized_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (duplicate_of) REFERENCES recordings(id),
                    FOREIGN KEY (folder_id) REFERENCES recording_folders(id) ON DELETE SET NULL
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
                    available_at TEXT,
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
                    expected_main_speakers INTEGER,
                    quality_tier TEXT NOT NULL DEFAULT 'torch',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS recording_speakers (
                    recording_id TEXT NOT NULL,
                    speaker_label TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (recording_id, speaker_label),
                    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS runtime_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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

                CREATE VIRTUAL TABLE IF NOT EXISTS conversation_fts USING fts5(
                    session_id UNINDEXED,
                    item_id UNINDEXED,
                    kind UNINDEXED,
                    speaker UNINDEXED,
                    text
                );

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

                CREATE TABLE IF NOT EXISTS skill_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id INTEGER,
                    domain TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL,
                    evidence_count INTEGER NOT NULL DEFAULT 0,
                    period_start TEXT,
                    period_end TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES coaching_goals(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_skill_scores_goal
                    ON skill_scores(goal_id, period_end DESC, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_skill_scores_domain_metric
                    ON skill_scores(domain, metric, period_end DESC, created_at DESC);

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

                CREATE TABLE IF NOT EXISTS memory_vectors (
                    memory_id INTEGER PRIMARY KEY,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedding_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memory_items(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_memory_vectors_model
                    ON memory_vectors(model, dimensions);

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    memory_id UNINDEXED,
                    kind UNINDEXED,
                    title,
                    text
                );

                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
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
                        expected_main_speakers INTEGER,
                        quality_tier TEXT NOT NULL DEFAULT 'torch',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
                    )
                    """
                )

            recording_setting_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(recording_settings)").fetchall()
            }
            if "expected_main_speakers" not in recording_setting_cols:
                conn.execute(
                    "ALTER TABLE recording_settings ADD COLUMN expected_main_speakers INTEGER"
                )
            if "quality_tier" not in recording_setting_cols:
                conn.execute(
                    "ALTER TABLE recording_settings ADD COLUMN quality_tier TEXT NOT NULL DEFAULT 'torch'"
                )

            if "jobs" in existing_tables:
                job_cols = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
                }
                if "available_at" not in job_cols:
                    conn.execute("ALTER TABLE jobs ADD COLUMN available_at TEXT")

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

            if "recordings" in existing_tables:
                recording_cols = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(recordings)").fetchall()
                }
                if "folder_id" not in recording_cols:
                    conn.execute(
                        """
                        ALTER TABLE recordings
                        ADD COLUMN folder_id TEXT
                            REFERENCES recording_folders(id) ON DELETE SET NULL
                        """
                    )
                if "title_origin" not in recording_cols:
                    # Existing titles may already be user-authored. Mark them as
                    # legacy so automatic title generation cannot replace them.
                    conn.execute(
                        """
                        ALTER TABLE recordings
                        ADD COLUMN title_origin TEXT NOT NULL DEFAULT 'legacy'
                        """
                    )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_recordings_folder
                    ON recordings(folder_id, created_at DESC)
                    """
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

            conversation_fts_migration = "conversation_fts_v1"
            conversation_fts_backfilled = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE name = ?",
                (conversation_fts_migration,),
            ).fetchone()
            if conversation_fts_backfilled is None:
                # Rebuild once so databases created before conversation search gain
                # a complete, duplicate-free index without paying this cost at each
                # startup.
                conn.execute("DELETE FROM conversation_fts")
                conn.execute(
                    """
                    INSERT INTO conversation_fts (
                        session_id, item_id, kind, speaker, text
                    )
                    SELECT session_id, id, 'utterance', speaker, text
                    FROM utterances
                    """
                )
                conn.execute(
                    """
                    INSERT INTO conversation_fts (
                        session_id, item_id, kind, speaker, text
                    )
                    SELECT session_id, id, 'assistant', 'assistant', text
                    FROM assistant_turns
                    """
                )
                conn.execute(
                    """
                    INSERT INTO schema_migrations (name, applied_at)
                    VALUES (?, ?)
                    """,
                    (conversation_fts_migration, utc_now()),
                )

    def create_recording(self, source_path: Path | str, title: str | None = None) -> str:
        recording_id = uuid.uuid4().hex
        now = utc_now()
        path = str(Path(source_path).expanduser().resolve())
        clean_title = " ".join(str(title or "").split())
        stored_title = clean_title or Path(path).name
        title_origin = "manual" if clean_title else "filename"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO recordings (
                    id, source_path, title, title_origin, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?)
                """,
                (recording_id, path, stored_title, title_origin, now, now),
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

    def list_recording_library(
        self,
        folder_id: str | None = None,
        status: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return a bounded, paginated projection for the recording library.

        ``folder_id=None`` includes every folder. An empty folder id selects
        recordings that have not been filed yet. ``status="processing"``
        includes every non-terminal pipeline status.
        """
        bounded_limit = max(min(limit, 1000), 0)
        if bounded_limit == 0:
            return []
        bounded_offset = max(offset, 0)
        where, params = self._recording_library_where(folder_id, status)
        query = f"""
            WITH selected_recordings AS (
                SELECT r.id, r.created_at
                FROM recordings r
                {where}
                ORDER BY r.created_at DESC, r.id DESC
                LIMIT ? OFFSET ?
            )
            SELECT
                r.id,
                r.title,
                r.title_origin,
                r.status,
                r.folder_id,
                r.created_at,
                r.updated_at,
                f.name AS folder_name,
                substr(s.text, 1, {RECORDING_LIBRARY_SUMMARY_EXCERPT_CHARS}) AS summary,
                (
                    SELECT MAX(segment.end)
                    FROM segments segment
                    WHERE segment.recording_id = r.id
                ) AS duration_seconds
            FROM selected_recordings selected
            JOIN recordings r ON r.id = selected.id
            LEFT JOIN recording_folders f ON f.id = r.folder_id
            LEFT JOIN summaries s ON s.recording_id = r.id
            ORDER BY selected.created_at DESC, selected.id DESC
        """
        params.extend([bounded_limit, bounded_offset])
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def count_recording_library(
        self,
        folder_id: str | None = None,
        status: str | None = None,
    ) -> int:
        where, params = self._recording_library_where(folder_id, status)
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS count FROM recordings r {where}",
                params,
            ).fetchone()
        return int(row["count"] if row is not None else 0)

    def recording_library_folder_counts(self) -> dict[str, Any]:
        """Return exact all, unfiled, and per-folder recording counts."""
        with self.connect() as conn:
            totals = conn.execute(
                """
                SELECT
                    COUNT(*) AS all_count,
                    COALESCE(SUM(CASE WHEN folder_id IS NULL THEN 1 ELSE 0 END), 0)
                        AS unfiled_count
                FROM recordings
                """
            ).fetchone()
            folder_rows = conn.execute(
                """
                SELECT f.id, COUNT(r.id) AS recording_count
                FROM recording_folders f
                LEFT JOIN recordings r ON r.folder_id = f.id
                GROUP BY f.id
                ORDER BY f.name COLLATE NOCASE ASC, f.id ASC
                """
            ).fetchall()
        return {
            "all": int(totals["all_count"]),
            "unfiled": int(totals["unfiled_count"]),
            "folders": {row["id"]: int(row["recording_count"]) for row in folder_rows},
        }

    @staticmethod
    def _recording_library_where(
        folder_id: str | None,
        status: str | None,
    ) -> tuple[str, list[Any]]:
        filters: list[str] = []
        params: list[Any] = []
        if folder_id == "":
            filters.append("r.folder_id IS NULL")
        elif folder_id is not None:
            filters.append("r.folder_id = ?")
            params.append(folder_id)
        if status == "processing":
            filters.append("r.status NOT IN ('done', 'failed', 'duplicate')")
        elif status is not None:
            filters.append("r.status = ?")
            params.append(status)
        return ("WHERE " + " AND ".join(filters) if filters else ""), params

    def create_recording_folder(self, name: str) -> str:
        folder_id = uuid.uuid4().hex
        clean_name = self._recording_folder_name(name)
        now = utc_now()
        try:
            with self.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO recording_folders (id, name, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (folder_id, clean_name, now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"A recording folder named '{clean_name}' already exists") from exc
        return folder_id

    def get_recording_folder(self, folder_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM recording_folders WHERE id = ?",
                (folder_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_recording_folders(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    f.id,
                    f.name,
                    f.created_at,
                    f.updated_at,
                    COUNT(r.id) AS recording_count
                FROM recording_folders f
                LEFT JOIN recordings r ON r.folder_id = f.id
                GROUP BY f.id, f.name, f.created_at, f.updated_at
                ORDER BY f.name COLLATE NOCASE ASC, f.id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def rename_recording_folder(self, folder_id: str, name: str) -> bool:
        clean_name = self._recording_folder_name(name)
        try:
            with self.connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE recording_folders
                    SET name = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (clean_name, utc_now(), folder_id),
                )
                return cursor.rowcount > 0
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"A recording folder named '{clean_name}' already exists") from exc

    def delete_recording_folder(self, folder_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM recording_folders WHERE id = ?",
                (folder_id,),
            )
            return cursor.rowcount > 0

    def set_recording_folder(
        self,
        recording_id: str,
        folder_id: str | None,
    ) -> bool:
        normalized_folder_id = str(folder_id or "").strip() or None
        with self.connect() as conn:
            if normalized_folder_id is not None:
                folder = conn.execute(
                    "SELECT 1 FROM recording_folders WHERE id = ?",
                    (normalized_folder_id,),
                ).fetchone()
                if folder is None:
                    raise ValueError(f"Unknown recording folder: {normalized_folder_id}")
            cursor = conn.execute(
                """
                UPDATE recordings
                SET folder_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (normalized_folder_id, utc_now(), recording_id),
            )
            return cursor.rowcount > 0

    def update_recording_title(
        self,
        recording_id: str,
        title: str,
        *,
        origin: str = "manual",
        expected_origin: str | None = None,
    ) -> bool:
        clean_title = " ".join(str(title or "").split())
        if not clean_title:
            raise ValueError("Recording title must not be empty")
        if len(clean_title) > 160:
            raise ValueError("Recording title must be 160 characters or fewer")
        allowed_origins = {"filename", "generated", "manual", "legacy"}
        if origin not in allowed_origins:
            raise ValueError(
                "Recording title origin must be one of: filename, generated, manual, legacy"
            )
        if expected_origin is not None and expected_origin not in allowed_origins:
            raise ValueError(f"Unknown expected recording title origin: {expected_origin}")
        query = """
            UPDATE recordings
            SET title = ?, title_origin = ?, updated_at = ?
            WHERE id = ?
        """
        params: list[Any] = [clean_title, origin, utc_now(), recording_id]
        if expected_origin is not None:
            query += " AND title_origin = ?"
            params.append(expected_origin)
        with self.connect() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount > 0

    @staticmethod
    def _recording_folder_name(name: str) -> str:
        clean_name = " ".join(str(name or "").split())
        if not clean_name:
            raise ValueError("Recording folder name must not be empty")
        if len(clean_name) > 80:
            raise ValueError("Recording folder name must be 80 characters or fewer")
        return clean_name

    def list_recent_recording_summaries(self, limit: int = 10) -> list[dict[str, Any]]:
        bounded_limit = max(min(limit, 100), 0)
        if bounded_limit == 0:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    r.id AS recording_id,
                    r.title,
                    r.source_path,
                    r.created_at,
                    s.text AS summary
                FROM summaries s
                JOIN recordings r ON r.id = s.recording_id
                WHERE trim(s.text) != ''
                  AND r.status = 'done'
                ORDER BY r.created_at DESC, r.id DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_recording_titles(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        bounded_limit = max(min(limit, 100), 0)
        if not tokens or bounded_limit == 0:
            return []
        title_filters = " AND ".join("lower(r.title) LIKE ?" for _ in tokens)
        params: list[Any] = [f"%{token.casefold()}%" for token in tokens]
        params.append(bounded_limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    r.id AS recording_id,
                    r.title,
                    r.source_path,
                    r.created_at,
                    r.status,
                    s.text AS summary
                FROM recordings r
                LEFT JOIN summaries s ON s.recording_id = r.id
                WHERE {title_filters}
                  AND r.status != 'duplicate'
                ORDER BY r.created_at DESC, r.id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def search_recording(
        self,
        recording_id: str,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search transcript and summary text within exactly one recording."""
        match = self._fts_query(query)
        bounded_limit = max(min(limit, 100), 0)
        if not recording_id or not match or bounded_limit == 0:
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
                  AND f.recording_id = ?
                ORDER BY f.rank ASC, f.rowid ASC
                LIMIT ?
                """,
                (match, recording_id, bounded_limit),
            ).fetchall()
        return [dict(row) for row in rows]

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

    def recover_running_jobs(self) -> int:
        """Return jobs abandoned by a previous single-worker process to the queue."""
        now = utc_now()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, recording_id FROM jobs WHERE status = 'running'"
            ).fetchall()
            if not rows:
                return 0
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    attempts = MAX(attempts - 1, 0),
                    error = 'Recovered after worker restart.',
                    started_at = NULL,
                    finished_at = NULL,
                    updated_at = ?,
                    available_at = NULL
                WHERE status = 'running'
                """,
                (now,),
            )
            conn.executemany(
                """
                UPDATE recordings
                SET status = 'queued', error = NULL, updated_at = ?
                WHERE id = ?
                """,
                [(now, str(row["recording_id"])) for row in rows],
            )
            return len(rows)

    def claim_next_job(self) -> sqlite3.Row | None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = utc_now()
            job = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                  AND (available_at IS NULL OR available_at <= ?)
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if job is None:
                return None
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    attempts = attempts + 1,
                    started_at = ?,
                    updated_at = ?,
                    error = NULL,
                    available_at = NULL
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

    def defer_job(
        self,
        job_id: int,
        message: str,
        *,
        delay_seconds: float = 30.0,
    ) -> sqlite3.Row:
        now_value = datetime.now(timezone.utc)
        now = now_value.isoformat(timespec="seconds")
        available_at = (now_value + timedelta(seconds=max(delay_seconds, 1.0))).isoformat(
            timespec="seconds"
        )
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
                raise ValueError(f"Unknown job id: {job_id}")
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    attempts = MAX(attempts - 1, 0),
                    error = ?,
                    started_at = NULL,
                    finished_at = NULL,
                    updated_at = ?,
                    available_at = ?
                WHERE id = ?
                """,
                (str(message)[:4000], now, available_at, job_id),
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
                SELECT s.*, rs.display_name
                FROM segments s
                LEFT JOIN recording_speakers rs
                    ON rs.recording_id = s.recording_id
                   AND rs.speaker_label = s.speaker
                WHERE s.recording_id = ?
                ORDER BY s.idx ASC
                """,
                (recording_id,),
            ).fetchall()
        return [
            {
                "idx": row["idx"],
                "start": row["start"],
                "end": row["end"],
                "speaker_label": row["speaker"],
                "speaker": row["display_name"] or row["speaker"],
                "text": row["text"],
                "words": json.loads(row["words_json"] or "[]"),
            }
            for row in rows
        ]

    def get_recording_segment_stats(self, recording_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS segment_count, MAX(end) AS duration_seconds
                FROM segments
                WHERE recording_id = ?
                """,
                (recording_id,),
            ).fetchone()
        return {
            "segment_count": int(row["segment_count"] or 0),
            "duration_seconds": (
                float(row["duration_seconds"])
                if row["duration_seconds"] is not None
                else None
            ),
        }

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
            return str(row["summary_template"] or "").strip() or None
        except Exception:
            return None

    def set_recording_processing_options(
        self,
        recording_id: str,
        *,
        expected_main_speakers: int | None,
        quality_tier: str,
    ) -> None:
        if expected_main_speakers is not None:
            expected_main_speakers = int(expected_main_speakers)
            if expected_main_speakers < 1 or expected_main_speakers > 20:
                raise ValueError("Expected main speakers must be between 1 and 20")
        tier = str(quality_tier or "").strip().lower()
        if tier not in {"light", "torch", "fire"}:
            raise ValueError("Quality tier must be light, torch, or fire")
        if self.get_recording(recording_id) is None:
            raise ValueError(f"Unknown recording: {recording_id}")
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO recording_settings (
                    recording_id,
                    summary_template,
                    expected_main_speakers,
                    quality_tier,
                    created_at,
                    updated_at
                )
                VALUES (?, '', ?, ?, ?, ?)
                ON CONFLICT(recording_id) DO UPDATE SET
                    expected_main_speakers = excluded.expected_main_speakers,
                    quality_tier = excluded.quality_tier,
                    updated_at = excluded.updated_at
                """,
                (recording_id, expected_main_speakers, tier, now, now),
            )

    def get_recording_processing_options(self, recording_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT expected_main_speakers, quality_tier
                FROM recording_settings
                WHERE recording_id = ?
                """,
                (recording_id,),
            ).fetchone()
        return {
            "expected_main_speakers": (
                int(row["expected_main_speakers"])
                if row is not None and row["expected_main_speakers"] is not None
                else None
            ),
            "quality_tier": (
                str(row["quality_tier"] or "torch")
                if row is not None
                else "torch"
            ),
        }

    def list_recording_speakers(self, recording_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    s.speaker AS speaker_label,
                    rs.display_name,
                    COUNT(*) AS segment_count,
                    SUM(MAX(s.end - s.start, 0.0)) AS talk_seconds,
                    MIN(s.start) AS first_seen
                FROM segments s
                LEFT JOIN recording_speakers rs
                    ON rs.recording_id = s.recording_id
                   AND rs.speaker_label = s.speaker
                WHERE s.recording_id = ?
                GROUP BY s.speaker, rs.display_name
                ORDER BY talk_seconds DESC, first_seen ASC, s.speaker ASC
                """,
                (recording_id,),
            ).fetchall()
        expected = self.get_recording_processing_options(recording_id)[
            "expected_main_speakers"
        ]
        speakers: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            label = str(row["speaker_label"])
            speakers.append(
                {
                    "speaker_label": label,
                    "display_name": str(row["display_name"] or ""),
                    "default_name": _friendly_speaker_name(label, index),
                    "segment_count": int(row["segment_count"] or 0),
                    "talk_seconds": round(float(row["talk_seconds"] or 0.0), 3),
                    "first_seen": float(row["first_seen"] or 0.0),
                    "is_main": expected is None or index < expected,
                }
            )
        return speakers

    def save_recording_speaker_names(
        self,
        recording_id: str,
        names: dict[str, str],
    ) -> list[dict[str, Any]]:
        speakers = self.list_recording_speakers(recording_id)
        allowed = {str(speaker["speaker_label"]) for speaker in speakers}
        cleaned: dict[str, str] = {}
        used_names: set[str] = set()
        for label, raw_name in names.items():
            if label not in allowed:
                raise ValueError(f"Unknown speaker label: {label}")
            name = " ".join(str(raw_name or "").split())
            if len(name) > 80:
                raise ValueError("Speaker names must be 80 characters or fewer")
            if not name:
                continue
            folded = name.casefold()
            if folded in used_names:
                raise ValueError("Each detected speaker needs a distinct name")
            used_names.add(folded)
            cleaned[label] = name

        now = utc_now()
        with self.connect() as conn:
            for label in allowed:
                name = cleaned.get(label)
                if name is None:
                    conn.execute(
                        """
                        DELETE FROM recording_speakers
                        WHERE recording_id = ? AND speaker_label = ?
                        """,
                        (recording_id, label),
                    )
                    continue
                conn.execute(
                    """
                    INSERT INTO recording_speakers (
                        recording_id, speaker_label, display_name, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(recording_id, speaker_label) DO UPDATE SET
                        display_name = excluded.display_name,
                        updated_at = excluded.updated_at
                    """,
                    (recording_id, label, name, now, now),
                )

            conn.execute(
                "DELETE FROM search_fts WHERE recording_id = ? AND kind = 'transcript'",
                (recording_id,),
            )
            rows = conn.execute(
                """
                SELECT s.speaker, s.text, rs.display_name
                FROM segments s
                LEFT JOIN recording_speakers rs
                    ON rs.recording_id = s.recording_id
                   AND rs.speaker_label = s.speaker
                WHERE s.recording_id = ?
                ORDER BY s.idx ASC
                """,
                (recording_id,),
            ).fetchall()
            conn.executemany(
                """
                INSERT INTO search_fts (recording_id, kind, speaker, text)
                VALUES (?, 'transcript', ?, ?)
                """,
                [
                    (
                        recording_id,
                        str(row["display_name"] or row["speaker"]),
                        str(row["text"] or ""),
                    )
                    for row in rows
                ],
            )
        return self.list_recording_speakers(recording_id)

    def set_runtime_setting(self, key: str, value: str) -> None:
        clean_key = str(key or "").strip()
        if not clean_key:
            raise ValueError("Runtime setting key is required")
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (clean_key, str(value), now),
            )

    def get_runtime_setting(self, key: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM runtime_settings WHERE key = ?",
                (str(key or "").strip(),),
            ).fetchone()
        return str(row["value"]) if row is not None else None

    def list_ambient_sessions(
        self,
        limit: int = 20,
        *,
        status: str | None = None,
        mode: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        search_query = query
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
        cleaned_query = (search_query or "").strip().lower()
        if cleaned_query:
            pattern = f"%{cleaned_query}%"
            filters.append(
                "("
                "lower(COALESCE(s.title, '')) LIKE ? OR "
                "lower(COALESCE(s.source, '')) LIKE ? OR "
                "EXISTS (SELECT 1 FROM utterances u "
                "WHERE u.session_id = s.id "
                "AND (lower(COALESCE(u.text, '')) LIKE ? "
                "OR lower(COALESCE(u.speaker, '')) LIKE ?)) OR "
                "EXISTS (SELECT 1 FROM assistant_turns t "
                "WHERE t.session_id = s.id AND lower(COALESCE(t.text, '')) LIKE ?)"
                ")"
            )
            params.extend([pattern, pattern, pattern, pattern, pattern])
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " GROUP BY s.id ORDER BY s.started_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def search_conversations(
        self,
        query: str,
        limit: int = 20,
        *,
        mode: str | None = None,
        exclude_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        match = self._fts_query(query)
        if not match or limit <= 0:
            return []

        sql = """
            SELECT
                f.session_id,
                CAST(f.item_id AS INTEGER) AS item_id,
                f.kind,
                f.speaker,
                snippet(conversation_fts, 4, '[', ']', '...', 20) AS snippet,
                s.title,
                s.mode,
                s.started_at,
                f.rank AS rank
            FROM conversation_fts f
            JOIN ambient_sessions s ON s.id = f.session_id
            WHERE conversation_fts MATCH ?
        """
        params: list[Any] = [match]
        if mode:
            sql += " AND s.mode = ?"
            params.append(mode)
        if exclude_session_id:
            sql += " AND f.session_id != ?"
            params.append(exclude_session_id)
        sql += " ORDER BY f.rank ASC, s.started_at DESC, f.rowid ASC LIMIT ?"
        params.append(limit)

        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def list_recent_conversation_excerpts(
        self,
        *,
        session_limit: int = 4,
        utterances_per_session: int = 2,
        mode: str | None = None,
        exclude_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        bounded_sessions = max(min(session_limit, 20), 0)
        bounded_utterances = max(min(utterances_per_session, 10), 0)
        if bounded_sessions == 0 or bounded_utterances == 0:
            return []

        filters = [
            "EXISTS (SELECT 1 FROM utterances existing WHERE existing.session_id = s.id)"
        ]
        params: list[Any] = []
        if mode:
            filters.append("s.mode = ?")
            params.append(mode)
        if exclude_session_id:
            filters.append("s.id != ?")
            params.append(exclude_session_id)
        where = " AND ".join(filters)
        params.extend([bounded_sessions, bounded_utterances])
        sql = f"""
            WITH recent_sessions AS (
                SELECT s.id, s.title, s.mode, s.started_at
                FROM ambient_sessions s
                WHERE {where}
                ORDER BY s.started_at DESC, s.id DESC
                LIMIT ?
            ),
            ranked_utterances AS (
                SELECT
                    u.id,
                    u.session_id,
                    u.idx,
                    u.speaker,
                    u.source_provider,
                    u.text,
                    ROW_NUMBER() OVER (
                        PARTITION BY u.session_id
                        ORDER BY u.idx DESC, u.id DESC
                    ) AS recent_rank
                FROM utterances u
                JOIN recent_sessions recent ON recent.id = u.session_id
                WHERE trim(u.text) != ''
            )
            SELECT
                recent.id AS session_id,
                recent.title,
                recent.mode,
                recent.started_at,
                utterance.id AS utterance_id,
                utterance.idx,
                utterance.speaker,
                utterance.source_provider,
                utterance.text
            FROM recent_sessions recent
            JOIN ranked_utterances utterance ON utterance.session_id = recent.id
            WHERE utterance.recent_rank <= ?
            ORDER BY recent.started_at DESC, recent.id DESC, utterance.idx ASC
        """
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        sessions: dict[str, dict[str, Any]] = {}
        for row in rows:
            session_id = str(row["session_id"])
            session = sessions.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "title": row["title"],
                    "mode": row["mode"],
                    "started_at": row["started_at"],
                    "utterances": [],
                },
            )
            session["utterances"].append(
                {
                    "id": int(row["utterance_id"]),
                    "idx": int(row["idx"]),
                    "speaker": row["speaker"],
                    "source_provider": row["source_provider"],
                    "text": row["text"],
                }
            )
        return list(sessions.values())

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
                "model_run_count": 0,
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
            linked_run_filter = f"""
                EXISTS (
                    SELECT 1 FROM utterances u
                    WHERE u.session_id IN ({placeholders})
                      AND (
                          model_runs.input_ref = 'utterance:' || u.id
                          OR model_runs.output_ref = 'utterance:' || u.id
                      )
                )
                OR EXISTS (
                    SELECT 1 FROM assistant_turns t
                    WHERE t.session_id IN ({placeholders})
                      AND (
                          model_runs.input_ref = 'assistant_turn:' || t.id
                          OR model_runs.output_ref = 'assistant_turn:' || t.id
                      )
                )
            """
            linked_run_params = [*unique_ids, *unique_ids]
            model_run_count = int(
                conn.execute(
                    f"SELECT COUNT(*) AS count FROM model_runs WHERE {linked_run_filter}",
                    linked_run_params,
                ).fetchone()["count"]
            )
            conn.execute(
                f"DELETE FROM model_runs WHERE {linked_run_filter}",
                linked_run_params,
            )
            conn.execute(
                f"DELETE FROM conversation_fts WHERE session_id IN ({placeholders})",
                unique_ids,
            )
            conn.execute(f"DELETE FROM ambient_sessions WHERE id IN ({placeholders})", unique_ids)
        return {
            "session_count": session_count,
            "utterance_count": utterance_count,
            "assistant_turn_count": assistant_turn_count,
            "model_run_count": model_run_count,
            "session_ids": unique_ids,
        }

    def delete_ambient_session(self, session_id: str) -> dict[str, Any]:
        return self.purge_privacy_sessions([session_id])

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
            utterance_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO conversation_fts (
                    session_id, item_id, kind, speaker, text
                ) VALUES (?, ?, 'utterance', ?, ?)
                """,
                (session_id, utterance_id, speaker, text),
            )
            return utterance_id

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
            turn_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO conversation_fts (
                    session_id, item_id, kind, speaker, text
                ) VALUES (?, ?, 'assistant', 'assistant', ?)
                """,
                (session_id, turn_id, text),
            )
            return turn_id

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
            conn.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))
            cursor = conn.execute("DELETE FROM memory_items WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    def upsert_memory_vector(
        self,
        memory_id: int,
        embedding: list[float],
        *,
        model: str,
    ) -> None:
        clean_embedding = _clean_embedding(embedding)
        if not clean_embedding:
            raise ValueError("Memory vector embedding must not be empty")
        now = utc_now()
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM memory_items WHERE id = ?", (memory_id,)).fetchone() is None:
                raise ValueError(f"Unknown memory item: {memory_id}")
            conn.execute(
                """
                INSERT INTO memory_vectors (
                    memory_id, model, dimensions, embedding_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    model = excluded.model,
                    dimensions = excluded.dimensions,
                    embedding_json = excluded.embedding_json,
                    updated_at = excluded.updated_at
                """,
                (
                    memory_id,
                    model,
                    len(clean_embedding),
                    json.dumps(clean_embedding),
                    now,
                ),
            )

    def search_memory_items_by_vector(
        self,
        embedding: list[float],
        *,
        model: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query_embedding = _clean_embedding(embedding)
        if not query_embedding:
            return []
        now = utc_now()
        sql = """
            SELECT
                m.*,
                v.model AS vector_model,
                v.embedding_json AS vector_embedding_json
            FROM memory_vectors v
            JOIN memory_items m ON m.id = v.memory_id
            WHERE v.dimensions = ?
                AND (m.valid_from IS NULL OR m.valid_from <= ?)
                AND (m.valid_until IS NULL OR m.valid_until >= ?)
        """
        params: list[Any] = [len(query_embedding), now, now]
        if model is not None:
            sql += " AND v.model = ?"
            params.append(model)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        scored: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                candidate_embedding = json.loads(str(item.pop("vector_embedding_json") or "[]"))
            except json.JSONDecodeError:
                continue
            score = _cosine_similarity(query_embedding, _clean_embedding(candidate_embedding))
            if score is None:
                continue
            item["vector_score"] = score
            scored.append(item)
        scored.sort(key=lambda item: (item["vector_score"], item.get("importance") or 0.0), reverse=True)
        return scored[: max(min(limit, 100), 1)]

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

    def record_skill_score(
        self,
        *,
        domain: str,
        metric: str,
        value: float,
        goal_id: int | None = None,
        evidence_count: int = 0,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> int:
        now = utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO skill_scores (
                    goal_id, domain, metric, value, evidence_count,
                    period_start, period_end, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal_id,
                    domain,
                    metric,
                    value,
                    max(evidence_count, 0),
                    period_start,
                    period_end,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def get_skill_score(self, score_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM skill_scores WHERE id = ?",
                (score_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_skill_scores(
        self,
        *,
        goal_id: int | None = None,
        domain: str | None = None,
        metric: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM skill_scores"
        params: list[Any] = []
        filters: list[str] = []
        if goal_id is not None:
            filters.append("goal_id = ?")
            params.append(goal_id)
        if domain is not None:
            filters.append("domain = ?")
            params.append(domain)
        if metric is not None:
            filters.append("metric = ?")
            params.append(metric)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY period_end DESC, created_at DESC, id DESC LIMIT ?"
        params.append(max(min(limit, 500), 1))
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


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

    def search(
        self,
        query: str,
        limit: int = 50,
        *,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(min(int(limit), 200), 0)
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        if not tokens or bounded_limit == 0:
            return []
        selected_kind = str(kind or "").strip().lower()
        if selected_kind not in {"", "summary", "transcript", "speaker"}:
            raise ValueError("Search kind must be summary, transcript, or speaker")

        if selected_kind == "speaker":
            pattern = "%" + " ".join(tokens).casefold() + "%"
            with self.connect() as conn:
                rows = conn.execute(
                    """
                    SELECT DISTINCT
                        f.recording_id,
                        'speaker' AS kind,
                        f.speaker,
                        f.speaker AS snippet,
                        r.title,
                        r.status,
                        r.created_at,
                        0.0 AS rank
                    FROM search_fts f
                    JOIN recordings r ON r.id = f.recording_id
                    WHERE f.kind = 'transcript'
                      AND lower(f.speaker) LIKE ?
                    ORDER BY r.created_at DESC
                    LIMIT ?
                    """,
                    (pattern, bounded_limit),
                ).fetchall()
            return [dict(row) for row in rows]

        def fetch(match: str) -> list[dict[str, Any]]:
            filters = ["search_fts MATCH ?"]
            params: list[Any] = [match]
            if selected_kind:
                filters.append("f.kind = ?")
                params.append(selected_kind)
            params.append(bounded_limit)
            with self.connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT
                        f.recording_id,
                        f.kind,
                        f.speaker,
                        snippet(search_fts, 3, '[', ']', '...', 28) AS snippet,
                        r.title,
                        r.status,
                        r.created_at,
                        f.rank AS rank
                    FROM search_fts f
                    JOIN recordings r ON r.id = f.recording_id
                    WHERE {" AND ".join(filters)}
                    ORDER BY f.rank ASC, r.created_at DESC, f.rowid ASC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
            return [dict(row) for row in rows]

        results = fetch(self._fts_query(query, operator="AND"))
        if not results and len(tokens) > 1:
            results = fetch(self._fts_query(query, operator="OR"))

        if not selected_kind:
            speaker_results = self.search(
                query,
                limit=bounded_limit,
                kind="speaker",
            )
            results = speaker_results + results
            known_recordings = {str(result["recording_id"]) for result in results}
            title_results = []
            for match in self.search_recording_titles(query, limit=bounded_limit):
                recording_id = str(match["recording_id"])
                if recording_id in known_recordings:
                    continue
                title_results.append(
                    {
                        "recording_id": recording_id,
                        "kind": "title",
                        "speaker": "",
                        "snippet": str(match["title"]),
                        "title": str(match["title"]),
                        "status": str(match.get("status") or "queued"),
                        "created_at": match.get("created_at"),
                        "rank": -1.0,
                    }
                )
            results = title_results + results
        return results[:bounded_limit]

    @staticmethod
    def _fts_query(query: str, *, operator: str = "OR") -> str:
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        joiner = " AND " if operator.upper() == "AND" else " OR "
        return joiner.join(f'"{token}"*' for token in tokens)


def _friendly_speaker_name(label: str, fallback_index: int) -> str:
    match = re.fullmatch(r"SPEAKER[_ -]?(\d+)", str(label or ""), flags=re.IGNORECASE)
    if match:
        return f"Speaker {int(match.group(1)) + 1}"
    if str(label or "").strip().lower() in {"unknown", "speaker_unknown"}:
        return "Unknown voice"
    return f"Speaker {fallback_index + 1}"


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _clean_embedding(embedding: list[float] | tuple[float, ...]) -> list[float]:
    clean: list[float] = []
    for value in embedding:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("Memory vector embedding values must be numeric") from None
        if not math.isfinite(number):
            raise ValueError("Memory vector embedding values must be finite")
        clean.append(number)
    return clean


def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    score = sum(left_value * right_value for left_value, right_value in zip(left, right)) / (
        left_norm * right_norm
    )
    return round(score, 6)
