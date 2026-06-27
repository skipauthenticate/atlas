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
        chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        now = utc_now()
        chunks_json = json.dumps(chunks or [])
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO summaries (recording_id, text, model, chunks_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(recording_id) DO UPDATE SET
                    text = excluded.text,
                    model = excluded.model,
                    chunks_json = excluded.chunks_json,
                    created_at = excluded.created_at
                """,
                (recording_id, text, model, chunks_json, now),
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
            "chunks": json.loads(row["chunks_json"] or "[]"),
            "created_at": row["created_at"],
        }

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
                    snippet(search_fts, 3, '<mark>', '</mark>', '...', 20) AS snippet,
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
