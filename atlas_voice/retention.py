from __future__ import annotations

import shutil
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import Settings
from .database import Database


@dataclass(frozen=True)
class AmbientRetentionResult:
    dry_run: bool
    audio_artifact_paths: tuple[Path, ...]
    session_ids: tuple[str, ...]

    @property
    def audio_artifact_count(self) -> int:
        return len(self.audio_artifact_paths)

    @property
    def session_count(self) -> int:
        return len(self.session_ids)


def apply_ambient_retention(
    settings: Settings,
    db: Database,
    *,
    now: datetime | None = None,
    dry_run: bool = True,
    limit: int = 10000,
) -> AmbientRetentionResult:
    current_time = _as_aware_utc(now or datetime.now(UTC))
    sessions = db.list_ambient_sessions(limit=limit, mode=None)
    transcript_session_ids: list[str] = []
    audio_session_ids: list[str] = []

    for session in sessions:
        if session.get("status") == "active":
            continue
        session_id = str(session.get("id") or "")
        started_at = _parse_timestamp(session.get("started_at"))
        if not session_id or started_at is None:
            continue
        if _transcript_expired(settings, started_at, current_time):
            transcript_session_ids.append(session_id)
            continue
        if _audio_expired(settings, started_at, current_time):
            audio_session_ids.append(session_id)

    session_ids = tuple(dict.fromkeys(transcript_session_ids))
    audio_artifact_paths = _artifact_paths(
        settings,
        [*audio_session_ids, *session_ids],
    )

    if not dry_run:
        for path in audio_artifact_paths:
            with suppress(FileNotFoundError):
                shutil.rmtree(path)
        if session_ids:
            db.purge_privacy_sessions(list(session_ids))

    return AmbientRetentionResult(
        dry_run=dry_run,
        audio_artifact_paths=audio_artifact_paths,
        session_ids=session_ids,
    )


def _transcript_expired(settings: Settings, started_at: datetime, now: datetime) -> bool:
    retention_days = settings.ambient_transcript_retention_days
    if retention_days is None:
        return False
    return _expired(started_at, now, float(max(retention_days, 0)))


def _audio_expired(settings: Settings, started_at: datetime, now: datetime) -> bool:
    return _expired(started_at, now, settings.ambient_raw_audio_retention_days)


def _expired(started_at: datetime, now: datetime, retention_days: float) -> bool:
    if retention_days <= 0:
        return started_at <= now
    return started_at <= now - timedelta(days=retention_days)


def _artifact_paths(settings: Settings, session_ids: list[str]) -> tuple[Path, ...]:
    ambient_artifacts_dir = (settings.artifacts_dir / "ambient").resolve()
    paths: list[Path] = []
    for session_id in dict.fromkeys(session_ids):
        path = (ambient_artifacts_dir / session_id).resolve()
        if not path.exists() or not _is_relative_to(path, ambient_artifacts_dir):
            continue
        paths.append(path)
    return tuple(paths)


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_aware_utc(parsed)


def _as_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
