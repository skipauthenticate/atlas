from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .config import Settings
from .database import Database


AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "audio"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FileStorage:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db

    def ingest_source(self, recording_id: str) -> dict[str, Any]:
        recording = self.db.get_recording(recording_id)
        if recording is None:
            raise ValueError(f"Unknown recording: {recording_id}")

        source_path = Path(recording["source_path"])
        if not source_path.exists():
            raise FileNotFoundError(f"Source audio does not exist: {source_path}")
        if not is_audio_file(source_path):
            raise ValueError(f"Unsupported audio extension: {source_path.suffix}")

        sha256 = sha256_file(source_path)
        duplicate = self.db.get_recording_by_sha(sha256, exclude_id=recording_id)
        if duplicate is not None:
            self.db.update_recording(
                recording_id,
                sha256=sha256,
                duplicate_of=duplicate["id"],
                status="duplicate",
                error=None,
            )
            return {"duplicate": True, "duplicate_of": duplicate["id"], "sha256": sha256}

        destination = self.settings.originals_dir / (
            f"{recording_id}-{safe_filename(source_path.name)}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

        self.db.update_recording(
            recording_id,
            title=source_path.stem,
            sha256=sha256,
            original_path=str(destination),
            status="ingested",
            error=None,
        )
        return {"duplicate": False, "path": destination, "sha256": sha256}

    def normalized_path(self, recording_id: str) -> Path:
        return self.settings.normalized_dir / f"{recording_id}.wav"

    def artifact_dir(self, recording_id: str) -> Path:
        path = self.settings.artifacts_dir / recording_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, recording_id: str, name: str, payload: Any) -> Path:
        path = self.artifact_dir(recording_id) / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return path

    def read_json(self, recording_id: str, name: str) -> Any:
        path = self.artifact_dir(recording_id) / name
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
