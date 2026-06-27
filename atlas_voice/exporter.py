from __future__ import annotations

import json
from typing import Any

from .database import Database, row_to_dict
from .merge import format_seconds


def export_recording(db: Database, recording_id: str, export_format: str) -> str:
    payload = export_payload(db, recording_id)
    if export_format == "json":
        return json.dumps(payload, indent=2, ensure_ascii=False)
    if export_format == "md":
        return export_markdown(payload)
    if export_format == "txt":
        return export_text(payload)
    raise ValueError("format must be one of: json, md, txt")


def export_payload(db: Database, recording_id: str) -> dict[str, Any]:
    recording = row_to_dict(db.get_recording(recording_id))
    if recording is None:
        raise ValueError(f"Unknown recording: {recording_id}")
    return {
        "recording": recording,
        "jobs": [dict(row) for row in db.jobs_for_recording(recording_id)],
        "segments": db.get_segments(recording_id),
        "summary": db.get_summary(recording_id),
    }


def export_markdown(payload: dict[str, Any]) -> str:
    recording = payload["recording"]
    summary = payload.get("summary") or {}
    lines = [
        f"# {recording['title']}",
        "",
        f"- Recording ID: `{recording['id']}`",
        f"- Status: `{recording['status']}`",
        f"- SHA-256: `{recording.get('sha256') or ''}`",
        "",
        "## Summary",
        "",
        summary.get("text") or "No summary available.",
        "",
        "## Transcript",
        "",
    ]
    for segment in payload["segments"]:
        lines.append(
            f"**{segment['speaker']}** "
            f"`{format_seconds(segment['start'])}-{format_seconds(segment['end'])}` "
            f"{segment['text']}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_text(payload: dict[str, Any]) -> str:
    recording = payload["recording"]
    summary = payload.get("summary") or {}
    lines = [
        recording["title"],
        "=" * len(recording["title"]),
        "",
        "Summary:",
        summary.get("text") or "No summary available.",
        "",
        "Transcript:",
    ]
    for segment in payload["segments"]:
        lines.append(
            f"[{format_seconds(segment['start'])}-{format_seconds(segment['end'])}] "
            f"{segment['speaker']}: {segment['text']}"
        )
    return "\n".join(lines).rstrip() + "\n"
