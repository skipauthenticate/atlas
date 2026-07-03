from __future__ import annotations

from typing import Any

import httpx

from .config import Settings
from .database import Database
from .exporter import export_payload
from .merge import format_seconds


class AnythingLLMError(RuntimeError):
    """Raised when AnythingLLM rejects or cannot receive a recording document."""


class AnythingLLMConfigError(AnythingLLMError):
    """Raised when the AnythingLLM integration is missing required settings."""


def anythingllm_configured(settings: Settings) -> bool:
    return all(
        _clean(value)
        for value in (
            settings.anythingllm_base_url,
            settings.anythingllm_api_key,
            settings.anythingllm_workspace_slug,
        )
    )


def build_anythingllm_document(
    db: Database,
    recording_id: str,
    *,
    workspace_slug: str | None = None,
) -> dict[str, Any]:
    """Build a rich AnythingLLM raw-text document without local filesystem paths."""
    payload = export_payload(db, recording_id)
    recording = payload["recording"]
    summary = payload.get("summary") or {}
    segments = payload.get("segments") or []

    summary_text = _clean(summary.get("text"))
    if not summary_text and not segments:
        raise ValueError("Recording has no transcript or summary to sync yet.")

    text_content = _document_text(payload)
    body: dict[str, Any] = {
        "textContent": text_content,
        "metadata": _document_metadata(recording),
    }
    if workspace_slug:
        body["addToWorkspaces"] = workspace_slug
    return body


def sync_recording_to_anythingllm(
    db: Database,
    recording_id: str,
    settings: Settings,
) -> dict[str, Any]:
    base_url, api_key, workspace_slug = _required_settings(settings)
    document = build_anythingllm_document(
        db,
        recording_id,
        workspace_slug=workspace_slug,
    )
    url = anythingllm_api_url(base_url, "/v1/document/raw-text")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        with httpx.Client(timeout=settings.anythingllm_timeout, follow_redirects=True) as client:
            response = client.post(url, headers=headers, json=document)
    except httpx.RequestError as exc:
        raise AnythingLLMError(f"AnythingLLM request failed: {exc}") from exc

    if response.status_code >= 400:
        raise AnythingLLMError(
            f"AnythingLLM returned HTTP {response.status_code}: {_safe_response_error(response)}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise AnythingLLMError("AnythingLLM returned a non-JSON response") from exc

    if data.get("success") is False:
        raise AnythingLLMError(
            f"AnythingLLM rejected the recording: {_safe_error_value(data.get('error'))}"
        )
    return data


def anythingllm_api_url(base_url: str, endpoint: str) -> str:
    endpoint = "/" + endpoint.lstrip("/")
    base = base_url.rstrip("/")
    if endpoint.startswith("/v1/") and (base.endswith("/api/v1") or base.endswith("/v1")):
        return base + endpoint[len("/v1") :]
    if base.endswith("/api"):
        return base + endpoint
    return base + "/api" + endpoint


def _required_settings(settings: Settings) -> tuple[str, str, str]:
    base_url = _clean(settings.anythingllm_base_url)
    api_key = _clean(settings.anythingllm_api_key)
    workspace_slug = _clean(settings.anythingllm_workspace_slug)
    missing = []
    if not base_url:
        missing.append("ANYTHINGLLM_BASE_URL")
    if not api_key:
        missing.append("ANYTHINGLLM_API_KEY")
    if not workspace_slug:
        missing.append("ANYTHINGLLM_WORKSPACE_SLUG")
    if missing:
        raise AnythingLLMConfigError(
            "Missing AnythingLLM setting(s): " + ", ".join(missing)
        )
    return base_url, api_key, workspace_slug


def _document_text(payload: dict[str, Any]) -> str:
    recording = payload["recording"]
    summary = payload.get("summary") or {}
    segments = payload.get("segments") or []
    jobs = payload.get("jobs") or []

    title = _clean(recording.get("title")) or recording["id"]
    lines = [
        f"# {title}",
        "",
        "## Recording",
        f"- Atlas Voice recording ID: {recording['id']}",
        f"- Status: {_clean(recording.get('status')) or 'unknown'}",
    ]
    if recording.get("created_at"):
        lines.append(f"- Created at: {recording['created_at']}")
    if recording.get("updated_at"):
        lines.append(f"- Updated at: {recording['updated_at']}")

    if jobs:
        lines.extend(["", "## Processing"])
        for job in jobs:
            attempts = job.get("attempts")
            max_attempts = job.get("max_attempts")
            attempts_label = ""
            if attempts is not None and max_attempts is not None:
                attempts_label = f" ({attempts}/{max_attempts} attempts)"
            lines.append(
                f"- {_clean(job.get('step')) or 'unknown'}: "
                f"{_clean(job.get('status')) or 'unknown'}{attempts_label}"
            )

    lines.extend(["", "## Summary", "", _clean(summary.get("text")) or "No summary available."])

    chunks = summary.get("chunks") or []
    chunk_lines = _summary_chunk_lines(chunks)
    if chunk_lines:
        lines.extend(["", "## Summary Chunks", "", *chunk_lines])

    lines.extend(["", "## Transcript", ""])
    if segments:
        for segment in segments:
            start = format_seconds(float(segment.get("start") or 0))
            end = format_seconds(float(segment.get("end") or 0))
            speaker = _clean(segment.get("speaker")) or "SPEAKER_UNKNOWN"
            text = _clean(segment.get("text"))
            lines.append(f"[{start}-{end}] {speaker}: {text}")
    else:
        lines.append("No transcript segments available.")

    return "\n".join(lines).rstrip() + "\n"


def _summary_chunk_lines(chunks: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        text = _clean(chunk.get("text"))
        if not text:
            continue
        chunk_index = chunk.get("chunk_index") or index
        lines.extend([f"### Chunk {chunk_index}", "", text, ""])
    return lines[:-1] if lines else []


def _document_metadata(recording: dict[str, Any]) -> dict[str, str]:
    recording_id = str(recording["id"])
    return {
        "title": _clean(recording.get("title")) or recording_id,
        "docAuthor": "Atlas Voice",
        "description": "Atlas Voice recording transcript, summary, and processing metadata.",
        "docSource": f"atlas-voice://recordings/{recording_id}",
        "chunkSource": f"atlas-voice://recordings/{recording_id}",
    }


def _safe_response_error(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return _safe_error_value(response.text)
    for key in ("error", "message", "detail"):
        value = data.get(key) if isinstance(data, dict) else None
        if value:
            return _safe_error_value(value)
    return response.reason_phrase or "request failed"


def _safe_error_value(value: Any) -> str:
    text = str(value or "request failed").replace("\n", " ").replace("\r", " ").strip()
    return text[:500]


def _clean(value: Any) -> str:
    return str(value or "").strip()
