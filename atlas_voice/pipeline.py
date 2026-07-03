from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from .anythingllm import AnythingLLMError, sync_recording_to_anythingllm
from .audio import normalize_audio
from .config import Settings
from .database import Database
from .merge import merge_transcript_with_diarization
from .providers.asr import transcribe_audio
from .providers.diarization import diarize_audio
from .storage import FileStorage
from .summarizer import (
    chunk_transcript,
    detect_template,
    get_template,
    summarize_with_llm,
)


PIPELINE_STEPS = ["ingest", "normalize", "transcribe", "diarize", "merge", "summarize"]


class PipelineProcessor:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.storage = FileStorage(settings, db)

    def enqueue_source(self, source_path: Path | str) -> str:
        self.settings.ensure_directories()
        self.db.initialize()
        existing = self.db.get_recording_by_source_path(source_path)
        if existing and existing["status"] in {"queued", "ingested", "processing", "done"}:
            return existing["id"]
        recording_id = self.db.create_recording(source_path)
        self.db.enqueue_job(recording_id, "ingest")
        return recording_id

    def process_next(self) -> bool:
        job = self.db.claim_next_job()
        if job is None:
            return False
        self.process_job(dict(job))
        return True

    def process_job(self, job: dict[str, Any]) -> None:
        recording_id = job["recording_id"]
        step = job["step"]
        try:
            self.db.update_recording(recording_id, status=f"{step}_running", error=None)
            should_continue = self._run_step(recording_id, step)
            self.db.complete_job(job["id"])
            if should_continue:
                next_step = self._next_step(step)
                if next_step:
                    self.db.enqueue_job(recording_id, next_step)
                    self.db.update_recording(recording_id, status="queued", error=None)
                else:
                    self.db.update_recording(recording_id, status="done", error=None)
                    if step == "summarize":
                        self._sync_anythingllm_after_summary(recording_id)
        except Exception as exc:
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            failed_job = self.db.fail_job(job["id"], error)
            status = "failed" if failed_job["status"] == "failed" else "queued"
            self.db.update_recording(recording_id, status=status, error=error)

    def _run_step(self, recording_id: str, step: str) -> bool:
        if step == "ingest":
            result = self.storage.ingest_source(recording_id)
            return not bool(result.get("duplicate"))
        if step == "normalize":
            self._normalize(recording_id)
            return True
        if step == "transcribe":
            self._transcribe(recording_id)
            return True
        if step == "diarize":
            self._diarize(recording_id)
            return True
        if step == "merge":
            self._merge(recording_id)
            return True
        if step == "summarize":
            self._summarize(recording_id)
            return True
        raise ValueError(f"Unknown pipeline step: {step}")

    @staticmethod
    def _next_step(step: str) -> str | None:
        try:
            index = PIPELINE_STEPS.index(step)
        except ValueError:
            return None
        next_index = index + 1
        return PIPELINE_STEPS[next_index] if next_index < len(PIPELINE_STEPS) else None

    def _recording(self, recording_id: str) -> dict[str, Any]:
        recording = self.db.get_recording(recording_id)
        if recording is None:
            raise ValueError(f"Unknown recording: {recording_id}")
        return dict(recording)

    def _normalize(self, recording_id: str) -> None:
        recording = self._recording(recording_id)
        original_path = recording.get("original_path")
        if not original_path:
            raise RuntimeError("Recording has no original audio path")
        normalized_path = self.storage.normalized_path(recording_id)
        normalize_audio(Path(original_path), normalized_path)
        self.db.update_recording(
            recording_id,
            normalized_path=str(normalized_path),
            status="normalized",
            error=None,
        )

    def _transcribe(self, recording_id: str) -> None:
        recording = self._recording(recording_id)
        normalized_path = recording.get("normalized_path")
        if not normalized_path:
            raise RuntimeError("Recording has no normalized audio path")
        transcript = transcribe_audio(Path(normalized_path), self.settings)
        self.storage.write_json(recording_id, "transcript.json", transcript)
        self.db.update_recording(recording_id, status="transcribed", error=None)

    def _diarize(self, recording_id: str) -> None:
        recording = self._recording(recording_id)
        normalized_path = recording.get("normalized_path")
        if not normalized_path:
            raise RuntimeError("Recording has no normalized audio path")
        transcript = None
        if self.settings.diarization_provider == "transcript":
            transcript = self.storage.read_json(recording_id, "transcript.json")
        diarization = diarize_audio(Path(normalized_path), self.settings, transcript=transcript)
        self.storage.write_json(recording_id, "diarization.json", diarization)
        self.db.update_recording(recording_id, status="diarized", error=None)

    def _merge(self, recording_id: str) -> None:
        transcript = self.storage.read_json(recording_id, "transcript.json")
        diarization = self.storage.read_json(recording_id, "diarization.json")
        segments = merge_transcript_with_diarization(transcript, diarization)
        self.storage.write_json(recording_id, "segments.json", segments)
        self.db.replace_segments(recording_id, segments)
        self.db.update_recording(recording_id, status="merged", error=None)

    def _summarize(self, recording_id: str) -> None:
        segments = self.db.get_segments(recording_id)
        # Determine template: explicit user preference > auto-detect
        preferred_id = self.db.get_recording_template(recording_id)
        if self.settings.stub_mode:
            text = (
                "Overview\n"
                "Atlas Voice processed a local test recording.\n\n"
                "Key Points\n"
                "- Stub mode is enabled.\n\n"
                "Action Items\n"
                "None\n\n"
                "Open Questions\n"
                "None"
            )
            chunks_payload = [{"chunk_index": 1, "text": text}]
            tpl_id = preferred_id or "meeting"
        else:
            chunks = chunk_transcript(segments)
            # Build a mini transcript for auto-detection (first 2000 chars)
            mini_transcript = "\n".join(
                s.get("text", "") for s in segments[:100]
            )[:2000]
            tpl = None
            if preferred_id:
                tpl = get_template(preferred_id)
            if tpl is None and mini_transcript.strip():
                tpl = detect_template(mini_transcript)
            tpl_id = tpl.id if tpl else "meeting"
            text, chunks_payload = summarize_with_llm(
                chunks, self.settings, template=tpl, template_id=preferred_id
            )
        self.db.save_summary(
            recording_id,
            text,
            model=self.settings.llm_model,
            template_id=tpl_id,
            chunks=chunks_payload,
        )
        self.db.update_recording(recording_id, status="done", error=None)

    def _sync_anythingllm_after_summary(self, recording_id: str) -> None:
        if not self.settings.anythingllm_auto_sync:
            return
        try:
            sync_recording_to_anythingllm(self.db, recording_id, self.settings)
        except (AnythingLLMError, ValueError) as exc:
            self.db.update_recording(
                recording_id,
                error=f"AnythingLLM auto-sync failed: {str(exc)[:500]}",
            )
