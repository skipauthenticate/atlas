from __future__ import annotations

import time
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
from .quality import normalize_quality_tier, settings_for_quality
from .resources import admit_pipeline_step
from .storage import FileStorage
from .summarizer import (
    chunk_transcript,
    detect_template,
    get_template,
    summarize_with_llm,
)
from .titles import generate_recording_title


PIPELINE_STEPS = ["ingest", "normalize", "transcribe", "diarize", "merge", "summarize"]


class PipelineProcessor:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.storage = FileStorage(settings, db)

    def enqueue_source(
        self,
        source_path: Path | str,
        *,
        expected_main_speakers: int | None = None,
        quality_tier: str | None = None,
    ) -> str:
        self.settings.ensure_directories()
        self.db.initialize()
        existing = self.db.get_recording_by_source_path(source_path)
        if existing and existing["status"] in {"queued", "ingested", "processing", "done"}:
            return existing["id"]
        recording_id = self.db.create_recording(source_path)
        tier = normalize_quality_tier(
            quality_tier,
            default=self.settings.default_quality_tier,
        )
        self.db.set_recording_processing_options(
            recording_id,
            expected_main_speakers=expected_main_speakers,
            quality_tier=tier,
        )
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
        if not self.settings.stub_mode and step in {"transcribe", "diarize"}:
            options = self.db.get_recording_processing_options(recording_id)
            admission = admit_pipeline_step(options["quality_tier"], step)
            if not admission.allowed:
                self.db.defer_job(
                    int(job["id"]),
                    admission.message,
                    delay_seconds=45,
                )
                self.db.update_recording(
                    recording_id, status="waiting_resources", error=None
                )
                return
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

    def _processing_settings(self, recording_id: str) -> tuple[Settings, dict[str, Any]]:
        options = self.db.get_recording_processing_options(recording_id)
        return settings_for_quality(self.settings, options["quality_tier"]), options

    def _normalize(self, recording_id: str) -> None:
        recording = self._recording(recording_id)
        original_path = recording.get("original_path")
        if not original_path:
            raise RuntimeError("Recording has no original audio path")
        normalized_path = self.storage.normalized_path(recording_id)
        processing_settings, _options = self._processing_settings(recording_id)
        normalize_audio(
            Path(original_path),
            normalized_path,
            cleanup_mode=processing_settings.audio_cleanup,
        )
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
        processing_settings, options = self._processing_settings(recording_id)
        transcript = transcribe_audio(Path(normalized_path), processing_settings)
        transcript.setdefault(
            "atlas",
            {
                "schema_version": 1,
                "quality_tier": options["quality_tier"],
                "provider": processing_settings.asr_provider,
                "model": processing_settings.asr_model
                or processing_settings.whisperx_model,
                "beam_size": processing_settings.asr_beam_size,
            },
        )
        self.storage.write_json(recording_id, "transcript.json", transcript)
        self.db.update_recording(recording_id, status="transcribed", error=None)

    def _diarize(self, recording_id: str) -> None:
        recording = self._recording(recording_id)
        normalized_path = recording.get("normalized_path")
        if not normalized_path:
            raise RuntimeError("Recording has no normalized audio path")
        processing_settings, options = self._processing_settings(recording_id)
        transcript = None
        if processing_settings.diarization_provider == "transcript":
            transcript = self.storage.read_json(recording_id, "transcript.json")
        diarization = diarize_audio(
            Path(normalized_path),
            processing_settings,
            transcript=transcript,
            expected_speakers=options["expected_main_speakers"],
        )
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
        started = time.perf_counter()
        provider = "stub" if self.settings.stub_mode else "openai-compatible"
        try:
            recording = self._recording(recording_id)
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
            if recording.get("title_origin") == "filename":
                self._generate_title(recording_id, text, provider=provider)
            self.db.update_recording(recording_id, status="done", error=None)
        except Exception as exc:
            self.db.log_model_run(
                provider=provider,
                model=self.settings.llm_model,
                task="summarize",
                input_ref=f"recording:{recording_id}",
                latency_ms=_elapsed_ms(started),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self.db.log_model_run(
            provider=provider,
            model=self.settings.llm_model,
            task="summarize",
            input_ref=f"recording:{recording_id}",
            output_ref=f"summary:{recording_id}",
            latency_ms=_elapsed_ms(started),
        )

    def _generate_title(self, recording_id: str, summary: str, *, provider: str) -> None:
        del provider
        started = time.perf_counter()
        try:
            title = generate_recording_title(summary, self.settings)
            if title:
                self.db.update_recording_title(
                    recording_id,
                    title,
                    origin="generated",
                    expected_origin="filename",
                )
        except Exception as exc:  # noqa: BLE001 - title generation is best effort.
            self.db.log_model_run(
                provider="local-deterministic",
                model="summary-headline-v1",
                task="generate_title",
                input_ref=f"summary:{recording_id}",
                latency_ms=_elapsed_ms(started),
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        self.db.log_model_run(
            provider="local-deterministic",
            model="summary-headline-v1",
            task="generate_title",
            input_ref=f"summary:{recording_id}",
            output_ref=f"recording:{recording_id}",
            latency_ms=_elapsed_ms(started),
        )

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


def _elapsed_ms(started: float) -> int:
    return max(int((time.perf_counter() - started) * 1000), 0)
