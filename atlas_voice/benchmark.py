from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import io
import gc
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse, urlunparse

from atlas_voice.audio import normalize_audio
from atlas_voice.assistant_config import AssistantConfig
from atlas_voice.config import Settings
from atlas_voice.providers.asr import transcribe_audio
from atlas_voice.profile_settings import settings_for_voice_profile
from atlas_voice.providers.diarization import diarize_audio
from atlas_voice.providers.transcript_utils import audio_duration_seconds
from atlas_voice.quality import QUALITY_TIER_ORDER, quality_profile, settings_for_quality
from atlas_voice.realtime import (
    RealtimeAudio,
    generate_realtime_reply,
    is_tts_sidecar_provider,
    normalize_tts_provider,
    synthesize_with_espeak_ng,
    synthesize_with_piper,
    synthesize_with_tts_sidecar,
)
from atlas_voice.voice_profiles import (
    VOICE_PROFILE_ORDER,
    normalize_voice_profile_id,
)

DEFAULT_VOICE_STACK_BENCHMARK_TEXT = "Answer in one short sentence: is Atlas Voice responsive?"
DEFAULT_VOICE_STACK_BENCHMARK_TTS_TEXT = (
    "Atlas Voice concurrent local speech benchmark. "
    "This checks whether Qwen and local TTS can run at the same time."
)

VOICE_PROFILE_BENCHMARK_SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True)
class VoiceTTSMeasurement:
    """Client-observed TTS timings without implying that bytes were playable."""

    audio: RealtimeAudio
    requested_model: str
    served_model: str | None
    served_model_source: str
    response_headers_ms: int | None
    first_audio_byte_ms: int | None
    full_response_ms: int
    server_generation_ms: int | None
    audio_duration_seconds: float | None
    real_time_factor: float | None
    sample_rate_hz: int | None
    channels: int | None
    sample_width_bytes: int | None
    audio_integrity_verified: bool
    delivery_mode: str


def run_asr_benchmark(
    audio_path: Path,
    settings: Settings,
    *,
    providers: list[str],
    reference_text: str | None = None,
    include_diarization: bool = False,
    expected_speakers: int | None = None,
) -> list[dict[str, Any]]:
    expected_speakers = _validate_expected_speakers(expected_speakers)
    results = []
    duration = audio_duration_seconds(audio_path, default=0.0)
    for provider in providers:
        provider_settings = _settings_for_provider(settings, provider)
        started = time.perf_counter()
        error = None
        transcript: dict[str, Any] | None = None
        diarization: list[dict[str, Any]] | None = None
        try:
            transcript = transcribe_audio(audio_path, provider_settings)
            if include_diarization:
                diarization_kwargs: dict[str, Any] = {"transcript": transcript}
                if expected_speakers is not None:
                    diarization_kwargs["expected_speakers"] = expected_speakers
                diarization = diarize_audio(audio_path, provider_settings, **diarization_kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        text = _transcript_text(transcript) if transcript else ""
        detected_speakers = (
            _detected_speaker_count(diarization) if diarization is not None else None
        )
        count_error = (
            abs(detected_speakers - expected_speakers)
            if detected_speakers is not None and expected_speakers is not None
            else None
        )
        if transcript is not None:
            repetition_warning, repetition_reason = _repetition_hallucination_warning(text)
        else:
            repetition_warning, repetition_reason = None, None
        result = {
            "provider": provider,
            "model": _model_for_provider(provider_settings, provider),
            "audio_seconds": duration,
            "elapsed_seconds": round(elapsed, 3),
            "rtf": round(elapsed / duration, 4) if duration > 0 else None,
            "max_rss_mb": _max_rss_mb(),
            "segment_count": len(transcript.get("segments", [])) if transcript else 0,
            "diarization_turn_count": len(diarization or []),
            "detected_speaker_count": detected_speakers,
            "expected_speaker_count": expected_speakers,
            "speaker_count_error": count_error,
            "repetition_hallucination_warning": repetition_warning,
            "repetition_hallucination_reason": repetition_reason,
            "reference_provided": reference_text is not None,
            "text_preview": text[:500],
            "error": error,
        }
        if reference_text is not None:
            result["wer"] = (
                word_error_rate(reference_text, text) if transcript is not None else None
            )
            result["cer"] = (
                character_error_rate(reference_text, text) if transcript is not None else None
            )
        results.append(result)
    return results


def run_quality_benchmark(
    audio_path: Path,
    settings: Settings,
    *,
    reference_text: str | None = None,
    expected_speakers: int | None = None,
    tiers: str | list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Run the selected product quality tiers without choosing a winner."""

    selected_tiers = _selected_quality_tiers(tiers)
    expected_speakers = _validate_expected_speakers(expected_speakers)
    duration = audio_duration_seconds(audio_path, default=0.0)
    results: list[dict[str, Any]] = []

    with TemporaryDirectory(prefix="atlas-voice-quality-") as temporary_dir:
        output_dir = Path(temporary_dir)
        for tier in selected_tiers:
            profile = quality_profile(tier, settings)
            tier_settings = settings_for_quality(settings, tier)
            provider = tier_settings.asr_provider.strip().lower()
            normalized_path = output_dir / f"{tier}.wav"
            started = time.perf_counter()

            try:
                normalize_audio(
                    audio_path,
                    normalized_path,
                    cleanup_mode=tier_settings.audio_cleanup,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                trial = _quality_failure_result(
                    provider=provider,
                    model=_model_for_provider(tier_settings, provider),
                    audio_seconds=duration,
                    elapsed_seconds=elapsed,
                    expected_speakers=expected_speakers,
                    reference_provided=reference_text is not None,
                    error=f"normalization: {type(exc).__name__}: {exc}",
                )
            else:
                trial = run_asr_benchmark(
                    normalized_path,
                    tier_settings,
                    providers=[provider],
                    reference_text=reference_text,
                    include_diarization=True,
                    expected_speakers=expected_speakers,
                )[0]
                elapsed = time.perf_counter() - started
                trial.update(
                    {
                        "audio_seconds": duration,
                        "elapsed_seconds": round(elapsed, 3),
                        "rtf": round(elapsed / duration, 4) if duration > 0 else None,
                        "max_rss_mb": _max_rss_mb(),
                    }
                )

            results.append(
                {
                    "tier": tier,
                    "tier_name": profile.name,
                    "rank": profile.rank,
                    **trial,
                }
            )
            _release_quality_trial_models()
    return results


def _release_quality_trial_models() -> None:
    from atlas_voice.providers.faster_whisper_provider import (
        _clear_model_cache as clear_faster_whisper,
    )
    from atlas_voice.providers.nemo_provider import _clear_model_cache as clear_nemo
    from atlas_voice.providers.pyannote_provider import _clear_pipeline_cache
    from atlas_voice.providers.whisperx_provider import _clear_model_caches

    clear_faster_whisper()
    clear_nemo()
    _clear_model_caches()
    _clear_pipeline_cache()
    gc.collect()
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
        cuda.empty_cache()


def run_voice_stack_benchmark(
    settings: Settings,
    *,
    rounds: int = 3,
    text: str = DEFAULT_VOICE_STACK_BENCHMARK_TEXT,
    tts_text: str = DEFAULT_VOICE_STACK_BENCHMARK_TTS_TEXT,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    target_dir = output_dir or settings.artifacts_dir / "voice-stack-benchmark"
    results: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        started = time.perf_counter()
        llm_reply = None
        tts_audio = None
        llm_error = None
        tts_error = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            llm_future = executor.submit(generate_realtime_reply, text, settings)
            tts_future = executor.submit(
                _synthesize_voice_stack_tts, tts_text, settings, target_dir
            )
            try:
                llm_reply = llm_future.result()
            except Exception as exc:  # noqa: BLE001 - benchmark reports component failures.
                llm_error = _component_error("llm", exc)
            try:
                tts_audio = tts_future.result()
            except Exception as exc:  # noqa: BLE001 - benchmark reports component failures.
                tts_error = _component_error("tts", exc)

        errors = [error for error in (llm_error, tts_error) if error]
        provider = normalize_tts_provider(settings.tts_provider)
        results.append(
            {
                "round": round_index,
                "llm_provider": "stub" if settings.stub_mode else "openai-compatible",
                "llm_model": settings.llm_model,
                "tts_provider": provider,
                "tts_model": _tts_model_for_benchmark(settings, provider),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "llm_latency_ms": llm_reply.latency_ms if llm_reply else None,
                "tts_latency_ms": tts_audio.latency_ms if tts_audio else None,
                "tokens_in": llm_reply.tokens_in if llm_reply else None,
                "tokens_out": llm_reply.tokens_out if llm_reply else None,
                "tts_audio_bytes": len(tts_audio.payload) if tts_audio else 0,
                "tts_audio_path": str(tts_audio.path) if tts_audio else None,
                "max_rss_mb": _max_rss_mb(),
                "error": "; ".join(errors) if errors else None,
            }
        )
    return results


def run_voice_profiles_benchmark(
    settings: Settings,
    assistant_config: AssistantConfig,
    *,
    profiles: str | list[str] | tuple[str, ...] | None = None,
    rounds: int = 3,
    text: str = DEFAULT_VOICE_STACK_BENCHMARK_TEXT,
    output_dir: Path | None = None,
    inventory_path: Path | None = None,
    router_models_url: str | None = None,
) -> list[dict[str, Any]]:
    """Benchmark profile turns sequentially and report the model actually served."""

    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    selected_profiles = _selected_voice_profiles(profiles)
    target_dir = output_dir or settings.artifacts_dir / "voice-profile-benchmark"
    prepared_artifacts = _prepare_voice_artifact_provenance(inventory_path, selected_profiles)
    results: list[dict[str, Any]] = []

    for profile_id in selected_profiles:
        profile_settings = settings_for_voice_profile(settings, assistant_config, profile_id)
        requested_model = str(profile_settings.llm_model)
        tts_provider = normalize_tts_provider(profile_settings.tts_provider)
        tts_requested_model = _tts_model_for_benchmark(profile_settings, tts_provider)
        for round_index in range(1, rounds + 1):
            turn_started = time.perf_counter()
            streamed_deltas: list[str] = []
            reply = None
            tts_audio = None
            tts_measurement: VoiceTTSMeasurement | None = None
            llm_error = None
            tts_error = None
            tts_started_from_turn_ms: int | None = None

            try:
                reply = generate_realtime_reply(
                    text,
                    profile_settings,
                    on_text_delta=streamed_deltas.append,
                )
            except Exception as exc:  # noqa: BLE001 - benchmark records component failures.
                llm_error = _component_error("llm", exc)

            if reply is not None:
                tts_started = time.perf_counter()
                tts_started_from_turn_ms = max(round((tts_started - turn_started) * 1000), 0)
                try:
                    tts_measurement = _measure_voice_profile_tts(
                        reply.text,
                        profile_settings,
                        target_dir / profile_id,
                    )
                    tts_audio = tts_measurement.audio
                    if (
                        str(profile_settings.tts_response_format).strip().lower() == "wav"
                        and not tts_measurement.audio_integrity_verified
                    ):
                        tts_error = "tts: invalid or empty WAV response"
                except Exception as exc:  # noqa: BLE001 - preserve the successful LLM result.
                    tts_error = _component_error("tts", exc)

            turn_elapsed_ms = max(round((time.perf_counter() - turn_started) * 1000), 0)
            served_model = _sanitize_served_model(reply.served_model if reply else None)
            model_relation = _model_relation(requested_model, served_model)
            models_endpoint = router_models_url or _router_models_url(
                str(profile_settings.llm_base_url)
            )
            artifact_provenance = _observe_voice_artifact_provenance(
                prepared_artifacts[profile_id],
                requested_model=requested_model,
                served_model=served_model,
                models_endpoint=models_endpoint,
            )
            text_to_first_audio_byte_ms = None
            if (
                tts_measurement is not None
                and tts_started_from_turn_ms is not None
                and tts_measurement.first_audio_byte_ms is not None
            ):
                text_to_first_audio_byte_ms = (
                    tts_started_from_turn_ms + tts_measurement.first_audio_byte_ms
                )
            errors = [error for error in (llm_error, tts_error) if error]
            results.append(
                {
                    "requested_profile": profile_id,
                    "profile_name": profile_id.title(),
                    "round": round_index,
                    "llm_provider": ("stub" if profile_settings.stub_mode else "openai-compatible"),
                    "requested_model": requested_model,
                    "served_model": served_model,
                    "model_relation": model_relation,
                    "routing_verified": model_relation in {"exact", "alias"},
                    "artifact_verified": artifact_provenance["verified"],
                    "artifact_provenance": artifact_provenance,
                    "llm_ttft_ms": reply.ttft_ms if reply else None,
                    "llm_ttft_observed": bool(reply and reply.ttft_ms is not None),
                    "llm_full_latency_ms": reply.latency_ms if reply else None,
                    "llm_latency_ms": reply.latency_ms if reply else None,
                    "llm_tokens_per_second": reply.tokens_per_second if reply else None,
                    "tokens_in": reply.tokens_in if reply else None,
                    "llm_endpoint": str(profile_settings.llm_base_url),
                    "tokens_out": reply.tokens_out if reply else None,
                    "streamed_delta_count": len(streamed_deltas),
                    "tts_provider": tts_provider,
                    "tts_model": tts_requested_model,
                    "tts_requested_model": tts_requested_model,
                    "tts_served_model": tts_measurement.served_model if tts_measurement else None,
                    "tts_served_model_source": tts_measurement.served_model_source
                    if tts_measurement
                    else "unreported",
                    "tts_model_verified": bool(
                        tts_measurement and tts_measurement.served_model == tts_requested_model
                    ),
                    "tts_routing_verified": bool(
                        tts_measurement and tts_measurement.served_model == tts_requested_model
                    ),
                    "tts_endpoint": str(profile_settings.tts_base_url),
                    "tts_voice": str(profile_settings.tts_voice),
                    "tts_response_headers_ms": tts_measurement.response_headers_ms
                    if tts_measurement
                    else None,
                    "tts_response_headers_observed": bool(
                        tts_measurement and tts_measurement.response_headers_ms is not None
                    ),
                    "tts_first_audio_byte_ms": tts_measurement.first_audio_byte_ms
                    if tts_measurement
                    else None,
                    "tts_first_audio_byte_observed": bool(
                        tts_measurement and tts_measurement.first_audio_byte_ms is not None
                    ),
                    "tts_first_playable_audio_ms": None,
                    "tts_first_playable_audio_observed": False,
                    "tts_full_response_latency_ms": tts_measurement.full_response_ms
                    if tts_measurement
                    else None,
                    "tts_full_wav_latency_ms": tts_measurement.full_response_ms
                    if tts_measurement and tts_measurement.audio_integrity_verified
                    else None,
                    "tts_latency_ms": tts_measurement.full_response_ms if tts_measurement else None,
                    "tts_server_generation_ms": tts_measurement.server_generation_ms
                    if tts_measurement
                    else None,
                    "tts_delivery_mode": tts_measurement.delivery_mode
                    if tts_measurement
                    else "unobserved",
                    "tts_endpoint_buffers_full_audio": is_tts_sidecar_provider(tts_provider),
                    "tts_delivery_semantics": "client-observed response; first body byte is not first playable audio",
                    "tts_audio_bytes": len(tts_audio.payload) if tts_audio else 0,
                    "tts_audio_path": str(tts_audio.path) if tts_audio else None,
                    "tts_audio_duration_seconds": tts_measurement.audio_duration_seconds
                    if tts_measurement
                    else None,
                    "tts_real_time_factor": tts_measurement.real_time_factor
                    if tts_measurement
                    else None,
                    "tts_sample_rate_hz": tts_measurement.sample_rate_hz
                    if tts_measurement
                    else None,
                    "tts_channels": tts_measurement.channels if tts_measurement else None,
                    "tts_sample_width_bytes": tts_measurement.sample_width_bytes
                    if tts_measurement
                    else None,
                    "tts_audio_integrity_verified": bool(
                        tts_measurement and tts_measurement.audio_integrity_verified
                    ),
                    "text_prompt_to_first_audio_byte_ms": text_to_first_audio_byte_ms,
                    "text_prompt_to_first_playable_audio_ms": None,
                    "text_prompt_to_full_wav_ms": turn_elapsed_ms
                    if tts_measurement and tts_measurement.audio_integrity_verified
                    else None,
                    "end_to_end_ms": turn_elapsed_ms,
                    "measurement_scope": "text prompt to complete buffered WAV; excludes microphone, VAD, ASR, and playback",
                    "max_rss_mb": _max_rss_mb(),
                    "error": "; ".join(errors) if errors else None,
                }
            )
    return results


def voice_profiles_benchmark_payload(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": VOICE_PROFILE_BENCHMARK_SCHEMA_VERSION,
        "benchmark": "voice_profiles_text_to_buffered_wav",
        "first_audio_byte_is_first_playable": False,
        "measurement_scope": (
            "text prompt through complete buffered WAV; no capture, ASR, or playback"
        ),
        "results": results,
    }


def validate_voice_profiles_benchmark_json_output(
    output_path: Path,
    *,
    protected_paths: Iterable[Path | str | None] = (),
) -> Path:
    target, _temporary = _voice_profiles_benchmark_json_paths(
        output_path, protected_paths=protected_paths
    )
    return target


def write_voice_profiles_benchmark_json(
    output_path: Path,
    results: list[dict[str, Any]],
    *,
    protected_paths: Iterable[Path | str | None] = (),
) -> Path:
    target, temporary = _voice_profiles_benchmark_json_paths(
        output_path, protected_paths=protected_paths
    )
    encoded = (json.dumps(voice_profiles_benchmark_payload(results), indent=2) + "\n").encode(
        "utf-8"
    )
    _atomic_create_private_file(target, temporary, encoded)
    return target


def _voice_profiles_benchmark_json_paths(
    output_path: Path,
    *,
    protected_paths: Iterable[Path | str | None],
) -> tuple[Path, Path]:
    raw_target = output_path.expanduser()
    target = Path(os.path.abspath(raw_target))
    if target.suffix.casefold() != ".json":
        raise ValueError("Voice profile benchmark JSON output must use a .json suffix")
    _reject_symlink_components(target.parent)
    if not target.parent.is_dir():
        raise ValueError(
            f"Voice profile benchmark JSON parent directory does not exist: {target.parent}"
        )
    if os.path.lexists(target):
        raise ValueError(f"Voice profile benchmark JSON output already exists: {target}")

    temporary = target.with_name(f".{target.name}.tmp")
    if os.path.lexists(temporary):
        raise ValueError(f"Stale voice profile benchmark JSON temporary file exists: {temporary}")

    for protected in protected_paths:
        if protected is None or not str(protected).strip():
            continue
        protected_path = Path(os.path.abspath(Path(protected).expanduser()))
        if target == protected_path:
            raise ValueError(
                "Voice profile benchmark JSON output collides with an input, model, "
                f"or audio path: {target}"
            )
        try:
            if (
                target.exists()
                and protected_path.exists()
                and os.path.samefile(target, protected_path)
            ):
                raise ValueError(
                    "Voice profile benchmark JSON output aliases an input, model, "
                    f"or audio path: {target}"
                )
        except OSError as exc:
            raise ValueError(
                f"Could not validate voice profile benchmark output collision: {exc}"
            ) from exc
    return target, temporary


def _reject_symlink_components(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"Voice profile benchmark JSON path contains a symlink: {current}")
        if current.parent == current:
            return
        current = current.parent


def _atomic_create_private_file(target: Path, temporary: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    created_temporary = False
    try:
        file_descriptor = os.open(temporary, flags, 0o600)
        created_temporary = True
        with os.fdopen(file_descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise ValueError(
                f"Voice profile benchmark JSON output appeared during publish: {target}"
            ) from exc
        os.unlink(temporary)
        created_temporary = False
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(target.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as exc:
        raise ValueError(
            f"Voice profile benchmark JSON temporary file already exists: {temporary}"
        ) from exc
    finally:
        if created_temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def print_voice_profiles_benchmark_results(
    results: list[dict[str, Any]], *, json_output: bool = False
) -> None:
    if json_output:
        print(json.dumps(voice_profiles_benchmark_payload(results), indent=2))
        return
    for result in results:
        print(
            f"profile={result['profile_name']} round={result['round']} "
            f"requested_model={result['requested_model']} "
            f"served_model={result['served_model'] or 'unreported'} "
            f"relation={result['model_relation']}"
        )
        print(
            f"  llm_ttft={result['llm_ttft_ms']}ms "
            f"llm_total={result['llm_latency_ms']}ms "
            f"llm_tok_s={result['llm_tokens_per_second']} "
            f"tts={result['tts_latency_ms']}ms end_to_end={result['end_to_end_ms']}ms"
        )
        if not result["routing_verified"]:
            print("  routing_verified=no; do not attribute this run to the requested model")
        elif not result["artifact_verified"]:
            print("  artifact_verified=no; verify the loaded artifact hash before promotion")
        if result.get("tts_audio_path"):
            print(f"  audio={result['tts_audio_path']}")
        if result.get("error"):
            print(f"  error: {result['error']}")


def print_voice_stack_benchmark_results(
    results: list[dict[str, Any]], *, json_output: bool = False
) -> None:
    if json_output:
        print(json.dumps({"results": results}, indent=2))
        return
    for result in results:
        print(
            f"round={result['round']} llm={result['llm_model']} "
            f"tts={result['tts_provider']}:{result['tts_model']}"
        )
        print(
            f"  elapsed={result['elapsed_seconds']}s "
            f"llm={result['llm_latency_ms']}ms tts={result['tts_latency_ms']}ms "
            f"tts_bytes={result['tts_audio_bytes']} max_rss={result['max_rss_mb']}MB"
        )
        if result.get("tts_audio_path"):
            print(f"  audio={result['tts_audio_path']}")
        if result.get("error"):
            print(f"  error: {result['error']}")


def make_smoke_audio(output_path: Path) -> str:
    text = "Atlas Voice benchmark smoke test. This recording checks speech recognition."
    output_path.parent.mkdir(parents=True, exist_ok=True)
    speaker = _first_available(["espeak-ng", "espeak"])
    if speaker is None:
        raise RuntimeError("espeak-ng or espeak is required to generate smoke-test speech")
    subprocess.check_call([speaker, "-w", str(output_path), text])
    return text


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = _words(reference)
    hyp_words = _words(hypothesis)
    return _error_rate(ref_words, hyp_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref_characters = list(" ".join(_words(reference)))
    hyp_characters = list(" ".join(_words(hypothesis)))
    return _error_rate(ref_characters, hyp_characters)


def _error_rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for i, reference_item in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_item in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (0 if reference_item == hypothesis_item else 1)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(reference)


def print_benchmark_results(results: list[dict[str, Any]], *, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps({"results": results}, indent=2))
        return
    for result in results:
        print(f"provider={result['provider']} model={result['model']}")
        if result.get("error"):
            print(f"  error: {result['error']}")
            continue
        print(
            f"  elapsed={result['elapsed_seconds']}s "
            f"rtf={result['rtf']} max_rss={result['max_rss_mb']}MB "
            f"segments={result['segment_count']} turns={result['diarization_turn_count']}"
        )
        if "wer" in result:
            print(f"  wer={result['wer']:.4f}")
        if result.get("text_preview"):
            print(f"  text={result['text_preview']}")


def print_quality_benchmark_results(
    results: list[dict[str, Any]], *, json_output: bool = False
) -> None:
    if json_output:
        print(json.dumps({"results": results}, indent=2))
        return

    for result in results:
        print(
            f"tier={result['tier_name']} rank={result['rank']} "
            f"provider={result['provider']} model={result['model']}"
        )
        rtf = result.get("rtf")
        print(
            f"  elapsed={result['elapsed_seconds']}s "
            f"rtf={rtf if rtf is not None else 'n/a'} "
            f"max_rss={result['max_rss_mb']}MB"
        )
        detected = result.get("detected_speaker_count")
        expected = result.get("expected_speaker_count")
        count_error = result.get("speaker_count_error")
        print(
            "  speakers="
            f"{detected if detected is not None else 'n/a'} "
            f"expected={expected if expected is not None else 'n/a'} "
            f"count_error={count_error if count_error is not None else 'n/a'}"
        )
        if result.get("reference_provided"):
            wer = result.get("wer")
            cer = result.get("cer")
            print(
                f"  wer={wer:.4f} cer={cer:.4f}"
                if wer is not None and cer is not None
                else "  wer=n/a cer=n/a"
            )
        warning = result.get("repetition_hallucination_warning")
        warning_text = "n/a" if warning is None else ("yes" if warning else "no")
        print(f"  repetition_hallucination_warning={warning_text}")
        if result.get("repetition_hallucination_reason"):
            print(f"  repetition_reason={result['repetition_hallucination_reason']}")
        if result.get("text_preview"):
            print(f"  text={result['text_preview']}")
        if result.get("error"):
            print(f"  error: {result['error']}")


def _synthesize_voice_stack_tts(text: str, settings: Settings, output_dir: Path):
    provider = normalize_tts_provider(settings.tts_provider)
    if is_tts_sidecar_provider(provider):
        return synthesize_with_tts_sidecar(text, settings, output_dir)
    if provider == "piper":
        return synthesize_with_piper(text, settings, output_dir)
    if provider == "espeak-ng":
        return synthesize_with_espeak_ng(text, settings, output_dir)
    raise RuntimeError(f"TTS provider {provider!r} does not produce benchmark audio")


def _measure_voice_profile_tts(
    text: str,
    settings: Settings,
    output_dir: Path,
) -> VoiceTTSMeasurement:
    provider = normalize_tts_provider(settings.tts_provider)
    if is_tts_sidecar_provider(provider):
        return _measure_tts_sidecar_for_benchmark(text, settings, output_dir)

    started = time.perf_counter()
    audio = _synthesize_voice_stack_tts(text, settings, output_dir)
    full_response_ms = (
        audio.latency_ms
        if audio.latency_ms is not None
        else max(round((time.perf_counter() - started) * 1000), 0)
    )
    duration, sample_rate, channels, sample_width, integrity = _wav_audio_metadata(audio.payload)
    return VoiceTTSMeasurement(
        audio=audio,
        requested_model=_tts_model_for_benchmark(settings, provider),
        served_model=None,
        served_model_source="unreported",
        response_headers_ms=None,
        first_audio_byte_ms=None,
        full_response_ms=full_response_ms,
        server_generation_ms=None,
        audio_duration_seconds=duration,
        real_time_factor=_audio_real_time_factor(full_response_ms, duration),
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        audio_integrity_verified=integrity,
        delivery_mode="buffered-process-output",
    )


def _measure_tts_sidecar_for_benchmark(
    text: str,
    settings: Settings,
    output_dir: Path,
) -> VoiceTTSMeasurement:
    cleaned = " ".join(str(text).split())
    if not cleaned:
        raise RuntimeError("TTS sidecar benchmark requires non-empty input text")

    import httpx

    requested_model = str(settings.tts_model)
    requested_format = str(settings.tts_response_format or "wav").strip().lower()
    request_payload = {
        "model": requested_model,
        "input": cleaned,
        "voice": settings.tts_voice,
        "response_format": requested_format,
    }
    started = time.perf_counter()
    response_headers: dict[str, str] = {}
    chunks: list[bytes] = []
    first_audio_byte_ms: int | None = None

    with httpx.Client(timeout=settings.tts_timeout) as client:
        with client.stream("POST", str(settings.tts_base_url), json=request_payload) as response:
            response_headers_ms = _milliseconds_since(started)
            response.raise_for_status()
            response_headers = {
                str(key).casefold(): str(value) for key, value in response.headers.items()
            }
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                if first_audio_byte_ms is None:
                    first_audio_byte_ms = _milliseconds_since(started)
                chunks.append(chunk)
    full_response_ms = _milliseconds_since(started)
    payload = b"".join(chunks)
    if not payload:
        raise RuntimeError("TTS sidecar returned empty audio")

    media_type = response_headers.get("content-type", "application/octet-stream")
    media_type = media_type.split(";", 1)[0].strip().lower()
    extension = "wav" if media_type in {"audio/wav", "audio/x-wav"} else requested_format
    extension = re.sub(r"[^a-z0-9]", "", extension) or "bin"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"assistant-benchmark-{time.time_ns()}.{extension}"
    output_path.write_bytes(payload)

    duration, sample_rate, channels, sample_width, integrity = _wav_audio_metadata(payload)
    header_sample_rate = _positive_int_header(response_headers.get("x-audio-sample-rate"))
    return VoiceTTSMeasurement(
        audio=RealtimeAudio(
            path=output_path,
            payload=payload,
            media_type=media_type,
            latency_ms=full_response_ms,
        ),
        requested_model=requested_model,
        served_model=_nonempty_header(response_headers.get("x-atlas-model")),
        served_model_source=(
            "x-atlas-model"
            if _nonempty_header(response_headers.get("x-atlas-model"))
            else "unreported"
        ),
        response_headers_ms=response_headers_ms,
        first_audio_byte_ms=first_audio_byte_ms,
        full_response_ms=full_response_ms,
        server_generation_ms=_nonnegative_int_header(
            response_headers.get("x-generation-latency-ms")
        ),
        audio_duration_seconds=duration,
        real_time_factor=_audio_real_time_factor(full_response_ms, duration),
        sample_rate_hz=sample_rate or header_sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        audio_integrity_verified=integrity,
        delivery_mode="buffered-http-response",
    )


def _wav_audio_metadata(
    payload: bytes,
) -> tuple[float | None, int | None, int | None, int | None, bool]:
    data_size = _riff_wave_data_size(payload)
    if data_size is None:
        return None, None, None, None, False
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            frame_rate = int(wav_file.getframerate())
            frame_count = int(wav_file.getnframes())
            channels = int(wav_file.getnchannels())
            sample_width = int(wav_file.getsampwidth())
            expected_frame_bytes = frame_count * channels * sample_width
            frame_payload = wav_file.readframes(frame_count)
            trailing_frame = wav_file.readframes(1)
    except (EOFError, wave.Error):
        return None, None, None, None, False
    if frame_rate <= 0 or frame_count <= 0 or channels <= 0 or sample_width <= 0:
        return None, frame_rate or None, channels or None, sample_width or None, False
    if (
        expected_frame_bytes != data_size
        or len(frame_payload) != expected_frame_bytes
        or trailing_frame
    ):
        return None, frame_rate, channels, sample_width, False
    return round(frame_count / frame_rate, 6), frame_rate, channels, sample_width, True


def _riff_wave_data_size(payload: bytes) -> int | None:
    """Validate the complete RIFF container and return its sole data-chunk size."""

    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        return None
    if int.from_bytes(payload[4:8], "little") + 8 != len(payload):
        return None

    offset = 12
    data_sizes: list[int] = []
    while offset < len(payload):
        if len(payload) - offset < 8:
            return None
        chunk_id = payload[offset : offset + 4]
        chunk_size = int.from_bytes(payload[offset + 4 : offset + 8], "little")
        chunk_end = offset + 8 + chunk_size
        if chunk_end > len(payload):
            return None
        if chunk_id == b"data":
            data_sizes.append(chunk_size)

        # RIFF chunks are word-aligned. Python's wave writer omits the optional
        # pad byte for an odd final data chunk, so retain compatibility with
        # that common output while still requiring padding before another chunk.
        padded_end = chunk_end + (chunk_size & 1)
        offset = chunk_end if chunk_end == len(payload) else padded_end
        if offset > len(payload):
            return None

    return data_sizes[0] if len(data_sizes) == 1 else None


def _audio_real_time_factor(full_response_ms: int, duration_seconds: float | None) -> float | None:
    if duration_seconds is None or duration_seconds <= 0:
        return None
    return round(full_response_ms / (duration_seconds * 1000), 4)


def _milliseconds_since(started: float) -> int:
    return max(round((time.perf_counter() - started) * 1000), 0)


def _nonempty_header(value: str | None) -> str | None:
    clean = str(value or "").strip()
    return clean or None


def _nonnegative_int_header(value: str | None) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _positive_int_header(value: str | None) -> int | None:
    parsed = _nonnegative_int_header(value)
    return parsed if parsed is not None and parsed > 0 else None


def _tts_model_for_benchmark(settings: Settings, provider: str) -> str:
    if is_tts_sidecar_provider(provider):
        return settings.tts_model
    if provider == "piper":
        return settings.piper_voice or settings.piper_executable
    if provider == "espeak-ng":
        return settings.tts_voice if settings.tts_voice != "default" else "espeak-ng"
    return provider


def _component_error(component: str, exc: Exception) -> str:
    return f"{component}: {type(exc).__name__}: {exc}"


def _prepare_voice_artifact_provenance(
    inventory_path: Path | None,
    selected_profiles: list[str],
) -> dict[str, dict[str, Any]]:
    if inventory_path is None:
        return {
            profile: {
                "profile": profile,
                "inventory_provided": False,
                "file_verified": False,
                "route_path_verified": False,
                "runtime_loaded": False,
                "verified": False,
                "status": "inventory_not_provided",
            }
            for profile in selected_profiles
        }

    hash_cache: dict[Path, str] = {}
    source_path = inventory_path.expanduser().resolve()
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read voice model inventory {source_path}: {exc}") from exc
    try:
        payload = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Voice model inventory is not valid JSON: {source_path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Voice model inventory must contain a models list")

    by_profile: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(payload["models"]):
        if not isinstance(raw, dict):
            raise ValueError(f"Voice model inventory models[{index}] must be an object")
        profile = str(raw.get("profile") or "").strip().lower()
        if profile in by_profile:
            raise ValueError(f"Duplicate profile in voice model inventory: {profile}")
        if profile:
            by_profile[profile] = raw

    inventory_sha256 = hashlib.sha256(source_bytes).hexdigest()
    prepared: dict[str, dict[str, Any]] = {}
    for profile in selected_profiles:
        raw = by_profile.get(profile)
        if raw is None:
            raise ValueError(f"Voice model inventory has no {profile!r} entry")
        model_id = str(raw.get("model_id") or "").strip()
        expected_sha256 = str(raw.get("sha256") or "").strip().lower()
        relative_path = str(raw.get("path") or "").strip()
        try:
            expected_size = int(raw.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid size_bytes for inventory profile {profile}") from exc
        if not model_id or not relative_path:
            raise ValueError(f"Inventory profile {profile} requires model_id and path")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"Invalid SHA-256 for inventory profile {profile}")
        if expected_size <= 0:
            raise ValueError(f"Invalid size_bytes for inventory profile {profile}")

        configured_path = Path(relative_path).expanduser()
        if not configured_path.is_absolute():
            configured_path = source_path.parent / configured_path
        artifact_path = configured_path.resolve()
        artifact_exists = artifact_path.is_file()
        observed_size = artifact_path.stat().st_size if artifact_exists else None
        observed_sha256 = hash_cache.get(artifact_path)
        if artifact_exists and observed_sha256 is None:
            observed_sha256 = _sha256_path(artifact_path)
            hash_cache[artifact_path] = observed_sha256
        size_verified = observed_size == expected_size
        sha256_verified = observed_sha256 == expected_sha256
        file_verified = bool(artifact_exists and size_verified and sha256_verified)
        prepared[profile] = {
            "profile": profile,
            "inventory_provided": True,
            "inventory_path": str(source_path),
            "inventory_sha256": inventory_sha256,
            "inventory_version": str(payload.get("inventory_version") or ""),
            "model_id": model_id,
            "artifact_path": str(artifact_path),
            "expected_size_bytes": expected_size,
            "observed_size_bytes": observed_size,
            "size_verified": size_verified,
            "expected_sha256": expected_sha256,
            "observed_sha256": observed_sha256,
            "sha256_verified": sha256_verified,
            "file_verified": file_verified,
            "source_url": str(raw.get("source_url") or ""),
            "source_revision": str(raw.get("source_revision") or ""),
            "license": str(raw.get("license") or ""),
            "route_path_verified": False,
            "runtime_loaded": False,
            "verified": False,
            "status": "file_verified" if file_verified else "file_verification_failed",
        }
    return prepared


def _observe_voice_artifact_provenance(
    prepared: dict[str, Any],
    *,
    requested_model: str,
    served_model: str | None,
    models_endpoint: str,
) -> dict[str, Any]:
    observation = dict(prepared)
    observation.update(
        {
            "requested_model": requested_model,
            "served_model": served_model,
            "router_models_endpoint": models_endpoint,
            "route_path_verified": False,
            "runtime_loaded": False,
            "verified": False,
        }
    )
    if not observation.get("inventory_provided"):
        return observation
    if requested_model != observation.get("model_id"):
        observation["status"] = "requested_model_inventory_mismatch"
        return observation
    if served_model != requested_model:
        observation["status"] = "served_model_mismatch"
        return observation

    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(models_endpoint)
            response.raise_for_status()
            payload = response.json()
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ValueError("router /models response has no data list")
        matches = [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("id") == requested_model
        ]
        if len(matches) != 1:
            raise ValueError(
                f"router /models returned {len(matches)} entries for {requested_model!r}"
            )
        entry = matches[0]
        status = entry.get("status")
        if not isinstance(status, dict):
            raise ValueError("router model entry has no status object")
        args = status.get("args")
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ValueError("router status.args is not a string list")
        model_flags = [index for index, item in enumerate(args) if item == "--model"]
        alternate_flags = [item for item in args if item == "-m" or item.startswith("--model=")]
        if len(model_flags) != 1 or alternate_flags:
            raise ValueError("router status.args must contain one separate --model flag")
        model_index = model_flags[0] + 1
        if model_index >= len(args) or not args[model_index]:
            raise ValueError("router --model flag has no path")
        args_path = Path(args[model_index]).expanduser()
        if not args_path.is_absolute():
            raise ValueError("router --model path is not absolute")

        path_field = entry.get("path")
        if path_field is not None:
            if not isinstance(path_field, str) or not path_field:
                raise ValueError("router path field is malformed")
            reported_path = Path(path_field).expanduser()
            if not reported_path.is_absolute():
                raise ValueError("router path field is not absolute")
            if not _same_artifact_path(reported_path, args_path):
                raise ValueError("router path and status.args disagree")
            path_source = "path+status.args"
        else:
            reported_path = args_path
            path_source = "status.args"

        expected_path = Path(str(observation["artifact_path"]))
        route_path_verified = _same_artifact_path(reported_path, expected_path)
        status_value = str(status.get("value") or "").strip().lower()
        runtime_loaded = status_value == "loaded"
        observation.update(
            {
                "router_reported_path": str(reported_path.resolve()),
                "router_path_source": path_source,
                "router_status": status_value or None,
                "route_path_verified": route_path_verified,
                "runtime_loaded": runtime_loaded,
            }
        )
        observation["verified"] = bool(
            observation.get("file_verified") and route_path_verified and runtime_loaded
        )
        if observation["verified"]:
            observation["status"] = "verified"
        elif not route_path_verified:
            observation["status"] = "router_path_mismatch"
        elif not runtime_loaded:
            observation["status"] = "router_model_not_loaded"
        else:
            observation["status"] = "file_verification_failed"
    except Exception as exc:  # noqa: BLE001 - preserve benchmark output on audit failure.
        observation["status"] = "router_verification_failed"
        observation["router_error"] = f"{type(exc).__name__}: {exc}"[:500]
    return observation


def _router_models_url(llm_endpoint: str) -> str:
    parsed = urlparse(llm_endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"LLM endpoint is not an absolute HTTP URL: {llm_endpoint!r}")
    return urlunparse((parsed.scheme, parsed.netloc, "/models", "", "", ""))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_artifact_path(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return left.resolve() == right.resolve()


def _selected_voice_profiles(
    profiles: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    if profiles is None:
        requested = list(VOICE_PROFILE_ORDER)
    elif isinstance(profiles, str):
        requested = [item.strip() for item in profiles.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in profiles if str(item).strip()]
    if not requested:
        raise ValueError("At least one voice profile is required.")
    normalized = [normalize_voice_profile_id(item) for item in requested]
    selected = set(normalized)
    return [profile_id for profile_id in VOICE_PROFILE_ORDER if profile_id in selected]


def _sanitize_served_model(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    clean = value.strip()
    parsed = urlparse(clean)
    if parsed.scheme and parsed.netloc:
        clean = Path(parsed.path).name or parsed.netloc
    else:
        clean = clean.replace("\\", "/").rsplit("/", 1)[-1]
    clean = clean.split("?", 1)[0].split("#", 1)[0]
    clean = re.sub(r"[^A-Za-z0-9._:+@ -]", "_", clean).strip()
    return clean[:120] or None


def _model_relation(requested_model: str, served_model: str | None) -> str:
    if not served_model:
        return "unreported"
    requested = _sanitize_served_model(requested_model)
    if not requested:
        return "unreported"
    if requested.casefold() == served_model.casefold():
        return "exact"
    if _canonical_model_alias(requested) == _canonical_model_alias(served_model):
        return "alias"
    return "mismatch"


def _canonical_model_alias(value: str) -> str:
    lowered = value.casefold()
    lowered = re.sub(r"\.(?:gguf|bin|safetensors)$", "", lowered)
    tokens = [token for token in re.split(r"[^a-z0-9.]+", lowered) if token]
    removable = {
        "model",
        "gguf",
        "instruct",
        "chat",
        "fp16",
        "f16",
        "bf16",
        "int4",
        "int8",
    }
    while tokens and (
        tokens[-1] in removable or re.fullmatch(r"i?q\d+(?:k|km|ks|m|s|l|xl|_[kms])+", tokens[-1])
    ):
        tokens.pop()
    return "-".join(tokens)


def _selected_quality_tiers(
    tiers: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    if tiers is None:
        requested = list(QUALITY_TIER_ORDER)
    elif isinstance(tiers, str):
        requested = [item.strip().lower() for item in tiers.split(",") if item.strip()]
    else:
        requested = [str(item).strip().lower() for item in tiers if str(item).strip()]

    if not requested:
        raise ValueError("At least one quality tier is required.")
    unknown = sorted(set(requested) - set(QUALITY_TIER_ORDER))
    if unknown:
        raise ValueError(
            "Unknown quality tier(s): " + ", ".join(unknown) + ". Use light, torch, or fire."
        )
    selected = set(requested)
    return [tier for tier in QUALITY_TIER_ORDER if tier in selected]


def _validate_expected_speakers(expected_speakers: int | None) -> int | None:
    if expected_speakers is None:
        return None
    if (
        isinstance(expected_speakers, bool)
        or not isinstance(expected_speakers, int)
        or expected_speakers < 1
    ):
        raise ValueError("expected_speakers must be a positive integer")
    return expected_speakers


def _detected_speaker_count(diarization: list[dict[str, Any]]) -> int:
    speakers = {
        str(turn.get("speaker")).strip()
        for turn in diarization
        if str(turn.get("speaker") or "").strip()
    }
    return len(speakers)


def _repetition_hallucination_warning(text: str) -> tuple[bool, str | None]:
    """Flag common transcript loops; this is not proof of hallucination."""

    words = _words(text)
    if not words:
        return False, None

    run_length = 1
    for index in range(1, len(words)):
        if words[index] == words[index - 1]:
            run_length += 1
            if run_length >= 5:
                return True, f"single word repeated {run_length} times consecutively"
        else:
            run_length = 1

    maximum_phrase = min(12, len(words) // 2)
    for phrase_length in range(maximum_phrase, 1, -1):
        minimum_repeats = 2 if phrase_length >= 3 else 3
        maximum_start = len(words) - (phrase_length * minimum_repeats)
        for start in range(maximum_start + 1):
            phrase = words[start : start + phrase_length]
            repeats = 1
            cursor = start + phrase_length
            while words[cursor : cursor + phrase_length] == phrase:
                repeats += 1
                cursor += phrase_length
            if repeats >= minimum_repeats:
                return (
                    True,
                    f"{phrase_length}-word phrase repeated {repeats} times consecutively",
                )
    return False, None


def _quality_failure_result(
    *,
    provider: str,
    model: str,
    audio_seconds: float,
    elapsed_seconds: float,
    expected_speakers: int | None,
    reference_provided: bool,
    error: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "audio_seconds": audio_seconds,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "rtf": (round(elapsed_seconds / audio_seconds, 4) if audio_seconds > 0 else None),
        "max_rss_mb": _max_rss_mb(),
        "segment_count": 0,
        "diarization_turn_count": 0,
        "detected_speaker_count": None,
        "expected_speaker_count": expected_speakers,
        "speaker_count_error": None,
        "repetition_hallucination_warning": None,
        "repetition_hallucination_reason": None,
        "reference_provided": reference_provided,
        "text_preview": "",
        "error": error,
    }
    if reference_provided:
        result["wer"] = None
        result["cer"] = None
    return result


def _settings_for_provider(settings: Settings, provider: str) -> Settings:
    provider = provider.strip().lower()
    diarization_provider = settings.diarization_provider
    if provider == "vibevoice" and diarization_provider == "pyannote":
        diarization_provider = "transcript"
    return replace(settings, asr_provider=provider, diarization_provider=diarization_provider)


def _model_for_provider(settings: Settings, provider: str) -> str:
    if settings.asr_model:
        return settings.asr_model
    if provider == "whisperx":
        return settings.whisperx_model
    if provider == "faster-whisper":
        return settings.faster_whisper_model
    if provider == "parakeet":
        return "nvidia/parakeet-tdt-0.6b-v3"
    if provider == "canary":
        return "nvidia/canary-1b-v2"
    if provider == "vibevoice":
        return settings.vibevoice_model
    return "unknown"


def _transcript_text(transcript: dict[str, Any] | None) -> str:
    if not transcript:
        return ""
    segment_text = " ".join(
        str(segment.get("text") or "").strip()
        for segment in transcript.get("segments") or []
        if str(segment.get("text") or "").strip()
    ).strip()
    if segment_text:
        return segment_text
    return str(transcript.get("text") or "").strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _max_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return round(usage / (1024 * 1024), 1)
    return round(usage / 1024, 1)


def _first_available(commands: list[str]) -> str | None:
    for command in commands:
        if shutil.which(command):
            return command
    return None
