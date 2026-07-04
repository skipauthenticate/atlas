from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import resource
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from atlas_voice.config import Settings
from atlas_voice.providers.asr import transcribe_audio
from atlas_voice.providers.diarization import diarize_audio
from atlas_voice.providers.transcript_utils import audio_duration_seconds
from atlas_voice.realtime import (
    generate_realtime_reply,
    is_tts_sidecar_provider,
    normalize_tts_provider,
    synthesize_with_piper,
    synthesize_with_tts_sidecar,
)

DEFAULT_VOICE_STACK_BENCHMARK_TEXT = "Answer in one short sentence: is Atlas Voice responsive?"
DEFAULT_VOICE_STACK_BENCHMARK_TTS_TEXT = (
    "Atlas Voice concurrent local speech benchmark. "
    "This checks whether Qwen and local TTS can run at the same time."
)


def run_asr_benchmark(
    audio_path: Path,
    settings: Settings,
    *,
    providers: list[str],
    reference_text: str | None = None,
    include_diarization: bool = False,
) -> list[dict[str, Any]]:
    results = []
    duration = audio_duration_seconds(audio_path, default=0.0)
    for provider in providers:
        provider_settings = _settings_for_provider(settings, provider)
        started = time.perf_counter()
        error = None
        transcript: dict[str, Any] | None = None
        diarization = None
        try:
            transcript = transcribe_audio(audio_path, provider_settings)
            if include_diarization:
                diarization = diarize_audio(
                    audio_path, provider_settings, transcript=transcript
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        text = _transcript_text(transcript) if transcript else ""
        result = {
            "provider": provider,
            "model": _model_for_provider(provider_settings, provider),
            "audio_seconds": duration,
            "elapsed_seconds": round(elapsed, 3),
            "rtf": round(elapsed / duration, 4) if duration > 0 else None,
            "max_rss_mb": _max_rss_mb(),
            "segment_count": len(transcript.get("segments", [])) if transcript else 0,
            "diarization_turn_count": len(diarization or []),
            "text_preview": text[:500],
            "error": error,
        }
        if reference_text is not None and text:
            result["wer"] = word_error_rate(reference_text, text)
        results.append(result)
    return results


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
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    previous = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp_words, start=1):
            substitution = previous[j - 1] + (0 if ref_word == hyp_word else 1)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(ref_words)


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


def _synthesize_voice_stack_tts(text: str, settings: Settings, output_dir: Path):
    provider = normalize_tts_provider(settings.tts_provider)
    if is_tts_sidecar_provider(provider):
        return synthesize_with_tts_sidecar(text, settings, output_dir)
    if provider == "piper":
        return synthesize_with_piper(text, settings, output_dir)
    raise RuntimeError(f"TTS provider {provider!r} does not produce benchmark audio")


def _tts_model_for_benchmark(settings: Settings, provider: str) -> str:
    if is_tts_sidecar_provider(provider):
        return settings.tts_model
    if provider == "piper":
        return settings.piper_voice or settings.piper_executable
    return provider


def _component_error(component: str, exc: Exception) -> str:
    return f"{component}: {type(exc).__name__}: {exc}"


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
    return " ".join(
        str(segment.get("text") or "").strip()
        for segment in transcript.get("segments") or []
        if str(segment.get("text") or "").strip()
    ).strip()


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
