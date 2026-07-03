from __future__ import annotations

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
