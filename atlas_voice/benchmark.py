from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

from atlas_voice.audio import normalize_audio
from atlas_voice.assistant_config import AssistantConfig
from atlas_voice.config import Settings
from atlas_voice.providers.asr import transcribe_audio
from atlas_voice.profile_settings import settings_for_voice_profile
from atlas_voice.providers.diarization import diarize_audio
from atlas_voice.providers.transcript_utils import audio_duration_seconds
from atlas_voice.quality import QUALITY_TIER_ORDER, quality_profile, settings_for_quality
from atlas_voice.realtime import (
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
) -> list[dict[str, Any]]:
    """Benchmark profile turns sequentially and report the model actually served."""

    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    selected_profiles = _selected_voice_profiles(profiles)
    target_dir = output_dir or settings.artifacts_dir / "voice-profile-benchmark"
    results: list[dict[str, Any]] = []

    for profile_id in selected_profiles:
        profile_settings = settings_for_voice_profile(settings, assistant_config, profile_id)
        requested_model = str(profile_settings.llm_model)
        tts_provider = normalize_tts_provider(profile_settings.tts_provider)
        for round_index in range(1, rounds + 1):
            turn_started = time.perf_counter()
            streamed_deltas: list[str] = []
            reply = None
            tts_audio = None
            llm_error = None
            tts_error = None
            measured_tts_latency_ms: int | None = None

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
                try:
                    tts_audio = _synthesize_voice_stack_tts(
                        reply.text,
                        profile_settings,
                        target_dir / profile_id,
                    )
                    measured_tts_latency_ms = (
                        tts_audio.latency_ms
                        if tts_audio.latency_ms is not None
                        else max(round((time.perf_counter() - tts_started) * 1000), 0)
                    )
                except Exception as exc:  # noqa: BLE001 - preserve the successful LLM result.
                    tts_error = _component_error("tts", exc)

            served_model = _sanitize_served_model(reply.served_model if reply else None)
            model_relation = _model_relation(requested_model, served_model)
            errors = [error for error in (llm_error, tts_error) if error]
            results.append(
                {
                    "requested_profile": profile_id,
                    "profile_name": profile_id.title(),
                    "round": round_index,
                    "llm_provider": (
                        "stub" if profile_settings.stub_mode else "openai-compatible"
                    ),
                    "requested_model": requested_model,
                    "served_model": served_model,
                    "model_relation": model_relation,
                    "routing_verified": model_relation in {"exact", "alias"},
                    "artifact_verified": False,
                    "llm_ttft_ms": reply.ttft_ms if reply else None,
                    "llm_latency_ms": reply.latency_ms if reply else None,
                    "llm_tokens_per_second": reply.tokens_per_second if reply else None,
                    "tokens_in": reply.tokens_in if reply else None,
                    "tokens_out": reply.tokens_out if reply else None,
                    "streamed_delta_count": len(streamed_deltas),
                    "tts_provider": tts_provider,
                    "tts_model": _tts_model_for_benchmark(profile_settings, tts_provider),
                    "tts_latency_ms": measured_tts_latency_ms,
                    "tts_audio_bytes": len(tts_audio.payload) if tts_audio else 0,
                    "tts_audio_path": str(tts_audio.path) if tts_audio else None,
                    "end_to_end_ms": max(round((time.perf_counter() - turn_started) * 1000), 0),
                    "max_rss_mb": _max_rss_mb(),
                    "error": "; ".join(errors) if errors else None,
                }
            )
    return results


def print_voice_profiles_benchmark_results(
    results: list[dict[str, Any]], *, json_output: bool = False
) -> None:
    if json_output:
        print(json.dumps({"results": results}, indent=2))
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
        tokens[-1] in removable
        or re.fullmatch(r"i?q\d+(?:k|km|ks|m|s|l|xl|_[kms])+", tokens[-1])
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
