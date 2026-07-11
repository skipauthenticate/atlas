from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_NAME = "atlas_voice.model_benchmark_report"
GENERATOR_VERSION = "2.0.0"
GENERATOR_SOURCE = Path(__file__).resolve()
DEFAULT_REPORT_SHELL = (
    Path(__file__).resolve().parent / "report_assets" / "voice_model_report_shell.html"
)
DEFAULT_PROTOCOL_DOCUMENT = REPOSITORY_ROOT / "docs" / "VOICE_MODEL_PROFILES_RESEARCH_2026.md"
REPORT_RUNTIME_MARKER = "<!-- ATLAS_VOICE_MODEL_REPORT_RUNTIME -->"
PROFILE_ORDER = ("light", "torch", "fire")
PROFILE_LABELS = {profile: profile.title() for profile in PROFILE_ORDER}
INITIAL_GATES = {
    "light": {"ttft_p95_ms_max": 800.0, "decode_tps_min": 20.0, "recall_min": 0.80},
    "torch": {"ttft_p95_ms_max": 1200.0, "decode_tps_min": 14.0, "recall_min": 0.88},
    "fire": {"ttft_p95_ms_max": 2500.0, "decode_tps_min": 10.0, "recall_min": 0.92},
}
VOICE_PATH_TARGETS_MS = {"light": 1500.0, "torch": 2000.0, "fire": 3000.0}
LLAMA_BENCH_SETTING_FIELDS = (
    "n_batch",
    "n_ubatch",
    "n_threads",
    "type_k",
    "type_v",
    "n_gpu_layers",
    "n_cpu_moe",
    "no_kv_offload",
    "flash_attn",
    "use_mmap",
)
PROTOCOL_SOURCE = {
    "source": "Atlas Voice engineering protocol",
    "kind": "Document",
    "name": "VOICE_MODEL_PROFILES_RESEARCH_2026.md",
}


class BenchmarkReportError(RuntimeError):
    """Raised when source artifacts cannot support an auditable report."""


@dataclass(frozen=True)
class BenchmarkReportInputs:
    smoke_runner: Path
    quality_runner: Path
    postprocessed_quality: Path
    llama_bench_light: Path
    llama_bench_torch: Path
    llama_bench_fire: Path
    protocol: Path = DEFAULT_PROTOCOL_DOCUMENT
    tts: Path | None = None
    tegrastats: Path | None = None

    def source_paths(self) -> dict[str, Path]:
        paths = {
            "smoke_runner": self.smoke_runner,
            "quality_runner": self.quality_runner,
            "postprocessed_quality": self.postprocessed_quality,
            "llama_bench_light": self.llama_bench_light,
            "llama_bench_torch": self.llama_bench_torch,
            "llama_bench_fire": self.llama_bench_fire,
            "protocol": self.protocol,
        }
        if self.tts is not None:
            paths["tts"] = self.tts
        if self.tegrastats is not None:
            paths["tegrastats"] = self.tegrastats
        return paths


SOURCE_LABELS = {
    "smoke_runner": {
        "source": "Atlas Voice benchmark runner",
        "kind": "File",
        "name": "runner smoke JSON",
    },
    "quality_runner": {
        "source": "Atlas Voice benchmark runner",
        "kind": "File",
        "name": "runner quality JSON",
    },
    "postprocessed_quality": {
        "source": "Atlas Voice deterministic quality scorer",
        "kind": "File",
        "name": "postprocessed quality JSON",
    },
    "llama_bench_light": {
        "source": "llama.cpp llama-bench",
        "kind": "File",
        "name": "Light llama-bench JSON",
    },
    "llama_bench_torch": {
        "source": "llama.cpp llama-bench",
        "kind": "File",
        "name": "Torch llama-bench JSON",
    },
    "llama_bench_fire": {
        "source": "llama.cpp llama-bench",
        "kind": "File",
        "name": "Fire llama-bench JSON",
    },
    "tts": {
        "source": "Atlas Voice voice-profile benchmark",
        "kind": "File",
        "name": "optional TTS JSON",
    },
    "tegrastats": {
        "source": "NVIDIA tegrastats",
        "kind": "File",
        "name": "optional tegrastats log",
    },
    "protocol": PROTOCOL_SOURCE,
}


def build_analysis_from_paths(inputs: BenchmarkReportInputs) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    source_manifest: dict[str, dict[str, Any]] = {}
    for role, raw_path in inputs.source_paths().items():
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise BenchmarkReportError(f"missing input for {role}: {path}")
        raw = path.read_bytes()
        source_manifest[role] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "label": SOURCE_LABELS[role],
        }
        if role in {"tegrastats", "protocol"}:
            try:
                payloads[role] = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise BenchmarkReportError(f"{role} input is not UTF-8: {path}") from exc
        else:
            try:
                payloads[role] = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BenchmarkReportError(f"invalid JSON for {role}: {path}: {exc}") from exc

    smoke = _object(payloads["smoke_runner"], "smoke runner")
    quality = _object(payloads["quality_runner"], "quality runner")
    postprocessed = _object(payloads["postprocessed_quality"], "postprocessed quality")
    generator = _generator_identity()

    checks: list[dict[str, Any]] = []

    def check(key: str, label: str, passed: bool, detail: str, roles: Sequence[str]) -> None:
        checks.append(
            {
                "key": key,
                "label": label,
                "passed": bool(passed),
                "detail": detail,
                "source_roles": list(roles),
            }
        )

    smoke_models = _runner_model_map(smoke, "smoke runner")
    quality_models = _runner_model_map(quality, "quality runner")
    post_models = _postprocessed_model_map(postprocessed)

    check(
        "runner_status",
        "Both runner artifacts completed without recorded errors",
        smoke.get("status") == "completed"
        and quality.get("status") == "completed"
        and _integer(smoke.get("error_count"), default=0) == 0
        and _integer(quality.get("error_count"), default=0) == 0,
        "Runner completion status and aggregate error counters were checked.",
        ("smoke_runner", "quality_runner"),
    )
    check(
        "postprocess_status",
        "Deterministic quality postprocessing completed",
        postprocessed.get("status") == "completed",
        "The scorer artifact reports a completed status.",
        ("postprocessed_quality",),
    )
    check(
        "profile_matrix",
        "All artifacts describe one exact, complete Light, Torch, and Fire model matrix",
        _has_exact_profile_matrix(smoke_models)
        and _has_exact_profile_matrix(quality_models)
        and _has_exact_profile_matrix(post_models)
        and all(
            smoke_models[profile]["model_id"]
            == quality_models[profile]["model_id"]
            == post_models[profile]["model_id"]
            for profile in PROFILE_ORDER
            if profile in smoke_models and profile in quality_models and profile in post_models
        )
        and _postprocessed_trials_match_matrix(postprocessed, post_models),
        "Unique profile and model identifiers were reconciled across inventories, summaries, and scorer trials.",
        ("smoke_runner", "quality_runner", "postprocessed_quality"),
    )

    post_input = postprocessed.get("inputs")
    benchmark_meta = post_input.get("benchmark") if isinstance(post_input, dict) else None
    post_benchmark_hash = benchmark_meta.get("sha256") if isinstance(benchmark_meta, dict) else None
    quality_hash = source_manifest["quality_runner"]["sha256"]
    check(
        "quality_hash_chain",
        "The scorer is chained to the supplied quality runner by SHA-256",
        isinstance(post_benchmark_hash, str)
        and post_benchmark_hash.casefold() == quality_hash.casefold(),
        "The scorer input hash was compared with the supplied quality-runner bytes.",
        ("quality_runner", "postprocessed_quality"),
    )
    quality_corpus = quality.get("corpus")
    post_corpus = post_input.get("corpus") if isinstance(post_input, dict) else None
    check(
        "corpus_hash_chain",
        "The quality corpus identity is consistent through scoring",
        isinstance(quality_corpus, dict)
        and isinstance(post_corpus, dict)
        and quality_corpus.get("sha256") == post_corpus.get("sha256")
        and quality_corpus.get("corpus_version") == post_corpus.get("dataset_version"),
        "Corpus version and SHA-256 were reconciled between runner and scorer metadata.",
        ("quality_runner", "postprocessed_quality"),
    )
    validation = postprocessed.get("validation")
    validation = validation if isinstance(validation, dict) else {}
    actual_trial_count = _integer(validation.get("actual_trial_count"))
    expected_trial_count = _integer(validation.get("expected_trial_count"))
    quality_protocol = quality.get("protocol")
    quality_protocol = quality_protocol if isinstance(quality_protocol, dict) else {}
    quality_rounds = _integer(quality_protocol.get("rounds"))
    quality_case_count = _integer(
        quality_corpus.get("case_count") if isinstance(quality_corpus, dict) else None
    )
    calculated_trial_count = (
        quality_rounds * len(quality_models) * quality_case_count
        if quality_rounds is not None and quality_case_count is not None
        else None
    )
    check(
        "quality_case_matrix",
        "The reviewed quality matrix contains the expected conversation checks",
        quality_case_count == 60
        and actual_trial_count is not None
        and actual_trial_count == expected_trial_count == calculated_trial_count,
        "Corpus cases, rounds, model count, and expected-versus-actual trials were checked.",
        ("quality_runner", "postprocessed_quality"),
    )

    for runner_role, runner in (("smoke_runner", smoke), ("quality_runner", quality)):
        runner_models = smoke_models if runner_role == "smoke_runner" else quality_models
        check(
            f"{runner_role}_artifact_verification",
            f"{runner_role.replace('_', ' ').title()} verified every pinned artifact",
            _verification_matrix_pass(runner.get("artifact_verification"), runner_models),
            "Exactly one true artifact-verification row must exist for every expected profile/model pair.",
            (runner_role,),
        )
        check(
            f"{runner_role}_route_verification",
            f"{runner_role.replace('_', ' ').title()} verified router paths and served models",
            _verification_matrix_pass(runner.get("route_verification"), runner_models)
            and _all_trial_routes_pass(runner.get("trials"), runner_models),
            "The exact route-verification matrix and every streamed served-model row must pass.",
            (runner_role,),
        )

    llama_by_profile: dict[str, dict[str, Any]] = {}
    for profile in PROFILE_ORDER:
        role = f"llama_bench_{profile}"
        rows = _llama_bench_rows(payloads[role], role)
        identity_verified = _llama_bench_identity_verified(
            payloads[role], rows, quality_models.get(profile, {})
        )
        check(
            f"{role}_identity",
            f"{PROFILE_LABELS[profile]} llama-bench path and SHA chain match inventory",
            identity_verified,
            "Wrapper path and SHA-256 plus every row filename were reconciled with inventory.",
            (role, "quality_runner"),
        )
        llama_by_profile[profile] = _summarize_llama_bench(rows)

    commits = {
        str(value)
        for summary in llama_by_profile.values()
        for value in summary.get("build_commits", [])
        if value
    }
    check(
        "llama_build_consistency",
        "All isolated performance rows use one llama.cpp build",
        len(commits) == 1,
        "The build commit reported by each llama-bench artifact was reconciled.",
        tuple(f"llama_bench_{profile}" for profile in PROFILE_ORDER),
    )
    settings = [llama_by_profile[profile].get("settings") for profile in PROFILE_ORDER]
    settings_complete = all(isinstance(item, dict) and item for item in settings)
    settings_consistent = (
        settings_complete and len({json.dumps(item, sort_keys=True) for item in settings}) == 1
    )
    check(
        "llama_settings_consistency",
        "All isolated performance rows use comparable runtime settings",
        settings_consistent,
        "Batching, threads, KV types, GPU layers, flash attention, mmap, and offload settings were reconciled.",
        tuple(f"llama_bench_{profile}" for profile in PROFILE_ORDER),
    )

    tegrastats = (
        _summarize_tegrastats(str(payloads["tegrastats"])) if "tegrastats" in payloads else None
    )
    if tegrastats is not None:
        check(
            "tegrastats_parse",
            "The optional tegrastats log contains parseable device samples",
            tegrastats["sample_count"] > 0,
            "RAM, swap, temperature, and timestamp fields were parsed when present.",
            ("tegrastats",),
        )

    tts_by_profile = (
        _summarize_tts(payloads["tts"], quality_models)
        if "tts" in payloads
        else {profile: None for profile in PROFILE_ORDER}
    )
    if "tts" in payloads:
        check(
            "tts_identity_routes_and_audio",
            "Optional voice-path rows verify artifacts, exact model IDs, routes, and audio",
            all(
                summary is not None
                and summary["error_count"] == 0
                and summary["identity_failure_count"] == 0
                and summary["route_failure_count"] == 0
                and summary["empty_audio_count"] == 0
                and summary["artifact_verification_reported"]
                for summary in tts_by_profile.values()
            ),
            "Artifact flags, requested/served IDs, routing, errors, and non-empty audio were checked.",
            ("tts", "quality_runner"),
        )

    smoke_summaries = _summary_map(smoke, smoke_models, "smoke runner")
    quality_summaries = _summary_map(quality, quality_models, "quality runner")
    quality_post_summaries = _postprocessed_model_map(postprocessed)
    swap = _summarize_swap_samples(quality.get("resource_samples"), quality_models)
    if not swap["whole_run"]["verified"] and tegrastats is not None:
        swap["whole_run"] = _whole_run_swap_from_tegrastats(tegrastats)

    models: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        runner_model = quality_models.get(profile, {})
        runner_summary = quality_summaries.get(profile, {})
        smoke_summary = smoke_summaries.get(profile, {})
        post_summary = quality_post_summaries.get(profile, {})
        overall = post_summary.get("overall") if isinstance(post_summary, dict) else None
        overall = overall if isinstance(overall, dict) else {}
        by_category = post_summary.get("by_category") if isinstance(post_summary, dict) else None
        category_map = {
            str(item.get("category")): item for item in by_category or [] if isinstance(item, dict)
        }
        multi_window = category_map.get("multi_window_synthesis", {})
        quality_metrics = {
            key: _metric_rate(overall, key)
            for key in (
                "forbidden_claim_hit_rate",
                "forbidden_claim_trial_rate",
                "exact_pass_rate",
                "abstention_accuracy",
                "unwarranted_abstention_rate",
                "attribution_target_accuracy",
                "timestamp_target_accuracy",
                "temporal_order_accuracy",
                "manual_review_rate",
            )
        }
        quality_metrics["overall_required_fact_recall"] = _metric_rate(
            overall, "required_fact_recall"
        )
        quality_metrics["required_fact_recall"] = _answerable_required_fact_recall(
            postprocessed, str(runner_model.get("model_id", ""))
        )
        quality_metrics["forbidden_claim_hit_count"] = _metric_numerator(
            overall.get("forbidden_claim_hit_rate")
        )
        quality_metrics["multi_window_required_fact_recall"] = _metric_rate(
            multi_window, "required_fact_recall"
        )
        quality_metrics["manual_review_count"] = _metric_numerator(
            overall.get("manual_review_rate")
        )
        model = {
            "profile": profile,
            "profile_label": PROFILE_LABELS[profile],
            "model_id": runner_model.get("model_id"),
            "artifact_sha256": runner_model.get("sha256"),
            "runner": {
                "smoke_accuracy_mean": _number(smoke_summary.get("accuracy_mean")),
                "smoke_error_count": _integer(smoke_summary.get("error_count"), default=0),
                "quality_error_count": _integer(runner_summary.get("error_count"), default=0),
                "ttft_p95_ms": _nested_number(runner_summary, "ttft_ms", "p95"),
                "ttft_p50_ms": _nested_number(runner_summary, "ttft_ms", "p50"),
                "total_latency_p95_ms": _nested_number(runner_summary, "total_latency_ms", "p95"),
                "client_estimated_decode_tps_p50": _nested_number(
                    runner_summary, "tokens_per_second", "p50"
                ),
                "decode_tps_p50": _nested_number(runner_summary, "server_tokens_per_second", "p50"),
                "decode_tps_p95": _nested_number(runner_summary, "server_tokens_per_second", "p95"),
                "load_p95_seconds": _nested_number(runner_summary, "load_seconds", "p95"),
                "mem_available_mb_min": _number(runner_summary.get("mem_available_mb_min")),
                "swap_used_mb_max": _number(runner_summary.get("swap_used_mb_max")),
                "process_tree_rss_mb_max": _number(runner_summary.get("process_tree_rss_mb_max")),
                "thermal_c_max": _number(runner_summary.get("thermal_c_max")),
            },
            "quality": quality_metrics,
            "resources": {"swap": swap["by_profile"][profile]},
            "llama_bench": llama_by_profile[profile],
            "voice_path": tts_by_profile.get(profile),
            "targets": {
                **INITIAL_GATES[profile],
                "voice_path_p95_ms_max": VOICE_PATH_TARGETS_MS[profile],
            },
            "gates": [],
        }
        models.append(model)

    source_checks_pass = all(check_item["passed"] for check_item in checks)
    model_by_profile = {model["profile"]: model for model in models}
    for model in models:
        model["gates"] = _build_model_gates(
            model,
            model_by_profile,
            source_checks_pass=source_checks_pass,
            swap_observation=swap["by_profile"][str(model["profile"])],
        )
        initial = [gate for gate in model["gates"] if gate["group"] == "initial"]
        all_required = [gate for gate in model["gates"] if gate.get("required_for_promotion")]
        model["initial_performance_gate_status"] = _aggregate_gate_status(initial)
        model["promotion_gate_status"] = _aggregate_gate_status(all_required)
        model["promotion_ready"] = model["promotion_gate_status"] == "pass"

    decision_supported = source_checks_pass and all(model["promotion_ready"] for model in models)
    if decision_supported:
        decision = {
            "supported": True,
            "status": "promotion_supported",
            "headline": "The evidence supports the planned tier mapping.",
            "summary": (
                "Every source-integrity, hard-safety, initial performance, and tier-"
                "differentiation gate is supported by the supplied artifacts."
            ),
        }
    else:
        decision = {
            "supported": False,
            "status": "promotion_not_supported",
            "headline": "No promotion decision is supported yet.",
            "summary": (
                "The controlled benchmark can compare observed speed and closed-world quality, "
                "but one or more required source, safety, soak, or differentiation gates failed "
                "or remain unverified."
            ),
        }

    return {
        "schema_version": 2,
        "generator": generator,
        "source_manifest": source_manifest,
        "source_checks": checks,
        "source_checks_pass": source_checks_pass,
        "decision": decision,
        "models": models,
        "device": {
            "swap": swap,
            "tegrastats": tegrastats,
        },
        "protocol": {
            "document": source_manifest["protocol"],
            "smoke": smoke.get("protocol"),
            "quality": quality.get("protocol"),
            "quality_corpus": quality.get("corpus"),
            "postprocessor_scorer": postprocessed.get("scorer"),
        },
    }


def _build_model_gates(
    model: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
    *,
    source_checks_pass: bool,
    swap_observation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    profile = str(model["profile"])
    runner = _object(model["runner"], "runner metrics")
    quality = _object(model["quality"], "quality metrics")
    targets = _object(model["targets"], "targets")
    gates: list[dict[str, Any]] = []

    def add(
        key: str,
        label: str,
        group: str,
        value: Any,
        threshold: Any,
        operator: str,
        passed: bool | None,
        source_roles: Sequence[str],
        *,
        required: bool = True,
        note: str = "",
    ) -> None:
        gates.append(
            {
                "key": key,
                "label": label,
                "group": group,
                "value": value,
                "threshold": threshold,
                "operator": operator,
                "status": "unverified" if passed is None else ("pass" if passed else "fail"),
                "source_roles": list(source_roles),
                "required_for_promotion": required,
                "note": note,
            }
        )

    add(
        "source_integrity",
        "Input identity and verification chain",
        "hard",
        source_checks_pass,
        True,
        "is",
        source_checks_pass,
        ("smoke_runner", "quality_runner", "postprocessed_quality"),
    )
    observed_errors = _integer(runner.get("quality_error_count"))
    add(
        "observed_runtime_failures",
        "No observed runner errors",
        "hard",
        observed_errors,
        0,
        "<=",
        None if observed_errors is None else observed_errors == 0,
        ("quality_runner",),
        note="This checks recorded trials; the full soak gate is separate.",
    )
    memory_min = _number(runner.get("mem_available_mb_min"))
    add(
        "memory_floor",
        "Warm-run memory floor (MiB)",
        "hard",
        memory_min,
        8192.0,
        ">=",
        None if memory_min is None else memory_min >= 8192.0,
        ("quality_runner", "protocol"),
    )
    swap_verified = swap_observation.get("verified") is True
    swap_growth_mb = _number(swap_observation.get("growth_mb")) if swap_verified else None
    add(
        "swap_growth",
        "Profile-attributed swap growth (MiB)",
        "hard",
        swap_growth_mb,
        256.0,
        "<=",
        None if swap_growth_mb is None else swap_growth_mb <= 256.0,
        ("quality_runner", "protocol"),
        note="Whole-run swap baseline and peak are diagnostic only and cannot be attributed to this profile.",
    )
    add(
        "profile_switch_p95",
        "Complete profile-switch p95 (seconds)",
        "hard",
        None,
        30.0,
        "<=",
        None,
        ("quality_runner", "protocol"),
        note="Load-only timing cannot satisfy unload, load, warm-up, and health-check timing.",
    )
    add(
        "thermal_decode_drop",
        "Thermal-soak decode degradation",
        "hard",
        None,
        0.10,
        "<=",
        None,
        ("tegrastats", "quality_runner", "protocol"),
        note="Temperature samples do not measure decode-throughput degradation.",
    )
    critical_claims = _integer(quality.get("forbidden_claim_hit_count"))
    add(
        "critical_fabricated_claims",
        "Critical fabricated claims",
        "hard",
        critical_claims,
        0,
        "<=",
        None if critical_claims is None else critical_claims == 0,
        ("postprocessed_quality", "protocol"),
        note="Lexical forbidden-claim hits are the available conservative proxy.",
    )
    add(
        "unsupported_minor_claims",
        "Unsupported minor-claim rate",
        "hard",
        None,
        0.02,
        "<=",
        None,
        ("postprocessed_quality", "protocol"),
        note="The deterministic lexical scorer does not classify unsupported minor claims.",
    )
    add(
        "evidence_precision",
        "Evidence precision",
        "hard",
        None,
        0.95,
        ">=",
        None,
        ("postprocessed_quality", "protocol"),
        note="Required-fact recall cannot be substituted for evidence precision.",
    )
    abstention = _number(quality.get("abstention_accuracy"))
    add(
        "abstention_accuracy",
        "Correct abstention rate",
        "hard",
        abstention,
        0.90,
        ">=",
        None if abstention is None else abstention >= 0.90,
        ("postprocessed_quality", "protocol"),
    )
    add(
        "full_protocol",
        "Complete load, warm-turn, and soak protocol",
        "hard",
        None,
        True,
        "is",
        None,
        ("quality_runner", "tegrastats", "protocol"),
        note="The supplied runner artifacts do not certify the complete soak contract.",
    )
    add(
        "audio_integrity",
        "No corrupted or missing voice output",
        "hard",
        None,
        True,
        "is",
        None,
        ("tts", "protocol"),
        note="Non-empty audio bytes do not establish that rendered audio is uncorrupted.",
    )

    ttft = _number(runner.get("ttft_p95_ms"))
    ttft_target = _number(targets.get("ttft_p95_ms_max"))
    add(
        "ttft",
        "Text time to first token p95 (ms)",
        "initial",
        ttft,
        ttft_target,
        "<=",
        None if ttft is None or ttft_target is None else ttft <= ttft_target,
        ("quality_runner", "protocol"),
    )
    decode = _number(runner.get("decode_tps_p50"))
    decode_target = _number(targets.get("decode_tps_min"))
    add(
        "decode",
        "Server-reported decode throughput p50 (tok/s)",
        "initial",
        decode,
        decode_target,
        ">=",
        None if decode is None or decode_target is None else decode >= decode_target,
        ("quality_runner", "protocol"),
        note="Uses llama.cpp timings.predicted_per_second, not the client end-minus-first-token estimate.",
    )
    recall = _number(quality.get("required_fact_recall"))
    recall_target = _number(targets.get("recall_min"))
    add(
        "required_fact_recall",
        "Answerable required-fact recall",
        "initial",
        recall,
        recall_target,
        ">=",
        None if recall is None or recall_target is None else recall >= recall_target,
        ("postprocessed_quality", "protocol"),
        note="Abstention-required cases are evaluated by the separate abstention gate.",
    )
    voice_target = _number(targets.get("voice_path_p95_ms_max"))
    add(
        "voice_path_p95",
        "End-of-speech to first playable audio p95 (ms)",
        "voice",
        None,
        voice_target,
        "<=",
        None,
        ("tts", "protocol"),
        note="The optional profile timer starts before the LLM request and is not this metric.",
    )

    if profile == "torch":
        baseline = _number(
            models.get("light", {}).get("quality", {}).get("multi_window_required_fact_recall")
        )
        current = _number(quality.get("multi_window_required_fact_recall"))
        delta = None if baseline is None or current is None else current - baseline
        add(
            "torch_multi_window_gain",
            "Multi-window gain over Light",
            "differentiation",
            delta,
            0.05,
            ">=",
            None if delta is None else delta >= 0.05,
            ("postprocessed_quality", "protocol"),
        )
    elif profile == "fire":
        baseline = _number(
            models.get("torch", {}).get("quality", {}).get("multi_window_required_fact_recall")
        )
        current = _number(quality.get("multi_window_required_fact_recall"))
        delta = None if baseline is None or current is None else current - baseline
        passed = True if delta is not None and delta >= 0.05 else None
        add(
            "fire_quality_differentiation",
            "Quality gain over Torch or blinded difficult-prompt win rate",
            "differentiation",
            {"multi_window_delta": delta, "blinded_win_rate": None},
            {"multi_window_delta": 0.05, "blinded_win_rate": 0.60},
            "either >=",
            passed,
            ("postprocessed_quality", "protocol"),
            note="A sub-threshold multi-window delta remains unverified until blind review exists.",
        )

    return gates


def _runner_model_map(payload: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    inventory = payload.get("inventory")
    raw_models = inventory.get("models") if isinstance(inventory, dict) else None
    if not isinstance(raw_models, list):
        raise BenchmarkReportError(f"{label} inventory.models must be a list")
    result: dict[str, dict[str, Any]] = {}
    model_ids: set[str] = set()
    for index, item in enumerate(raw_models):
        if not isinstance(item, dict):
            raise BenchmarkReportError(f"{label} inventory.models[{index}] must be an object")
        profile = str(item.get("profile", "")).strip().lower()
        model_id = str(item.get("model_id", "")).strip()
        if not profile or not model_id or profile in result or model_id in model_ids:
            raise BenchmarkReportError(f"{label} has an invalid or duplicate model entry")
        result[profile] = dict(item)
        model_ids.add(model_id)
    return result


def _postprocessed_model_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("model_summaries")
    if not isinstance(raw, list):
        raise BenchmarkReportError("postprocessed quality model_summaries must be a list")
    result: dict[str, dict[str, Any]] = {}
    model_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise BenchmarkReportError(f"model_summaries[{index}] must be an object")
        profile = str(item.get("profile", "")).strip().lower()
        model_id = str(item.get("model_id", "")).strip()
        if not profile or not model_id or profile in result or model_id in model_ids:
            raise BenchmarkReportError("postprocessed quality has invalid model summaries")
        result[profile] = dict(item)
        model_ids.add(model_id)
    return result


def _answerable_required_fact_recall(
    payload: Mapping[str, Any],
    model_id: str,
) -> float | None:
    trials = payload.get("trials")
    if not isinstance(trials, list):
        raise BenchmarkReportError("postprocessed quality trials must be a list")
    hits = 0
    targets = 0
    for trial in trials:
        if not isinstance(trial, dict) or trial.get("model_id") != model_id or trial.get("error"):
            continue
        scored = trial.get("postprocessed_quality")
        if not isinstance(scored, dict):
            raise BenchmarkReportError("postprocessed trial is missing quality details")
        abstention = scored.get("abstention")
        if not isinstance(abstention, dict):
            raise BenchmarkReportError("postprocessed trial is missing abstention details")
        if abstention.get("required") is True:
            continue
        required_facts = scored.get("required_facts")
        if not isinstance(required_facts, list):
            raise BenchmarkReportError("postprocessed trial is missing required-fact details")
        targets += len(required_facts)
        hits += sum(isinstance(fact, dict) and fact.get("hit") is True for fact in required_facts)
    return round(hits / targets, 4) if targets else None


def _summary_map(
    payload: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    raw = payload.get("summaries")
    if not isinstance(raw, list):
        raise BenchmarkReportError(f"{label} summaries must be a list")
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise BenchmarkReportError(f"{label} summaries[{index}] must be an object")
        profile = str(item.get("profile", "")).strip().lower()
        model_id = str(item.get("model_id", "")).strip()
        expected = models.get(profile)
        if not profile or not model_id or profile in result or expected is None:
            raise BenchmarkReportError(f"{label} has an invalid, duplicate, or unknown summary")
        if model_id != expected.get("model_id"):
            raise BenchmarkReportError(
                f"{label} summary model_id mismatch for {profile}: {model_id!r}"
            )
        result[profile] = dict(item)
    if set(result) != set(models):
        raise BenchmarkReportError(f"{label} summary profile matrix is incomplete")
    return result


def _has_exact_profile_matrix(models: Mapping[str, Mapping[str, Any]]) -> bool:
    model_ids = [str(model.get("model_id", "")).strip() for model in models.values()]
    return (
        set(models) == set(PROFILE_ORDER)
        and len(model_ids) == len(set(model_ids)) == len(PROFILE_ORDER)
        and all(model_ids)
    )


def _postprocessed_trials_match_matrix(
    payload: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
) -> bool:
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        return False
    seen_profiles: set[str] = set()
    for trial in trials:
        if not isinstance(trial, dict):
            return False
        profile = str(trial.get("profile", "")).strip().lower()
        expected = models.get(profile)
        if expected is None or trial.get("model_id") != expected.get("model_id"):
            return False
        seen_profiles.add(profile)
    return seen_profiles == set(PROFILE_ORDER)


def _verification_matrix_pass(
    value: Any,
    models: Mapping[str, Mapping[str, Any]],
) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != len(PROFILE_ORDER)
        or not _has_exact_profile_matrix(models)
    ):
        return False
    expected_pairs = {(profile, str(model["model_id"])) for profile, model in models.items()}
    observed_pairs: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, dict) or item.get("verified") is not True:
            return False
        pair = (
            str(item.get("profile", "")).strip().lower(),
            str(item.get("model_id", "")).strip(),
        )
        if pair not in expected_pairs or pair in observed_pairs:
            return False
        observed_pairs.add(pair)
    return observed_pairs == expected_pairs


def _all_trial_routes_pass(
    value: Any,
    models: Mapping[str, Mapping[str, Any]],
) -> bool:
    if not isinstance(value, list) or not value or not _has_exact_profile_matrix(models):
        return False
    expected_pairs = {(profile, str(model["model_id"])) for profile, model in models.items()}
    observed_pairs: set[tuple[str, str]] = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or item.get("route_verified") is not True
            or bool(item.get("error"))
        ):
            return False
        pair = (
            str(item.get("profile", "")).strip().lower(),
            str(item.get("model_id", "")).strip(),
        )
        if pair not in expected_pairs:
            return False
        observed_pairs.add(pair)
    return observed_pairs == expected_pairs


def _llama_bench_rows(payload: Any, label: str) -> list[dict[str, Any]]:
    rows: Any
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (
                payload[key]
                for key in ("results", "benchmarks", "rows", "data")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
    else:
        rows = None
    if not isinstance(rows, list) or not rows:
        raise BenchmarkReportError(f"{label} must contain a non-empty llama-bench row list")
    normalized = [dict(item) for item in rows if isinstance(item, dict)]
    if len(normalized) != len(rows) or not all(
        _number(item.get("avg_ts")) is not None for item in normalized
    ):
        raise BenchmarkReportError(f"{label} rows must be objects with numeric avg_ts")
    return normalized


def _llama_bench_identity_verified(
    payload: Any,
    rows: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> bool:
    if not isinstance(payload, dict):
        return False
    artifact = payload.get("artifact")
    artifact = artifact if isinstance(artifact, dict) else {}
    reported_path = payload.get("artifact_path", artifact.get("path"))
    reported_sha256 = payload.get("artifact_sha256", artifact.get("sha256"))
    expected_path = model.get("path")
    expected_sha256 = model.get("sha256")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (reported_path, reported_sha256, expected_path, expected_sha256)
    ):
        return False
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(reported_sha256)):
        return False
    if str(reported_sha256).casefold() != str(expected_sha256).casefold():
        return False
    if (
        Path(str(reported_path)).expanduser().resolve()
        != Path(str(expected_path)).expanduser().resolve()
    ):
        return False
    expected_filename = Path(str(expected_path)).name
    reported_filenames = {
        Path(str(row["model_filename"])).name
        for row in rows
        if isinstance(row.get("model_filename"), str) and row.get("model_filename")
    }
    return reported_filenames == {expected_filename}


def _llama_bench_settings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    signatures: list[dict[str, Any]] = []
    for row in rows:
        if any(field not in row for field in LLAMA_BENCH_SETTING_FIELDS):
            return {}
        signatures.append({field: row[field] for field in LLAMA_BENCH_SETTING_FIELDS})
    encoded = {json.dumps(signature, sort_keys=True) for signature in signatures}
    return signatures[0] if signatures and len(encoded) == 1 else {}


def _summarize_llama_bench(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_values: list[float] = []
    generation_values: list[float] = []
    build_commits: list[str] = []
    repetitions = 0
    for row in rows:
        avg_ts = _number(row.get("avg_ts"))
        n_prompt = _integer(row.get("n_prompt"), default=0)
        n_gen = _integer(row.get("n_gen"), default=0)
        if avg_ts is not None:
            if n_gen > 0:
                generation_values.append(avg_ts)
            elif n_prompt > 0:
                prompt_values.append(avg_ts)
        commit = row.get("build_commit")
        if isinstance(commit, str) and commit and commit not in build_commits:
            build_commits.append(commit)
        repetitions += max(_integer(row.get("repetitions"), default=1), 1)
    return {
        "row_count": len(rows),
        "reported_repetitions": repetitions,
        "prompt_tps_median": _median(prompt_values),
        "generation_tps_median": _median(generation_values),
        "prompt_test_count": len(prompt_values),
        "generation_test_count": len(generation_values),
        "build_commits": build_commits,
        "settings": _llama_bench_settings(rows),
    }


def _summarize_tts(
    payload: Any,
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or any(not isinstance(item, dict) for item in rows):
        raise BenchmarkReportError("optional TTS JSON must contain object result rows")
    unknown_profiles = {
        str(item.get("requested_profile", "")).strip().lower()
        for item in rows
        if str(item.get("requested_profile", "")).strip().lower() not in PROFILE_ORDER
    }
    if unknown_profiles:
        raise BenchmarkReportError(
            "optional TTS JSON contains unknown profiles: " + ", ".join(sorted(unknown_profiles))
        )
    result: dict[str, dict[str, Any] | None] = {profile: None for profile in PROFILE_ORDER}
    for profile in PROFILE_ORDER:
        selected = [
            item
            for item in rows
            if str(item.get("requested_profile", "")).strip().lower() == profile
        ]
        if not selected:
            continue
        expected_model_id = str(models.get(profile, {}).get("model_id", ""))
        successful = [item for item in selected if not item.get("error")]
        result[profile] = {
            "trial_count": len(selected),
            "error_count": len(selected) - len(successful),
            "identity_failure_count": sum(
                item.get("requested_model") != expected_model_id
                or item.get("served_model") != expected_model_id
                for item in selected
            ),
            "route_failure_count": sum(
                item.get("routing_verified") is not True for item in selected
            ),
            "empty_audio_count": sum(
                _integer(item.get("tts_audio_bytes"), default=0) <= 0 for item in selected
            ),
            "tts_latency_p95_ms": _nearest_rank_percentile(
                [_number(item.get("tts_latency_ms")) for item in successful], 0.95
            ),
            "end_to_end_p95_ms": _nearest_rank_percentile(
                [_number(item.get("end_to_end_ms")) for item in successful], 0.95
            ),
            "artifact_verification_reported": all(
                item.get("artifact_verified") is True for item in selected
            ),
        }
    return result


def _summarize_tegrastats(text: str) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    timestamp_pattern = re.compile(r"(?P<date>\d{2}-\d{2}-\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})")
    ram_pattern = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
    swap_pattern = re.compile(r"\bSWAP\s+(\d+)/(\d+)MB")
    temp_pattern = re.compile(r"\b[A-Za-z0-9_]+@(-?\d+(?:\.\d+)?)C")
    for line in text.splitlines():
        ram_match = ram_pattern.search(line)
        swap_match = swap_pattern.search(line)
        timestamp_match = timestamp_pattern.search(line)
        temperatures = [float(value) for value in temp_pattern.findall(line)]
        if not any((ram_match, swap_match, timestamp_match, temperatures)):
            continue
        timestamp = None
        if timestamp_match:
            try:
                timestamp = datetime.strptime(
                    f"{timestamp_match.group('date')} {timestamp_match.group('time')}",
                    "%m-%d-%Y %H:%M:%S",
                )
            except ValueError:
                timestamp = None
        samples.append(
            {
                "timestamp": timestamp,
                "ram_used_mb": float(ram_match.group(1)) if ram_match else None,
                "ram_total_mb": float(ram_match.group(2)) if ram_match else None,
                "swap_used_mb": float(swap_match.group(1)) if swap_match else None,
                "temperature_c_max": max(temperatures) if temperatures else None,
            }
        )
    available = [
        sample["ram_total_mb"] - sample["ram_used_mb"]
        for sample in samples
        if sample["ram_total_mb"] is not None and sample["ram_used_mb"] is not None
    ]
    swap = [sample["swap_used_mb"] for sample in samples if sample["swap_used_mb"] is not None]
    temperatures = [
        sample["temperature_c_max"] for sample in samples if sample["temperature_c_max"] is not None
    ]
    timestamps = [sample["timestamp"] for sample in samples if sample["timestamp"] is not None]
    return {
        "sample_count": len(samples),
        "mem_available_mb_min": min(available) if available else None,
        "swap_sample_count": len(swap),
        "swap_used_mb_baseline": swap[0] if swap else None,
        "swap_used_mb_min": min(swap) if swap else None,
        "swap_used_mb_peak": max(swap) if swap else None,
        "swap_used_mb_max": max(swap) if swap else None,
        "swap_growth_mb": max(swap) - swap[0] if swap else None,
        "temperature_c_max": max(temperatures) if temperatures else None,
        "duration_seconds": (
            (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) >= 2 else None
        ),
    }


def _swap_observation(
    values: Sequence[float],
    *,
    source_role: str,
    verified: bool,
) -> dict[str, Any]:
    baseline = values[0] if values else None
    peak = max(values) if values else None
    effective_verified = verified and baseline is not None and peak is not None
    return {
        "verified": effective_verified,
        "source_role": source_role,
        "sample_count": len(values),
        "baseline_mb": baseline,
        "minimum_mb": min(values) if values else None,
        "peak_mb": peak,
        "growth_mb": peak - baseline if effective_verified else None,
    }


def _summarize_swap_samples(
    samples: Any,
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    raw_rows = samples if isinstance(samples, list) else []
    numeric_rows: list[tuple[Mapping[str, Any], float]] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        value = _number(item.get("swap_used_mb"))
        if value is not None:
            numeric_rows.append((item, value))

    expected_pairs = {(profile, str(model["model_id"])) for profile, model in models.items()}
    attributed_rows = [
        item
        for item in raw_rows
        if isinstance(item, dict)
        and (str(item.get("profile", "")).strip() or str(item.get("model_id", "")).strip())
    ]
    attributed_pairs = [
        (
            str(item.get("profile", "")).strip().lower(),
            str(item.get("model_id", "")).strip(),
        )
        for item in attributed_rows
    ]
    attribution_valid = (
        bool(attributed_rows)
        and all(
            pair in expected_pairs and _number(item.get("swap_used_mb")) is not None
            for pair, item in zip(attributed_pairs, attributed_rows)
        )
        and set(attributed_pairs) == expected_pairs
    )
    by_profile: dict[str, dict[str, Any]] = {}
    for profile in PROFILE_ORDER:
        expected_pair = (profile, str(models.get(profile, {}).get("model_id", "")))
        profile_values = [
            value
            for item, value in numeric_rows
            if (
                str(item.get("profile", "")).strip().lower(),
                str(item.get("model_id", "")).strip(),
            )
            == expected_pair
        ]
        by_profile[profile] = _swap_observation(
            profile_values,
            source_role="quality_runner",
            verified=attribution_valid and len(profile_values) >= 2,
        )
    return {
        "whole_run": _swap_observation(
            [value for _, value in numeric_rows],
            source_role="quality_runner",
            verified=bool(numeric_rows),
        ),
        "by_profile": by_profile,
    }


def _whole_run_swap_from_tegrastats(summary: Mapping[str, Any]) -> dict[str, Any]:
    baseline = _number(summary.get("swap_used_mb_baseline"))
    peak = _number(summary.get("swap_used_mb_peak"))
    verified = baseline is not None and peak is not None
    return {
        "verified": verified,
        "source_role": "tegrastats",
        "sample_count": _integer(summary.get("swap_sample_count"), default=0),
        "baseline_mb": baseline,
        "minimum_mb": _number(summary.get("swap_used_mb_min")),
        "peak_mb": peak,
        "growth_mb": peak - baseline if verified else None,
    }


def build_report_payload(analysis: Mapping[str, Any]) -> dict[str, Any]:
    models = analysis["models"]
    charts = []
    charts.append(
        _chart_payload(
            "required-fact-recall",
            "Answerable required-fact recall against target",
            [
                {
                    "profile": model["profile_label"],
                    "series": series,
                    "value": value,
                }
                for model in models
                for series, value in (
                    ("Observed", model["quality"]["required_fact_recall"]),
                    ("Target", model["targets"]["recall_min"]),
                )
                if value is not None
            ],
            value_label="Recall rate",
            value_format="percent",
        )
    )
    charts.append(
        _chart_payload(
            "ttft-p95",
            "Text TTFT p95 against maximum",
            [
                {
                    "profile": model["profile_label"],
                    "series": series,
                    "value": value,
                }
                for model in models
                for series, value in (
                    ("Observed", model["runner"]["ttft_p95_ms"]),
                    ("Maximum", model["targets"]["ttft_p95_ms_max"]),
                )
                if value is not None
            ],
            value_label="Milliseconds",
            value_format="number",
        )
    )
    charts.append(
        _chart_payload(
            "decode-throughput",
            "Server-reported decode throughput against minimum",
            [
                {
                    "profile": model["profile_label"],
                    "series": series,
                    "value": value,
                }
                for model in models
                for series, value in (
                    ("Observed", model["runner"]["decode_tps_p50"]),
                    ("Minimum", model["targets"]["decode_tps_min"]),
                )
                if value is not None
            ],
            value_label="Tokens per second",
            value_format="number",
        )
    )
    return {"charts": charts}


def _chart_payload(
    chart_id: str,
    title: str,
    rows: list[dict[str, Any]],
    *,
    value_label: str,
    value_format: str,
) -> dict[str, Any]:
    return {
        "id": chart_id,
        "height": 320,
        "type": "bar",
        "dataset": {
            "id": chart_id,
            "title": title,
            "data": rows,
            "chart_spec": {
                "id": chart_id,
                "dataset": chart_id,
                "title": title,
                "type": "bar",
                "encodings": {
                    "x": {"field": "profile", "type": "nominal"},
                    "y": {"field": "value", "label": value_label, "type": "quantitative"},
                    "color": {"field": "series", "type": "nominal"},
                },
                "xAxisTitle": "",
                "yAxisTitle": value_label,
                "valueFormat": value_format,
            },
        },
    }


class _TooltipBuilder:
    def __init__(self) -> None:
        self.counter = 0

    def value(self, text: str, roles: Sequence[str], *, tag: str = "span") -> str:
        self.counter += 1
        tooltip_id = f"source-tooltip-{self.counter}"
        role_labels = [SOURCE_LABELS[role] for role in roles if role in SOURCE_LABELS]
        tooltip = "<br>".join(
            f"Source: {escape(str(label['source']))}<br>{escape(str(label['kind']))}: {escape(str(label['name']))}"
            for label in role_labels
        )
        attributes = f'class="source-tooltip" tabindex="0" aria-describedby="{tooltip_id}"'
        return (
            f"<{tag} {attributes}>{escape(text)}"
            f'<span class="source-tooltip-content" id="{tooltip_id}" role="tooltip">'
            f'<span class="source-tooltip-heading">Source details</span>{tooltip}</span></{tag}>'
        )

    def figure_source(self, roles: Sequence[str]) -> str:
        self.counter += 1
        tooltip_id = f"source-tooltip-{self.counter}"
        labels = [SOURCE_LABELS[role] for role in roles if role in SOURCE_LABELS]
        tooltip = "<br>".join(
            f"Source: {escape(str(label['source']))}<br>{escape(str(label['kind']))}: {escape(str(label['name']))}"
            for label in labels
        )
        return (
            f'<button type="button" class="source-tooltip" aria-describedby="{tooltip_id}">'
            f'Source<span class="source-tooltip-content" id="{tooltip_id}" role="tooltip">'
            f'<span class="source-tooltip-heading">Source details</span>{tooltip}</span></button>'
        )


def build_report_shell(
    analysis: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    shell_path: Path = DEFAULT_REPORT_SHELL,
) -> str:
    shell = shell_path.read_text(encoding="utf-8")
    if '<meta name="color-scheme" content="light dark">' not in shell:
        raise BenchmarkReportError(
            "required report shell is missing automatic color-scheme support"
        )
    tooltip = _TooltipBuilder()
    title = "Voice Model Candidate Benchmark"
    decision = analysis["decision"]
    models = analysis["models"]

    initial_status = ", ".join(
        f"{model['profile_label']} {model['initial_performance_gate_status']}" for model in models
    )
    summary_html = (
        f"<p><strong>{escape(str(decision['headline']))}</strong> "
        f"{escape(str(decision['summary']))}</p>"
        f"<p>Initial target status: {escape(initial_status)}. Initial targets are engineering "
        "screening thresholds; they do not replace the complete hard-gate protocol.</p>"
    )

    metric_cards = []
    for model in models:
        recall = _format_percent_tooltip(
            model["quality"]["required_fact_recall"], tooltip, ("postprocessed_quality",)
        )
        ttft = _format_number_tooltip(
            model["runner"]["ttft_p95_ms"], tooltip, ("quality_runner",), suffix=" ms"
        )
        metric_cards.append(
            '<div class="metric">'
            f'<div class="metric-label">{escape(model["profile_label"])} answerable fact recall</div>'
            f'<div class="metric-value">{recall}</div>'
            f'<div class="metric-note">TTFT p95 {ttft} · initial gate '
            f'<span class="status status-{escape(model["initial_performance_gate_status"])}">'
            f"{escape(model['initial_performance_gate_status'])}</span></div></div>"
        )
    metric_cards.append(
        '<div class="metric">'
        '<div class="metric-label">Promotion decision</div>'
        f'<div class="metric-value metric-value-text">{escape(decision["status"].replace("_", " "))}</div>'
        '<div class="metric-note">Fail-closed across source, safety, soak, and differentiation gates.</div>'
        "</div>"
    )

    chart_map = {chart["id"]: chart for chart in payload["charts"]}
    charts_html = "".join(
        _build_chart_figure(
            chart_map[chart_id],
            tooltip,
            roles=roles,
            subtitle=subtitle,
            alt_text=alt_text,
        )
        for chart_id, roles, subtitle, alt_text in (
            (
                "required-fact-recall",
                ("postprocessed_quality", "protocol"),
                "Observed closed-world lexical recall and the predeclared minimum by profile.",
                "Grouped bars comparing answerable required-fact recall with each profile target.",
            ),
            (
                "ttft-p95",
                ("quality_runner", "protocol"),
                "Lower is better; warmups are retained by the runner but excluded from summaries.",
                "Grouped bars comparing p95 time to first token with each profile maximum.",
            ),
            (
                "decode-throughput",
                ("quality_runner", "protocol"),
                "Higher is better; these are routed response measurements, not isolated kernels.",
                "Grouped bars comparing median server-reported decode throughput with each profile minimum.",
            ),
        )
    )

    performance_rows = []
    for model in models:
        llama_role = f"llama_bench_{model['profile']}"
        performance_rows.append(
            "<tr>"
            f"<td>{escape(model['profile_label'])}</td>"
            f"<td>{_format_percent_tooltip(model['quality']['required_fact_recall'], tooltip, ('postprocessed_quality',))}</td>"
            f"<td>{_format_percent_tooltip(model['quality']['multi_window_required_fact_recall'], tooltip, ('postprocessed_quality',))}</td>"
            f"<td>{_format_number_tooltip(model['runner']['ttft_p95_ms'], tooltip, ('quality_runner',), suffix=' ms')}</td>"
            f"<td>{_format_number_tooltip(model['runner']['decode_tps_p50'], tooltip, ('quality_runner',), suffix=' tok/s')}</td>"
            f"<td>{_format_number_tooltip(model['llama_bench']['generation_tps_median'], tooltip, (llama_role,), suffix=' tok/s')}</td>"
            f"<td>{_format_number_tooltip(model['runner']['mem_available_mb_min'], tooltip, ('quality_runner',), suffix=' MiB', decimals=0)}</td>"
            f'<td><span class="status status-{escape(model["promotion_gate_status"])}">{escape(model["promotion_gate_status"])}</span></td>'
            "</tr>"
        )
    performance_table = (
        "<table><thead><tr><th>Profile</th><th>Answerable fact recall</th>"
        "<th>Multi-window recall</th><th>TTFT p95</th><th>Server decode p50</th>"
        "<th>llama-bench generation</th><th>Min. available memory</th><th>Promotion gate</th>"
        f"</tr></thead><tbody>{''.join(performance_rows)}</tbody></table>"
    )

    gate_sections = "".join(_gate_table(model, tooltip) for model in models)
    source_rows = "".join(
        "<tr>"
        f"<td>{escape(check['label'])}</td>"
        f'<td><span class="status status-{"pass" if check["passed"] else "fail"}">'
        f"{'pass' if check['passed'] else 'fail'}</span></td>"
        f"<td>{escape(check['detail'])}</td></tr>"
        for check in analysis["source_checks"]
    )
    source_table = (
        "<table><thead><tr><th>Source check</th><th>Status</th><th>Method</th></tr></thead>"
        f"<tbody>{source_rows}</tbody></table>"
    )

    quality_protocol = analysis.get("protocol", {}).get("quality")
    quality_protocol = quality_protocol if isinstance(quality_protocol, dict) else {}
    rounds = _integer(quality_protocol.get("rounds"))
    warmups = _integer(quality_protocol.get("warmups_per_model_load"))
    methods_values = []
    if rounds is not None:
        methods_values.append(f"runner rounds {tooltip.value(str(rounds), ('quality_runner',))}")
    if warmups is not None:
        methods_values.append(
            f"warmups per load {tooltip.value(str(warmups), ('quality_runner',))}"
        )
    methods_values.extend(
        (
            f"temperature {tooltip.value('0', ('quality_runner',))}",
            f"top-k {tooltip.value('1', ('quality_runner',))}",
            f"top-p {tooltip.value('1.0', ('quality_runner',))}",
        )
    )
    methods_sentence = ", ".join(methods_values)

    manual_review_total = sum(
        _integer(model["quality"].get("manual_review_count"), default=0) for model in models
    )
    manual_review_value = tooltip.value(str(manual_review_total), ("postprocessed_quality",))
    required_case_count = tooltip.value("60", ("quality_runner", "postprocessed_quality"))
    hard_protocol = (
        f"{tooltip.value('five', ('protocol',))} cold loads, "
        f"{tooltip.value('thirty', ('protocol',))} warm turns, and a "
        f"{tooltip.value('30-minute', ('protocol',))} alternating soak"
    )

    main = f"""
    <main data-report-audience="technical">
      <article class="reading">
        <div class="kicker">On-device technical benchmark</div>
        <header data-contract-section="title"><h1>{escape(title)}</h1></header>
        <p class="deck">A fail-closed comparison of the Light, Torch, and Fire candidates using pinned runner, quality, isolated-performance, and optional voice-path evidence.</p>
        <section class="summary" data-contract-section="technical-summary">
          <div class="summary-label">Technical summary</div>
          <div class="summary-body">{summary_html}</div>
        </section>
        <section class="metrics">{"".join(metric_cards)}</section>
      </article>

      <section data-contract-section="key-findings">
        <article class="reading narrative">
          <h2>Observed quality and responsiveness are separable from promotion readiness</h2>
          <p>The figures compare the measured initial targets on common axes. A passed initial target does not override an unverified hard gate; the report therefore keeps screening performance and deployment readiness distinct.</p>
        </article>
        <div class="wide chart-grid">{charts_html}</div>
        <article class="reading">
          <section class="card table-card">
            <div class="card-head"><h3>Exact candidate comparison</h3><p>Routed response metrics, deterministic lexical quality, isolated llama-bench throughput, and resource floor.</p></div>
            <div class="table-scroll">{performance_table}</div>
          </section>
        </article>
      </section>

      <article class="reading">
        <section class="narrative" data-contract-section="scope-data-and-metric-definitions">
          <h2>The evidence chain is explicit and closed-world quality remains scoped</h2>
          <p>The reviewed suite contains {required_case_count} synthetic conversation questions covering local detail, multi-window synthesis, temporal order, speaker/decision/action attribution, and contradiction or unanswerable cases. Answerable required-fact recall is normalized lexical value-or-alias recovery on cases that do not require abstention; abstention has its own accuracy gate. TTFT is request start to first streamed text; server decode uses llama.cpp timing metadata, while the client-derived rate remains diagnostic only; llama-bench is an isolated engine diagnostic.</p>
          <section class="card table-card">
            <div class="card-head"><h3>Source and identity checks</h3><p>Any failed check blocks a promotion conclusion but remains visible for audit.</p></div>
            <div class="table-scroll">{source_table}</div>
          </section>
        </section>

        <section class="narrative" data-contract-section="methodology">
          <h2>Deterministic routed trials and isolated engine tests answer different questions</h2>
          <p>The router runner used {methods_sentence}, disabled prompt caching, disabled thinking, alternated model order, retained answers, and excluded warmups from reported summaries. The postprocessor rechecked the supplied quality-runner SHA-256 before applying normalized lexical rules. Separate llama-bench artifacts measure prompt and generation kernels without the HTTP, routing, and answer-shape overhead present in the routed runner.</p>
          <div class="gate-stack">{gate_sections}</div>
        </section>

        <section class="narrative" data-contract-section="limitations-uncertainty-and-robustness-checks">
          <h2>Missing soak and claim-review evidence blocks overconfident deployment claims</h2>
          <p>The complete promotion contract requires {hard_protocol}, full switch timing, thermal throughput degradation, evidence precision, unsupported-minor-claim review, and voice-output integrity. Raw temperature alone cannot prove the absence of throughput throttling. The lexical scorer flags answer patterns for review but does not establish semantic correctness; {manual_review_value} supplied trial results remain flagged across the candidate matrix.</p>
          <div class="caveat"><strong>Robustness boundary.</strong> Percentiles describe the supplied trials only. No confidence interval or population-level generalization is claimed, and the optional voice-profile end-to-end timer is not equivalent to end-of-speech-to-first-playable-audio instrumentation.</div>
        </section>

        <section class="narrative" data-contract-section="recommended-next-steps">
          <h2>Close the missing hard gates before changing the production default</h2>
          <ol>
            <li>Run the complete load, warm-turn, and alternating voice soak while recording per-turn throughput, switch boundaries, tegrastats, process memory, and audio integrity.</li>
            <li>Review every scorer-flagged answer and record evidence precision, unsupported minor claims, critical fabricated claims, and blinded difficult-prompt judgments.</li>
            <li>Capture end-of-speech, final ASR, retrieval, first LLM token, first TTS chunk, and playback start as separate monotonic events.</li>
            <li>Regenerate this report from the immutable artifacts; the decision logic will only mark promotion supported when every required gate is present and passing.</li>
          </ol>
        </section>

        <section class="narrative" data-contract-section="further-questions">
          <h2>Further questions</h2>
          <ul>
            <li>Does Torch retain a meaningful multi-window advantage after manual semantic review?</li>
            <li>Does Fire add enough difficult-context quality to justify its load, memory, and voice-path cost?</li>
            <li>Which latency stage dominates first playable audio for each profile under concurrent ASR and TTS?</li>
          </ul>
        </section>
      </article>
    </main>
    """

    main_start = shell.find("<main ")
    main_end = shell.find("</main>")
    if main_start < 0 or main_end < 0:
        raise BenchmarkReportError(
            "required report shell does not contain a replaceable main element"
        )
    main_end += len("</main>")
    authored = shell[:main_start] + main + shell[main_end:]
    authored = authored.replace("{{TITLE}}", escape(title))
    authored = authored.replace("{{SOURCE_AND_DATE}}", "Immutable local benchmark artifacts")
    authored = authored.replace("{{REPORT_AUDIENCE}}", "technical")
    authored = authored.replace("Data Analytics</div>", "Atlas Voice Engineering</div>")
    extra_css = """
    .metric-value-text { font-size: 17px; line-height: 24px; letter-spacing: -0.2px; text-transform: capitalize; }
    .chart-grid { display: grid; grid-template-columns: 1fr; gap: 18px; }
    .status { display: inline-flex; align-items: center; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 650; text-transform: capitalize; }
    .status-pass { background: var(--positive-bg); color: var(--positive); }
    .status-fail { background: var(--warning-bg); color: var(--warning); }
    .status-unverified { background: var(--surface-tertiary); color: var(--secondary); }
    .gate-stack { display: grid; gap: 14px; margin-top: 20px; }
    .gate-card { overflow: hidden; border: 1px solid var(--border); border-radius: 18px; }
    .gate-card h3 { padding: 13px 16px; border-bottom: 1px solid var(--border); }
    .gate-card table { margin: 0; }
    code { color: var(--secondary); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
    ol, ul { margin: 10px 0 0; padding-left: 22px; color: var(--secondary); }
    @media (max-width: 800px) { .chart-grid { display: block; } .chart-grid .card { margin-bottom: 18px; } }
    """
    authored = authored.replace("</style>", f"{extra_css}</style>", 1)
    if authored.count(REPORT_RUNTIME_MARKER) != 1:
        raise BenchmarkReportError("authored report shell must retain exactly one runtime marker")
    if re.search(r"\{\{[A-Z][A-Z0-9_]*\}\}", authored):
        raise BenchmarkReportError("authored report shell contains unresolved placeholders")
    for role in (
        "title",
        "technical-summary",
        "key-findings",
        "scope-data-and-metric-definitions",
        "methodology",
        "limitations-uncertainty-and-robustness-checks",
        "recommended-next-steps",
        "further-questions",
    ):
        if f'data-contract-section="{role}"' not in authored:
            raise BenchmarkReportError(f"authored report is missing technical section role: {role}")
    return authored


def _build_chart_figure(
    chart: Mapping[str, Any],
    tooltip: _TooltipBuilder,
    *,
    roles: Sequence[str],
    subtitle: str,
    alt_text: str,
) -> str:
    chart_id = str(chart["id"])
    rows = chart["dataset"]["data"]
    value_format = chart["dataset"]["chart_spec"]["valueFormat"]
    fallback = _grouped_bar_svg(rows, alt_text=alt_text, value_format=value_format)
    return f"""
      <figure class="card source-figure">
        <div class="card-head"><h3>{escape(str(chart["dataset"]["title"]))}</h3><p>{escape(subtitle)}</p></div>
        <div class="chart-wrap">
          <div data-recharts-chart="{escape(chart_id)}">
            <div class="chart-fallback" data-recharts-fallback>{fallback}</div>
            <div data-recharts-live aria-hidden="true"></div>
          </div>
        </div>
        <figcaption class="chart-note">Observed and target marks use the same reviewed rows in the live chart and static fallback.</figcaption>
        {tooltip.figure_source(roles)}
      </figure>
    """


def _grouped_bar_svg(rows: Sequence[Mapping[str, Any]], *, alt_text: str, value_format: str) -> str:
    profiles = list(dict.fromkeys(str(row["profile"]) for row in rows))
    series = list(dict.fromkeys(str(row["series"]) for row in rows))
    values = [_number(row.get("value")) for row in rows]
    numeric = [value for value in values if value is not None]
    maximum = max(numeric, default=1.0)
    if value_format == "percent":
        axis_max = max(1.0, math.ceil(maximum * 10) / 10)
    else:
        magnitude = 10 ** max(math.floor(math.log10(maximum)) - 1, 0) if maximum > 0 else 1
        axis_max = math.ceil(maximum / magnitude) * magnitude if magnitude else maximum
    axis_max = axis_max or 1.0
    width, height = 960, 420
    left, right, top, bottom = 76, 28, 36, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / max(len(profiles), 1)
    bar_width = min(64.0, group_width / max(len(series) + 1, 2))
    colors = {
        "Observed": "#0169cc",
        "Target": "#8f8f8f",
        "Maximum": "#8f8f8f",
        "Minimum": "#8f8f8f",
    }
    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(alt_text)}">',
        '<g fill="none" stroke="var(--grid)" stroke-width="1">',
    ]
    for tick in range(5):
        fraction = tick / 4
        y = top + plot_height * (1 - fraction)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
    elements.append("</g>")
    elements.append(
        '<g fill="var(--secondary)" font-family="system-ui, sans-serif" font-size="12">'
    )
    for tick in range(5):
        fraction = tick / 4
        y = top + plot_height * (1 - fraction)
        label_value = axis_max * fraction
        label = f"{label_value * 100:.0f}%" if value_format == "percent" else f"{label_value:g}"
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{escape(label)}</text>'
        )
    elements.append("</g>")
    lookup = {(str(row["profile"]), str(row["series"])): _number(row.get("value")) for row in rows}
    for profile_index, profile in enumerate(profiles):
        center = left + group_width * (profile_index + 0.5)
        total_bars_width = bar_width * len(series) + 8 * max(len(series) - 1, 0)
        start_x = center - total_bars_width / 2
        for series_index, series_name in enumerate(series):
            value = lookup.get((profile, series_name))
            if value is None:
                continue
            bar_height = max(value / axis_max * plot_height, 0)
            x = start_x + series_index * (bar_width + 8)
            y = top + plot_height - bar_height
            color = colors.get(series_name, "#8046d9")
            label = f"{value * 100:.1f}%" if value_format == "percent" else f"{value:.1f}"
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="4" fill="{color}" stroke="#3f3f3f" stroke-width="0.6"/>'
            )
            label_y = max(y - 7, 15)
            elements.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" fill="var(--secondary)" font-family="system-ui, sans-serif" font-size="11">{escape(label)}</text>'
            )
        elements.append(
            f'<text x="{center:.1f}" y="{height - 42}" text-anchor="middle" fill="var(--text)" font-family="system-ui, sans-serif" font-size="13" font-weight="600">{escape(profile)}</text>'
        )
    legend_x = left
    for index, series_name in enumerate(series):
        x = legend_x + index * 140
        color = colors.get(series_name, "#8046d9")
        elements.append(
            f'<rect x="{x}" y="{height - 20}" width="12" height="12" rx="2" fill="{color}"/>'
        )
        elements.append(
            f'<text x="{x + 18}" y="{height - 10}" fill="var(--secondary)" font-family="system-ui, sans-serif" font-size="11">{escape(series_name)}</text>'
        )
    elements.append("</svg>")
    return "".join(elements)


def _gate_table(model: Mapping[str, Any], tooltip: _TooltipBuilder) -> str:
    rows = []
    for gate in model["gates"]:
        value = _format_gate_value(gate, tooltip)
        threshold = _format_gate_threshold(gate, tooltip)
        status = escape(str(gate["status"]))
        rows.append(
            "<tr>"
            f"<td>{escape(str(gate['label']))}</td>"
            f"<td>{value}</td><td>{escape(str(gate['operator']))} {threshold}</td>"
            f'<td><span class="status status-{status}">{status}</span></td>'
            "</tr>"
        )
    return (
        '<section class="gate-card">'
        f"<h3>{escape(str(model['profile_label']))} gate ledger</h3>"
        '<div class="table-scroll"><table><thead><tr><th>Gate</th><th>Observed</th>'
        "<th>Required</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _format_gate_value(gate: Mapping[str, Any], tooltip: _TooltipBuilder) -> str:
    value = gate.get("value")
    roles = tuple(str(role) for role in gate.get("source_roles", []))
    if value is None:
        return "—"
    if isinstance(value, bool):
        return escape("yes" if value else "no")
    if isinstance(value, dict):
        present = [item for item in value.values() if _number(item) is not None]
        if not present:
            return "—"
        return " / ".join(
            _format_percent_tooltip(_number(item), tooltip, roles) for item in present
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if (
            "rate" in str(gate.get("key"))
            or "precision" in str(gate.get("key"))
            or "drop" in str(gate.get("key"))
            or "gain" in str(gate.get("key"))
        ):
            return _format_percent_tooltip(float(value), tooltip, roles)
        return _format_number_tooltip(float(value), tooltip, roles)
    return escape(str(value))


def _format_gate_threshold(gate: Mapping[str, Any], tooltip: _TooltipBuilder) -> str:
    threshold = gate.get("threshold")
    if threshold is None:
        return "—"
    if isinstance(threshold, bool):
        return "yes" if threshold else "no"
    if isinstance(threshold, dict):
        return " / ".join(
            _format_percent_tooltip(_number(value), tooltip, ("protocol",))
            for value in threshold.values()
            if _number(value) is not None
        )
    if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        if (
            "rate" in str(gate.get("key"))
            or "precision" in str(gate.get("key"))
            or "drop" in str(gate.get("key"))
            or "gain" in str(gate.get("key"))
            or "recall" in str(gate.get("key"))
        ):
            return _format_percent_tooltip(float(threshold), tooltip, ("protocol",))
        return _format_number_tooltip(float(threshold), tooltip, ("protocol",))
    return escape(str(threshold))


def _format_number_tooltip(
    value: float | int | None,
    tooltip: _TooltipBuilder,
    roles: Sequence[str],
    *,
    suffix: str = "",
    decimals: int = 1,
) -> str:
    number = _number(value)
    if number is None:
        return "—"
    rendered = f"{number:,.{decimals}f}{suffix}"
    return tooltip.value(rendered, roles)


def _format_percent_tooltip(
    value: float | int | None,
    tooltip: _TooltipBuilder,
    roles: Sequence[str],
) -> str:
    number = _number(value)
    if number is None:
        return "—"
    return tooltip.value(f"{number * 100:.1f}%", roles)


def write_executed_notebook(
    analysis: Mapping[str, Any],
    inputs: BenchmarkReportInputs,
    output_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise BenchmarkReportError(
            "notebook execution requires nbformat and nbclient; run this generator with "
            "the repository .venv Python"
        ) from exc

    source_paths = {
        role: str(path.expanduser().resolve()) for role, path in inputs.source_paths().items()
    }
    expected_hashes = {
        role: source["sha256"] for role, source in analysis["source_manifest"].items()
    }
    analysis_hash = _canonical_json_sha256(analysis)
    protocol_hash = str(analysis["source_manifest"]["protocol"]["sha256"])
    decision = analysis["decision"]
    tldr = (
        f"## tl;dr\n\n**{decision['headline']}** {decision['summary']} "
        "The notebook recomputes every source hash and rebuilds the analysis from the supplied artifacts."
    )
    takeaways = [
        f"- **{model['profile_label']}:** initial performance gate "
        f"`{model['initial_performance_gate_status']}`; promotion gate "
        f"`{model['promotion_gate_status']}`."
        for model in analysis["models"]
    ]
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
            "atlas_voice_report": {
                "generator": analysis["generator"],
                "analysis_sha256": analysis_hash,
                "protocol_sha256": protocol_hash,
                "canonicalization": "json-sort-keys-compact-utf8",
            },
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# Voice Model Candidate Benchmark\n\n" + tldr),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "This is a reproducible companion to the technical HTML report. It uses only local, "
            "explicitly supplied artifacts; no network calls or synthetic result substitution occur.\n\n"
            "### Key Assumptions\n\n"
            "- Runner and scorer schemas retain the profile, model, routing, and hash fields documented by Atlas Voice.\n"
            "- `llama-bench` files contain JSON rows with `avg_ts`, prompt/generation token counts, model filename, and build commit.\n"
            "- Missing hard-gate evidence is unverified, never silently passed."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import hashlib\n"
            "import json\n"
            "import sys\n\n"
            f"REPOSITORY_ROOT = Path({str(repository_root.resolve())!r})\n"
            "if str(REPOSITORY_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(REPOSITORY_ROOT))\n\n"
            "from atlas_voice.model_benchmark_report import (\n"
            "    BenchmarkReportInputs, build_analysis_from_paths\n"
            ")\n\n"
            f"SOURCE_PATHS = {source_paths!r}\n"
            f"EXPECTED_SHA256 = {expected_hashes!r}\n"
            f"EXPECTED_ANALYSIS_SHA256 = {analysis_hash!r}\n"
            f"EXPECTED_GENERATOR = {dict(analysis['generator'])!r}\n"
            f"EXPECTED_PROTOCOL_SHA256 = {protocol_hash!r}\n"
        ),
        nbformat.v4.new_markdown_cell("## Data\n\n### 1. Verify immutable source bytes"),
        nbformat.v4.new_code_cell(
            "source_checks = {}\n"
            "for role, raw_path in SOURCE_PATHS.items():\n"
            "    path = Path(raw_path)\n"
            "    digest = hashlib.sha256(path.read_bytes()).hexdigest()\n"
            "    source_checks[role] = {\n"
            "        'exists': path.is_file(),\n"
            "        'sha256': digest,\n"
            "        'matches_expected': digest == EXPECTED_SHA256[role],\n"
            "    }\n"
            "assert all(item['exists'] and item['matches_expected'] for item in source_checks.values())\n"
            "print(json.dumps(source_checks, indent=2, sort_keys=True))"
        ),
        nbformat.v4.new_markdown_cell("### 2. Rebuild the normalized comparison"),
        nbformat.v4.new_code_cell(
            "input_kwargs = {\n"
            "    'smoke_runner': Path(SOURCE_PATHS['smoke_runner']),\n"
            "    'quality_runner': Path(SOURCE_PATHS['quality_runner']),\n"
            "    'postprocessed_quality': Path(SOURCE_PATHS['postprocessed_quality']),\n"
            "    'llama_bench_light': Path(SOURCE_PATHS['llama_bench_light']),\n"
            "    'llama_bench_torch': Path(SOURCE_PATHS['llama_bench_torch']),\n"
            "    'llama_bench_fire': Path(SOURCE_PATHS['llama_bench_fire']),\n"
            "    'protocol': Path(SOURCE_PATHS['protocol']),\n"
            "}\n"
            "if 'tts' in SOURCE_PATHS:\n"
            "    input_kwargs['tts'] = Path(SOURCE_PATHS['tts'])\n"
            "if 'tegrastats' in SOURCE_PATHS:\n"
            "    input_kwargs['tegrastats'] = Path(SOURCE_PATHS['tegrastats'])\n"
            "rebuilt = build_analysis_from_paths(BenchmarkReportInputs(**input_kwargs))\n"
            "canonical_analysis = json.dumps(rebuilt, ensure_ascii=False, separators=(',', ':'), sort_keys=True)\n"
            "rebuilt_analysis_sha256 = hashlib.sha256(canonical_analysis.encode('utf-8')).hexdigest()\n"
            "assert rebuilt_analysis_sha256 == EXPECTED_ANALYSIS_SHA256\n"
            "assert rebuilt['generator'] == EXPECTED_GENERATOR\n"
            "assert rebuilt['protocol']['document']['sha256'] == EXPECTED_PROTOCOL_SHA256\n"
            "print(f'Canonical analysis SHA-256: {rebuilt_analysis_sha256}')\n"
            "print(json.dumps(rebuilt['source_checks'], indent=2, sort_keys=True))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Results\n\n### 3. Compare observed metrics and gate status"
        ),
        nbformat.v4.new_code_cell(
            "headers = ['Profile', 'Answerable recall', 'Multi-window', 'TTFT p95 ms', 'Server decode p50 tok/s', 'llama-bench gen tok/s', 'Initial', 'Promotion']\n"
            "rows = []\n"
            "for model in rebuilt['models']:\n"
            "    rows.append([\n"
            "        model['profile_label'],\n"
            "        model['quality']['required_fact_recall'],\n"
            "        model['quality']['multi_window_required_fact_recall'],\n"
            "        model['runner']['ttft_p95_ms'],\n"
            "        model['runner']['decode_tps_p50'],\n"
            "        model['llama_bench']['generation_tps_median'],\n"
            "        model['initial_performance_gate_status'],\n"
            "        model['promotion_gate_status'],\n"
            "    ])\n"
            "widths = [max(len(str(value)) for value in [header, *[row[index] for row in rows]]) for index, header in enumerate(headers)]\n"
            "def render(row):\n"
            "    return ' | '.join(str(value).ljust(widths[index]) for index, value in enumerate(row))\n"
            "print(render(headers))\n"
            "print('-+-'.join('-' * width for width in widths))\n"
            "for row in rows:\n"
            "    print(render(row))\n"
        ),
        nbformat.v4.new_code_cell(
            "gate_failures = {\n"
            "    model['profile']: [\n"
            "        {'gate': gate['key'], 'status': gate['status'], 'note': gate['note']}\n"
            "        for gate in model['gates']\n"
            "        if gate['required_for_promotion'] and gate['status'] != 'pass'\n"
            "    ]\n"
            "    for model in rebuilt['models']\n"
            "}\n"
            "print(json.dumps({'decision': rebuilt['decision'], 'blocking_gates': gate_failures}, indent=2, sort_keys=True))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            + "\n".join(takeaways)
            + "\n\nA production mapping should change only after all hard, initial, and differentiation gates pass from immutable rerun artifacts."
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        allow_errors=False,
        record_timing=False,
        resources={"metadata": {"path": str(repository_root.resolve())}},
    )
    client.execute(cwd=str(repository_root.resolve()))
    nbformat.write(notebook, output_path)


def generate_report_artifacts(
    inputs: BenchmarkReportInputs,
    output_dir: Path,
    *,
    shell_path: Path = DEFAULT_REPORT_SHELL,
) -> dict[str, str]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = build_analysis_from_paths(inputs)
    payload = build_report_payload(analysis)
    report_shell = build_report_shell(analysis, payload, shell_path=shell_path)

    analysis_path = output_dir / "analysis.json"
    payload_path = output_dir / "report-payload.json"
    authored_shell_path = output_dir / "report-shell.html"
    report_path = output_dir / "report.html"
    notebook_path = output_dir / "voice-model-benchmark.ipynb"
    manifest_path = output_dir / "source-manifest.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    authored_shell_path.write_text(report_shell, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(analysis["source_manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_executed_notebook(analysis, inputs, notebook_path)

    delivered = _embed_report(report_shell, payload)
    report_path.write_text(delivered, encoding="utf-8")
    _validate_embedded_report(delivered, payload)
    return {
        "report": str(report_path),
        "notebook": str(notebook_path),
        "analysis": str(analysis_path),
        "source_manifest": str(manifest_path),
        "authored_shell": str(authored_shell_path),
        "chart_payload": str(payload_path),
    }


def _safe_json_script_text(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        rendered.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _embed_report(shell: str, payload: Mapping[str, Any]) -> str:
    if shell.count(REPORT_RUNTIME_MARKER) != 1:
        raise BenchmarkReportError("authored report must contain exactly one runtime marker")
    encoded = _safe_json_script_text(payload)
    runtime = f'<script id="analytics-payload" type="application/json">{encoded}</script>'
    return shell.replace(REPORT_RUNTIME_MARKER, runtime)


def _validate_embedded_report(html: str, payload: Mapping[str, Any]) -> None:
    if REPORT_RUNTIME_MARKER in html:
        raise BenchmarkReportError("embedded report still contains the runtime marker")
    if re.search(r"<(?:script|link)[^>]+(?:src|href)=[\"']https?://", html, flags=re.I):
        raise BenchmarkReportError("embedded report contains a remote script or stylesheet")
    if 'id="analytics-payload"' not in html:
        raise BenchmarkReportError("embedded report is missing the packaged chart payload")
    for chart in payload.get("charts", []):
        chart_id = str(chart["id"])
        if html.count(f'data-recharts-chart="{chart_id}"') != 1:
            raise BenchmarkReportError(f"embedded report has an invalid host count for {chart_id}")
    if html.count('class="source-tooltip-content"') < 1:
        raise BenchmarkReportError("embedded report has no source tooltip content")
    if 'data-report-audience="technical"' not in html:
        raise BenchmarkReportError("embedded report lost the technical audience contract")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an executed notebook and self-contained technical HTML report from voice-model benchmark artifacts."
    )
    parser.add_argument("--smoke-runner", type=Path, required=True)
    parser.add_argument("--quality-runner", type=Path, required=True)
    parser.add_argument("--postprocessed-quality", type=Path, required=True)
    parser.add_argument("--llama-bench-light", type=Path, required=True)
    parser.add_argument("--llama-bench-torch", type=Path, required=True)
    parser.add_argument("--llama-bench-fire", type=Path, required=True)
    parser.add_argument("--tts", type=Path)
    parser.add_argument("--tegrastats", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_DOCUMENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-shell", type=Path, default=DEFAULT_REPORT_SHELL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = BenchmarkReportInputs(
        smoke_runner=args.smoke_runner,
        quality_runner=args.quality_runner,
        postprocessed_quality=args.postprocessed_quality,
        llama_bench_light=args.llama_bench_light,
        llama_bench_torch=args.llama_bench_torch,
        llama_bench_fire=args.llama_bench_fire,
        protocol=args.protocol,
        tts=args.tts,
        tegrastats=args.tegrastats,
    )
    try:
        outputs = generate_report_artifacts(
            inputs,
            args.output_dir,
            shell_path=args.report_shell,
        )
    except (BenchmarkReportError, OSError) as exc:
        print(f"report generation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


def _repository_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BenchmarkReportError(f"required repository file is missing: {resolved}")
    raw = resolved.read_bytes()
    try:
        reported_path = str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        reported_path = str(resolved)
    return {
        "path": reported_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _generator_identity() -> dict[str, Any]:
    return {
        "name": GENERATOR_NAME,
        "version": GENERATOR_VERSION,
        "source": _repository_file_identity(GENERATOR_SOURCE),
        "report_shell": _repository_file_identity(DEFAULT_REPORT_SHELL),
    }


def _canonical_json_sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkReportError(f"{label} must be a JSON object")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any, *, default: int | None = None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _nested_number(value: Mapping[str, Any], first: str, second: str) -> float | None:
    nested = value.get(first)
    return _number(nested.get(second)) if isinstance(nested, dict) else None


def _metric_rate(summary: Mapping[str, Any], key: str) -> float | None:
    metric = summary.get(key)
    return _number(metric.get("rate")) if isinstance(metric, dict) else None


def _metric_numerator(metric: Any) -> int | None:
    if not isinstance(metric, dict):
        return None
    for key in ("hit_count", "trial_hit_count", "pass_count", "flagged_count"):
        value = metric.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _median(values: Sequence[float]) -> float | None:
    return round(float(statistics.median(values)), 4) if values else None


def _nearest_rank_percentile(values: Iterable[float | None], percentile: float) -> float | None:
    present = sorted(value for value in values if value is not None)
    if not present:
        return None
    index = max(math.ceil(percentile * len(present)) - 1, 0)
    return round(float(present[index]), 4)


def _aggregate_gate_status(gates: Sequence[Mapping[str, Any]]) -> str:
    statuses = [str(gate.get("status")) for gate in gates]
    if not statuses or "unverified" in statuses:
        return "unverified"
    if "fail" in statuses:
        return "fail"
    return "pass"


if __name__ == "__main__":
    raise SystemExit(main())
