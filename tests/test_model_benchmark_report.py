from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from atlas_voice.model_benchmark_report import (
    DEFAULT_REPORT_SHELL,
    GENERATOR_SOURCE,
    REPORT_RUNTIME_MARKER,
    BenchmarkReportError,
    BenchmarkReportInputs,
    build_analysis_from_paths,
    build_report_payload,
    build_report_shell,
    generate_report_artifacts,
)


class ModelBenchmarkReportTests(unittest.TestCase):
    def test_analysis_fails_closed_when_hard_gate_evidence_is_absent(self) -> None:
        with TemporaryDirectory() as temporary:
            inputs = _write_fixture(Path(temporary), include_promotion_evidence=False)

            analysis = build_analysis_from_paths(inputs)

        self.assertTrue(analysis["source_checks_pass"])
        self.assertFalse(analysis["decision"]["supported"])
        self.assertEqual(
            [model["initial_performance_gate_status"] for model in analysis["models"]],
            ["pass", "pass", "pass"],
        )
        self.assertEqual(
            [model["promotion_gate_status"] for model in analysis["models"]],
            ["unverified", "unverified", "unverified"],
        )
        torch_quality = analysis["models"][1]["quality"]
        self.assertEqual(torch_quality["required_fact_recall"], 0.90)
        self.assertEqual(torch_quality["overall_required_fact_recall"], 0.80)

    def test_hand_edited_promotion_evidence_is_not_trusted(self) -> None:
        with TemporaryDirectory() as temporary:
            inputs = _write_fixture(Path(temporary), include_promotion_evidence=True)

            analysis = build_analysis_from_paths(inputs)

        self.assertTrue(analysis["source_checks_pass"])
        self.assertFalse(analysis["decision"]["supported"])
        self.assertTrue(all(not model["promotion_ready"] for model in analysis["models"]))

    def test_unwrapped_llama_bench_rows_fail_source_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            inputs = _write_fixture(Path(temporary), include_promotion_evidence=False)
            wrapped = json.loads(inputs.llama_bench_light.read_text(encoding="utf-8"))
            _write_json(inputs.llama_bench_light, wrapped["results"])

            analysis = build_analysis_from_paths(inputs)

        identity = next(
            check
            for check in analysis["source_checks"]
            if check["key"] == "llama_bench_light_identity"
        )
        self.assertFalse(identity["passed"])
        self.assertFalse(analysis["source_checks_pass"])

    def test_source_verification_requires_an_exact_unique_matrix(self) -> None:
        cases = {
            "missing": lambda rows: rows[:-1],
            "duplicate": lambda rows: [*rows, dict(rows[0])],
            "unknown": lambda rows: [
                *rows[:-1],
                {**rows[-1], "profile": "ember", "model_id": "qwen-ember"},
            ],
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                inputs = _write_fixture(Path(temporary), include_promotion_evidence=False)
                smoke = json.loads(inputs.smoke_runner.read_text(encoding="utf-8"))
                smoke["artifact_verification"] = mutate(smoke["artifact_verification"])
                _write_json(inputs.smoke_runner, smoke)

                analysis = build_analysis_from_paths(inputs)

            verification = next(
                check
                for check in analysis["source_checks"]
                if check["key"] == "smoke_runner_artifact_verification"
            )
            self.assertFalse(verification["passed"])
            self.assertFalse(analysis["source_checks_pass"])

    def test_duplicate_candidate_model_ids_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            inputs = _write_fixture(Path(temporary), include_promotion_evidence=False)
            smoke = json.loads(inputs.smoke_runner.read_text(encoding="utf-8"))
            smoke["inventory"]["models"][1]["model_id"] = smoke["inventory"]["models"][0][
                "model_id"
            ]
            _write_json(inputs.smoke_runner, smoke)

            with self.assertRaisesRegex(BenchmarkReportError, "duplicate model entry"):
                build_analysis_from_paths(inputs)

    def test_whole_run_swap_is_diagnostic_not_profile_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            inputs = _write_fixture(Path(temporary), include_promotion_evidence=False)

            analysis = build_analysis_from_paths(inputs)

        whole_run = analysis["device"]["swap"]["whole_run"]
        self.assertEqual(whole_run["baseline_mb"], 100.0)
        self.assertEqual(whole_run["peak_mb"], 120.0)
        self.assertEqual(whole_run["growth_mb"], 20.0)
        for model in analysis["models"]:
            self.assertFalse(model["resources"]["swap"]["verified"])
            gate = next(gate for gate in model["gates"] if gate["key"] == "swap_growth")
            self.assertEqual(gate["status"], "unverified")
            self.assertIsNone(gate["value"])

    def test_profile_attributed_swap_samples_can_verify_each_gate(self) -> None:
        with TemporaryDirectory() as temporary:
            inputs = _write_fixture(Path(temporary), include_promotion_evidence=False)
            quality = json.loads(inputs.quality_runner.read_text(encoding="utf-8"))
            quality["resource_samples"] = [
                {
                    "profile": model["profile"],
                    "model_id": model["model_id"],
                    "swap_used_mb": value,
                }
                for model in quality["inventory"]["models"]
                for value in (100.0, 120.0)
            ]
            _write_json(inputs.quality_runner, quality)
            postprocessed = json.loads(inputs.postprocessed_quality.read_text(encoding="utf-8"))
            postprocessed["inputs"]["benchmark"]["sha256"] = hashlib.sha256(
                inputs.quality_runner.read_bytes()
            ).hexdigest()
            _write_json(inputs.postprocessed_quality, postprocessed)

            analysis = build_analysis_from_paths(inputs)

        self.assertTrue(analysis["source_checks_pass"])
        for model in analysis["models"]:
            self.assertTrue(model["resources"]["swap"]["verified"])
            self.assertEqual(model["resources"]["swap"]["growth_mb"], 20.0)
            gate = next(gate for gate in model["gates"] if gate["key"] == "swap_growth")
            self.assertEqual(gate["status"], "pass")

    def test_generator_and_protocol_have_hashed_repo_owned_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            inputs = _write_fixture(Path(temporary), include_promotion_evidence=False)

            analysis = build_analysis_from_paths(inputs)

        self.assertNotIn(".codex", str(DEFAULT_REPORT_SHELL))
        self.assertTrue(DEFAULT_REPORT_SHELL.is_file())
        self.assertEqual(analysis["generator"]["name"], "atlas_voice.model_benchmark_report")
        self.assertEqual(
            analysis["generator"]["source"]["sha256"],
            hashlib.sha256(GENERATOR_SOURCE.read_bytes()).hexdigest(),
        )
        protocol = analysis["source_manifest"]["protocol"]
        self.assertEqual(
            protocol["sha256"],
            hashlib.sha256(inputs.protocol.read_bytes()).hexdigest(),
        )

    def test_technical_shell_has_same_data_fallbacks_and_source_tooltips(self) -> None:
        with TemporaryDirectory() as temporary:
            inputs = _write_fixture(Path(temporary), include_promotion_evidence=False)
            analysis = build_analysis_from_paths(inputs)
            payload = build_report_payload(analysis)

            shell = build_report_shell(analysis, payload)

        self.assertIn('data-report-audience="technical"', shell)
        self.assertIn('data-contract-section="technical-summary"', shell)
        self.assertIn(
            'data-contract-section="limitations-uncertainty-and-robustness-checks"', shell
        )
        self.assertEqual(shell.count("data-recharts-chart="), 3)
        self.assertEqual(shell.count("data-recharts-fallback>"), 3)
        self.assertGreater(shell.count('class="source-tooltip"'), 20)
        self.assertEqual(shell.count(REPORT_RUNTIME_MARKER), 1)
        self.assertNotIn("https://", shell)

    def test_generator_executes_notebook_and_embeds_repo_owned_payload(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _write_fixture(root, include_promotion_evidence=False)

            outputs = generate_report_artifacts(inputs, root / "output")

            report = Path(outputs["report"]).read_text(encoding="utf-8")
            notebook = json.loads(Path(outputs["notebook"]).read_text(encoding="utf-8"))
            analysis = json.loads(Path(outputs["analysis"]).read_text(encoding="utf-8"))

        self.assertNotIn(REPORT_RUNTIME_MARKER, report)
        self.assertIn('id="analytics-payload"', report)
        self.assertNotIn('src="http', report)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        canonical = json.dumps(
            analysis, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        metadata = notebook["metadata"]["atlas_voice_report"]
        self.assertEqual(metadata["analysis_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(
            metadata["protocol_sha256"], analysis["source_manifest"]["protocol"]["sha256"]
        )
        code_source = "\n".join(
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in code_cells
        )
        self.assertIn("assert rebuilt_analysis_sha256 == EXPECTED_ANALYSIS_SHA256", code_source)
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.get("execution_count") is not None for cell in code_cells))
        self.assertTrue(
            all(
                output.get("output_type") != "error"
                for cell in code_cells
                for output in cell.get("outputs", [])
            )
        )


def _write_fixture(root: Path, *, include_promotion_evidence: bool) -> BenchmarkReportInputs:
    model_rows = [
        {
            "profile": "light",
            "model_id": "qwen-light",
            "path": "/models/Qwen-Light-Q8_0.gguf",
            "sha256": "1" * 64,
        },
        {
            "profile": "torch",
            "model_id": "qwen-torch",
            "path": "/models/Qwen-Torch-Q8_0.gguf",
            "sha256": "2" * 64,
        },
        {
            "profile": "fire",
            "model_id": "qwen-fire",
            "path": "/models/Qwen-Fire-Q8_0.gguf",
            "sha256": "3" * 64,
        },
    ]
    observed = {
        "light": {"ttft": 700.0, "decode": 22.0, "recall": 0.82, "multi": 0.81},
        "torch": {"ttft": 1000.0, "decode": 16.0, "recall": 0.90, "multi": 0.87},
        "fire": {"ttft": 2200.0, "decode": 11.0, "recall": 0.96, "multi": 0.94},
    }
    promotion_evidence = None
    if include_promotion_evidence:
        promotion_evidence = {
            profile: {
                "profile_switch_p95_seconds": 18.0,
                "thermal_decode_throughput_drop_rate": 0.04,
                "unsupported_minor_claim_rate": 0.01,
                "evidence_precision": 0.98,
                "full_protocol_completed": True,
                "audio_corruption_free": True,
            }
            for profile in ("light", "torch", "fire")
        }

    smoke = _runner_payload(
        model_rows,
        observed,
        case_count=8,
        corpus_version="voice_model_smoke_v1",
        corpus_sha="a" * 64,
        promotion_evidence=promotion_evidence,
    )
    quality = _runner_payload(
        model_rows,
        observed,
        case_count=60,
        corpus_version="voice_model_quality_v1",
        corpus_sha="b" * 64,
        promotion_evidence=promotion_evidence,
    )
    smoke_path = _write_json(root / "smoke.json", smoke)
    quality_path = _write_json(root / "quality.json", quality)
    quality_sha = hashlib.sha256(quality_path.read_bytes()).hexdigest()

    postprocessed = {
        "schema_version": "1.0.0",
        "status": "completed",
        "inputs": {
            "benchmark": {"sha256": quality_sha, "status": "completed"},
            "corpus": {
                "sha256": "b" * 64,
                "dataset_version": "voice_model_quality_v1",
                "case_count": 60,
            },
        },
        "validation": {
            "expected_trial_count": 180,
            "actual_trial_count": 180,
            "rounds": 1,
            "model_count": 3,
        },
        "scorer": {
            "version": "fixture-lexical-v1",
            "claim_scope": "Deterministic fixture scorer.",
        },
        "model_summaries": [
            _postprocessed_summary(row, observed[row["profile"]]) for row in model_rows
        ],
        "trials": _postprocessed_trials(model_rows, observed),
    }
    post_path = _write_json(root / "postprocessed.json", postprocessed)

    llama_paths: dict[str, Path] = {}
    for index, row in enumerate(model_rows, start=1):
        profile = row["profile"]
        llama_paths[profile] = _write_json(
            root / f"llama-{profile}.json",
            {
                "artifact": {"path": row["path"], "sha256": row["sha256"]},
                "results": [
                    _llama_row(row, n_prompt=512, n_gen=0, avg_ts=100.0 / index),
                    _llama_row(
                        row, n_prompt=0, n_gen=128, avg_ts=observed[profile]["decode"] + 1.0
                    ),
                ],
            },
        )

    return BenchmarkReportInputs(
        smoke_runner=smoke_path,
        quality_runner=quality_path,
        postprocessed_quality=post_path,
        llama_bench_light=llama_paths["light"],
        llama_bench_torch=llama_paths["torch"],
        llama_bench_fire=llama_paths["fire"],
    )


def _llama_row(
    model: dict[str, object],
    *,
    n_prompt: int,
    n_gen: int,
    avg_ts: float,
) -> dict[str, object]:
    return {
        "build_commit": "fixture-commit",
        "model_filename": model["path"],
        "n_batch": 2048,
        "n_ubatch": 512,
        "n_threads": 12,
        "type_k": "q8_0",
        "type_v": "q8_0",
        "n_gpu_layers": -1,
        "n_cpu_moe": 0,
        "no_kv_offload": False,
        "flash_attn": 1,
        "use_mmap": True,
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "avg_ts": avg_ts,
        "repetitions": 5,
    }


def _postprocessed_trials(
    model_rows: list[dict[str, object]],
    observed: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    trials: list[dict[str, object]] = []
    for model in model_rows:
        profile = str(model["profile"])
        hit_count = round(observed[profile]["recall"] * 100)
        trials.append(
            {
                "round": 1,
                "profile": profile,
                "model_id": model["model_id"],
                "prompt_id": f"{profile}-answerable",
                "category": "local_detail",
                "error": None,
                "postprocessed_quality": {
                    "abstention": {"required": False},
                    "required_facts": [
                        {"fact_id": f"fact-{index}", "hit": index < hit_count}
                        for index in range(100)
                    ],
                },
            }
        )
    return trials


def _runner_payload(
    model_rows: list[dict[str, object]],
    observed: dict[str, dict[str, float]],
    *,
    case_count: int,
    corpus_version: str,
    corpus_sha: str,
    promotion_evidence: dict[str, dict[str, object]] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "error_count": 0,
        "protocol": {
            "rounds": 1,
            "warmups_per_model_load": 1,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "enable_thinking": False,
        },
        "inventory": {"models": model_rows},
        "corpus": {
            "sha256": corpus_sha,
            "corpus_version": corpus_version,
            "case_count": case_count,
        },
        "artifact_verification": [
            {"profile": row["profile"], "model_id": row["model_id"], "verified": True}
            for row in model_rows
        ],
        "route_verification": [
            {"profile": row["profile"], "model_id": row["model_id"], "verified": True}
            for row in model_rows
        ],
        "trials": [
            {
                "profile": row["profile"],
                "model_id": row["model_id"],
                "route_verified": True,
                "error": None,
            }
            for row in model_rows
        ],
        "summaries": [
            {
                "profile": row["profile"],
                "model_id": row["model_id"],
                "accuracy_mean": observed[str(row["profile"])]["recall"],
                "error_count": 0,
                "ttft_ms": {
                    "p50": observed[str(row["profile"])]["ttft"] - 50,
                    "p95": observed[str(row["profile"])]["ttft"],
                },
                "total_latency_ms": {"p50": 2500.0, "p95": 3000.0},
                "tokens_per_second": {
                    "p50": observed[str(row["profile"])]["decode"],
                    "p95": observed[str(row["profile"])]["decode"] + 1,
                },
                "server_tokens_per_second": {
                    "p50": observed[str(row["profile"])]["decode"],
                    "p95": observed[str(row["profile"])]["decode"] + 1,
                },
                "load_seconds": {"p50": 10.0, "p95": 12.0},
                "mem_available_mb_min": 12000.0,
                "swap_used_mb_max": 120.0,
                "process_tree_rss_mb_max": 24000.0,
                "thermal_c_max": 62.0,
            }
            for row in model_rows
        ],
        "resource_samples": [
            {"swap_used_mb": 100.0},
            {"swap_used_mb": 120.0},
        ],
    }
    if promotion_evidence is not None:
        payload["promotion_evidence"] = promotion_evidence
    return payload


def _postprocessed_summary(
    model: dict[str, object], metrics: dict[str, float]
) -> dict[str, object]:
    overall = {
        "required_fact_recall": _metric(max(metrics["recall"] - 0.10, 0.0), "hit_count"),
        "forbidden_claim_hit_rate": _metric(0.0, "hit_count"),
        "forbidden_claim_trial_rate": _metric(0.0, "trial_hit_count"),
        "exact_pass_rate": _metric(metrics["recall"], "pass_count"),
        "abstention_accuracy": _metric(1.0, "pass_count", target_count=10),
        "unwarranted_abstention_rate": _metric(0.0, "hit_count"),
        "attribution_target_accuracy": _metric(metrics["recall"], "pass_count"),
        "timestamp_target_accuracy": _metric(metrics["recall"], "pass_count"),
        "temporal_order_accuracy": _metric(metrics["recall"], "pass_count", target_count=10),
        "manual_review_rate": _metric(0.0, "flagged_count"),
    }
    return {
        "profile": model["profile"],
        "model_id": model["model_id"],
        "overall": overall,
        "by_category": [
            {
                "category": "multi_window_synthesis",
                "required_fact_recall": _metric(metrics["multi"], "hit_count"),
            }
        ],
    }


def _metric(rate: float, numerator_name: str, *, target_count: int = 100) -> dict[str, object]:
    return {
        numerator_name: round(rate * target_count),
        "target_count": target_count,
        "rate": rate,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
