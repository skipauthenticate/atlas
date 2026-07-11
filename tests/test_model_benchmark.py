from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from atlas_voice.model_benchmark import (
    ArtifactVerificationError,
    CorpusCase,
    FactRule,
    InputValidationError,
    ModelArtifact,
    NullSampler,
    RouteVerificationError,
    RouterClient,
    _load_api_key,
    _observed_decode_tps,
    _validate_output_path,
    _write_failure_report,
    _write_report,
    _validate_router_target,
    alternating_model_orders,
    build_parser,
    load_corpus,
    load_inventory,
    main,
    run_benchmark,
    score_answer,
    verify_artifacts,
)


class ModelBenchmarkTests(unittest.TestCase):
    def test_artifact_verification_checks_size_and_sha256(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "model.gguf"
            artifact.write_bytes(b"pinned model bytes")
            inventory_path = _write_inventory(root, [("light", "model-a", artifact)])
            inventory = load_inventory(inventory_path)

            verified = verify_artifacts(inventory)
            self.assertTrue(verified[0]["verified"])
            self.assertEqual(verified[0]["actual_sha256"], _sha256(artifact))

            payload = json.loads(inventory_path.read_text(encoding="utf-8"))
            payload["models"][0]["sha256"] = "0" * 64
            inventory_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactVerificationError, "SHA-256 mismatch"):
                verify_artifacts(load_inventory(inventory_path))

    def test_observed_decode_tps_uses_first_to_last_token_interval(self) -> None:
        self.assertEqual(
            _observed_decode_tps(11, first_text_at=4.0, last_text_at=4.5),
            20.0,
        )
        self.assertIsNone(_observed_decode_tps(1, first_text_at=4.0, last_text_at=4.5))
        self.assertIsNone(_observed_decode_tps(11, first_text_at=4.5, last_text_at=4.5))

    def test_output_path_rejects_collisions_symlinks_stale_temp_and_wrong_suffix(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = root / "inventory.json"
            inventory.write_text("{}", encoding="utf-8")
            model = root / "model.gguf"
            model.write_bytes(b"model")

            with self.assertRaisesRegex(InputValidationError, "collides"):
                _validate_output_path(inventory, protected_paths=(inventory,))
            with self.assertRaisesRegex(InputValidationError, r"\.json extension"):
                _validate_output_path(root / "result.txt", protected_paths=())

            existing_output = root / "existing.json"
            existing_output.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(InputValidationError, "overwrite existing"):
                _validate_output_path(existing_output, protected_paths=())

            symlink_target = root / "target.json"
            symlink_target.write_text("{}", encoding="utf-8")
            symlink_output = root / "symlink.json"
            symlink_output.symlink_to(symlink_target)
            with self.assertRaisesRegex(InputValidationError, "symlink"):
                _validate_output_path(symlink_output, protected_paths=())

            hardlink_output = root / "model-alias.json"
            os.link(model, hardlink_output)
            with self.assertRaisesRegex(InputValidationError, "collides"):
                _validate_output_path(hardlink_output, protected_paths=(model,))

            output = root / "result.json"
            output.with_suffix(".json.tmp").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(InputValidationError, "temporary"):
                _validate_output_path(output, protected_paths=())

            protected_temporary = root / "protected.json.tmp"
            with self.assertRaisesRegex(InputValidationError, "collides"):
                _validate_output_path(
                    root / "protected.json",
                    protected_paths=(protected_temporary,),
                )
            symlink_temporary = root / "symlink-temp.json.tmp"
            symlink_temporary.symlink_to(symlink_target)
            with self.assertRaisesRegex(InputValidationError, "symlink"):
                _validate_output_path(root / "symlink-temp.json", protected_paths=())

    def test_success_and_failure_reports_do_not_clobber_a_racing_target(self) -> None:
        for filename, failure_report in (
            ("success.json", False),
            ("failure.json", True),
        ):
            with self.subTest(filename=filename):
                with TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    output = root / filename
                    racer_bytes = b"independent writer"
                    original_link = os.link

                    def racing_link(
                        source,
                        destination,
                        *,
                        src_dir_fd=None,
                        dst_dir_fd=None,
                        follow_symlinks=True,
                    ):
                        descriptor = os.open(
                            destination,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o600,
                            dir_fd=dst_dir_fd,
                        )
                        try:
                            os.write(descriptor, racer_bytes)
                        finally:
                            os.close(descriptor)
                        return original_link(
                            source,
                            destination,
                            src_dir_fd=src_dir_fd,
                            dst_dir_fd=dst_dir_fd,
                            follow_symlinks=follow_symlinks,
                        )

                    with patch(
                        "atlas_voice.model_benchmark.os.link",
                        side_effect=racing_link,
                    ):
                        if failure_report:
                            _write_failure_report(output, "benchmark_failed", RuntimeError("boom"))
                        else:
                            with self.assertRaisesRegex(
                                InputValidationError, "appeared during publication"
                            ):
                                _write_report(output, {"status": "completed"})

                    self.assertEqual(output.read_bytes(), racer_bytes)
                    self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_cli_refuses_to_overwrite_inventory(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "model.gguf"
            artifact.write_bytes(b"model")
            inventory_path = _write_inventory(root, [("light", "model-a", artifact)])
            corpus_path = _write_corpus(root)
            original = inventory_path.read_bytes()

            exit_code = main(
                [
                    "--inventory",
                    str(inventory_path),
                    "--corpus",
                    str(corpus_path),
                    "--output",
                    str(inventory_path),
                    "--no-resource-sampling",
                ]
            )

            self.assertEqual(exit_code, 2)
            self.assertEqual(inventory_path.read_bytes(), original)

    def test_router_port_8080_requires_explicit_mutation_opt_in(self) -> None:
        self.assertEqual(
            build_parser().parse_args(["--inventory", "inventory.json"]).base_url,
            "http://127.0.0.1:18080",
        )
        with self.assertRaisesRegex(InputValidationError, "port 8080"):
            _validate_router_target("http://127.0.0.1:8080", allow_router_mutation=False)
        _validate_router_target("http://127.0.0.1:8080", allow_router_mutation=True)
        _validate_router_target("http://127.0.0.1:18080", allow_router_mutation=False)

    def test_api_key_uses_environment_or_private_regular_file(self) -> None:
        with patch.dict(os.environ, {"ATLAS_TEST_ROUTER_KEY": "environment-secret"}):
            self.assertEqual(
                _load_api_key(None, "ATLAS_TEST_ROUTER_KEY"),
                "environment-secret",
            )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "router.key"
            path.write_text("file-secret\n", encoding="utf-8")
            path.chmod(0o600)
            self.assertEqual(_load_api_key(path, None), "file-secret")
            path.chmod(0o644)
            with self.assertRaisesRegex(InputValidationError, "group/other"):
                _load_api_key(path, None)

    def test_benchmark_wrapper_bootstraps_source_checkout(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark-voice-models.py"
        with TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, "-I", str(script), "--help"],
                cwd=temporary,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--allow-router-mutation", completed.stdout)

    def test_alternating_order_reverses_then_rotates(self) -> None:
        models = tuple(
            ModelArtifact(profile=name, model_id=name, path=Path(f"/{name}"), sha256="a" * 64)
            for name in ("light", "torch", "fire")
        )

        orders = alternating_model_orders(models, 4)

        self.assertEqual([item.profile for item in orders[0]], ["light", "torch", "fire"])
        self.assertEqual([item.profile for item in orders[1]], ["fire", "torch", "light"])
        self.assertEqual([item.profile for item in orders[2]], ["torch", "fire", "light"])
        self.assertEqual([item.profile for item in orders[3]], ["light", "fire", "torch"])

    def test_score_answer_handles_required_forbidden_and_abstention_rules(self) -> None:
        answerable = CorpusCase(
            prompt_id="decision",
            category="decision",
            evidence=("Cobalt was selected; Emerald was rejected.",),
            question="Which channel?",
            required_facts=(FactRule("selected", ("Cobalt",)),),
            forbidden_claims=(FactRule("rejected", ("selected Emerald",)),),
            should_abstain=False,
        )
        abstention_phrases = ("not provided in the evidence",)

        passing = score_answer("Cobalt was selected.", answerable, abstention_phrases)
        hallucinated = score_answer("Cobalt and selected Emerald.", answerable, abstention_phrases)
        self.assertTrue(passing["passed"])
        self.assertEqual(passing["score"], 1.0)
        self.assertFalse(hallucinated["passed"])
        self.assertEqual(hallucinated["score"], 0.0)
        self.assertEqual(hallucinated["forbidden_claim_hits"], ["rejected"])

        unanswerable = CorpusCase(
            prompt_id="budget",
            category="unanswerable",
            evidence=("No budget appears in the packet.",),
            question="What is the budget?",
            required_facts=(),
            forbidden_claims=(FactRule("invented", ("120 thousand dollars",)),),
            should_abstain=True,
        )
        abstained = score_answer("Not provided in the evidence.", unanswerable, abstention_phrases)
        self.assertTrue(abstained["passed"])
        self.assertEqual(abstained["score"], 1.0)

    def test_warmups_are_retained_but_excluded_from_summaries(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_a = root / "a.gguf"
            artifact_b = root / "b.gguf"
            artifact_a.write_bytes(b"model a")
            artifact_b.write_bytes(b"model b")
            inventory = load_inventory(
                _write_inventory(
                    root,
                    [("light", "model-a", artifact_a), ("torch", "model-b", artifact_b)],
                )
            )
            corpus = load_corpus(_write_corpus(root))
            client = _FakeRouter(inventory.models)

            report = run_benchmark(
                inventory,
                corpus,
                client,
                rounds=2,
                warmups=2,
                sampler=NullSampler(),
            )

        self.assertEqual(len(report["warmup_trials"]), 8)
        self.assertTrue(all(item["excluded_from_summary"] for item in report["warmup_trials"]))
        self.assertEqual(len(report["trials"]), 4)
        self.assertTrue(all(not item["excluded_from_summary"] for item in report["trials"]))
        self.assertEqual([item["trial_count"] for item in report["summaries"]], [2, 2])
        self.assertEqual(
            [item.model_id for item in client.load_events],
            [
                "model-a",
                "model-b",
                "model-b",
                "model-a",
            ],
        )

    def test_route_failure_after_load_cleans_up_and_restores_initial_model(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_a = root / "a.gguf"
            artifact_b = root / "b.gguf"
            artifact_a.write_bytes(b"model a")
            artifact_b.write_bytes(b"model b")
            inventory = load_inventory(
                _write_inventory(
                    root,
                    [("light", "model-a", artifact_a), ("torch", "model-b", artifact_b)],
                )
            )
            corpus = load_corpus(_write_corpus(root))
            client = _RouteCheckFailingRouter(
                inventory.models,
                initial_loaded_model="model-b",
                fail_model_id="model-a",
            )

            with self.assertRaisesRegex(RouteVerificationError, "post-load route failure"):
                run_benchmark(
                    inventory,
                    corpus,
                    client,
                    rounds=1,
                    warmups=0,
                    sampler=NullSampler(),
                )

        self.assertEqual(client.current_loaded_model, "model-b")
        self.assertIn("model-a", client.unload_events)
        self.assertEqual([item.model_id for item in client.load_events], ["model-a", "model-b"])

    def test_router_stream_uses_deterministic_settings_and_allows_missing_token_usage(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            body = (
                'data: {"model":"model-a","choices":[{"delta":{"content":"R7"},'
                '"finish_reason":null}]}\n\n'
                'data: {"model":"model-a","choices":[{"delta":{},'
                '"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        client = RouterClient("http://router.invalid")
        client._client.close()
        client._client = httpx.Client(
            base_url="http://router.invalid", transport=httpx.MockTransport(handler)
        )
        try:
            result = client.complete(
                "model-a", [{"role": "user", "content": "code?"}], max_tokens=8, seed=17
            )
        finally:
            client.close()

        self.assertEqual(result.answer, "R7")
        self.assertEqual(result.served_model, "model-a")
        self.assertIsNone(result.output_tokens)
        self.assertEqual(result.output_tokens_source, "unavailable")
        self.assertEqual(captured["temperature"], 0.0)
        self.assertEqual(captured["top_k"], 1)
        self.assertEqual(captured["top_p"], 1.0)
        self.assertFalse(captured["cache_prompt"])
        self.assertEqual(captured["chat_template_kwargs"], {"enable_thinking": False})

    def test_stream_reports_client_interval_tps_and_retains_server_tps(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            body = (
                'data: {"model":"model-a","choices":[{"delta":{"content":"alpha "},'
                '"finish_reason":null}]}\n\n'
                'data: {"model":"model-a","choices":[{"delta":{"content":"beta"},'
                '"finish_reason":"stop"}],"usage":{"completion_tokens":3,"prompt_tokens":4},'
                '"timings":{"predicted_per_second":7.5}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                text=body,
                headers={"content-type": "text/event-stream"},
            )

        client = RouterClient("http://router.invalid")
        client._client.close()
        client._client = httpx.Client(
            base_url="http://router.invalid",
            transport=httpx.MockTransport(handler),
        )
        try:
            with patch(
                "atlas_voice.model_benchmark._perf_counter",
                side_effect=(10.0, 10.1, 10.6, 10.7),
            ):
                result = client.complete(
                    "model-a",
                    [{"role": "user", "content": "hello"}],
                    max_tokens=8,
                    seed=17,
                )
        finally:
            client.close()

        self.assertEqual(result.answer, "alpha beta")
        self.assertEqual(result.tokens_per_second, 4.0)
        self.assertEqual(result.server_tokens_per_second, 7.5)

    def test_streamed_route_mismatch_raises_and_cli_returns_nonzero(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            body = (
                'data: {"model":"wrong-model","choices":[{"delta":{"content":"x"},'
                '"finish_reason":"stop"}],"usage":{"completion_tokens":1}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        client = RouterClient("http://router.invalid")
        client._client.close()
        client._client = httpx.Client(
            base_url="http://router.invalid", transport=httpx.MockTransport(handler)
        )
        try:
            with self.assertRaisesRegex(RouteVerificationError, "route mismatch"):
                client.complete(
                    "model-a", [{"role": "user", "content": "hello"}], max_tokens=8, seed=17
                )
        finally:
            client.close()

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "model.gguf"
            artifact.write_bytes(b"model")
            inventory_path = _write_inventory(root, [("light", "model-a", artifact)])
            corpus_path = _write_corpus(root)
            output_path = root / "failure.json"
            fake_client = SimpleNamespace(close=lambda: None)
            with (
                patch("atlas_voice.model_benchmark.RouterClient", return_value=fake_client),
                patch(
                    "atlas_voice.model_benchmark.run_benchmark",
                    side_effect=RouteVerificationError("wrong route"),
                ),
            ):
                exit_code = main(
                    [
                        "--inventory",
                        str(inventory_path),
                        "--corpus",
                        str(corpus_path),
                        "--output",
                        str(output_path),
                        "--no-resource-sampling",
                    ]
                )
            failure = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 3)
        self.assertEqual(failure["status"], "route_verification_failed")


class _FakeRouter:
    def __init__(
        self,
        models: tuple[ModelArtifact, ...],
        *,
        initial_loaded_model: str | None = None,
    ):
        self.models = {model.model_id: model for model in models}
        self.load_events: list[ModelArtifact] = []
        self.unload_events: list[str] = []
        self.current_loaded_model = initial_loaded_model

    def verify_routes(self, models):
        return [self.verify_route(model) for model in models]

    def verify_route(self, model):
        return {
            "profile": model.profile,
            "model_id": model.model_id,
            "inventory_path": str(model.path),
            "router_path": str(model.path),
            "router_status": "loaded",
            "verified": True,
        }

    def snapshot_loaded_model(self):
        return self.current_loaded_model

    def restore_loaded_model(self, initial_model_id, managed_model_ids):
        del managed_model_ids
        actions = []
        if self.current_loaded_model is not None and self.current_loaded_model != initial_model_id:
            current = self.current_loaded_model
            actions.append({"action": "unload", "model_id": current})
            self.unload_model(current)
        if initial_model_id is not None and self.current_loaded_model != initial_model_id:
            actions.append({"action": "load", "model_id": initial_model_id})
            self.load_model(initial_model_id)
        return {
            "initial_loaded_model": initial_model_id,
            "final_loaded_model": self.current_loaded_model,
            "actions": actions,
            "restored": self.current_loaded_model == initial_model_id,
        }

    def load_model(self, model_id):
        self.load_events.append(self.models[model_id])
        self.current_loaded_model = model_id
        return 0.01

    def unload_model(self, model_id):
        self.unload_events.append(model_id)
        if self.current_loaded_model == model_id:
            self.current_loaded_model = None
        return 0.005

    def complete(self, model_id, messages, *, max_tokens, seed):
        del max_tokens, seed
        answer = "READY" if messages[-1]["content"] == "Reply with exactly READY." else "R7"
        return SimpleNamespace(
            answer=answer,
            served_model=model_id,
            ttft_ms=10.0,
            total_latency_ms=20.0,
            output_tokens=2,
            output_tokens_source="usage.completion_tokens",
            tokens_per_second=100.0,
            server_tokens_per_second=90.0,
            prompt_tokens=8,
            finish_reason="stop",
        )


class _RouteCheckFailingRouter(_FakeRouter):
    def __init__(
        self,
        models: tuple[ModelArtifact, ...],
        *,
        initial_loaded_model: str,
        fail_model_id: str,
    ):
        super().__init__(models, initial_loaded_model=initial_loaded_model)
        self.fail_model_id = fail_model_id

    def verify_route(self, model):
        if model.model_id == self.fail_model_id and self.current_loaded_model == self.fail_model_id:
            raise RouteVerificationError("post-load route failure")
        return super().verify_route(model)


def _write_inventory(root: Path, models: list[tuple[str, str, Path]]) -> Path:
    inventory = {
        "schema_version": 1,
        "inventory_version": "test-1",
        "models": [
            {
                "profile": profile,
                "model_id": model_id,
                "path": artifact.name,
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            }
            for profile, model_id, artifact in models
        ],
    }
    path = root / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    return path


def _write_corpus(root: Path) -> Path:
    corpus = {
        "schema_version": 1,
        "corpus_version": "test-1",
        "description": "Test corpus",
        "abstention_phrases": ["not provided in the evidence"],
        "cases": [
            {
                "id": "code",
                "category": "local_detail",
                "evidence": ["The current code is R7.", "B9 is obsolete."],
                "question": "What is the current code?",
                "required_facts": [{"id": "code", "any_of": ["R7"]}],
                "forbidden_claims": [{"id": "obsolete", "any_of": ["code is B9"]}],
                "should_abstain": False,
            }
        ],
    }
    path = root / "corpus.json"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
