from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atlas_voice.llama_bench_wrapper import (
    ArtifactVerificationError,
    BenchmarkExecutionError,
    InputValidationError,
    build_command,
    load_inventory,
    parse_and_validate_results,
    run_llama_bench,
    select_model,
    validate_output_path,
    verify_model,
)


class LlamaBenchWrapperTests(unittest.TestCase):
    def test_run_verifies_and_publishes_report_shape(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path, artifact = _write_inventory(root)
            binary = _write_binary(root)
            output = root / "run.json"
            model = select_model(load_inventory(inventory_path), "light")
            stdout = json.dumps(_rows(model.path))

            with patch(
                "atlas_voice.llama_bench_wrapper.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=stdout,
                    stderr="device diagnostics\n",
                ),
            ) as run:
                report = run_llama_bench(
                    profile="LIGHT",
                    inventory_path=inventory_path,
                    binary_path=binary,
                    output_path=output,
                )

            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                saved["artifact"],
                {
                    "path": str(artifact),
                    "sha256": _sha256(artifact),
                },
            )
            self.assertEqual(len(saved["results"]), 2)
            self.assertEqual(saved["results"][0]["repetitions"], 5)
            self.assertEqual(saved["provenance"]["protocol"]["observed_threads"], 12)
            self.assertEqual(saved["provenance"]["binary"]["build_commit"], "abc123")
            self.assertEqual(report["artifact"], saved["artifact"])
            command = run.call_args.args[0]
            self.assertNotIn("--threads", command)
            self.assertEqual(command, build_command(binary, model))
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_artifact_size_and_sha_are_verified(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path, artifact = _write_inventory(root)
            model = select_model(load_inventory(inventory_path), "light")
            verify_model(model)
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ArtifactVerificationError, "size mismatch"):
                verify_model(model)

    def test_output_rejects_model_inventory_existing_and_symlink_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path, artifact = _write_inventory(root)
            with self.assertRaisesRegex(InputValidationError, "collides"):
                validate_output_path(inventory_path, protected_paths=[inventory_path, artifact])
            with self.assertRaisesRegex(InputValidationError, "collides"):
                validate_output_path(artifact, protected_paths=[inventory_path, artifact])

            existing = root / "existing.json"
            existing.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(InputValidationError, "overwrite"):
                validate_output_path(existing, protected_paths=[inventory_path, artifact])

            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(InputValidationError, "symlink"):
                validate_output_path(linked / "result.json", protected_paths=[])

    def test_rows_require_exact_model_and_consistent_protocol(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path, _ = _write_inventory(root)
            model = select_model(load_inventory(inventory_path), "light")
            rows = _rows(model.path)
            parse_and_validate_results(json.dumps(rows), model)

            wrong_path = _rows(model.path)
            wrong_path[1]["model_filename"] = model.path.name
            with self.assertRaisesRegex(BenchmarkExecutionError, "model_filename mismatch"):
                parse_and_validate_results(json.dumps(wrong_path), model)

            wrong_threads = _rows(model.path)
            wrong_threads[1]["n_threads"] = 8
            with self.assertRaisesRegex(BenchmarkExecutionError, "inconsistent"):
                parse_and_validate_results(json.dumps(wrong_threads), model)

            wrong_test = _rows(model.path)
            wrong_test[1]["n_gen"] = 127
            with self.assertRaisesRegex(BenchmarkExecutionError, "test rows mismatch"):
                parse_and_validate_results(json.dumps(wrong_test), model)

            wrong_samples = _rows(model.path)
            wrong_samples[1]["samples_ts"] = [24.0] * 4
            with self.assertRaisesRegex(BenchmarkExecutionError, "exactly 5 samples"):
                parse_and_validate_results(json.dumps(wrong_samples), model)

    def test_nonzero_exit_does_not_publish_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path, _ = _write_inventory(root)
            binary = _write_binary(root)
            output = root / "run.json"
            with patch(
                "atlas_voice.llama_bench_wrapper.subprocess.run",
                return_value=SimpleNamespace(returncode=7, stdout="", stderr="GPU error"),
            ):
                with self.assertRaisesRegex(BenchmarkExecutionError, "status 7"):
                    run_llama_bench(
                        profile="light",
                        inventory_path=inventory_path,
                        binary_path=binary,
                        output_path=output,
                    )
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_midrun_artifact_change_is_detected_without_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path, artifact = _write_inventory(root)
            binary = _write_binary(root)
            output = root / "run.json"
            model = select_model(load_inventory(inventory_path), "light")

            def mutate(*args: object, **kwargs: object) -> SimpleNamespace:
                artifact.write_bytes(b"mutated")
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(_rows(model.path)),
                    stderr="",
                )

            with patch("atlas_voice.llama_bench_wrapper.subprocess.run", side_effect=mutate):
                with self.assertRaises(ArtifactVerificationError):
                    run_llama_bench(
                        profile="light",
                        inventory_path=inventory_path,
                        binary_path=binary,
                        output_path=output,
                    )
            self.assertFalse(output.exists())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inventory(root: Path) -> tuple[Path, Path]:
    artifact = root / "model.gguf"
    artifact.write_bytes(b"pinned model")
    inventory = root / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inventory_version": "fixture-v1",
                "models": [
                    {
                        "profile": "light",
                        "model_id": "model-a",
                        "path": artifact.name,
                        "size_bytes": artifact.stat().st_size,
                        "sha256": _sha256(artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return inventory, artifact


def _write_binary(root: Path) -> Path:
    binary = root / "llama-bench"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o700)
    return binary


def _rows(model_path: Path) -> list[dict[str, object]]:
    common: dict[str, object] = {
        "build_number": 9913,
        "build_commit": "abc123",
        "model_filename": str(model_path),
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
    }
    return [
        {
            **common,
            "n_prompt": 512,
            "n_gen": 0,
            "avg_ts": 1024.0,
            "samples_ns": [500_000_000] * 5,
            "samples_ts": [1024.0] * 5,
        },
        {
            **common,
            "n_prompt": 0,
            "n_gen": 128,
            "avg_ts": 24.0,
            "samples_ns": [5_333_333_333] * 5,
            "samples_ts": [24.0] * 5,
        },
    ]


if __name__ == "__main__":
    unittest.main()
