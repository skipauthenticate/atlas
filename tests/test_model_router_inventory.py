from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from atlas_voice.model_benchmark import (
    ModelArtifact,
    RouteVerificationError,
    RouterClient,
)


class ModelRouterInventoryTests(unittest.TestCase):
    def test_unloaded_preset_path_is_verified_from_status_args(self) -> None:
        model = ModelArtifact(
            profile="light",
            model_id="qwen3.5-2b",
            path=Path("/models/light.gguf"),
            sha256="a" * 64,
        )
        client = RouterClient("http://router.invalid")
        try:
            result = client._verify_route_from_entries(
                model,
                [
                    {
                        "id": model.model_id,
                        "path": None,
                        "status": {
                            "value": "unloaded",
                            "args": ["llama-server", "--model", str(model.path)],
                        },
                    }
                ],
            )
        finally:
            client.close()

        self.assertTrue(result["verified"])
        self.assertEqual(result["router_path"], str(model.path))
        self.assertEqual(result["router_path_source"], "status.args")

    def test_router_inventory_rejects_unexpected_default_preset(self) -> None:
        model = ModelArtifact(
            profile="light",
            model_id="qwen3.5-2b",
            path=Path("/models/light.gguf"),
            sha256="a" * 64,
        )
        entries = [
            {"id": model.model_id, "path": str(model.path), "status": {"value": "unloaded"}},
            {"id": "default", "path": None, "status": {"value": "unloaded"}},
        ]
        client = RouterClient("http://router.invalid")
        try:
            with patch.object(client, "list_models", return_value=entries):
                with self.assertRaisesRegex(RouteVerificationError, "inventory mismatch"):
                    client.verify_routes((model,))
        finally:
            client.close()

    def test_router_inventory_rejects_duplicate_expected_id(self) -> None:
        model = ModelArtifact(
            profile="light",
            model_id="qwen3.5-2b",
            path=Path("/models/light.gguf"),
            sha256="a" * 64,
        )
        entry = {
            "id": model.model_id,
            "path": str(model.path),
            "status": {
                "value": "unloaded",
                "args": ["llama-server", "--model", str(model.path)],
            },
        }
        client = RouterClient("http://router.invalid")
        try:
            with patch.object(client, "list_models", return_value=[entry, dict(entry)]):
                with self.assertRaisesRegex(RouteVerificationError, "counts"):
                    client.verify_routes((model,))
        finally:
            client.close()

    def test_route_rejects_multiple_or_relative_model_flags(self) -> None:
        model = ModelArtifact(
            profile="light",
            model_id="qwen3.5-2b",
            path=Path("/models/light.gguf"),
            sha256="a" * 64,
        )
        client = RouterClient("http://router.invalid")
        try:
            with self.assertRaisesRegex(RouteVerificationError, "exactly one"):
                client._verify_route_from_entries(
                    model,
                    [
                        {
                            "id": model.model_id,
                            "path": str(model.path),
                            "status": {
                                "value": "unloaded",
                                "args": [
                                    "llama-server",
                                    "--model",
                                    "/models/wrong.gguf",
                                    "--model",
                                    str(model.path),
                                ],
                            },
                        }
                    ],
                )
            with self.assertRaisesRegex(RouteVerificationError, "must be absolute"):
                client._verify_route_from_entries(
                    model,
                    [
                        {
                            "id": model.model_id,
                            "path": None,
                            "status": {
                                "value": "unloaded",
                                "args": ["llama-server", "--model", "models/light.gguf"],
                            },
                        }
                    ],
                )
        finally:
            client.close()

    def test_route_rejects_disagreement_between_path_and_status_args(self) -> None:
        model = ModelArtifact(
            profile="light",
            model_id="qwen3.5-2b",
            path=Path("/models/light.gguf"),
            sha256="a" * 64,
        )
        client = RouterClient("http://router.invalid")
        try:
            with self.assertRaisesRegex(RouteVerificationError, "disagree"):
                client._verify_route_from_entries(
                    model,
                    [
                        {
                            "id": model.model_id,
                            "path": str(model.path),
                            "status": {
                                "value": "unloaded",
                                "args": ["llama-server", "--model", "/models/wrong.gguf"],
                            },
                        }
                    ],
                )
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
