from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from atlas_voice.model_installer import (
    InstallerError,
    _partial_path,
    _publish_partial,
    build_aria2_command,
    install_models,
    inspect_artifact,
    load_inventory,
    sha256_file_stable,
    verify_models,
)


class ModelInstallerTests(unittest.TestCase):
    def test_load_inventory_derives_immutable_urls_and_provenance(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = _load_inventory(root)

        self.assertEqual([model.profile for model in inventory.models], ["light", "torch", "fire"])
        self.assertEqual(inventory.total_size_bytes, 15)
        self.assertIn(
            "/resolve/" + "1" * 40 + "/light.gguf",
            inventory.models[0].download_url,
        )
        self.assertEqual(inventory.models[0].download_url.count("?"), 1)
        self.assertNotIn("blob", inventory.models[0].download_url)
        self.assertEqual(inventory.models[0].provenance.exact_upstream_lineage, "undeclared")
        self.assertIn(
            "does not declare",
            inventory.models[0].provenance.lineage_limitation or "",
        )

    def test_load_inventory_rejects_floating_or_mismatched_revision(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_inventory(root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["models"][0]["source_url"] = (
                "https://huggingface.co/example/light/blob/main/light.gguf"
            )
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(InstallerError, "revision"):
                load_inventory(path, allowed_models_root=root / "models")

    def test_load_inventory_rejects_destination_escape(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_inventory(root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["models"][0]["path"] = "../escaped/light.gguf"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(InstallerError, "escapes allowed models root"):
                load_inventory(path, allowed_models_root=root / "models")

    def test_load_inventory_rejects_symlinked_allowed_root(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.mkdir()
            (root / "models").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(InstallerError, "symlinked allowed models root"):
                load_inventory(_write_inventory(root), allowed_models_root=root / "models")

    def test_load_inventory_requires_limitation_for_undeclared_lineage(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_inventory(root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["models"][0]["provenance"]["lineage_limitation"]
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(InstallerError, "lineage_limitation is required"):
                load_inventory(path, allowed_models_root=root / "models")

    def test_command_uses_unique_partial_checksum_and_secure_aria_flags(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = _load_inventory(root).models[0]
            command = build_aria2_command("/usr/bin/aria2c", model, connections=4)

        self.assertIn("--no-conf=true", command)
        self.assertIn("--check-certificate=true", command)
        self.assertIn("--continue=true", command)
        self.assertIn(f"--checksum=sha-256={model.sha256}", command)
        self.assertIn("--max-connection-per-server=4", command)
        output = _command_value(command, "--out=")
        self.assertNotEqual(output, model.path.name)
        self.assertIn(model.sha256[:16], output)
        self.assertTrue(output.endswith(".partial"))
        self.assertEqual(command[-1], model.download_url)

    def test_install_is_sequential_atomic_read_only_and_records_observed_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = _contents()
            inventory = _load_inventory(root, contents=contents)
            calls: list[str] = []

            def fake_aria(command, **_kwargs):
                if _is_version(command):
                    return _version_result(command)
                filename = _write_completed_download(command, contents)
                calls.append(filename)
                current_model = next(
                    model for model in inventory.models if model.path.name == filename
                )
                self.assertFalse(current_model.path.exists())
                return subprocess.CompletedProcess(command, 0)

            summary = install_models(
                inventory,
                aria2c="/bin/true",
                safety_margin_bytes=8,
                disk_free_bytes=1024,
                command_runner=fake_aria,
            )

            self.assertEqual(calls, ["light.gguf", "torch.gguf", "fire.gguf"])
            self.assertEqual(summary["status"], "verified")
            self.assertEqual(summary["download_tool"]["version"], "1.37.0")
            self.assertEqual([item["status"] for item in summary["models"]], ["verified"] * 3)
            for item, model in zip(summary["models"], inventory.models):
                self.assertEqual(item["observed_sha256"], model.sha256)
                self.assertEqual(item["observed_size_bytes"], model.size_bytes)
                self.assertTrue(item["read_only"])
                self.assertEqual(stat.S_IMODE(model.path.stat().st_mode), 0o444)
                self.assertFalse(_partial_path(model).exists())
            self.assertNotIn("source_url", summary["models"][0])

    def test_failed_multi_connection_download_resumes_with_one_connection(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = _contents()
            inventory = _load_inventory(root, contents=contents)
            failed_once = False

            def fake_aria(command, **_kwargs):
                nonlocal failed_once
                if _is_version(command):
                    return _version_result(command)
                filename = _filename_for_command(command, contents)
                output = _output_path(command)
                connections = int(_command_value(command, "--max-connection-per-server="))
                if filename == "light.gguf" and connections == 4 and not failed_once:
                    failed_once = True
                    output.write_bytes(b"li")
                    Path(f"{output}.aria2").write_bytes(b"resume")
                    return subprocess.CompletedProcess(command, 22)
                _write_completed_download(command, contents)
                return subprocess.CompletedProcess(command, 0)

            summary = install_models(
                inventory,
                aria2c="/bin/true",
                safety_margin_bytes=0,
                disk_free_bytes=1024,
                command_runner=fake_aria,
            )

        light_attempts = summary["download_attempts"][0]["attempts"]
        self.assertEqual(
            light_attempts,
            [
                {"connections": 4, "exit_status": 22},
                {"connections": 1, "exit_status": 0},
            ],
        )

    def test_existing_hash_mismatch_is_refused_before_download(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = _load_inventory(root)
            inventory.model_directory.mkdir(parents=True)
            inventory.models[0].path.write_bytes(b"wrong")
            downloads = 0

            def should_not_download(command, **_kwargs):
                nonlocal downloads
                if _is_version(command):
                    return _version_result(command)
                downloads += 1
                return subprocess.CompletedProcess(command, 0)

            with self.assertRaisesRegex(InstallerError, "SHA-256 mismatch"):
                install_models(
                    inventory,
                    aria2c="/bin/true",
                    disk_free_bytes=1024,
                    command_runner=should_not_download,
                )

        self.assertEqual(downloads, 0)

    def test_preflight_refuses_insufficient_space_without_downloading(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = _load_inventory(root)
            with self.assertRaisesRegex(InstallerError, "insufficient free disk space"):
                install_models(
                    inventory,
                    aria2c="/bin/true",
                    safety_margin_bytes=10,
                    disk_free_bytes=20,
                    preflight_only=True,
                    command_runner=_version_only_runner,
                )

    def test_disk_capacity_is_rechecked_immediately_before_download(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = _load_inventory(root)
            free_values = iter((1024, 20))
            calls = 0

            def free_provider(_path: Path) -> int:
                nonlocal calls
                calls += 1
                return next(free_values)

            with self.assertRaisesRegex(InstallerError, "before light primary download"):
                install_models(
                    inventory,
                    aria2c="/bin/true",
                    safety_margin_bytes=8,
                    disk_free_provider=free_provider,
                    command_runner=_version_only_runner,
                )

        self.assertEqual(calls, 2)

    def test_sparse_partial_uses_allocated_bytes_for_capacity(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = {
                "light.gguf": b"x" * (8 * 1024 * 1024),
                "torch.gguf": b"torch",
                "fire.gguf": b"fire!",
            }
            inventory = _load_inventory(root, contents=contents)
            model = inventory.models[0]
            model.path.parent.mkdir(parents=True)
            partial = _partial_path(model)
            with partial.open("wb") as handle:
                handle.truncate(model.size_bytes)
            Path(f"{partial}.aria2").write_bytes(b"resume metadata")

            state = inspect_artifact(model)

        self.assertEqual(state.status, "partial")
        self.assertEqual(state.current_size_bytes, model.size_bytes)
        self.assertLess(state.allocated_size_bytes, model.size_bytes)
        self.assertEqual(
            state.remaining_bytes,
            model.size_bytes - state.allocated_size_bytes,
        )

    def test_verify_only_requires_exact_read_only_artifacts_and_observed_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = _contents()
            inventory = _load_inventory(root, contents=contents)
            inventory.model_directory.mkdir(parents=True)
            for model in inventory.models:
                model.path.write_bytes(contents[model.path.name])
                model.path.chmod(0o444)

            summary = verify_models(inventory)

        self.assertEqual(summary["operation"], "verify")
        self.assertEqual(summary["artifact_count"], 3)
        self.assertEqual(
            [item["observed_sha256"] for item in summary["models"]],
            [model.sha256 for model in inventory.models],
        )

    def test_verify_only_rejects_writable_final_artifacts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = _contents()
            inventory = _load_inventory(root, contents=contents)
            inventory.model_directory.mkdir(parents=True)
            for model in inventory.models:
                model.path.write_bytes(contents[model.path.name])
                model.path.chmod(0o444)
            inventory.models[1].path.chmod(0o644)

            with self.assertRaisesRegex(InstallerError, "artifacts are writable"):
                verify_models(inventory)

    def test_symlinked_lock_is_refused_without_following(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = _load_inventory(root)
            inventory.model_directory.mkdir(parents=True)
            outside = root / "outside.lock"
            outside.write_text("do not touch", encoding="utf-8")
            (inventory.model_directory / ".install-voice-models.lock").symlink_to(outside)

            with self.assertRaisesRegex(InstallerError, "symlinked .*lock"):
                install_models(
                    inventory,
                    aria2c="/bin/true",
                    disk_free_bytes=1024,
                    command_runner=_version_only_runner,
                )

            self.assertEqual(outside.read_text(encoding="utf-8"), "do not touch")

    def test_symlinked_partial_is_refused(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = _load_inventory(root)
            inventory.model_directory.mkdir(parents=True)
            outside = root / "outside.gguf"
            outside.write_bytes(b"outside")
            _partial_path(inventory.models[0]).symlink_to(outside)

            with self.assertRaisesRegex(InstallerError, "symlinked partial"):
                inspect_artifact(inventory.models[0])

    def test_publish_is_no_clobber_when_final_target_appears_during_race(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = _contents()
            inventory = _load_inventory(root, contents=contents)
            inventory.model_directory.mkdir(parents=True)
            model = inventory.models[0]
            partial = _partial_path(model)
            partial.write_bytes(contents[model.path.name])
            racer_bytes = b"independent target"
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
                    0o444,
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

            with (
                patch("atlas_voice.model_installer.os.link", side_effect=racing_link),
                self.assertRaisesRegex(InstallerError, "appeared during publication"),
            ):
                _publish_partial(model)

            self.assertEqual(model.path.read_bytes(), racer_bytes)
            self.assertEqual(partial.read_bytes(), contents[model.path.name])

    def test_final_verification_rehashes_every_published_artifact(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = _contents()
            inventory = _load_inventory(root, contents=contents)
            hashed_paths: list[Path] = []

            def recording_hash(path: Path, *, chunk_size: int = 8 * 1024 * 1024):
                hashed_paths.append(path)
                return sha256_file_stable(path, chunk_size=chunk_size)

            def fake_aria(command, **_kwargs):
                if _is_version(command):
                    return _version_result(command)
                _write_completed_download(command, contents)
                return subprocess.CompletedProcess(command, 0)

            with patch(
                "atlas_voice.model_installer.sha256_file_stable",
                side_effect=recording_hash,
            ):
                install_models(
                    inventory,
                    aria2c="/bin/true",
                    safety_margin_bytes=0,
                    disk_free_bytes=1024,
                    command_runner=fake_aria,
                )

            counts = Counter(hashed_paths)
            for model in inventory.models:
                self.assertGreaterEqual(counts[model.path], 2)
                self.assertGreaterEqual(counts[_partial_path(model)], 1)

    def test_unrecognized_aria_version_is_refused(self) -> None:
        with TemporaryDirectory() as temporary:
            inventory = _load_inventory(Path(temporary))

            def bad_version(command, **_kwargs):
                return subprocess.CompletedProcess(command, 0, stdout="unexpected tool", stderr="")

            with self.assertRaisesRegex(InstallerError, "expected 'aria2 version"):
                install_models(
                    inventory,
                    aria2c="/bin/true",
                    disk_free_bytes=1024,
                    command_runner=bad_version,
                )


def _contents() -> dict[str, bytes]:
    return {
        "light.gguf": b"light",
        "torch.gguf": b"torch",
        "fire.gguf": b"fire!",
    }


def _load_inventory(root: Path, *, contents: dict[str, bytes] | None = None):
    return load_inventory(
        _write_inventory(root, contents=contents),
        allowed_models_root=root / "models",
    )


def _write_inventory(root: Path, *, contents: dict[str, bytes] | None = None) -> Path:
    contents = contents or _contents()
    models = []
    for index, (profile, filename) in enumerate(
        (("light", "light.gguf"), ("torch", "torch.gguf"), ("fire", "fire.gguf")),
        start=1,
    ):
        payload = contents[filename]
        revision = str(index) * 40
        models.append(
            {
                "profile": profile,
                "model_id": f"model-{profile}",
                "path": f"models/{filename}",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_url": (
                    f"https://huggingface.co/example/{profile}/blob/{revision}/{filename}"
                ),
                "source_revision": revision,
                "provenance": {
                    "artifact_repository": f"example/{profile}",
                    "conversion_kind": "community_gguf_conversion",
                    "upstream_model": f"upstream/{profile}",
                    "upstream_reference_revision": "a" * 40,
                    "exact_upstream_lineage": "undeclared",
                    "lineage_limitation": (
                        "The publisher does not declare the exact upstream commit."
                    ),
                },
            }
        )
    path = root / "inventory.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inventory_version": "test-inventory",
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    return path


def _is_version(command: list[str]) -> bool:
    return len(command) == 2 and command[1] == "--version"


def _version_result(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout="aria2 version 1.37.0\n", stderr="")


def _version_only_runner(command, **_kwargs):
    if _is_version(command):
        return _version_result(command)
    raise AssertionError(f"unexpected download command: {command}")


def _command_value(command: list[str], prefix: str) -> str:
    return next(value.split("=", 1)[1] for value in command if value.startswith(prefix))


def _output_path(command: list[str]) -> Path:
    return Path(
        _command_value(command, "--dir="),
        _command_value(command, "--out="),
    )


def _filename_for_command(command: list[str], contents: dict[str, bytes]) -> str:
    output_name = _command_value(command, "--out=")
    return next(filename for filename in contents if filename in output_name)


def _write_completed_download(command: list[str], contents: dict[str, bytes]) -> str:
    filename = _filename_for_command(command, contents)
    output = _output_path(command)
    output.write_bytes(contents[filename])
    control = Path(f"{output}.aria2")
    if control.exists():
        control.unlink()
    return filename


if __name__ == "__main__":
    unittest.main()
