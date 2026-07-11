from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import unquote, urlsplit, urlunsplit


SCHEMA_VERSION = 1
EXPECTED_PROFILES = frozenset({"light", "torch", "fire"})
LINEAGE_STATUSES = frozenset({"declared", "undeclared"})
DEFAULT_SAFETY_MARGIN_BYTES = 8 * 1024**3
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY_PATH = REPOSITORY_ROOT / "benchmarks" / "voice_models_q8_inventory.json"
DEFAULT_MODELS_ROOT = REPOSITORY_ROOT / "models" / "llm" / "voice"


class InstallerError(RuntimeError):
    """An expected, user-actionable installer failure."""


@dataclass(frozen=True)
class ModelProvenance:
    artifact_repository: str
    conversion_kind: str
    upstream_model: str
    upstream_reference_revision: str
    exact_upstream_lineage: str
    lineage_limitation: str | None


@dataclass(frozen=True)
class ModelSpec:
    profile: str
    model_id: str
    path: Path
    size_bytes: int
    sha256: str
    download_url: str
    source_revision: str
    allowed_models_root: Path
    provenance: ModelProvenance


@dataclass(frozen=True)
class Inventory:
    source_path: Path
    source_sha256: str
    inventory_version: str
    model_directory: Path
    allowed_models_root: Path
    models: tuple[ModelSpec, ...]

    @property
    def total_size_bytes(self) -> int:
        return sum(model.size_bytes for model in self.models)


@dataclass(frozen=True)
class ArtifactState:
    model: ModelSpec
    status: str
    current_size_bytes: int
    allocated_size_bytes: int
    observed_sha256: str | None = None
    read_only: bool | None = None

    @property
    def remaining_bytes(self) -> int:
        if self.status == "verified":
            return 0
        return self.model.size_bytes - self.allocated_size_bytes


CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]
DiskFreeProvider = Callable[[Path], int]


def load_inventory(path: Path, *, allowed_models_root: Path | None = None) -> Inventory:
    source_path = path.expanduser().resolve()
    root = _absolute_without_symlink_resolution(
        allowed_models_root if allowed_models_root is not None else DEFAULT_MODELS_ROOT
    )
    _reject_symlink_components(root, label="allowed models root")
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise InstallerError(f"cannot read inventory {source_path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InstallerError(f"inventory is not valid UTF-8 JSON: {source_path}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise InstallerError(f"inventory schema_version must be {SCHEMA_VERSION}")
    inventory_version = _required_string(payload, "inventory_version", "inventory")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or len(raw_models) != len(EXPECTED_PROFILES):
        raise InstallerError("inventory must contain exactly the light, torch, and fire models")

    models: list[ModelSpec] = []
    seen_profiles: set[str] = set()
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, item in enumerate(raw_models):
        label = f"models[{index}]"
        if not isinstance(item, dict):
            raise InstallerError(f"{label} must be an object")
        profile = _required_string(item, "profile", label).strip().lower()
        model_id = _required_string(item, "model_id", label).strip()
        path_value = _required_string(item, "path", label)
        size_bytes = item.get("size_bytes")
        sha256 = _required_string(item, "sha256", label).strip().lower()
        source_url = _required_string(item, "source_url", label).strip()
        source_revision = _required_string(item, "source_revision", label).strip().lower()
        provenance = _parse_provenance(item.get("provenance"), label=label)

        if profile in seen_profiles:
            raise InstallerError(f"duplicate inventory profile: {profile}")
        if model_id in seen_ids:
            raise InstallerError(f"duplicate inventory model_id: {model_id}")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise InstallerError(f"size_bytes must be a positive integer for {model_id}")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise InstallerError(f"invalid SHA-256 for {model_id}")
        if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
            raise InstallerError(
                f"source_revision must be an immutable 40-hex commit for {model_id}"
            )

        artifact_path = Path(path_value).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = source_path.parent / artifact_path
        artifact_path = _absolute_without_symlink_resolution(artifact_path)
        _validate_destination_path(artifact_path, root, label=f"artifact path for {model_id}")
        if artifact_path in seen_paths:
            raise InstallerError(f"duplicate artifact path: {artifact_path}")
        download_url, source_filename, source_repository = _immutable_download_url(
            source_url, expected_revision=source_revision, model_id=model_id
        )
        if provenance.artifact_repository != source_repository:
            raise InstallerError(
                f"{label}.provenance.artifact_repository does not match source_url"
            )
        if unquote(source_filename) != artifact_path.name:
            raise InstallerError(
                f"source filename does not match artifact path for {model_id}: "
                f"{unquote(source_filename)!r} != {artifact_path.name!r}"
            )

        seen_profiles.add(profile)
        seen_ids.add(model_id)
        seen_paths.add(artifact_path)
        models.append(
            ModelSpec(
                profile=profile,
                model_id=model_id,
                path=artifact_path,
                size_bytes=size_bytes,
                sha256=sha256,
                download_url=download_url,
                source_revision=source_revision,
                allowed_models_root=root,
                provenance=provenance,
            )
        )

    if seen_profiles != EXPECTED_PROFILES:
        missing = sorted(EXPECTED_PROFILES - seen_profiles)
        unexpected = sorted(seen_profiles - EXPECTED_PROFILES)
        raise InstallerError(
            "inventory profiles must be light, torch, and fire; "
            f"missing={missing}, unexpected={unexpected}"
        )
    model_directories = {model.path.parent for model in models}
    if len(model_directories) != 1:
        raise InstallerError("all candidate artifacts must share one model directory")

    return Inventory(
        source_path=source_path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        inventory_version=inventory_version,
        model_directory=next(iter(model_directories)),
        allowed_models_root=root,
        models=tuple(models),
    )


def inspect_artifact(model: ModelSpec) -> ArtifactState:
    _validate_destination_path(
        model.path, model.allowed_models_root, label=f"artifact path for {model.model_id}"
    )
    partial_path = _partial_path(model)
    final_control_path = Path(f"{model.path}.aria2")
    partial_control_path = Path(f"{partial_path}.aria2")
    for candidate, label in (
        (model.path, "final artifact"),
        (final_control_path, "legacy aria2 control file"),
        (partial_path, "partial artifact"),
        (partial_control_path, "partial aria2 control file"),
    ):
        _reject_symlink_components(candidate, label=label)

    if final_control_path.exists():
        _regular_file_stat(final_control_path, label="legacy aria2 control file")
        raise InstallerError(
            "refusing legacy aria2 state at the final artifact path; "
            f"move it aside before retrying: {final_control_path}"
        )

    if model.path.exists():
        if partial_path.exists() or partial_control_path.exists():
            raise InstallerError(
                f"final and partial artifacts coexist for {model.model_id}; "
                "move the partial state aside before retrying"
            )
        file_stat = _regular_file_stat(model.path, label="artifact")
        if file_stat.st_size != model.size_bytes:
            raise InstallerError(
                f"artifact size mismatch for {model.model_id}: "
                f"expected {model.size_bytes}, got {file_stat.st_size}"
            )
        actual_sha256 = sha256_file_stable(model.path)
        if not hmac.compare_digest(actual_sha256, model.sha256):
            raise InstallerError(
                f"artifact SHA-256 mismatch for {model.model_id}: "
                f"expected {model.sha256}, got {actual_sha256}; move it aside and retry"
            )
        return ArtifactState(
            model=model,
            status="verified",
            current_size_bytes=file_stat.st_size,
            allocated_size_bytes=_allocated_bytes(file_stat, model.size_bytes),
            observed_sha256=actual_sha256,
            read_only=(stat.S_IMODE(file_stat.st_mode) & 0o222) == 0,
        )

    if partial_control_path.exists() and not partial_path.exists():
        _regular_file_stat(partial_control_path, label="aria2 control file")
        raise InstallerError(
            f"orphaned aria2 control file without partial artifact: {partial_control_path}"
        )
    if not partial_path.exists():
        return ArtifactState(
            model=model,
            status="missing",
            current_size_bytes=0,
            allocated_size_bytes=0,
        )

    partial_stat = _regular_file_stat(partial_path, label="partial artifact")
    if partial_control_path.exists():
        _regular_file_stat(partial_control_path, label="aria2 control file")
    elif partial_stat.st_size != model.size_bytes:
        raise InstallerError(
            f"incomplete partial artifact has no aria2 resume file for {model.model_id}: "
            f"expected {model.size_bytes}, got {partial_stat.st_size}; move it aside and retry"
        )
    if partial_stat.st_size > model.size_bytes:
        raise InstallerError(
            f"partial artifact is larger than pinned size for {model.model_id}: "
            f"expected {model.size_bytes}, got {partial_stat.st_size}"
        )
    return ArtifactState(
        model=model,
        status="partial",
        current_size_bytes=partial_stat.st_size,
        allocated_size_bytes=_allocated_bytes(partial_stat, model.size_bytes),
    )


def sha256_file_stable(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise InstallerError("hash chunk size must be positive")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise InstallerError("this platform cannot enforce no-follow artifact reads")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InstallerError(
            f"cannot open artifact without following symlinks: {path}: {exc}"
        ) from exc

    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise InstallerError(f"artifact path is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    fingerprint_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    fingerprint_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if fingerprint_before != fingerprint_after:
        raise InstallerError(f"artifact changed while hashing: {path}")
    try:
        current = os.lstat(path)
    except OSError as exc:
        raise InstallerError(f"artifact disappeared while hashing: {path}") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != after.st_dev
        or current.st_ino != after.st_ino
    ):
        raise InstallerError(f"artifact path changed while hashing: {path}")
    return digest.hexdigest()


def build_aria2_command(aria2c: str, model: ModelSpec, *, connections: int = 4) -> list[str]:
    if not 1 <= connections <= 8:
        raise InstallerError("aria2 connection count must be between 1 and 8")
    partial_path = _partial_path(model)
    return [
        aria2c,
        "--no-conf=true",
        "--check-certificate=true",
        "--continue=true",
        "--allow-overwrite=false",
        "--auto-file-renaming=false",
        "--check-integrity=true",
        "--file-allocation=none",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        "--min-split-size=64M",
        "--max-tries=5",
        "--retry-wait=5",
        "--connect-timeout=30",
        "--timeout=60",
        "--summary-interval=10",
        f"--checksum=sha-256={model.sha256}",
        f"--dir={partial_path.parent}",
        f"--out={partial_path.name}",
        model.download_url,
    ]


def install_models(
    inventory: Inventory,
    *,
    aria2c: str | None = None,
    connections: int = 4,
    safety_margin_bytes: int = DEFAULT_SAFETY_MARGIN_BYTES,
    preflight_only: bool = False,
    command_runner: CommandRunner = subprocess.run,
    disk_free_bytes: int | None = None,
    disk_free_provider: DiskFreeProvider | None = None,
) -> dict[str, Any]:
    if safety_margin_bytes < 0:
        raise InstallerError("safety margin cannot be negative")
    if not 1 <= connections <= 8:
        raise InstallerError("aria2 connection count must be between 1 and 8")
    if disk_free_bytes is not None and disk_free_provider is not None:
        raise InstallerError("provide either disk_free_bytes or disk_free_provider, not both")

    _prepare_inventory_directory(inventory, create=True)
    executable = _resolve_aria2c(aria2c)
    download_tool = _capture_aria2_provenance(executable, command_runner)
    capacity_checks: list[dict[str, Any]] = []
    downloads: list[dict[str, Any]] = []

    with _installer_lock(inventory):
        states = [inspect_artifact(model) for model in inventory.models]
        required_bytes = sum(state.remaining_bytes for state in states)
        free_bytes = _check_disk_capacity(
            inventory,
            states,
            safety_margin_bytes=safety_margin_bytes,
            disk_free_bytes=disk_free_bytes,
            disk_free_provider=disk_free_provider,
            stage="preflight",
            checks=capacity_checks,
        )

        if preflight_only:
            return _summary(
                inventory,
                states,
                operation="preflight",
                status="ready",
                required_download_bytes=required_bytes,
                disk_free_bytes=free_bytes,
                safety_margin_bytes=safety_margin_bytes,
                download_tool=download_tool,
                downloads=downloads,
                capacity_checks=capacity_checks,
            )

        for index, state in enumerate(states):
            if state.status == "verified":
                _make_read_only(state.model.path)
                continue

            partial_path = _partial_path(state.model)
            partial_control = _partial_control_path(state.model)
            if (
                state.status == "partial"
                and state.current_size_bytes == state.model.size_bytes
                and not partial_control.exists()
            ):
                states[index] = _publish_partial(state.model)
                continue

            _check_disk_capacity(
                inventory,
                states[index:],
                safety_margin_bytes=safety_margin_bytes,
                disk_free_bytes=disk_free_bytes,
                disk_free_provider=disk_free_provider,
                stage=f"before {state.model.profile} primary download",
                checks=capacity_checks,
            )
            attempts: list[dict[str, int]] = []
            result = _run_aria2(
                build_aria2_command(executable, state.model, connections=connections),
                command_runner,
                profile=state.model.profile,
            )
            attempts.append({"connections": connections, "exit_status": result.returncode})

            if result.returncode != 0 and connections > 1:
                resumed = inspect_artifact(state.model)
                states[index] = resumed
                _check_disk_capacity(
                    inventory,
                    states[index:],
                    safety_margin_bytes=safety_margin_bytes,
                    disk_free_bytes=disk_free_bytes,
                    disk_free_provider=disk_free_provider,
                    stage=f"before {state.model.profile} single-connection fallback",
                    checks=capacity_checks,
                )
                result = _run_aria2(
                    build_aria2_command(executable, state.model, connections=1),
                    command_runner,
                    profile=state.model.profile,
                )
                attempts.append({"connections": 1, "exit_status": result.returncode})

            downloads.append(
                {
                    "profile": state.model.profile,
                    "model_id": state.model.model_id,
                    "partial_path": str(partial_path),
                    "attempts": attempts,
                }
            )
            if result.returncode != 0:
                statuses = ", ".join(
                    f"{attempt['connections']} connection(s): {attempt['exit_status']}"
                    for attempt in attempts
                )
                raise InstallerError(
                    f"aria2c failed for {state.model.profile}/{state.model.model_id} "
                    f"({statuses}); resumable state remains at {partial_path}"
                )
            states[index] = _publish_partial(state.model)

        for model in inventory.models:
            _make_read_only(model.path)

        # Deliberately discard all prior hash results and rehash every final artifact.
        final_states = [inspect_artifact(model) for model in inventory.models]
        incomplete = [state for state in final_states if state.status != "verified"]
        writable = [state for state in final_states if not state.read_only]
        if incomplete or writable:
            raise InstallerError(
                "final artifact verification did not produce three read-only files"
            )

        return _summary(
            inventory,
            final_states,
            operation="install",
            status="verified",
            required_download_bytes=required_bytes,
            disk_free_bytes=free_bytes,
            safety_margin_bytes=safety_margin_bytes,
            download_tool=download_tool,
            downloads=downloads,
            capacity_checks=capacity_checks,
        )


def verify_models(inventory: Inventory) -> dict[str, Any]:
    _prepare_inventory_directory(inventory, create=False)
    with _installer_lock(inventory):
        states = [inspect_artifact(model) for model in inventory.models]
        incomplete = [state for state in states if state.status != "verified"]
        if incomplete:
            labels = ", ".join(f"{item.model.profile} ({item.status})" for item in incomplete)
            raise InstallerError(f"inventory is not fully installed: {labels}")
        writable = [state.model.profile for state in states if not state.read_only]
        if writable:
            raise InstallerError(
                "verified artifacts are writable; run the installer once to normalize "
                f"read-only permissions: {', '.join(writable)}"
            )
        return _summary(
            inventory,
            states,
            operation="verify",
            status="verified",
            required_download_bytes=0,
            disk_free_bytes=None,
            safety_margin_bytes=None,
            download_tool=None,
            downloads=[],
            capacity_checks=[],
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially install and verify the three immutable Q8 voice-model candidates."
        )
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    mode = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help="trusted root that must contain every destination path",
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate inventory, aria2c, resume state, and disk capacity without downloading",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="verify final sizes and SHA-256 values without requiring aria2c",
    )
    parser.add_argument(
        "--aria2c",
        help="aria2c executable path (default: resolve aria2c from PATH)",
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=4,
        help="connections per artifact, 1-8 (artifacts are always downloaded sequentially)",
    )
    parser.add_argument(
        "--safety-margin-gib",
        type=float,
        default=8.0,
        help="free space retained beyond unfinished artifact bytes (default: 8 GiB)",
    )
    args = parser.parse_args(argv)

    try:
        inventory = load_inventory(args.inventory, allowed_models_root=args.models_root)
        if args.verify_only:
            summary = verify_models(inventory)
        else:
            if args.safety_margin_gib < 0:
                raise InstallerError("--safety-margin-gib cannot be negative")
            summary = install_models(
                inventory,
                aria2c=args.aria2c,
                connections=args.connections,
                safety_margin_bytes=int(args.safety_margin_gib * 1024**3),
                preflight_only=args.preflight_only,
            )
    except InstallerError as exc:
        print(f"voice-model installer: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _immutable_download_url(
    source_url: str, *, expected_revision: str, model_id: str
) -> tuple[str, str, str]:
    parsed = urlsplit(source_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise InstallerError(
            f"source_url must be a credential-free Hugging Face HTTPS URL: {model_id}"
        )
    parts = parsed.path.split("/")
    if len(parts) < 6 or parts[0] or parts[3] not in {"blob", "resolve"}:
        raise InstallerError(f"unsupported Hugging Face artifact URL for {model_id}")
    revision = parts[4].lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or revision != expected_revision:
        raise InstallerError(f"source_url revision does not match source_revision for {model_id}")
    if not all(parts[index] for index in (1, 2, 5)):
        raise InstallerError(f"source_url has an empty repository or filename for {model_id}")
    resolved_path = "/".join(["", parts[1], parts[2], "resolve", expected_revision, *parts[5:]])
    download_url = urlunsplit(("https", "huggingface.co", resolved_path, "download=true", ""))
    return download_url, parts[-1], f"{parts[1]}/{parts[2]}"


def _parse_provenance(payload: Any, *, label: str) -> ModelProvenance:
    provenance_label = f"{label}.provenance"
    if not isinstance(payload, dict):
        raise InstallerError(f"{provenance_label} must be an object")
    artifact_repository = _required_string(payload, "artifact_repository", provenance_label).strip()
    conversion_kind = _required_string(payload, "conversion_kind", provenance_label).strip()
    upstream_model = _required_string(payload, "upstream_model", provenance_label).strip()
    upstream_reference_revision = (
        _required_string(payload, "upstream_reference_revision", provenance_label).strip().lower()
    )
    exact_upstream_lineage = (
        _required_string(payload, "exact_upstream_lineage", provenance_label).strip().lower()
    )
    limitation_value = payload.get("lineage_limitation")
    lineage_limitation: str | None
    if limitation_value is None:
        lineage_limitation = None
    elif isinstance(limitation_value, str) and limitation_value.strip():
        lineage_limitation = limitation_value.strip()
    else:
        raise InstallerError(
            f"{provenance_label}.lineage_limitation must be null or a non-empty string"
        )

    repository_pattern = r"[^/\s]+/[^/\s]+"
    if not re.fullmatch(repository_pattern, artifact_repository):
        raise InstallerError(f"{provenance_label}.artifact_repository must be an owner/repository")
    if not re.fullmatch(repository_pattern, upstream_model):
        raise InstallerError(f"{provenance_label}.upstream_model must be an owner/repository")
    if not re.fullmatch(r"[0-9a-f]{40}", upstream_reference_revision):
        raise InstallerError(
            f"{provenance_label}.upstream_reference_revision must be a 40-hex commit"
        )
    if exact_upstream_lineage not in LINEAGE_STATUSES:
        raise InstallerError(
            f"{provenance_label}.exact_upstream_lineage must be declared or undeclared"
        )
    if exact_upstream_lineage == "undeclared" and lineage_limitation is None:
        raise InstallerError(
            f"{provenance_label}.lineage_limitation is required when exact "
            "upstream-to-conversion lineage is undeclared"
        )
    return ModelProvenance(
        artifact_repository=artifact_repository,
        conversion_kind=conversion_kind,
        upstream_model=upstream_model,
        upstream_reference_revision=upstream_reference_revision,
        exact_upstream_lineage=exact_upstream_lineage,
        lineage_limitation=lineage_limitation,
    )


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = _absolute_without_symlink_resolution(path)
    if not absolute.is_absolute():
        raise InstallerError(f"{label} is not absolute: {path}")
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            current_stat = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise InstallerError(f"cannot inspect {label} component {current}: {exc}") from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise InstallerError(f"refusing symlinked {label} component: {current}")


def _validate_destination_path(path: Path, root: Path, *, label: str) -> None:
    absolute_path = _absolute_without_symlink_resolution(path)
    absolute_root = _absolute_without_symlink_resolution(root)
    if absolute_path != path:
        raise InstallerError(f"{label} must be normalized and absolute: {path}")
    try:
        relative = absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise InstallerError(
            f"{label} escapes allowed models root {absolute_root}: {absolute_path}"
        ) from exc
    if not relative.parts:
        raise InstallerError(f"{label} must name a file below {absolute_root}")
    _reject_symlink_components(absolute_root, label="allowed models root")
    _reject_symlink_components(absolute_path, label=label)


def _prepare_inventory_directory(inventory: Inventory, *, create: bool) -> None:
    root = inventory.allowed_models_root
    _reject_symlink_components(root, label="allowed models root")
    for model in inventory.models:
        if model.allowed_models_root != root:
            raise InstallerError("model and inventory allowed roots do not match")
        _validate_destination_path(model.path, root, label=f"artifact path for {model.model_id}")
        if model.path.parent != inventory.model_directory:
            raise InstallerError("inventory model directory changed after parsing")
    try:
        inventory.model_directory.relative_to(root)
    except ValueError as exc:
        raise InstallerError("inventory model directory escapes allowed models root") from exc

    if create:
        try:
            inventory.model_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise InstallerError(
                f"cannot create model directory {inventory.model_directory}: {exc}"
            ) from exc
    if not inventory.model_directory.exists():
        raise InstallerError(f"model directory does not exist: {inventory.model_directory}")
    _reject_symlink_components(inventory.model_directory, label="model directory")
    try:
        directory_stat = os.lstat(inventory.model_directory)
    except OSError as exc:
        raise InstallerError(
            f"cannot inspect model directory {inventory.model_directory}: {exc}"
        ) from exc
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise InstallerError(f"model directory is not a directory: {inventory.model_directory}")


def _partial_path(model: ModelSpec) -> Path:
    partial = model.path.with_name(f".{model.path.name}.{model.sha256[:16]}.partial")
    _validate_destination_path(
        partial, model.allowed_models_root, label=f"partial path for {model.model_id}"
    )
    return partial


def _partial_control_path(model: ModelSpec) -> Path:
    return Path(f"{_partial_path(model)}.aria2")


def _regular_file_stat(path: Path, *, label: str) -> os.stat_result:
    try:
        path_stat = os.lstat(path)
    except OSError as exc:
        raise InstallerError(f"cannot inspect {label} {path}: {exc}") from exc
    if not stat.S_ISREG(path_stat.st_mode):
        raise InstallerError(f"{label} is not a regular file: {path}")
    if path_stat.st_nlink != 1:
        raise InstallerError(f"{label} must not have multiple hard links: {path}")
    return path_stat


def _allocated_bytes(path_stat: os.stat_result, expected_size: int) -> int:
    return min(
        expected_size,
        path_stat.st_size,
        max(0, int(getattr(path_stat, "st_blocks", 0))) * 512,
    )


@contextmanager
def _installer_lock(inventory: Inventory) -> Iterator[None]:
    lock_path = inventory.model_directory / ".install-voice-models.lock"
    _validate_destination_path(lock_path, inventory.allowed_models_root, label="lock path")
    _reject_symlink_components(lock_path, label="installer lock")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise InstallerError("this platform cannot enforce a no-follow installer lock")
    flags = os.O_RDWR | os.O_CREAT | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise InstallerError(f"cannot securely open installer lock {lock_path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(lock_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
        ):
            raise InstallerError(f"installer lock is not a unique regular file: {lock_path}")
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InstallerError(
                f"another voice-model installer holds the lock: {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _capture_aria2_provenance(executable: str, command_runner: CommandRunner) -> dict[str, str]:
    try:
        result = command_runner(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise InstallerError(f"could not inspect aria2c version: {exc}") from exc
    if result.returncode != 0:
        raise InstallerError(f"aria2c --version failed with exit status {result.returncode}")
    stdout = result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout
    stderr = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
    output = "\n".join(part for part in (stdout, stderr) if part)
    first_line = next((item.strip() for item in output.splitlines() if item.strip()), "")
    match = re.fullmatch(r"aria2 version (\S+)", first_line)
    if match is None:
        raise InstallerError(
            "aria2c --version did not return the expected 'aria2 version <version>' line"
        )
    return {
        "name": "aria2",
        "executable": executable,
        "version": match.group(1),
        "version_output": first_line,
    }


def _run_aria2(
    command: list[str],
    command_runner: CommandRunner,
    *,
    profile: str,
) -> subprocess.CompletedProcess[Any]:
    try:
        return command_runner(command, check=False)
    except OSError as exc:
        raise InstallerError(f"could not start aria2c for {profile}: {exc}") from exc


def _free_disk_bytes(
    inventory: Inventory,
    *,
    disk_free_bytes: int | None,
    disk_free_provider: DiskFreeProvider | None,
) -> int:
    if disk_free_bytes is not None:
        value = disk_free_bytes
    elif disk_free_provider is not None:
        value = disk_free_provider(inventory.model_directory)
    else:
        value = shutil.disk_usage(inventory.model_directory).free
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InstallerError("disk free-space provider returned an invalid byte count")
    return value


def _check_disk_capacity(
    inventory: Inventory,
    states: Sequence[ArtifactState],
    *,
    safety_margin_bytes: int,
    disk_free_bytes: int | None,
    disk_free_provider: DiskFreeProvider | None,
    stage: str,
    checks: list[dict[str, Any]],
) -> int:
    required_bytes = sum(state.remaining_bytes for state in states)
    free_bytes = _free_disk_bytes(
        inventory,
        disk_free_bytes=disk_free_bytes,
        disk_free_provider=disk_free_provider,
    )
    required_with_margin = required_bytes + safety_margin_bytes
    checks.append(
        {
            "stage": stage,
            "free_bytes": free_bytes,
            "remaining_download_bytes": required_bytes,
            "required_with_margin_bytes": required_with_margin,
        }
    )
    if required_bytes and free_bytes < required_with_margin:
        raise InstallerError(
            "insufficient free disk space: "
            f"need {_format_bytes(required_with_margin)} "
            f"({_format_bytes(required_bytes)} remaining downloads + "
            f"{_format_bytes(safety_margin_bytes)} safety margin), "
            f"have {_format_bytes(free_bytes)} at {stage}"
        )
    return free_bytes


def _publish_partial(model: ModelSpec) -> ArtifactState:
    partial_path = _partial_path(model)
    control_path = _partial_control_path(model)
    _reject_symlink_components(partial_path, label="partial artifact")
    _reject_symlink_components(control_path, label="aria2 control file")
    _reject_symlink_components(model.path, label="final artifact")
    if control_path.exists():
        raise InstallerError(
            f"aria2 left resume metadata after success for {model.model_id}: {control_path}"
        )
    if model.path.exists():
        raise InstallerError(f"refusing to overwrite existing final artifact: {model.path}")

    partial_stat = _regular_file_stat(partial_path, label="partial artifact")
    if partial_stat.st_size != model.size_bytes:
        raise InstallerError(
            f"downloaded partial size mismatch for {model.model_id}: "
            f"expected {model.size_bytes}, got {partial_stat.st_size}"
        )
    actual_sha256 = sha256_file_stable(partial_path)
    if not hmac.compare_digest(actual_sha256, model.sha256):
        raise InstallerError(
            f"downloaded partial SHA-256 mismatch for {model.model_id}: "
            f"expected {model.sha256}, got {actual_sha256}"
        )

    _make_read_only(partial_path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise InstallerError("this platform cannot securely publish artifacts")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | nofollow
    descriptor = os.open(model.path.parent, directory_flags)
    try:
        try:
            os.link(
                partial_path.name,
                model.path.name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise InstallerError(
                f"refusing final artifact that appeared during publication: {model.path}"
            ) from exc
        os.unlink(partial_path.name, dir_fd=descriptor)
        os.fsync(descriptor)
    except OSError as exc:
        raise InstallerError(f"could not atomically publish {model.model_id}: {exc}") from exc
    finally:
        os.close(descriptor)

    published = inspect_artifact(model)
    if published.status != "verified" or not published.read_only:
        raise InstallerError(f"published artifact did not verify: {model.path}")
    return published


def _make_read_only(path: Path) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise InstallerError("this platform cannot enforce no-follow permissions")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InstallerError(f"cannot securely open artifact for chmod: {path}: {exc}") from exc
    try:
        path_stat = os.fstat(descriptor)
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
            raise InstallerError(f"artifact is not a unique regular file: {path}")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _regular_file_stat(path, label="read-only artifact")
    if stat.S_IMODE(os.lstat(path).st_mode) & 0o222:
        raise InstallerError(f"could not make artifact read-only: {path}")


def _resolve_aria2c(explicit: str | None) -> str:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.parent != Path("."):
            resolved = candidate.resolve()
            if not resolved.is_file() or not os.access(resolved, os.X_OK):
                raise InstallerError(f"aria2c is not executable: {resolved}")
            return str(resolved)
        executable = shutil.which(explicit)
    else:
        executable = shutil.which("aria2c")
    if executable is None:
        raise InstallerError("aria2c is required but was not found on PATH")
    return executable


def _required_string(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InstallerError(f"{label}.{key} must be a non-empty string")
    return value


def _summary(
    inventory: Inventory,
    states: Sequence[ArtifactState],
    *,
    operation: str,
    status: str,
    required_download_bytes: int,
    disk_free_bytes: int | None,
    safety_margin_bytes: int | None,
    download_tool: dict[str, str] | None,
    downloads: Sequence[dict[str, Any]],
    capacity_checks: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": status,
        "operation": operation,
        "inventory_version": inventory.inventory_version,
        "inventory_sha256": inventory.source_sha256,
        "inventory_path": str(inventory.source_path),
        "model_directory": str(inventory.model_directory),
        "allowed_models_root": str(inventory.allowed_models_root),
        "artifact_count": len(states),
        "expected_total_bytes": inventory.total_size_bytes,
        "required_download_bytes": required_download_bytes,
        "disk_free_bytes_at_preflight": disk_free_bytes,
        "safety_margin_bytes": safety_margin_bytes,
        "download_tool": download_tool,
        "download_attempts": list(downloads),
        "disk_capacity_checks": list(capacity_checks),
        "models": [
            {
                "profile": state.model.profile,
                "model_id": state.model.model_id,
                "path": str(state.model.path),
                "status": state.status,
                "observed_size_bytes": state.current_size_bytes,
                "allocated_size_bytes": state.allocated_size_bytes,
                "expected_size_bytes": state.model.size_bytes,
                "sha256": state.model.sha256,
                "expected_sha256": state.model.sha256,
                "observed_sha256": state.observed_sha256,
                "read_only": state.read_only,
                "source_revision": state.model.source_revision,
                "provenance": {
                    "artifact_repository": (state.model.provenance.artifact_repository),
                    "conversion_kind": state.model.provenance.conversion_kind,
                    "upstream_model": state.model.provenance.upstream_model,
                    "upstream_reference_revision": (
                        state.model.provenance.upstream_reference_revision
                    ),
                    "exact_upstream_lineage": (state.model.provenance.exact_upstream_lineage),
                    "lineage_limitation": (state.model.provenance.lineage_limitation),
                },
            }
            for state in states
        ],
    }


def _format_bytes(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


if __name__ == "__main__":
    raise SystemExit(main())
