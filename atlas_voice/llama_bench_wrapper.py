"""Run one pinned GGUF through a fixed, report-compatible llama-bench protocol.

The wrapper is deliberately narrow: it selects one profile from the committed
inventory, verifies the artifact before and after the subprocess, validates the
standard llama-bench JSON rows, and publishes a no-clobber JSON artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "atlas-voice-llama-bench-q8-v1"
PROMPT_TOKENS = 512
GENERATION_TOKENS = 128
REPETITIONS = 5
BATCH_SIZE = 2048
UBATCH_SIZE = 512
GPU_LAYERS = -1
KV_TYPE = "q8_0"
DEFAULT_INVENTORY_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "voice_models_q8_inventory.json"
)
CONSISTENT_ROW_FIELDS = (
    "build_number",
    "build_commit",
    "model_filename",
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


class LlamaBenchWrapperError(RuntimeError):
    """Base class for audited wrapper failures."""


class InputValidationError(LlamaBenchWrapperError):
    """Raised when an input or output path is unsafe or malformed."""


class ArtifactVerificationError(LlamaBenchWrapperError):
    """Raised when the selected GGUF does not match the pinned inventory."""


class BenchmarkExecutionError(LlamaBenchWrapperError):
    """Raised when llama-bench fails or emits unsupported results."""


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    size_bytes: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class InventoryModel:
    profile: str
    model_id: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class LoadedInventory:
    path: Path
    identity: FileIdentity
    inventory_version: str | None
    models: tuple[InventoryModel, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_hex_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _first_symlink_component(path: Path) -> Path | None:
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return current
        if not current.exists():
            break
    return None


def _regular_file_identity(path: Path, *, executable: bool = False) -> FileIdentity:
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    symlink = _first_symlink_component(absolute)
    if symlink is not None:
        raise InputValidationError(f"input path contains a symlink: {symlink}")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise InputValidationError(f"cannot open regular input file {absolute}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise InputValidationError(f"input is not a regular file: {absolute}")
        if executable and not os.access(absolute, os.X_OK):
            raise InputValidationError(f"llama-bench binary is not executable: {absolute}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return FileIdentity(
            path=absolute,
            size_bytes=metadata.st_size,
            sha256=digest.hexdigest(),
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mtime_ns=metadata.st_mtime_ns,
        )
    finally:
        os.close(descriptor)


def _same_file_identity(left: FileIdentity, right: FileIdentity) -> bool:
    return (
        left.path == right.path
        and left.size_bytes == right.size_bytes
        and left.sha256 == right.sha256
        and left.device == right.device
        and left.inode == right.inode
        and left.mtime_ns == right.mtime_ns
    )


def load_inventory(path: Path) -> LoadedInventory:
    identity = _regular_file_identity(path)
    try:
        payload = json.loads(identity.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"cannot parse inventory {identity.path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise InputValidationError("inventory must be a schema_version 1 JSON object")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise InputValidationError("inventory models must be a non-empty list")

    models: list[InventoryModel] = []
    profiles: set[str] = set()
    for index, raw in enumerate(raw_models):
        if not isinstance(raw, dict):
            raise InputValidationError(f"inventory model {index} must be an object")
        profile = raw.get("profile")
        model_id = raw.get("model_id")
        raw_path = raw.get("path")
        size_bytes = raw.get("size_bytes")
        sha256 = raw.get("sha256")
        if not isinstance(profile, str) or not profile.strip():
            raise InputValidationError(f"inventory model {index} has an invalid profile")
        profile = profile.strip().casefold()
        if profile in profiles:
            raise InputValidationError(f"inventory profile is not unique: {profile}")
        profiles.add(profile)
        if not isinstance(model_id, str) or not model_id.strip():
            raise InputValidationError(f"inventory model {profile} has an invalid model_id")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise InputValidationError(f"inventory model {profile} has an invalid path")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise InputValidationError(f"inventory model {profile} has an invalid size_bytes")
        if not _is_hex_sha256(sha256):
            raise InputValidationError(f"inventory model {profile} has an invalid sha256")
        artifact_path = Path(raw_path).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = identity.path.parent / artifact_path
        artifact_path = Path(os.path.abspath(str(artifact_path)))
        models.append(
            InventoryModel(
                profile=profile,
                model_id=model_id.strip(),
                path=artifact_path,
                size_bytes=size_bytes,
                sha256=str(sha256).casefold(),
            )
        )

    inventory_version = payload.get("inventory_version")
    if inventory_version is not None and not isinstance(inventory_version, str):
        raise InputValidationError("inventory_version must be a string when present")
    return LoadedInventory(
        path=identity.path,
        identity=identity,
        inventory_version=inventory_version,
        models=tuple(models),
    )


def select_model(inventory: LoadedInventory, profile: str) -> InventoryModel:
    normalized = profile.strip().casefold()
    selected = [model for model in inventory.models if model.profile == normalized]
    if len(selected) != 1:
        available = ", ".join(model.profile for model in inventory.models)
        raise InputValidationError(
            f"profile {profile!r} is not present exactly once; available profiles: {available}"
        )
    return selected[0]


def verify_model(model: InventoryModel) -> FileIdentity:
    identity = _regular_file_identity(model.path)
    if identity.size_bytes != model.size_bytes:
        raise ArtifactVerificationError(
            f"artifact size mismatch for {model.profile}: expected {model.size_bytes}, "
            f"observed {identity.size_bytes}"
        )
    if identity.sha256 != model.sha256:
        raise ArtifactVerificationError(
            f"artifact SHA-256 mismatch for {model.profile}: expected {model.sha256}, "
            f"observed {identity.sha256}"
        )
    return identity


def _paths_collide(path: Path, protected: Path) -> bool:
    left = Path(os.path.abspath(os.path.expanduser(str(path))))
    right = Path(os.path.abspath(os.path.expanduser(str(protected))))
    if left == right:
        return True
    if left.exists() and right.exists():
        try:
            return os.path.samefile(left, right)
        except OSError:
            return False
    return False


def validate_output_path(path: Path, *, protected_paths: Sequence[Path]) -> Path:
    output = Path(os.path.abspath(os.path.expanduser(str(path))))
    for protected in protected_paths:
        if _paths_collide(output, protected):
            raise InputValidationError(
                f"output path collides with protected input/artifact: {protected}"
            )
    if output.suffix.casefold() != ".json":
        raise InputValidationError(f"output must have a .json extension: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    for label, candidate in (("output", output), ("temporary output", temporary)):
        symlink = _first_symlink_component(candidate)
        if symlink is not None:
            raise InputValidationError(f"{label} path contains a symlink: {symlink}")
        for protected in protected_paths:
            if _paths_collide(candidate, protected):
                raise InputValidationError(
                    f"{label} path collides with protected input/artifact: {protected}"
                )
        if candidate.exists():
            raise InputValidationError(f"refusing to overwrite existing {label}: {candidate}")
    return output


def build_command(binary: Path, model: InventoryModel) -> list[str]:
    """Return the fixed protocol command; intentionally omit ``--threads``.

    llama-bench chooses its platform default thread count. The emitted rows are
    required to agree on that count, and it is copied into protocol provenance.
    """

    return [
        str(binary),
        "--model",
        str(model.path),
        "--n-prompt",
        str(PROMPT_TOKENS),
        "--n-gen",
        str(GENERATION_TOKENS),
        "--repetitions",
        str(REPETITIONS),
        "--batch-size",
        str(BATCH_SIZE),
        "--ubatch-size",
        str(UBATCH_SIZE),
        "--n-gpu-layers",
        str(GPU_LAYERS),
        "--n-cpu-moe",
        "0",
        "--no-kv-offload",
        "0",
        "--flash-attn",
        "on",
        "--cache-type-k",
        KV_TYPE,
        "--cache-type-v",
        KV_TYPE,
        "--mmap",
        "1",
        "--offline",
        "--output",
        "json",
    ]


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkExecutionError(f"llama-bench row field {field} must be an integer")
    return value


def _enabled(value: object) -> bool:
    return (
        value is True
        or value == 1
        or (isinstance(value, str) and value.strip().casefold() in {"1", "on", "true"})
    )


def _disabled(value: object) -> bool:
    return (
        value is False
        or value == 0
        or (isinstance(value, str) and value.strip().casefold() in {"0", "off", "false"})
    )


def parse_and_validate_results(stdout: str, model: InventoryModel) -> list[dict[str, Any]]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise BenchmarkExecutionError(f"llama-bench stdout is not valid JSON: {exc}") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or any(not isinstance(row, dict) for row in payload)
    ):
        raise BenchmarkExecutionError(
            "llama-bench JSON must contain exactly one prompt row and one generation row"
        )
    rows = [dict(row) for row in payload]

    for index, row in enumerate(rows):
        missing = [field for field in CONSISTENT_ROW_FIELDS if field not in row]
        if missing:
            raise BenchmarkExecutionError(
                f"llama-bench row {index} is missing fields: {', '.join(missing)}"
            )
        if row.get("error") not in (None, ""):
            raise BenchmarkExecutionError(f"llama-bench row {index} reports an error")
        build_commit = row.get("build_commit")
        if not isinstance(build_commit, str) or not build_commit.strip():
            raise BenchmarkExecutionError(
                f"llama-bench row {index} build_commit must be a non-empty string"
            )
        if _integer(row.get("build_number"), field="build_number") <= 0:
            raise BenchmarkExecutionError(f"llama-bench row {index} build_number must be positive")
        filename = row.get("model_filename")
        if filename != str(model.path):
            raise BenchmarkExecutionError(
                f"llama-bench row {index} model_filename mismatch: {filename!r}"
            )
        avg_ts = row.get("avg_ts")
        if isinstance(avg_ts, bool) or not isinstance(avg_ts, (int, float)):
            raise BenchmarkExecutionError(f"llama-bench row {index} avg_ts must be numeric")
        if not math.isfinite(float(avg_ts)) or float(avg_ts) <= 0:
            raise BenchmarkExecutionError(f"llama-bench row {index} avg_ts must be positive")
        for samples_field in ("samples_ns", "samples_ts"):
            samples = row.get(samples_field)
            if not isinstance(samples, list) or len(samples) != REPETITIONS:
                raise BenchmarkExecutionError(
                    f"llama-bench row {index} {samples_field} must contain "
                    f"exactly {REPETITIONS} samples"
                )
            if any(
                isinstance(sample, bool)
                or not isinstance(sample, (int, float))
                or not math.isfinite(float(sample))
                or float(sample) <= 0
                for sample in samples
            ):
                raise BenchmarkExecutionError(
                    f"llama-bench row {index} {samples_field} contains an invalid sample"
                )

    signatures = {
        json.dumps({field: row[field] for field in CONSISTENT_ROW_FIELDS}, sort_keys=True)
        for row in rows
    }
    if len(signatures) != 1:
        raise BenchmarkExecutionError("llama-bench rows use inconsistent build or engine settings")

    first = rows[0]
    expected_settings: tuple[tuple[str, object], ...] = (
        ("n_batch", BATCH_SIZE),
        ("n_ubatch", UBATCH_SIZE),
        ("type_k", KV_TYPE),
        ("type_v", KV_TYPE),
        ("n_gpu_layers", GPU_LAYERS),
        ("n_cpu_moe", 0),
    )
    for field, expected in expected_settings:
        if first.get(field) != expected:
            raise BenchmarkExecutionError(
                f"llama-bench {field} mismatch: expected {expected!r}, observed {first.get(field)!r}"
            )
    if not _disabled(first.get("no_kv_offload")):
        raise BenchmarkExecutionError("llama-bench unexpectedly disabled KV offload")
    if not _enabled(first.get("flash_attn")):
        raise BenchmarkExecutionError("llama-bench did not enable flash attention")
    if not _enabled(first.get("use_mmap")):
        raise BenchmarkExecutionError("llama-bench did not enable mmap")
    if _integer(first.get("n_threads"), field="n_threads") <= 0:
        raise BenchmarkExecutionError("llama-bench reported a non-positive default thread count")

    tests = {
        (_integer(row.get("n_prompt"), field="n_prompt"), _integer(row.get("n_gen"), field="n_gen"))
        for row in rows
    }
    expected_tests = {(PROMPT_TOKENS, 0), (0, GENERATION_TOKENS)}
    if tests != expected_tests:
        raise BenchmarkExecutionError(
            f"llama-bench test rows mismatch: expected {sorted(expected_tests)}, observed {sorted(tests)}"
        )
    # Standard llama-bench JSON records repetitions as the sample-array length,
    # not as a named field. Add the derived value for the report summarizer.
    for row in rows:
        row["repetitions"] = REPETITIONS
    return rows


def _assert_unchanged(label: str, before: FileIdentity, after: FileIdentity) -> None:
    if not _same_file_identity(before, after):
        raise BenchmarkExecutionError(f"{label} changed while llama-bench was running")


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Recheck after mkdir so a newly created parent cannot conceal a symlink.
    validate_output_path(path, protected_paths=())
    temporary = path.with_suffix(path.suffix + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails rather than clobbering an
        # output created after validation. Both paths live in the same folder.
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory = -1
        if directory >= 0:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except FileExistsError as exc:
        raise InputValidationError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run_llama_bench(
    *,
    profile: str,
    inventory_path: Path,
    binary_path: Path,
    output_path: Path,
    timeout_seconds: float = 7200.0,
) -> dict[str, Any]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise InputValidationError("timeout_seconds must be positive and finite")
    inventory = load_inventory(inventory_path)
    model = select_model(inventory, profile)
    binary_before = _regular_file_identity(binary_path, executable=True)
    protected = [inventory.path, binary_before.path, *(item.path for item in inventory.models)]
    output = validate_output_path(output_path, protected_paths=protected)
    artifact_before = verify_model(model)
    command = build_command(binary_before.path, model)

    started_utc = _utc_now()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(inventory.path.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise BenchmarkExecutionError(
            f"llama-bench exceeded the {timeout_seconds:g}-second timeout"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise BenchmarkExecutionError(f"could not execute llama-bench: {exc}") from exc
    elapsed_seconds = time.monotonic() - started
    ended_utc = _utc_now()
    if completed.returncode != 0:
        stderr_tail = completed.stderr[-4096:].strip()
        detail = f": {stderr_tail}" if stderr_tail else ""
        raise BenchmarkExecutionError(
            f"llama-bench exited with status {completed.returncode}{detail}"
        )

    rows = parse_and_validate_results(completed.stdout, model)
    artifact_after = verify_model(model)
    inventory_after = _regular_file_identity(inventory.path)
    binary_after = _regular_file_identity(binary_before.path, executable=True)
    _assert_unchanged("model artifact", artifact_before, artifact_after)
    _assert_unchanged("inventory", inventory.identity, inventory_after)
    _assert_unchanged("llama-bench binary", binary_before, binary_after)

    build_commits = {str(row["build_commit"]) for row in rows}
    build_numbers = {row["build_number"] for row in rows}
    # The fields are also part of the consistent signature; these explicit
    # checks make the provenance contract clear and avoid empty version data.
    if len(build_commits) != 1 or not next(iter(build_commits)).strip():
        raise BenchmarkExecutionError("llama-bench rows lack a consistent build commit")
    if len(build_numbers) != 1:
        raise BenchmarkExecutionError("llama-bench rows lack a consistent build number")

    stderr_bytes = completed.stderr.encode("utf-8")
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact": {
            "path": str(model.path),
            "sha256": artifact_before.sha256,
        },
        "results": rows,
        "provenance": {
            "protocol": {
                "id": PROTOCOL_ID,
                "prompt_tokens": PROMPT_TOKENS,
                "generation_tokens": GENERATION_TOKENS,
                "repetitions": REPETITIONS,
                "batch_size": BATCH_SIZE,
                "ubatch_size": UBATCH_SIZE,
                "n_gpu_layers": GPU_LAYERS,
                "flash_attention": "on",
                "kv_type_k": KV_TYPE,
                "kv_type_v": KV_TYPE,
                "mmap": True,
                "kv_offload": True,
                "threads_source": "llama-bench platform default",
                "observed_threads": rows[0]["n_threads"],
            },
            "profile": model.profile,
            "model_id": model.model_id,
            "artifact_size_bytes": artifact_before.size_bytes,
            "inventory": {
                "path": str(inventory.path),
                "sha256": inventory.identity.sha256,
                "inventory_version": inventory.inventory_version,
            },
            "binary": {
                "path": str(binary_before.path),
                "sha256": binary_before.sha256,
                "size_bytes": binary_before.size_bytes,
                "build_commit": next(iter(build_commits)),
                "build_number": next(iter(build_numbers)),
            },
            "execution": {
                "argv": command,
                "cwd": str(inventory.path.parent),
                "started_utc": started_utc,
                "ended_utc": ended_utc,
                "elapsed_seconds": elapsed_seconds,
                "exit_code": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
                "stderr_tail": completed.stderr[-4096:],
            },
        },
    }
    # Repeat collision checks after the benchmark, immediately before publish.
    output = validate_output_path(output, protected_paths=protected)
    try:
        _publish_json(output, report)
    except LlamaBenchWrapperError:
        raise
    except OSError as exc:
        raise InputValidationError(f"could not publish output {output}: {exc}") from exc
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one pinned voice GGUF and run the fixed, report-compatible "
            "llama-bench Q8 protocol. The output is atomically published and must not exist."
        )
    )
    parser.add_argument(
        "--profile", required=True, help="Pinned inventory profile (light/torch/fire)"
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY_PATH,
        help=f"Pinned inventory JSON (default: {DEFAULT_INVENTORY_PATH})",
    )
    parser.add_argument("--llama-bench", type=Path, required=True, help="llama-bench executable")
    parser.add_argument("--output", type=Path, required=True, help="New .json result path")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=7200.0,
        help="Subprocess timeout in seconds (default: 7200)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_llama_bench(
            profile=args.profile,
            inventory_path=args.inventory,
            binary_path=args.llama_bench,
            output_path=args.output,
            timeout_seconds=args.timeout_seconds,
        )
    except LlamaBenchWrapperError as exc:
        print(f"llama-bench wrapper failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "profile": report["provenance"]["profile"],
                "output": str(Path(args.output).expanduser().absolute()),
                "artifact_sha256": report["artifact"]["sha256"],
                "build_commit": report["provenance"]["binary"]["build_commit"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
