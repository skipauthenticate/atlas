from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


NORMALIZE_TIMEOUT_SECONDS = 30 * 60.0
CLEANUP_FILTERS = {
    "none": None,
    "adaptive": "highpass=f=70,lowpass=f=7800,dynaudnorm=f=250:g=9:p=0.95",
    "enhanced": (
        "highpass=f=80,lowpass=f=7600,afftdn=nr=12:nf=-35:tn=1,dynaudnorm=f=150:g=12:p=0.90"
    ),
}


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    *,
    cleanup_mode: str = "none",
) -> list[str]:
    cleanup_mode = _normalize_cleanup_mode(cleanup_mode)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-map_metadata",
        "-1",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    cleanup_filter = CLEANUP_FILTERS[cleanup_mode]
    if cleanup_filter:
        command.extend(["-af", cleanup_filter])
    command.extend(
        [
            "-c:a",
            "pcm_s16le",
            "-sample_fmt",
            "s16",
            "-f",
            "wav",
            str(output_path),
        ]
    )
    return command


def normalize_audio(
    input_path: Path,
    output_path: Path,
    *,
    cleanup_mode: str = "none",
    timeout_seconds: float = NORMALIZE_TIMEOUT_SECONDS,
) -> None:
    """Normalize audio into an atomically published 16 kHz mono PCM WAV.

    Cleanup filters are best-effort. If an optional filter chain is not supported
    by the host FFmpeg build (or rejects the source), normalization is retried
    without cleanup before the destination is replaced.
    """

    cleanup_mode = _normalize_cleanup_mode(cleanup_mode)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        try:
            _run_ffmpeg(
                build_ffmpeg_command(
                    input_path,
                    temporary_path,
                    cleanup_mode=cleanup_mode,
                ),
                timeout_seconds=timeout_seconds,
            )
        except subprocess.CalledProcessError as cleanup_error:
            if cleanup_mode == "none":
                raise _normalization_error(input_path, cleanup_error) from cleanup_error

            temporary_path.unlink(missing_ok=True)
            try:
                _run_ffmpeg(
                    build_ffmpeg_command(input_path, temporary_path),
                    timeout_seconds=timeout_seconds,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as fallback_error:
                message = _normalization_error(input_path, fallback_error)
                cleanup_detail = _ffmpeg_error_detail(cleanup_error)
                raise RuntimeError(
                    f"{message} Cleanup mode {cleanup_mode!r} also failed: {cleanup_detail}"
                ) from fallback_error
        except subprocess.TimeoutExpired as exc:
            raise _normalization_error(input_path, exc) from exc

        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _run_ffmpeg(command: list[str], *, timeout_seconds: float) -> None:
    subprocess.run(
        command,
        check=True,
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
    )


def _normalize_cleanup_mode(cleanup_mode: str) -> str:
    mode = str(cleanup_mode or "none").strip().lower()
    if mode not in CLEANUP_FILTERS:
        choices = ", ".join(CLEANUP_FILTERS)
        raise ValueError(f"Unknown audio cleanup mode {cleanup_mode!r}; use {choices}.")
    return mode


def _normalization_error(
    input_path: Path,
    error: subprocess.CalledProcessError | subprocess.TimeoutExpired,
) -> RuntimeError:
    if isinstance(error, subprocess.TimeoutExpired):
        detail = f"timed out after {error.timeout} seconds"
    else:
        detail = _ffmpeg_error_detail(error)
    return RuntimeError(f"FFmpeg could not normalize {input_path}: {detail}")


def _ffmpeg_error_detail(error: subprocess.CalledProcessError) -> str:
    stderr = error.stderr
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    detail = " ".join(str(stderr or "").split())
    if detail:
        return detail[-1000:]
    return f"process exited with status {error.returncode}"
