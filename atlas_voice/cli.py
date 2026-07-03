from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .anythingllm import AnythingLLMError, sync_recording_to_anythingllm
from .config import Settings
from .database import Database
from .exporter import export_recording
from .pipeline import PipelineProcessor
from .benchmark import make_smoke_audio, print_benchmark_results, run_asr_benchmark
from .worker import Worker


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return int(args.func(args) or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas-voice")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Run the repository setup script")
    setup.add_argument("setup_args", nargs=argparse.REMAINDER)
    setup.set_defaults(func=cmd_setup)

    for command in ("start", "stop", "status", "logs"):
        sub = subparsers.add_parser(command, help=f"Docker Compose {command}")
        sub.set_defaults(func=cmd_compose)

    ingest = subparsers.add_parser("ingest", help="Queue an audio file for processing")
    ingest.add_argument("path", type=Path)
    ingest.set_defaults(func=cmd_ingest)

    retry = subparsers.add_parser("retry", help="Retry failed jobs for a recording")
    retry.add_argument("recording_id")
    retry.set_defaults(func=cmd_retry)

    export = subparsers.add_parser("export", help="Export a processed recording")
    export.add_argument("recording_id")
    export.add_argument("--format", choices=("json", "md", "txt"), default="json")
    export.set_defaults(func=cmd_export)

    sync_anythingllm = subparsers.add_parser(
        "sync-anythingllm", help="Sync a recording into an AnythingLLM workspace"
    )
    sync_anythingllm.add_argument("recording_id")
    sync_anythingllm.set_defaults(func=cmd_sync_anythingllm)

    benchmark = subparsers.add_parser("benchmark-asr", help="Benchmark ASR providers")
    benchmark.add_argument("audio", type=Path, nargs="?", help="Audio file to benchmark")
    benchmark.add_argument(
        "--providers",
        default="whisperx,parakeet,canary,vibevoice",
        help="Comma-separated providers: whisperx, parakeet, canary, vibevoice",
    )
    benchmark.add_argument("--reference", type=Path, help="Optional reference transcript text")
    benchmark.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    benchmark.add_argument(
        "--include-diarization",
        action="store_true",
        help="Also run the configured diarization provider after ASR",
    )
    benchmark.set_defaults(func=cmd_benchmark_asr)

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.set_defaults(func=cmd_worker)
    return parser


def settings_and_db() -> tuple[Settings, Database]:
    settings = Settings.from_env()
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    return settings, db


def cmd_setup(args: argparse.Namespace) -> int:
    script = Path.cwd() / "setup.sh"
    if not script.exists():
        print("setup.sh was not found in the current directory", file=sys.stderr)
        return 1
    return subprocess.call([str(script), *args.setup_args])


def cmd_compose(args: argparse.Namespace) -> int:
    mapping = {
        "start": ["up", "-d"],
        "stop": ["down"],
        "status": ["ps"],
        "logs": ["logs", "-f"],
    }
    return subprocess.call(["docker", "compose", *mapping[args.command]])


def cmd_ingest(args: argparse.Namespace) -> int:
    settings, db = settings_and_db()
    processor = PipelineProcessor(settings, db)
    recording_id = processor.enqueue_source(args.path)
    print(recording_id)
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    _settings, db = settings_and_db()
    if db.get_recording(args.recording_id) is None:
        print(f"Unknown recording: {args.recording_id}", file=sys.stderr)
        return 1
    db.retry_recording(args.recording_id)
    print(f"queued retry for {args.recording_id}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    _settings, db = settings_and_db()
    try:
        print(export_recording(db, args.recording_id, args.format), end="")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def cmd_sync_anythingllm(args: argparse.Namespace) -> int:
    settings, db = settings_and_db()
    try:
        result = sync_recording_to_anythingllm(db, args.recording_id, settings)
    except (ValueError, AnythingLLMError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    documents = result.get("documents") or []
    if documents and documents[0].get("location"):
        print(f"synced to AnythingLLM: {documents[0]['location']}")
    else:
        print("synced to AnythingLLM")
    return 0


def cmd_benchmark_asr(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    audio_path = args.audio
    reference_text = args.reference.read_text().strip() if args.reference else None
    if audio_path is None:
        audio_path = Path("/tmp/atlas-voice-asr-smoke.wav")
        reference_text = make_smoke_audio(audio_path)
        print(f"generated smoke audio: {audio_path}")
    providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    results = run_asr_benchmark(
        audio_path,
        settings,
        providers=providers,
        reference_text=reference_text,
        include_diarization=args.include_diarization,
    )
    print_benchmark_results(results, json_output=args.json)
    return 1 if any(result.get("error") for result in results) else 0


def cmd_worker(_args: argparse.Namespace) -> int:
    settings, db = settings_and_db()
    Worker(settings, db).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
