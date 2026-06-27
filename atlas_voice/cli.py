from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .config import Settings
from .database import Database
from .exporter import export_recording
from .pipeline import PipelineProcessor
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


def cmd_worker(_args: argparse.Namespace) -> int:
    settings, db = settings_and_db()
    Worker(settings, db).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
