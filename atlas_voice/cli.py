from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .ambient import (
    AMBIENT_MODES,
    AmbientResult,
    process_ambient_file,
    process_ambient_path,
    process_microphone_once,
    validate_microphone_asr,
)
from .anythingllm import AnythingLLMError, sync_recording_to_anythingllm
from .assistant_config import AssistantConfigError, load_assistant_config
from .config import Settings
from .database import Database
from .exporter import export_recording
from .pipeline import PipelineProcessor
from .privacy import privacy_summary
from .memory import extract_memories_from_ambient_session
from .retention import apply_ambient_retention
from .benchmark import make_smoke_audio, print_benchmark_results, run_asr_benchmark
from .storage import is_audio_file
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

    privacy = subparsers.add_parser("privacy", help="Privacy and local-only checks")
    privacy_subparsers = privacy.add_subparsers(dest="privacy_command", required=True)
    privacy_status = privacy_subparsers.add_parser("status", help="Show local-only status")
    privacy_status.set_defaults(func=cmd_privacy_status)
    privacy_audit = privacy_subparsers.add_parser("audit-egress", help="Audit configured egress URLs")
    privacy_audit.set_defaults(func=cmd_privacy_audit_egress)
    privacy_purge = privacy_subparsers.add_parser(
        "purge",
        help="Dry-run or delete assistant/ambient sessions matching privacy filters",
    )
    privacy_purge.add_argument("--session", dest="session_id", help="Exact session id to purge")
    privacy_purge.add_argument("--keyword", help="Match session title, utterance text, or reply text")
    privacy_purge.add_argument("--person", help="Match speaker, title, or utterance text")
    privacy_purge.add_argument("--date", dest="started_on", help="Match session start date YYYY-MM-DD")
    privacy_purge.add_argument("--before", help="Match sessions before YYYY-MM-DD")
    privacy_purge.add_argument("--after", help="Match sessions on or after YYYY-MM-DD")
    privacy_purge.add_argument("--mode", help="Optional session mode filter, such as ambient")
    privacy_purge.add_argument("--limit", type=int, default=100, help="Maximum matching sessions")
    privacy_purge.add_argument("--yes", action="store_true", help="Actually delete matching sessions")
    privacy_purge.set_defaults(func=cmd_privacy_purge)
    privacy_retention = privacy_subparsers.add_parser(
        "retention",
        help="Dry-run or apply configured ambient retention windows",
    )
    privacy_retention.add_argument("--limit", type=int, default=10000, help="Maximum sessions to scan")
    privacy_retention.add_argument("--yes", action="store_true", help="Actually apply retention")
    privacy_retention.set_defaults(func=cmd_privacy_retention)

    memory = subparsers.add_parser("memory", help="Local memory extraction and retrieval")
    memory_subparsers = memory.add_subparsers(dest="memory_command", required=True)
    memory_extract_ambient = memory_subparsers.add_parser(
        "extract-ambient",
        help="Dry-run or store memory candidates from an ambient or meeting session",
    )
    memory_extract_ambient.add_argument("--session", required=True, help="Ambient session id")
    memory_extract_ambient.add_argument("--limit", type=int, default=20, help="Maximum candidates")
    memory_extract_ambient.add_argument("--yes", action="store_true", help="Actually store candidates")
    memory_extract_ambient.set_defaults(func=cmd_memory_extract_ambient)

    ambient = subparsers.add_parser("ambient", help="Run the ambient listener MVP")
    ambient.add_argument(
        "--source",
        help="Audio file, directory, or 'mic'. Defaults to ATLAS_VOICE_AMBIENT_SOURCE.",
    )
    ambient.add_argument(
        "--mode",
        choices=sorted(AMBIENT_MODES),
        help="ambient, meeting, direct, private, or paused.",
    )
    ambient.add_argument("--once", action="store_true", help="Process one source/chunk and exit")
    ambient.add_argument("--chunk-seconds", type=float, help="Mic capture chunk duration")
    ambient.add_argument("--poll-seconds", type=float, help="Directory/mic polling interval")
    ambient.add_argument("--mic-device", help="ALSA device name for mic capture")
    ambient.add_argument("--vad-threshold", type=float, help="Energy threshold for VAD")
    ambient.add_argument("--min-speech-seconds", type=float, help="Minimum segment duration")
    ambient.add_argument(
        "--retain-audio",
        action="store_true",
        default=None,
        help="Keep transient ambient audio artifacts",
    )
    ambient.set_defaults(func=cmd_ambient)

    validate_brio = subparsers.add_parser(
        "validate-brio",
        help="Capture Logitech BRIO ALSA audio and validate real ASR transcription",
    )
    validate_brio.add_argument("--device", default="plughw:2,0", help="ALSA capture device")
    validate_brio.add_argument("--seconds", type=float, default=5.0, help="Capture duration")
    validate_brio.add_argument(
        "--allow-stub",
        action="store_true",
        help="Allow stub mode for command smoke tests; real validation should leave this off",
    )
    validate_brio.add_argument(
        "--min-transcript-chars",
        type=int,
        default=1,
        help="Minimum transcript characters required for success",
    )
    validate_brio.set_defaults(func=cmd_validate_brio)

    benchmark = subparsers.add_parser("benchmark-asr", help="Benchmark ASR providers")
    benchmark.add_argument("audio", type=Path, nargs="?", help="Audio file to benchmark")
    benchmark.add_argument(
        "--providers",
        default="whisperx,faster-whisper,hyprwhspr,parakeet,canary,vibevoice",
        help="Comma-separated providers: whisperx, faster-whisper, hyprwhspr, parakeet, canary, vibevoice",
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


def cmd_privacy_status(_args: argparse.Namespace) -> int:
    try:
        _settings, _db, report = _privacy_report("privacy.status")
    except AssistantConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Local-only status: {report['status']}")
    print(f"Assistant config: {report['assistant_config_path']}")
    print("Assistant config loaded: " + ("yes" if report["assistant_config_loaded"] else "no"))
    if report["issues"]:
        for issue in report["issues"]:
            print(f"[{issue['severity']}] {issue['check']}: {issue['message']}")
    else:
        print("No local-only issues found.")
    return 1 if report["status"] == "error" else 0


def cmd_privacy_audit_egress(_args: argparse.Namespace) -> int:
    try:
        settings, _db, report = _privacy_report("privacy.audit_egress")
    except AssistantConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Configured endpoints:")
    print(f"- web: {settings.host}:{settings.port}")
    print(f"- llm: {settings.llm_base_url}")
    if (
        settings.anythingllm_auto_sync
        or settings.anythingllm_api_key
        or settings.anythingllm_workspace_slug
    ):
        print(f"- anythingllm: {settings.anythingllm_base_url}")
    print(f"Local-only status: {report['status']}")
    if report["issues"]:
        for issue in report["issues"]:
            print(f"[{issue['severity']}] {issue['check']}: {issue['message']}")
    else:
        print("No configured external egress found.")
    return 1 if report["status"] == "error" else 0


def cmd_privacy_purge(args: argparse.Namespace) -> int:
    if not _privacy_purge_has_filter(args):
        print("privacy purge requires at least one purge filter", file=sys.stderr)
        return 1
    settings, db = settings_and_db()
    matches = db.find_privacy_purge_sessions(
        session_id=args.session_id,
        keyword=args.keyword,
        person=args.person,
        started_on=args.started_on,
        before=args.before,
        after=args.after,
        mode=args.mode,
        limit=args.limit,
    )
    if not matches:
        print("No matching sessions found.")
        return 0

    _print_privacy_purge_matches(matches, dry_run=not args.yes)
    if not args.yes:
        print("dry run only; rerun with --yes to delete matching sessions")
        return 0

    session_ids = [str(session["id"]) for session in matches]
    result = db.purge_privacy_sessions(session_ids)
    artifact_count = _purge_realtime_artifacts(settings, session_ids)
    db.log_privacy_event(
        "privacy.purge",
        f"Purged {result['session_count']} privacy session(s).",
        severity="info",
        metadata={
            "filters": _privacy_purge_filter_metadata(args),
            "session_ids": session_ids,
            "session_count": result["session_count"],
            "utterance_count": result["utterance_count"],
            "assistant_turn_count": result["assistant_turn_count"],
            "artifact_count": artifact_count,
        },
    )
    print(
        f"purged {result['session_count']} session(s), "
        f"{result['utterance_count']} utterance(s), "
        f"{result['assistant_turn_count']} assistant turn(s), "
        f"{artifact_count} artifact dir(s)"
    )
    return 0


def cmd_privacy_retention(args: argparse.Namespace) -> int:
    settings, db = settings_and_db()
    result = apply_ambient_retention(
        settings,
        db,
        dry_run=not args.yes,
        limit=args.limit,
    )
    suffix = "dry run" if result.dry_run else "applied"
    print(
        f"privacy retention {suffix}: "
        f"{result.audio_artifact_count} audio artifact dir(s), "
        f"{result.session_count} session(s)"
    )
    if result.dry_run:
        print("dry run only; rerun with --yes to apply configured retention windows")
        return 0

    db.log_privacy_event(
        "privacy.retention",
        "Applied configured ambient retention windows.",
        severity="info",
        metadata={
            "audio_artifact_count": result.audio_artifact_count,
            "session_count": result.session_count,
            "session_ids": list(result.session_ids),
            "ambient_raw_audio_retention_days": settings.ambient_raw_audio_retention_days,
            "ambient_transcript_retention_days": settings.ambient_transcript_retention_days,
        },
    )
    return 0


def _privacy_purge_has_filter(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None)
        for name in ("session_id", "keyword", "person", "started_on", "before", "after", "mode")
    )


def _privacy_purge_filter_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: value
        for name in ("session_id", "keyword", "person", "started_on", "before", "after", "mode")
        if (value := getattr(args, name, None))
    }


def _print_privacy_purge_matches(sessions: list[dict[str, object]], *, dry_run: bool) -> None:
    prefix = "privacy purge dry run" if dry_run else "privacy purge confirmed"
    print(f"{prefix}: {len(sessions)} matching session(s)")
    for session in sessions:
        title = session.get("title") or "untitled"
        print(
            f"- {session['id']} {session.get('mode')} {session.get('status')} "
            f"{session.get('started_at')} {title} "
            f"({session.get('utterance_count', 0)} utterance(s), "
            f"{session.get('assistant_turn_count', 0)} turn(s))"
        )


def _purge_realtime_artifacts(settings: Settings, session_ids: list[str]) -> int:
    count = 0
    realtime_dir = settings.artifacts_dir / "realtime"
    for session_id in session_ids:
        artifact_dir = realtime_dir / session_id
        if not artifact_dir.exists():
            continue
        shutil.rmtree(artifact_dir)
        count += 1
    return count


def _privacy_report(event_type: str) -> tuple[Settings, Database, dict[str, object]]:
    settings, db = settings_and_db()
    assistant_config = load_assistant_config(settings.assistant_config_path)
    report = privacy_summary(settings, assistant_config)
    db.log_privacy_event(
        event_type,
        f"Local-only status: {report['status']}",
        severity="error" if report["status"] == "error" else "info",
        metadata={
            "issue_count": report["issue_count"],
            "assistant_config_loaded": report["assistant_config_loaded"],
        },
    )
    return settings, db, report


def cmd_memory_extract_ambient(args: argparse.Namespace) -> int:
    _settings, db = settings_and_db()
    try:
        result = extract_memories_from_ambient_session(
            db,
            args.session,
            dry_run=not args.yes,
            max_items=args.limit,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    suffix = "dry run" if result.dry_run else "applied"
    print(
        f"memory extraction {suffix}: "
        f"{len(result.candidates)} candidate(s), "
        f"{len(result.created_ids)} created, "
        f"{result.skipped_duplicates} duplicate(s) skipped"
    )
    if result.status == "skipped":
        print("session mode is not eligible for ambient memory extraction")
    elif result.dry_run:
        print("dry run only; rerun with --yes to store memory candidates")
    return 0


def cmd_ambient(args: argparse.Namespace) -> int:
    settings, db = settings_and_db()
    source = args.source or settings.ambient_source
    mode = args.mode or settings.ambient_mode
    retain_audio = args.retain_audio
    if mode in {"private", "paused"}:
        db.log_privacy_event(
            "ambient.skipped",
            f"Ambient mode {mode} skipped audio processing.",
            metadata={"source": source, "mode": mode},
        )
        print(f"ambient listener {mode}; no audio processed")
        return 0

    try:
        if source == "mic":
            return _run_ambient_mic(args, settings, db, mode, retain_audio)
        source_path = Path(source)
        if source_path.is_dir() and not args.once:
            return _run_ambient_directory_loop(args, settings, db, mode, source_path, retain_audio)
        results = process_ambient_path(
            source_path,
            settings,
            db,
            mode=mode,
            retain_audio=retain_audio,
            vad_threshold=args.vad_threshold,
            min_speech_seconds=args.min_speech_seconds,
        )
    except KeyboardInterrupt:
        print("ambient listener stopped")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for result in results:
        _print_ambient_result(result)
    return 0


def _run_ambient_mic(
    args: argparse.Namespace,
    settings: Settings,
    db: Database,
    mode: str,
    retain_audio: bool | None,
) -> int:
    while True:
        result = process_microphone_once(
            settings,
            db,
            mode=mode,
            seconds=args.chunk_seconds,
            device=args.mic_device,
            retain_audio=retain_audio,
        )
        _print_ambient_result(result)
        if args.once:
            return 0
        time.sleep(args.poll_seconds or settings.ambient_poll_seconds)


def _run_ambient_directory_loop(
    args: argparse.Namespace,
    settings: Settings,
    db: Database,
    mode: str,
    source_path: Path,
    retain_audio: bool | None,
) -> int:
    seen: set[Path] = set()
    while True:
        for path in sorted(source_path.iterdir()):
            resolved = path.resolve()
            if resolved in seen or not is_audio_file(path):
                continue
            result = process_ambient_file(
                path,
                settings,
                db,
                mode=mode,
                retain_audio=retain_audio,
                vad_threshold=args.vad_threshold,
                min_speech_seconds=args.min_speech_seconds,
            )
            seen.add(resolved)
            _print_ambient_result(result)
        time.sleep(args.poll_seconds or settings.ambient_poll_seconds)


def _print_ambient_result(result: AmbientResult) -> None:
    session = result.session_id or "-"
    print(
        f"ambient session {session}: {result.status}; "
        f"{result.utterance_count} utterances from {result.segment_count} segments"
    )


def cmd_validate_brio(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    try:
        result = validate_microphone_asr(
            settings,
            device=args.device,
            seconds=args.seconds,
            allow_stub=args.allow_stub,
            min_transcript_chars=args.min_transcript_chars,
        )
    except Exception as exc:
        print(f"BRIO validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"BRIO validation ok: device={result.device} provider={result.provider} "
        f"audio={result.audio_path} transcript={result.transcript[:120]!r}"
    )
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
