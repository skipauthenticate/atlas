from __future__ import annotations

import logging
import signal
import time
from pathlib import Path

from .config import Settings
from .database import Database
from .pipeline import PipelineProcessor
from .storage import is_audio_file


LOGGER = logging.getLogger("atlas_voice.worker")


class Worker:
    def __init__(self, settings: Settings, db: Database, poll_seconds: float = 2.0):
        self.settings = settings
        self.db = db
        self.processor = PipelineProcessor(settings, db)
        self.poll_seconds = poll_seconds
        self._stop = False

    def install_signal_handlers(self) -> None:
        def _handle_stop(signum: int, _frame: object) -> None:
            LOGGER.info("Received signal %s; stopping after current job", signum)
            self._stop = True

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

    def run_forever(self) -> None:
        self.settings.ensure_directories()
        self.db.initialize()
        self.install_signal_handlers()
        LOGGER.info("Worker started; watching %s", self.settings.inbox_dir)
        while not self._stop:
            self.scan_inbox()
            did_work = self.processor.process_next()
            if not did_work:
                time.sleep(self.poll_seconds)

    def scan_inbox(self) -> list[str]:
        self.settings.inbox_dir.mkdir(parents=True, exist_ok=True)
        recording_ids: list[str] = []
        for path in sorted(self.settings.inbox_dir.rglob("*")):
            if not is_audio_file(path) or not self._stable(path):
                continue
            existing = self.db.get_recording_by_source_path(path)
            if existing is not None:
                continue
            recording_id = self.processor.enqueue_source(path)
            LOGGER.info("Queued inbox file %s as %s", path, recording_id)
            recording_ids.append(recording_id)
        return recording_ids

    @staticmethod
    def _stable(path: Path) -> bool:
        try:
            return time.time() - path.stat().st_mtime > 2
        except FileNotFoundError:
            return False
