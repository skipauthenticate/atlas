from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest import mock

from atlas_voice.providers.whisperx_provider import (
    _clear_model_caches,
    _ensure_ctranslate2_cuda,
    transcribe_audio,
)


class WhisperXProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_model_caches()

    def tearDown(self) -> None:
        _clear_model_caches()

    def test_cuda_device_requires_ctranslate2_cuda(self) -> None:
        fake_ctranslate2 = types.ModuleType("ctranslate2")
        fake_ctranslate2.get_cuda_device_count = lambda: 0

        with mock.patch.dict(sys.modules, {"ctranslate2": fake_ctranslate2}):
            with self.assertRaisesRegex(RuntimeError, "CTranslate2 CUDA is not available"):
                _ensure_ctranslate2_cuda("cuda")

    def test_reuses_asr_and_alignment_models(self) -> None:
        calls = {
            "load_model": 0,
            "load_align_model": 0,
            "transcribe": 0,
            "align": 0,
        }

        class Model:
            def transcribe(self, _path: str, *, batch_size: int):
                calls["transcribe"] += 1
                self.batch_size = batch_size
                return {
                    "language": "en",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
                }

        fake_whisperx = types.ModuleType("whisperx")

        def load_model(*_args, **_kwargs):
            calls["load_model"] += 1
            return Model()

        def load_align_model(**_kwargs):
            calls["load_align_model"] += 1
            return object(), {"language": "en"}

        def align(segments, *_args, **_kwargs):
            calls["align"] += 1
            return {"segments": list(segments)}

        fake_whisperx.load_model = load_model
        fake_whisperx.load_align_model = load_align_model
        fake_whisperx.align = align

        provider_settings = self._settings()
        with mock.patch.dict(sys.modules, {"whisperx": fake_whisperx}):
            first = transcribe_audio(Path("one.wav"), provider_settings)
            second = transcribe_audio(Path("two.wav"), provider_settings)

        self.assertEqual(calls["load_model"], 1)
        self.assertEqual(calls["load_align_model"], 1)
        self.assertEqual(calls["transcribe"], 2)
        self.assertEqual(calls["align"], 2)
        self.assertEqual(first["language"], "en")
        self.assertEqual(second["language"], "en")

    def test_serializes_inference_for_a_cached_model(self) -> None:
        state_lock = threading.Lock()
        active = 0
        max_active = 0

        class Model:
            def transcribe(self, _path: str, *, batch_size: int):
                nonlocal active, max_active
                self.batch_size = batch_size
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.02)
                with state_lock:
                    active -= 1
                return {"language": None, "segments": []}

        fake_whisperx = types.ModuleType("whisperx")
        fake_whisperx.load_model = lambda *_args, **_kwargs: Model()
        provider_settings = self._settings()

        with mock.patch.dict(sys.modules, {"whisperx": fake_whisperx}):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        transcribe_audio,
                        Path(f"audio-{index}.wav"),
                        provider_settings,
                    )
                    for index in range(2)
                ]
                for future in futures:
                    future.result()

        self.assertEqual(max_active, 1)

    @staticmethod
    def _settings() -> types.SimpleNamespace:
        return types.SimpleNamespace(
            stub_mode=False,
            whisperx_device="cpu",
            whisperx_model="tiny.en",
            whisperx_compute_type="int8",
            models_dir=Path("/tmp/atlas-whisperx-models"),
        )


if __name__ == "__main__":
    unittest.main()
