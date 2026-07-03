import sys
import types
import unittest
from unittest import mock

from atlas_voice.providers.whisperx_provider import _ensure_ctranslate2_cuda


class WhisperXProviderTests(unittest.TestCase):
    def test_cuda_device_requires_ctranslate2_cuda(self) -> None:
        fake_ctranslate2 = types.ModuleType("ctranslate2")
        fake_ctranslate2.get_cuda_device_count = lambda: 0

        with mock.patch.dict(sys.modules, {"ctranslate2": fake_ctranslate2}):
            with self.assertRaisesRegex(RuntimeError, "CTranslate2 CUDA is not available"):
                _ensure_ctranslate2_cuda("cuda")


if __name__ == "__main__":
    unittest.main()
