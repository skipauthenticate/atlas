from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

from atlas_voice.config import Settings


class ConfigTests(unittest.TestCase):
    def test_from_env_loads_dotenv_without_overriding_environment(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "ATLAS_VOICE_PORT=9999\n"
                "WHISPERX_MODEL=tiny.en\n"
                "ATLAS_VOICE_ALLOW_SINGLE_SPEAKER_FALLBACK=true\n"
            )
            with patch.dict(os.environ, {"ATLAS_VOICE_PORT": "7777"}, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)

        self.assertEqual(settings.port, 7777)
        self.assertEqual(settings.whisperx_model, "tiny.en")
        self.assertTrue(settings.allow_single_speaker_fallback)


if __name__ == "__main__":
    unittest.main()
