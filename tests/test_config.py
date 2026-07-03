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
                "ATLAS_VOICE_ASR_PROVIDER=vibevoice\n"
                "ATLAS_VOICE_ASR_MODEL=microsoft/VibeVoice-ASR\n"
                "ATLAS_VOICE_DIARIZATION_PROVIDER=transcript\n"
                "ATLAS_VOICE_NEMO_SOURCE_LANG=en\n"
                "ATLAS_VOICE_NEMO_TARGET_LANG=fr\n"
                "ANYTHINGLLM_BASE_URL=http://127.0.0.1:3001/api/v1\n"
                "ANYTHINGLLM_API_KEY=test-key\n"
                "ANYTHINGLLM_WORKSPACE_SLUG=memory\n"
                "ANYTHINGLLM_TIMEOUT=12.5\n"
                "ANYTHINGLLM_AUTO_SYNC=true\n"
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
        self.assertEqual(settings.asr_provider, "vibevoice")
        self.assertEqual(settings.asr_model, "microsoft/VibeVoice-ASR")
        self.assertEqual(settings.diarization_provider, "transcript")
        self.assertEqual(settings.nemo_source_lang, "en")
        self.assertEqual(settings.nemo_target_lang, "fr")
        self.assertEqual(settings.anythingllm_base_url, "http://127.0.0.1:3001/api/v1")
        self.assertEqual(settings.anythingllm_api_key, "test-key")
        self.assertEqual(settings.anythingllm_workspace_slug, "memory")
        self.assertEqual(settings.anythingllm_timeout, 12.5)
        self.assertTrue(settings.anythingllm_auto_sync)


if __name__ == "__main__":
    unittest.main()
