from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

from atlas_voice.assistant_config import load_assistant_config
from atlas_voice.config import Settings


class ConfigTests(unittest.TestCase):
    def test_assistant_enabled_defaults_off_and_reads_exact_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {}, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)

        self.assertFalse(settings.assistant_enabled)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"ATLAS_ASSISTANT_ENABLED": "true"}, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)

        self.assertTrue(settings.assistant_enabled)

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
                "ATLAS_VOICE_ASSISTANT_CONFIG=./assistant.yaml\n"
                "ATLAS_VOICE_TTS_PROVIDER=piper\n"
                "ATLAS_TTS_BASE_URL=http://127.0.0.1:8008/v1/audio/speech\n"
                "ATLAS_TTS_HEALTH_URL=http://127.0.0.1:8008/health\n"
                "ATLAS_TTS_MODEL=faster-qwen3-tts-0.6b\n"
                "ATLAS_TTS_VOICE=atlas\n"
                "ATLAS_TTS_RESPONSE_FORMAT=wav\n"
                "ATLAS_TTS_TIMEOUT=12\n"
                "ATLAS_TTS_HEALTH_TIMEOUT=0.75\n"
                "ATLAS_VOICE_PIPER_EXECUTABLE=/usr/local/bin/piper\n"
                "ATLAS_VOICE_PIPER_VOICE=/models/voice.onnx\n"
                "ATLAS_VOICE_REALTIME_AUDIO_SAMPLE_RATE=16000\n"
                "ATLAS_VOICE_REALTIME_AUDIO_CHANNELS=2\n"
                "ATLAS_VOICE_AMBIENT_SOURCE=./ambient-inbox\n"
                "ATLAS_VOICE_AMBIENT_MODE=meeting\n"
                "ATLAS_VOICE_AMBIENT_CHUNK_SECONDS=3.5\n"
                "ATLAS_VOICE_AMBIENT_POLL_SECONDS=0.5\n"
                "ATLAS_VOICE_AMBIENT_MIC_DEVICE=hw:1,0\n"
                "ATLAS_VOICE_AMBIENT_VAD_THRESHOLD=700\n"
                "ATLAS_VOICE_AMBIENT_MIN_SPEECH_SECONDS=0.25\n"
                "ATLAS_VOICE_AMBIENT_RETAIN_AUDIO=true\n"
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
        self.assertEqual(settings.assistant_config_path, (Path(tmp) / "assistant.yaml").resolve())
        self.assertEqual(settings.tts_provider, "piper")
        self.assertEqual(settings.tts_base_url, "http://127.0.0.1:8008/v1/audio/speech")
        self.assertEqual(settings.tts_health_url, "http://127.0.0.1:8008/health")
        self.assertEqual(settings.tts_model, "faster-qwen3-tts-0.6b")
        self.assertEqual(settings.tts_voice, "atlas")
        self.assertEqual(settings.tts_response_format, "wav")
        self.assertEqual(settings.tts_timeout, 12)
        self.assertEqual(settings.tts_health_timeout, 0.75)
        self.assertEqual(settings.piper_executable, "/usr/local/bin/piper")
        self.assertEqual(settings.piper_voice, "/models/voice.onnx")
        self.assertEqual(settings.realtime_audio_sample_rate, 16000)
        self.assertEqual(settings.realtime_audio_channels, 2)
        self.assertEqual(settings.ambient_source, "./ambient-inbox")
        self.assertEqual(settings.ambient_mode, "meeting")
        self.assertEqual(settings.ambient_chunk_seconds, 3.5)
        self.assertEqual(settings.ambient_poll_seconds, 0.5)
        self.assertEqual(settings.ambient_mic_device, "hw:1,0")
        self.assertEqual(settings.ambient_vad_threshold, 700)
        self.assertEqual(settings.ambient_min_speech_seconds, 0.25)
        self.assertTrue(settings.ambient_retain_audio)

    def test_assistant_config_loader_merges_yaml_with_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "assistant.yaml"
            path.write_text(
                "profiles:\n"
                "  direct_voice:\n"
                "    enabled: true\n"
                "    realtime_backend: hf-speech-to-speech\n"
                "privacy:\n"
                "  allowed_hosts: [127.0.0.1, localhost, llm]\n"
                "  raw_audio_retention_seconds: 15\n"
            )

            config = load_assistant_config(path)

        self.assertTrue(config.loaded)
        self.assertTrue(config.profiles["direct_voice"]["enabled"])
        self.assertEqual(
            config.profiles["direct_voice"]["realtime_backend"],
            "hf-speech-to-speech",
        )
        self.assertIn("ambient", config.profiles)
        self.assertEqual(config.privacy["raw_audio_retention_seconds"], 15)
        self.assertIn("llm", config.privacy["allowed_hosts"])


if __name__ == "__main__":
    unittest.main()
