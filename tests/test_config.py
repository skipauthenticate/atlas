from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

from atlas_voice.assistant_config import load_assistant_config
from atlas_voice.prompts import PromptRegistryError, load_prompt_registry
from atlas_voice.tools import ToolRegistryError, load_tool_registry
from atlas_voice.config import Settings
from atlas_voice.profile_settings import settings_for_profile


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

    def test_exact_realtime_host_flag_overrides_voice_host(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "ATLAS_REALTIME_HOST": "127.0.0.1",
                "ATLAS_VOICE_HOST": "0.0.0.0",
            }
            with patch.dict(os.environ, env, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)

        self.assertEqual(settings.host, "127.0.0.1")

    def test_exact_tts_base_url_flag_overrides_namespaced_alias(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "ATLAS_TTS_BASE_URL": "http://127.0.0.1:8008/v1/audio/speech",
                "ATLAS_VOICE_TTS_BASE_URL": "http://127.0.0.1:9999/v1/audio/speech",
            }
            with patch.dict(os.environ, env, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)

        self.assertEqual(settings.tts_base_url, "http://127.0.0.1:8008/v1/audio/speech")

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
                "ATLAS_VOICE_HYPRWHSPR_HEALTH_URL=http://127.0.0.1:9000/health\n"
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
                "ATLAS_VOICE_REALTIME_VAD_ENABLED=false\n"
                "ATLAS_VOICE_REALTIME_VAD_THRESHOLD=1200\n"
                "ATLAS_VOICE_REALTIME_VAD_MIN_SPEECH_MS=150\n"
                "ATLAS_VOICE_REALTIME_VAD_SILENCE_MS=350\n"
                "ATLAS_VOICE_AMBIENT_SOURCE=./ambient-inbox\n"
                "ATLAS_VOICE_AMBIENT_MODE=meeting\n"
                "ATLAS_VOICE_AMBIENT_CHUNK_SECONDS=3.5\n"
                "ATLAS_VOICE_AMBIENT_POLL_SECONDS=0.5\n"
                "ATLAS_VOICE_AMBIENT_MIC_DEVICE=hw:1,0\n"
                "ATLAS_VOICE_AMBIENT_VAD_THRESHOLD=700\n"
                "ATLAS_VOICE_AMBIENT_MIN_SPEECH_SECONDS=0.25\n"
                "ATLAS_VOICE_AMBIENT_RETAIN_AUDIO=true\n"
                "ATLAS_VOICE_AMBIENT_RAW_AUDIO_RETENTION_DAYS=2.5\n"
                "ATLAS_VOICE_AMBIENT_TRANSCRIPT_RETENTION_DAYS=14\n"
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
        self.assertEqual(settings.hyprwhspr_health_url, "http://127.0.0.1:9000/health")
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
        self.assertFalse(settings.realtime_vad_enabled)
        self.assertEqual(settings.realtime_vad_threshold, 1200)
        self.assertEqual(settings.realtime_vad_min_speech_ms, 150)
        self.assertEqual(settings.realtime_vad_silence_ms, 350)
        self.assertEqual(settings.ambient_source, "./ambient-inbox")
        self.assertEqual(settings.ambient_mode, "meeting")
        self.assertEqual(settings.ambient_chunk_seconds, 3.5)
        self.assertEqual(settings.ambient_poll_seconds, 0.5)
        self.assertEqual(settings.ambient_mic_device, "hw:1,0")
        self.assertEqual(settings.ambient_vad_threshold, 700)
        self.assertEqual(settings.ambient_min_speech_seconds, 0.25)
        self.assertTrue(settings.ambient_retain_audio)
        self.assertEqual(settings.ambient_raw_audio_retention_days, 2.5)
        self.assertEqual(settings.ambient_transcript_retention_days, 14)

    def test_ambient_transcript_retention_defaults_to_indefinite(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {}, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)

        self.assertEqual(settings.ambient_raw_audio_retention_days, 0)
        self.assertIsNone(settings.ambient_transcript_retention_days)

    def test_prompt_registry_loads_default_and_custom_yaml_prompts(self) -> None:
        with TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp) / "config" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "custom.yaml").write_text(
                "prompts:\n"
                "  custom_reflection:\n"
                "    name: Custom Reflection\n"
                "    domain: coaching\n"
                "    version: 1\n"
                "    system: Ask one direct reflective question.\n"
                "    user: Summarize the user's current goal.\n"
                "    rubric:\n"
                "      clarity: Check whether the ask is specific.\n"
                "      follow_through: Check whether a next action exists.\n"
            )

            registry = load_prompt_registry(prompts_dir)

        default_prompt = registry.get("direct_voice_assistant")
        custom_prompt = registry.get("custom_reflection")
        self.assertIn("direct_voice_assistant", registry.ids())
        self.assertEqual(default_prompt.domain, "voice")
        self.assertIn("private local realtime assistant", default_prompt.system.lower())
        self.assertEqual(custom_prompt.name, "Custom Reflection")
        self.assertEqual(custom_prompt.version, "1")
        self.assertEqual(custom_prompt.rubric["clarity"], "Check whether the ask is specific.")
        self.assertIn("Custom Reflection", registry.as_dict()["custom_reflection"]["name"])

    def test_prompt_registry_rejects_invalid_prompt_yaml(self) -> None:
        with TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp) / "prompts"
            prompts_dir.mkdir()
            (prompts_dir / "broken.yaml").write_text(
                "prompts:\n"
                "  broken:\n"
                "    name: Broken\n"
                "    domain: coaching\n"
                "    system: Missing user template.\n"
            )

            with self.assertRaises(PromptRegistryError):
                load_prompt_registry(prompts_dir)

    def test_tool_registry_loads_default_and_custom_yaml_tools(self) -> None:
        with TemporaryDirectory() as tmp:
            tools_dir = Path(tmp) / "config" / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "custom.yaml").write_text(
                "tools:\n"
                "  draft_message:\n"
                "    name: Draft Message\n"
                "    description: Draft local text without sending it.\n"
                "    handler: atlas_voice.tools.draft_message\n"
                "    mutating: false\n"
                "    permission: allow\n"
                "    parameters:\n"
                "      recipient: Person receiving the draft.\n"
                "      topic: Topic to draft about.\n"
            )

            registry = load_tool_registry(tools_dir)

        default_tool = registry.get("search_recordings")
        custom_tool = registry.get("draft_message")
        self.assertIn("search_recordings", registry.ids())
        self.assertEqual(default_tool.permission, "allow")
        self.assertFalse(default_tool.requires_confirmation)
        self.assertEqual(custom_tool.name, "Draft Message")
        self.assertEqual(custom_tool.parameters["recipient"], "Person receiving the draft.")
        self.assertFalse(custom_tool.requires_confirmation)
        self.assertTrue(registry.get("privacy_purge").requires_confirmation)
        self.assertEqual(registry.as_dict()["draft_message"]["handler"], "atlas_voice.tools.draft_message")

    def test_tool_registry_rejects_invalid_permission_policy(self) -> None:
        with TemporaryDirectory() as tmp:
            tools_dir = Path(tmp) / "tools"
            tools_dir.mkdir()
            (tools_dir / "broken.yaml").write_text(
                "tools:\n"
                "  broken:\n"
                "    name: Broken\n"
                "    description: Invalid permission.\n"
                "    handler: atlas_voice.tools.broken\n"
                "    permission: maybe\n"
            )

            with self.assertRaises(ToolRegistryError):
                load_tool_registry(tools_dir)

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

    def test_profile_settings_apply_explicit_provider_overrides(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "assistant.yaml"
            path.write_text(
                "profiles:\n"
                "  direct_voice:\n"
                "    stt_provider: parakeet\n"
                "    asr_model: nvidia/parakeet-tdt-0.6b-v3\n"
                "    tts_provider: piper\n"
                "    tts_model: piper-local\n"
                "  ambient:\n"
                "    stt_provider: hyprwhspr\n"
                "    source: ./ambient-inbox\n"
                "    chunk_seconds: 2.5\n"
                "  reflection:\n"
                "    asr_provider: canary\n"
                "    diarization_provider: none\n"
            )
            env = {
                "ATLAS_VOICE_ASSISTANT_CONFIG": str(path),
                "ATLAS_VOICE_ASR_PROVIDER": "whisperx",
                "ATLAS_VOICE_ASR_MODEL": "env-model",
                "ATLAS_VOICE_TTS_PROVIDER": "none",
                "ATLAS_VOICE_AMBIENT_SOURCE": "mic",
                "ATLAS_VOICE_AMBIENT_CHUNK_SECONDS": "15",
            }
            with patch.dict(os.environ, env, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)
                config = load_assistant_config(settings.assistant_config_path)

        direct = settings_for_profile(settings, config, "direct_voice")
        ambient = settings_for_profile(settings, config, "ambient")
        reflection = settings_for_profile(settings, config, "reflection")

        self.assertEqual(settings.asr_provider, "whisperx")
        self.assertEqual(direct.asr_provider, "parakeet")
        self.assertEqual(direct.asr_model, "nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(direct.tts_provider, "piper")
        self.assertEqual(direct.tts_model, "piper-local")
        self.assertEqual(ambient.asr_provider, "hyprwhspr")
        self.assertEqual(ambient.ambient_source, "./ambient-inbox")
        self.assertEqual(ambient.ambient_chunk_seconds, 2.5)
        self.assertEqual(reflection.asr_provider, "canary")
        self.assertEqual(reflection.diarization_provider, "none")


    def test_direct_voice_defaults_to_faster_qwen3_tts_when_tts_unset(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {}, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)
                config = load_assistant_config(root / "missing.yaml")

        direct = settings_for_profile(settings, config, "direct_voice")

        self.assertEqual(config.profiles["direct_voice"]["tts_provider"], "faster-qwen3-tts")
        self.assertEqual(direct.tts_provider, "faster-qwen3-tts")
        self.assertEqual(direct.tts_model, "faster-qwen3-tts-0.6b")
        self.assertEqual(direct.tts_base_url, "http://127.0.0.1:8008/v1/audio/speech")

    def test_explicit_tts_none_disables_direct_voice_default_tts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"ATLAS_VOICE_TTS_PROVIDER": "none"}, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)
                config = load_assistant_config(root / "missing.yaml")

        direct = settings_for_profile(settings, config, "direct_voice")

        self.assertEqual(direct.tts_provider, "none")

    def test_default_profile_values_do_not_override_environment(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "ATLAS_VOICE_ASSISTANT_CONFIG": str(root / "missing.yaml"),
                "ATLAS_VOICE_ASR_PROVIDER": "vibevoice",
                "ATLAS_VOICE_ASR_MODEL": "env-model",
                "ATLAS_VOICE_TTS_PROVIDER": "piper",
            }
            with patch.dict(os.environ, env, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)
                config = load_assistant_config(settings.assistant_config_path)

        ambient = settings_for_profile(settings, config, "ambient")
        direct = settings_for_profile(settings, config, "direct_voice")

        self.assertFalse(config.loaded)
        self.assertEqual(ambient.asr_provider, "vibevoice")
        self.assertEqual(ambient.asr_model, "env-model")
        self.assertEqual(direct.tts_provider, "piper")

    def test_qwen_deep_llm_profile_defaults_to_on_demand_roles(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"LLM_MODEL": "qwen-local"}, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)
                config = load_assistant_config(root / "missing.yaml")

        profile = config.llm_profiles["qwen-deep"]
        reflection = settings_for_profile(settings, config, "reflection")

        self.assertEqual(profile["model"], "qwen-27b-instruct")
        self.assertEqual(profile["load_policy"], "on_demand")
        self.assertIn("deep_coaching", profile["roles"])
        self.assertIn("weekly_reviews", profile["roles"])
        self.assertEqual(settings.llm_model, "qwen-local")
        self.assertEqual(reflection.llm_model, "qwen-27b-instruct")

    def test_qwen_deep_llm_profile_accepts_local_overrides(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "assistant.yaml"
            path.write_text(
                "profiles:\n"
                "  reflection:\n"
                "    llm_profile: qwen-deep\n"
                "llm_profiles:\n"
                "  qwen-deep:\n"
                "    model: local-qwen-27b\n"
                "    base_url: http://127.0.0.1:8088/v1/chat/completions\n"
                "    temperature: 0.35\n"
                "    max_tokens: 4096\n"
                "    load_policy: on_demand\n"
                "    roles: [deep_coaching, complex_reasoning, weekly_reviews]\n"
            )
            env = {
                "ATLAS_VOICE_ASSISTANT_CONFIG": str(path),
                "LLM_MODEL": "qwen-local",
                "LLM_BASE_URL": "http://127.0.0.1:8080/v1/chat/completions",
                "LLM_TEMPERATURE": "0.2",
                "LLM_MAX_TOKENS": "1200",
            }
            with patch.dict(os.environ, env, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)
                config = load_assistant_config(settings.assistant_config_path)

        reflection = settings_for_profile(settings, config, "reflection")
        direct = settings_for_profile(settings, config, "direct_voice")

        self.assertEqual(reflection.llm_model, "local-qwen-27b")
        self.assertEqual(reflection.llm_base_url, "http://127.0.0.1:8088/v1/chat/completions")
        self.assertEqual(reflection.llm_temperature, 0.35)
        self.assertEqual(reflection.llm_max_tokens, 4096)
        self.assertEqual(direct.llm_model, "qwen2.5-7b-instruct")
        self.assertNotEqual(direct.llm_model, reflection.llm_model)

    def test_smaller_llm_profiles_drive_voice_and_classifier_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"LLM_MODEL": "qwen-local"}, clear=True):
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    settings = Settings.from_env()
                finally:
                    os.chdir(cwd)
                config = load_assistant_config(root / "missing.yaml")

        voice_profile = config.llm_profiles["qwen-voice"]
        classifier_profile = config.llm_profiles["small-classifier"]
        direct = settings_for_profile(settings, config, "direct_voice")
        ambient = settings_for_profile(settings, config, "ambient")

        self.assertEqual(voice_profile["load_policy"], "warm_optional")
        self.assertIn("fast_voice_replies", voice_profile["roles"])
        self.assertEqual(direct.llm_model, "qwen2.5-7b-instruct")
        self.assertLessEqual(direct.llm_max_tokens, 800)
        self.assertEqual(classifier_profile["load_policy"], "hot_optional")
        self.assertIn("intent_classification", classifier_profile["roles"])
        self.assertEqual(ambient.llm_model, "qwen2.5-0.5b-instruct")
        self.assertLessEqual(ambient.llm_max_tokens, 256)


if __name__ == "__main__":
    unittest.main()
