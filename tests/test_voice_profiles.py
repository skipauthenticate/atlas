import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from atlas_voice.assistant_config import load_assistant_config
from atlas_voice.config import Settings
from atlas_voice.profile_settings import settings_for_voice_profile
from atlas_voice.voice_profiles import (
    VOICE_PROFILE_ORDER,
    VoiceProfileError,
    default_voice_profile_id,
    normalize_voice_profile_id,
    public_voice_profiles,
    voice_profile,
    voice_profiles,
)


class VoiceProfileTests(unittest.TestCase):
    def test_defaults_are_allowlisted_ranked_and_torch_preserves_qwen_voice(self) -> None:
        with TemporaryDirectory() as tmp:
            config = load_assistant_config(Path(tmp) / "missing.yaml")

        profiles = voice_profiles(config)

        self.assertEqual(tuple(profiles), VOICE_PROFILE_ORDER)
        self.assertEqual([profile.rank for profile in profiles.values()], [1, 2, 3])
        self.assertEqual(default_voice_profile_id(config), "torch")
        self.assertEqual(profiles["light"].llm_profile, "qwen-voice-light")
        self.assertEqual(profiles["torch"].llm_profile, "qwen-voice")
        self.assertEqual(profiles["fire"].llm_profile, "qwen-voice-fire")
        self.assertEqual(
            [profile.context_budget_chars for profile in profiles.values()],
            [3_000, 6_000, 12_000],
        )
        self.assertEqual(config.llm_profiles["qwen-voice-light"]["model"], "qwen3.5-2b")
        self.assertEqual(config.llm_profiles["qwen-voice"]["model"], "qwen3.5-9b")
        self.assertEqual(config.llm_profiles["qwen-voice-fire"]["model"], "qwen3.6-35b-a3b")
        self.assertNotEqual(profiles["fire"].llm_profile, "qwen-deep")

    def test_router_preset_ids_match_requested_model_ids(self) -> None:
        preset = Path("config/voice-models.example.ini").read_text()

        for model_id in ("qwen3.5-2b", "qwen3.5-9b", "qwen3.6-35b-a3b"):
            self.assertIn(f"[{model_id}]", preset)
        for stale_alias in ("atlas-light", "atlas-torch", "atlas-fire"):
            self.assertNotIn(f"[{stale_alias}]", preset)
        self.assertNotIn("version=1", preset)
        self.assertNotIn("[default]", preset)
        self.assertIn('chat-template-kwargs = {"enable_thinking": false}', preset)
        for filename in (
            "Qwen_Qwen3.5-2B-Q8_0.gguf",
            "Qwen_Qwen3.5-9B-Q8_0.gguf",
            "Qwen3.6-35B-A3B-Q8_0.gguf",
        ):
            self.assertIn(filename, preset)
        self.assertEqual(preset.count("load-on-startup = true"), 1)
        torch_section = preset.split("[qwen3.5-9b]", 1)[1].split("[qwen3.6-35b-a3b]", 1)[0]
        self.assertIn("load-on-startup = true", torch_section)

    def test_unknown_profiles_are_not_configurable_or_resolvable(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "assistant.yaml"
            path.write_text("voice_profiles:\n  spark:\n    llm_profile: qwen-voice\n")
            config = load_assistant_config(path)

        self.assertNotIn("spark", config.voice_profiles)
        with self.assertRaisesRegex(VoiceProfileError, "light, torch, fire"):
            voice_profile(config, "spark")
        with self.assertRaises(VoiceProfileError):
            normalize_voice_profile_id("")

    def test_public_metadata_does_not_expose_urls_or_internal_llm_profile_names(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "assistant.yaml"
            path.write_text(
                "voice_profiles:\n"
                "  torch:\n"
                "    llm_profile: private-torch\n"
                "    context_budget_chars: 2400\n"
                "    base_url: http://private-model.internal/v1/chat/completions\n"
                "llm_profiles:\n"
                "  private-torch:\n"
                "    model: local-private-model\n"
                "    base_url: http://127.0.0.1:9999/v1/chat/completions\n"
            )
            config = load_assistant_config(path)

        public = public_voice_profiles(config)
        torch = next(profile for profile in public if profile["id"] == "torch")
        serialized = json.dumps(public)

        self.assertEqual(torch["context_budget_chars"], 2400)
        self.assertEqual(
            [profile["icon"] for profile in public],
            ["feather.svg", "flame.svg", "flame-kindling.svg"],
        )
        self.assertEqual(
            set(torch),
            {
                "id",
                "name",
                "rank",
                "promise",
                "description",
                "icon",
                "context_budget_chars",
            },
        )
        self.assertNotIn("private-torch", serialized)
        self.assertNotIn("http://", serialized)
        self.assertNotIn("base_url", serialized)

    def test_session_local_resolution_changes_only_the_llm_settings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "assistant.yaml"
            path.write_text(
                "profiles:\n"
                "  direct_voice:\n"
                "    stt_provider: faster-whisper\n"
                "    llm_profile: legacy-voice\n"
                "    tts_provider: piper\n"
                "voice_profiles:\n"
                "  light:\n"
                "    llm_profile: local-light\n"
                "  torch:\n"
                "    llm_profile: local-torch\n"
                "  fire:\n"
                "    llm_profile: local-fire\n"
                "llm_profiles:\n"
                "  legacy-voice:\n"
                "    model: legacy-model\n"
                "    base_url: http://127.0.0.1:8100/v1/chat/completions\n"
                "  local-light:\n"
                "    model: light-model\n"
                "    base_url: http://127.0.0.1:8101/v1/chat/completions\n"
                "    temperature: 0.1\n"
                "    max_tokens: 300\n"
                "  local-torch:\n"
                "    model: torch-model\n"
                "    base_url: http://127.0.0.1:8102/v1/chat/completions\n"
                "    temperature: 0.2\n"
                "    max_tokens: 500\n"
                "  local-fire:\n"
                "    model: fire-model\n"
                "    base_url: http://127.0.0.1:8103/v1/chat/completions\n"
                "    temperature: 0.3\n"
                "    max_tokens: 700\n"
            )
            settings = self._settings(root, assistant_config=path)
            config = load_assistant_config(path)

        legacy = settings_for_voice_profile(settings, config)
        light = settings_for_voice_profile(settings, config, "light")
        torch = settings_for_voice_profile(settings, config, "torch")
        fire = settings_for_voice_profile(settings, config, "fire")

        self.assertEqual(settings.llm_model, "environment-model")
        self.assertEqual(legacy.llm_model, "legacy-model")
        self.assertEqual(light.llm_model, "light-model")
        self.assertEqual(torch.llm_model, "torch-model")
        self.assertEqual(fire.llm_model, "fire-model")
        self.assertEqual(light.llm_base_url, "http://127.0.0.1:8101/v1/chat/completions")
        self.assertEqual(torch.llm_max_tokens, 500)
        self.assertEqual(fire.llm_temperature, 0.3)
        for resolved in (legacy, light, torch, fire):
            self.assertEqual(resolved.asr_provider, "faster-whisper")
            self.assertEqual(resolved.tts_provider, "piper")

    def test_explicit_default_profile_is_applied_without_a_session_override(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "assistant.yaml"
            path.write_text(
                "profiles:\n"
                "  direct_voice:\n"
                "    llm_profile: qwen-voice\n"
                "    voice_profile: fire\n"
                "voice_profiles:\n"
                "  fire:\n"
                "    llm_profile: configured-fire\n"
                "llm_profiles:\n"
                "  configured-fire:\n"
                "    model: configured-fire-model\n"
                "    max_tokens: 900\n"
            )
            settings = self._settings(root, assistant_config=path)
            config = load_assistant_config(path)

        resolved = settings_for_voice_profile(settings, config)

        self.assertEqual(resolved.llm_model, "configured-fire-model")
        self.assertEqual(resolved.llm_max_tokens, 900)

    def test_active_llm_profile_override_replaces_the_candidate_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "assistant.yaml"
            path.write_text(
                "llm_profiles:\n"
                "  qwen-voice:\n"
                "    model: active-tuned-model\n"
                "    base_url: http://127.0.0.1:8181/v1/chat/completions\n"
                "    max_tokens: 555\n"
            )
            settings = self._settings(root, assistant_config=path)
            config = load_assistant_config(path)

        torch = settings_for_voice_profile(settings, config, "torch")

        self.assertEqual(torch.llm_model, "active-tuned-model")
        self.assertEqual(torch.llm_base_url, "http://127.0.0.1:8181/v1/chat/completions")
        self.assertEqual(torch.llm_max_tokens, 555)
        self.assertEqual(config.llm_profiles["qwen-voice-light"]["model"], "qwen3.5-2b")
        self.assertEqual(config.llm_profiles["qwen-voice-fire"]["model"], "qwen3.6-35b-a3b")

    def test_invalid_profile_configuration_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "assistant.yaml"
            path.write_text(
                "voice_profiles:\n"
                "  light:\n"
                "    llm_profile: missing-profile\n"
                "  fire:\n"
                "    context_budget_chars: 100\n"
            )
            config = load_assistant_config(path)

        with self.assertRaisesRegex(VoiceProfileError, "unknown LLM profile"):
            voice_profile(config, "light")

        # Isolate the budget validation from the invalid Light mapping.
        config.raw["voice_profiles"]["light"]["llm_profile"] = "qwen-voice-light"
        with self.assertRaisesRegex(VoiceProfileError, "between 256 and 32000"):
            voice_profile(config, "fire")

    @staticmethod
    def _settings(root: Path, *, assistant_config: Path) -> Settings:
        env = {
            "ATLAS_VOICE_ENV_FILE": str(root / "missing.env"),
            "ATLAS_VOICE_ASSISTANT_CONFIG": str(assistant_config),
            "LLM_MODEL": "environment-model",
            "LLM_BASE_URL": "http://127.0.0.1:8000/v1/chat/completions",
            "ATLAS_VOICE_TTS_PROVIDER": "none",
        }
        with patch.dict(os.environ, env, clear=True):
            return Settings.from_env()


if __name__ == "__main__":
    unittest.main()
