from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from atlas_voice.assistant_config import load_assistant_config
from atlas_voice.config import Settings
from atlas_voice.privacy import privacy_summary, validate_local_only


class PrivacyTests(unittest.TestCase):
    def test_default_local_settings_have_no_privacy_issues(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = _settings(root)
            config = load_assistant_config(root / "missing.yaml")

            report = privacy_summary(settings, config)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["issue_count"], 0)

    def test_external_urls_and_binds_are_errors(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "assistant.yaml"
            config_path.write_text(
                "profiles:\n"
                "  direct_voice:\n"
                "    enabled: true\n"
                "    bind: 0.0.0.0\n"
                "privacy:\n"
                "  telemetry: true\n"
            )
            settings = _settings(
                root,
                host="0.0.0.0",
                llm_base_url="https://api.example.com/v1/chat/completions",
            )
            config = load_assistant_config(config_path)

            issues = validate_local_only(settings, config)

        checks = {issue.check for issue in issues}
        self.assertIn("web_bind", checks)
        self.assertIn("llm_base_url", checks)
        self.assertIn("telemetry", checks)
        self.assertIn("profile.direct_voice.bind", checks)


def _settings(root: Path, **overrides: object) -> Settings:
    values = {
        "host": "127.0.0.1",
        "port": 8787,
        "data_dir": root / "data",
        "models_dir": root / "models",
        "hf_cache_dir": root / "cache" / "huggingface",
        "whisperx_model": "tiny.en",
        "whisperx_device": "cpu",
        "whisperx_compute_type": "int8",
        "pyannote_model": "pyannote/speaker-diarization-community-1",
        "hf_token": None,
        "llm_base_url": "http://127.0.0.1:8080/v1/chat/completions",
        "llm_model": "qwen-local",
        "llm_temperature": 0.2,
        "llm_max_tokens": 1200,
        "stub_mode": True,
        "anythingllm_base_url": "http://127.0.0.1:3001/api",
    }
    values.update(overrides)
    return Settings(**values)


if __name__ == "__main__":
    unittest.main()
