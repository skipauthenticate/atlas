from pathlib import Path
import unittest

from atlas_voice.config import Settings
from atlas_voice.providers.pyannote_provider import diarize_audio


def settings(root: Path, *, fallback: bool) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8787,
        data_dir=root / "data",
        models_dir=root / "models",
        hf_cache_dir=root / "cache" / "huggingface",
        whisperx_model="tiny.en",
        whisperx_device="cpu",
        whisperx_compute_type="int8",
        pyannote_model="pyannote/speaker-diarization-community-1",
        hf_token=None,
        llm_base_url="http://127.0.0.1:8080/v1/chat/completions",
        llm_model="qwen-local",
        llm_temperature=0.2,
        llm_max_tokens=100,
        stub_mode=False,
        allow_single_speaker_fallback=fallback,
    )


class PyannoteProviderTests(unittest.TestCase):
    def test_single_speaker_fallback_does_not_require_hf_token(self) -> None:
        turns = diarize_audio(Path("missing.wav"), settings(Path("/tmp/atlas-test"), fallback=True))

        self.assertEqual(turns, [{"start": 0.0, "end": 3600.0, "speaker": "SPEAKER_00"}])

    def test_missing_token_is_strict_by_default(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "HF_TOKEN is required"):
            diarize_audio(Path("missing.wav"), settings(Path("/tmp/atlas-test"), fallback=False))


if __name__ == "__main__":
    unittest.main()
