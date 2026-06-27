from pathlib import Path
import sys
import types
import unittest
from unittest import mock

from atlas_voice.config import Settings
from atlas_voice.providers.pyannote_provider import diarize_audio


def settings(root: Path, *, fallback: bool, token: str | None = None) -> Settings:
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
        hf_token=token,
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

    def test_uses_pyannote_v4_token_keyword_and_preloaded_audio(self) -> None:
        class Turn:
            start = 1.0
            end = 2.5

        class Diarization:
            def itertracks(self, yield_label: bool = False):
                yield Turn(), None, "SPEAKER_01"

        class Pipeline:
            kwargs = None
            audio = None

            @classmethod
            def from_pretrained(cls, model: str, **kwargs):
                cls.kwargs = kwargs
                return cls()

            def __call__(self, audio):
                type(self).audio = audio
                return Diarization()

        fake_pyannote = types.ModuleType("pyannote")
        fake_audio = types.ModuleType("pyannote.audio")
        fake_audio.Pipeline = Pipeline
        fake_torchaudio = types.ModuleType("torchaudio")
        fake_torchaudio.load = lambda path: ("waveform", 16000)

        modules = {
            "pyannote": fake_pyannote,
            "pyannote.audio": fake_audio,
            "torchaudio": fake_torchaudio,
        }
        with mock.patch.dict(sys.modules, modules):
            turns = diarize_audio(
                Path("audio.wav"),
                settings(Path("/tmp/atlas-test"), fallback=False, token="hf_test"),
            )

        self.assertEqual(Pipeline.kwargs, {"token": "hf_test"})
        self.assertEqual(Pipeline.audio, {"waveform": "waveform", "sample_rate": 16000})
        self.assertEqual(turns, [{"start": 1.0, "end": 2.5, "speaker": "SPEAKER_01"}])

    def test_falls_back_to_legacy_pyannote_auth_keyword(self) -> None:
        class Turn:
            start = 0.0
            end = 1.0

        class Diarization:
            def itertracks(self, yield_label: bool = False):
                yield Turn(), None, "SPEAKER_00"

        class Pipeline:
            kwargs = None

            @classmethod
            def from_pretrained(cls, model: str, **kwargs):
                if "token" in kwargs:
                    raise TypeError("unexpected keyword argument 'token'")
                cls.kwargs = kwargs
                return cls()

            def __call__(self, audio):
                return Diarization()

        fake_pyannote = types.ModuleType("pyannote")
        fake_audio = types.ModuleType("pyannote.audio")
        fake_audio.Pipeline = Pipeline
        fake_torchaudio = types.ModuleType("torchaudio")
        fake_torchaudio.load = lambda path: ("waveform", 16000)
        modules = {
            "pyannote": fake_pyannote,
            "pyannote.audio": fake_audio,
            "torchaudio": fake_torchaudio,
        }
        with mock.patch.dict(sys.modules, modules):
            diarize_audio(
                Path("audio.wav"),
                settings(Path("/tmp/atlas-test"), fallback=False, token="hf_test"),
            )

        self.assertEqual(Pipeline.kwargs, {"use_auth_token": "hf_test"})

    def test_reads_pyannote_v4_diarize_output_wrapper(self) -> None:
        class Turn:
            start = 3.0
            end = 4.0

        class Annotation:
            def itertracks(self, yield_label: bool = False):
                yield Turn(), None, "SPEAKER_02"

        class Output:
            exclusive_speaker_diarization = Annotation()
            speaker_diarization = None

        class Pipeline:
            @classmethod
            def from_pretrained(cls, model: str, **kwargs):
                return cls()

            def __call__(self, audio):
                return Output()

        fake_pyannote = types.ModuleType("pyannote")
        fake_audio = types.ModuleType("pyannote.audio")
        fake_audio.Pipeline = Pipeline
        fake_torchaudio = types.ModuleType("torchaudio")
        fake_torchaudio.load = lambda path: ("waveform", 16000)
        modules = {
            "pyannote": fake_pyannote,
            "pyannote.audio": fake_audio,
            "torchaudio": fake_torchaudio,
        }
        with mock.patch.dict(sys.modules, modules):
            turns = diarize_audio(
                Path("audio.wav"),
                settings(Path("/tmp/atlas-test"), fallback=False, token="hf_test"),
            )

        self.assertEqual(turns, [{"start": 3.0, "end": 4.0, "speaker": "SPEAKER_02"}])


if __name__ == "__main__":
    unittest.main()
