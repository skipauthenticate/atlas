from pathlib import Path
import sys
import tempfile
import types
import unittest
import wave
from unittest import mock

from atlas_voice.config import Settings
from atlas_voice.providers import pyannote_provider
from atlas_voice.providers.pyannote_provider import diarize_audio


def settings(
    root: Path, *, fallback: bool, token: str | None = None, device: str = "cpu"
) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8787,
        data_dir=root / "data",
        models_dir=root / "models",
        hf_cache_dir=root / "cache" / "huggingface",
        whisperx_model="tiny.en",
        whisperx_device=device,
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
    def setUp(self) -> None:
        pyannote_provider._clear_pipeline_cache()

    def test_single_speaker_fallback_does_not_require_hf_token(self) -> None:
        turns = diarize_audio(Path("missing.wav"), settings(Path("/tmp/atlas-test"), fallback=True))

        self.assertEqual(turns, [{"start": 0.0, "end": 3600.0, "speaker": "SPEAKER_00"}])

    def test_single_speaker_fallback_uses_audio_duration_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.wav"
            sample_rate = 16000
            seconds = 3.25
            frame_count = int(sample_rate * seconds)
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(sample_rate)
                audio.writeframes(b"\0\0" * frame_count)

            turns = diarize_audio(audio_path, settings(Path(tmpdir), fallback=True))

        self.assertEqual(turns[0]["start"], 0.0)
        self.assertAlmostEqual(turns[0]["end"], seconds)
        self.assertEqual(turns[0]["speaker"], "SPEAKER_00")

    def test_missing_token_is_strict_by_default(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "HF_TOKEN is required"):
            diarize_audio(Path("missing.wav"), settings(Path("/tmp/atlas-test"), fallback=False))

    def test_fallback_setting_does_not_skip_an_available_pipeline(self) -> None:
        class Turn:
            start = 0.0
            end = 1.0

        class Diarization:
            def itertracks(self, yield_label: bool = False):
                yield Turn(), None, "SPEAKER_07"

        class Pipeline:
            loads = 0
            calls = 0

            @classmethod
            def from_pretrained(cls, model: str, **kwargs):
                cls.loads += 1
                cls.model = model
                cls.kwargs = kwargs
                return cls()

            def __call__(self, audio):
                type(self).calls += 1
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
        provider_settings = settings(
            Path("/tmp/atlas-test"),
            fallback=True,
            token="hf_test",
        )

        with mock.patch.dict(sys.modules, modules):
            first = diarize_audio(Path("one.wav"), provider_settings)
            second = diarize_audio(Path("two.wav"), provider_settings)

        self.assertEqual(Pipeline.loads, 1)
        self.assertEqual(Pipeline.calls, 2)
        self.assertEqual(first[0]["speaker"], "SPEAKER_07")
        self.assertEqual(second[0]["speaker"], "SPEAKER_07")

    def test_expected_speakers_must_be_a_positive_integer(self) -> None:
        provider_settings = settings(Path("/tmp/atlas-test"), fallback=True)

        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    diarize_audio(
                        Path("missing.wav"),
                        provider_settings,
                        expected_speakers=invalid,
                    )

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

    def test_long_audio_is_diarized_in_overlapped_chunks(self) -> None:
        class AudioInfo:
            sample_rate = 100
            num_frames = 2500

        load_calls = []

        def load(path, frame_offset=0, num_frames=-1):
            load_calls.append((frame_offset, num_frames))
            return f"waveform-{len(load_calls)}", 100

        class Turn:
            def __init__(self, start: float, end: float):
                self.start = start
                self.end = end

        class Diarization:
            def __init__(self, rows):
                self.rows = rows

            def itertracks(self, yield_label: bool = False):
                for start, end, speaker in self.rows:
                    yield Turn(start, end), None, speaker

        chunks = [
            [(0.0, 9.0, "SPEAKER_00"), (8.5, 11.5, "SPEAKER_01")],
            [(0.5, 2.0, "SPEAKER_00"), (2.1, 12.0, "SPEAKER_00")],
            [(0.0, 7.0, "SPEAKER_00")],
        ]

        class Pipeline:
            calls = []
            speaker_kwargs = []

            @classmethod
            def from_pretrained(cls, model: str, **kwargs):
                return cls()

            def __call__(self, audio, **kwargs):
                call_index = len(type(self).calls)
                type(self).calls.append(audio)
                type(self).speaker_kwargs.append(kwargs)
                return Diarization(chunks[call_index])

        fake_pyannote = types.ModuleType("pyannote")
        fake_audio = types.ModuleType("pyannote.audio")
        fake_audio.Pipeline = Pipeline
        fake_torchaudio = types.ModuleType("torchaudio")
        fake_torchaudio.info = lambda path: AudioInfo()
        fake_torchaudio.load = load

        modules = {
            "pyannote": fake_pyannote,
            "pyannote.audio": fake_audio,
            "torchaudio": fake_torchaudio,
        }
        with mock.patch.dict(sys.modules, modules):
            with mock.patch.object(pyannote_provider, "DIARIZATION_CHUNK_SECONDS", 10.0):
                with mock.patch.object(pyannote_provider, "DIARIZATION_CHUNK_OVERLAP_SECONDS", 2.0):
                    turns = diarize_audio(
                        Path("audio.wav"),
                        settings(Path("/tmp/atlas-test"), fallback=False, token="hf_test"),
                        expected_speakers=2,
                    )

        self.assertEqual(load_calls, [(0, 1200), (800, 1400), (1800, 700)])
        self.assertEqual(len(Pipeline.calls), 3)
        self.assertEqual(
            Pipeline.speaker_kwargs,
            [{"min_speakers": 1, "max_speakers": 6}] * 3,
        )
        self.assertTrue(all("num_speakers" not in kwargs for kwargs in Pipeline.speaker_kwargs))
        self.assertEqual(
            turns,
            [
                {"start": 0.0, "end": 9.0, "speaker": "SPEAKER_00"},
                {"start": 8.5, "end": 25.0, "speaker": "SPEAKER_01"},
            ],
        )

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

    def test_cuda_device_requires_torch_cuda(self) -> None:
        class Pipeline:
            @classmethod
            def from_pretrained(cls, model: str, **kwargs):
                return cls()

        fake_pyannote = types.ModuleType("pyannote")
        fake_audio = types.ModuleType("pyannote.audio")
        fake_audio.Pipeline = Pipeline
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

        modules = {
            "pyannote": fake_pyannote,
            "pyannote.audio": fake_audio,
            "torch": fake_torch,
        }
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "PyTorch CUDA is not available"):
                diarize_audio(
                    Path("audio.wav"),
                    settings(
                        Path("/tmp/atlas-test"),
                        fallback=False,
                        token="hf_test",
                        device="cuda",
                    ),
                )

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
