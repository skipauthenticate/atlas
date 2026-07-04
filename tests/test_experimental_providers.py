from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import types
import unittest
from unittest import mock

from atlas_voice.benchmark import run_asr_benchmark, word_error_rate
from atlas_voice.config import Settings
from atlas_voice.providers.asr import (
    realtime_asr_provider_chain,
    select_realtime_asr_provider,
    transcribe_audio,
)
from atlas_voice.providers.diarization import diarize_audio
from atlas_voice.providers.faster_whisper_provider import transcribe_faster_whisper
from atlas_voice.providers.hyprwhspr_provider import (
    hyprwhspr_available,
    hyprwhspr_reliable,
    transcribe_hyprwhspr,
)
from atlas_voice.providers.nemo_provider import transcribe_parakeet
from atlas_voice.providers.transcript_utils import (
    diarization_from_transcript,
    transcript_from_vibevoice_result,
)


def settings(
    root: Path, *, asr_provider: str = "whisperx", diarization_provider: str = "pyannote"
) -> Settings:
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
        hf_token="hf_test",
        llm_base_url="http://127.0.0.1:8080/v1/chat/completions",
        llm_model="qwen-local",
        llm_temperature=0.2,
        llm_max_tokens=100,
        stub_mode=False,
        asr_provider=asr_provider,
        diarization_provider=diarization_provider,
    )


class ExperimentalProviderTests(unittest.TestCase):

    def test_realtime_asr_prefers_available_hyprwhspr_cli(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = root / "hyprwhspr"
            cli.write_text("#!/bin/sh\n")
            cli.chmod(0o755)
            provider_settings = replace(
                settings(root),
                hyprwhspr_cli=str(cli),
                hyprwhspr_endpoint=None,
                realtime_asr_prefer_hyprwhspr=True,
            )

            self.assertTrue(hyprwhspr_available(provider_settings))
            self.assertEqual(select_realtime_asr_provider(provider_settings), "hyprwhspr")

    def test_realtime_asr_falls_back_when_hyprwhspr_is_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider_settings = replace(
                settings(root),
                hyprwhspr_cli=str(root / "missing-hyprwhspr"),
                hyprwhspr_endpoint=None,
                realtime_asr_prefer_hyprwhspr=True,
            )

            self.assertFalse(hyprwhspr_available(provider_settings))
            self.assertEqual(select_realtime_asr_provider(provider_settings), "faster-whisper")


    def test_realtime_asr_uses_hyprwhspr_only_after_cli_reliability_probe(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = root / "hyprwhspr"
            cli.write_text("#!/bin/sh\nexit 1\n")
            cli.chmod(0o755)
            provider_settings = replace(
                settings(root),
                hyprwhspr_cli=str(cli),
                hyprwhspr_endpoint=None,
                realtime_asr_prefer_hyprwhspr=True,
            )

            self.assertTrue(hyprwhspr_available(provider_settings))
            self.assertFalse(hyprwhspr_reliable(provider_settings))
            self.assertEqual(select_realtime_asr_provider(provider_settings), "faster-whisper")

            cli.write_text("#!/bin/sh\nexit 0\n")
            cli.chmod(0o755)

            self.assertTrue(hyprwhspr_reliable(provider_settings))
            self.assertEqual(select_realtime_asr_provider(provider_settings), "hyprwhspr")

    def test_realtime_asr_uses_hyprwhspr_socket_after_health_probe(self) -> None:
        provider_settings = replace(
            settings(Path("/tmp/atlas-test")),
            hyprwhspr_endpoint="http://127.0.0.1:9000/transcribe",
            hyprwhspr_health_url="http://127.0.0.1:9000/health",
            realtime_asr_prefer_hyprwhspr=True,
        )
        healthy_response = types.SimpleNamespace(status_code=200)
        unhealthy_response = types.SimpleNamespace(status_code=503)

        with mock.patch("httpx.get", return_value=healthy_response):
            self.assertTrue(hyprwhspr_reliable(provider_settings))
            self.assertEqual(select_realtime_asr_provider(provider_settings), "hyprwhspr")

        with mock.patch("httpx.get", return_value=unhealthy_response):
            self.assertFalse(hyprwhspr_reliable(provider_settings))
            self.assertEqual(select_realtime_asr_provider(provider_settings), "faster-whisper")

    def test_hyprwhspr_cli_transcription_parses_json_stdout(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = root / "hyprwhspr"
            cli.write_text("#!/bin/sh\n")
            cli.chmod(0o755)
            provider_settings = replace(
                settings(root),
                hyprwhspr_cli=str(cli),
                hyprwhspr_endpoint=None,
                hyprwhspr_timeout=2.0,
            )
            completed = types.SimpleNamespace(
                stdout=json.dumps({"text": "hello local", "segments": [{"text": "hello local"}]}),
                stderr="",
            )

            with mock.patch("subprocess.run", return_value=completed) as run_mock:
                transcript = transcribe_hyprwhspr(Path("audio.wav"), provider_settings)

        run_mock.assert_called_once()
        self.assertEqual(transcript["provider"], "hyprwhspr")
        self.assertEqual(transcript["text"], "hello local")
        self.assertEqual(transcript["segments"][0]["text"], "hello local")


    def test_realtime_asr_chain_uses_faster_whisper_before_configured_provider(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider_settings = replace(
                settings(root),
                hyprwhspr_cli=str(root / "missing-hyprwhspr"),
                hyprwhspr_endpoint=None,
                realtime_asr_prefer_hyprwhspr=True,
                realtime_asr_fallback_provider="faster-whisper",
            )

            self.assertEqual(
                realtime_asr_provider_chain(provider_settings),
                ["faster-whisper", "whisperx"],
            )
            self.assertEqual(select_realtime_asr_provider(provider_settings), "faster-whisper")

    def test_realtime_asr_chain_deduplicates_configured_fallback_provider(self) -> None:
        provider_settings = replace(
            settings(Path("/tmp/atlas-test"), asr_provider="faster-whisper"),
            hyprwhspr_cli="/tmp/missing-hyprwhspr",
            hyprwhspr_endpoint=None,
            realtime_asr_fallback_provider="faster-whisper",
        )

        self.assertEqual(realtime_asr_provider_chain(provider_settings), ["faster-whisper"])


    def test_realtime_asr_chain_falls_back_when_configured_hyprwhspr_is_unreliable(self) -> None:
        provider_settings = replace(
            settings(Path("/tmp/atlas-test"), asr_provider="hyprwhspr"),
            hyprwhspr_cli="/tmp/missing-hyprwhspr",
            hyprwhspr_endpoint=None,
            realtime_asr_prefer_hyprwhspr=True,
            realtime_asr_fallback_provider="faster-whisper",
        )

        self.assertEqual(realtime_asr_provider_chain(provider_settings), ["faster-whisper", "hyprwhspr"])
        self.assertEqual(select_realtime_asr_provider(provider_settings), "faster-whisper")


    def test_realtime_transcription_falls_back_to_configured_provider(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider_settings = replace(
                settings(root),
                hyprwhspr_cli=str(root / "missing-hyprwhspr"),
                hyprwhspr_endpoint=None,
                realtime_asr_fallback_provider="faster-whisper",
            )
            with (
                mock.patch(
                    "atlas_voice.providers.asr.transcribe_faster_whisper",
                    side_effect=RuntimeError("faster-whisper busy"),
                ) as faster_mock,
                mock.patch(
                    "atlas_voice.providers.asr.transcribe_whisperx",
                    return_value={"text": "whisperx fallback", "segments": []},
                ) as whisperx_mock,
            ):
                transcript = transcribe_audio(Path("audio.wav"), provider_settings, realtime=True)

        self.assertEqual(transcript["text"], "whisperx fallback")
        faster_mock.assert_called_once()
        whisperx_mock.assert_called_once()

    def test_faster_whisper_provider_normalizes_segments_and_words(self) -> None:
        class Word:
            start = 0.0
            end = 0.4
            word = "hello"
            probability = 0.9

        class Segment:
            start = 0.0
            end = 1.0
            text = "hello local"
            words = [Word()]

        class Info:
            language = "en"
            language_probability = 0.99

        class WhisperModel:
            requested = None
            kwargs = None

            def __init__(self, model_name, **kwargs):
                self.__class__.requested = model_name
                self.__class__.kwargs = kwargs

            def transcribe(self, path, **kwargs):
                self.path = path
                self.transcribe_kwargs = kwargs
                return [Segment()], Info()

        fake_module = types.ModuleType("faster_whisper")
        fake_module.WhisperModel = WhisperModel
        provider_settings = replace(
            settings(Path("/tmp/atlas-test")),
            faster_whisper_model="distil-large-v3",
            whisperx_device="cpu",
            whisperx_compute_type="int8",
        )

        with mock.patch.dict(sys.modules, {"faster_whisper": fake_module}):
            transcript = transcribe_faster_whisper(Path("audio.wav"), provider_settings)

        self.assertEqual(WhisperModel.requested, "distil-large-v3")
        self.assertEqual(WhisperModel.kwargs["device"], "cpu")
        self.assertEqual(WhisperModel.kwargs["compute_type"], "int8")
        self.assertEqual(transcript["provider"], "faster-whisper")
        self.assertEqual(transcript["language"], "en")
        self.assertEqual(transcript["text"], "hello local")
        self.assertEqual(transcript["segments"][0]["words"][0]["word"], "hello")

    def test_word_error_rate_counts_word_edits(self) -> None:
        self.assertEqual(word_error_rate("alpha beta", "alpha beta"), 0.0)
        self.assertAlmostEqual(word_error_rate("one two three", "one four three"), 1 / 3)

    def test_benchmark_uses_transcript_diarization_for_vibevoice(self) -> None:
        captured = {}

        def fake_transcribe(_audio_path, provider_settings):
            captured["asr_provider"] = provider_settings.asr_provider
            return {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "speaker": "SPEAKER_00",
                        "text": "hello world",
                    }
                ],
                "diarization": [
                    {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                ],
            }

        def fake_diarize(_audio_path, provider_settings, *, transcript):
            captured["diarization_provider"] = provider_settings.diarization_provider
            return transcript["diarization"]

        with (
            mock.patch("atlas_voice.benchmark.audio_duration_seconds", return_value=1.0),
            mock.patch("atlas_voice.benchmark.transcribe_audio", side_effect=fake_transcribe),
            mock.patch("atlas_voice.benchmark.diarize_audio", side_effect=fake_diarize),
        ):
            results = run_asr_benchmark(
                Path("audio.wav"),
                settings(Path("/tmp/atlas-test"), diarization_provider="pyannote"),
                providers=["vibevoice"],
                reference_text="hello world",
                include_diarization=True,
            )

        self.assertEqual(captured["asr_provider"], "vibevoice")
        self.assertEqual(captured["diarization_provider"], "transcript")
        self.assertEqual(results[0]["diarization_turn_count"], 1)
        self.assertEqual(results[0]["wer"], 0.0)

    def test_diarization_from_transcript_uses_embedded_turns(self) -> None:
        transcript = {
            "diarization": [
                {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            ]
        }

        turns = diarization_from_transcript(transcript)

        self.assertEqual(turns, transcript["diarization"])

    def test_transcript_diarization_provider_requires_speaker_turns(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not contain speaker turns"):
            diarize_audio(
                Path("audio.wav"),
                settings(Path("/tmp/atlas-test"), diarization_provider="transcript"),
                transcript={"segments": []},
            )

    def test_vibevoice_result_converts_segments_and_speaker_turns(self) -> None:
        result = {
            "raw_text": "hello",
            "segments": [
                {
                    "start_time": "00:00:01.000",
                    "end_time": "00:00:02.500",
                    "speaker_id": "1",
                    "text": "hello there",
                }
            ],
        }

        transcript = transcript_from_vibevoice_result(
            result, Path("audio.wav"), provider="vibevoice", model="test"
        )

        self.assertEqual(transcript["segments"][0]["speaker"], "SPEAKER_01")
        self.assertEqual(transcript["segments"][0]["start"], 1.0)
        self.assertEqual(transcript["segments"][0]["end"], 2.5)
        self.assertEqual(transcript["diarization"][0]["speaker"], "SPEAKER_01")

    def test_parakeet_provider_uses_nemo_output(self) -> None:
        class Output:
            text = "hello world"
            timestamp = {
                "segment": [{"start": 0.0, "end": 1.0, "segment": "hello world"}],
                "word": [
                    {"start": 0.0, "end": 0.4, "word": "hello"},
                    {"start": 0.5, "end": 1.0, "word": "world"},
                ],
            }

        class ASRModel:
            instance = None
            requested_model = None

            def __init__(self):
                self.cfg = types.SimpleNamespace(decoding={"greedy": {}})
                self.changed_decoding_cfg = None

            @classmethod
            def from_pretrained(cls, model_name):
                cls.requested_model = model_name
                cls.instance = cls()
                return cls.instance

            def change_decoding_strategy(self, decoding_cfg, verbose=True):
                self.changed_decoding_cfg = decoding_cfg

            def transcribe(self, paths, **kwargs):
                self.paths = paths
                self.kwargs = kwargs
                return [Output()]

        fake_nemo = types.ModuleType("nemo")
        fake_collections = types.ModuleType("nemo.collections")
        fake_asr = types.ModuleType("nemo.collections.asr")
        fake_models = types.ModuleType("nemo.collections.asr.models")
        fake_models.ASRModel = ASRModel
        modules = {
            "nemo": fake_nemo,
            "nemo.collections": fake_collections,
            "nemo.collections.asr": fake_asr,
            "nemo.collections.asr.models": fake_models,
        }
        with mock.patch.dict(sys.modules, modules):
            transcript = transcribe_parakeet(
                Path("audio.wav"), settings(Path("/tmp/atlas-test"), asr_provider="parakeet")
            )

        self.assertEqual(ASRModel.requested_model, "nvidia/parakeet-tdt-0.6b-v3")
        self.assertIsNotNone(ASRModel.instance)
        self.assertFalse(
            ASRModel.instance.cfg.decoding["greedy"]["use_cuda_graph_decoder"]
        )
        self.assertIs(ASRModel.instance.changed_decoding_cfg, ASRModel.instance.cfg.decoding)
        self.assertEqual(transcript["provider"], "parakeet")
        self.assertEqual(transcript["segments"][0]["words"][0]["word"], "hello")


if __name__ == "__main__":
    unittest.main()
