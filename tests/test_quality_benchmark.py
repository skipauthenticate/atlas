from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from atlas_voice.benchmark import (
    character_error_rate,
    print_quality_benchmark_results,
    run_quality_benchmark,
)
from atlas_voice.cli import build_parser, main
from atlas_voice.config import Settings
from atlas_voice.quality import settings_for_quality


def _settings(root: Path) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8787,
        data_dir=root / "data",
        models_dir=root / "models",
        hf_cache_dir=root / "cache" / "huggingface",
        whisperx_model="default-whisperx",
        whisperx_device="cpu",
        whisperx_compute_type="int8",
        pyannote_model="pyannote/speaker-diarization-community-1",
        hf_token="hf_test",
        llm_base_url="http://127.0.0.1:8080/v1/chat/completions",
        llm_model="qwen-local",
        llm_temperature=0.2,
        llm_max_tokens=100,
        stub_mode=False,
        diarization_provider="pyannote",
        quality_light_asr_provider="faster-whisper",
        quality_light_asr_model="light-model",
        quality_torch_asr_provider="faster-whisper",
        quality_torch_asr_model="torch-model",
        quality_fire_asr_provider="whisperx",
        quality_fire_asr_model="fire-model",
    )


class QualityBenchmarkTests(unittest.TestCase):
    def test_runs_all_product_tiers_with_literal_quality_metrics(self) -> None:
        provider_settings = _settings(Path("/tmp/atlas-quality-test"))
        normalizations: list[tuple[str, str]] = []
        transcriptions: list[tuple[str, str | None, int]] = []
        speaker_hints: list[int | None] = []

        def fake_normalize(_source: Path, destination: Path, *, cleanup_mode: str) -> None:
            normalizations.append((destination.name, cleanup_mode))

        def fake_transcribe(_audio: Path, trial_settings: Settings):
            transcriptions.append(
                (
                    trial_settings.asr_provider,
                    trial_settings.asr_model,
                    trial_settings.asr_beam_size,
                )
            )
            return {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "hello world",
                    }
                ]
            }

        def fake_diarize(
            _audio: Path,
            _trial_settings: Settings,
            *,
            transcript,
            expected_speakers: int | None = None,
        ):
            self.assertTrue(transcript["segments"])
            speaker_hints.append(expected_speakers)
            return [
                {"start": 0.0, "end": 0.7, "speaker": "SPEAKER_00"},
                {"start": 0.7, "end": 1.4, "speaker": "SPEAKER_01"},
                {"start": 1.4, "end": 2.0, "speaker": "SPEAKER_02"},
            ]

        with (
            mock.patch("atlas_voice.benchmark.audio_duration_seconds", return_value=10.0),
            mock.patch("atlas_voice.benchmark.normalize_audio", side_effect=fake_normalize),
            mock.patch("atlas_voice.benchmark.transcribe_audio", side_effect=fake_transcribe),
            mock.patch("atlas_voice.benchmark.diarize_audio", side_effect=fake_diarize),
            mock.patch("atlas_voice.benchmark._max_rss_mb", return_value=512.5),
            mock.patch(
                "atlas_voice.benchmark.settings_for_quality",
                wraps=settings_for_quality,
            ) as quality_settings,
        ):
            results = run_quality_benchmark(
                Path("meeting.wav"),
                provider_settings,
                reference_text="hello world",
                expected_speakers=2,
            )

        self.assertEqual([result["tier"] for result in results], ["light", "torch", "fire"])
        self.assertEqual([result["tier_name"] for result in results], ["Light", "Torch", "Fire"])
        self.assertEqual([result["rank"] for result in results], [1, 2, 3])
        self.assertEqual(
            [result["provider"] for result in results],
            ["faster-whisper", "faster-whisper", "whisperx"],
        )
        self.assertEqual(
            [result["model"] for result in results],
            ["light-model", "torch-model", "fire-model"],
        )
        self.assertEqual(
            normalizations,
            [
                ("light.wav", "none"),
                ("torch.wav", "adaptive"),
                ("fire.wav", "enhanced"),
            ],
        )
        self.assertEqual(
            transcriptions,
            [
                ("faster-whisper", "light-model", 1),
                ("faster-whisper", "torch-model", 3),
                ("whisperx", "fire-model", 5),
            ],
        )
        self.assertEqual(speaker_hints, [2, 2, 2])
        self.assertEqual(
            [call.args[1] for call in quality_settings.call_args_list],
            ["light", "torch", "fire"],
        )
        for result in results:
            self.assertEqual(result["wer"], 0.0)
            self.assertEqual(result["cer"], 0.0)
            self.assertEqual(result["detected_speaker_count"], 3)
            self.assertEqual(result["speaker_count_error"], 1)
            self.assertFalse(result["repetition_hallucination_warning"])
            self.assertEqual(result["max_rss_mb"], 512.5)
            self.assertIsNone(result["error"])
            self.assertNotIn("accuracy_score", result)
            self.assertNotIn("winner", result)
            self.assertNotIn("recommended_tier", result)

    def test_tier_subset_runs_in_rank_order_and_rejects_unknown_tiers(self) -> None:
        provider_settings = _settings(Path("/tmp/atlas-quality-test"))

        with (
            mock.patch("atlas_voice.benchmark.audio_duration_seconds", return_value=5.0),
            mock.patch("atlas_voice.benchmark.normalize_audio"),
            mock.patch(
                "atlas_voice.benchmark.transcribe_audio",
                return_value={"segments": [{"text": "short transcript"}]},
            ),
            mock.patch("atlas_voice.benchmark.diarize_audio", return_value=[]),
        ):
            results = run_quality_benchmark(
                Path("meeting.wav"),
                provider_settings,
                tiers="fire,light,light",
            )

        self.assertEqual([result["tier"] for result in results], ["light", "fire"])
        with self.assertRaisesRegex(ValueError, "Unknown quality tier"):
            run_quality_benchmark(
                Path("meeting.wav"),
                provider_settings,
                tiers="light,spark",
            )

    def test_normalization_failure_reports_error_without_invented_metrics(self) -> None:
        provider_settings = _settings(Path("/tmp/atlas-quality-test"))

        with (
            mock.patch("atlas_voice.benchmark.audio_duration_seconds", return_value=5.0),
            mock.patch(
                "atlas_voice.benchmark.normalize_audio",
                side_effect=RuntimeError("ffmpeg failed"),
            ),
            mock.patch("atlas_voice.benchmark.transcribe_audio") as transcribe,
        ):
            result = run_quality_benchmark(
                Path("meeting.wav"),
                provider_settings,
                tiers="torch",
                reference_text="known reference",
                expected_speakers=2,
            )[0]

        transcribe.assert_not_called()
        self.assertIn("normalization: RuntimeError: ffmpeg failed", result["error"])
        self.assertIsNone(result["wer"])
        self.assertIsNone(result["cer"])
        self.assertIsNone(result["detected_speaker_count"])
        self.assertIsNone(result["speaker_count_error"])
        self.assertIsNone(result["repetition_hallucination_warning"])

    def test_repetition_warning_flags_a_consecutive_phrase_loop(self) -> None:
        provider_settings = _settings(Path("/tmp/atlas-quality-test"))

        with (
            mock.patch("atlas_voice.benchmark.audio_duration_seconds", return_value=5.0),
            mock.patch("atlas_voice.benchmark.normalize_audio"),
            mock.patch(
                "atlas_voice.benchmark.transcribe_audio",
                return_value={
                    "segments": [
                        {
                            "text": "please review this please review this",
                        }
                    ]
                },
            ),
            mock.patch("atlas_voice.benchmark.diarize_audio", return_value=[]),
        ):
            result = run_quality_benchmark(
                Path("meeting.wav"),
                provider_settings,
                tiers="light",
            )[0]

        self.assertTrue(result["repetition_hallucination_warning"])
        self.assertIn("phrase repeated", result["repetition_hallucination_reason"])

    def test_character_error_rate_uses_normalized_characters(self) -> None:
        self.assertAlmostEqual(character_error_rate("cat", "cut"), 1 / 3)
        self.assertEqual(character_error_rate("", ""), 0.0)
        self.assertEqual(character_error_rate("", "text"), 1.0)

    def test_cli_parses_quality_benchmark_options(self) -> None:
        args = build_parser().parse_args(
            [
                "benchmark-quality",
                "meeting.wav",
                "--reference",
                "meeting.reference.txt",
                "--expected-speakers",
                "3",
                "--tiers",
                "fire,light",
                "--json",
            ]
        )

        self.assertEqual(args.audio, Path("meeting.wav"))
        self.assertEqual(args.reference, Path("meeting.reference.txt"))
        self.assertEqual(args.expected_speakers, 3)
        self.assertEqual(args.tiers, "fire,light")
        self.assertTrue(args.json)

    def test_cli_reads_reference_and_forwards_quality_options(self) -> None:
        provider_settings = _settings(Path("/tmp/atlas-quality-test"))
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio = root / "meeting.wav"
            reference = root / "meeting.reference.txt"
            audio.write_bytes(b"fixture")
            reference.write_text("ground truth transcript")

            with (
                mock.patch(
                    "atlas_voice.cli.Settings.from_env",
                    return_value=provider_settings,
                ),
                mock.patch(
                    "atlas_voice.cli.admit_pipeline_step",
                    return_value=mock.Mock(allowed=True),
                ),
                mock.patch(
                    "atlas_voice.cli.run_quality_benchmark",
                    return_value=[{"error": None}],
                ) as runner,
                mock.patch("atlas_voice.cli.print_quality_benchmark_results") as printer,
            ):
                exit_code = main(
                    [
                        "benchmark-quality",
                        str(audio),
                        "--reference",
                        str(reference),
                        "--expected-speakers",
                        "2",
                        "--tiers",
                        "torch",
                        "--json",
                    ]
                )

        self.assertEqual(exit_code, 0)
        runner.assert_called_once_with(
            audio,
            provider_settings,
            reference_text="ground truth transcript",
            expected_speakers=2,
            tiers=["torch"],
        )
        printer.assert_called_once_with([{"error": None}], json_output=True)

    def test_cli_refuses_a_tier_when_device_memory_is_unsafe(self) -> None:
        provider_settings = _settings(Path("/tmp/atlas-quality-test"))
        error_output = StringIO()
        with (
            mock.patch(
                "atlas_voice.cli.Settings.from_env",
                return_value=provider_settings,
            ),
            mock.patch(
                "atlas_voice.cli.admit_pipeline_step",
                return_value=mock.Mock(
                    allowed=False,
                    message="Fire processing is waiting for device memory.",
                ),
            ),
            mock.patch("atlas_voice.cli.run_quality_benchmark") as runner,
            redirect_stderr(error_output),
        ):
            exit_code = main(
                ["benchmark-quality", "meeting.wav", "--tiers", "fire"]
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("waiting for device memory", error_output.getvalue())
        self.assertIn("--tiers", error_output.getvalue())
        runner.assert_not_called()

    def test_json_report_keeps_literal_results(self) -> None:
        literal_result = {
            "tier": "light",
            "rank": 1,
            "wer": 0.25,
            "error": None,
        }
        output = StringIO()
        with redirect_stdout(output):
            print_quality_benchmark_results([literal_result], json_output=True)

        self.assertEqual(json.loads(output.getvalue()), {"results": [literal_result]})


if __name__ == "__main__":
    unittest.main()
