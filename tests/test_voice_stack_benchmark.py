from __future__ import annotations

from pathlib import Path
from contextlib import redirect_stdout
import io
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atlas_voice.benchmark import (
    print_voice_profiles_benchmark_results,
    run_voice_profiles_benchmark,
    run_voice_stack_benchmark,
)
from atlas_voice.cli import build_parser
from atlas_voice.realtime import RealtimeAudio, RealtimeReply


class VoiceStackBenchmarkTests(unittest.TestCase):
    def test_voice_stack_benchmark_runs_llm_and_tts_concurrently(self) -> None:
        settings = _settings()

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            def fake_reply(_text, _settings):
                return RealtimeReply(
                    text="reply",
                    latency_ms=17,
                    tokens_in=4,
                    tokens_out=5,
                )

            def fake_tts(_text, _settings, _output_dir):
                return RealtimeAudio(
                    path=output_dir / "assistant.wav",
                    payload=b"RIFFvoice",
                    media_type="audio/wav",
                    latency_ms=11,
                )

            with (
                patch("atlas_voice.benchmark.generate_realtime_reply", side_effect=fake_reply),
                patch("atlas_voice.benchmark.synthesize_with_tts_sidecar", side_effect=fake_tts),
            ):
                results = run_voice_stack_benchmark(
                    settings,
                    rounds=2,
                    text="Can you hear me?",
                    tts_text="Atlas voice benchmark",
                    output_dir=output_dir,
                )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["llm_model"], "qwen-27b-instruct")
        self.assertEqual(results[0]["tts_provider"], "faster-qwen3-tts")
        self.assertEqual(results[0]["tts_model"], "faster-qwen3-tts-0.6b")
        self.assertEqual(results[0]["llm_latency_ms"], 17)
        self.assertEqual(results[0]["tts_latency_ms"], 11)
        self.assertEqual(results[0]["tts_audio_bytes"], len(b"RIFFvoice"))
        self.assertIsNone(results[0]["error"])

    def test_voice_stack_benchmark_reports_tts_error_without_hiding_llm_result(self) -> None:
        settings = _settings()

        with TemporaryDirectory() as tmp:
            with (
                patch(
                    "atlas_voice.benchmark.generate_realtime_reply",
                    return_value=RealtimeReply(text="reply", latency_ms=9),
                ),
                patch(
                    "atlas_voice.benchmark.synthesize_with_tts_sidecar",
                    side_effect=RuntimeError("sidecar refused"),
                ),
            ):
                results = run_voice_stack_benchmark(
                    settings,
                    rounds=1,
                    output_dir=Path(tmp),
                )

        self.assertEqual(results[0]["llm_latency_ms"], 9)
        self.assertIsNone(results[0]["tts_latency_ms"])
        self.assertIn("tts: RuntimeError: sidecar refused", results[0]["error"])

    def test_voice_profiles_benchmark_is_sequential_and_reports_router_mismatch(self) -> None:
        base_settings = _settings()
        events: list[tuple[str, str]] = []
        requested_models = {
            "light": "qwen3.5-2b",
            "torch": "qwen3.5-9b",
        }
        served_models = {
            "qwen3.5-2b": "Qwen/Qwen3.5-2B-instruct",
            "qwen3.5-9b": "/models/qwen27-q8-tuned.gguf",
        }

        def resolve_profile(_settings, _config, profile_id):
            resolved = _settings_copy(base_settings)
            resolved.llm_model = requested_models[profile_id]
            return resolved

        def fake_reply(text, current_settings, *, on_text_delta):
            events.append(("llm", current_settings.llm_model))
            on_text_delta("streamed")
            return RealtimeReply(
                text=f"reply from {current_settings.llm_model}: {text}",
                latency_ms=80,
                tokens_in=12,
                tokens_out=6,
                served_model=served_models[current_settings.llm_model],
                ttft_ms=25,
                tokens_per_second=18.5,
            )

        def fake_tts(text, _settings, output_dir):
            events.append(("tts", text))
            return RealtimeAudio(
                path=output_dir / "assistant.wav",
                payload=b"RIFFvoice",
                latency_ms=14,
            )

        with TemporaryDirectory() as tmp:
            with (
                patch(
                    "atlas_voice.benchmark.settings_for_voice_profile",
                    side_effect=resolve_profile,
                ),
                patch("atlas_voice.benchmark.generate_realtime_reply", side_effect=fake_reply),
                patch(
                    "atlas_voice.benchmark.synthesize_with_tts_sidecar",
                    side_effect=fake_tts,
                ),
                patch("atlas_voice.benchmark._max_rss_mb", return_value=123.0),
            ):
                results = run_voice_profiles_benchmark(
                    base_settings,
                    object(),
                    profiles="torch,light,light",
                    rounds=1,
                    text="Benchmark prompt",
                    output_dir=Path(tmp),
                )

        self.assertEqual([result["requested_profile"] for result in results], ["light", "torch"])
        self.assertEqual(
            [event[0] for event in events],
            ["llm", "tts", "llm", "tts"],
        )
        self.assertIn("reply from qwen3.5-2b", events[1][1])
        self.assertIn("reply from qwen3.5-9b", events[3][1])
        self.assertEqual(results[0]["served_model"], "Qwen3.5-2B-instruct")
        self.assertEqual(results[0]["model_relation"], "alias")
        self.assertTrue(results[0]["routing_verified"])
        self.assertFalse(results[0]["artifact_verified"])
        self.assertEqual(results[1]["served_model"], "qwen27-q8-tuned.gguf")
        self.assertEqual(results[1]["model_relation"], "mismatch")
        self.assertFalse(results[1]["routing_verified"])
        self.assertFalse(results[1]["artifact_verified"])
        self.assertEqual(results[1]["llm_ttft_ms"], 25)
        self.assertEqual(results[1]["llm_latency_ms"], 80)
        self.assertEqual(results[1]["llm_tokens_per_second"], 18.5)
        self.assertEqual(results[1]["tts_latency_ms"], 14)
        self.assertGreaterEqual(results[1]["end_to_end_ms"], 0)

        output = io.StringIO()
        with redirect_stdout(output):
            print_voice_profiles_benchmark_results(results)
        self.assertIn("relation=mismatch", output.getvalue())
        self.assertIn("routing_verified=no", output.getvalue())

    def test_voice_profiles_benchmark_cli_exposes_profile_and_round_options(self) -> None:
        args = build_parser().parse_args(
            ["benchmark-voice-profiles", "--profiles", "fire,light", "--rounds", "2", "--json"]
        )

        self.assertEqual(args.profiles, "fire,light")
        self.assertEqual(args.rounds, 2)
        self.assertTrue(args.json)

    def test_voice_stack_benchmark_cli_exposes_rounds_text_output_and_json_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args([
            "benchmark-voice-stack",
            "--rounds",
            "5",
            "--text",
            "Question",
            "--tts-text",
            "Spoken answer",
            "--output-dir",
            "/tmp/voice-stack",
            "--json",
        ])

        self.assertEqual(args.rounds, 5)
        self.assertEqual(args.text, "Question")
        self.assertEqual(args.tts_text, "Spoken answer")
        self.assertEqual(args.output_dir, Path("/tmp/voice-stack"))
        self.assertTrue(args.json)


def _settings_copy(settings: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(**vars(settings))



def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        stub_mode=False,
        llm_base_url="http://127.0.0.1:8088/v1/chat/completions",
        llm_model="qwen-27b-instruct",
        llm_temperature=0.2,
        llm_max_tokens=256,
        tts_provider="faster-qwen3-tts",
        tts_model="faster-qwen3-tts-0.6b",
        tts_base_url="http://127.0.0.1:8008/v1/audio/speech",
        tts_voice="default",
        tts_response_format="wav",
        tts_timeout=60.0,
        artifacts_dir=Path("/tmp/atlas-voice-artifacts"),
    )


if __name__ == "__main__":
    unittest.main()
