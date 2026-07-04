from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atlas_voice.benchmark import run_voice_stack_benchmark
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
