from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atlas_voice.cli import build_parser
from atlas_voice.tts_validation import validate_tts_sidecar
from atlas_voice.realtime import RealtimeAudio


class TTSValidationTests(unittest.TestCase):
    def test_validate_tts_sidecar_writes_audio_and_reports_latency(self) -> None:
        settings = SimpleNamespace(
            tts_provider="faster-qwen3-tts",
            tts_model="faster-qwen3-tts-0.6b",
            tts_base_url="http://127.0.0.1:8008/v1/audio/speech",
            tts_health_url="http://127.0.0.1:8008/health",
        )

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            audio_path = output_dir / "validation.wav"
            with (
                patch(
                    "atlas_voice.tts_validation.check_tts_sidecar_health",
                    return_value={
                        "status": "ok",
                        "detail": "ok",
                        "url": settings.tts_health_url,
                        "latency_ms": 3,
                    },
                ),
                patch(
                    "atlas_voice.tts_validation.synthesize_with_tts_sidecar",
                    return_value=RealtimeAudio(
                        path=audio_path,
                        payload=b"RIFFvalidation",
                        media_type="audio/wav",
                        latency_ms=11,
                    ),
                ) as synth_mock,
            ):
                result = validate_tts_sidecar(
                    settings,
                    text="Atlas sidecar validation",
                    output_dir=output_dir,
                )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.provider, "faster-qwen3-tts")
        self.assertEqual(result.model, "faster-qwen3-tts-0.6b")
        self.assertEqual(result.health_latency_ms, 3)
        self.assertEqual(result.synthesis_latency_ms, 11)
        self.assertEqual(result.audio_path, audio_path)
        self.assertEqual(result.audio_bytes, len(b"RIFFvalidation"))
        synth_mock.assert_called_once_with("Atlas sidecar validation", settings, output_dir)

    def test_validate_tts_sidecar_rejects_disabled_provider(self) -> None:
        settings = SimpleNamespace(tts_provider="none")

        with self.assertRaisesRegex(RuntimeError, "TTS sidecar provider"):
            validate_tts_sidecar(settings)

    def test_validate_tts_sidecar_rejects_non_loopback_urls_before_network_call(self) -> None:
        settings = SimpleNamespace(
            tts_provider="faster-qwen3-tts",
            tts_model="faster-qwen3-tts-0.6b",
            tts_base_url="https://api.example.test/v1/audio/speech",
            tts_health_url="http://127.0.0.1:8008/health",
        )

        with (
            patch("atlas_voice.tts_validation.check_tts_sidecar_health") as health_mock,
            patch("atlas_voice.tts_validation.synthesize_with_tts_sidecar") as synth_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "loopback"):
                validate_tts_sidecar(settings)

        health_mock.assert_not_called()
        synth_mock.assert_not_called()

    def test_validate_tts_sidecar_rejects_non_loopback_health_url(self) -> None:
        settings = SimpleNamespace(
            tts_provider="faster-qwen3-tts",
            tts_model="faster-qwen3-tts-0.6b",
            tts_base_url="http://127.0.0.1:8008/v1/audio/speech",
            tts_health_url="http://192.0.2.20:8008/health",
        )

        with patch("atlas_voice.tts_validation.check_tts_sidecar_health") as health_mock:
            with self.assertRaisesRegex(RuntimeError, "loopback"):
                validate_tts_sidecar(settings)

        health_mock.assert_not_called()

    def test_validate_tts_sidecar_requires_openai_audio_speech_endpoint(self) -> None:
        settings = SimpleNamespace(
            tts_provider="faster-qwen3-tts",
            tts_model="faster-qwen3-tts-0.6b",
            tts_base_url="http://127.0.0.1:8008/synthesize",
            tts_health_url="http://127.0.0.1:8008/health",
        )

        with patch("atlas_voice.tts_validation.check_tts_sidecar_health") as health_mock:
            with self.assertRaisesRegex(RuntimeError, "/v1/audio/speech"):
                validate_tts_sidecar(settings)

        health_mock.assert_not_called()

    def test_validate_tts_sidecar_fails_on_unhealthy_sidecar(self) -> None:
        settings = SimpleNamespace(
            tts_provider="faster-qwen3-tts",
            tts_model="faster-qwen3-tts-0.6b",
            tts_base_url="http://127.0.0.1:8008/v1/audio/speech",
            tts_health_url="http://127.0.0.1:8008/health",
        )

        with patch(
            "atlas_voice.tts_validation.check_tts_sidecar_health",
            return_value={
                "status": "error",
                "detail": "connection refused",
                "url": settings.tts_health_url,
                "latency_ms": 2,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "connection refused"):
                validate_tts_sidecar(settings)

    def test_validate_tts_sidecar_cli_exposes_output_text_and_json_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args([
            "validate-tts-sidecar",
            "--text",
            "Hello from Atlas",
            "--output-dir",
            "/tmp/tts-validation",
            "--json",
        ])

        self.assertEqual(args.text, "Hello from Atlas")
        self.assertEqual(args.output_dir, Path("/tmp/tts-validation"))
        self.assertTrue(args.json)


if __name__ == "__main__":
    unittest.main()
