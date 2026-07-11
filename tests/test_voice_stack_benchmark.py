from __future__ import annotations

from pathlib import Path
from contextlib import redirect_stdout
import hashlib
import io
import json
import wave
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atlas_voice.benchmark import (
    VoiceTTSMeasurement,
    _measure_tts_sidecar_for_benchmark,
    _observe_voice_artifact_provenance,
    _prepare_voice_artifact_provenance,
    _wav_audio_metadata,
    print_voice_profiles_benchmark_results,
    run_voice_profiles_benchmark,
    run_voice_stack_benchmark,
    validate_voice_profiles_benchmark_json_output,
    write_voice_profiles_benchmark_json,
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
            return _fake_tts_measurement(
                output_dir / "assistant.wav",
                requested_model=_settings.tts_model,
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
                    "atlas_voice.benchmark._measure_voice_profile_tts",
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
        self.assertEqual(results[1]["llm_full_latency_ms"], 80)
        self.assertEqual(results[1]["tts_served_model"], "faster-qwen3-tts-0.6b")
        self.assertTrue(results[1]["tts_model_verified"])
        self.assertEqual(results[1]["tts_first_audio_byte_ms"], 2)
        self.assertTrue(results[1]["tts_first_audio_byte_observed"])
        self.assertIsNone(results[1]["tts_first_playable_audio_ms"])
        self.assertFalse(results[1]["tts_first_playable_audio_observed"])
        self.assertEqual(results[1]["tts_audio_duration_seconds"], 1.0)
        self.assertEqual(results[1]["tts_real_time_factor"], 0.014)
        self.assertEqual(results[1]["artifact_provenance"]["status"], "inventory_not_provided")
        self.assertGreaterEqual(results[1]["end_to_end_ms"], 0)

        output = io.StringIO()
        with redirect_stdout(output):
            print_voice_profiles_benchmark_results(results)
        self.assertIn("relation=mismatch", output.getvalue())

        json_output = io.StringIO()
        with redirect_stdout(json_output):
            print_voice_profiles_benchmark_results(results, json_output=True)
        payload = json.loads(json_output.getvalue())
        self.assertEqual(payload["schema_version"], "2.0.0")
        self.assertFalse(payload["first_audio_byte_is_first_playable"])
        self.assertEqual(len(payload["results"]), 2)
        self.assertIn("routing_verified=no", output.getvalue())

    def test_tts_sidecar_observer_reports_buffered_byte_timing_and_wav_rtf(self) -> None:
        import httpx

        payload = _wav_payload(seconds=1.0, sample_rate=8000)

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.assertEqual(body["model"], "faster-qwen3-tts-0.6b")
            self.assertEqual(body["response_format"], "wav")
            return httpx.Response(
                200,
                content=payload,
                headers={
                    "content-type": "audio/wav",
                    "x-atlas-model": "faster-qwen3-tts-0.6b",
                    "x-audio-sample-rate": "8000",
                    "x-generation-latency-ms": "7",
                },
            )

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client

        def client_factory(*args, **kwargs):
            return real_client(*args, transport=transport, **kwargs)

        with TemporaryDirectory() as tmp:
            with patch("httpx.Client", side_effect=client_factory):
                measurement = _measure_tts_sidecar_for_benchmark(
                    "A short benchmark reply.",
                    _settings(),
                    Path(tmp),
                )

            self.assertTrue(measurement.audio.path.is_file())

        self.assertEqual(measurement.served_model, "faster-qwen3-tts-0.6b")
        self.assertEqual(measurement.served_model_source, "x-atlas-model")
        self.assertIsNotNone(measurement.response_headers_ms)
        self.assertIsNotNone(measurement.first_audio_byte_ms)
        self.assertGreaterEqual(measurement.full_response_ms, measurement.first_audio_byte_ms or 0)
        self.assertEqual(measurement.server_generation_ms, 7)
        self.assertEqual(measurement.audio_duration_seconds, 1.0)
        self.assertEqual(measurement.sample_rate_hz, 8000)
        self.assertEqual(measurement.channels, 1)
        self.assertEqual(measurement.sample_width_bytes, 2)
        self.assertTrue(measurement.audio_integrity_verified)
        self.assertEqual(measurement.delivery_mode, "buffered-http-response")

    def test_wav_integrity_rejects_truncated_header_and_frame_payload(self) -> None:
        payload = _wav_payload(seconds=1.0, sample_rate=8000)

        self.assertTrue(_wav_audio_metadata(payload)[-1])
        self.assertEqual(
            _wav_audio_metadata(payload[:20]),
            (None, None, None, None, False),
        )
        self.assertEqual(
            _wav_audio_metadata(payload[:-8000]),
            (None, None, None, None, False),
        )

    def test_wav_integrity_rejects_bytes_outside_declared_riff_container(self) -> None:
        payload = _wav_payload(seconds=1.0, sample_rate=8000)

        self.assertEqual(
            _wav_audio_metadata(payload + b"trailing"),
            (None, None, None, None, False),
        )

    def test_artifact_provenance_hashes_once_and_binds_loaded_router_path(self) -> None:
        import httpx
        import atlas_voice.benchmark as benchmark_module

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "candidate.gguf"
            artifact.write_bytes(b"candidate")
            inventory = root / "inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "inventory_version": "test-1",
                        "models": [
                            {
                                "profile": "light",
                                "model_id": "qwen-light",
                                "path": artifact.name,
                                "size_bytes": artifact.stat().st_size,
                                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                "source_url": "https://example.test/candidate",
                                "source_revision": "abc123",
                                "license": "Apache-2.0",
                            }
                        ],
                    }
                )
            )

            with patch(
                "atlas_voice.benchmark._sha256_path",
                wraps=benchmark_module._sha256_path,
            ) as hash_path:
                prepared = _prepare_voice_artifact_provenance(inventory, ["light"])
            self.assertEqual(hash_path.call_count, 1)

            def handler(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "qwen-light",
                                "path": None,
                                "status": {
                                    "value": "loaded",
                                    "args": ["llama-server", "--model", str(artifact.resolve())],
                                },
                            }
                        ]
                    },
                )

            transport = httpx.MockTransport(handler)
            real_client = httpx.Client

            def client_factory(*args, **kwargs):
                return real_client(*args, transport=transport, **kwargs)

            with patch("httpx.Client", side_effect=client_factory):
                observation = _observe_voice_artifact_provenance(
                    prepared["light"],
                    requested_model="qwen-light",
                    served_model="qwen-light",
                    models_endpoint="http://127.0.0.1:18080/models",
                )

        self.assertTrue(observation["verified"])
        self.assertTrue(observation["file_verified"])
        self.assertTrue(observation["route_path_verified"])
        self.assertTrue(observation["runtime_loaded"])
        self.assertEqual(observation["status"], "verified")
        self.assertEqual(observation["router_path_source"], "status.args")
        self.assertEqual(observation["source_revision"], "abc123")

    def test_voice_profile_json_output_is_private_atomic_and_matches_stdout(self) -> None:
        results = [{"requested_profile": "light", "round": 1}]

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            print_voice_profiles_benchmark_results(results, json_output=True)
        stdout_payload = json.loads(stdout.getvalue())

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "voice-path.json"
            written = write_voice_profiles_benchmark_json(target, results)

            self.assertEqual(written, target)
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(target.read_text()), stdout_payload)
            self.assertFalse((root / ".voice-path.json.tmp").exists())

            with self.assertRaisesRegex(ValueError, "already exists"):
                write_voice_profiles_benchmark_json(target, results)

    def test_voice_profile_json_output_rejects_stale_temp_symlinks_and_collisions(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            stale_target = root / "stale.json"
            stale_temporary = root / ".stale.json.tmp"
            stale_temporary.write_text("leftover")
            with self.assertRaisesRegex(ValueError, "Stale"):
                validate_voice_profiles_benchmark_json_output(stale_target)
            self.assertTrue(stale_temporary.is_file())

            collision_target = root / "model.json"
            with self.assertRaisesRegex(ValueError, "collides"):
                validate_voice_profiles_benchmark_json_output(
                    collision_target,
                    protected_paths=[collision_target],
                )

            symlink_target = root / "linked.json"
            symlink_target.symlink_to(root / "missing.json")
            with self.assertRaisesRegex(ValueError, "already exists"):
                validate_voice_profiles_benchmark_json_output(symlink_target)

            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                validate_voice_profiles_benchmark_json_output(linked_parent / "voice.json")

            with self.assertRaisesRegex(ValueError, ".json suffix"):
                validate_voice_profiles_benchmark_json_output(root / "voice.txt")

    def test_voice_profiles_benchmark_cli_exposes_profile_and_round_options(self) -> None:
        args = build_parser().parse_args(
            [
                "benchmark-voice-profiles",
                "--profiles",
                "fire,light",
                "--rounds",
                "2",
                "--inventory",
                "benchmarks/voice_models_q8_inventory.json",
                "--router-models-url",
                "http://127.0.0.1:18080/models",
                "--json-output",
                "/tmp/voice-path.json",
                "--json",
            ]
        )

        self.assertEqual(args.profiles, "fire,light")
        self.assertEqual(args.rounds, 2)
        self.assertEqual(args.inventory, Path("benchmarks/voice_models_q8_inventory.json"))
        self.assertEqual(args.router_models_url, "http://127.0.0.1:18080/models")
        self.assertEqual(args.json_output, Path("/tmp/voice-path.json"))
        self.assertTrue(args.json)

    def test_voice_stack_benchmark_cli_exposes_rounds_text_output_and_json_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
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
            ]
        )

        self.assertEqual(args.rounds, 5)
        self.assertEqual(args.text, "Question")
        self.assertEqual(args.tts_text, "Spoken answer")
        self.assertEqual(args.output_dir, Path("/tmp/voice-stack"))
        self.assertTrue(args.json)


def _fake_tts_measurement(
    path: Path,
    *,
    requested_model: str,
    latency_ms: int,
) -> VoiceTTSMeasurement:
    payload = _wav_payload(seconds=1.0, sample_rate=8000)
    return VoiceTTSMeasurement(
        audio=RealtimeAudio(
            path=path,
            payload=payload,
            media_type="audio/wav",
            latency_ms=latency_ms,
        ),
        requested_model=requested_model,
        served_model=requested_model,
        served_model_source="x-atlas-model",
        response_headers_ms=1,
        first_audio_byte_ms=2,
        full_response_ms=latency_ms,
        server_generation_ms=latency_ms,
        audio_duration_seconds=1.0,
        real_time_factor=latency_ms / 1000,
        sample_rate_hz=8000,
        channels=1,
        sample_width_bytes=2,
        audio_integrity_verified=True,
        delivery_mode="buffered-http-response",
    )


def _wav_payload(*, seconds: float, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * round(seconds * sample_rate))
    return output.getvalue()


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
