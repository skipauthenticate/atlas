from __future__ import annotations

import wave
from io import BytesIO
import unittest

from fastapi.testclient import TestClient

from atlas_voice.tts_sidecar import (
    QwenTTSConfig,
    QwenTTSRuntime,
    SpeechRequest,
    create_app,
)


class FakeQwenModel:
    sample_rate = 24000

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        return [[0.0, 0.25, -0.25, 0.5, -0.5]], self.sample_rate


class TtsSidecarTests(unittest.TestCase):
    def make_runtime(self) -> tuple[QwenTTSRuntime, FakeQwenModel]:
        model = FakeQwenModel()
        runtime = QwenTTSRuntime(
            config=QwenTTSConfig(
                model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                voice="Aiden",
                language="English",
                instruction="Speak naturally.",
            ),
            model=model,
            state="ready",
            detail="Model is warm and ready",
        )
        return runtime, model

    def test_runtime_generates_valid_wav_with_configured_default_voice(self) -> None:
        runtime, model = self.make_runtime()

        payload, media_type, latency_ms = runtime.synthesize(
            SpeechRequest(input="  Hello   there.  ", voice="default", response_format="wav")
        )

        self.assertEqual(media_type, "audio/wav")
        self.assertGreaterEqual(latency_ms, 0)
        with wave.open(BytesIO(payload), "rb") as audio:
            self.assertEqual(audio.getframerate(), 24000)
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getnframes(), 5)
        self.assertEqual(model.calls[0]["text"], "Hello there.")
        self.assertEqual(model.calls[0]["speaker"], "aiden")
        self.assertEqual(model.calls[0]["instruct"], "Speak naturally.")

    def test_runtime_supports_raw_pcm_and_requested_qwen_speaker(self) -> None:
        runtime, model = self.make_runtime()

        payload, media_type, _latency_ms = runtime.synthesize(
            SpeechRequest(input="Hello", voice="Ryan", response_format="pcm")
        )

        self.assertEqual(media_type, "audio/pcm")
        self.assertEqual(len(payload), 10)
        self.assertEqual(model.calls[0]["speaker"], "ryan")

    def test_health_and_speech_endpoints_report_warm_model(self) -> None:
        runtime, _model = self.make_runtime()
        app = create_app(runtime)

        with TestClient(app) as client:
            health = client.get("/health")
            speech = client.post(
                "/v1/audio/speech",
                json={
                    "model": "tts-1",
                    "input": "Hello Atlas",
                    "voice": "atlas",
                    "response_format": "wav",
                },
            )

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ready")
        self.assertTrue(health.json()["model_loaded"])
        self.assertEqual(speech.status_code, 200)
        self.assertEqual(speech.headers["content-type"], "audio/wav")
        self.assertEqual(speech.headers["x-atlas-voice"], "Aiden")
        self.assertEqual(speech.headers["x-audio-sample-rate"], "24000")

    def test_speech_endpoint_returns_503_while_model_loads(self) -> None:
        runtime = QwenTTSRuntime(state="loading", detail="Loading model")
        app = create_app(runtime)

        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello", "voice": "default", "response_format": "wav"},
            )

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
