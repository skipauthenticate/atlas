from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import base64
import json
import unittest
import wave

import httpx

from atlas_voice.realtime import (
    audio_delta_payload,
    check_tts_sidecar_health,
    chunk_text,
    decode_audio_delta,
    extract_text_input,
    normalize_tts_provider,
    synthesize_with_tts_sidecar,
    transcript_text,
    write_realtime_audio,
)


class RealtimeTests(unittest.TestCase):
    def test_extract_text_input_from_openai_style_item(self) -> None:
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "content": [
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": "Atlas"},
                ],
            },
        }

        self.assertEqual(extract_text_input(event), "Hello\nAtlas")

    def test_decode_audio_delta_requires_base64(self) -> None:
        payload = b"\0\1"
        encoded = base64.b64encode(payload).decode("ascii")

        self.assertEqual(decode_audio_delta({"audio": encoded}), payload)
        with self.assertRaisesRegex(ValueError, "base64"):
            decode_audio_delta({"audio": "not base64"})

    def test_write_realtime_audio_wraps_pcm16_as_wav(self) -> None:
        with TemporaryDirectory() as tmp:
            output = write_realtime_audio(
                b"\0\0" * 160,
                Path(tmp),
                sample_rate=16000,
                channels=1,
            )

            with wave.open(str(output), "rb") as audio:
                self.assertEqual(audio.getframerate(), 16000)
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getsampwidth(), 2)

    def test_transcript_text_extracts_segments(self) -> None:
        payload = {"segments": [{"text": "hello"}, {"text": "there"}]}

        self.assertEqual(transcript_text(payload), "hello there")

    def test_chunk_text_and_audio_payload(self) -> None:
        chunks = chunk_text("one two three four", chunk_size=8)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(base64.b64decode(audio_delta_payload(b"wav")), b"wav")

    def test_synthesize_with_tts_sidecar_posts_openai_style_request(self) -> None:
        settings = SimpleNamespace(
            tts_base_url="http://tts.test/v1/audio/speech",
            tts_model="faster-qwen3-tts-0.6b",
            tts_voice="atlas",
            tts_response_format="wav",
            tts_timeout=5.0,
        )
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, content=b"RIFFtest", headers={"content-type": "audio/wav"})

        real_client = httpx.Client
        transport = httpx.MockTransport(handler)
        with TemporaryDirectory() as tmp:
            with patch("httpx.Client", lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs)):
                audio = synthesize_with_tts_sidecar("Hello Atlas", settings, Path(tmp))

        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["path"], "/v1/audio/speech")
        self.assertEqual(
            seen["payload"],
            {
                "model": "faster-qwen3-tts-0.6b",
                "input": "Hello Atlas",
                "voice": "atlas",
                "response_format": "wav",
            },
        )
        self.assertEqual(audio.payload, b"RIFFtest")
        self.assertEqual(audio.media_type, "audio/wav")
        self.assertEqual(audio.path.suffix, ".wav")
        self.assertIsNotNone(audio.latency_ms)

    def test_check_tts_sidecar_health_uses_configured_health_url(self) -> None:
        settings = SimpleNamespace(
            tts_health_url="http://tts.test/health",
            tts_health_timeout=0.5,
        )

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.url.path, "/health")
            return httpx.Response(200, json={"status": "ok"})

        real_client = httpx.Client
        transport = httpx.MockTransport(handler)
        with patch("httpx.Client", lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs)):
            health = check_tts_sidecar_health(settings)

        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["detail"], "ok")
        self.assertEqual(health["url"], "http://tts.test/health")

    def test_normalize_tts_provider_aliases_sidecar_names(self) -> None:
        self.assertEqual(normalize_tts_provider("qwen3_tts"), "faster-qwen3-tts")
        self.assertEqual(normalize_tts_provider("piper"), "piper")


if __name__ == "__main__":
    unittest.main()
