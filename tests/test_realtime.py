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
    generate_realtime_reply,
    normalize_tts_provider,
    synthesize_with_tts_sidecar,
    transcript_text,
    transcribe_realtime_audio,
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

    def test_write_realtime_audio_preserves_browser_container_audio(self) -> None:
        with TemporaryDirectory() as tmp:
            output = write_realtime_audio(
                b"\x1aE\xdf\xa3webm",
                Path(tmp),
                media_type="audio/webm;codecs=opus",
            )

            self.assertEqual(output.suffix, ".webm")
            self.assertEqual(output.read_bytes(), b"\x1aE\xdf\xa3webm")

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


    def test_transcribe_realtime_audio_marks_asr_call_realtime(self) -> None:
        settings = SimpleNamespace(
            stub_mode=False,
            realtime_audio_sample_rate=16000,
            realtime_audio_channels=1,
        )

        with TemporaryDirectory() as tmp:
            with patch(
                "atlas_voice.providers.asr.transcribe_audio",
                return_value={"text": "hello realtime"},
            ) as transcribe_mock:
                text, audio_path = transcribe_realtime_audio(b"\0\0" * 12, settings, Path(tmp))

        self.assertEqual(text, "hello realtime")
        self.assertEqual(audio_path.suffix, ".wav")
        transcribe_mock.assert_called_once()
        self.assertTrue(transcribe_mock.call_args.kwargs["realtime"])

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

    def test_generate_realtime_reply_extracts_openai_style_tool_calls(self) -> None:
        settings = SimpleNamespace(
            stub_mode=False,
            llm_base_url="http://llm.test/v1/chat/completions",
            llm_model="qwen-local",
            llm_temperature=0.2,
            llm_max_tokens=256,
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "I can search locally.",
                                "tool_calls": [
                                    {
                                        "id": "call_search",
                                        "type": "function",
                                        "function": {
                                            "name": "search_recordings",
                                            "arguments": '{"query": "Atlas"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                },
            )

        real_client = httpx.Client
        transport = httpx.MockTransport(handler)
        with patch("httpx.Client", lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs)):
            reply = generate_realtime_reply("Find Atlas", settings)

        self.assertEqual(reply.text, "I can search locally.")
        self.assertEqual(
            reply.tool_calls,
            [
                {
                    "id": "call_search",
                    "name": "search_recordings",
                    "arguments": {"query": "Atlas"},
                    "mutating": False,
                }
            ],
        )

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
