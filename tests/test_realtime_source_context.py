from __future__ import annotations

import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from atlas_voice.realtime import RealtimeReply


class RealtimeSourceContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.environment = patch.dict(
            os.environ,
            {
                "ATLAS_VOICE_DATA_DIR": str(self.root / "data"),
                "ATLAS_VOICE_MODELS_DIR": str(self.root / "models"),
                "ATLAS_VOICE_HF_CACHE": str(self.root / "cache" / "huggingface"),
                "ATLAS_VOICE_ASSISTANT_CONFIG": str(self.root / "config" / "atlas.assistant.yaml"),
                "ATLAS_ASSISTANT_ENABLED": "true",
                "ATLAS_VOICE_TTS_PROVIDER": "none",
                "ATLAS_VOICE_STUB_MODE": "true",
                "WHISPERX_DEVICE": "cpu",
                "WHISPERX_MODEL": "tiny.en",
                "WHISPERX_COMPUTE_TYPE": "int8",
            },
            clear=False,
        )
        self.environment.start()

        import atlas_voice.web.app as web_app

        self.web_app = importlib.reload(web_app)
        self.web_app.settings.ensure_directories()
        self.web_app.db.initialize()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary_directory.cleanup()

    def test_source_context_scopes_local_data_without_leaking_paths(self) -> None:
        seeded = self._seed_sources()

        recordings = self.web_app._build_realtime_source_context(
            "roadmap",
            "recordings",
            current_session_id=seeded["current_session_id"],
        ).lower()
        uploads = self.web_app._build_realtime_source_context(
            "roadmap",
            "uploads",
            current_session_id=seeded["current_session_id"],
        ).lower()
        voice = self.web_app._build_realtime_source_context(
            "roadmap",
            "voice",
            current_session_id=seeded["current_session_id"],
        ).lower()
        all_sources = self.web_app._build_realtime_source_context(
            "roadmap",
            "",
            current_session_id=seeded["current_session_id"],
        ).lower()

        self.assertIn("roadmap recording evidence", recordings)
        self.assertIn("roadmap upload evidence", recordings)
        self.assertNotIn("roadmap voice evidence", recordings)

        self.assertIn("roadmap upload evidence", uploads)
        self.assertNotIn("roadmap recording evidence", uploads)
        self.assertNotIn("roadmap voice evidence", uploads)

        self.assertIn("roadmap voice evidence", voice)
        self.assertIn("roadmap voice conclusion", voice)
        self.assertNotIn("roadmap recording evidence", voice)
        self.assertNotIn("roadmap upload evidence", voice)

        self.assertIn("roadmap recording evidence", all_sources)
        self.assertIn("roadmap upload evidence", all_sources)
        self.assertIn("roadmap voice evidence", all_sources)
        for context in (recordings, uploads, voice, all_sources):
            self.assertNotIn("roadmap ambient secret", context)
            self.assertNotIn("roadmap current secret", context)
            self.assertNotIn(str(seeded["external_path"]).lower(), context)
            self.assertNotIn(str(seeded["upload_path"]).lower(), context)
            self.assertLessEqual(
                len(context),
                self.web_app.MAX_REALTIME_SOURCE_CONTEXT_CHARS,
            )

    def test_source_context_is_bounded_to_six_thousand_characters(self) -> None:
        context = self.web_app._bounded_realtime_source_context(["x" * 7000])

        self.assertEqual(len(context), 6000)

    def test_invalid_source_scope_emits_error_without_persisting_input(self) -> None:
        client = TestClient(self.web_app.app)

        with patch.object(self.web_app, "generate_realtime_reply") as generate_reply:
            with client.websocket_connect("/v1/realtime") as websocket:
                created = websocket.receive_json()
                websocket.send_json(
                    {
                        "type": "input_text",
                        "text": "roadmap",
                        "source_context": "filesystem",
                    }
                )
                error = websocket.receive_json()

        self.assertEqual(error["type"], "error")
        self.assertEqual(error["error"]["event_type"], "input_text")
        self.assertIn("source_context", error["error"]["message"])
        self.assertEqual(
            self.web_app.db.list_utterances(created["session"]["id"]),
            [],
        )
        generate_reply.assert_not_called()

    def test_websocket_adds_context_only_to_turn_instructions(self) -> None:
        self._seed_sources()
        client = TestClient(self.web_app.app)
        captured: list[dict[str, object]] = []
        base_instructions = self.web_app._realtime_instructions()

        def reply_with_capture(
            _text: str,
            _settings: object,
            *,
            instructions: str,
            history: list[dict[str, str]],
        ) -> RealtimeReply:
            captured.append(
                {
                    "instructions": instructions,
                    "history": [dict(item) for item in history],
                }
            )
            return RealtimeReply(text="Scoped reply", latency_ms=1)

        with patch.object(
            self.web_app,
            "generate_realtime_reply",
            side_effect=reply_with_capture,
        ):
            with client.websocket_connect("/v1/realtime") as websocket:
                websocket.receive_json()
                websocket.send_json(
                    {
                        "type": "input_text",
                        "text": "roadmap",
                        "source_context": "uploads",
                    }
                )
                _receive_until(websocket, "response.done")
                websocket.send_json(
                    {
                        "type": "input_text",
                        "text": "roadmap",
                        "source_context": "voice",
                    }
                )
                _receive_until(websocket, "response.done")

        first_instructions = str(captured[0]["instructions"]).lower()
        second_instructions = str(captured[1]["instructions"]).lower()
        self.assertTrue(str(captured[0]["instructions"]).startswith(base_instructions))
        self.assertIn("roadmap upload evidence", first_instructions)
        self.assertNotIn("roadmap recording evidence", first_instructions)
        self.assertNotIn("roadmap voice evidence", first_instructions)
        self.assertIn("roadmap voice evidence", second_instructions)
        self.assertNotIn("roadmap upload evidence", second_instructions)
        self.assertEqual(captured[0]["history"], [])
        self.assertEqual(
            captured[1]["history"],
            [
                {"role": "user", "content": "roadmap"},
                {"role": "assistant", "content": "Scoped reply"},
            ],
        )
        self.assertNotIn("local_context", str(captured[1]["history"]))
        self.assertEqual(self.web_app._realtime_instructions(), base_instructions)

    def test_session_scope_applies_to_spoken_commit(self) -> None:
        self._seed_sources()
        client = TestClient(self.web_app.app)
        captured_instructions: list[str] = []

        def reply_with_capture(
            _text: str,
            _settings: object,
            *,
            instructions: str,
            history: list[dict[str, str]],
        ) -> RealtimeReply:
            del history
            captured_instructions.append(instructions)
            return RealtimeReply(text="Scoped voice reply", latency_ms=1)

        with patch.object(
            self.web_app,
            "generate_realtime_reply",
            side_effect=reply_with_capture,
        ):
            with client.websocket_connect("/v1/realtime") as websocket:
                websocket.receive_json()
                websocket.send_json(
                    {"type": "session.update", "source_context": "recordings"}
                )
                updated = _receive_until(websocket, "session.updated")[-1]
                websocket.send_json(
                    {
                        "type": "input_audio_buffer.commit",
                        "transcript": "roadmap",
                    }
                )
                _receive_until(websocket, "response.done")

        self.assertEqual(updated["session"]["source_context"], "recordings")
        instructions = captured_instructions[0].lower()
        self.assertIn("roadmap recording evidence", instructions)
        self.assertIn("roadmap upload evidence", instructions)
        self.assertNotIn("roadmap voice evidence", instructions)

    def _seed_sources(self) -> dict[str, object]:
        external_path = self.root / "external" / "meeting.wav"
        external_id = self._recording(
            external_path,
            title="External meeting",
            text="roadmap recording evidence",
        )
        upload_path = self.web_app.settings.inbox_dir / "uploads" / "uploaded.wav"
        upload_id = self._recording(
            upload_path,
            title="Dashboard upload",
            text="roadmap upload evidence",
        )
        self.assertNotEqual(external_id, upload_id)

        voice_session_id = self.web_app.db.create_ambient_session(
            mode="direct_voice",
            source="websocket",
            title="Past roadmap chat",
        )
        utterance_id = self.web_app.db.add_utterance(
            session_id=voice_session_id,
            text="roadmap voice evidence",
            source_provider="text",
        )
        self.web_app.db.add_assistant_turn(
            session_id=voice_session_id,
            user_utterance_id=utterance_id,
            text="roadmap voice conclusion",
            model="test",
        )
        self.web_app.db.end_ambient_session(voice_session_id)

        ambient_session_id = self.web_app.db.create_ambient_session(
            mode="ambient",
            source="mic",
            title="Passive room audio",
        )
        self.web_app.db.add_utterance(
            session_id=ambient_session_id,
            text="roadmap ambient secret",
            source_provider="test",
        )
        self.web_app.db.end_ambient_session(ambient_session_id)

        current_session_id = self.web_app.db.create_ambient_session(
            mode="direct_voice",
            source="websocket",
            title="Current chat",
        )
        self.web_app.db.add_utterance(
            session_id=current_session_id,
            text="roadmap current secret",
            source_provider="text",
        )
        return {
            "current_session_id": current_session_id,
            "external_path": external_path.resolve(),
            "upload_path": upload_path.resolve(),
        }

    def _recording(self, path: Path, *, title: str, text: str) -> str:
        recording_id = self.web_app.db.create_recording(path, title=title)
        self.web_app.db.replace_segments(
            recording_id,
            [
                {
                    "start": 0,
                    "end": 1,
                    "speaker": "SPEAKER_00",
                    "text": text,
                }
            ],
        )
        return recording_id


def _receive_until(websocket: object, event_type: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for _ in range(100):
        event = websocket.receive_json()
        events.append(event)
        if event.get("type") == event_type:
            return events
    raise AssertionError(f"Realtime event not received: {event_type}")


if __name__ == "__main__":
    unittest.main()
