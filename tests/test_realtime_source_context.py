from __future__ import annotations

import importlib
import os
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from atlas_voice.realtime import RealtimeReply
from atlas_voice.web_search import WebSearchResponse, WebSearchResult


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

    def test_source_context_is_bounded_for_low_latency_voice_prefill(self) -> None:
        context = self.web_app._bounded_realtime_source_context(["x" * 7000])

        self.assertEqual(len(context), 1800)

    def test_generic_history_browses_real_evidence_and_filters_stale_refusals(self) -> None:
        recording_id = self._recording(
            self.root / "Akshaya-planning.wav",
            title="Akshaya planning",
            text="little bit of a little bit repeated noise",
        )
        self.web_app.db.update_recording(recording_id, status="done")
        self.web_app.db.save_summary(
            recording_id,
            "Akshaya pricing and launch strategy were discussed.",
            model="test",
        )
        past_id = self.web_app.db.create_ambient_session(
            mode="direct_voice",
            source="websocket",
            title="Travel chat",
        )
        utterance_id = self.web_app.db.add_utterance(
            session_id=past_id,
            text="I planned two weeks in Vietnam and then Bali for a wedding.",
            source_provider="faster-whisper",
        )
        self.web_app.db.add_utterance(
            session_id=past_id,
            text="Can you tell me about my past conversations?",
            source_provider="faster-whisper",
        )
        self.web_app.db.add_assistant_turn(
            session_id=past_id,
            user_utterance_id=utterance_id,
            text=(
                "I don't have access to your past conversations or any history with Akshaya. "
                "I only know what we discuss in our current session."
            ),
            model="test",
        )
        current_id = self.web_app.db.create_ambient_session(
            mode="direct_voice",
            source="websocket",
            title="Current chat",
        )

        exact_query = "Tell me a little bit about some of my past conversations."
        context, hit_count = self.web_app._build_realtime_source_context_result(
            exact_query,
            "",
            current_session_id=current_id,
        )
        named_context = self.web_app._build_realtime_source_context(
            "What did I discuss with Akshaya?",
            "",
            current_session_id=current_id,
        )

        self.assertEqual(self.web_app._local_retrieval_query(exact_query), "")
        self.assertGreaterEqual(hit_count, 2)
        self.assertIn("Vietnam and then Bali", context)
        self.assertIn("Akshaya pricing and launch strategy", context)
        self.assertNotIn("little bit of a little bit", context)
        self.assertNotIn("don't have access", context.lower())
        self.assertNotIn("current session", context.lower())
        self.assertNotIn("Can you tell me about my past conversations", context)
        self.assertIn("Recording title match: Akshaya planning", named_context)
        self.assertIn("Akshaya pricing and launch strategy", named_context)
        self.assertNotIn("don't have access", named_context.lower())
        self.assertLessEqual(len(context), self.web_app.MAX_REALTIME_SOURCE_CONTEXT_CHARS)

    def test_websocket_switches_voice_profile_and_preserves_history(self) -> None:
        client = TestClient(self.web_app.app)
        settings_view = self.web_app._voice_settings_view()
        captured: list[dict[str, object]] = []
        retrieval_budgets: list[int | None] = []

        def reply_with_capture(
            text: str,
            current_settings: object,
            *,
            instructions: str,
            history: list[dict[str, str]],
            retrieval_context: str | None = None,
        ) -> RealtimeReply:
            del instructions, retrieval_context
            captured.append(
                {
                    "text": text,
                    "model": current_settings.llm_model,
                    "history": [dict(item) for item in history],
                }
            )
            return RealtimeReply(text=f"Reply to {text}", latency_ms=1)

        async def retrieval_with_capture(
            _query: str,
            *,
            max_chars: int | None = None,
            **_kwargs: object,
        ) -> object:
            retrieval_budgets.append(max_chars)
            return self.web_app.RealtimeRetrieval(
                context="",
                searched_local=True,
                local_hit_count=0,
                web=None,
                latency_ms=0,
            )

        with (
            patch.object(
                self.web_app,
                "generate_realtime_reply",
                side_effect=reply_with_capture,
            ),
            patch.object(
                self.web_app,
                "_build_realtime_retrieval",
                side_effect=retrieval_with_capture,
            ),
        ):
            with client.websocket_connect(
                f"/v1/realtime?token={self.web_app._REALTIME_ACCESS_TOKEN}"
            ) as websocket:
                created = websocket.receive_json()
                websocket.send_json(
                    {
                        "type": "input_text",
                        "text": "first tier request",
                        "source_context": "recordings",
                    }
                )
                _receive_until(websocket, "response.done")
                websocket.send_json(
                    {"type": "session.update", "model_tier": "light"}
                )
                updated = _receive_until(websocket, "session.updated")[-1]
                websocket.send_json(
                    {
                        "type": "input_text",
                        "text": "second tier request",
                        "source_context": "recordings",
                    }
                )
                done = _receive_until(websocket, "response.done")[-1]
                websocket.send_json(
                    {"type": "session.update", "voice_profile": "spark"}
                )
                error = websocket.receive_json()

        session = created["session"]
        serialized_profiles = str(session["voice_profiles"])
        self.assertEqual(settings_view["default_voice_profile"], "torch")
        self.assertEqual(
            [profile["id"] for profile in settings_view["voice_profiles"]],
            ["light", "torch", "fire"],
        )
        self.assertEqual(session["voice_profile"], "torch")
        self.assertEqual(session["model_tier"], "torch")
        self.assertEqual(session["model"], "qwen3.5-9b")
        self.assertEqual(
            [profile["id"] for profile in session["voice_profiles"]],
            ["light", "torch", "fire"],
        )
        self.assertNotIn("llm_profile", serialized_profiles)
        self.assertNotIn("http://", serialized_profiles)

        self.assertEqual(updated["session"]["voice_profile"], "light")
        self.assertEqual(updated["session"]["model_tier"], "light")
        self.assertEqual(updated["session"]["model"], "qwen3.5-2b")
        self.assertEqual(updated["session"]["context_budget_chars"], 3000)
        self.assertEqual(done["response"]["voice_profile"], "light")
        self.assertEqual(done["response"]["model_tier"], "light")
        self.assertEqual(done["response"]["model"], "qwen3.5-2b")
        self.assertEqual(retrieval_budgets, [6000, 3000])
        self.assertEqual(captured[0]["history"], [])
        self.assertEqual(
            captured[1]["history"],
            [
                {"role": "user", "content": "first tier request"},
                {"role": "assistant", "content": "Reply to first tier request"},
            ],
        )
        self.assertEqual(error["type"], "error")
        self.assertIn("light, torch, fire", error["error"]["message"])

    def test_voice_profile_switch_cancels_active_response_first(self) -> None:
        client = TestClient(self.web_app.app)
        started = threading.Event()
        release = threading.Event()

        def slow_reply(*_args: object, **_kwargs: object) -> RealtimeReply:
            started.set()
            release.wait(2.0)
            return RealtimeReply(text="Stale reply", latency_ms=100)

        with patch.object(
            self.web_app,
            "generate_realtime_reply",
            side_effect=slow_reply,
        ):
            with client.websocket_connect(
                f"/v1/realtime?token={self.web_app._REALTIME_ACCESS_TOKEN}"
            ) as websocket:
                try:
                    websocket.receive_json()
                    websocket.send_json(
                        {"type": "input_text", "text": "keep working"}
                    )
                    _receive_until(websocket, "response.created")
                    self.assertTrue(started.wait(1.0))
                    websocket.send_json(
                        {
                            "type": "session.update",
                            "session": {"voice_profile": "fire"},
                        }
                    )
                    events = _receive_until(websocket, "session.updated")
                finally:
                    release.set()

        cancelled = next(
            event for event in events if event["type"] == "response.cancelled"
        )
        updated = events[-1]["session"]
        self.assertEqual(cancelled["reason"], "voice_profile_changed")
        self.assertEqual(updated["voice_profile"], "fire")
        self.assertEqual(updated["model_tier"], "fire")
        self.assertEqual(updated["model"], "qwen3.6-35b-a3b")

    def test_websocket_forwards_streaming_deltas_and_metrics(self) -> None:
        client = TestClient(self.web_app.app)
        release = threading.Event()
        finished = threading.Event()

        def streaming_reply(
            _text: str,
            _settings: object,
            *,
            on_text_delta: object,
            **_kwargs: object,
        ) -> RealtimeReply:
            on_text_delta("Hello")
            release.wait(2.0)
            on_text_delta(" world")
            finished.set()
            return RealtimeReply(
                text="Hello world",
                latency_ms=80,
                tokens_in=10,
                tokens_out=2,
                served_model="/private/models/Qwen 3.5:9B.gguf",
                ttft_ms=25,
                tokens_per_second=36.5,
            )

        with patch.object(
            self.web_app,
            "generate_realtime_reply",
            side_effect=streaming_reply,
        ):
            with client.websocket_connect(
                f"/v1/realtime?token={self.web_app._REALTIME_ACCESS_TOKEN}"
            ) as websocket:
                try:
                    created = websocket.receive_json()
                    session_id = created["session"]["id"]
                    websocket.send_json(
                        {"type": "input_text", "text": "stream this"}
                    )
                    first_events = _receive_until(websocket, "response.text.delta")
                    self.assertEqual(first_events[-1]["delta"], "Hello")
                    self.assertFalse(finished.is_set())
                    release.set()
                    remaining_events = _receive_until(websocket, "response.done")
                finally:
                    release.set()

        events = first_events + remaining_events
        deltas = [
            event["delta"]
            for event in events
            if event["type"] == "response.text.delta"
        ]
        done = events[-1]["response"]
        self.assertEqual(deltas, ["Hello", " world"])
        self.assertEqual(done["model"], "Qwen_3.5:9B.gguf")
        self.assertEqual(done["requested_model"], "qwen3.5-9b")
        self.assertEqual(done["served_model"], "Qwen_3.5:9B.gguf")
        self.assertEqual(done["ttft_ms"], 25)
        self.assertEqual(done["tokens_per_second"], 36.5)
        self.assertEqual(
            self.web_app.db.list_assistant_turns(session_id)[0]["model"],
            "Qwen_3.5:9B.gguf",
        )
        model_runs = [
            run
            for run in self.web_app.db.list_model_runs()
            if run["task"] == "realtime_chat"
        ]
        self.assertEqual(model_runs[0]["model"], "Qwen_3.5:9B.gguf")

    def test_websocket_cancellation_stops_stream_callback_work(self) -> None:
        client = TestClient(self.web_app.app)
        allow_next_delta = threading.Event()
        callback_stopped = threading.Event()
        continued_work = threading.Event()

        def cancellable_stream(
            _text: str,
            _settings: object,
            *,
            on_text_delta: object,
            **_kwargs: object,
        ) -> RealtimeReply:
            try:
                on_text_delta("first")
                allow_next_delta.wait(2.0)
                on_text_delta("stale")
            except Exception:
                callback_stopped.set()
                raise
            continued_work.set()
            return RealtimeReply(text="first stale", latency_ms=100)

        with patch.object(
            self.web_app,
            "generate_realtime_reply",
            side_effect=cancellable_stream,
        ):
            with client.websocket_connect(
                f"/v1/realtime?token={self.web_app._REALTIME_ACCESS_TOKEN}"
            ) as websocket:
                try:
                    websocket.receive_json()
                    websocket.send_json(
                        {"type": "input_text", "text": "cancel this stream"}
                    )
                    first_events = _receive_until(websocket, "response.text.delta")
                    self.assertEqual(first_events[-1]["delta"], "first")
                    websocket.send_json(
                        {"type": "response.cancel", "reason": "test_cancel"}
                    )
                    cancelled = websocket.receive_json()
                    allow_next_delta.set()
                    self.assertTrue(callback_stopped.wait(1.0))
                    websocket.send_json(
                        {"type": "session.update", "instructions": "Still connected"}
                    )
                    after_cancel = _receive_until(websocket, "session.updated")
                finally:
                    allow_next_delta.set()

        self.assertEqual(cancelled["type"], "response.cancelled")
        self.assertEqual(cancelled["reason"], "test_cancel")
        self.assertFalse(continued_work.is_set())
        self.assertNotIn(
            "response.text.delta",
            [event["type"] for event in after_cancel],
        )

    def test_invalid_source_scope_emits_error_without_persisting_input(self) -> None:
        client = TestClient(self.web_app.app)

        with patch.object(self.web_app, "generate_realtime_reply") as generate_reply:
            with client.websocket_connect(f"/v1/realtime?token={self.web_app._REALTIME_ACCESS_TOKEN}") as websocket:
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
            retrieval_context: str | None = None,
        ) -> RealtimeReply:
            captured.append(
                {
                    "instructions": instructions,
                    "history": [dict(item) for item in history],
                    "retrieval_context": retrieval_context,
                }
            )
            return RealtimeReply(text="Scoped reply", latency_ms=1)

        with patch.object(
            self.web_app,
            "generate_realtime_reply",
            side_effect=reply_with_capture,
        ):
            with client.websocket_connect(f"/v1/realtime?token={self.web_app._REALTIME_ACCESS_TOKEN}") as websocket:
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

        first_context = str(captured[0]["retrieval_context"]).lower()
        second_context = str(captured[1]["retrieval_context"]).lower()
        self.assertEqual(captured[0]["instructions"], base_instructions)
        self.assertEqual(captured[1]["instructions"], base_instructions)
        self.assertIn("roadmap upload evidence", first_context)
        self.assertNotIn("roadmap recording evidence", first_context)
        self.assertNotIn("roadmap voice evidence", first_context)
        self.assertIn("roadmap voice evidence", second_context)
        self.assertNotIn("roadmap upload evidence", second_context)
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
            retrieval_context: str | None = None,
        ) -> RealtimeReply:
            del history
            self.assertEqual(instructions, self.web_app._realtime_instructions())
            captured_instructions.append(str(retrieval_context or ""))
            return RealtimeReply(text="Scoped voice reply", latency_ms=1)

        with patch.object(
            self.web_app,
            "generate_realtime_reply",
            side_effect=reply_with_capture,
        ):
            with client.websocket_connect(f"/v1/realtime?token={self.web_app._REALTIME_ACCESS_TOKEN}") as websocket:
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

    def test_web_scope_emits_provenance_and_passes_untrusted_evidence(self) -> None:
        client = TestClient(self.web_app.app)
        captured_context: list[str] = []
        search_response = WebSearchResponse(
            query="latest JetPack",
            provider="searxng",
            results=(
                WebSearchResult(
                    title="JetPack SDK",
                    url="https://developer.nvidia.com/embedded/jetpack",
                    snippet="JetPack release evidence",
                    source="duckduckgo",
                ),
            ),
            latency_ms=12,
        )

        def reply_with_capture(
            _text: str,
            _settings: object,
            *,
            instructions: str,
            history: list[dict[str, str]],
            retrieval_context: str | None = None,
        ) -> RealtimeReply:
            del instructions, history
            captured_context.append(str(retrieval_context or ""))
            return RealtimeReply(text="Current answer", latency_ms=1)

        with (
            patch.object(self.web_app, "search_web", return_value=search_response),
            patch.object(
                self.web_app,
                "generate_realtime_reply",
                side_effect=reply_with_capture,
            ),
        ):
            with client.websocket_connect(f"/v1/realtime?token={self.web_app._REALTIME_ACCESS_TOKEN}") as websocket:
                websocket.receive_json()
                websocket.send_json(
                    {
                        "type": "input_text",
                        "text": "latest JetPack",
                        "source_context": "web",
                    }
                )
                events = _receive_until(websocket, "response.done")

        by_type = {event["type"]: event for event in events}
        self.assertIn("response.retrieval.started", by_type)
        self.assertEqual(by_type["response.retrieval.done"]["web_result_count"], 1)
        self.assertEqual(
            by_type["response.retrieval.done"]["sources"][0]["url"],
            "https://developer.nvidia.com/embedded/jetpack",
        )
        self.assertIn("JetPack release evidence", captured_context[0])
        self.assertNotIn("https://developer.nvidia.com", captured_context[0])
        self.assertEqual(self.web_app.db.list_model_runs()[0]["task"], "realtime_chat")
        self.assertEqual(self.web_app.db.list_model_runs()[1]["task"], "web_search")

    def test_auto_scope_gates_retrieval_by_intent(self) -> None:
        current_settings = self.web_app._direct_voice_settings()

        self.assertEqual(
            self.web_app._realtime_retrieval_plan("Tell me a joke", "", current_settings),
            (False, False),
        )
        self.assertEqual(
            self.web_app._realtime_retrieval_plan(
                "What did we say in the last meeting?", "", current_settings
            ),
            (True, False),
        )
        self.assertEqual(
            self.web_app._realtime_retrieval_plan(
                "latest JetPack news", "", current_settings
            ),
            (False, True),
        )
        self.assertEqual(
            self.web_app._realtime_retrieval_plan(
                "What was the latest thing we said in the Project Zephyr meeting?",
                "",
                current_settings,
            ),
            (True, False),
        )

    def test_selected_recording_context_is_exact_and_disables_web(self) -> None:
        first_id = self._recording(
            self.root / "first.wav",
            title="First project",
            text="zephyr launch belongs only to first",
        )
        second_id = self._recording(
            self.root / "second.wav",
            title="Second project",
            text="zephyr budget belongs only to second",
        )
        self.web_app.db.save_summary(
            first_id,
            "## Snapshot\nFirst project launch plan.",
            model="test",
        )
        self.web_app.db.save_summary(
            second_id,
            "## Snapshot\nSecond project budget plan.",
            model="test",
        )
        current_id = self.web_app.db.create_ambient_session(
            mode="direct_voice",
            source="websocket",
            title="Current",
        )

        context = self.web_app._build_realtime_source_context(
            "zephyr",
            "recordings",
            current_session_id=current_id,
            recording_id=first_id,
        ).lower()
        plan = self.web_app._realtime_retrieval_plan(
            "latest zephyr news",
            "web",
            self.web_app._direct_voice_settings(),
            recording_id=first_id,
        )

        self.assertIn("exclusive selected recording: first project", context)
        self.assertIn("first project launch plan", context)
        self.assertIn("belongs only to first", context)
        self.assertNotIn("second project", context)
        self.assertNotIn("belongs only to second", context)
        self.assertEqual(plan, (True, False))

    def test_selected_recording_gist_spans_raw_beginning_middle_and_end(self) -> None:
        facts = {
            0: "BEGIN_RAW_FACT customer retention opened the discussion.",
            15: "MIDDLE_RAW_FACT pricing experiment changed direction.",
            29: "END_RAW_FACT Morgan accepted the final action.",
        }
        segments = [
            {
                "start": index * 3.0,
                "end": index * 3.0 + 2.0,
                "speaker": f"Speaker {index % 3}",
                "text": self._padded_context(
                    facts.get(index, f"Routine discussion number {index}.")
                ),
            }
            for index in range(30)
        ]
        selected_id = self._recording_with_segments(
            self.root / "long-selected.wav",
            title="Long selected strategy",
            segments=segments,
        )
        self.web_app.db.save_summary(
            selected_id,
            "A derived summary that deliberately omits the raw timeline facts.",
            model="test",
        )
        self._recording(
            self.root / "other-long.wav",
            title="Other recording",
            text="OTHER_RECORDING_SECRET must never leak",
        )

        context, evidence_count = self.web_app._build_realtime_source_context_result(
            "Give me the gist of this whole conversation",
            "recordings",
            current_session_id="unused",
            recording_id=selected_id,
            max_chars=3600,
        )

        self.assertGreaterEqual(evidence_count, 3)
        self.assertIn("Transcript evidence", context)
        self.assertIn("BEGIN_RAW_FACT", context)
        self.assertIn("MIDDLE_RAW_FACT", context)
        self.assertIn("END_RAW_FACT", context)
        self.assertNotIn("OTHER_RECORDING_SECRET", context)
        self.assertLessEqual(len(context), 3600)

    def test_selected_recording_speaker_match_keeps_adjacent_turns(self) -> None:
        turns = [
            ("Drew", "Opening context."),
            ("Bob", "BEFORE_TURN the supplier raised a constraint."),
            ("Alice", "ALICE_TITANIUM said titanium is the safer material."),
            ("Carol", "AFTER_TURN agreed with that recommendation."),
            ("Drew", "The meeting moved to another subject."),
            ("Bob", "Closing context."),
        ]
        recording_id = self._recording_with_segments(
            self.root / "speaker-selected.wav",
            title="Materials review",
            segments=[
                {
                    "start": index * 3.0,
                    "end": index * 3.0 + 2.0,
                    "speaker": speaker,
                    "text": self._padded_context(text, 230),
                }
                for index, (speaker, text) in enumerate(turns)
            ],
        )

        context, evidence_count = self.web_app._build_realtime_source_context_result(
            "What did Alice say about titanium?",
            "recordings",
            current_session_id="unused",
            recording_id=recording_id,
            max_chars=1350,
        )

        self.assertGreaterEqual(evidence_count, 1)
        self.assertIn("Bob: BEFORE_TURN", context)
        self.assertIn("Alice: ALICE_TITANIUM", context)
        self.assertIn("Carol: AFTER_TURN", context)
        self.assertLessEqual(len(context), 1350)

    def test_selected_recording_follow_up_uses_only_prior_user_turns(self) -> None:
        target = "TARGET_TESTING_FAILED the launch schedule slipped because testing failed."
        assistant_only = "OMEGA_ASSISTANT transcript marker from an unrelated topic."
        segments = [
            {
                "start": index * 3.0,
                "end": index * 3.0 + 2.0,
                "speaker": (
                    "Alice"
                    if index == 4
                    else "Morgan"
                    if index == 22
                    else f"Participant {index % 3}"
                ),
                "text": self._padded_context(
                    target
                    if index == 4
                    else assistant_only
                    if index == 22
                    else f"Routine neutral discussion number {index}."
                ),
            }
            for index in range(30)
        ]
        recording_id = self._recording_with_segments(
            self.root / "follow-up-selected.wav",
            title="Launch retrospective",
            segments=segments,
        )
        torch_budget = self.web_app.voice_profile(
            self.web_app.assistant_config,
            "torch",
        ).context_budget_chars
        no_history = self.web_app._build_realtime_source_context(
            "what about her concern?",
            "recordings",
            current_session_id="unused",
            recording_id=recording_id,
            max_chars=torch_budget,
        )
        captured_context: list[str] = []

        def reply_with_capture(
            text: str,
            _settings: object,
            *,
            instructions: str,
            history: list[dict[str, str]],
            retrieval_context: str | None = None,
        ) -> RealtimeReply:
            del instructions, history
            captured_context.append(str(retrieval_context or ""))
            reply_text = "OMEGA_ASSISTANT" if len(captured_context) == 1 else "Follow-up"
            return RealtimeReply(text=reply_text, latency_ms=1)

        client = TestClient(self.web_app.app)
        with patch.object(
            self.web_app,
            "generate_realtime_reply",
            side_effect=reply_with_capture,
        ):
            with client.websocket_connect(
                f"/v1/realtime?token={self.web_app._REALTIME_ACCESS_TOKEN}"
            ) as websocket:
                websocket.receive_json()
                websocket.send_json(
                    {
                        "type": "input_text",
                        "text": "What did Alice say about the launch schedule?",
                        "recording_id": recording_id,
                    }
                )
                _receive_until(websocket, "response.done")
                websocket.send_json(
                    {"type": "input_text", "text": "what about her concern?"}
                )
                _receive_until(websocket, "response.done")

        self.assertNotIn("TARGET_TESTING_FAILED", no_history)
        self.assertIn("TARGET_TESTING_FAILED", captured_context[1])
        self.assertNotIn("OMEGA_ASSISTANT transcript marker", captured_context[1])

    def test_selected_recording_profile_budget_changes_context_without_leakage(self) -> None:
        recording_id = self._recording_with_segments(
            self.root / "budget-selected.wav",
            title="Migration review",
            segments=[
                {
                    "start": index * 3.0,
                    "end": index * 3.0 + 2.0,
                    "speaker": f"Speaker {index % 2}",
                    "text": self._padded_context(
                        "COBALT_MIGRATION the migration begins Tuesday."
                        if index == 5
                        else f"Neutral migration context {index}.",
                        240,
                    ),
                }
                for index in range(14)
            ],
        )
        self._recording(
            self.root / "budget-other.wav",
            title="Other budget recording",
            text="BUDGET_LEAK from another recording",
        )

        small_context, small_evidence = self.web_app._build_realtime_source_context_result(
            "What was decided about the cobalt migration?",
            "recordings",
            current_session_id="unused",
            recording_id=recording_id,
            max_chars=256,
        )
        large_context, large_evidence = self.web_app._build_realtime_source_context_result(
            "What was decided about the cobalt migration?",
            "recordings",
            current_session_id="unused",
            recording_id=recording_id,
            max_chars=1800,
        )

        self.assertLessEqual(len(small_context), 256)
        self.assertLessEqual(len(large_context), 1800)
        self.assertGreater(len(large_context), len(small_context))
        self.assertGreater(large_evidence, small_evidence)
        self.assertIn("COBALT_MIGRATION", large_context)
        self.assertNotIn("BUDGET_LEAK", small_context)
        self.assertNotIn("BUDGET_LEAK", large_context)

    def test_selected_recording_profile_budgets_scale_raw_timeline_coverage(self) -> None:
        recording_id = self._recording_with_segments(
            self.root / "profile-budget-selected.wav",
            title="Long profile budget review",
            segments=[
                {
                    "start": index * 3.0,
                    "end": index * 3.0 + 2.0,
                    "speaker": f"Speaker {index % 2}",
                    "text": self._padded_context(
                        f"COVERAGE_{index:03d} selected timeline detail.",
                        220,
                    ),
                }
                for index in range(120)
            ],
        )
        self._recording(
            self.root / "profile-budget-other.wav",
            title="Other profile budget recording",
            text="PROFILE_BUDGET_LEAK must never appear",
        )

        contexts: dict[str, str] = {}
        evidence_counts: dict[str, int] = {}
        coverage: dict[str, int] = {}
        for profile_id, budget in (("light", 3000), ("torch", 6000), ("fire", 12000)):
            context, evidence_count = self.web_app._build_realtime_source_context_result(
                "Give me the gist of this whole conversation",
                "recordings",
                current_session_id="unused",
                recording_id=recording_id,
                max_chars=budget,
            )
            contexts[profile_id] = context
            evidence_counts[profile_id] = evidence_count
            coverage[profile_id] = sum(
                f"COVERAGE_{index:03d}" in context
                for index in range(120)
            )
            self.assertLessEqual(len(context), budget)
            self.assertNotIn("PROFILE_BUDGET_LEAK", context)

        for context in contexts.values():
            for marker in ("COVERAGE_000", "COVERAGE_059", "COVERAGE_119"):
                self.assertIn(marker, context)

        self.assertGreater(evidence_counts["torch"], evidence_counts["light"])
        self.assertGreater(evidence_counts["fire"], evidence_counts["torch"])
        self.assertGreater(coverage["torch"], coverage["light"] + 4)
        self.assertGreater(coverage["fire"], coverage["torch"] + 6)

    def test_switching_selected_recordings_clears_history(self) -> None:
        first_id = self._recording(
            self.root / "first.wav",
            title="First project",
            text="alpha evidence",
        )
        second_id = self._recording(
            self.root / "second.wav",
            title="Second project",
            text="beta evidence",
        )
        client = TestClient(self.web_app.app)
        captured: list[dict[str, object]] = []

        def reply_with_capture(
            _text: str,
            _settings: object,
            *,
            instructions: str,
            history: list[dict[str, str]],
            retrieval_context: str | None = None,
        ) -> RealtimeReply:
            del instructions
            captured.append(
                {
                    "history": [dict(item) for item in history],
                    "context": str(retrieval_context or ""),
                }
            )
            return RealtimeReply(text="Scoped reply", latency_ms=1)

        with (
            patch.object(self.web_app, "generate_realtime_reply", side_effect=reply_with_capture),
            patch.object(self.web_app, "search_web") as search_web,
        ):
            with client.websocket_connect(
                f"/v1/realtime?token={self.web_app._REALTIME_ACCESS_TOKEN}"
            ) as websocket:
                websocket.receive_json()
                websocket.send_json(
                    {
                        "type": "input_text",
                        "text": "alpha",
                        "source_context": "all",
                        "recording_id": first_id,
                    }
                )
                _receive_until(websocket, "response.done")
                websocket.send_json(
                    {
                        "type": "input_text",
                        "text": "beta",
                        "source_context": "all",
                        "recording_id": second_id,
                    }
                )
                _receive_until(websocket, "response.done")

        self.assertEqual(captured[0]["history"], [])
        self.assertEqual(captured[1]["history"], [])
        self.assertIn("First project", captured[0]["context"])
        self.assertNotIn("Second project", captured[0]["context"])
        self.assertIn("Second project", captured[1]["context"])
        self.assertNotIn("First project", captured[1]["context"])
        search_web.assert_not_called()

    def test_unknown_selected_recording_is_rejected_before_persistence(self) -> None:
        client = TestClient(self.web_app.app)
        missing_id = "f" * 32

        with patch.object(self.web_app, "generate_realtime_reply") as generate_reply:
            with client.websocket_connect(
                f"/v1/realtime?token={self.web_app._REALTIME_ACCESS_TOKEN}"
            ) as websocket:
                created = websocket.receive_json()
                websocket.send_json(
                    {
                        "type": "input_text",
                        "text": "private question",
                        "recording_id": missing_id,
                    }
                )
                error = websocket.receive_json()

        self.assertEqual(error["type"], "error")
        self.assertIn("not found", error["error"]["message"])
        self.assertEqual(
            self.web_app.db.list_utterances(created["session"]["id"]),
            [],
        )
        generate_reply.assert_not_called()

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

    def _recording_with_segments(
        self,
        path: Path,
        *,
        title: str,
        segments: list[dict[str, object]],
    ) -> str:
        recording_id = self.web_app.db.create_recording(path, title=title)
        self.web_app.db.replace_segments(recording_id, segments)
        return recording_id

    @staticmethod
    def _padded_context(text: str, minimum_length: int = 280) -> str:
        padding = " supporting detail" * 80
        return (text + padding)[:minimum_length].rstrip() + "."

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
