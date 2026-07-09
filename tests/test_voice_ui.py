import re
from pathlib import Path
import unittest


class VoiceUiStaticTests(unittest.TestCase):
    @staticmethod
    def _last_css_block(css: str, selector: str) -> str:
        matches = list(re.finditer(rf"{re.escape(selector)}\s*{{(?P<body>.*?)}}", css, flags=re.S))
        if not matches:
            raise AssertionError(f"Missing CSS selector: {selector}")
        return max((match.group("body") for match in matches), key=len)

    @staticmethod
    def _script_between(script: str, start: str, end: str) -> str:
        start_index = script.find(start)
        if start_index == -1:
            raise AssertionError(f"Missing script marker: {start}")
        end_index = script.find(end, start_index)
        if end_index == -1:
            raise AssertionError(f"Missing script marker: {end}")
        return script[start_index:end_index]

    def test_voice_page_uses_shared_shell_and_focused_workspace_order(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        base = Path("atlas_voice/web/templates/base.html").read_text()

        self.assertIn("{% block body_class %}voice-page{% endblock %}", template)
        self.assertIn('data-layout="atlas-voice-workbench"', template)
        self.assertIn('<aside class="sidebar"', base)
        self.assertIn('id="app-settings-dialog"', base)
        self.assertNotIn('<nav class="voice-rail"', template)
        ordered = [
            'class="voice-appbar"',
            'class="voice-tool-tabs"',
            'class="voice-session-layout"',
            'class="voice-main"',
            'class="voice-transcript-panel"',
            'class="voice-transport"',
        ]
        positions = [template.index(marker) for marker in ordered]
        self.assertEqual(positions, sorted(positions))

    def test_voice_ui_uses_clean_white_black_theme_without_purple(self) -> None:
        css = Path("atlas_voice/web/static/app.css").read_text().lower()

        for purple in ("#5867d8", "#3743a4", "#9eabff", "#c8d0ff", "#6865b8"):
            self.assertNotIn(purple, css)
        self.assertIn("color-scheme: light", css)
        self.assertIn("--surface: #ffffff", css)
        self.assertIn("--ink: #171717", css)
        self.assertIn("--accent: #171717", css)
        self.assertIn("--voice-accent: #ff5b35", css)
        self.assertIn("background: #ffffff", self._last_css_block(css, ".voice-workbench"))

    def test_voice_page_uses_local_lucide_assets_without_reference_branding(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text().lower()
        base = Path("atlas_voice/web/templates/base.html").read_text().lower()
        production = template + base + Path("atlas_voice/web/static/voice.js").read_text().lower()

        self.assertNotIn("elevenlabs", production)
        self.assertNotIn("eleven labs", production)
        referenced = set(re.findall(r'/static/icons/([^"?]+\.svg)', template + base))
        self.assertGreaterEqual(len(referenced), 12)
        for name in referenced:
            self.assertTrue(Path("atlas_voice/web/static/icons", name).is_file(), name)

    def test_voice_bubble_is_canvas_driven_by_input_and_output_audio(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        script = Path("atlas_voice/web/static/voice.js").read_text()

        self.assertIn("data-voice-bubble", template)
        self.assertIn('data-voice-state="idle"', template)
        self.assertIn("class VoiceBubble", script)
        self.assertIn("createMediaElementSource", script)
        self.assertIn("getByteTimeDomainData", script)
        self.assertIn("bubble.setLevel", script)
        for state in ("listening", "thinking", "synthesizing", "speaking"):
            self.assertIn(f"'{state}'", script)

    def test_browser_capture_streams_pcm_for_server_vad(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()

        self.assertIn("navigator.mediaDevices.getUserMedia", script)
        self.assertIn("AudioWorkletNode", script)
        self.assertIn("createScriptProcessor", script)
        self.assertIn("downsampleToPCM16", script)
        self.assertIn("echoCancellation: true", script)
        self.assertIn("noiseSuppression: true", script)
        self.assertIn("input_audio_buffer.append", script)
        self.assertIn("input_audio_buffer.commit", script)
        self.assertIn("input_audio_buffer.clear", script)
        self.assertIn("media_type: 'audio/pcm'", script)
        self.assertNotIn("new MediaRecorder", script)
        self.assertNotIn("audio/webm", script)

    def test_voice_client_handles_automatic_turn_and_barge_in_events(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()

        for event_type in (
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "conversation.item.input_audio_transcription.done",
            "response.cancelled",
            "response.interrupted",
        ):
            self.assertIn(event_type, script)
        self.assertIn("clearPlayback()", script)

    def test_pause_and_private_stop_capture_without_committing(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()
        pause_handler = self._script_between(
            script,
            "controls.get('pause')?.addEventListener",
            "controls.get('private')?.addEventListener",
        )
        private_handler = self._script_between(
            script,
            "controls.get('private')?.addEventListener",
            "controls.get('interrupt')?.addEventListener",
        )

        for handler in (pause_handler, private_handler):
            self.assertIn("if (", handler)
            self.assertIn("micEnabled", handler)
            self.assertIn("stopPcmCapture({ commit: false })", handler)
            self.assertIn("markMicState(false)", handler)

    def test_interrupt_cancels_response_without_dropping_live_microphone(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()
        handler = self._script_between(
            script,
            "controls.get('interrupt')?.addEventListener",
            "controls.get('play')?.addEventListener",
        )

        self.assertIn("response.cancel", handler)
        self.assertIn("clearPlayback()", handler)
        self.assertNotIn("stopPcmCapture", handler)

    def test_call_controls_have_explicit_lifecycle_and_reset(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        end_call = self._script_between(script, "const endCall =", "handleSocketClosed")

        self.assertIn('data-transport-action="start-call"', template)
        self.assertIn('data-transport-action="end-call"', template)
        self.assertIn("if (!enabled || callActive) return", script)
        self.assertIn("socket.close(1000, 'call ended')", end_call)
        self.assertIn("stopPcmCapture({ commit: false })", end_call)
        self.assertIn("stopTimer()", end_call)
        self.assertIn("clearPlayback()", end_call)
        self.assertIn("callActive = false", end_call)

    def test_transport_uses_stable_icon_controls_and_volume_slider(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        css = Path("atlas_voice/web/static/app.css").read_text()
        block = self._last_css_block(css, ".voice-transport")

        for action in ("mic", "pause", "private", "interrupt", "play"):
            self.assertRegex(
                template,
                rf'data-transport-action="{action}"[^>]+aria-label="[^"]+"[^>]+title="[^"]+"',
            )
        self.assertIn('type="range"', template)
        self.assertIn("grid-template-columns", block)
        self.assertIn("min-height: 72px", block)

    def test_voice_layout_has_stable_desktop_and_mobile_dimensions(self) -> None:
        css = Path("atlas_voice/web/static/app.css").read_text()
        workbench = self._last_css_block(css, ".voice-workbench")
        bubble = self._last_css_block(css, ".voice-bubble")

        self.assertIn("height: 100dvh", workbench)
        self.assertIn("grid-template-rows: 48px 44px minmax(0, 1fr) 72px", workbench)
        self.assertIn("aspect-ratio: 1", bubble)
        self.assertIn("@media (max-width: 760px)", css)
        self.assertIn('.voice-workbench[data-active-view="transcript"] .voice-main', css)
        self.assertIn("grid-template-columns: 12px minmax(0, 1fr)", css)

    def test_voice_states_and_transcript_are_accessible(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text()

        self.assertIn('aria-live="polite"', template)
        self.assertIn('aria-label="Realtime call"', template)
        self.assertIn('aria-label="Conversation transcript"', template)
        self.assertIn('role="tablist"', template)
        self.assertIn('role="tab"', template)
        self.assertIn('aria-label="Assistant volume"', template)

    def test_settings_keep_diagnostics_out_of_primary_voice_surface(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        content, settings = template.split("{% block settings_dialog_extra %}", 1)

        self.assertNotIn("Voice Playground", content)
        self.assertNotIn("Privacy Events", content)
        self.assertIn("Voice Playground", settings)
        self.assertIn("Privacy and sessions", settings)
        self.assertIn("Memory and coaching", settings)
        self.assertIn("data-tts-playground", settings)


if __name__ == "__main__":
    unittest.main()
