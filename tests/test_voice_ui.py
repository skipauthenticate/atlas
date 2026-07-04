import re
from pathlib import Path
import unittest


class VoiceUiStaticTests(unittest.TestCase):
    @staticmethod
    def _css_block(css: str, selector: str) -> str:
        match = re.search(rf"{re.escape(selector)}\s*{{(?P<body>.*?)}}", css, flags=re.S)
        if match is None:
            raise AssertionError(f"Missing CSS selector: {selector}")
        return match.group("body")

    @staticmethod
    def _media_block(css: str, query: str) -> str:
        marker = f"@media {query} {{"
        start = css.find(marker)
        if start == -1:
            raise AssertionError(f"Missing media query: {query}")
        index = start + len(marker)
        depth = 1
        while index < len(css) and depth:
            char = css[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            index += 1
        if depth != 0:
            raise AssertionError(f"Unclosed media query: {query}")
        return css[start + len(marker) : index - 1]

    def test_voice_workbench_desktop_layout_has_bounded_columns(self) -> None:
        css = Path("atlas_voice/web/static/app.css").read_text()
        workbench = self._css_block(css, ".voice-workbench")

        self.assertIn(
            "grid-template-columns: minmax(120px, 150px) minmax(0, 1fr) minmax(260px, 300px)",
            workbench,
        )
        self.assertIn("max-width: 100%", workbench)

    def test_voice_workbench_tablet_and_mobile_layouts_prevent_overflow(self) -> None:
        css = Path("atlas_voice/web/static/app.css").read_text()
        tablet = self._media_block(css, "(max-width: 980px)")
        mobile = self._media_block(css, "(max-width: 640px)")

        tablet_workbench = self._css_block(tablet, ".voice-workbench")
        tablet_transport = self._css_block(tablet, ".voice-transport")
        mobile_titlebar = self._css_block(mobile, ".voice-titlebar")
        mobile_transport = self._css_block(mobile, ".voice-transport")
        mobile_transport_group = self._css_block(
            mobile,
            ".voice-transport label,\n  .voice-playback-status,\n  .voice-transport audio",
        )
        mobile_transport_range_group = self._css_block(
            mobile, '.voice-transport input[type="range"],\n  .voice-transport audio'
        )
        mobile_waveform = self._css_block(mobile, ".voice-waveform")

        self.assertIn("grid-template-columns: 1fr", tablet_workbench)
        self.assertIn("flex-wrap: wrap", tablet_transport)
        self.assertIn("display: grid", mobile_titlebar)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", mobile_transport)
        self.assertIn("grid-column: 1 / -1", mobile_transport_group)
        self.assertIn("min-width: 0", mobile_transport_group)
        self.assertIn("width: 100%", mobile_transport_range_group)
        self.assertIn("grid-template-columns: repeat(12, minmax(3px, 1fr))", mobile_waveform)

    @staticmethod
    def _script_between(script: str, start: str, end: str) -> str:
        start_index = script.find(start)
        if start_index == -1:
            raise AssertionError(f"Missing script marker: {start}")
        end_index = script.find(end, start_index)
        if end_index == -1:
            raise AssertionError(f"Missing script marker: {end}")
        return script[start_index:end_index]

    def test_pause_and_private_transport_stop_active_microphone_audio(self) -> None:
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
            self.assertIn("if (micEnabled)", handler)
            self.assertIn("stopMicStreaming({ commit: false })", handler)
            self.assertIn("togglePressed(controls.get('mic'), false)", handler)
            self.assertIn("stopTimer()", handler)

    def test_interrupt_transport_discards_active_microphone_audio(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()

        self.assertIn("input_audio_buffer.clear", script)
        self.assertIn("stopMicStreaming({ commit: false })", script)
        self.assertIn("micDiscardPending", script)

    def test_voice_playground_sections_are_not_card_framed(self) -> None:
        css = Path("atlas_voice/web/static/app.css").read_text()
        playground = self._css_block(css, ".voice-playground")

        self.assertNotIn("border:", playground)
        self.assertNotIn("background:", playground)
        self.assertNotIn("box-shadow", playground)

    def test_repeated_voice_items_keep_restrained_row_framing(self) -> None:
        css = Path("atlas_voice/web/static/app.css").read_text()
        row_block = self._css_block(css, ".voice-timeline-row")
        grouped_block = self._css_block(
            css,
            ".voice-setting-list div,\n.voice-model-list div,\n.voice-session-row",
        )

        self.assertIn("border: 1px solid var(--line)", row_block)
        self.assertIn("border-radius: 8px", row_block)
        self.assertIn("border: 1px solid var(--line)", grouped_block)
        self.assertIn("border-radius: 8px", grouped_block)


if __name__ == "__main__":
    unittest.main()
