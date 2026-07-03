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
