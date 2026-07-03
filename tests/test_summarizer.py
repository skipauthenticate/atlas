import unittest

from atlas_voice.summarizer import (
    _SUMMARY_TEMPLATES,
    _strip_summary_markup,
    build_summary_prompt,
    chunk_transcript,
    detect_template,
    get_template,
    list_templates,
    summary_to_blocks,
    summary_to_sections,
)


class TemplateRegistryTests(unittest.TestCase):
    """Test that all built-in templates are registered and accessible."""

    def test_all_built_in_templates_registered(self) -> None:
        """Every _build_* function should have registered a template."""
        ids = list(_SUMMARY_TEMPLATES.keys())
        self.assertIn("meeting", ids)
        self.assertIn("personal", ids)
        self.assertIn("catchup", ids)
        self.assertIn("call", ids)
        self.assertIn("interview", ids)
        self.assertIn("debrief", ids)
        self.assertIn("sales", ids)
        self.assertIn("education", ids)

    def test_get_template_returns_none_for_unknown(self) -> None:
        self.assertIsNone(get_template("nonexistent"))

    def test_get_template_returns_template_for_known(self) -> None:
        tpl = get_template("meeting")
        self.assertIsNotNone(tpl)
        self.assertEqual(tpl.id, "meeting")
        self.assertTrue(tpl.default)

    def test_list_templates_returns_all(self) -> None:
        templates = list_templates()
        self.assertEqual(len(templates), 8)

    def test_template_to_dict(self) -> None:
        tpl = get_template("personal")
        d = tpl.to_dict()
        self.assertEqual(d["id"], "personal")
        self.assertEqual(d["name"], "Personal Reflection")
        self.assertEqual(d["default"], False)
        self.assertIsInstance(d["section_instructions"], dict)
        self.assertIsInstance(d["keywords"], list)


class AutoDetectionTests(unittest.TestCase):
    """Test template auto-detection by keyword scoring."""

    def test_meeting_transcript_detects_meeting(self) -> None:
        transcript = (
            "We had a team meeting today to discuss the Q3 roadmap. "
            "John proposed a new feature and we decided to move forward. "
            "Action item: Sarah will write the spec by Friday."
        )
        tpl = detect_template(transcript)
        self.assertEqual(tpl.id, "meeting")

    def test_personal_transcript_detects_personal(self) -> None:
        transcript = (
            "I've been feeling really grateful today. "
            "I learned a lot about myself through meditation. "
            "I'm thankful for my family and friends."
        )
        tpl = detect_template(transcript)
        self.assertEqual(tpl.id, "personal")

    def test_interview_transcript_detects_interview(self) -> None:
        transcript = (
            "Welcome to the podcast. Today we have our guest "
            "who will discuss their latest book. Thanks for having me. "
            "Listeners, what do you think?"
        )
        tpl = detect_template(transcript)
        self.assertEqual(tpl.id, "interview")

    def test_catchup_transcript_detects_catchup(self) -> None:
        transcript = (
            "Hey, how have you been? It's been so long since we last "
            "met. Let's plan to catch up soon. Remember when we went "
            "to the beach last summer?"
        )
        tpl = detect_template(transcript)
        self.assertEqual(tpl.id, "catchup")

    def test_debrief_transcript_detects_debrief(self) -> None:
        transcript = (
            "Let's do a quick debrief on the presentation. "
            "What went well was the demo. What could improve "
            "is the slides. Lessons learned for next time."
        )
        tpl = detect_template(transcript)
        self.assertEqual(tpl.id, "debrief")

    def test_sales_transcript_detects_sales(self) -> None:
        transcript = (
            "We met with the prospect about their pricing concerns. "
            "They need a proposal for the enterprise tier. "
            "Next steps: send the contract by Monday."
        )
        tpl = detect_template(transcript)
        self.assertEqual(tpl.id, "sales")

    def test_education_transcript_detects_education(self) -> None:
        transcript = (
            "Today's lesson covers the key concepts of linear algebra. "
            "The formula for matrix multiplication is... "
            "For homework, solve exercises 5 through 12."
        )
        tpl = detect_template(transcript)
        self.assertEqual(tpl.id, "education")

    def test_empty_transcript_falls_back_to_default(self) -> None:
        tpl = detect_template("")
        self.assertEqual(tpl.id, "meeting")

    def test_neutral_transcript_falls_back_to_meeting(self) -> None:
        transcript = "The sky is blue. The grass is green. Today is Tuesday."
        tpl = detect_template(transcript)
        # Should fall back to meeting (the default)
        self.assertEqual(tpl.id, "meeting")


class BuildSummaryPromptTests(unittest.TestCase):
    """Test prompt generation with various templates."""

    def test_prompt_includes_template_name(self) -> None:
        tpl = get_template("personal")
        prompt = build_summary_prompt("Some text", template=tpl)
        self.assertIn("Personal Reflection", prompt)

    def test_prompt_includes_template_description(self) -> None:
        tpl = get_template("interview")
        prompt = build_summary_prompt("Some text", template=tpl)
        self.assertIn("interview", prompt.lower())

    def test_prompt_includes_section_headers(self) -> None:
        tpl = get_template("sales")
        prompt = build_summary_prompt("Some text", template=tpl)
        self.assertIn("## Call Context", prompt)
        self.assertIn("## Client Needs", prompt)
        self.assertIn("## Objections", prompt)
        self.assertIn("## Pipeline Notes", prompt)

    def test_default_template_when_none(self) -> None:
        prompt = build_summary_prompt("text", template=None)
        self.assertIn("Meeting Minutes", prompt)
        self.assertIn("## Overview", prompt)

    def test_prompt_includes_chunk_position(self) -> None:
        prompt = build_summary_prompt("text", chunk_index=2, chunk_count=5)
        self.assertIn("Chunk 2 of 5", prompt)


class ChunkTranscriptTests(unittest.TestCase):
    def test_chunk_transcript_respects_character_budget(self) -> None:
        segments = [
            {"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "alpha beta gamma"},
            {"start": 1, "end": 2, "speaker": "SPEAKER_01", "text": "delta epsilon zeta"},
        ]

        chunks = chunk_transcript(segments, max_chars=55)

        self.assertEqual(len(chunks), 2)
        self.assertIn("SPEAKER_00", chunks[0])
        self.assertIn("SPEAKER_01", chunks[1])

    def test_single_chunk_for_short_transcript(self) -> None:
        segments = [
            {"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "hello world"},
        ]
        chunks = chunk_transcript(segments, max_chars=6000)
        self.assertEqual(len(chunks), 1)


class SummaryParsingTests(unittest.TestCase):
    def test_summary_to_blocks_formats_markdown_sections_and_bullets(self) -> None:
        blocks = summary_to_blocks(
            "**Overview**\n"
            "A short recap.\n\n"
            "**Key Points**\n"
            "* **Audio Issues:** volume changed during setup.\n"
            "  * Nested detail.\n\n"
            "## Action Items\n"
            "- None"
        )

        self.assertEqual(blocks[0], {"type": "heading", "text": "Overview"})
        self.assertEqual(blocks[1], {"type": "paragraph", "text": "A short recap."})
        self.assertEqual(blocks[2], {"type": "heading", "text": "Key Points"})
        self.assertEqual(blocks[3]["type"], "list")
        self.assertEqual(blocks[3]["items"][0]["label"], "Audio Issues")
        self.assertEqual(blocks[3]["items"][1]["depth"], 1)
        self.assertEqual(blocks[5]["items"][0]["text"], "None")
        self.assertTrue(blocks[5]["items"][0]["is_none"])

    def test_summary_to_blocks_handles_arbitrary_sections(self) -> None:
        """Sections from non-meeting templates should still be parsed."""
        blocks = summary_to_blocks(
            "## Mood Overview\n"
            "Feeling good today.\n\n"
            "## Gratitude Notes\n"
            "- **Morning:** sunshine\n"
            "- **Evening:** coffee\n\n"
            "## Looking Ahead\n"
            "- Visit mom next weekend"
        )

        headings = [b["text"] for b in blocks if b["type"] == "heading"]
        self.assertIn("Mood Overview", headings)
        self.assertIn("Gratitude Notes", headings)
        self.assertIn("Looking Ahead", headings)

    def test_summary_to_sections_merges_chunk_summaries_by_heading(self) -> None:
        sections = summary_to_sections(
            "Chunk 1 summary:\n"
            "## Overview\n"
            "First overview.\n\n"
            "## Key Points\n"
            "- **Topic:** first point.\n\n"
            "Chunk 2 summary:\n"
            "## Overview\n"
            "Second overview.\n\n"
            "## Key Points\n"
            "- **Topic:** first point.\n"
            "- Second point.\n"
            "## Action Items\n"
            "- None"
        )

        # Sections with only "- None" items are excluded (empty sections)
        self.assertEqual(
            [section["title"] for section in sections],
            ["Overview", "Key Points"],
        )
        self.assertEqual(sections[0]["paragraphs"], ["First overview.", "Second overview."])
        self.assertEqual(len(sections[1]["items"]), 2)
        self.assertEqual(sections[1]["items"][0]["label"], "Topic")

    def test_summary_to_sections_handles_personal_template_sections(self) -> None:
        sections = summary_to_sections(
            "## Mood Overview\n"
            "Feeling optimistic.\n\n"
            "## Lessons Learned\n"
            "- Be more patient.\n\n"
            "## Gratitude Notes\n"
            "- Family\n"
            "- Good health\n\n"
            "## Looking Ahead\n"
            "- None"
        )

        titles = [s["title"] for s in sections]
        self.assertIn("Mood Overview", titles)
        self.assertIn("Lessons Learned", titles)
        self.assertIn("Gratitude Notes", titles)
        # "Looking Ahead" has "None" so should be excluded
        self.assertNotIn("Looking Ahead", titles)

    def test_summary_to_sections_handles_interview_template_sections(self) -> None:
        sections = summary_to_sections(
            "## Notable Quotes\n"
            "- **Guest:** \"The future is now.\"\n\n"
            "## Takeaways\n"
            "- Innovation matters\n"
            "- Keep learning"
        )

        titles = [s["title"] for s in sections]
        self.assertIn("Notable Quotes", titles)
        self.assertIn("Takeaways", titles)


class StripMarkupTests(unittest.TestCase):
    def test_strips_bold(self) -> None:
        self.assertEqual(_strip_summary_markup("**hello**"), "hello")

    def test_strips_inline_code(self) -> None:
        self.assertEqual(_strip_summary_markup("`world`"), "world")

    def test_passes_none(self) -> None:
        self.assertEqual(_strip_summary_markup(None), "")


if __name__ == "__main__":
    unittest.main()
