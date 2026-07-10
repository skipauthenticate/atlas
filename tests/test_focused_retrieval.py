from __future__ import annotations

import unittest

from atlas_voice.retrieval import build_focused_recording_context, estimate_tokens


class FocusedRecordingRetrievalTests(unittest.TestCase):
    def test_short_transcript_is_injected_in_full(self) -> None:
        segments = [
            self._segment(0, "Alice", "We should launch on Tuesday."),
            self._segment(1, "Bob", "I will prepare the release checklist."),
            self._segment(2, "Alice", "The final review is Monday afternoon."),
        ]

        result = build_focused_recording_context(
            {"id": "rec-1", "title": "Launch planning"},
            {"text": "A short launch discussion."},
            segments,
            "When is the review?",
            max_chars=1800,
            max_tokens=500,
        )

        self.assertTrue(result.used_full_transcript)
        self.assertEqual(result.mode, "full_transcript")
        self.assertEqual(len(result.evidence), 1)
        for segment in segments:
            self.assertIn(segment["text"], result.context)
        self.assertIn("Complete timestamped transcript [00:00-00:08", result.context)

    def test_overview_covers_beginning_middle_and_end(self) -> None:
        segments = [
            self._segment(
                index,
                "Alice" if index % 2 == 0 else "Bob",
                self._padded(
                    {
                        0: "BEGIN_FACT the team opened with customer retention.",
                        6: "MIDDLE_FACT they changed the pricing experiment.",
                        11: "END_FACT Bob accepted the follow-up action.",
                    }.get(index, f"Discussion detail number {index}."),
                    210,
                ),
            )
            for index in range(12)
        ]

        result = build_focused_recording_context(
            {"id": "rec-2", "title": "Long strategy conversation"},
            {"text": "x" * 5000, "chunks": [{"chunk_index": 1, "text": "y" * 3000}]},
            segments,
            "Give me the gist of this whole conversation",
            max_chars=2800,
            max_tokens=900,
            target_window_chars=360,
        )

        self.assertFalse(result.used_full_transcript)
        self.assertEqual(result.mode, "overview")
        self.assertIn("BEGIN_FACT", result.context)
        self.assertIn("MIDDLE_FACT", result.context)
        self.assertIn("END_FACT", result.context)
        self.assertEqual(
            [item.reason for item in result.evidence],
            ["timeline_start", "timeline_middle", "timeline_end"],
        )

    def test_long_single_speaker_recording_keeps_raw_evidence(self) -> None:
        segments = [
            self._segment(
                index,
                "Narrator",
                self._padded(f"SINGLE_SPEAKER_{index:02d} monologue detail.", 150),
            )
            for index in range(48)
        ]

        result = build_focused_recording_context(
            {"id": "rec-single", "title": "Long single-speaker monologue"},
            None,
            segments,
            "Give me the gist of this whole recording",
            max_chars=3000,
            max_tokens=800,
            target_window_chars=500,
        )

        self.assertFalse(result.used_full_transcript)
        self.assertEqual(
            [item.reason for item in result.evidence],
            ["timeline_start", "timeline_middle", "timeline_end"],
        )
        for marker in ("SINGLE_SPEAKER_00", "SINGLE_SPEAKER_25", "SINGLE_SPEAKER_47"):
            self.assertIn(marker, result.context)
        self.assertLessEqual(len(result.context), 3000)

    def test_speaker_match_returns_adjacent_complete_turns(self) -> None:
        segments = [
            self._segment(0, "Drew", self._padded("Opening context.", 230)),
            self._segment(1, "Bob", self._padded("The supplier raised a constraint.", 230)),
            self._segment(2, "Alice", self._padded("Titanium is the safer material.", 230)),
            self._segment(3, "Carol", self._padded("I agree with that recommendation.", 230)),
            self._segment(4, "Drew", self._padded("We moved to another subject.", 230)),
            self._segment(5, "Bob", self._padded("Closing context.", 230)),
        ]

        result = build_focused_recording_context(
            {"id": "rec-3", "title": "Materials review"},
            None,
            segments,
            "What did Alice say about titanium?",
            max_chars=1300,
            max_tokens=400,
            target_window_chars=500,
        )

        self.assertEqual(result.mode, "focused")
        self.assertTrue(result.evidence)
        window = result.evidence[0]
        self.assertIn("Bob: The supplier raised", window.transcript)
        self.assertIn("Alice: Titanium is the safer", window.transcript)
        self.assertIn("Carol: I agree", window.transcript)
        self.assertIn("Alice", window.speakers)

    def test_large_summary_and_chunk_summaries_cannot_starve_transcript(self) -> None:
        segments = [
            self._segment(index, "A" if index % 2 == 0 else "B", self._padded(text, 180))
            for index, text in enumerate(
                [
                    "Unrelated opening.",
                    "More background.",
                    "ORCHID_FACT the renewal date is September ninth.",
                    "That date was confirmed.",
                    "Unrelated ending.",
                    "Final pleasantries.",
                ]
            )
        ]
        summary = {
            "text": "S" * 9000,
            "chunks": [
                {"chunk_index": index, "text": "C" * 4000}
                for index in range(1, 8)
            ],
        }

        result = build_focused_recording_context(
            {"id": "rec-4", "title": "Renewal call"},
            summary,
            segments,
            "When is the orchid renewal date?",
            max_chars=1150,
            max_tokens=350,
            target_window_chars=420,
        )

        self.assertTrue(result.evidence)
        self.assertIn("ORCHID_FACT", result.context)
        self.assertNotIn("S" * 100, result.context)
        self.assertNotIn("C" * 100, result.context)
        self.assertLessEqual(len(result.context), 1150)

    def test_packing_keeps_complete_timestamped_blocks_inside_both_budgets(self) -> None:
        segments = [
            self._segment(
                index,
                f"Speaker {index}",
                self._padded(f"TOPIC_{index} exact complete ending {index}.", 210),
            )
            for index in range(10)
        ]

        result = build_focused_recording_context(
            {"id": "rec-5", "title": "Budget test"},
            {"text": "Derived overview sentence. " * 20},
            segments,
            "Tell me about TOPIC_5",
            max_chars=1050,
            max_tokens=270,
            target_window_chars=500,
        )

        self.assertTrue(result.evidence)
        self.assertLessEqual(len(result.context), 1050)
        self.assertLessEqual(result.estimated_tokens, 270)
        self.assertEqual(result.estimated_tokens, estimate_tokens(result.context))
        for evidence in result.evidence:
            self.assertIn(evidence.transcript, result.context)
            self.assertIn("exact complete ending", evidence.transcript)
        self.assertRegex(result.context, r"Transcript evidence E1 \[00:\d{2}-00:\d{2}; segments")

    def test_follow_up_expands_only_from_recent_user_turns(self) -> None:
        segments = [
            self._segment(0, "Bob", self._padded("Unrelated financial context.", 230)),
            self._segment(
                1,
                "Alice",
                self._padded("The launch schedule slipped because testing failed.", 230),
            ),
            self._segment(2, "Bob", self._padded("We can recover two days.", 230)),
            self._segment(3, "Carol", self._padded("Different hiring discussion.", 230)),
            self._segment(4, "Drew", self._padded("More unrelated details.", 230)),
        ]

        result = build_focused_recording_context(
            {"id": "rec-6", "title": "Launch retrospective"},
            {"text": "A derived note about Zephyr that must not drive raw retrieval."},
            segments,
            "Why did she say that?",
            recent_user_turns=("What did Alice say about the launch schedule?",),
            max_chars=1200,
            max_tokens=350,
            target_window_chars=480,
        )

        self.assertEqual(result.mode, "focused_followup")
        self.assertIn("alice", result.resolved_query)
        self.assertIn("launch", result.resolved_query)
        self.assertIn("schedule", result.resolved_query)
        self.assertIn("testing failed", result.context)
        self.assertNotIn("zephyr", result.resolved_query)

    @staticmethod
    def _segment(index: int, speaker: str, text: str) -> dict[str, object]:
        return {
            "idx": index,
            "start": index * 3.0,
            "end": index * 3.0 + 2.0,
            "speaker": speaker,
            "text": text,
        }

    @staticmethod
    def _padded(text: str, minimum_length: int) -> str:
        padding = " supporting context" * 30
        return (text + padding)[:minimum_length].rstrip() + "."


if __name__ == "__main__":
    unittest.main()
