import unittest

from atlas_voice.summarizer import build_summary_prompt, chunk_transcript


class SummarizerTests(unittest.TestCase):
    def test_chunk_transcript_respects_character_budget(self) -> None:
        segments = [
            {"start": 0, "end": 1, "speaker": "SPEAKER_00", "text": "alpha beta gamma"},
            {"start": 1, "end": 2, "speaker": "SPEAKER_01", "text": "delta epsilon zeta"},
        ]

        chunks = chunk_transcript(segments, max_chars=55)

        self.assertEqual(len(chunks), 2)
        self.assertIn("SPEAKER_00", chunks[0])
        self.assertIn("SPEAKER_01", chunks[1])

    def test_prompt_includes_grounding_instruction_and_chunk_position(self) -> None:
        prompt = build_summary_prompt("SPEAKER_00: hello", 2, 3)

        self.assertIn("Use only the transcript content", prompt)
        self.assertIn("Chunk 2 of 3", prompt)
        self.assertIn("Action Items", prompt)


if __name__ == "__main__":
    unittest.main()
