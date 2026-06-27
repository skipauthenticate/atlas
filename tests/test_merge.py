import unittest

from atlas_voice.merge import merge_transcript_with_diarization, speaker_for_interval


class MergeTests(unittest.TestCase):
    def test_speaker_for_interval_uses_largest_overlap(self) -> None:
        diarization = [
            {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
            {"start": 1.2, "end": 3.0, "speaker": "SPEAKER_01"},
        ]

        self.assertEqual(speaker_for_interval(1.0, 2.0, diarization), "SPEAKER_01")

    def test_merge_splits_words_on_speaker_change(self) -> None:
        transcript = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello there",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 1.0},
                        {"word": "there", "start": 1.0, "end": 2.0},
                    ],
                }
            ]
        }
        diarization = [
            {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
            {"start": 1.2, "end": 3.0, "speaker": "SPEAKER_01"},
        ]

        merged = merge_transcript_with_diarization(transcript, diarization)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["speaker"], "SPEAKER_00")
        self.assertEqual(merged[0]["text"], "hello")
        self.assertEqual(merged[1]["speaker"], "SPEAKER_01")
        self.assertEqual(merged[1]["text"], "there")


if __name__ == "__main__":
    unittest.main()
