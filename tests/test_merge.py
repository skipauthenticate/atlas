import unittest
from unittest import mock

from atlas_voice.merge import (
    merge_transcript_with_diarization,
    overlap_seconds,
    speaker_for_interval,
)


class MergeTests(unittest.TestCase):
    def test_speaker_for_interval_uses_largest_overlap(self) -> None:
        diarization = [
            {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
            {"start": 1.2, "end": 3.0, "speaker": "SPEAKER_01"},
        ]

        self.assertEqual(speaker_for_interval(1.0, 2.0, diarization), "SPEAKER_01")

    def test_speaker_ties_preserve_original_diarization_order(self) -> None:
        diarization = [
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_FIRST"},
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_SECOND"},
        ]

        self.assertEqual(
            speaker_for_interval(0.5, 1.5, diarization),
            "SPEAKER_FIRST",
        )

    def test_speaker_for_interval_returns_unknown_without_overlap(self) -> None:
        diarization = [{"start": 2.0, "end": 3.0, "speaker": "SPEAKER_00"}]
        self.assertEqual(speaker_for_interval(0.0, 1.0, diarization), "SPEAKER_UNKNOWN")

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

    def test_merge_preserves_word_confidence_fields(self) -> None:
        transcript = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.4, "score": 0.91},
                        {
                            "word": "world",
                            "start": 0.5,
                            "end": 1.0,
                            "probability": 0.82,
                            "confidence": 0.80,
                        },
                    ],
                }
            ]
        }
        diarization = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

        merged = merge_transcript_with_diarization(transcript, diarization)

        self.assertEqual(merged[0]["words"][0]["score"], 0.91)
        self.assertEqual(merged[0]["words"][1]["probability"], 0.82)
        self.assertEqual(merged[0]["words"][1]["confidence"], 0.80)

    def test_merge_preserves_segment_confidence_without_words(self) -> None:
        transcript = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "words": [],
                    "avg_logprob": -0.25,
                    "no_speech_prob": 0.05,
                }
            ]
        }
        diarization = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

        merged = merge_transcript_with_diarization(transcript, diarization)

        self.assertEqual(merged[0]["avg_logprob"], -0.25)
        self.assertEqual(merged[0]["no_speech_prob"], 0.05)

    def test_merge_checks_only_nearby_diarization_turns(self) -> None:
        turn_count = 2000
        diarization = [
            {
                "start": float(index),
                "end": float(index + 1),
                "speaker": f"SPEAKER_{index % 2:02d}",
            }
            for index in range(turn_count)
        ]
        transcript = {
            "segments": [
                {
                    "start": 0.0,
                    "end": float(turn_count),
                    "text": "word " * turn_count,
                    "words": [
                        {
                            "word": "word",
                            "start": index + 0.1,
                            "end": index + 0.2,
                        }
                        for index in range(turn_count)
                    ],
                }
            ]
        }

        with mock.patch("atlas_voice.merge.overlap_seconds", wraps=overlap_seconds) as overlap:
            merged = merge_transcript_with_diarization(transcript, diarization)

        self.assertEqual(len(merged), turn_count)
        self.assertLess(overlap.call_count, turn_count * 3)


if __name__ == "__main__":
    unittest.main()
