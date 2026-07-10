import unittest

from atlas_voice.titles import clean_recording_title, fallback_recording_title


class RecordingTitleTests(unittest.TestCase):
    def test_clean_title_removes_model_wrapping_and_limits_words(self) -> None:
        self.assertEqual(
            clean_recording_title(
                '## Title: "Atlas Voice Retrieval Planning and Performance Review Meeting Notes"'
            ),
            "Atlas Voice Retrieval Planning and Performance Review Meeting Notes",
        )

    def test_fallback_uses_snapshot_instead_of_a_filename(self) -> None:
        title = fallback_recording_title(
            "## Snapshot\n"
            "The meeting focused on reducing voice retrieval latency for Jetson users.\n\n"
            "## Action Items\n"
            "- Benchmark local search."
        )

        self.assertEqual(title, "reducing voice retrieval latency for Jetson users")


if __name__ == "__main__":
    unittest.main()
