from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess
import sys
import unittest

from atlas_voice.quality_corpus import decode_quality_jsonl


class QualityCorpusAdapterTests(unittest.TestCase):
    def test_reviewed_context_thresholds_bands_and_leakage_validate(self) -> None:
        path = Path("benchmarks/voice_model_quality_v1.jsonl").resolve()
        builder = path.with_name("build_voice_model_quality_context.py")

        result = subprocess.run(
            [sys.executable, str(builder), "--corpus", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("| `multi_window_synthesis` | 15 | 21/36/51 |", result.stdout)
        self.assertIn("| short | 5 | 4,000-6,000 |", result.stdout)
        self.assertIn("| medium | 5 | 8,000-10,000 |", result.stdout)
        self.assertIn("| long | 5 | 12,000-16,000 |", result.stdout)

    def test_versioned_quality_corpus_maps_all_grading_and_evidence_fields(self) -> None:
        path = Path("benchmarks/voice_model_quality_v1.jsonl").resolve()

        payload = decode_quality_jsonl(path.read_bytes(), path)

        self.assertEqual(payload["corpus_version"], "voice_model_quality_v1")
        self.assertEqual(len(payload["cases"]), 60)
        self.assertEqual(
            Counter(case["category"] for case in payload["cases"]),
            {
                "local_detail": 15,
                "multi_window_synthesis": 15,
                "temporal_order": 10,
                "speaker_decision_action": 10,
                "contradiction_unanswerable": 10,
            },
        )
        local = payload["cases"][0]
        self.assertIn("[local_detail_001_t02] Window W1;", "\n".join(local["evidence"]))
        self.assertEqual(local["required_facts"][0]["id"], "courier_code")
        self.assertEqual(
            local["required_facts"][0]["evidence_turn_ids"],
            ["local_detail_001_t02"],
        )
        self.assertEqual(
            local["attribution_targets"][0],
            {
                "fact_id": "courier_code",
                "value": "Omar",
            },
        )
        self.assertNotIn(local["gold_rationale"], "\n".join(local["evidence"]))

        temporal = next(case for case in payload["cases"] if case["id"] == "temporal_order_001")
        self.assertEqual(temporal["ordered_fact_ids"], ["first", "second", "third"])
        self.assertEqual(len(temporal["timestamp_targets"]), 3)

        abstention = next(case for case in payload["cases"] if case["id"] == "abstention_010")
        self.assertTrue(abstention["should_abstain"])
        self.assertEqual(abstention["required_facts"][0]["id"], "answer_status")
        self.assertGreater(len(abstention["forbidden_claims"]), 0)


if __name__ == "__main__":
    unittest.main()
