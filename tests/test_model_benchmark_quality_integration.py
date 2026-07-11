from __future__ import annotations

from pathlib import Path
import unittest

from atlas_voice.model_benchmark import load_corpus, score_answer


class ModelBenchmarkQualityIntegrationTests(unittest.TestCase):
    def test_loads_and_scores_the_versioned_sixty_case_jsonl(self) -> None:
        corpus = load_corpus(Path("benchmarks/voice_model_quality_v1.jsonl"))

        self.assertEqual(corpus.corpus_version, "voice_model_quality_v1")
        self.assertEqual(len(corpus.cases), 60)
        local = next(case for case in corpus.cases if case.prompt_id == "local_detail_001")
        passing = score_answer(
            "Couriers should use 7314, and 2048 is disabled.",
            local,
            corpus.abstention_phrases,
        )
        forbidden = score_answer(
            "Couriers should use 2048.",
            local,
            corpus.abstention_phrases,
        )
        self.assertTrue(passing["passed"])
        self.assertEqual(passing["required_fact_recall"], 1.0)
        self.assertFalse(forbidden["passed"])
        self.assertEqual(forbidden["forbidden_claim_hits"], ["forbidden-01"])

        unanswerable = next(case for case in corpus.cases if case.prompt_id == "abstention_010")
        abstained = score_answer(
            "No winner is available yet.",
            unanswerable,
            corpus.abstention_phrases,
        )
        self.assertTrue(abstained["abstention_detected"])
        self.assertTrue(abstained["passed"])


if __name__ == "__main__":
    unittest.main()
