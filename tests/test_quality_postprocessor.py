from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from atlas_voice.quality_postprocessor import (
    QualityPostprocessError,
    load_quality_corpus,
    normalize_tokens,
    postprocess_quality,
)


REAL_CORPUS = Path("benchmarks/voice_model_quality_v1.jsonl")


class QualityPostprocessorTests(unittest.TestCase):
    def test_normalization_handles_case_punctuation_numeric_grouping_and_zeroes(
        self,
    ) -> None:
        self.assertEqual(
            normalize_tokens("CAFÉ — $3,800 at 09:05"),
            ("cafe", "3800", "at", "9", "5"),
        )

    def test_synthetic_answers_score_linked_targets_and_temporal_order(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = root / "quality.jsonl"
            rows = _synthetic_rows()
            corpus_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            corpus = load_quality_corpus(corpus_path)
            benchmark = _benchmark(
                corpus,
                {
                    "detail": "By ALICE, the selected code is r-7 on July 16.",
                    "order": "Third: doors at 18:15; first: soundcheck at 17:00; "
                    "second: safety walk at 17:30.",
                    "unknown": "Not provided in the evidence.",
                },
            )

            report = postprocess_quality(benchmark, corpus)

        summary = report["model_summaries"][0]["overall"]
        # Explicit abstention passes the case but does not fabricate a hit for the
        # separate required-fact recall metric.
        self.assertEqual(summary["required_fact_recall"]["rate"], 0.8)
        self.assertEqual(summary["forbidden_claim_hit_rate"]["rate"], 0.0)
        self.assertEqual(summary["exact_pass_rate"]["rate"], 1.0)
        self.assertEqual(summary["abstention_accuracy"]["rate"], 1.0)
        self.assertEqual(summary["attribution_target_accuracy"]["rate"], 1.0)
        self.assertEqual(summary["timestamp_target_accuracy"]["rate"], 1.0)
        self.assertEqual(summary["temporal_order_accuracy"]["rate"], 0.0)
        temporal = next(item for item in report["trials"] if item["prompt_id"] == "order")
        scored = temporal["postprocessed_quality"]
        self.assertFalse(scored["temporal_order"]["passed"])
        self.assertIn(
            "temporal_order_mismatch",
            [item["code"] for item in scored["manual_review_reasons"]],
        )

    def test_attribution_and_timestamp_require_the_linked_fact(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "quality.jsonl"
            path.write_text(json.dumps(_synthetic_rows()[0]) + "\n", encoding="utf-8")
            corpus = load_quality_corpus(path)
            benchmark = _benchmark(corpus, {"detail": "Alice spoke on July 16."})

            report = postprocess_quality(benchmark, corpus)

        quality = report["trials"][0]["postprocessed_quality"]
        self.assertFalse(quality["required_facts"][0]["hit"])
        self.assertTrue(quality["attribution_targets"][0]["target_present"])
        self.assertFalse(quality["attribution_targets"][0]["passed"])
        self.assertTrue(quality["timestamp_targets"][0]["target_present"])
        self.assertFalse(quality["timestamp_targets"][0]["passed"])

    def test_prompt_matrix_mismatches_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "quality.jsonl"
            path.write_text(json.dumps(_synthetic_rows()[0]) + "\n", encoding="utf-8")
            corpus = load_quality_corpus(path)
            benchmark = _benchmark(corpus, {"detail": "R-7"})
            benchmark["trials"][0]["prompt_id"] = "unknown"

            with self.assertRaisesRegex(QualityPostprocessError, "unknown prompt_id"):
                postprocess_quality(benchmark, corpus)

    def test_real_sixty_case_corpus_scores_all_declared_targets(self) -> None:
        corpus = load_quality_corpus(REAL_CORPUS)
        answers: dict[str, str] = {}
        for case in corpus.cases:
            parts = [fact.value for fact in case.required_facts]
            parts.extend(target.value for target in case.attribution_targets)
            parts.extend(target.value for target in case.timestamp_targets)
            answers[case.prompt_id] = "; ".join(parts)
        benchmark = _benchmark(corpus, answers)

        report = postprocess_quality(benchmark, corpus)

        self.assertEqual(len(corpus.cases), 60)
        self.assertEqual(len(report["trials"]), 60)
        summary = report["model_summaries"][0]["overall"]
        self.assertEqual(summary["required_fact_recall"]["target_count"], 134)
        self.assertEqual(summary["forbidden_claim_hit_rate"]["target_count"], 155)
        self.assertEqual(summary["attribution_target_accuracy"]["target_count"], 84)
        self.assertEqual(summary["timestamp_target_accuracy"]["target_count"], 41)
        self.assertEqual(summary["temporal_order_accuracy"]["target_count"], 10)
        self.assertEqual(summary["required_fact_recall"]["rate"], 1.0)
        self.assertEqual(summary["attribution_target_accuracy"]["rate"], 1.0)
        self.assertEqual(summary["timestamp_target_accuracy"]["rate"], 1.0)
        self.assertEqual(summary["temporal_order_accuracy"]["rate"], 1.0)
        self.assertEqual(
            [item["category"] for item in report["model_summaries"][0]["by_category"]],
            [
                "local_detail",
                "multi_window_synthesis",
                "temporal_order",
                "speaker_decision_action",
                "contradiction_unanswerable",
            ],
        )


def _synthetic_rows() -> list[dict[str, object]]:
    common = {
        "schema_version": "1.0.0",
        "dataset_version": "synthetic_quality_v1",
        "evidence_packet": {"closed_world": True},
        "gold_rationale": "Synthetic test fixture.",
    }
    return [
        {
            **common,
            "id": "detail",
            "category": "local_detail",
            "question": "Which code?",
            "required_facts": [{"fact_id": "code", "value": "R-7", "aliases": ["code R7"]}],
            "forbidden_claims": ["The code is X-9."],
            "attribution_targets": [{"fact_id": "code", "speaker": "Alice"}],
            "timestamp_targets": [{"fact_id": "code", "timestamp": "2026-07-16"}],
            "abstention_required": False,
        },
        {
            **common,
            "id": "order",
            "category": "temporal_order",
            "question": "What happened in order?",
            "required_facts": [
                {"fact_id": "first", "value": "soundcheck at 17:00", "aliases": []},
                {"fact_id": "second", "value": "safety walk at 17:30", "aliases": []},
                {"fact_id": "third", "value": "doors at 18:15", "aliases": []},
            ],
            "forbidden_claims": [],
            "attribution_targets": [],
            "timestamp_targets": [],
            "abstention_required": False,
        },
        {
            **common,
            "id": "unknown",
            "category": "contradiction_unanswerable",
            "question": "Who won?",
            "required_facts": [
                {
                    "fact_id": "answer_status",
                    "value": "no winner is available",
                    "aliases": ["winner unknown"],
                }
            ],
            "forbidden_claims": ["Variant A won."],
            "attribution_targets": [],
            "timestamp_targets": [],
            "abstention_required": True,
        },
    ]


def _benchmark(corpus, answers: dict[str, str]) -> dict[str, object]:
    trials = [
        {
            "round": 1,
            "profile": "light",
            "model_id": "model-light",
            "prompt_id": case.prompt_id,
            "category": case.category,
            "excluded_from_summary": False,
            "answer": answers[case.prompt_id],
            "finish_reason": "stop",
            "error": None,
            "ttft_ms": 10.0,
        }
        for case in corpus.cases
    ]
    return {
        "schema_version": 1,
        "status": "completed",
        "protocol": {"rounds": 1},
        "inventory": {"models": [{"profile": "light", "model_id": "model-light"}]},
        "corpus": {
            "sha256": hashlib.sha256(corpus.path.read_bytes()).hexdigest(),
            "corpus_version": corpus.dataset_version,
            "case_count": len(corpus.cases),
        },
        "trials": trials,
    }


if __name__ == "__main__":
    unittest.main()
