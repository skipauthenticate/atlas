from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from atlas_voice.blinded_quality_review import (
    BlindedReviewError,
    JUDGMENTS_TYPE,
    SCHEMA_VERSION,
    _json_bytes,
    apply_judgments,
    build_review_packet,
    load_review_corpus,
    load_runner_artifact,
    main,
    validate_judgments,
)


class BlindedQualityReviewTests(unittest.TestCase):
    def test_packet_is_deterministic_blinded_and_order_independent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = _write_corpus(root)
            corpus = load_review_corpus(corpus_path)
            first_path = _write_runner(root, corpus_path, "model-a", "baseline", correct=True)
            second_path = _write_runner(root, corpus_path, "model-b", "candidate", correct=False)
            first = load_runner_artifact(first_path, corpus)
            second = load_runner_artifact(second_path, corpus)

            packet, mapping = build_review_packet(corpus, [first, second], seed="fixed-seed")
            reversed_packet, reversed_mapping = build_review_packet(
                corpus, [second, first], seed="fixed-seed"
            )

            self.assertEqual(packet, reversed_packet)
            self.assertEqual(mapping, reversed_mapping)
            public = json.dumps(packet, sort_keys=True)
            self.assertNotIn("model-a", public)
            self.assertNotIn("model-b", public)
            self.assertNotIn("baseline", public)
            self.assertNotIn("candidate", public)
            self.assertEqual(packet["inputs"]["model_count"], 2)
            self.assertEqual(len(packet["review_items"]), 2)
            self.assertEqual(
                {answer["model_label"] for answer in packet["review_items"][0]["answers"]},
                {"Model A", "Model B"},
            )
            self.assertEqual(mapping["packet"]["sha256"], _sha256_bytes(_json_bytes(packet)))

    def test_runner_validation_rejects_bad_corpus_hash_and_route(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = _write_corpus(root)
            corpus = load_review_corpus(corpus_path)
            runner_path = _write_runner(root, corpus_path, "model-a", "baseline", correct=True)
            payload = json.loads(runner_path.read_text(encoding="utf-8"))
            payload["corpus"]["sha256"] = "0" * 64
            runner_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(BlindedReviewError, "corpus sha256"):
                load_runner_artifact(runner_path, corpus)

            runner_path = _write_runner(
                root, corpus_path, "model-a", "baseline", correct=True, suffix="-route"
            )
            payload = json.loads(runner_path.read_text(encoding="utf-8"))
            payload["trials"][0]["route_verified"] = False
            runner_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(BlindedReviewError, "did not verify its route"):
                load_runner_artifact(runner_path, corpus)

    def test_runner_allows_explicit_unverified_baseline_artifact_limitation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = _write_corpus(root)
            corpus = load_review_corpus(corpus_path)
            runner_path = _write_runner(root, corpus_path, "model-a", "baseline", correct=True)
            payload = json.loads(runner_path.read_text(encoding="utf-8"))
            payload["inventory"]["models"][0]["sha256"] = "not-rehashed-quality-only"
            payload["artifact_verification"][0] = {
                "profile": "baseline",
                "model_id": "model-a",
                "path": payload["inventory"]["models"][0]["path"],
                "verified": False,
                "limitation": "Artifact was not rehashed in this quality-only baseline.",
            }
            runner_path.write_text(json.dumps(payload), encoding="utf-8")

            artifact = load_runner_artifact(runner_path, corpus)

            self.assertFalse(artifact.models[0]["artifact_verification"]["verified"])
            payload["artifact_verification"][0].pop("limitation")
            runner_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(BlindedReviewError, "declare a limitation"):
                load_runner_artifact(runner_path, corpus)

    def test_apply_uses_reviewer_semantics_and_computes_deblinded_aggregates(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = _write_corpus(root)
            corpus = load_review_corpus(corpus_path)
            first_path = _write_runner(root, corpus_path, "model-a", "baseline", correct=True)
            second_path = _write_runner(root, corpus_path, "model-b", "candidate", correct=False)
            first = load_runner_artifact(first_path, corpus)
            second = load_runner_artifact(second_path, corpus)
            packet, mapping = build_review_packet(corpus, [first, second], seed="review-seed")
            packet_bytes = _json_bytes(packet)
            packet_sha = _sha256_bytes(packet_bytes)
            judgments = _make_judgments(packet, packet_sha)

            validated = validate_judgments(packet, mapping, judgments, packet_sha256=packet_sha)
            self.assertEqual(validated["review_item_count"], 2)
            self.assertEqual(validated["answer_judgment_count"], 4)

            result = apply_judgments(
                packet,
                mapping,
                judgments,
                packet_path=root / "packet.json",
                packet_sha256=packet_sha,
                mapping_path=root / "mapping.json",
                mapping_sha256="a" * 64,
                judgments_path=root / "judgments.json",
                judgments_sha256="b" * 64,
            )
            summaries = {row["model_id"]: row for row in result["model_summaries"]}
            correct = summaries["model-a"]["overall"]
            wrong = summaries["model-b"]["overall"]
            self.assertEqual(correct["required_fact_semantic_recall"]["rate"], 1.0)
            self.assertEqual(correct["abstention_correct_rate"]["rate"], 1.0)
            self.assertEqual(correct["preference"]["credit_rate"], 1.0)
            self.assertEqual(wrong["required_fact_semantic_recall"]["rate"], 0.0)
            self.assertEqual(wrong["abstention_correct_rate"]["rate"], 0.0)
            self.assertEqual(wrong["forbidden_claim_answer_rate"]["rate"], 1.0)
            self.assertEqual(wrong["unsupported_claim_answer_rate"]["rate"], 1.0)
            self.assertIn(
                "runner lexical accuracy/self-grade fields are not present",
                result["methodology"]["score_source"],
            )

    def test_judgment_validation_rejects_missing_fact_and_tampered_mapping(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = _write_corpus(root)
            corpus = load_review_corpus(corpus_path)
            artifacts = [
                load_runner_artifact(
                    _write_runner(root, corpus_path, "model-a", "baseline", correct=True),
                    corpus,
                ),
                load_runner_artifact(
                    _write_runner(root, corpus_path, "model-b", "candidate", correct=False),
                    corpus,
                ),
            ]
            packet, mapping = build_review_packet(corpus, artifacts, seed="seed")
            packet_sha = _sha256_bytes(_json_bytes(packet))
            judgments = _make_judgments(packet, packet_sha)
            judgments["reviews"][0]["answer_judgments"][0]["required_fact_judgments"] = []
            with self.assertRaisesRegex(BlindedReviewError, "target matrix mismatch"):
                validate_judgments(packet, mapping, judgments, packet_sha256=packet_sha)

            judgments = _make_judgments(packet, packet_sha)
            mapping["private_payload"]["models"][0]["model_id"] = "tampered"
            with self.assertRaisesRegex(BlindedReviewError, "commitment"):
                validate_judgments(packet, mapping, judgments, packet_sha256=packet_sha)

    def test_cli_build_protects_outputs_and_private_mapping_mode(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = _write_corpus(root)
            first = _write_runner(root, corpus_path, "model-a", "baseline", correct=True)
            second = _write_runner(root, corpus_path, "model-b", "candidate", correct=False)
            packet_path = root / "packet.json"
            mapping_path = root / "mapping.json"
            exit_code = main(
                [
                    "build",
                    "--corpus",
                    str(corpus_path),
                    "--benchmark",
                    str(first),
                    "--benchmark",
                    str(second),
                    "--packet",
                    str(packet_path),
                    "--mapping",
                    str(mapping_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(mapping_path.stat().st_mode & 0o777, 0o600)
            generated_seed = json.loads(mapping_path.read_text(encoding="utf-8"))[
                "private_payload"
            ]["seed"]
            self.assertEqual(len(generated_seed), 64)
            self.assertNotIn(generated_seed, packet_path.read_text(encoding="utf-8"))
            self.assertEqual(
                main(
                    [
                        "build",
                        "--corpus",
                        str(corpus_path),
                        "--benchmark",
                        str(first),
                        "--benchmark",
                        str(second),
                        "--packet",
                        str(packet_path),
                        "--mapping",
                        str(root / "other.json"),
                    ]
                ),
                2,
            )

            target = root / "target.json"
            target.write_text("keep", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            self.assertEqual(
                main(
                    [
                        "build",
                        "--corpus",
                        str(corpus_path),
                        "--benchmark",
                        str(first),
                        "--benchmark",
                        str(second),
                        "--packet",
                        str(link),
                        "--mapping",
                        str(root / "third.json"),
                    ]
                ),
                2,
            )
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_cli_seed_file_is_private_deterministic_and_unique(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_path = _write_corpus(root)
            first = _write_runner(root, corpus_path, "model-a", "baseline", correct=True)
            second = _write_runner(root, corpus_path, "model-b", "candidate", correct=False)
            seed = "deterministic-review-seed-" + "a" * 32
            seed_path = root / "seed.txt"
            seed_path.write_text(seed + "\n", encoding="utf-8")
            seed_path.chmod(0o600)

            def build_arguments(packet: Path, mapping: Path, source: Path) -> list[str]:
                return [
                    "build",
                    "--corpus",
                    str(corpus_path),
                    "--benchmark",
                    str(first),
                    "--benchmark",
                    str(second),
                    "--seed-file",
                    str(source),
                    "--packet",
                    str(packet),
                    "--mapping",
                    str(mapping),
                ]

            first_packet = root / "seeded-packet-1.json"
            unsafe_arguments = build_arguments(
                root / "unsafe-seed-packet.json",
                root / "unsafe-seed-mapping.json",
                seed_path,
            )
            unsafe_arguments[unsafe_arguments.index("--seed-file")] = "--seed"
            with self.assertRaises(SystemExit):
                main(unsafe_arguments)
            first_mapping = root / "seeded-mapping-1.json"
            second_packet = root / "seeded-packet-2.json"
            second_mapping = root / "seeded-mapping-2.json"
            self.assertEqual(main(build_arguments(first_packet, first_mapping, seed_path)), 0)
            self.assertEqual(main(build_arguments(second_packet, second_mapping, seed_path)), 0)
            self.assertEqual(first_packet.read_bytes(), second_packet.read_bytes())
            self.assertEqual(
                json.loads(first_mapping.read_text(encoding="utf-8"))["private_payload"]["seed"],
                seed,
            )

            seed_path.chmod(0o644)
            self.assertEqual(
                main(
                    build_arguments(
                        root / "public-seed-packet.json",
                        root / "public-seed-mapping.json",
                        seed_path,
                    )
                ),
                2,
            )
            seed_path.chmod(0o600)
            seed_symlink = root / "seed-link.txt"
            seed_symlink.symlink_to(seed_path)
            self.assertEqual(
                main(
                    build_arguments(
                        root / "linked-seed-packet.json",
                        root / "linked-seed-mapping.json",
                        seed_symlink,
                    )
                ),
                2,
            )
            seed_hardlink = root / "seed-hardlink.txt"
            os.link(seed_path, seed_hardlink)
            self.assertEqual(
                main(
                    build_arguments(
                        root / "hardlinked-seed-packet.json",
                        root / "hardlinked-seed-mapping.json",
                        seed_hardlink,
                    )
                ),
                2,
            )

    def test_source_checkout_cli_help(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "review-voice-model-quality.py"
        with TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, "-I", str(script), "--help"],
                cwd=temporary,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("{build,validate,apply}", completed.stdout)
        self.assertIn("No model calls are made", completed.stdout)


def _write_corpus(root: Path) -> Path:
    rows = [
        {
            "schema_version": "1.0.0",
            "dataset_version": "quality-fixture-v1",
            "id": "answerable-001",
            "category": "local_detail",
            "evidence_packet": {
                "recording_id": "recording-1",
                "title": "Channel decision",
                "participants": ["Ada", "Bo"],
                "time_range": "2026-07-01T10:00:00Z/2026-07-01T10:05:00Z",
                "overview": "The team chooses a channel.",
                "closed_world": True,
                "turns": [
                    {
                        "turn_id": "turn-1",
                        "window": "W1",
                        "timestamp": "2026-07-01T10:01:00Z",
                        "speaker": "Ada",
                        "text": "Cobalt is the selected channel; Emerald is rejected.",
                    }
                ],
            },
            "question": "Which channel was selected?",
            "required_facts": [
                {
                    "fact_id": "channel",
                    "value": "Cobalt",
                    "aliases": ["the Cobalt channel"],
                    "evidence_turn_ids": ["turn-1"],
                }
            ],
            "forbidden_claims": ["Emerald was selected."],
            "attribution_targets": [{"fact_id": "channel", "speaker": "Ada"}],
            "timestamp_targets": [],
            "abstention_required": False,
            "gold_rationale": "Ada explicitly selects Cobalt.",
        },
        {
            "schema_version": "1.0.0",
            "dataset_version": "quality-fixture-v1",
            "id": "abstention-001",
            "category": "contradiction_unanswerable",
            "evidence_packet": {
                "recording_id": "recording-2",
                "title": "Route decision",
                "participants": ["Cy", "Dee"],
                "time_range": "2026-07-02T10:00:00Z/2026-07-02T10:05:00Z",
                "overview": "The route remains open.",
                "closed_world": True,
                "turns": [
                    {
                        "turn_id": "turn-2",
                        "window": "W1",
                        "timestamp": "2026-07-02T10:01:00Z",
                        "speaker": "Cy",
                        "text": "The route decision is still pending.",
                    }
                ],
            },
            "question": "Which route was selected?",
            "required_facts": [
                {
                    "fact_id": "answer_status",
                    "value": "Not provided in the evidence",
                    "aliases": ["still pending"],
                    "evidence_turn_ids": ["turn-2"],
                }
            ],
            "forbidden_claims": ["The east route was selected."],
            "attribution_targets": [],
            "timestamp_targets": [],
            "abstention_required": True,
            "gold_rationale": "No route is selected.",
        },
    ]
    path = root / "quality.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _write_runner(
    root: Path,
    corpus_path: Path,
    model_id: str,
    profile: str,
    *,
    correct: bool,
    suffix: str = "",
) -> Path:
    corpus_raw = corpus_path.read_bytes()
    corpus_rows = [json.loads(line) for line in corpus_raw.decode().splitlines() if line]
    model_sha = hashlib.sha256(model_id.encode()).hexdigest()
    artifact_path = root / f"{model_id}.gguf"
    payload = {
        "schema_version": 1,
        "status": "completed",
        "error_count": 0,
        "protocol": {
            "rounds": 1,
            "max_tokens": 160,
            "seed": 3407,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "cache_prompt": False,
            "enable_thinking": False,
        },
        "inventory": {
            "models": [
                {
                    "profile": profile,
                    "model_id": model_id,
                    "path": str(artifact_path),
                    "sha256": model_sha,
                    "size_bytes": 123,
                    "source_url": "https://example.invalid/model",
                    "source_revision": "revision",
                    "license": "Apache-2.0",
                }
            ]
        },
        "corpus": {
            "sha256": hashlib.sha256(corpus_raw).hexdigest(),
            "corpus_version": "quality-fixture-v1",
            "case_count": len(corpus_rows),
        },
        "artifact_verification": [
            {
                "profile": profile,
                "model_id": model_id,
                "path": str(artifact_path),
                "expected_sha256": model_sha,
                "actual_sha256": model_sha,
                "verified": True,
            }
        ],
        "route_verification": [
            {
                "profile": profile,
                "model_id": model_id,
                "inventory_path": str(artifact_path),
                "router_path": str(artifact_path),
                "verified": True,
            }
        ],
        "trials": [],
    }
    answers = (
        ["Cobalt was selected.", "Not provided in the evidence."]
        if correct
        else [
            "Emerald was selected, plus a bonus claim.",
            "The east route was selected, plus a bonus claim.",
        ]
    )
    for row, answer in zip(corpus_rows, answers):
        payload["trials"].append(
            {
                "round": 1,
                "profile": profile,
                "model_id": model_id,
                "prompt_id": row["id"],
                "category": row["category"],
                "excluded_from_summary": False,
                "served_model": model_id,
                "route_verified": True,
                "answer": answer,
                "error": None,
                "accuracy": {"score": 1.0 if not correct else 0.0},
            }
        )
    path = root / f"runner-{model_id}{suffix}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _make_judgments(packet: dict[str, object], packet_sha: str) -> dict[str, object]:
    reviews = []
    for item in packet["review_items"]:
        answer_judgments = []
        preferred = None
        for answer in item["answers"]:
            text = answer["text"]
            correct = text in {"Cobalt was selected.", "Not provided in the evidence."}
            if correct:
                preferred = answer["answer_id"]
            answer_judgments.append(
                {
                    "answer_id": answer["answer_id"],
                    "required_fact_judgments": [
                        {"fact_id": fact["fact_id"], "semantic_hit": correct}
                        for fact in item["gold"]["required_facts"]
                    ],
                    "abstention_correct": correct,
                    "forbidden_claim_judgments": [
                        {"claim_id": claim["claim_id"], "asserted": not correct}
                        for claim in item["gold"]["forbidden_claims"]
                    ],
                    "unsupported_claims": [] if correct else ["bonus claim"],
                }
            )
        reviews.append(
            {
                "review_item_id": item["review_item_id"],
                "answer_judgments": answer_judgments,
                "preference": {"kind": "preferred", "answer_ids": [preferred]},
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": JUDGMENTS_TYPE,
        "packet": {"packet_id": packet["packet_id"], "sha256": packet_sha},
        "reviewer": {"reviewer_id": "fixture-reviewer"},
        "reviews": reviews,
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    unittest.main()
