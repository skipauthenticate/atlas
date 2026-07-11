from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import secrets
import stat
import sys
from typing import Any, Iterable, Sequence

from atlas_voice.model_benchmark import DEFAULT_SYSTEM_PROMPT
from atlas_voice.quality_corpus import decode_quality_jsonl


SCHEMA_VERSION = "1.0.0"
TOOL_VERSION = "voice_model_blinded_semantic_review_v1"
PACKET_TYPE = "voice_model_blinded_review_packet"
MAPPING_TYPE = "voice_model_blinded_review_private_mapping"
JUDGMENTS_TYPE = "voice_model_blinded_review_judgments"
RESULT_TYPE = "voice_model_blinded_review_result"


class BlindedReviewError(ValueError):
    """Raised when a review input is incomplete, incompatible, or unsafe."""


@dataclass(frozen=True)
class ReviewCorpus:
    path: Path
    sha256: str
    dataset_version: str
    cases: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RunnerArtifact:
    path: Path
    sha256: str
    rounds: int
    protocol_fingerprint: dict[str, Any]
    models: tuple[dict[str, Any], ...]
    trials: tuple[dict[str, Any], ...]


def load_review_corpus(path: Path) -> ReviewCorpus:
    source_path = _resolve_input(path, "quality corpus")
    if source_path.suffix.casefold() != ".jsonl":
        raise BlindedReviewError("quality corpus must use the .jsonl extension")
    raw = _read_bytes(source_path, "quality corpus")
    try:
        converted = decode_quality_jsonl(raw, source_path)
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BlindedReviewError(f"invalid quality corpus {source_path}: {exc}") from exc

    converted_cases = converted.get("cases")
    if not isinstance(converted_cases, list) or len(converted_cases) != len(rows):
        raise BlindedReviewError("quality corpus conversion produced an invalid case matrix")
    review_cases: list[dict[str, Any]] = []
    for raw_row, runner_case in zip(rows, converted_cases):
        if not isinstance(raw_row, dict) or not isinstance(runner_case, dict):
            raise BlindedReviewError("quality corpus rows and converted cases must be objects")
        prompt_id = _required_string(raw_row, "id", "quality corpus row")
        if runner_case.get("id") != prompt_id:
            raise BlindedReviewError(f"quality corpus ID conversion mismatch for {prompt_id}")
        evidence = runner_case.get("evidence")
        question = runner_case.get("question")
        if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            raise BlindedReviewError(f"converted evidence is invalid for {prompt_id}")
        if not isinstance(question, str) or not question:
            raise BlindedReviewError(f"converted question is invalid for {prompt_id}")
        user_prompt = (
            "EVIDENCE PACKET:\n"
            + "\n".join(f"- {item}" for item in evidence)
            + f"\n\nQUESTION:\n{question}\n\nRespond in one concise sentence."
        )
        required_facts = []
        for fact in _required_list(raw_row, "required_facts", prompt_id):
            if not isinstance(fact, dict):
                raise BlindedReviewError(f"{prompt_id}.required_facts must contain objects")
            required_facts.append(
                {
                    "fact_id": _required_string(fact, "fact_id", prompt_id),
                    "value": _required_string(fact, "value", prompt_id),
                    "aliases": _string_list(fact.get("aliases"), f"{prompt_id}.aliases"),
                    "evidence_turn_ids": _string_list(
                        fact.get("evidence_turn_ids"),
                        f"{prompt_id}.evidence_turn_ids",
                    ),
                }
            )
        forbidden_claims = [
            {"claim_id": f"forbidden-{index:02d}", "text": claim}
            for index, claim in enumerate(
                _string_list(
                    raw_row.get("forbidden_claims"),
                    f"{prompt_id}.forbidden_claims",
                ),
                start=1,
            )
        ]
        evidence_packet = raw_row.get("evidence_packet")
        if not isinstance(evidence_packet, dict):
            raise BlindedReviewError(f"{prompt_id}.evidence_packet must be an object")
        review_cases.append(
            {
                "prompt_id": prompt_id,
                "category": _required_string(raw_row, "category", prompt_id),
                "prompt": {
                    "system": DEFAULT_SYSTEM_PROMPT,
                    "user": user_prompt,
                },
                "evidence": deepcopy(evidence_packet),
                "question": question,
                "gold": {
                    "required_facts": required_facts,
                    "forbidden_claims": forbidden_claims,
                    "abstention_required": _required_bool(
                        raw_row, "abstention_required", prompt_id
                    ),
                    "attribution_targets": deepcopy(
                        _required_list(raw_row, "attribution_targets", prompt_id)
                    ),
                    "timestamp_targets": deepcopy(
                        _required_list(raw_row, "timestamp_targets", prompt_id)
                    ),
                    "rationale": _required_string(raw_row, "gold_rationale", prompt_id),
                },
            }
        )
    return ReviewCorpus(
        path=source_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        dataset_version=_required_string(converted, "corpus_version", "corpus"),
        cases=tuple(review_cases),
    )


def load_runner_artifact(path: Path, corpus: ReviewCorpus) -> RunnerArtifact:
    source_path = _resolve_input(path, "runner artifact")
    raw = _read_bytes(source_path, "runner artifact")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BlindedReviewError(f"invalid runner JSON {source_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BlindedReviewError(f"runner artifact must be a JSON object: {source_path}")
    if payload.get("schema_version") != 1:
        raise BlindedReviewError(f"runner schema_version must be 1: {source_path}")
    if payload.get("status") != "completed":
        raise BlindedReviewError(f"runner status must be completed: {source_path}")
    if payload.get("error_count") != 0:
        raise BlindedReviewError(f"runner error_count must be zero: {source_path}")
    _verify_runner_corpus(payload, corpus, source_path)

    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise BlindedReviewError(f"runner protocol must be an object: {source_path}")
    rounds = protocol.get("rounds")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise BlindedReviewError(f"runner protocol.rounds must be positive: {source_path}")
    fingerprint_keys = (
        "rounds",
        "max_tokens",
        "seed",
        "temperature",
        "top_k",
        "top_p",
        "cache_prompt",
        "enable_thinking",
    )
    protocol_fingerprint = {key: protocol.get(key) for key in fingerprint_keys}
    if any(key not in protocol for key in fingerprint_keys):
        missing = [key for key in fingerprint_keys if key not in protocol]
        raise BlindedReviewError(
            f"runner protocol is missing comparison settings {missing}: {source_path}"
        )

    inventory = payload.get("inventory")
    raw_models = inventory.get("models") if isinstance(inventory, dict) else None
    if not isinstance(raw_models, list) or not raw_models:
        raise BlindedReviewError(f"runner inventory.models must be non-empty: {source_path}")
    models: list[dict[str, Any]] = []
    model_ids: set[str] = set()
    profiles: set[str] = set()
    for index, row in enumerate(raw_models):
        if not isinstance(row, dict):
            raise BlindedReviewError(f"inventory.models[{index}] must be an object")
        model_id = _required_string(row, "model_id", f"inventory.models[{index}]")
        profile = _required_string(row, "profile", f"inventory.models[{index}]")
        if model_id in model_ids or profile in profiles:
            raise BlindedReviewError(
                f"runner inventory contains duplicate model/profile: {source_path}"
            )
        model_ids.add(model_id)
        profiles.add(profile)
        sha256 = row.get("sha256")
        if sha256 is not None and not isinstance(sha256, str):
            raise BlindedReviewError(f"inventory model {model_id}.sha256 must be a string or null")
        models.append(
            {
                "profile": profile,
                "model_id": model_id,
                "path": _required_string(row, "path", f"inventory model {model_id}"),
                "sha256": sha256,
                "size_bytes": row.get("size_bytes"),
                "source_url": row.get("source_url"),
                "source_revision": row.get("source_revision"),
                "license": row.get("license"),
            }
        )
    model_by_id = {str(row["model_id"]): row for row in models}
    _verify_artifact_rows(payload.get("artifact_verification"), model_by_id, source_path)
    _verify_route_rows(payload.get("route_verification"), model_by_id, source_path)

    cases_by_id = {str(case["prompt_id"]): case for case in corpus.cases}
    expected = {
        (model_id, round_index, prompt_id)
        for model_id in model_by_id
        for round_index in range(1, rounds + 1)
        for prompt_id in cases_by_id
    }
    raw_trials = payload.get("trials")
    if not isinstance(raw_trials, list):
        raise BlindedReviewError(f"runner trials must be a list: {source_path}")
    actual: set[tuple[str, int, str]] = set()
    trials: list[dict[str, Any]] = []
    for index, trial in enumerate(raw_trials):
        if not isinstance(trial, dict):
            raise BlindedReviewError(f"runner trials[{index}] must be an object")
        model_id = trial.get("model_id")
        prompt_id = trial.get("prompt_id")
        round_index = trial.get("round")
        if model_id not in model_by_id:
            raise BlindedReviewError(f"trial uses unknown model_id {model_id!r}")
        if prompt_id not in cases_by_id:
            raise BlindedReviewError(f"trial uses unknown prompt_id {prompt_id!r}")
        if not isinstance(round_index, int) or isinstance(round_index, bool):
            raise BlindedReviewError(f"trial {index} round must be an integer")
        key = (str(model_id), round_index, str(prompt_id))
        if key in actual:
            raise BlindedReviewError(f"duplicate runner trial key: {key}")
        actual.add(key)
        model = model_by_id[str(model_id)]
        case = cases_by_id[str(prompt_id)]
        if trial.get("profile") != model["profile"]:
            raise BlindedReviewError(f"trial profile mismatch for {model_id}")
        if trial.get("category") != case["category"]:
            raise BlindedReviewError(f"trial category mismatch for {prompt_id}")
        if trial.get("excluded_from_summary") is not False:
            raise BlindedReviewError(f"trial {key} must not be excluded from summary")
        if trial.get("error") is not None:
            raise BlindedReviewError(f"trial {key} contains an error")
        if trial.get("route_verified") is not True:
            raise BlindedReviewError(f"trial {key} did not verify its route")
        if trial.get("served_model") != model_id:
            raise BlindedReviewError(f"trial {key} served a different model")
        if not isinstance(trial.get("answer"), str):
            raise BlindedReviewError(f"trial {key} answer must be a string")
        trials.append(
            {
                "round": round_index,
                "profile": model["profile"],
                "model_id": model_id,
                "prompt_id": prompt_id,
                "category": trial["category"],
                "answer": trial["answer"],
            }
        )
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise BlindedReviewError(
            "runner prompt/model/round matrix mismatch; "
            f"missing={missing}, extra={extra}: {source_path}"
        )
    return RunnerArtifact(
        path=source_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        rounds=rounds,
        protocol_fingerprint=protocol_fingerprint,
        models=tuple(models),
        trials=tuple(trials),
    )


def build_review_packet(
    corpus: ReviewCorpus,
    artifacts: Sequence[RunnerArtifact],
    *,
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(artifacts) < 2:
        raise BlindedReviewError("at least two completed runner artifacts are required")
    if not seed:
        raise BlindedReviewError("randomization seed must be non-empty")
    artifact_hashes = [artifact.sha256 for artifact in artifacts]
    if len(set(artifact_hashes)) != len(artifact_hashes):
        raise BlindedReviewError("runner artifacts must be distinct")
    rounds = {artifact.rounds for artifact in artifacts}
    if len(rounds) != 1:
        raise BlindedReviewError("runner artifacts use different round counts")
    fingerprints = {_canonical_json(artifact.protocol_fingerprint) for artifact in artifacts}
    if len(fingerprints) != 1:
        raise BlindedReviewError("runner artifacts use different comparison protocols")

    canonical_artifacts = sorted(artifacts, key=lambda item: (item.sha256, str(item.path)))
    source_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    global_model_ids: set[str] = set()
    trial_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for source_index, artifact in enumerate(canonical_artifacts, start=1):
        source_id = f"runner-{source_index:02d}"
        source_rows.append(
            {
                "source_id": source_id,
                "path": str(artifact.path),
                "sha256": artifact.sha256,
                "protocol": deepcopy(artifact.protocol_fingerprint),
            }
        )
        for model in artifact.models:
            model_id = str(model["model_id"])
            if model_id in global_model_ids:
                raise BlindedReviewError(
                    f"model_id appears in more than one runner artifact: {model_id}"
                )
            global_model_ids.add(model_id)
            model_key = (
                "model-"
                + hashlib.sha256(f"{artifact.sha256}\0{model_id}".encode("utf-8")).hexdigest()[:20]
            )
            model_rows.append(
                {
                    "model_key": model_key,
                    "source_id": source_id,
                    "profile": model["profile"],
                    "model_id": model_id,
                    "artifact": {
                        **{
                            key: deepcopy(model.get(key))
                            for key in (
                                "path",
                                "sha256",
                                "size_bytes",
                                "source_url",
                                "source_revision",
                                "license",
                            )
                        },
                        "verification": deepcopy(model.get("artifact_verification")),
                    },
                }
            )
        for trial in artifact.trials:
            key = (str(trial["model_id"]), int(trial["round"]), str(trial["prompt_id"]))
            trial_by_key[key] = trial
    if len(model_rows) < 2:
        raise BlindedReviewError("runner artifacts must contain at least two total models")

    seed_digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    randomized_models = sorted(model_rows, key=lambda item: str(item["model_key"]))
    random.Random(_derived_seed(seed, "model-labels")).shuffle(randomized_models)
    for index, model in enumerate(randomized_models):
        model["label"] = _neutral_label(index)
    models_by_id = {str(model["model_id"]): model for model in model_rows}
    labels = sorted(str(model["label"]) for model in model_rows)
    case_order = {str(case["prompt_id"]): index for index, case in enumerate(corpus.cases)}
    answer_mapping: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    round_count = next(iter(rounds))
    for round_index in range(1, round_count + 1):
        for case in corpus.cases:
            prompt_id = str(case["prompt_id"])
            review_item_id = (
                "review-"
                + hashlib.sha256(
                    f"{corpus.sha256}\0{round_index}\0{prompt_id}".encode("utf-8")
                ).hexdigest()[:20]
            )
            answers: list[dict[str, Any]] = []
            for model_id, model in sorted(models_by_id.items()):
                trial = trial_by_key[(model_id, round_index, prompt_id)]
                answer_id = (
                    "answer-"
                    + hashlib.sha256(
                        (
                            f"{seed}\0{review_item_id}\0{model['model_key']}\0{trial['answer']}"
                        ).encode("utf-8")
                    ).hexdigest()[:20]
                )
                answers.append(
                    {
                        "answer_id": answer_id,
                        "model_label": model["label"],
                        "text": trial["answer"],
                    }
                )
                answer_mapping.append(
                    {
                        "review_item_id": review_item_id,
                        "prompt_id": prompt_id,
                        "round": round_index,
                        "answer_id": answer_id,
                        "model_key": model["model_key"],
                        "model_label": model["label"],
                    }
                )
            random.Random(_derived_seed(seed, review_item_id)).shuffle(answers)
            review_items.append(
                {
                    "review_item_id": review_item_id,
                    "prompt_id": prompt_id,
                    "round": round_index,
                    "category": case["category"],
                    "prompt": deepcopy(case["prompt"]),
                    "evidence": deepcopy(case["evidence"]),
                    "question": case["question"],
                    "gold": deepcopy(case["gold"]),
                    "answers": answers,
                }
            )
    review_items.sort(key=lambda item: (int(item["round"]), case_order[str(item["prompt_id"])]))

    private_payload = {
        "seed": seed,
        "sources": source_rows,
        "models": sorted(model_rows, key=lambda item: str(item["label"])),
        "answers": sorted(answer_mapping, key=lambda item: str(item["answer_id"])),
    }
    mapping_commitment = _sha256_json(private_payload)
    runner_set_commitment = _sha256_json(sorted(artifact_hashes))
    packet_id = (
        "packet-"
        + hashlib.sha256(
            f"{corpus.sha256}\0{runner_set_commitment}\0{seed_digest}".encode("utf-8")
        ).hexdigest()[:20]
    )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": PACKET_TYPE,
        "tool_version": TOOL_VERSION,
        "packet_id": packet_id,
        "status": "ready_for_review",
        "inputs": {
            "corpus": {
                "sha256": corpus.sha256,
                "dataset_version": corpus.dataset_version,
                "case_count": len(corpus.cases),
            },
            "runner_set_commitment_sha256": runner_set_commitment,
            "runner_artifact_count": len(artifacts),
            "model_count": len(model_rows),
            "rounds": round_count,
        },
        "randomization": {
            "method": "SHA-256-derived deterministic Python random shuffles",
            "seed_sha256": seed_digest,
            "mapping_commitment_sha256": mapping_commitment,
            "neutral_model_labels": labels,
            "answer_order_shuffled_per_review_item": True,
        },
        "review_guidance": {
            "blinding": (
                "Model identities, profiles, source paths, and runtime metrics are withheld. "
                "Answer text is preserved verbatim and is not scrubbed for self-identification."
            ),
            "semantic_hit": (
                "Mark true when the answer communicates the required fact correctly, including "
                "a valid paraphrase; do not rely only on exact string overlap."
            ),
            "abstention_correct": (
                "For abstention-required cases, mark true only for an appropriately uncertain "
                "answer that does not select an unsupported outcome. For answerable cases, mark "
                "true only when the answer does not wrongly abstain."
            ),
            "forbidden_asserted": (
                "Mark true only when the answer asserts or endorses the forbidden claim, not "
                "when it explicitly rejects or corrects it."
            ),
            "unsupported_claims": (
                "List each material claim not supported by the closed-world evidence; use an "
                "empty list when none are present."
            ),
            "preference": (
                "Choose one preferred answer, or mark a tie among two or more answer IDs. "
                "Judge grounding, completeness, correctness, and concise usefulness."
            ),
        },
        "judgment_contract": {
            "artifact_type": JUDGMENTS_TYPE,
            "per_answer_required_fields": [
                "answer_id",
                "required_fact_judgments[{fact_id, semantic_hit}]",
                "abstention_correct",
                "forbidden_claim_judgments[{claim_id, asserted}]",
                "unsupported_claims[]",
            ],
            "per_item_preference": {
                "preferred": "kind=preferred with exactly one answer_id",
                "tie": "kind=tie with at least two distinct answer_ids",
            },
        },
        "review_items": review_items,
    }
    packet_sha256 = hashlib.sha256(_json_bytes(packet)).hexdigest()
    mapping = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MAPPING_TYPE,
        "tool_version": TOOL_VERSION,
        "packet": {"packet_id": packet_id, "sha256": packet_sha256},
        "commitment": {
            "algorithm": "sha256-canonical-json",
            "sha256": mapping_commitment,
            "covered_field": "private_payload",
        },
        "private_payload": private_payload,
    }
    return packet, mapping


def validate_judgments(
    packet: dict[str, Any],
    mapping: dict[str, Any],
    judgments: dict[str, Any],
    *,
    packet_sha256: str,
) -> dict[str, Any]:
    _verify_packet_and_mapping(packet, mapping, packet_sha256)
    _expect_type(judgments, JUDGMENTS_TYPE, "judgments")
    judgment_packet = judgments.get("packet")
    if not isinstance(judgment_packet, dict):
        raise BlindedReviewError("judgments.packet must be an object")
    if judgment_packet.get("packet_id") != packet.get("packet_id"):
        raise BlindedReviewError("judgments packet_id does not match the review packet")
    if judgment_packet.get("sha256") != packet_sha256:
        raise BlindedReviewError("judgments packet SHA-256 does not match the review packet")
    reviewer = judgments.get("reviewer")
    if not isinstance(reviewer, dict):
        raise BlindedReviewError("judgments.reviewer must be an object")
    _required_string(reviewer, "reviewer_id", "judgments.reviewer")

    raw_reviews = judgments.get("reviews")
    if not isinstance(raw_reviews, list):
        raise BlindedReviewError("judgments.reviews must be a list")
    packet_items = packet.get("review_items")
    if not isinstance(packet_items, list):
        raise BlindedReviewError("packet.review_items must be a list")
    items_by_id = {
        _required_string(item, "review_item_id", "packet review item"): item
        for item in packet_items
        if isinstance(item, dict)
    }
    if len(items_by_id) != len(packet_items):
        raise BlindedReviewError("packet contains duplicate or invalid review_item_id values")
    reviews_by_id: dict[str, dict[str, Any]] = {}
    normalized_reviews: list[dict[str, Any]] = []
    for review in raw_reviews:
        if not isinstance(review, dict):
            raise BlindedReviewError("judgments reviews must contain objects")
        review_item_id = _required_string(review, "review_item_id", "judgment review")
        if review_item_id in reviews_by_id:
            raise BlindedReviewError(f"duplicate judgment review_item_id: {review_item_id}")
        item = items_by_id.get(review_item_id)
        if item is None:
            raise BlindedReviewError(f"unknown judgment review_item_id: {review_item_id}")
        normalized = _validate_item_judgment(review, item)
        reviews_by_id[review_item_id] = normalized
        normalized_reviews.append(normalized)
    if set(reviews_by_id) != set(items_by_id):
        missing = sorted(set(items_by_id) - set(reviews_by_id))[:10]
        extra = sorted(set(reviews_by_id) - set(items_by_id))[:10]
        raise BlindedReviewError(
            f"judgment review-item matrix mismatch; missing={missing}, extra={extra}"
        )
    return {
        "reviewer": deepcopy(reviewer),
        "reviews": normalized_reviews,
        "review_item_count": len(normalized_reviews),
        "answer_judgment_count": sum(
            len(review["answer_judgments"]) for review in normalized_reviews
        ),
    }


def apply_judgments(
    packet: dict[str, Any],
    mapping: dict[str, Any],
    judgments: dict[str, Any],
    *,
    packet_path: Path,
    packet_sha256: str,
    mapping_path: Path,
    mapping_sha256: str,
    judgments_path: Path,
    judgments_sha256: str,
) -> dict[str, Any]:
    validated = validate_judgments(
        packet,
        mapping,
        judgments,
        packet_sha256=packet_sha256,
    )
    private_payload = mapping["private_payload"]
    private_models = private_payload["models"]
    answer_mapping = {str(item["answer_id"]): item for item in private_payload["answers"]}
    models_by_key = {str(item["model_key"]): item for item in private_models}
    packet_items = {str(item["review_item_id"]): item for item in packet["review_items"]}

    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    deblinded_reviews: list[dict[str, Any]] = []
    for review in validated["reviews"]:
        review_item_id = str(review["review_item_id"])
        packet_item = packet_items[review_item_id]
        preferred_ids = set(review["preference"]["answer_ids"])
        preference_credit = 1.0 / len(preferred_ids)
        deblinded_answers: list[dict[str, Any]] = []
        for answer_judgment in review["answer_judgments"]:
            answer_id = str(answer_judgment["answer_id"])
            private_answer = answer_mapping[answer_id]
            model = models_by_key[str(private_answer["model_key"])]
            fact_hits = sum(
                bool(item["semantic_hit"]) for item in answer_judgment["required_fact_judgments"]
            )
            forbidden_hits = sum(
                bool(item["asserted"]) for item in answer_judgment["forbidden_claim_judgments"]
            )
            selected = answer_id in preferred_ids
            observation = {
                "review_item_id": review_item_id,
                "prompt_id": packet_item["prompt_id"],
                "round": packet_item["round"],
                "category": packet_item["category"],
                "abstention_required": packet_item["gold"]["abstention_required"],
                "required_fact_hit_count": fact_hits,
                "required_fact_target_count": len(answer_judgment["required_fact_judgments"]),
                "abstention_correct": answer_judgment["abstention_correct"],
                "forbidden_claim_hit_count": forbidden_hits,
                "forbidden_claim_target_count": len(answer_judgment["forbidden_claim_judgments"]),
                "unsupported_claim_count": len(answer_judgment["unsupported_claims"]),
                "preference_selected": selected,
                "preference_kind": review["preference"]["kind"],
                "preference_credit": preference_credit if selected else 0.0,
            }
            observations[str(model["model_key"])].append(observation)
            deblinded_answers.append(
                {
                    "answer_id": answer_id,
                    "model_label": model["label"],
                    "profile": model["profile"],
                    "model_id": model["model_id"],
                    "judgment": deepcopy(answer_judgment),
                    "preference_selected": selected,
                    "preference_credit": round(observation["preference_credit"], 6),
                }
            )
        deblinded_reviews.append(
            {
                "review_item_id": review_item_id,
                "prompt_id": packet_item["prompt_id"],
                "round": packet_item["round"],
                "category": packet_item["category"],
                "preference": deepcopy(review["preference"]),
                "answers": deblinded_answers,
            }
        )

    model_summaries: list[dict[str, Any]] = []
    categories = list(dict.fromkeys(str(item["category"]) for item in packet["review_items"]))
    for model in sorted(private_models, key=lambda item: str(item["label"])):
        selected = observations[str(model["model_key"])]
        model_summaries.append(
            {
                "model_label": model["label"],
                "profile": model["profile"],
                "model_id": model["model_id"],
                "source_id": model["source_id"],
                "artifact": deepcopy(model["artifact"]),
                "overall": _summarize_observations(selected),
                "by_category": [
                    {
                        "category": category,
                        **_summarize_observations(
                            [item for item in selected if item["category"] == category]
                        ),
                    }
                    for category in categories
                ],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RESULT_TYPE,
        "tool_version": TOOL_VERSION,
        "status": "completed",
        "inputs": {
            "packet": {"path": str(packet_path), "sha256": packet_sha256},
            "mapping": {"path": str(mapping_path), "sha256": mapping_sha256},
            "judgments": {"path": str(judgments_path), "sha256": judgments_sha256},
            "corpus": deepcopy(packet["inputs"]["corpus"]),
        },
        "validation": {
            "packet_mapping_commitment_verified": True,
            "corpus_provenance_verified_when_packet_was_built": True,
            "runner_artifact_count": packet["inputs"]["runner_artifact_count"],
            "runner_routes_and_trial_matrix_verified_when_packet_was_built": True,
            "review_item_matrix_verified": True,
            "answer_fact_and_forbidden_target_matrices_verified": True,
            "review_item_count": validated["review_item_count"],
            "answer_judgment_count": validated["answer_judgment_count"],
        },
        "methodology": {
            "score_source": (
                "Only explicit reviewer judgments in the supplied judgments artifact are "
                "aggregated; runner lexical accuracy/self-grade fields are not present in the "
                "packet and are not trusted."
            ),
            "preference_credit": (
                "A sole preference contributes 1.0. A tie divides 1.0 equally across the "
                "selected answers. Each model has exactly one answer per review item."
            ),
            "reviewer_independence_caveat": (
                "The tool validates reviewer attribution and matrix completeness but cannot "
                "prove that the reviewer was independent or human."
            ),
        },
        "reviewer": validated["reviewer"],
        "metric_definitions": _metric_definitions(),
        "model_summaries": model_summaries,
        "deblinded_reviews": deblinded_reviews,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Build, validate, and apply an auditable blinded semantic review for completed "
            "voice-model benchmark artifacts. No model calls are made."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        allow_abbrev=False,
        help="Build a public blinded packet and private model mapping.",
    )
    build.add_argument("--corpus", type=Path, required=True, help="Versioned quality JSONL.")
    build.add_argument(
        "--benchmark",
        type=Path,
        action="append",
        required=True,
        help="Completed runner JSON; repeat for at least two artifacts.",
    )
    build.add_argument(
        "--seed-file",
        type=Path,
        help=(
            "Read a private deterministic shuffle seed from a unique, owner-controlled "
            "mode-0600 regular file. By default a cryptographically random seed is generated "
            "internally and retained only in the private mapping."
        ),
    )
    build.add_argument("--packet", type=Path, required=True, help="Public packet JSON output.")
    build.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="Private model mapping JSON output (created mode 0600).",
    )

    validate = subparsers.add_parser(
        "validate",
        allow_abbrev=False,
        help="Validate packet, private mapping, and completed reviewer judgments.",
    )
    _add_review_inputs(validate)

    apply = subparsers.add_parser(
        "apply",
        allow_abbrev=False,
        help="Validate judgments and write deblinded per-model/category aggregates.",
    )
    _add_review_inputs(apply)
    apply.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Private deblinded aggregate JSON output (created mode 0600).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            if len(args.benchmark) < 2:
                raise BlindedReviewError("repeat --benchmark for at least two artifacts")
            corpus = load_review_corpus(args.corpus)
            artifacts = [load_runner_artifact(path, corpus) for path in args.benchmark]
            seed = _load_build_seed(args.seed_file)
            packet, mapping = build_review_packet(corpus, artifacts, seed=seed)
            protected = [corpus.path, *(artifact.path for artifact in artifacts)]
            packet_path = _prepare_output(args.packet, protected)
            mapping_path = _prepare_output(args.mapping, [*protected, packet_path])
            if packet_path == mapping_path:
                raise BlindedReviewError("packet and mapping outputs must be distinct")
            _atomic_write_json(packet_path, packet, mode=0o644)
            try:
                _atomic_write_json(mapping_path, mapping, mode=0o600)
            except Exception:
                packet_path.unlink(missing_ok=True)
                raise
            print(f"Blinded review packet: {packet_path}")
            print(f"Private review mapping: {mapping_path}")
            return 0

        packet, packet_sha256, packet_path = _load_typed_json(args.packet, PACKET_TYPE)
        mapping, mapping_sha256, mapping_path = _load_typed_json(args.mapping, MAPPING_TYPE)
        judgments, judgments_sha256, judgments_path = _load_typed_json(
            args.judgments, JUDGMENTS_TYPE
        )
        if args.command == "validate":
            validated = validate_judgments(packet, mapping, judgments, packet_sha256=packet_sha256)
            print(
                "Validated blinded review: "
                f"{validated['review_item_count']} items, "
                f"{validated['answer_judgment_count']} answer judgments."
            )
            return 0
        result = apply_judgments(
            packet,
            mapping,
            judgments,
            packet_path=packet_path,
            packet_sha256=packet_sha256,
            mapping_path=mapping_path,
            mapping_sha256=mapping_sha256,
            judgments_path=judgments_path,
            judgments_sha256=judgments_sha256,
        )
        output = _prepare_output(args.output, (packet_path, mapping_path, judgments_path))
        _atomic_write_json(output, result, mode=0o600)
        print(f"Blinded semantic review result: {output}")
        return 0
    except (BlindedReviewError, OSError) as exc:
        print(f"Blinded review failed: {exc}", file=sys.stderr)
        return 2


def _verify_runner_corpus(payload: dict[str, Any], corpus: ReviewCorpus, source_path: Path) -> None:
    metadata = payload.get("corpus")
    if not isinstance(metadata, dict):
        raise BlindedReviewError(f"runner corpus metadata must be an object: {source_path}")
    expected = {
        "sha256": corpus.sha256,
        "corpus_version": corpus.dataset_version,
        "case_count": len(corpus.cases),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise BlindedReviewError(
                f"runner corpus {key} does not match the supplied corpus: {source_path}"
            )


def _verify_artifact_rows(
    value: Any, models_by_id: dict[str, dict[str, Any]], source_path: Path
) -> None:
    rows = _indexed_verification_rows(value, models_by_id, "artifact", source_path)
    for model_id, row in rows.items():
        model = models_by_id[model_id]
        verified = row.get("verified")
        if not isinstance(verified, bool):
            raise BlindedReviewError(f"artifact verification status must be boolean: {model_id}")
        if verified:
            model_sha = model.get("sha256")
            expected = row.get("expected_sha256")
            actual = row.get("actual_sha256")
            if (
                not isinstance(model_sha, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", model_sha) is None
                or expected != model_sha
                or actual != model_sha
            ):
                raise BlindedReviewError(f"artifact SHA-256 verification mismatch: {model_id}")
        else:
            limitation = row.get("limitation")
            if not isinstance(limitation, str) or not limitation.strip():
                raise BlindedReviewError(
                    f"unverified artifact must declare a limitation: {model_id}"
                )
        model["artifact_verification"] = {
            "verified": verified,
            "expected_sha256": row.get("expected_sha256"),
            "actual_sha256": row.get("actual_sha256"),
            "observed_sha256": row.get("sha256"),
            "limitation": row.get("limitation"),
        }


def _verify_route_rows(
    value: Any, models_by_id: dict[str, dict[str, Any]], source_path: Path
) -> None:
    rows = _indexed_verification_rows(value, models_by_id, "route", source_path)
    for model_id, row in rows.items():
        if row.get("verified") is not True:
            raise BlindedReviewError(f"router path was not verified: {model_id}")
        if row.get("model_id") != model_id:
            raise BlindedReviewError(f"route verification model mismatch: {model_id}")
        model_path = _absolute_path_value(
            models_by_id[model_id].get("path"), f"inventory path for {model_id}"
        )
        if "served_model_id" in row:
            if row.get("served_model_id") != model_id:
                raise BlindedReviewError(f"route served-model mismatch: {model_id}")
            route_path = _absolute_path_value(row.get("path"), f"route path for {model_id}")
        else:
            inventory_path = _absolute_path_value(
                row.get("inventory_path"), f"route inventory_path for {model_id}"
            )
            route_path = _absolute_path_value(
                row.get("router_path"), f"route router_path for {model_id}"
            )
            if inventory_path != model_path:
                raise BlindedReviewError(f"route inventory-path mismatch: {model_id}")
        if route_path != model_path:
            raise BlindedReviewError(f"route model-path mismatch: {model_id}")


def _absolute_path_value(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BlindedReviewError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise BlindedReviewError(f"{field} must be absolute")
    return path.resolve()


def _indexed_verification_rows(
    value: Any,
    models_by_id: dict[str, dict[str, Any]],
    label: str,
    source_path: Path,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise BlindedReviewError(f"runner {label}_verification must be a list: {source_path}")
    rows: dict[str, dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, dict):
            raise BlindedReviewError(f"runner {label}_verification rows must be objects")
        model_id = row.get("model_id")
        if model_id not in models_by_id:
            raise BlindedReviewError(f"{label} verification uses unknown model_id {model_id!r}")
        if model_id in rows:
            raise BlindedReviewError(f"duplicate {label} verification for {model_id}")
        if row.get("profile") != models_by_id[str(model_id)]["profile"]:
            raise BlindedReviewError(f"{label} verification profile mismatch for {model_id}")
        rows[str(model_id)] = row
    if set(rows) != set(models_by_id):
        raise BlindedReviewError(f"{label} verification model matrix is incomplete")
    return rows


def _validate_item_judgment(review: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    review_item_id = str(item["review_item_id"])
    answers = item.get("answers")
    gold = item.get("gold")
    if not isinstance(answers, list) or not isinstance(gold, dict):
        raise BlindedReviewError(f"packet review item is malformed: {review_item_id}")
    answer_ids = {
        _required_string(answer, "answer_id", review_item_id)
        for answer in answers
        if isinstance(answer, dict)
    }
    if len(answer_ids) != len(answers):
        raise BlindedReviewError(f"packet has duplicate answer IDs: {review_item_id}")
    fact_ids = {
        _required_string(fact, "fact_id", review_item_id)
        for fact in _required_list(gold, "required_facts", review_item_id)
        if isinstance(fact, dict)
    }
    forbidden_ids = {
        _required_string(claim, "claim_id", review_item_id)
        for claim in _required_list(gold, "forbidden_claims", review_item_id)
        if isinstance(claim, dict)
    }
    raw_answers = review.get("answer_judgments")
    if not isinstance(raw_answers, list):
        raise BlindedReviewError(f"{review_item_id}.answer_judgments must be a list")
    normalized_answers: list[dict[str, Any]] = []
    seen_answers: set[str] = set()
    for judgment in raw_answers:
        if not isinstance(judgment, dict):
            raise BlindedReviewError(f"{review_item_id} answer judgments must be objects")
        answer_id = _required_string(judgment, "answer_id", review_item_id)
        if answer_id not in answer_ids or answer_id in seen_answers:
            raise BlindedReviewError(
                f"{review_item_id} has unknown or duplicate answer judgment {answer_id}"
            )
        seen_answers.add(answer_id)
        fact_judgments = _validate_boolean_targets(
            judgment.get("required_fact_judgments"),
            expected_ids=fact_ids,
            id_key="fact_id",
            bool_key="semantic_hit",
            field=f"{review_item_id}/{answer_id}.required_fact_judgments",
        )
        forbidden_judgments = _validate_boolean_targets(
            judgment.get("forbidden_claim_judgments"),
            expected_ids=forbidden_ids,
            id_key="claim_id",
            bool_key="asserted",
            field=f"{review_item_id}/{answer_id}.forbidden_claim_judgments",
        )
        abstention_correct = judgment.get("abstention_correct")
        if not isinstance(abstention_correct, bool):
            raise BlindedReviewError(
                f"{review_item_id}/{answer_id}.abstention_correct must be boolean"
            )
        unsupported = _string_list(
            judgment.get("unsupported_claims"),
            f"{review_item_id}/{answer_id}.unsupported_claims",
        )
        normalized_answers.append(
            {
                "answer_id": answer_id,
                "required_fact_judgments": fact_judgments,
                "abstention_correct": abstention_correct,
                "forbidden_claim_judgments": forbidden_judgments,
                "unsupported_claims": unsupported,
                **({"notes": judgment["notes"]} if "notes" in judgment else {}),
            }
        )
    if seen_answers != answer_ids:
        missing = sorted(answer_ids - seen_answers)
        raise BlindedReviewError(
            f"{review_item_id} answer-judgment matrix is incomplete; missing={missing}"
        )
    preference = review.get("preference")
    if not isinstance(preference, dict):
        raise BlindedReviewError(f"{review_item_id}.preference must be an object")
    kind = preference.get("kind")
    preferred_ids = preference.get("answer_ids")
    if kind not in {"preferred", "tie"}:
        raise BlindedReviewError(f"{review_item_id}.preference.kind must be preferred or tie")
    if not isinstance(preferred_ids, list) or not all(
        isinstance(answer_id, str) and answer_id for answer_id in preferred_ids
    ):
        raise BlindedReviewError(
            f"{review_item_id}.preference.answer_ids must be non-empty strings"
        )
    if len(preferred_ids) != len(set(preferred_ids)):
        raise BlindedReviewError(f"{review_item_id}.preference contains duplicate IDs")
    if not set(preferred_ids).issubset(answer_ids):
        raise BlindedReviewError(f"{review_item_id}.preference uses unknown answer IDs")
    if kind == "preferred" and len(preferred_ids) != 1:
        raise BlindedReviewError(
            f"{review_item_id} preferred choice must contain exactly one answer"
        )
    if kind == "tie" and len(preferred_ids) < 2:
        raise BlindedReviewError(f"{review_item_id} tie must contain at least two answers")
    return {
        "review_item_id": review_item_id,
        "answer_judgments": normalized_answers,
        "preference": {"kind": kind, "answer_ids": list(preferred_ids)},
        **({"notes": review["notes"]} if "notes" in review else {}),
    }


def _validate_boolean_targets(
    value: Any,
    *,
    expected_ids: set[str],
    id_key: str,
    bool_key: str,
    field: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise BlindedReviewError(f"{field} must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, dict):
            raise BlindedReviewError(f"{field} must contain objects")
        target_id = _required_string(row, id_key, field)
        flag = row.get(bool_key)
        if target_id not in expected_ids or target_id in seen:
            raise BlindedReviewError(f"{field} has unknown or duplicate ID {target_id}")
        if not isinstance(flag, bool):
            raise BlindedReviewError(f"{field}.{target_id}.{bool_key} must be boolean")
        seen.add(target_id)
        normalized.append({id_key: target_id, bool_key: flag})
    if seen != expected_ids:
        raise BlindedReviewError(
            f"{field} target matrix mismatch; missing={sorted(expected_ids - seen)}"
        )
    return normalized


def _verify_packet_and_mapping(
    packet: dict[str, Any], mapping: dict[str, Any], packet_sha256: str
) -> None:
    _expect_type(packet, PACKET_TYPE, "packet")
    _expect_type(mapping, MAPPING_TYPE, "mapping")
    mapping_packet = mapping.get("packet")
    if not isinstance(mapping_packet, dict):
        raise BlindedReviewError("mapping.packet must be an object")
    if mapping_packet.get("packet_id") != packet.get("packet_id"):
        raise BlindedReviewError("private mapping packet_id does not match packet")
    if mapping_packet.get("sha256") != packet_sha256:
        raise BlindedReviewError("private mapping packet SHA-256 does not match packet bytes")
    private_payload = mapping.get("private_payload")
    commitment = mapping.get("commitment")
    if not isinstance(private_payload, dict) or not isinstance(commitment, dict):
        raise BlindedReviewError("private mapping payload/commitment is malformed")
    actual_commitment = _sha256_json(private_payload)
    packet_randomization = packet.get("randomization")
    packet_commitment = (
        packet_randomization.get("mapping_commitment_sha256")
        if isinstance(packet_randomization, dict)
        else None
    )
    if commitment.get("sha256") != actual_commitment or packet_commitment != actual_commitment:
        raise BlindedReviewError("private mapping commitment does not match the packet")
    models = private_payload.get("models")
    answers = private_payload.get("answers")
    if not isinstance(models, list) or not isinstance(answers, list):
        raise BlindedReviewError("private mapping models/answers must be lists")
    model_keys = {
        _required_string(model, "model_key", "private model")
        for model in models
        if isinstance(model, dict)
    }
    labels = {
        _required_string(model, "label", "private model")
        for model in models
        if isinstance(model, dict)
    }
    if len(model_keys) != len(models) or len(labels) != len(models):
        raise BlindedReviewError("private mapping contains duplicate/invalid models or labels")
    public_answer_rows = [
        answer
        for item in packet.get("review_items", [])
        if isinstance(item, dict)
        for answer in item.get("answers", [])
        if isinstance(answer, dict)
    ]
    public_answers = {
        str(answer.get("answer_id")): str(answer.get("model_label"))
        for answer in public_answer_rows
    }
    if len(public_answers) != len(public_answer_rows):
        raise BlindedReviewError("packet contains duplicate answer IDs")
    private_answers: dict[str, str] = {}
    for answer in answers:
        if not isinstance(answer, dict):
            raise BlindedReviewError("private answer mappings must be objects")
        answer_id = _required_string(answer, "answer_id", "private answer")
        model_key = _required_string(answer, "model_key", "private answer")
        label = _required_string(answer, "model_label", "private answer")
        if model_key not in model_keys or label not in labels or answer_id in private_answers:
            raise BlindedReviewError("private answer mapping contains an invalid identity")
        private_answers[answer_id] = label
    if private_answers != public_answers:
        raise BlindedReviewError("private/public answer-label matrices do not match")


def _summarize_observations(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fact_targets = sum(int(item["required_fact_target_count"]) for item in items)
    fact_hits = sum(int(item["required_fact_hit_count"]) for item in items)
    answerable = [item for item in items if not item["abstention_required"]]
    answerable_targets = sum(int(item["required_fact_target_count"]) for item in answerable)
    answerable_hits = sum(int(item["required_fact_hit_count"]) for item in answerable)
    abstention_required = [item for item in items if item["abstention_required"]]
    forbidden_targets = sum(int(item["forbidden_claim_target_count"]) for item in items)
    forbidden_hits = sum(int(item["forbidden_claim_hit_count"]) for item in items)
    preference_credit = sum(float(item["preference_credit"]) for item in items)
    return {
        "answer_count": len(items),
        "required_fact_semantic_recall": _metric(fact_hits, fact_targets, "hit_count"),
        "answerable_required_fact_semantic_recall": _metric(
            answerable_hits, answerable_targets, "hit_count"
        ),
        "abstention_correct_rate": _metric(
            sum(bool(item["abstention_correct"]) for item in items),
            len(items),
            "pass_count",
        ),
        "abstention_required_accuracy": _metric(
            sum(bool(item["abstention_correct"]) for item in abstention_required),
            len(abstention_required),
            "pass_count",
        ),
        "answerable_nonabstention_accuracy": _metric(
            sum(bool(item["abstention_correct"]) for item in answerable),
            len(answerable),
            "pass_count",
        ),
        "forbidden_claim_assertion_rate": _metric(
            forbidden_hits, forbidden_targets, "asserted_count"
        ),
        "forbidden_claim_answer_rate": _metric(
            sum(int(item["forbidden_claim_hit_count"]) > 0 for item in items),
            len(items),
            "flagged_answer_count",
        ),
        "unsupported_claim_answer_rate": _metric(
            sum(int(item["unsupported_claim_count"]) > 0 for item in items),
            len(items),
            "flagged_answer_count",
        ),
        "unsupported_claim_count": sum(int(item["unsupported_claim_count"]) for item in items),
        "preference": {
            "credit": round(preference_credit, 6),
            "review_item_count": len(items),
            "credit_rate": _rate(preference_credit, len(items)),
            "sole_preference_count": sum(
                item["preference_selected"] and item["preference_kind"] == "preferred"
                for item in items
            ),
            "tie_selection_count": sum(
                item["preference_selected"] and item["preference_kind"] == "tie" for item in items
            ),
        },
    }


def _metric(numerator: float, denominator: int, numerator_name: str) -> dict[str, Any]:
    return {
        numerator_name: int(numerator) if float(numerator).is_integer() else numerator,
        "target_count": denominator,
        "rate": _rate(numerator, denominator),
    }


def _rate(numerator: float, denominator: int) -> float | None:
    return round(float(numerator) / denominator, 6) if denominator else None


def _metric_definitions() -> dict[str, str]:
    return {
        "required_fact_semantic_recall": (
            "Reviewer-marked semantic fact hits divided by declared required facts."
        ),
        "answerable_required_fact_semantic_recall": (
            "The same semantic recall restricted to cases where abstention_required is false."
        ),
        "abstention_correct_rate": (
            "Reviewer-marked abstention behavior correctness across all answers."
        ),
        "forbidden_claim_assertion_rate": (
            "Reviewer-marked asserted/endorsed forbidden claims divided by declared targets."
        ),
        "unsupported_claim_answer_rate": (
            "Answers with one or more reviewer-recorded unsupported material claims."
        ),
        "preference.credit_rate": (
            "Per-item preference credit; ties split one credit equally among selected answers."
        ),
    }


def _add_review_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--packet", type=Path, required=True, help="Public packet JSON.")
    parser.add_argument("--mapping", type=Path, required=True, help="Private mapping JSON.")
    parser.add_argument(
        "--judgments", type=Path, required=True, help="Completed reviewer judgments JSON."
    )


def _load_typed_json(path: Path, expected_type: str) -> tuple[dict[str, Any], str, Path]:
    source_path = _resolve_input(path, expected_type)
    raw = _read_bytes(source_path, expected_type)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BlindedReviewError(f"invalid JSON {source_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BlindedReviewError(f"{expected_type} must be a JSON object")
    _expect_type(payload, expected_type, expected_type)
    return payload, hashlib.sha256(raw).hexdigest(), source_path


def _expect_type(payload: dict[str, Any], expected_type: str, label: str) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise BlindedReviewError(f"{label} schema_version must be {SCHEMA_VERSION}")
    if payload.get("artifact_type") != expected_type:
        raise BlindedReviewError(f"{label} artifact_type must be {expected_type}")


def _prepare_output(path: Path, protected: Iterable[Path]) -> Path:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if absolute.suffix.casefold() != ".json":
        raise BlindedReviewError(f"output path must use the .json extension: {absolute}")
    _reject_symlink_components(absolute)
    protected_resolved = {item.expanduser().resolve() for item in protected}
    normalized = absolute.parent.resolve() / absolute.name
    if normalized in protected_resolved:
        raise BlindedReviewError(f"output path collides with an input: {normalized}")
    try:
        metadata = absolute.lstat()
    except FileNotFoundError:
        metadata = None
    except OSError as exc:
        raise BlindedReviewError(f"could not inspect output path {absolute}: {exc}") from exc
    if metadata is not None:
        kind = "symlink" if stat.S_ISLNK(metadata.st_mode) else "existing path"
        raise BlindedReviewError(f"refusing to overwrite {kind}: {absolute}")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(absolute.parent)
    return absolute.parent.resolve() / absolute.name


def _load_build_seed(seed_file: Path | None) -> str:
    if seed_file is None:
        return _generate_review_seed()
    return _read_private_seed_file(seed_file)


def _generate_review_seed() -> str:
    return secrets.token_hex(32)


def _read_private_seed_file(path: Path) -> str:
    expanded = path.expanduser()
    absolute = Path(os.path.abspath(os.fspath(expanded)))
    _reject_symlink_components(absolute, label="seed file path")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise BlindedReviewError("this platform cannot securely read a private seed file")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise BlindedReviewError(f"could not securely open seed file {absolute}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BlindedReviewError(f"seed file is not a regular file: {absolute}")
        if stat.S_IMODE(before.st_mode) != 0o600:
            raise BlindedReviewError(
                f"seed file must have mode 0600: {absolute} "
                f"(mode {stat.S_IMODE(before.st_mode):04o})"
            )
        if before.st_uid != os.geteuid():
            raise BlindedReviewError(f"seed file must be owned by the current user: {absolute}")
        if before.st_nlink != 1:
            raise BlindedReviewError(f"seed file must have exactly one hard link: {absolute}")
        if before.st_size > 4096:
            raise BlindedReviewError("seed file must contain at most 4096 bytes")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_nlink", "st_size")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise BlindedReviewError(f"seed file changed while it was being read: {absolute}")
        for field in ("st_mtime_ns", "st_ctime_ns"):
            if getattr(before, field, None) != getattr(after, field, None):
                raise BlindedReviewError(f"seed file changed while it was being read: {absolute}")
    finally:
        os.close(descriptor)
    if len(raw) > 4096:
        raise BlindedReviewError("seed file must contain at most 4096 bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BlindedReviewError(f"seed file is not valid UTF-8: {absolute}") from exc
    seed = text.rstrip("\r\n")
    if not seed or "\n" in seed or "\r" in seed:
        raise BlindedReviewError("seed file must contain exactly one non-empty line")
    if len(seed.encode("utf-8")) < 32:
        raise BlindedReviewError("seed file value must contain at least 32 UTF-8 bytes")
    return seed


def _reject_symlink_components(path: Path, *, label: str = "output path") -> None:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise BlindedReviewError(f"could not inspect path component {current}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise BlindedReviewError(f"{label} must not traverse a symlink: {current}")


def _atomic_write_json(path: Path, payload: dict[str, Any], *, mode: int) -> None:
    data = _json_bytes(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _resolve_input(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BlindedReviewError(f"{label} is not a regular file: {resolved}")
    return resolved


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise BlindedReviewError(f"could not read {label} {path}: {exc}") from exc


def _required_list(payload: dict[str, Any], key: str, field: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise BlindedReviewError(f"{field}.{key} must be a list")
    return value


def _required_string(payload: dict[str, Any], key: str, field: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BlindedReviewError(f"{field}.{key} must be a non-empty string")
    return value.strip()


def _required_bool(payload: dict[str, Any], key: str, field: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise BlindedReviewError(f"{field}.{key} must be boolean")
    return value


def _required_sha256(payload: dict[str, Any], key: str, field: str) -> str:
    value = _required_string(payload, key, field).casefold()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise BlindedReviewError(f"{field}.{key} must be a SHA-256 hex digest")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise BlindedReviewError(f"{field} must be a list")
    strings = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(strings) != len(value):
        raise BlindedReviewError(f"{field} must contain only non-empty strings")
    return strings


def _neutral_label(index: int) -> str:
    value = index
    letters = ""
    while True:
        value, remainder = divmod(value, 26)
        letters = chr(ord("A") + remainder) + letters
        if value == 0:
            break
        value -= 1
    return f"Model {letters}"


def _derived_seed(seed: str, scope: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{scope}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
