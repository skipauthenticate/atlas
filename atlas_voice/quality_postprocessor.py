from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence
import unicodedata

from atlas_voice.quality_corpus import DEFAULT_ABSTENTION_PHRASES


SCHEMA_VERSION = "1.0.0"
SCORER_VERSION = "voice_model_quality_lexical_v1"
TEMPORAL_CATEGORY = "temporal_order"
NEGATION_TOKENS = frozenset({"cannot", "isnt", "never", "no", "not", "wasnt", "without"})


class QualityPostprocessError(ValueError):
    """Raised when a corpus or completed benchmark report is incompatible."""


@dataclass(frozen=True)
class FactSpec:
    fact_id: str
    value: str
    aliases: tuple[str, ...]

    @property
    def variants(self) -> tuple[tuple[str, str], ...]:
        return (("value", self.value), *(("alias", item) for item in self.aliases))


@dataclass(frozen=True)
class TargetSpec:
    fact_id: str
    value: str


@dataclass(frozen=True)
class QualityCase:
    prompt_id: str
    category: str
    required_facts: tuple[FactSpec, ...]
    forbidden_claims: tuple[str, ...]
    attribution_targets: tuple[TargetSpec, ...]
    timestamp_targets: tuple[TargetSpec, ...]
    abstention_required: bool


@dataclass(frozen=True)
class QualityCorpus:
    path: Path
    sha256: str
    dataset_version: str
    cases: tuple[QualityCase, ...]


def load_quality_corpus(path: Path) -> QualityCorpus:
    source_path = path.expanduser().resolve()
    raw = _read_bytes(source_path, "quality corpus")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualityPostprocessError(f"quality corpus is not UTF-8: {source_path}") from exc

    cases: list[QualityCase] = []
    seen_ids: set[str] = set()
    dataset_versions: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QualityPostprocessError(
                f"invalid JSON at {source_path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise QualityPostprocessError(f"quality corpus row {line_number} must be an object")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise QualityPostprocessError(
                f"quality corpus row {line_number} schema_version must be {SCHEMA_VERSION}"
            )
        dataset_versions.add(_required_string(row, "dataset_version", line_number))
        prompt_id = _required_string(row, "id", line_number)
        if prompt_id in seen_ids:
            raise QualityPostprocessError(f"duplicate corpus prompt ID: {prompt_id}")
        seen_ids.add(prompt_id)
        category = _required_string(row, "category", line_number)
        evidence_packet = row.get("evidence_packet")
        if not isinstance(evidence_packet, dict) or evidence_packet.get("closed_world") is not True:
            raise QualityPostprocessError(f"{prompt_id}.evidence_packet.closed_world must be true")
        abstention_required = row.get("abstention_required")
        if not isinstance(abstention_required, bool):
            raise QualityPostprocessError(f"{prompt_id}.abstention_required must be boolean")

        required_facts = _load_required_facts(row.get("required_facts"), prompt_id)
        fact_ids = {item.fact_id for item in required_facts}
        forbidden_claims = _string_list(
            row.get("forbidden_claims"), f"{prompt_id}.forbidden_claims"
        )
        attribution_targets = _load_targets(
            row.get("attribution_targets"), prompt_id, "speaker", fact_ids
        )
        timestamp_targets = _load_targets(
            row.get("timestamp_targets"), prompt_id, "timestamp", fact_ids
        )
        cases.append(
            QualityCase(
                prompt_id=prompt_id,
                category=category,
                required_facts=required_facts,
                forbidden_claims=forbidden_claims,
                attribution_targets=attribution_targets,
                timestamp_targets=timestamp_targets,
                abstention_required=abstention_required,
            )
        )

    if not cases:
        raise QualityPostprocessError(f"quality corpus contains no cases: {source_path}")
    if len(dataset_versions) != 1:
        raise QualityPostprocessError(
            f"quality corpus mixes dataset versions: {sorted(dataset_versions)}"
        )
    return QualityCorpus(
        path=source_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        dataset_version=dataset_versions.pop(),
        cases=tuple(cases),
    )


def load_completed_benchmark(path: Path) -> tuple[dict[str, Any], str, Path]:
    source_path = path.expanduser().resolve()
    raw = _read_bytes(source_path, "benchmark report")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualityPostprocessError(f"invalid benchmark JSON {source_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise QualityPostprocessError("benchmark report must be a JSON object")
    if payload.get("schema_version") != 1:
        raise QualityPostprocessError("benchmark report schema_version must be 1")
    if payload.get("status") not in {"completed", "completed_with_errors"}:
        raise QualityPostprocessError(
            "benchmark report status must be completed or completed_with_errors"
        )
    return payload, hashlib.sha256(raw).hexdigest(), source_path


def postprocess_quality(
    benchmark: dict[str, Any],
    corpus: QualityCorpus,
    *,
    benchmark_path: Path | None = None,
    benchmark_sha256: str | None = None,
) -> dict[str, Any]:
    models, trials, rounds = _validate_benchmark(benchmark, corpus)
    cases_by_id = {case.prompt_id: case for case in corpus.cases}
    evaluated_trials: list[dict[str, Any]] = []
    for trial in trials:
        case = cases_by_id[str(trial["prompt_id"])]
        preserved = deepcopy(trial)
        preserved["postprocessed_quality"] = score_trial(trial, case)
        evaluated_trials.append(preserved)

    categories = list(dict.fromkeys(case.category for case in corpus.cases))
    model_summaries: list[dict[str, Any]] = []
    for model in models:
        model_id = model["model_id"]
        selected = [item for item in evaluated_trials if item["model_id"] == model_id]
        model_summaries.append(
            {
                "profile": model["profile"],
                "model_id": model_id,
                "overall": _summarize(selected),
                "by_category": [
                    {
                        "category": category,
                        **_summarize([item for item in selected if item["category"] == category]),
                    }
                    for category in categories
                ],
            }
        )

    review_trials = [
        item for item in evaluated_trials if item["postprocessed_quality"]["manual_review_required"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scorer": {
            "version": SCORER_VERSION,
            "claim_scope": (
                "Deterministic normalized value/alias matching over closed-world synthetic "
                "answers. Rates are lexical grading signals, not semantic accuracy claims."
            ),
            "explicit_mention_limitation": (
                "Attribution and timestamp rates require explicit speaker/time mentions and "
                "may understate correct concise answers when the question did not request a "
                "citation."
            ),
            "normalization": (
                "Unicode NFKD, case folding, punctuation/underscore folding, numeric grouping "
                "removal, and leading-zero normalization; matching uses whole token sequences."
            ),
            "temporal_order_rule": (
                "All declared required facts must match and their earliest alias positions must "
                "be strictly increasing in required_facts order."
            ),
        },
        "inputs": {
            "benchmark": {
                "path": str(benchmark_path.resolve()) if benchmark_path else None,
                "sha256": benchmark_sha256,
                "status": benchmark["status"],
            },
            "corpus": {
                "path": str(corpus.path),
                "sha256": corpus.sha256,
                "dataset_version": corpus.dataset_version,
                "case_count": len(corpus.cases),
            },
        },
        "validation": {
            "prompt_ids_verified": True,
            "corpus_hash_verified": True,
            "trial_matrix_verified": True,
            "rounds": rounds,
            "model_count": len(models),
            "expected_trial_count": rounds * len(models) * len(corpus.cases),
            "actual_trial_count": len(trials),
        },
        "metric_definitions": _metric_definitions(),
        "model_summaries": model_summaries,
        "manual_review": {
            "trial_count": len(review_trials),
            "trial_keys": [
                {
                    "round": item["round"],
                    "profile": item["profile"],
                    "model_id": item["model_id"],
                    "prompt_id": item["prompt_id"],
                    "reason_codes": [
                        reason["code"]
                        for reason in item["postprocessed_quality"]["manual_review_reasons"]
                    ],
                }
                for item in review_trials
            ],
        },
        "trials": evaluated_trials,
    }


def score_trial(trial: dict[str, Any], case: QualityCase) -> dict[str, Any]:
    answer = trial.get("answer")
    if not isinstance(answer, str):
        raise QualityPostprocessError(f"trial {case.prompt_id} answer must be a string")
    answer_tokens = normalize_tokens(answer)
    facts_by_id = {fact.fact_id: fact for fact in case.required_facts}
    required_results = [
        _score_variants(answer_tokens, fact.fact_id, fact.variants) for fact in case.required_facts
    ]
    required_by_id = {item["fact_id"]: item for item in required_results}
    forbidden_results = [
        _score_variants(
            answer_tokens,
            f"forbidden-{index:02d}",
            (("claim", claim),),
        )
        for index, claim in enumerate(case.forbidden_claims, start=1)
    ]
    required_hits = sum(1 for item in required_results if item["hit"])
    forbidden_hits = sum(1 for item in forbidden_results if item["hit"])
    required_recall = _rate(required_hits, len(required_results))

    abstention_variants: list[tuple[str, str]] = [
        ("default_phrase", phrase) for phrase in DEFAULT_ABSTENTION_PHRASES
    ]
    if case.abstention_required:
        answer_status = next(
            (fact for fact in case.required_facts if fact.fact_id == "answer_status"), None
        )
        if answer_status is not None:
            abstention_variants.extend(answer_status.variants)
    abstention_matches = _variant_matches(answer_tokens, abstention_variants)
    abstention_detected = bool(abstention_matches)
    abstention_correct = (
        abstention_detected and forbidden_hits == 0
        if case.abstention_required
        else not abstention_detected
    )

    attribution_results = []
    for target in case.attribution_targets:
        fact_hit = bool(required_by_id[target.fact_id]["hit"])
        target_matches = _variant_matches(answer_tokens, (("speaker", target.value),))
        attribution_results.append(
            {
                "fact_id": target.fact_id,
                "target_speaker": target.value,
                "linked_fact_hit": fact_hit,
                "target_present": bool(target_matches),
                "passed": fact_hit and bool(target_matches),
                "matches": target_matches,
            }
        )

    timestamp_results = []
    for target in case.timestamp_targets:
        fact_hit = bool(required_by_id[target.fact_id]["hit"])
        variants = tuple(
            ("timestamp_alias", item)
            for item in _timestamp_aliases(target.value, facts_by_id[target.fact_id])
        )
        target_matches = _variant_matches(answer_tokens, variants)
        timestamp_results.append(
            {
                "fact_id": target.fact_id,
                "target_timestamp": target.value,
                "linked_fact_hit": fact_hit,
                "target_present": bool(target_matches),
                "passed": fact_hit and bool(target_matches),
                "aliases_checked": [item[1] for item in variants],
                "matches": target_matches,
            }
        )

    temporal_result = None
    if case.category == TEMPORAL_CATEGORY:
        declared = [fact.fact_id for fact in case.required_facts]
        positions = [required_by_id[fact_id]["first_start_token"] for fact_id in declared]
        all_hit = all(position is not None for position in positions)
        ordered = all_hit and all(
            int(left) < int(right) for left, right in zip(positions, positions[1:])
        )
        temporal_result = {
            "declared_fact_ids": declared,
            "earliest_alias_positions": positions,
            "all_required_facts_hit": all_hit,
            "passed": ordered,
        }

    if case.abstention_required:
        exact_pass = abstention_correct
    else:
        exact_pass = (
            required_hits == len(required_results)
            and forbidden_hits == 0
            and not abstention_detected
        )

    manual_review_reasons = _manual_review_reasons(
        trial=trial,
        answer_tokens=answer_tokens,
        case=case,
        required_results=required_results,
        forbidden_results=forbidden_results,
        abstention_detected=abstention_detected,
        attribution_results=attribution_results,
        timestamp_results=timestamp_results,
        temporal_result=temporal_result,
    )
    return {
        "scorer_version": SCORER_VERSION,
        "lexical_only": True,
        "required_facts": required_results,
        "required_fact_recall": required_recall,
        "forbidden_claims": forbidden_results,
        "forbidden_claim_hit_count": forbidden_hits,
        "exact_pass": exact_pass,
        "abstention": {
            "required": case.abstention_required,
            "detected": abstention_detected,
            "correct": abstention_correct,
            "matches": abstention_matches,
        },
        "attribution_targets": attribution_results,
        "timestamp_targets": timestamp_results,
        "temporal_order": temporal_result,
        "manual_review_required": bool(manual_review_reasons),
        "manual_review_reasons": manual_review_reasons,
    }


def normalize_tokens(value: str) -> tuple[str, ...]:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    without_marks = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    without_grouping = re.sub(r"(?<=\d)[,_](?=\d{3}(?:\D|$))", "", without_marks)
    raw_tokens = re.findall(r"[^\W_]+", without_grouping, flags=re.UNICODE)
    return tuple(_normalize_numeric_token(token) for token in raw_tokens)


def _score_variants(
    answer_tokens: tuple[str, ...],
    fact_id: str,
    variants: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    matches = _variant_matches(answer_tokens, variants)
    starts = [int(item["start_token"]) for item in matches]
    return {
        "fact_id": fact_id,
        "hit": bool(matches),
        "first_start_token": min(starts) if starts else None,
        "matches": matches,
    }


def _variant_matches(
    answer_tokens: tuple[str, ...], variants: Iterable[tuple[str, str]]
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen_variants: set[tuple[str, ...]] = set()
    for variant_kind, variant in variants:
        phrase_tokens = normalize_tokens(variant)
        if not phrase_tokens or phrase_tokens in seen_variants:
            continue
        seen_variants.add(phrase_tokens)
        for start in _find_positions(answer_tokens, phrase_tokens):
            matches.append(
                {
                    "variant_kind": variant_kind,
                    "variant": variant,
                    "normalized": " ".join(phrase_tokens),
                    "start_token": start,
                    "end_token": start + len(phrase_tokens),
                }
            )
    matches.sort(
        key=lambda item: (
            int(item["start_token"]),
            int(item["end_token"]),
            str(item["normalized"]),
        )
    )
    return matches


def _find_positions(answer_tokens: tuple[str, ...], phrase_tokens: tuple[str, ...]) -> list[int]:
    width = len(phrase_tokens)
    return [
        index
        for index in range(len(answer_tokens) - width + 1)
        if answer_tokens[index : index + width] == phrase_tokens
    ]


def _timestamp_aliases(value: str, linked_fact: FactSpec) -> tuple[str, ...]:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return (value,)

    month = parsed.strftime("%B")
    month_short = parsed.strftime("%b")
    date_aliases = [
        f"{parsed.year:04d}-{parsed.month:02d}-{parsed.day:02d}",
        f"{parsed.month}/{parsed.day}/{parsed.year}",
        f"{month} {parsed.day}, {parsed.year}",
        f"{month} {parsed.day} {parsed.year}",
        f"{month_short} {parsed.day}, {parsed.year}",
        f"{month} {parsed.day}",
        f"{month_short} {parsed.day}",
    ]
    if "T" not in value:
        return tuple(dict.fromkeys([value, *date_aliases]))

    hour_12 = parsed.hour % 12 or 12
    meridiem = "AM" if parsed.hour < 12 else "PM"
    time_aliases = [
        f"{parsed.hour:02d}:{parsed.minute:02d}",
        f"{parsed.hour}:{parsed.minute:02d}",
        f"{hour_12}:{parsed.minute:02d} {meridiem}",
        f"{hour_12}:{parsed.minute:02d}{meridiem}",
    ]
    linked_variants = [item[1] for item in linked_fact.variants]
    date_is_part_of_fact = _any_variant_contains(linked_variants, date_aliases)
    time_is_part_of_fact = _any_variant_contains(linked_variants, time_aliases)
    if time_is_part_of_fact:
        selected = time_aliases
    elif date_is_part_of_fact:
        selected = date_aliases
    else:
        selected = [
            f"{month} {parsed.day} {parsed.hour:02d}:{parsed.minute:02d}",
            f"{month} {parsed.day} at {parsed.hour:02d}:{parsed.minute:02d}",
            (
                f"{parsed.year:04d}-{parsed.month:02d}-{parsed.day:02d} "
                f"{parsed.hour:02d}:{parsed.minute:02d}"
            ),
        ]
    return tuple(dict.fromkeys([value, *selected]))


def _any_variant_contains(variants: Sequence[str], candidates: Sequence[str]) -> bool:
    for variant in variants:
        variant_tokens = normalize_tokens(variant)
        for candidate in candidates:
            candidate_tokens = normalize_tokens(candidate)
            if candidate_tokens and _find_positions(variant_tokens, candidate_tokens):
                return True
    return False


def _manual_review_reasons(
    *,
    trial: dict[str, Any],
    answer_tokens: tuple[str, ...],
    case: QualityCase,
    required_results: Sequence[dict[str, Any]],
    forbidden_results: Sequence[dict[str, Any]],
    abstention_detected: bool,
    attribution_results: Sequence[dict[str, Any]],
    timestamp_results: Sequence[dict[str, Any]],
    temporal_result: dict[str, Any] | None,
) -> list[dict[str, str]]:
    reasons: dict[str, str] = {}

    def add(code: str, explanation: str) -> None:
        reasons.setdefault(code, explanation)

    if trial.get("error"):
        add("trial_error", "The source benchmark recorded a trial error; quality was not scored.")
    if not answer_tokens:
        add("empty_answer", "The answer is empty after normalization.")
    required_hits = sum(bool(item["hit"]) for item in required_results)
    forbidden_hits = sum(bool(item["hit"]) for item in forbidden_results)
    if 0 < required_hits < len(required_results):
        add(
            "partial_required_fact_match",
            "Only some required value/alias rules matched; unmatched paraphrases need review.",
        )
    if answer_tokens and required_hits == 0 and not abstention_detected:
        add(
            "no_required_or_abstention_match",
            "No required value/alias or explicit abstention phrase matched the non-empty answer.",
        )
    if required_hits and forbidden_hits:
        add(
            "required_and_forbidden_match",
            "Both required and forbidden lexical rules matched; assertion scope may be ambiguous.",
        )
    if abstention_detected and forbidden_hits:
        add(
            "abstention_with_forbidden_match",
            "An abstention phrase and a forbidden claim both matched.",
        )
    if case.abstention_required and not abstention_detected:
        add(
            "abstention_not_detected",
            "No explicit abstention phrase or answer_status alias matched.",
        )
    if trial.get("finish_reason") == "length":
        add("truncated_answer", "The source trial stopped at the output-token limit.")

    all_rule_results = [*required_results, *forbidden_results]
    for result in all_rule_results:
        for match in result["matches"]:
            start = int(match["start_token"])
            if any(token in NEGATION_TOKENS for token in answer_tokens[max(0, start - 3) : start]):
                add(
                    "possible_negated_lexical_match",
                    (
                        "A matched phrase has a nearby preceding negation; lexical polarity "
                        "may be wrong."
                    ),
                )
                break

    if any(item["linked_fact_hit"] and not item["target_present"] for item in attribution_results):
        add(
            "attribution_target_implicit_or_missing",
            (
                "A linked fact matched without the target speaker name; implicit attribution "
                "needs review."
            ),
        )
    if any(item["linked_fact_hit"] and not item["target_present"] for item in timestamp_results):
        add(
            "timestamp_target_implicit_or_missing",
            "A linked fact matched without a recognized target timestamp form.",
        )
    if temporal_result is not None:
        duplicate_fact_mentions = any(
            len({match["start_token"] for match in item["matches"]}) > 1
            for item in required_results
        )
        if duplicate_fact_mentions:
            add(
                "multiple_temporal_fact_matches",
                (
                    "A temporal fact matched more than once; earliest-position ordering is "
                    "conservative."
                ),
            )
        if temporal_result["all_required_facts_hit"] and not temporal_result["passed"]:
            add(
                "temporal_order_mismatch",
                (
                    "All temporal facts matched, but earliest alias positions were not in "
                    "declared order."
                ),
            )
    return [{"code": code, "explanation": explanation} for code, explanation in reasons.items()]


def _summarize(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in trials if not item.get("error")]
    errors = len(trials) - len(successful)
    quality = [item["postprocessed_quality"] for item in successful]
    required_targets = sum(len(item["required_facts"]) for item in quality)
    required_hits = sum(
        sum(bool(fact["hit"]) for fact in item["required_facts"]) for item in quality
    )
    forbidden_targets = sum(len(item["forbidden_claims"]) for item in quality)
    forbidden_hits = sum(
        sum(bool(claim["hit"]) for claim in item["forbidden_claims"]) for item in quality
    )
    forbidden_trial_hits = sum(bool(item["forbidden_claim_hit_count"]) for item in quality)
    exact_passes = sum(bool(item["exact_pass"]) for item in quality)
    abstention_items = [item for item in quality if item["abstention"]["required"]]
    answerable_items = [item for item in quality if not item["abstention"]["required"]]
    attribution = [target for item in quality for target in item["attribution_targets"]]
    timestamp = [target for item in quality for target in item["timestamp_targets"]]
    temporal = [item["temporal_order"] for item in quality if item["temporal_order"] is not None]
    review_count = sum(bool(item["manual_review_required"]) for item in quality)
    return {
        "source_trial_count": len(trials),
        "scored_trial_count": len(successful),
        "error_trial_count": errors,
        "required_fact_recall": _metric(required_hits, required_targets, "hit_count"),
        "forbidden_claim_hit_rate": _metric(forbidden_hits, forbidden_targets, "hit_count"),
        "forbidden_claim_trial_rate": _metric(
            forbidden_trial_hits, len(successful), "trial_hit_count"
        ),
        "exact_pass_rate": _metric(exact_passes, len(successful), "pass_count"),
        "abstention_accuracy": _metric(
            sum(bool(item["abstention"]["correct"]) for item in abstention_items),
            len(abstention_items),
            "pass_count",
        ),
        "unwarranted_abstention_rate": _metric(
            sum(bool(item["abstention"]["detected"]) for item in answerable_items),
            len(answerable_items),
            "hit_count",
        ),
        "attribution_target_accuracy": _metric(
            sum(bool(item["passed"]) for item in attribution),
            len(attribution),
            "pass_count",
        ),
        "timestamp_target_accuracy": _metric(
            sum(bool(item["passed"]) for item in timestamp),
            len(timestamp),
            "pass_count",
        ),
        "temporal_order_accuracy": _metric(
            sum(bool(item["passed"]) for item in temporal),
            len(temporal),
            "pass_count",
        ),
        "manual_review_rate": _metric(review_count, len(successful), "flagged_count"),
    }


def _metric(numerator: int, denominator: int, numerator_name: str) -> dict[str, Any]:
    return {
        numerator_name: numerator,
        "target_count": denominator,
        "rate": _rate(numerator, denominator),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _validate_benchmark(
    benchmark: dict[str, Any], corpus: QualityCorpus
) -> tuple[list[dict[str, str]], list[dict[str, Any]], int]:
    if benchmark.get("schema_version") != 1:
        raise QualityPostprocessError("benchmark report schema_version must be 1")
    if benchmark.get("status") not in {"completed", "completed_with_errors"}:
        raise QualityPostprocessError(
            "benchmark report status must be completed or completed_with_errors"
        )
    corpus_meta = benchmark.get("corpus")
    if not isinstance(corpus_meta, dict):
        raise QualityPostprocessError("benchmark corpus metadata must be an object")
    if corpus_meta.get("sha256") != corpus.sha256:
        raise QualityPostprocessError("benchmark corpus SHA-256 does not match the supplied corpus")
    if corpus_meta.get("corpus_version") != corpus.dataset_version:
        raise QualityPostprocessError(
            "benchmark corpus_version does not match the supplied corpus dataset_version"
        )
    if corpus_meta.get("case_count") != len(corpus.cases):
        raise QualityPostprocessError("benchmark corpus case_count does not match the corpus")

    protocol = benchmark.get("protocol")
    rounds = protocol.get("rounds") if isinstance(protocol, dict) else None
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise QualityPostprocessError("benchmark protocol.rounds must be a positive integer")

    inventory = benchmark.get("inventory")
    raw_models = inventory.get("models") if isinstance(inventory, dict) else None
    if not isinstance(raw_models, list) or not raw_models:
        raise QualityPostprocessError("benchmark inventory.models must be a non-empty list")
    models: list[dict[str, str]] = []
    model_ids: set[str] = set()
    profiles: set[str] = set()
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise QualityPostprocessError(f"inventory.models[{index}] must be an object")
        model_id = raw_model.get("model_id")
        profile = raw_model.get("profile")
        if not isinstance(model_id, str) or not model_id:
            raise QualityPostprocessError(f"inventory.models[{index}].model_id is invalid")
        if not isinstance(profile, str) or not profile:
            raise QualityPostprocessError(f"inventory.models[{index}].profile is invalid")
        if model_id in model_ids or profile in profiles:
            raise QualityPostprocessError("benchmark inventory contains duplicate model/profile")
        model_ids.add(model_id)
        profiles.add(profile)
        models.append({"model_id": model_id, "profile": profile})

    raw_trials = benchmark.get("trials")
    if not isinstance(raw_trials, list):
        raise QualityPostprocessError("benchmark trials must be a list")
    cases_by_id = {case.prompt_id: case for case in corpus.cases}
    model_by_id = {item["model_id"]: item for item in models}
    expected = {
        (item["model_id"], round_index, case.prompt_id)
        for item in models
        for round_index in range(1, rounds + 1)
        for case in corpus.cases
    }
    actual: set[tuple[str, int, str]] = set()
    trials: list[dict[str, Any]] = []
    for index, trial in enumerate(raw_trials):
        if not isinstance(trial, dict):
            raise QualityPostprocessError(f"trials[{index}] must be an object")
        model_id = trial.get("model_id")
        prompt_id = trial.get("prompt_id")
        round_index = trial.get("round")
        profile = trial.get("profile")
        if model_id not in model_by_id:
            raise QualityPostprocessError(f"trial uses unknown model_id: {model_id!r}")
        if prompt_id not in cases_by_id:
            raise QualityPostprocessError(f"trial uses unknown prompt_id: {prompt_id!r}")
        if not isinstance(round_index, int) or isinstance(round_index, bool):
            raise QualityPostprocessError(f"trial {index} round must be an integer")
        key = (str(model_id), round_index, str(prompt_id))
        if key in actual:
            raise QualityPostprocessError(f"duplicate benchmark trial key: {key}")
        actual.add(key)
        if profile != model_by_id[str(model_id)]["profile"]:
            raise QualityPostprocessError(f"trial profile mismatch for model {model_id}")
        if trial.get("category") != cases_by_id[str(prompt_id)].category:
            raise QualityPostprocessError(f"trial category mismatch for prompt {prompt_id}")
        if trial.get("excluded_from_summary") is not False:
            raise QualityPostprocessError(
                f"scored trial {model_id}/{round_index}/{prompt_id} must not be excluded"
            )
        if not isinstance(trial.get("answer"), str):
            raise QualityPostprocessError(
                f"trial {model_id}/{round_index}/{prompt_id} answer must be a string"
            )
        trials.append(trial)
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise QualityPostprocessError(
            f"benchmark prompt/model/round matrix mismatch; missing={missing}, extra={extra}"
        )
    return models, trials, rounds


def _metric_definitions() -> dict[str, str]:
    return {
        "required_fact_recall": (
            "Matched required canonical values or declared aliases divided by required facts."
        ),
        "forbidden_claim_hit_rate": (
            "Matched normalized forbidden-claim strings divided by declared forbidden claims; "
            "a lexical hit does not prove asserted semantics."
        ),
        "exact_pass_rate": (
            "For answerable cases: every required fact, no forbidden claim, and no abstention. "
            "For abstention cases: explicit abstention with no forbidden claim."
        ),
        "abstention_accuracy": (
            "Explicit abstention with no forbidden claim among abstention-required trials only."
        ),
        "attribution_target_accuracy": (
            "A target passes only when its linked required fact matches and the target speaker "
            "name is explicitly present. This can understate correct concise answers when the "
            "question did not request attribution."
        ),
        "timestamp_target_accuracy": (
            "A target passes only when its linked required fact matches and an exact or derived "
            "ISO/date/time target form is explicitly present. This can understate correct "
            "concise answers when the question did not request a timestamp."
        ),
        "temporal_order_accuracy": (
            "Among temporal_order trials, every required fact must match and earliest matched "
            "alias positions must follow declared required_facts order."
        ),
    }


def _load_required_facts(value: Any, prompt_id: str) -> tuple[FactSpec, ...]:
    if not isinstance(value, list) or not value:
        raise QualityPostprocessError(f"{prompt_id}.required_facts must be non-empty")
    results: list[FactSpec] = []
    ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise QualityPostprocessError(f"{prompt_id}.required_facts[{index}] must be an object")
        fact_id = item.get("fact_id")
        fact_value = item.get("value")
        if not isinstance(fact_id, str) or not fact_id:
            raise QualityPostprocessError(f"{prompt_id}.required_facts[{index}].fact_id invalid")
        if fact_id in ids:
            raise QualityPostprocessError(f"duplicate fact ID {fact_id!r} in {prompt_id}")
        ids.add(fact_id)
        if not isinstance(fact_value, str) or not fact_value.strip():
            raise QualityPostprocessError(f"{prompt_id}.{fact_id}.value must be non-empty")
        aliases = _string_list(item.get("aliases"), f"{prompt_id}.{fact_id}.aliases")
        results.append(FactSpec(fact_id, fact_value.strip(), aliases))
    return tuple(results)


def _load_targets(
    value: Any,
    prompt_id: str,
    value_key: str,
    fact_ids: set[str],
) -> tuple[TargetSpec, ...]:
    if not isinstance(value, list):
        raise QualityPostprocessError(f"{prompt_id}.{value_key}_targets must be a list")
    results: list[TargetSpec] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise QualityPostprocessError(
                f"{prompt_id}.{value_key}_targets[{index}] must be an object"
            )
        fact_id = item.get("fact_id")
        target_value = item.get(value_key)
        if fact_id not in fact_ids:
            raise QualityPostprocessError(
                f"{prompt_id}.{value_key}_targets[{index}] cites unknown fact {fact_id!r}"
            )
        if not isinstance(target_value, str) or not target_value.strip():
            raise QualityPostprocessError(
                f"{prompt_id}.{value_key}_targets[{index}].{value_key} is invalid"
            )
        results.append(TargetSpec(str(fact_id), target_value.strip()))
    return tuple(results)


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise QualityPostprocessError(f"{field} must be a list")
    strings = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(strings) != len(value):
        raise QualityPostprocessError(f"{field} must contain only non-empty strings")
    return strings


def _required_string(payload: dict[str, Any], key: str, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise QualityPostprocessError(f"row {line_number} {key} must be non-empty")
    return value.strip()


def _normalize_numeric_token(token: str) -> str:
    if token.isdecimal():
        return str(int(token))
    return token


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise QualityPostprocessError(f"could not read {label} {path}: {exc}") from exc


def write_quality_report(path: Path, report: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess a completed voice-model benchmark with deterministic lexical quality "
            "metrics."
        )
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        benchmark, benchmark_sha256, benchmark_path = load_completed_benchmark(args.benchmark)
        corpus = load_quality_corpus(args.corpus)
        if args.output.expanduser().resolve() in {benchmark_path, corpus.path}:
            raise QualityPostprocessError("output path must not overwrite an input")
        report = postprocess_quality(
            benchmark,
            corpus,
            benchmark_path=benchmark_path,
            benchmark_sha256=benchmark_sha256,
        )
        write_quality_report(args.output, report)
    except (QualityPostprocessError, OSError) as exc:
        print(f"Quality postprocessing failed: {exc}", file=sys.stderr)
        return 2
    print(f"Quality report: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
