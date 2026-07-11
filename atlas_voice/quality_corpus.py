from __future__ import annotations

import json
from pathlib import Path
from typing import Any


QUALITY_SCHEMA_VERSION = "1.0.0"
DEFAULT_ABSTENTION_PHRASES = [
    "not provided in the evidence",
    "cannot determine from the evidence",
    "cannot be established from the evidence",
    "the evidence does not say",
    "unknown from the evidence",
    "no decision has been made",
    "no winner is available",
]


def decode_quality_jsonl(raw: bytes, source_path: Path) -> dict[str, Any]:
    """Convert the versioned quality JSONL contract to the runner's corpus shape."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"quality corpus is not valid UTF-8: {source_path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid quality JSONL at {source_path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"quality JSONL row {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"quality JSONL contains no cases: {source_path}")

    dataset_versions = {_string(row, "dataset_version") for row in rows}
    if len(dataset_versions) != 1:
        raise ValueError(f"quality JSONL mixes dataset versions: {sorted(dataset_versions)}")
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(rows, start=1):
        if row.get("schema_version") != QUALITY_SCHEMA_VERSION:
            raise ValueError(
                f"quality row {row_index} schema_version must be {QUALITY_SCHEMA_VERSION}"
            )
        case_id = _string(row, "id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate quality case ID: {case_id}")
        seen_ids.add(case_id)
        category = _string(row, "category")
        packet = row.get("evidence_packet")
        if not isinstance(packet, dict):
            raise ValueError(f"{case_id}.evidence_packet must be an object")
        if packet.get("closed_world") is not True:
            raise ValueError(f"{case_id}.evidence_packet.closed_world must be true")
        turns = packet.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{case_id}.evidence_packet.turns must be non-empty")

        evidence = [
            f"Recording: {_string(packet, 'title')} ({_string(packet, 'recording_id')}).",
            f"Participants: {', '.join(_strings(packet.get('participants'), f'{case_id}.participants'))}.",
            f"Time range: {_string(packet, 'time_range')}.",
            f"Overview: {_string(packet, 'overview')}",
        ]
        turn_ids: set[str] = set()
        for turn_index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                raise ValueError(f"{case_id}.turns[{turn_index}] must be an object")
            turn_id = _string(turn, "turn_id")
            if turn_id in turn_ids:
                raise ValueError(f"{case_id} contains duplicate turn ID {turn_id}")
            turn_ids.add(turn_id)
            evidence.append(
                f"[{turn_id}] Window {_string(turn, 'window')}; "
                f"{_string(turn, 'timestamp')}; {_string(turn, 'speaker')}: "
                f"{_string(turn, 'text')}"
            )

        required_facts_raw = row.get("required_facts")
        if not isinstance(required_facts_raw, list) or not required_facts_raw:
            raise ValueError(f"{case_id}.required_facts must be non-empty")
        required_facts: list[dict[str, Any]] = []
        fact_ids: set[str] = set()
        for fact_index, fact in enumerate(required_facts_raw, start=1):
            if not isinstance(fact, dict):
                raise ValueError(f"{case_id}.required_facts[{fact_index}] must be an object")
            fact_id = _string(fact, "fact_id")
            if fact_id in fact_ids:
                raise ValueError(f"{case_id} contains duplicate fact ID {fact_id}")
            fact_ids.add(fact_id)
            value = _string(fact, "value")
            aliases = _strings(fact.get("aliases"), f"{case_id}.{fact_id}.aliases")
            support = _strings(
                fact.get("evidence_turn_ids"), f"{case_id}.{fact_id}.evidence_turn_ids"
            )
            unknown_turns = set(support) - turn_ids
            if unknown_turns:
                raise ValueError(
                    f"{case_id}.{fact_id} cites unknown turns: {sorted(unknown_turns)}"
                )
            required_facts.append(
                {
                    "id": fact_id,
                    "any_of": list(dict.fromkeys([value, *aliases])),
                    "evidence_turn_ids": support,
                }
            )

        forbidden = _strings(row.get("forbidden_claims"), f"{case_id}.forbidden_claims")
        attribution_targets = _targets(
            row.get("attribution_targets"),
            case_id,
            fact_ids,
            value_key="speaker",
        )
        timestamp_targets = _targets(
            row.get("timestamp_targets"),
            case_id,
            fact_ids,
            value_key="timestamp",
        )
        abstention_required = row.get("abstention_required")
        if not isinstance(abstention_required, bool):
            raise ValueError(f"{case_id}.abstention_required must be boolean")
        cases.append(
            {
                "id": case_id,
                "category": category,
                "evidence": evidence,
                "question": _string(row, "question"),
                "required_facts": required_facts,
                "forbidden_claims": [
                    {"id": f"forbidden-{index:02d}", "any_of": [claim]}
                    for index, claim in enumerate(forbidden, start=1)
                ],
                "should_abstain": abstention_required,
                "attribution_targets": attribution_targets,
                "timestamp_targets": timestamp_targets,
                "ordered_fact_ids": (
                    [fact["id"] for fact in required_facts] if category == "temporal_order" else []
                ),
                "gold_rationale": _string(row, "gold_rationale"),
                "source_metadata": {
                    "recording_id": _string(packet, "recording_id"),
                    "title": _string(packet, "title"),
                    "participants": _strings(packet.get("participants"), f"{case_id}.participants"),
                    "time_range": _string(packet, "time_range"),
                    "closed_world": True,
                },
            }
        )
    dataset_version = dataset_versions.pop()
    return {
        "schema_version": 1,
        "corpus_version": dataset_version,
        "description": (
            f"Versioned closed-world voice-model quality corpus {dataset_version}; "
            f"{len(cases)} cases loaded from JSONL."
        ),
        "abstention_phrases": DEFAULT_ABSTENTION_PHRASES,
        "cases": cases,
    }


def _targets(
    value: Any,
    case_id: str,
    fact_ids: set[str],
    *,
    value_key: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{case_id}.{value_key}_targets must be a list")
    targets: list[dict[str, str]] = []
    for target_index, target in enumerate(value, start=1):
        if not isinstance(target, dict):
            raise ValueError(f"{case_id}.{value_key}_targets[{target_index}] must be an object")
        fact_id = _string(target, "fact_id")
        if fact_id not in fact_ids:
            raise ValueError(f"{case_id}.{value_key}_targets cites unknown fact ID {fact_id}")
        targets.append({"fact_id": fact_id, "value": _string(target, value_key)})
    return targets


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    strings = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(strings) != len(value):
        raise ValueError(f"{field} must contain only non-empty strings")
    return strings
