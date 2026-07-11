from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import threading
import time
from typing import Any, Protocol, Sequence
import unicodedata
from urllib.parse import urlsplit

import httpx

from atlas_voice.quality_corpus import decode_quality_jsonl


SCHEMA_VERSION = 1
DEFAULT_ROUTER_URL = "http://127.0.0.1:18080"
DEFAULT_API_KEY_ENV = "ATLAS_VOICE_BENCHMARK_API_KEY"
DEFAULT_SYSTEM_PROMPT = (
    "Answer only from the supplied evidence packet. Preserve exact names, identifiers, "
    "dates, times, quantities, decisions, and negations. If the evidence does not answer "
    'the question, reply exactly: "Not provided in the evidence." Keep the answer concise.'
)
DEFAULT_WARMUP_PROMPT = "Reply with exactly READY."
DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "voice_model_smoke_v1.json"
)


class BenchmarkError(RuntimeError):
    """Base class for benchmark failures that should produce a non-zero exit."""


class InputValidationError(BenchmarkError):
    pass


class ArtifactVerificationError(BenchmarkError):
    pass


class RouteVerificationError(BenchmarkError):
    pass


class RouterError(BenchmarkError):
    pass


@dataclass(frozen=True)
class ModelArtifact:
    profile: str
    model_id: str
    path: Path
    sha256: str
    size_bytes: int | None = None
    source_url: str | None = None
    source_revision: str | None = None
    license: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "model_id": self.model_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_url": self.source_url,
            "source_revision": self.source_revision,
            "license": self.license,
        }


@dataclass(frozen=True)
class Inventory:
    source_path: Path
    source_sha256: str
    inventory_version: str
    models: tuple[ModelArtifact, ...]


@dataclass(frozen=True)
class FactRule:
    fact_id: str
    any_of: tuple[str, ...]


@dataclass(frozen=True)
class CorpusCase:
    prompt_id: str
    category: str
    evidence: tuple[str, ...]
    question: str
    required_facts: tuple[FactRule, ...]
    forbidden_claims: tuple[FactRule, ...]
    should_abstain: bool

    def messages(self) -> list[dict[str, str]]:
        evidence = "\n".join(f"- {item}" for item in self.evidence)
        return [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"EVIDENCE PACKET:\n{evidence}\n\nQUESTION:\n{self.question}\n\n"
                    "Respond in one concise sentence."
                ),
            },
        ]


@dataclass(frozen=True)
class Corpus:
    source_path: Path
    source_sha256: str
    corpus_version: str
    description: str
    abstention_phrases: tuple[str, ...]
    cases: tuple[CorpusCase, ...]


@dataclass(frozen=True)
class CompletionMeasurement:
    answer: str
    served_model: str
    ttft_ms: float | None
    total_latency_ms: float
    output_tokens: int | None
    output_tokens_source: str
    tokens_per_second: float | None
    server_tokens_per_second: float | None
    prompt_tokens: int | None
    finish_reason: str | None


class Sampler(Protocol):
    def start(self) -> None: ...

    def capture(self) -> None: ...

    def stop(self) -> None: ...

    def window(self, started: float, ended: float) -> dict[str, float | None]: ...

    @property
    def samples(self) -> list[dict[str, Any]]: ...


class NullSampler:
    @property
    def samples(self) -> list[dict[str, Any]]:
        return []

    def start(self) -> None:
        return None

    def capture(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def window(self, started: float, ended: float) -> dict[str, float | None]:
        del started, ended
        return {
            "mem_available_mb_min": None,
            "swap_used_mb_max": None,
            "process_tree_rss_mb_max": None,
            "thermal_c_max": None,
        }


class ProcMemorySampler:
    """Best-effort Linux memory/thermal sampler with no Jetson-only dependency."""

    def __init__(self, *, process_pid: int | None = None, interval_seconds: float = 1.0):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        self.process_pid = process_pid
        self.interval_seconds = interval_seconds
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def samples(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._samples)

    def start(self) -> None:
        if self._thread is not None:
            return
        self.capture()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds * 2, 2.0))
        self.capture()

    def capture(self) -> None:
        sample = _read_resource_sample(self.process_pid)
        with self._lock:
            self._samples.append(sample)

    def window(self, started: float, ended: float) -> dict[str, float | None]:
        with self._lock:
            selected = [
                item
                for item in self._samples
                if started <= float(item["monotonic_seconds"]) <= ended
            ]
        return {
            "mem_available_mb_min": _extreme(selected, "mem_available_mb", min),
            "swap_used_mb_max": _extreme(selected, "swap_used_mb", max),
            "process_tree_rss_mb_max": _extreme(selected, "process_tree_rss_mb", max),
            "thermal_c_max": _extreme(selected, "thermal_c_max", max),
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.capture()


def load_inventory(path: Path) -> Inventory:
    source_path = path.expanduser().resolve()
    raw = _read_json_bytes(source_path, "inventory")
    payload = _decode_json(raw, source_path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InputValidationError(
            f"inventory schema_version must be {SCHEMA_VERSION}: {source_path}"
        )
    inventory_version = _required_string(payload, "inventory_version", source_path)
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise InputValidationError(f"inventory models must be a non-empty list: {source_path}")

    models: list[ModelArtifact] = []
    profiles: set[str] = set()
    model_ids: set[str] = set()
    for index, item in enumerate(raw_models):
        if not isinstance(item, dict):
            raise InputValidationError(f"inventory models[{index}] must be an object")
        profile = _required_string(item, "profile", source_path).strip().lower()
        model_id = _required_string(item, "model_id", source_path).strip()
        path_value = _required_string(item, "path", source_path)
        sha256 = _required_string(item, "sha256", source_path).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise InputValidationError(f"invalid SHA-256 for model {model_id!r}")
        if profile in profiles:
            raise InputValidationError(f"duplicate profile in inventory: {profile}")
        if model_id in model_ids:
            raise InputValidationError(f"duplicate model_id in inventory: {model_id}")
        profiles.add(profile)
        model_ids.add(model_id)
        artifact_path = Path(path_value).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = source_path.parent / artifact_path
        size_bytes = item.get("size_bytes")
        if size_bytes is not None and (not isinstance(size_bytes, int) or size_bytes <= 0):
            raise InputValidationError(f"size_bytes must be a positive integer for {model_id}")
        models.append(
            ModelArtifact(
                profile=profile,
                model_id=model_id,
                path=artifact_path.resolve(),
                sha256=sha256,
                size_bytes=size_bytes,
                source_url=_optional_string(item, "source_url"),
                source_revision=_optional_string(item, "source_revision"),
                license=_optional_string(item, "license"),
            )
        )
    return Inventory(
        source_path=source_path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        inventory_version=inventory_version,
        models=tuple(models),
    )


def load_corpus(path: Path) -> Corpus:
    source_path = path.expanduser().resolve()
    raw = _read_json_bytes(source_path, "corpus")
    try:
        payload = (
            decode_quality_jsonl(raw, source_path)
            if source_path.suffix.casefold() == ".jsonl"
            else _decode_json(raw, source_path)
        )
    except ValueError as exc:
        raise InputValidationError(f"invalid quality corpus {source_path}: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InputValidationError(f"corpus schema_version must be {SCHEMA_VERSION}: {source_path}")
    corpus_version = _required_string(payload, "corpus_version", source_path)
    description = _required_string(payload, "description", source_path)
    abstention_phrases = _string_list(payload.get("abstention_phrases"), "abstention_phrases")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise InputValidationError(f"corpus cases must be a non-empty list: {source_path}")

    cases: list[CorpusCase] = []
    prompt_ids: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise InputValidationError(f"corpus cases[{index}] must be an object")
        prompt_id = _required_string(item, "id", source_path)
        if prompt_id in prompt_ids:
            raise InputValidationError(f"duplicate corpus case id: {prompt_id}")
        prompt_ids.add(prompt_id)
        should_abstain = item.get("should_abstain")
        if not isinstance(should_abstain, bool):
            raise InputValidationError(f"should_abstain must be boolean for {prompt_id}")
        required = _fact_rules(item.get("required_facts"), prompt_id, "required_facts")
        if not should_abstain and not required:
            raise InputValidationError(
                f"answerable case {prompt_id} must define at least one required fact"
            )
        cases.append(
            CorpusCase(
                prompt_id=prompt_id,
                category=_required_string(item, "category", source_path),
                evidence=_string_list(item.get("evidence"), f"{prompt_id}.evidence"),
                question=_required_string(item, "question", source_path),
                required_facts=required,
                forbidden_claims=_fact_rules(
                    item.get("forbidden_claims", []), prompt_id, "forbidden_claims"
                ),
                should_abstain=should_abstain,
            )
        )
    return Corpus(
        source_path=source_path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        corpus_version=corpus_version,
        description=description,
        abstention_phrases=abstention_phrases,
        cases=tuple(cases),
    )


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(inventory: Inventory) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for model in inventory.models:
        if not model.path.is_file():
            raise ArtifactVerificationError(
                f"artifact does not exist or is not a file: {model.path}"
            )
        stat = model.path.stat()
        if model.size_bytes is not None and stat.st_size != model.size_bytes:
            raise ArtifactVerificationError(
                f"artifact size mismatch for {model.model_id}: expected {model.size_bytes}, "
                f"got {stat.st_size}"
            )
        actual_sha256 = sha256_file(model.path)
        if not _constant_time_equal(actual_sha256, model.sha256):
            raise ArtifactVerificationError(
                f"artifact SHA-256 mismatch for {model.model_id}: expected {model.sha256}, "
                f"got {actual_sha256}"
            )
        verified.append(
            {
                "profile": model.profile,
                "model_id": model.model_id,
                "path": str(model.path),
                "size_bytes": stat.st_size,
                "expected_sha256": model.sha256,
                "actual_sha256": actual_sha256,
                "verified": True,
                "stat_fingerprint": {
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "mtime_ns": stat.st_mtime_ns,
                },
            }
        )
    return verified


def score_answer(
    answer: str, case: CorpusCase, abstention_phrases: Sequence[str]
) -> dict[str, Any]:
    required_hits = [rule.fact_id for rule in case.required_facts if _matches_rule(answer, rule)]
    forbidden_hits = [rule.fact_id for rule in case.forbidden_claims if _matches_rule(answer, rule)]
    required_count = len(case.required_facts)
    required_recall = len(required_hits) / required_count if required_count else None
    abstained = any(_contains_phrase(answer, phrase) for phrase in abstention_phrases)
    if case.should_abstain and required_count and len(required_hits) == required_count:
        abstained = True
    if case.should_abstain:
        score = 1.0 if abstained and not forbidden_hits else 0.0
        passed = score == 1.0
    else:
        score = float(required_recall or 0.0)
        if abstained or forbidden_hits:
            score = 0.0
        passed = score == 1.0
    return {
        "score": round(score, 4),
        "passed": passed,
        "should_abstain": case.should_abstain,
        "abstention_detected": abstained,
        "required_fact_count": required_count,
        "required_fact_hits": required_hits,
        "required_fact_misses": [
            rule.fact_id for rule in case.required_facts if rule.fact_id not in required_hits
        ],
        "required_fact_recall": (
            round(required_recall, 4) if required_recall is not None else None
        ),
        "forbidden_claim_hits": forbidden_hits,
    }


def alternating_model_orders(
    models: Sequence[ModelArtifact], rounds: int
) -> list[tuple[ModelArtifact, ...]]:
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if not models:
        raise ValueError("models must not be empty")
    ordered = tuple(models)
    schedules: list[tuple[ModelArtifact, ...]] = []
    for round_index in range(rounds):
        rotation = (round_index // 2) % len(ordered)
        base = ordered[rotation:] + ordered[:rotation]
        schedules.append(tuple(reversed(base)) if round_index % 2 else base)
    return schedules


class RouterClient:
    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_seconds: float = 180.0,
        load_timeout_seconds: float = 900.0,
        api_key: str | None = None,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(request_timeout_seconds),
            headers=headers,
        )
        self.load_timeout_seconds = load_timeout_seconds

    def close(self) -> None:
        self._client.close()

    def list_models(self, *, reload: bool = False) -> list[dict[str, Any]]:
        response = self._client.get("/models", params={"reload": 1} if reload else None)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise RouterError("router /models response did not contain a data list")
        if not all(isinstance(item, dict) for item in models):
            raise RouterError("router /models data must contain only objects")
        return [dict(item) for item in models]

    def verify_routes(self, models: Sequence[ModelArtifact]) -> list[dict[str, Any]]:
        entries = self.list_models(reload=True)
        expected_ids = {model.model_id for model in models}
        raw_ids = [item.get("id") for item in entries]
        if not all(isinstance(item, str) and item for item in raw_ids):
            raise RouteVerificationError(
                "router model inventory contains an entry without a non-empty string ID"
            )
        id_counts = Counter(str(item) for item in raw_ids)
        actual_ids = set(id_counts)
        duplicates = sorted(model_id for model_id, count in id_counts.items() if count != 1)
        if actual_ids != expected_ids or duplicates or len(entries) != len(expected_ids):
            raise RouteVerificationError(
                "router model inventory mismatch: "
                f"expected one entry each for {sorted(expected_ids)!r}, "
                f"got counts {dict(sorted(id_counts.items()))!r}"
            )
        return [self._verify_route_from_entries(model, entries) for model in models]

    def verify_route(self, model: ModelArtifact) -> dict[str, Any]:
        return self._verify_route_from_entries(model, self.list_models())

    def snapshot_loaded_model(self) -> str | None:
        loaded = self._loaded_model_ids(self.list_models())
        if len(loaded) > 1:
            raise RouterError(
                "benchmark requires a one-model router; "
                f"found multiple loaded models: {sorted(loaded)!r}"
            )
        return next(iter(loaded), None)

    def restore_loaded_model(
        self, initial_model_id: str | None, managed_model_ids: Sequence[str]
    ) -> dict[str, Any]:
        managed = set(managed_model_ids)
        actions: list[dict[str, Any]] = []
        loaded = self._loaded_model_ids(self.list_models())
        for model_id in sorted((loaded & managed) - {initial_model_id}):
            actions.append(
                {
                    "action": "unload",
                    "model_id": model_id,
                    "seconds": round(self.unload_model(model_id), 4),
                }
            )
        loaded = self._loaded_model_ids(self.list_models())
        if initial_model_id is not None and initial_model_id not in loaded:
            actions.append(
                {
                    "action": "load",
                    "model_id": initial_model_id,
                    "seconds": round(self.load_model(initial_model_id), 4),
                }
            )
        restored = self._loaded_model_ids(self.list_models())
        expected = {initial_model_id} if initial_model_id is not None else set()
        if restored != expected:
            raise RouterError(
                "router state restoration mismatch: "
                f"expected loaded model {initial_model_id!r}, got {sorted(restored)!r}"
            )
        return {
            "initial_loaded_model": initial_model_id,
            "final_loaded_model": initial_model_id,
            "actions": actions,
            "restored": True,
        }

    @staticmethod
    def _loaded_model_ids(entries: Sequence[dict[str, Any]]) -> set[str]:
        loaded: set[str] = set()
        for entry in entries:
            model_id = entry.get("id")
            status = entry.get("status")
            status_value = status.get("value") if isinstance(status, dict) else None
            if status_value == "loaded" and isinstance(model_id, str) and model_id:
                loaded.add(model_id)
        return loaded

    def load_model(self, model_id: str) -> float:
        started = time.perf_counter()
        response = self._client.post("/models/load", json={"model": model_id})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise RouterError(f"router refused to load model {model_id}: {payload!r}")
        self._wait_for_status(model_id, {"loaded"})
        return time.perf_counter() - started

    def unload_model(self, model_id: str) -> float:
        started = time.perf_counter()
        response = self._client.post("/models/unload", json={"model": model_id})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise RouterError(f"router refused to unload model {model_id}: {payload!r}")
        self._wait_for_status(model_id, {"unloaded"})
        return time.perf_counter() - started

    def complete(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        seed: int,
    ) -> CompletionMeasurement:
        request = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": seed,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = _perf_counter()
        first_text_at: float | None = None
        last_text_at: float | None = None
        answer_parts: list[str] = []
        served_models: set[str] = set()
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        finish_reason: str | None = None
        with self._client.stream("POST", "/v1/chat/completions", json=request) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RouterError(f"invalid JSON in router SSE response: {data[:160]}") from exc
                if not isinstance(event, dict):
                    continue
                served_model = event.get("model")
                if isinstance(served_model, str) and served_model:
                    served_models.add(served_model)
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage = event_usage
                event_timings = event.get("timings")
                if isinstance(event_timings, dict):
                    timings = event_timings
                choices = event.get("choices")
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    reason = choice.get("finish_reason")
                    if isinstance(reason, str):
                        finish_reason = reason
                    delta = choice.get("delta")
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(content, str) and content:
                        received_at = _perf_counter()
                        if first_text_at is None:
                            first_text_at = received_at
                        last_text_at = received_at
                        answer_parts.append(content)
        ended = _perf_counter()
        if served_models != {model_id}:
            reported = sorted(served_models) if served_models else ["<missing>"]
            raise RouteVerificationError(
                f"completion route mismatch: requested {model_id!r}, served {reported!r}"
            )
        output_tokens = _integer_metric(usage.get("completion_tokens"))
        output_tokens_source = "usage.completion_tokens"
        if output_tokens is None:
            output_tokens = _integer_metric(timings.get("predicted_n"))
            output_tokens_source = "timings.predicted_n"
        if output_tokens is None:
            output_tokens_source = "unavailable"
        prompt_tokens = _integer_metric(usage.get("prompt_tokens"))
        server_tps = _float_metric(timings.get("predicted_per_second"))
        observed_tps = _observed_decode_tps(
            output_tokens, first_text_at=first_text_at, last_text_at=last_text_at
        )
        return CompletionMeasurement(
            answer="".join(answer_parts),
            served_model=model_id,
            ttft_ms=(first_text_at - started) * 1000 if first_text_at is not None else None,
            total_latency_ms=(ended - started) * 1000,
            output_tokens=output_tokens,
            output_tokens_source=output_tokens_source,
            tokens_per_second=observed_tps,
            server_tokens_per_second=server_tps,
            prompt_tokens=prompt_tokens,
            finish_reason=finish_reason,
        )

    def _verify_route_from_entries(
        self, model: ModelArtifact, entries: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        matches = [item for item in entries if item.get("id") == model.model_id]
        if len(matches) != 1:
            raise RouteVerificationError(
                f"router /models must contain exactly one entry for {model.model_id!r}; "
                f"found {len(matches)}"
            )
        entry = matches[0]
        status = entry.get("status")
        args = status.get("args") if isinstance(status, dict) else None
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise RouteVerificationError(
                f"router status.args must be a string list for {model.model_id!r}"
            )
        model_flags = [index for index, item in enumerate(args) if item == "--model"]
        alternate_flags = [item for item in args if item == "-m" or item.startswith("--model=")]
        if len(model_flags) != 1 or alternate_flags:
            raise RouteVerificationError(
                f"router status.args must contain exactly one separate --model flag for "
                f"{model.model_id!r}"
            )
        model_index = model_flags[0] + 1
        if model_index >= len(args) or not args[model_index]:
            raise RouteVerificationError(f"router --model flag has no path for {model.model_id!r}")
        args_path = Path(args[model_index]).expanduser()
        if not args_path.is_absolute():
            raise RouteVerificationError(
                f"router --model path must be absolute for {model.model_id!r}: {args_path}"
            )
        router_path = entry.get("path")
        if router_path is not None and (not isinstance(router_path, str) or not router_path):
            raise RouteVerificationError(f"router path field is malformed for {model.model_id!r}")
        if isinstance(router_path, str):
            reported_path = Path(router_path).expanduser()
            if not reported_path.is_absolute():
                raise RouteVerificationError(
                    f"router path must be absolute for {model.model_id!r}: {reported_path}"
                )
            if not _paths_are_same(reported_path, args_path):
                raise RouteVerificationError(
                    f"router path and status.args disagree for {model.model_id!r}: "
                    f"{reported_path} != {args_path}"
                )
        else:
            reported_path = args_path
        if not _paths_are_same(reported_path, model.path):
            raise RouteVerificationError(
                f"router path mismatch for {model.model_id}: expected {model.path}, "
                f"got {reported_path}"
            )
        status_value = status.get("value") if isinstance(status, dict) else None
        return {
            "profile": model.profile,
            "model_id": model.model_id,
            "inventory_path": str(model.path),
            "router_path": str(reported_path.resolve()),
            "router_path_source": (
                "path+status.args" if router_path is not None else "status.args"
            ),
            "router_status": status_value,
            "verified": True,
        }

    def _wait_for_status(self, model_id: str, desired: set[str]) -> None:
        deadline = time.monotonic() + self.load_timeout_seconds
        last_status: str | None = None
        while time.monotonic() < deadline:
            for entry in self.list_models():
                if entry.get("id") != model_id:
                    continue
                status = entry.get("status")
                last_status = status.get("value") if isinstance(status, dict) else None
                if last_status in desired:
                    return
                if isinstance(status, dict) and status.get("failed"):
                    raise RouterError(
                        f"model {model_id} failed while waiting for status {sorted(desired)}: "
                        f"{status!r}"
                    )
            time.sleep(0.5)
        raise RouterError(
            f"timed out waiting for {model_id} to reach {sorted(desired)}; "
            f"last status was {last_status!r}"
        )


def run_benchmark(
    inventory: Inventory,
    corpus: Corpus,
    client: Any,
    *,
    rounds: int = 2,
    warmups: int = 1,
    max_tokens: int = 96,
    seed: int = 3407,
    sampler: Sampler | None = None,
) -> dict[str, Any]:
    if warmups < 0:
        raise ValueError("warmups must not be negative")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    schedules = alternating_model_orders(inventory.models, rounds)
    artifact_verification = verify_artifacts(inventory)
    active_sampler: Sampler = sampler or NullSampler()
    trials: list[dict[str, Any]] = []
    warmup_trials: list[dict[str, Any]] = []
    load_events: list[dict[str, Any]] = []
    started_utc = _utc_now()
    started_monotonic = time.perf_counter()
    managed_model_ids = tuple(model.model_id for model in inventory.models)
    initial_loaded_model = client.snapshot_loaded_model()
    route_verification: list[dict[str, Any]] = []
    router_state: dict[str, Any] = {
        "initial_loaded_model": initial_loaded_model,
        "final_loaded_model": None,
        "actions": [],
        "restored": False,
    }
    sampler_started = False
    primary_error: BaseException | None = None
    try:
        route_verification = client.verify_routes(inventory.models)
        active_sampler.start()
        sampler_started = True
        for round_index, order in enumerate(schedules, start=1):
            for model in order:
                load_event: dict[str, Any] = {
                    "round": round_index,
                    "profile": model.profile,
                    "model_id": model.model_id,
                    "load_seconds": None,
                    "route_after_load": None,
                    "unload_seconds": None,
                    "cleanup_error": None,
                }
                load_events.append(load_event)
                model_error: BaseException | None = None
                try:
                    load_event["load_seconds"] = round(client.load_model(model.model_id), 4)
                    load_event["route_after_load"] = client.verify_route(model)
                    for warmup_index in range(1, warmups + 1):
                        warmup_trials.append(
                            _run_prompt(
                                client,
                                active_sampler,
                                model,
                                round_index=round_index,
                                prompt_id=f"warmup-{warmup_index}",
                                category="warmup",
                                messages=[{"role": "user", "content": DEFAULT_WARMUP_PROMPT}],
                                max_tokens=min(max_tokens, 16),
                                seed=seed,
                                scorer=None,
                                excluded_from_summary=True,
                            )
                        )
                    for case in corpus.cases:
                        trials.append(
                            _run_prompt(
                                client,
                                active_sampler,
                                model,
                                round_index=round_index,
                                prompt_id=case.prompt_id,
                                category=case.category,
                                messages=case.messages(),
                                max_tokens=max_tokens,
                                seed=seed,
                                scorer=lambda answer, current=case: score_answer(
                                    answer, current, corpus.abstention_phrases
                                ),
                                excluded_from_summary=False,
                            )
                        )
                except BaseException as exc:
                    model_error = exc
                    raise
                finally:
                    try:
                        load_event["unload_seconds"] = round(client.unload_model(model.model_id), 4)
                    except Exception as cleanup_error:
                        load_event["cleanup_error"] = (
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                        if model_error is not None:
                            _append_exception_context(
                                model_error,
                                f"cleanup also failed for {model.model_id}: {cleanup_error}",
                            )
                        else:
                            raise
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        finalization_errors: list[Exception] = []
        if sampler_started:
            try:
                active_sampler.stop()
            except Exception as exc:
                finalization_errors.append(exc)
        try:
            router_state = client.restore_loaded_model(initial_loaded_model, managed_model_ids)
        except Exception as exc:
            finalization_errors.append(exc)
        if primary_error is not None:
            for error in finalization_errors:
                _append_exception_context(
                    primary_error,
                    f"benchmark finalization also failed: {type(error).__name__}: {error}",
                )
        elif finalization_errors:
            first_error = finalization_errors[0]
            for error in finalization_errors[1:]:
                _append_exception_context(
                    first_error,
                    f"additional finalization failure: {type(error).__name__}: {error}",
                )
            raise first_error

    _assert_input_unchanged(inventory.source_path, inventory.source_sha256, "inventory")
    _assert_input_unchanged(corpus.source_path, corpus.source_sha256, "corpus")
    _assert_artifacts_unchanged(inventory, artifact_verification)
    ended_monotonic = time.perf_counter()
    summaries = summarize_trials(trials, load_events)
    error_count = sum(bool(item.get("error")) for item in trials + warmup_trials)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_with_errors" if error_count else "completed",
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "elapsed_seconds": round(ended_monotonic - started_monotonic, 3),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "protocol": {
            "rounds": rounds,
            "warmups_per_model_load": warmups,
            "warmups_excluded_from_summary": True,
            "max_tokens": max_tokens,
            "seed": seed,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "cache_prompt": False,
            "enable_thinking": False,
            "ordering": "alternating forward/reverse with a two-round rotation",
        },
        "inventory": {
            "path": str(inventory.source_path),
            "sha256": inventory.source_sha256,
            "inventory_version": inventory.inventory_version,
            "models": [model.as_dict() for model in inventory.models],
        },
        "corpus": {
            "path": str(corpus.source_path),
            "sha256": corpus.source_sha256,
            "corpus_version": corpus.corpus_version,
            "description": corpus.description,
            "case_count": len(corpus.cases),
            "claim_scope": (
                "Deterministic evidence-grounding smoke comparison only; this is not a "
                "general-purpose model accuracy benchmark."
            ),
        },
        "schedule": [
            {
                "round": index,
                "profiles": [model.profile for model in order],
                "model_ids": [model.model_id for model in order],
            }
            for index, order in enumerate(schedules, start=1)
        ],
        "artifact_verification": artifact_verification,
        "route_verification": route_verification,
        "load_events": load_events,
        "warmup_trials": warmup_trials,
        "trials": trials,
        "summaries": summaries,
        "resource_samples": active_sampler.samples,
        "error_count": error_count,
        "router_state": router_state,
    }


def summarize_trials(
    trials: Sequence[dict[str, Any]], load_events: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    model_ids = list(dict.fromkeys(str(item["model_id"]) for item in trials))
    summaries: list[dict[str, Any]] = []
    for model_id in model_ids:
        selected = [item for item in trials if item["model_id"] == model_id]
        successful = [item for item in selected if not item.get("error")]
        scores = [float(item["accuracy"]["score"]) for item in successful]
        passes = [bool(item["accuracy"]["passed"]) for item in successful]
        model_loads = [
            float(item["load_seconds"]) for item in load_events if item["model_id"] == model_id
        ]
        summaries.append(
            {
                "profile": selected[0]["profile"],
                "model_id": model_id,
                "trial_count": len(selected),
                "successful_trial_count": len(successful),
                "error_count": len(selected) - len(successful),
                "accuracy_mean": _mean(scores),
                "accuracy_pass_rate": _mean([1.0 if value else 0.0 for value in passes]),
                "ttft_ms": _percentiles(successful, "ttft_ms"),
                "total_latency_ms": _percentiles(successful, "total_latency_ms"),
                "tokens_per_second": _percentiles(successful, "tokens_per_second"),
                "server_tokens_per_second": _percentiles(successful, "server_tokens_per_second"),
                "output_tokens_total": sum(
                    int(item["output_tokens"])
                    for item in successful
                    if isinstance(item.get("output_tokens"), int)
                ),
                "load_seconds": _value_percentiles(model_loads),
                "mem_available_mb_min": _minimum_nested(
                    successful, "resources", "mem_available_mb_min"
                ),
                "swap_used_mb_max": _maximum_nested(successful, "resources", "swap_used_mb_max"),
                "process_tree_rss_mb_max": _maximum_nested(
                    successful, "resources", "process_tree_rss_mb_max"
                ),
                "thermal_c_max": _maximum_nested(successful, "resources", "thermal_c_max"),
            }
        )
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark pinned local GGUF candidates through a llama.cpp model router."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--base-url", default=DEFAULT_ROUTER_URL)
    parser.add_argument(
        "--allow-router-mutation",
        action="store_true",
        help=(
            "allow load/unload/reload calls against port 8080; omitted by default to protect "
            "the production router"
        ),
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--load-timeout-seconds", type=float, default=900.0)
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument(
        "--api-key-file",
        type=Path,
        help="read one API key from a non-symlinked file whose mode grants no group/other access",
    )
    authentication.add_argument(
        "--api-key-env",
        metavar="NAME",
        help=f"read the API key from NAME (default: {DEFAULT_API_KEY_ENV})",
    )
    parser.add_argument("--process-pid", type=int)
    parser.add_argument("--resource-sample-interval", type=float, default=1.0)
    parser.add_argument("--no-resource-sampling", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = (args.output or _default_output_path()).expanduser()
    output_is_safe = False
    client: RouterClient | None = None
    try:
        output_path = _validate_output_path(
            output_path, protected_paths=(args.inventory, args.corpus)
        )
        output_is_safe = True
        _validate_router_target(args.base_url, allow_router_mutation=args.allow_router_mutation)
        inventory = load_inventory(args.inventory)
        corpus = load_corpus(args.corpus)
        output_is_safe = False
        output_path = _validate_output_path(
            output_path,
            protected_paths=(
                inventory.source_path,
                corpus.source_path,
                *(model.path for model in inventory.models),
            ),
        )
        output_is_safe = True
        api_key = _load_api_key(args.api_key_file, args.api_key_env)
        sampler: Sampler
        if args.no_resource_sampling:
            sampler = NullSampler()
        else:
            sampler = ProcMemorySampler(
                process_pid=args.process_pid,
                interval_seconds=args.resource_sample_interval,
            )
        client = RouterClient(
            args.base_url,
            request_timeout_seconds=args.request_timeout_seconds,
            load_timeout_seconds=args.load_timeout_seconds,
            api_key=api_key,
        )
        report = run_benchmark(
            inventory,
            corpus,
            client,
            rounds=args.rounds,
            warmups=args.warmups,
            max_tokens=args.max_tokens,
            seed=args.seed,
            sampler=sampler,
        )
        _write_report(output_path, report)
        print(f"Benchmark report: {output_path.resolve()}")
        for summary in report["summaries"]:
            print(
                f"{summary['profile']}: accuracy={summary['accuracy_mean']} "
                f"ttft_p95_ms={summary['ttft_ms']['p95']} "
                f"tok_s_p50={summary['tokens_per_second']['p50']} "
                f"errors={summary['error_count']}"
            )
        return 4 if report["error_count"] else 0
    except (InputValidationError, ArtifactVerificationError) as exc:
        if output_is_safe:
            _write_failure_report(output_path, "input_or_artifact_verification_failed", exc)
        print(f"Benchmark verification failed: {exc}", file=sys.stderr)
        return 2
    except RouteVerificationError as exc:
        if output_is_safe:
            _write_failure_report(output_path, "route_verification_failed", exc)
        print(f"Benchmark route verification failed: {exc}", file=sys.stderr)
        return 3
    except (RouterError, httpx.HTTPError, OSError, ValueError) as exc:
        if output_is_safe:
            _write_failure_report(output_path, "benchmark_failed", exc)
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


def _run_prompt(
    client: Any,
    sampler: Sampler,
    model: ModelArtifact,
    *,
    round_index: int,
    prompt_id: str,
    category: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    seed: int,
    scorer: Any,
    excluded_from_summary: bool,
) -> dict[str, Any]:
    sampler.capture()
    started = time.perf_counter()
    try:
        result = client.complete(
            model.model_id,
            messages,
            max_tokens=max_tokens,
            seed=seed,
        )
    except RouteVerificationError:
        raise
    except Exception as exc:  # noqa: BLE001 - trials retain component errors for audit.
        sampler.capture()
        ended = time.perf_counter()
        return {
            "round": round_index,
            "profile": model.profile,
            "model_id": model.model_id,
            "prompt_id": prompt_id,
            "category": category,
            "excluded_from_summary": excluded_from_summary,
            "served_model": None,
            "route_verified": False,
            "answer": "",
            "ttft_ms": None,
            "total_latency_ms": round((ended - started) * 1000, 3),
            "output_tokens": None,
            "output_tokens_source": "unavailable",
            "tokens_per_second": None,
            "server_tokens_per_second": None,
            "prompt_tokens": None,
            "finish_reason": None,
            "accuracy": scorer("") if scorer is not None else None,
            "resources": sampler.window(started, ended),
            "error": f"{type(exc).__name__}: {exc}",
        }
    sampler.capture()
    ended = time.perf_counter()
    return {
        "round": round_index,
        "profile": model.profile,
        "model_id": model.model_id,
        "prompt_id": prompt_id,
        "category": category,
        "excluded_from_summary": excluded_from_summary,
        "served_model": result.served_model,
        "route_verified": result.served_model == model.model_id,
        "answer": result.answer,
        "ttft_ms": _round_optional(result.ttft_ms),
        "total_latency_ms": round(result.total_latency_ms, 3),
        "output_tokens": result.output_tokens,
        "output_tokens_source": result.output_tokens_source,
        "tokens_per_second": _round_optional(result.tokens_per_second),
        "server_tokens_per_second": _round_optional(result.server_tokens_per_second),
        "prompt_tokens": result.prompt_tokens,
        "finish_reason": result.finish_reason,
        "accuracy": scorer(result.answer) if scorer is not None else None,
        "resources": sampler.window(started, ended),
        "error": None,
    }


def _read_resource_sample(process_pid: int | None) -> dict[str, Any]:
    meminfo = _read_meminfo()
    return {
        "utc": _utc_now(),
        "monotonic_seconds": time.perf_counter(),
        "mem_available_mb": _kb_to_mb(meminfo.get("MemAvailable")),
        "swap_used_mb": _swap_used_mb(meminfo),
        "process_tree_rss_mb": _process_tree_rss_mb(process_pid),
        "thermal_c_max": _thermal_c_max(),
    }


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            match = re.search(r"\d+", raw)
            if match:
                values[key] = int(match.group())
    except (OSError, ValueError):
        pass
    return values


def _process_tree_rss_mb(root_pid: int | None) -> float | None:
    if root_pid is None or root_pid <= 0 or not Path("/proc").exists():
        return None
    parents: dict[int, int] = {}
    try:
        proc_entries = list(Path("/proc").iterdir())
    except OSError:
        return None
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2 :].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    total_kb = 0
    found = False
    for pid in descendants:
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
        if match:
            total_kb += int(match.group(1))
            found = True
    return round(total_kb / 1024, 3) if found else None


def _thermal_c_max() -> float | None:
    temperatures: list[float] = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            raw = float(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        temperatures.append(raw / 1000 if raw > 1000 else raw)
    return round(max(temperatures), 3) if temperatures else None


def _run_values(items: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [
        float(item[key])
        for item in items
        if isinstance(item.get(key), (int, float)) and not isinstance(item.get(key), bool)
    ]


def _percentiles(items: Sequence[dict[str, Any]], key: str) -> dict[str, float | None]:
    return _value_percentiles(_run_values(items, key))


def _value_percentiles(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    result = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return round(result, 3)


def _minimum_nested(items: Sequence[dict[str, Any]], outer: str, inner: str) -> float | None:
    return _nested_extreme(items, outer, inner, min)


def _maximum_nested(items: Sequence[dict[str, Any]], outer: str, inner: str) -> float | None:
    return _nested_extreme(items, outer, inner, max)


def _nested_extreme(
    items: Sequence[dict[str, Any]], outer: str, inner: str, operation: Any
) -> float | None:
    values = [
        float(item[outer][inner])
        for item in items
        if isinstance(item.get(outer), dict) and isinstance(item[outer].get(inner), (int, float))
    ]
    return round(operation(values), 3) if values else None


def _extreme(samples: Sequence[dict[str, Any]], key: str, operation: Any) -> float | None:
    values = [
        float(item[key])
        for item in samples
        if isinstance(item.get(key), (int, float)) and not isinstance(item.get(key), bool)
    ]
    return round(operation(values), 3) if values else None


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _fact_rules(value: Any, prompt_id: str, field: str) -> tuple[FactRule, ...]:
    if not isinstance(value, list):
        raise InputValidationError(f"{prompt_id}.{field} must be a list")
    rules: list[FactRule] = []
    ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise InputValidationError(f"{prompt_id}.{field}[{index}] must be an object")
        fact_id = item.get("id")
        if not isinstance(fact_id, str) or not fact_id.strip():
            raise InputValidationError(f"{prompt_id}.{field}[{index}].id must be a string")
        if fact_id in ids:
            raise InputValidationError(f"duplicate rule ID {fact_id!r} in {prompt_id}.{field}")
        ids.add(fact_id)
        rules.append(
            FactRule(
                fact_id=fact_id,
                any_of=_string_list(item.get("any_of"), f"{prompt_id}.{field}.{fact_id}"),
            )
        )
    return tuple(rules)


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise InputValidationError(f"{field} must be a non-empty list of strings")
    values = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(values) != len(value):
        raise InputValidationError(f"{field} must contain only non-empty strings")
    return values


def _required_string(payload: dict[str, Any], key: str, source: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{key} must be a non-empty string: {source}")
    return value.strip()


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{key} must be a non-empty string when provided")
    return value.strip()


def _read_json_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise InputValidationError(f"could not read {label} {path}: {exc}") from exc


def _decode_json(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputValidationError(f"top-level JSON must be an object: {path}")
    return payload


def _assert_input_unchanged(path: Path, expected_sha256: str, label: str) -> None:
    actual = sha256_file(path)
    if not _constant_time_equal(actual, expected_sha256):
        raise InputValidationError(
            f"{label} changed during benchmark: expected {expected_sha256}, got {actual}"
        )


def _assert_artifacts_unchanged(
    inventory: Inventory, verification: Sequence[dict[str, Any]]
) -> None:
    by_model = {str(item["model_id"]): item for item in verification}
    for model in inventory.models:
        expected = by_model[model.model_id]
        try:
            stat = model.path.stat()
        except OSError as exc:
            raise ArtifactVerificationError(
                f"artifact disappeared during benchmark: {model.path}"
            ) from exc
        fingerprint = expected["stat_fingerprint"]
        if (
            stat.st_size != expected["size_bytes"]
            or stat.st_dev != fingerprint["device"]
            or stat.st_ino != fingerprint["inode"]
            or stat.st_mtime_ns != fingerprint["mtime_ns"]
        ):
            raise ArtifactVerificationError(f"artifact changed during benchmark: {model.path}")


def _matches_rule(answer: str, rule: FactRule) -> bool:
    return any(_contains_phrase(answer, phrase) for phrase in rule.any_of)


def _contains_phrase(answer: str, phrase: str) -> bool:
    normalized_answer = f" {_normalize_text(answer)} "
    normalized_phrase = _normalize_text(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_answer


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", decomposed).split())


def _paths_are_same(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left.expanduser(), right.expanduser())
    except OSError:
        return left.expanduser().resolve() == right.expanduser().resolve()


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def _observed_decode_tps(
    output_tokens: int | None,
    *,
    first_text_at: float | None,
    last_text_at: float | None,
) -> float | None:
    if (
        output_tokens is None
        or output_tokens <= 1
        or first_text_at is None
        or last_text_at is None
        or last_text_at <= first_text_at
    ):
        return None
    return (output_tokens - 1) / (last_text_at - first_text_at)


def _append_exception_context(exc: BaseException, message: str) -> None:
    current = str(exc)
    updated = f"{current}; {message}" if current else message
    exc.args = (updated, *exc.args[1:])


def _perf_counter() -> float:
    return time.perf_counter()


def _integer_metric(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _float_metric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _round_optional(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _kb_to_mb(value: int | None) -> float | None:
    return round(value / 1024, 3) if value is not None else None


def _swap_used_mb(meminfo: dict[str, int]) -> float | None:
    total = meminfo.get("SwapTotal")
    free = meminfo.get("SwapFree")
    if total is None or free is None:
        return None
    return round((total - free) / 1024, 3)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_router_target(base_url: str, *, allow_router_mutation: bool) -> None:
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as exc:
        raise InputValidationError(f"invalid router base URL: {base_url!r}") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise InputValidationError(
            "router base URL must be a credential-free HTTP(S) URL without a query or fragment"
        )
    if port == 8080 and not allow_router_mutation:
        raise InputValidationError(
            "refusing to mutate a router on port 8080; use the dedicated benchmark port "
            "18080 or pass --allow-router-mutation after quiescing production clients"
        )


def _load_api_key(api_key_file: Path | None, api_key_env: str | None) -> str | None:
    if api_key_file is not None:
        path = _absolute_lexical_path(api_key_file)
        symlink = _first_symlink_component(path)
        if symlink is not None:
            raise InputValidationError(f"API key file path contains a symlink: {symlink}")
        if not path.is_file():
            raise InputValidationError(f"API key file is not a regular file: {path}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise InputValidationError(
                f"API key file must not grant group/other permissions: {path} (mode {mode:04o})"
            )
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise InputValidationError(f"could not read API key file {path}: {exc}") from exc
        value = raw.rstrip("\r\n")
        if not value or "\n" in value or "\r" in value:
            raise InputValidationError("API key file must contain exactly one non-empty line")
        return value

    environment_name = api_key_env or DEFAULT_API_KEY_ENV
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", environment_name):
        raise InputValidationError(
            f"invalid API key environment variable name: {environment_name!r}"
        )
    value = os.environ.get(environment_name)
    if value is None:
        return None
    if not value.strip() or "\n" in value or "\r" in value:
        raise InputValidationError(
            f"API key environment variable {environment_name} must be one non-empty line"
        )
    return value


def _absolute_lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _first_symlink_component(path: Path) -> Path | None:
    candidate = _absolute_lexical_path(path)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if current.is_symlink():
            return current
        if not current.exists():
            break
    return None


def _temporary_output_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _validate_output_path(path: Path, *, protected_paths: Sequence[Path]) -> Path:
    destination = _absolute_lexical_path(path)
    if destination.suffix.casefold() != ".json":
        raise InputValidationError(f"benchmark output must use a .json extension: {destination}")
    temporary = _temporary_output_path(destination)
    protected = tuple(_absolute_lexical_path(item) for item in protected_paths)
    for label, candidate in (("output", destination), ("temporary output", temporary)):
        symlink = _first_symlink_component(candidate)
        if symlink is not None:
            raise InputValidationError(f"{label} path contains a symlink and is unsafe: {symlink}")
        if candidate.exists() and not candidate.is_file():
            raise InputValidationError(f"{label} path is not a regular file: {candidate}")
        for protected_path in protected:
            if _paths_are_same(candidate, protected_path):
                raise InputValidationError(
                    f"{label} path collides with protected input/artifact: {protected_path}"
                )
    if temporary.exists():
        raise InputValidationError(f"refusing to reuse an existing temporary output: {temporary}")
    if destination.exists():
        raise InputValidationError(
            f"refusing to overwrite existing benchmark output: {destination}"
        )
    return destination


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "artifacts" / "voice-model-benchmark" / f"voice-models-{timestamp}.json"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path = _validate_output_path(path, protected_paths=())
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _validate_output_path(path, protected_paths=())
    temporary = _temporary_output_path(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise InputValidationError("this platform cannot securely publish benchmark output")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | nofollow
    directory_descriptor = os.open(path.parent, directory_flags)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= nofollow
    descriptor = -1
    temporary_created = False
    try:
        descriptor = os.open(temporary.name, flags, 0o600, dir_fd=directory_descriptor)
        temporary_created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary.name,
                path.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise InputValidationError(
                f"refusing benchmark output that appeared during publication: {path}"
            ) from exc
        os.unlink(temporary.name, dir_fd=directory_descriptor)
        temporary_created = False
        os.fsync(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_created:
            try:
                os.unlink(temporary.name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
        os.close(directory_descriptor)


def _write_failure_report(path: Path, status: str, exc: Exception) -> None:
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "ended_utc": _utc_now(),
        "error": f"{type(exc).__name__}: {exc}",
    }
    try:
        _write_report(path, report)
    except (OSError, InputValidationError):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
