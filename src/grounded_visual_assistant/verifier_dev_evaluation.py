"""Evaluation helpers for the POPE-isolated verifier development set."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .pope_evaluation import POPE_SYSTEM_PROMPT, binary_metrics


VERIFIER_DEV_BASELINE_PROTOCOL = "verifier_dev_qwen_baseline_v1"
VERIFIER_DEV_SYSTEM_PROMPT = POPE_SYSTEM_PROMPT


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def aggregate_verifier_dev_metrics(
    predictions: Iterable[Mapping[str, Any]],
    *,
    expected_samples: int,
    error_attempts: int = 0,
    status: str = "completed",
) -> dict[str, Any]:
    """Aggregate binary, pair-level, group, runtime, and memory metrics."""
    records = [dict(item) for item in predictions]
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_supercategory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_role[str(item["pair_role"])].append(item)
        by_supercategory[str(item["supercategory"])].append(item)
        by_pair[str(item["pair_id"])].append(item)

    pair_outcomes = Counter()
    completed_pairs = 0
    for items in by_pair.values():
        if len(items) != 2:
            continue
        completed_pairs += 1
        correct = sum(bool(item["evaluation"]["is_correct"]) for item in items)
        pair_outcomes[
            "both_correct"
            if correct == 2
            else "one_correct"
            if correct == 1
            else "both_wrong"
        ] += 1
    completed = len(records)
    latencies = [
        float(item.get("latency_seconds", 0.0)) for item in records
    ]
    generated_tokens = [
        int(item["generated_tokens"])
        for item in records
        if item.get("generated_tokens") is not None
    ]
    memory = [
        float(item["cuda_peak_memory_allocated_gb"])
        for item in records
        if item.get("cuda_peak_memory_allocated_gb") is not None
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": VERIFIER_DEV_BASELINE_PROTOCOL,
        "status": status,
        "coverage": {
            "expected": expected_samples,
            "completed": completed,
            "remaining": max(expected_samples - completed, 0),
            "completion_rate": (
                round(completed / expected_samples, 6)
                if expected_samples
                else 0.0
            ),
            "error_attempts": error_attempts,
        },
        "overall": binary_metrics(records),
        "pair_roles": {
            name: binary_metrics(items)
            for name, items in sorted(by_role.items())
        },
        "supercategories": {
            name: binary_metrics(items)
            for name, items in sorted(by_supercategory.items())
        },
        "pairs": {
            "expected": expected_samples // 2,
            "completed": completed_pairs,
            "both_correct": pair_outcomes["both_correct"],
            "one_correct": pair_outcomes["one_correct"],
            "both_wrong": pair_outcomes["both_wrong"],
            "both_correct_rate": (
                round(
                    pair_outcomes["both_correct"] / completed_pairs, 6
                )
                if completed_pairs
                else 0.0
            ),
        },
        "latency_seconds": {
            "total": round(sum(latencies), 6),
            "mean": _mean(latencies),
            "throughput_samples_per_second": (
                round(completed / sum(latencies), 6)
                if latencies and sum(latencies) > 0
                else 0.0
            ),
        },
        "generation": {
            "system_prompt": VERIFIER_DEV_SYSTEM_PROMPT,
            "generated_tokens_mean": _mean(generated_tokens),
            "token_limit_hits": sum(
                bool(item.get("hit_max_new_tokens")) for item in records
            ),
            "strict_parse_valid_rate": binary_metrics(records)[
                "strict_parse_valid_rate"
            ],
        },
        "cuda_memory_gb": {
            "peak_allocated_max": round(max(memory), 6) if memory else 0.0
        },
    }
