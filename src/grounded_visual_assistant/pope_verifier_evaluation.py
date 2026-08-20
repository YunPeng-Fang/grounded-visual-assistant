"""Paired POPE evaluation for Grounding-aware answer verification."""

from __future__ import annotations

import hashlib
import statistics
from math import comb
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .grounding_answer_verifier import (
    GroundingAnswerVerifierConfig,
    compact_grounding_result,
    verify_binary_answer,
)
from .pope_dataset import POPE_STRATEGIES
from .pope_evaluation import binary_metrics, evaluate_answer


POPE_VERIFIER_BATCH_PROTOCOL = "pope_grounding_positive_rescue_batch_v1"


def verification_query_key(record: Mapping[str, Any]) -> str:
    """Hash only inference inputs, never the ground-truth answer."""
    payload = "\n".join(
        (
            str(record["image_id"]),
            str(record["object"]).strip().lower(),
            " ".join(str(record["question"]).strip().lower().split()),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def group_verification_queries(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group repeated POPE questions for one cached grounding inference."""
    grouped: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        record = dict(raw_record)
        key = verification_query_key(record)
        signature = (
            int(record["image_id"]),
            str(record["image"]),
            str(record["object"]).strip().lower(),
            " ".join(str(record["question"]).strip().split()),
        )
        target = str(record["gt_answer"]).strip().lower()
        if key not in grouped:
            grouped[key] = {
                "query_key": key,
                "image_id": signature[0],
                "image": signature[1],
                "object": signature[2],
                "question": signature[3],
                "gt_answer": target,
                "records": [],
            }
        group = grouped[key]
        current_signature = (
            group["image_id"],
            group["image"],
            group["object"],
            group["question"],
        )
        if current_signature != signature:
            raise RuntimeError(
                f"Verification query hash collision for {key}."
            )
        if group["gt_answer"] != target:
            raise RuntimeError(
                f"POPE query {key} has conflicting ground-truth labels."
            )
        group["records"].append(record)

    return [
        {
            **group,
            "baseline_ids": [
                str(record["id"]) for record in group["records"]
            ],
            "strategies": sorted(
                {str(record["strategy"]) for record in group["records"]},
                key=POPE_STRATEGIES.index,
            ),
        }
        for group in grouped.values()
    ]


def _validate_baseline(record: Mapping[str, Any]) -> dict[str, Any]:
    expected = evaluate_answer(
        str(record["prediction"]), str(record["gt_answer"])
    )
    saved = record.get("evaluation")
    if not isinstance(saved, Mapping):
        raise ValueError(
            f"Baseline prediction {record.get('id')} has no evaluation."
        )
    for field, value in expected.items():
        if saved.get(field) != value:
            raise RuntimeError(
                f"Baseline evaluation does not reproduce for "
                f"{record.get('id')} field {field!r}."
            )
    return expected


def build_verified_prediction(
    baseline: Mapping[str, Any],
    evidence_record: Mapping[str, Any],
    *,
    config: GroundingAnswerVerifierConfig,
) -> dict[str, Any]:
    """Fuse one saved Qwen answer with one cached grounding result."""
    baseline_evaluation = _validate_baseline(baseline)
    expected_key = verification_query_key(baseline)
    if evidence_record.get("query_key") != expected_key:
        raise RuntimeError(
            f"Grounding evidence key mismatch for {baseline['id']}."
        )
    grounding = evidence_record.get("grounding")
    if not isinstance(grounding, Mapping):
        raise ValueError(
            f"Grounding evidence {expected_key} has no result payload."
        )
    verification = verify_binary_answer(
        str(baseline["prediction"]),
        target=str(baseline["object"]),
        annotations=grounding.get("annotations", []),
        image_width=int(grounding["img_width"]),
        image_height=int(grounding["img_height"]),
        config=config,
    )
    final_answer = str(verification["final_answer"])
    final_evaluation = evaluate_answer(
        final_answer, str(baseline["gt_answer"])
    )
    baseline_latency = float(baseline.get("latency_seconds", 0.0))
    verification_latency = float(
        (grounding.get("latency_seconds") or {}).get("total", 0.0)
    )
    result = {
        key: baseline.get(key)
        for key in (
            "id",
            "strategy",
            "question_id",
            "image",
            "image_id",
            "question",
            "object",
            "gt_answer",
            "model",
        )
        if key in baseline
    }
    result.update(
        {
            "query_key": expected_key,
            "baseline_prediction": baseline["prediction"],
            "baseline_evaluation": baseline_evaluation,
            "prediction": final_answer,
            "evaluation": final_evaluation,
            "verification": verification,
            "grounding": compact_grounding_result(grounding),
            "baseline_latency_seconds": round(baseline_latency, 6),
            "verification_latency_seconds": round(
                verification_latency, 6
            ),
            "projected_end_to_end_latency_seconds": round(
                baseline_latency + verification_latency, 6
            ),
        }
    )
    for key in (
        "generated_tokens",
        "cuda_peak_memory_allocated_gb",
        "cuda_memory_reserved_gb",
    ):
        if key in baseline:
            result[f"baseline_{key}"] = baseline[key]
    if evidence_record.get("cuda_peak_memory_allocated_gb") is not None:
        result["verification_cuda_peak_memory_allocated_gb"] = (
            evidence_record["cuda_peak_memory_allocated_gb"]
        )
    return result


def _metric_delta(
    baseline: Mapping[str, Any], verified: Mapping[str, Any]
) -> dict[str, float]:
    return {
        name: round(
            float(verified[name]) - float(baseline[name]),
            6,
        )
        for name in ("accuracy", "precision", "recall", "f1", "yes_ratio")
    }


def exact_mcnemar_p_value(
    baseline_only_correct: int,
    verified_only_correct: int,
) -> float:
    """Return the exact two-sided McNemar/binomial p-value."""
    if baseline_only_correct < 0 or verified_only_correct < 0:
        raise ValueError("McNemar discordant counts must be non-negative.")
    discordant = baseline_only_correct + verified_only_correct
    if discordant == 0:
        return 1.0
    lower = min(baseline_only_correct, verified_only_correct)
    tail = sum(comb(discordant, index) for index in range(lower + 1))
    return round(min(1.0, 2.0 * tail / (2**discordant)), 8)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def aggregate_pope_verifier_metrics(
    predictions: Iterable[Mapping[str, Any]],
    *,
    expected_samples: int,
    expected_queries: int,
    completed_queries: int,
    error_attempts: int = 0,
    status: str = "completed",
    protocol: str = POPE_VERIFIER_BATCH_PROTOCOL,
) -> dict[str, Any]:
    """Aggregate paired baseline/V1 metrics and transparent runtime costs."""
    records = [dict(item) for item in predictions]
    baseline_view = [
        {"evaluation": item["baseline_evaluation"]} for item in records
    ]
    baseline_metrics = binary_metrics(baseline_view)
    verified_metrics = binary_metrics(records)

    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_strategy[str(item["strategy"])].append(item)
    strategy_metrics = {}
    for strategy in POPE_STRATEGIES:
        items = by_strategy.get(strategy, [])
        if not items:
            continue
        baseline = binary_metrics(
            [{"evaluation": item["baseline_evaluation"]} for item in items]
        )
        verified = binary_metrics(items)
        strategy_metrics[strategy] = {
            "baseline": baseline,
            "verified": verified,
            "delta": _metric_delta(baseline, verified),
        }

    paired = Counter()
    correction_status = Counter()
    correction_directions = Counter()
    beneficial = harmful = 0
    for item in records:
        baseline_correct = bool(item["baseline_evaluation"]["is_correct"])
        verified_correct = bool(item["evaluation"]["is_correct"])
        if baseline_correct and verified_correct:
            paired["both_correct"] += 1
        elif baseline_correct:
            paired["baseline_only_correct"] += 1
        elif verified_correct:
            paired["verified_only_correct"] += 1
        else:
            paired["both_wrong"] += 1
        verification = item["verification"]
        correction_status[str(verification["status"])] += 1
        if verification["changed"]:
            correction_directions[
                str(verification["correction_direction"])
            ] += 1
            beneficial += int(not baseline_correct and verified_correct)
            harmful += int(baseline_correct and not verified_correct)

    unique_grounding: dict[str, dict[str, Any]] = {}
    for item in records:
        unique_grounding.setdefault(str(item["query_key"]), item)
    baseline_latencies = [
        float(item["baseline_latency_seconds"]) for item in records
    ]
    cached_verification_latencies = [
        float(item["verification_latency_seconds"])
        for item in unique_grounding.values()
    ]
    uncached_verification_latencies = [
        float(item["verification_latency_seconds"]) for item in records
    ]
    baseline_total = sum(baseline_latencies)
    cached_verification_total = sum(cached_verification_latencies)
    uncached_verification_total = sum(uncached_verification_latencies)
    completed = len(records)

    baseline_memory = [
        float(item["baseline_cuda_peak_memory_allocated_gb"])
        for item in records
        if item.get("baseline_cuda_peak_memory_allocated_gb") is not None
    ]
    verification_memory = [
        float(item["verification_cuda_peak_memory_allocated_gb"])
        for item in unique_grounding.values()
        if item.get("verification_cuda_peak_memory_allocated_gb") is not None
    ]
    result: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "status": status,
        "coverage": {
            "expected_samples": expected_samples,
            "completed_samples": completed,
            "remaining_samples": max(expected_samples - completed, 0),
            "sample_completion_rate": (
                round(completed / expected_samples, 6)
                if expected_samples
                else 0.0
            ),
            "expected_unique_queries": expected_queries,
            "completed_unique_queries": completed_queries,
            "remaining_unique_queries": max(
                expected_queries - completed_queries, 0
            ),
            "query_completion_rate": (
                round(completed_queries / expected_queries, 6)
                if expected_queries
                else 0.0
            ),
            "grounding_queries_saved_by_deduplication": max(
                completed - completed_queries, 0
            ),
            "error_attempts": error_attempts,
        },
        "baseline": baseline_metrics,
        "verified": verified_metrics,
        "delta": _metric_delta(baseline_metrics, verified_metrics),
        "strategies": strategy_metrics,
        "paired_outcomes": {
            "both_correct": paired["both_correct"],
            "baseline_only_correct": paired["baseline_only_correct"],
            "verified_only_correct": paired["verified_only_correct"],
            "both_wrong": paired["both_wrong"],
            "beneficial_corrections": beneficial,
            "harmful_corrections": harmful,
            "net_correct_corrections": beneficial - harmful,
            "mcnemar_exact_two_sided_p_value": exact_mcnemar_p_value(
                paired["baseline_only_correct"],
                paired["verified_only_correct"],
            ),
        },
        "corrections": {
            "changed_answers": beneficial + harmful,
            "change_rate": (
                round((beneficial + harmful) / completed, 6)
                if completed
                else 0.0
            ),
            "directions": dict(sorted(correction_directions.items())),
            "status_counts": dict(sorted(correction_status.items())),
        },
        "latency_seconds": {
            "baseline_total": round(baseline_total, 6),
            "baseline_mean": _mean(baseline_latencies),
            "verification_unique_query_total": round(
                cached_verification_total, 6
            ),
            "verification_unique_query_mean": _mean(
                cached_verification_latencies
            ),
            "offline_cached_end_to_end_total": round(
                baseline_total + cached_verification_total, 6
            ),
            "offline_cached_end_to_end_mean_per_question": (
                round(
                    (baseline_total + cached_verification_total) / completed,
                    6,
                )
                if completed
                else 0.0
            ),
            "uncached_projected_end_to_end_total": round(
                baseline_total + uncached_verification_total, 6
            ),
            "uncached_projected_end_to_end_mean_per_question": (
                round(
                    (baseline_total + uncached_verification_total) / completed,
                    6,
                )
                if completed
                else 0.0
            ),
        },
    }
    if baseline_memory or verification_memory:
        baseline_max = max(baseline_memory) if baseline_memory else 0.0
        verification_max = (
            max(verification_memory) if verification_memory else 0.0
        )
        result["cuda_memory_gb"] = {
            "baseline_peak_max": round(baseline_max, 6),
            "verification_peak_max": round(verification_max, 6),
            "sequential_projected_peak_max": round(
                max(baseline_max, verification_max), 6
            ),
            "note": (
                "Sequential projection; the saved baseline and verifier were "
                "not resident on GPU simultaneously."
            ),
        }
    return result
