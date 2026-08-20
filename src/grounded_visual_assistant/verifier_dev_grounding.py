"""GT-free query selection and diagnostics for Verifier Dev grounding."""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .evaluation import parse_yes_no
from .pope_evaluation import evaluate_answer
from .pope_verifier_evaluation import verification_query_key


VERIFIER_DEV_GROUNDING_PROTOCOL = "verifier_dev_grounding_evidence_v1"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def ordered_query_keys_sha256(
    jobs: Iterable[Mapping[str, Any]],
) -> str:
    payload = "\n".join(str(item["query_key"]) for item in jobs) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_negative_grounding_jobs(
    baseline_predictions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select strict baseline-No queries without copying GT metadata."""
    jobs = []
    for raw_item in baseline_predictions:
        item = dict(raw_item)
        evaluation = evaluate_answer(
            str(item["prediction"]), str(item["gt_answer"])
        )
        if evaluation != item.get("evaluation"):
            raise RuntimeError(
                f"Baseline evaluation mismatch for {item.get('id')}."
            )
        parsed = parse_yes_no(str(item["prediction"]))
        if parsed is None:
            raise ValueError(
                f"Baseline answer is not strict Yes/No: {item.get('id')}."
            )
        if parsed != "no":
            continue
        jobs.append(
            {
                "query_key": verification_query_key(item),
                "baseline_id": str(item["id"]),
                "image": str(item["image"]),
                "image_id": int(item["image_id"]),
                "question": str(item["question"]),
                "object": str(item["object"]),
            }
        )
    query_keys = [str(item["query_key"]) for item in jobs]
    baseline_ids = [str(item["baseline_id"]) for item in jobs]
    if len(query_keys) != len(set(query_keys)):
        raise ValueError("Verifier Dev grounding jobs contain duplicate queries.")
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("Verifier Dev grounding jobs contain duplicate IDs.")
    return jobs


def aggregate_verifier_dev_grounding_metrics(
    evidence_records: Iterable[Mapping[str, Any]],
    *,
    baseline_by_id: Mapping[str, Mapping[str, Any]],
    expected_queries: int,
    error_attempts: int = 0,
    status: str = "completed",
) -> dict[str, Any]:
    """Aggregate evidence coverage and label-aware Dev diagnostics."""
    records = [dict(item) for item in evidence_records]
    candidate_counts = []
    max_scores = []
    latencies = []
    memory = []
    by_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    score_bins = Counter()
    for item in records:
        baseline = baseline_by_id[str(item["baseline_id"])]
        target = str(baseline["gt_answer"])
        outcome = "false_negative" if target == "yes" else "true_negative"
        grounding = item["grounding"]
        annotations = list(grounding.get("annotations", []))
        count = len(annotations)
        score = max(
            (float(annotation.get("score", 0.0)) for annotation in annotations),
            default=0.0,
        )
        diagnostic = {
            "candidate_count": count,
            "max_grounding_score": score,
        }
        by_outcome[outcome].append(diagnostic)
        candidate_counts.append(count)
        max_scores.append(score)
        latency = float(
            (grounding.get("latency_seconds") or {}).get("total", 0.0)
        )
        latencies.append(latency)
        peak = item.get(
            "cuda_peak_memory_allocated_gb",
            grounding.get("cuda_peak_memory_allocated_gb"),
        )
        if peak is not None:
            memory.append(float(peak))
        score_bins[
            "none"
            if count == 0
            else "0.30-0.39"
            if score < 0.4
            else "0.40-0.49"
            if score < 0.5
            else "0.50-0.69"
            if score < 0.7
            else "0.70-1.00"
        ] += 1

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        with_candidates = sum(item["candidate_count"] > 0 for item in items)
        return {
            "queries": len(items),
            "queries_with_candidates": with_candidates,
            "candidate_presence_rate": (
                round(with_candidates / len(items), 6) if items else 0.0
            ),
            "candidate_count_mean": _mean(
                item["candidate_count"] for item in items
            ),
            "max_grounding_score_mean": _mean(
                item["max_grounding_score"] for item in items
            ),
        }

    completed = len(records)
    queries_with_candidates = sum(value > 0 for value in candidate_counts)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": VERIFIER_DEV_GROUNDING_PROTOCOL,
        "status": status,
        "coverage": {
            "expected_queries": expected_queries,
            "completed_queries": completed,
            "remaining_queries": max(expected_queries - completed, 0),
            "completion_rate": (
                round(completed / expected_queries, 6)
                if expected_queries
                else 0.0
            ),
            "error_attempts": error_attempts,
        },
        "candidates": {
            "queries_with_candidates": queries_with_candidates,
            "queries_without_candidates": completed - queries_with_candidates,
            "candidate_presence_rate": (
                round(queries_with_candidates / completed, 6)
                if completed
                else 0.0
            ),
            "annotations_total": sum(candidate_counts),
            "candidate_count_mean": _mean(candidate_counts),
            "max_grounding_score_mean": _mean(max_scores),
            "max_score_bins": dict(sorted(score_bins.items())),
        },
        "dev_diagnostics": {
            name: summarize(items)
            for name, items in sorted(by_outcome.items())
        },
        "latency_seconds": {
            "total": round(sum(latencies), 6),
            "mean": _mean(latencies),
            "throughput_queries_per_second": (
                round(completed / sum(latencies), 6)
                if latencies and sum(latencies) > 0
                else 0.0
            ),
        },
        "cuda_memory_gb": {
            "peak_allocated_max": round(max(memory), 6) if memory else 0.0
        },
        "methodology": {
            "inference_selection_uses_ground_truth": False,
            "diagnostic_labels_used_after_inference": True,
            "selection_rule": (
                "all and only strict No answers from the frozen Dev110 Qwen "
                "baseline"
            ),
        },
    }
