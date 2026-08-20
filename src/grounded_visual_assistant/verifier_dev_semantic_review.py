"""Candidate jobs and diagnostics for Verifier Dev semantic review."""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .semantic_answer_verifier import (
    SemanticAnswerVerifierConfig,
    normalize_semantic_review,
    select_semantic_candidates,
    semantic_candidate_key,
    semantic_review_question,
)


VERIFIER_DEV_SEMANTIC_REVIEW_PROTOCOL = "verifier_dev_semantic_review_v1"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def ordered_candidate_keys_sha256(
    jobs: Iterable[Mapping[str, Any]],
) -> str:
    payload = "\n".join(str(item["candidate_key"]) for item in jobs) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_dev_semantic_review_jobs(
    evidence_records: Iterable[Mapping[str, Any]],
    *,
    config: SemanticAnswerVerifierConfig,
) -> list[dict[str, Any]]:
    """Build candidate crop jobs without adding ground-truth metadata."""
    jobs = []
    for raw_record in evidence_records:
        record = dict(raw_record)
        grounding = record["grounding"]
        candidates, _ = select_semantic_candidates(
            grounding.get("annotations", []),
            target=str(record["object"]),
            image_width=int(grounding["img_width"]),
            image_height=int(grounding["img_height"]),
            config=config,
        )
        for candidate in candidates:
            annotation_index = int(candidate["annotation_index"])
            jobs.append(
                {
                    "candidate_key": semantic_candidate_key(
                        str(record["query_key"]), annotation_index
                    ),
                    "query_key": str(record["query_key"]),
                    "baseline_id": str(record["baseline_id"]),
                    "image": str(record["image"]),
                    "image_id": int(record["image_id"]),
                    "question": str(record["question"]),
                    "object": str(record["object"]),
                    "annotation_index": annotation_index,
                    "grounding_score": candidate["grounding_score"],
                    "mask_score": candidate["mask_score"],
                    "mask_area_ratio": candidate["mask_area_ratio"],
                    "bbox": list(candidate["bbox"]),
                    "semantic_question": semantic_review_question(
                        str(record["object"])
                    ),
                }
            )
    candidate_keys = [str(item["candidate_key"]) for item in jobs]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("Dev semantic jobs contain duplicate candidate keys.")
    return jobs


def aggregate_dev_semantic_review_metrics(
    reviews: Iterable[Mapping[str, Any]],
    *,
    jobs: Iterable[Mapping[str, Any]],
    evidence_records: Iterable[Mapping[str, Any]],
    baseline_by_id: Mapping[str, Mapping[str, Any]],
    error_attempts: int = 0,
    status: str = "completed",
) -> dict[str, Any]:
    """Aggregate review coverage and post-inference Dev diagnostics."""
    jobs = [dict(item) for item in jobs]
    evidence = [dict(item) for item in evidence_records]
    normalized = [normalize_semantic_review(item) for item in reviews]
    reviews_by_key = {
        str(item["candidate_key"]): item for item in normalized
    }
    if len(reviews_by_key) != len(normalized):
        raise ValueError("Dev semantic reviews contain duplicate keys.")
    jobs_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in jobs:
        jobs_by_query[str(item["query_key"])].append(item)

    parsed = Counter(
        str(item.get("parsed_answer") or "invalid")
        for item in normalized
    )
    exact = sum(bool(item.get("exact_answer")) for item in normalized)
    latencies = [
        float(
            item.get(
                "end_to_end_latency_seconds",
                item.get("latency_seconds", 0.0),
            )
        )
        for item in normalized
    ]
    memory = [
        float(item["cuda_peak_memory_allocated_gb"])
        for item in normalized
        if item.get("cuda_peak_memory_allocated_gb") is not None
    ]
    by_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in evidence:
        baseline = baseline_by_id[str(record["baseline_id"])]
        outcome = (
            "false_negative"
            if str(baseline["gt_answer"]) == "yes"
            else "true_negative"
        )
        query_jobs = jobs_by_query.get(str(record["query_key"]), [])
        query_reviews = [
            reviews_by_key[str(item["candidate_key"])]
            for item in query_jobs
            if str(item["candidate_key"]) in reviews_by_key
        ]
        by_outcome[outcome].append(
            {
                "has_candidates": bool(query_jobs),
                "candidate_count": len(query_jobs),
                "completed_reviews": len(query_reviews),
                "any_semantic_yes": any(
                    item.get("parsed_answer") == "yes"
                    and item.get("exact_answer")
                    for item in query_reviews
                ),
            }
        )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        candidate_queries = sum(item["has_candidates"] for item in items)
        semantic_yes_queries = sum(
            item["any_semantic_yes"] for item in items
        )
        return {
            "queries": len(items),
            "candidate_queries": candidate_queries,
            "candidate_presence_rate": (
                round(candidate_queries / len(items), 6)
                if items
                else 0.0
            ),
            "candidate_reviews": sum(
                item["completed_reviews"] for item in items
            ),
            "semantic_yes_queries": semantic_yes_queries,
            "semantic_yes_rate_among_candidate_queries": (
                round(semantic_yes_queries / candidate_queries, 6)
                if candidate_queries
                else 0.0
            ),
        }

    expected = len(jobs)
    completed = len(normalized)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": VERIFIER_DEV_SEMANTIC_REVIEW_PROTOCOL,
        "status": status,
        "coverage": {
            "expected_candidates": expected,
            "completed_candidates": completed,
            "remaining_candidates": max(expected - completed, 0),
            "completion_rate": (
                round(completed / expected, 6) if expected else 1.0
            ),
            "candidate_queries": len(jobs_by_query),
            "error_attempts": error_attempts,
        },
        "answers": {
            "parsed": dict(sorted(parsed.items())),
            "exact_answer_rate": (
                round(exact / completed, 6) if completed else 0.0
            ),
            "token_limit_hits": sum(
                bool(item.get("hit_max_new_tokens")) for item in normalized
            ),
        },
        "dev_diagnostics": {
            name: summarize(items)
            for name, items in sorted(by_outcome.items())
        },
        "latency_seconds": {
            "total": round(sum(latencies), 6),
            "mean": _mean(latencies),
            "throughput_candidates_per_second": (
                round(completed / sum(latencies), 6)
                if latencies and sum(latencies) > 0
                else 0.0
            ),
        },
        "cuda_memory_gb": {
            "peak_allocated_max": round(max(memory), 6) if memory else 0.0
        },
        "methodology": {
            "inference_jobs_use_ground_truth": False,
            "diagnostic_labels_used_after_inference": True,
            "candidate_union": (
                "all cached score>=0.30, mask-score>=0.50 candidates, "
                "maximum two per query, before the 0.90 area ablation"
            ),
        },
    }
