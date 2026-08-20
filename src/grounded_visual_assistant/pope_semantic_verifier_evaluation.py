"""Paired POPE evaluation for the V2 semantic crop verifier."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Iterable, Mapping

from .evaluation import parse_yes_no
from .grounding_answer_verifier import compact_grounding_result
from .pope_evaluation import evaluate_answer
from .pope_verifier_evaluation import aggregate_pope_verifier_metrics
from .semantic_answer_verifier import (
    SemanticAnswerVerifierConfig,
    select_semantic_candidates,
    semantic_candidate_key,
    semantic_review_question,
    verify_binary_answer_v2,
)


POPE_SEMANTIC_VERIFIER_BATCH_PROTOCOL = (
    "pope_grounding_semantic_rescue_batch_v2"
)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def build_semantic_review_jobs(
    groups: Iterable[Mapping[str, Any]],
    *,
    records_by_id: Mapping[str, Mapping[str, Any]],
    evidence_by_key: Mapping[str, Mapping[str, Any]],
    config: SemanticAnswerVerifierConfig,
) -> list[dict[str, Any]]:
    """Build GT-free crop review jobs for negative baseline queries."""
    jobs = []
    for raw_group in groups:
        group = dict(raw_group)
        query_key = str(group["query_key"])
        records = [
            records_by_id[str(sample_id)]
            for sample_id in group["baseline_ids"]
        ]
        answers = {
            parse_yes_no(str(record["prediction"])) for record in records
        }
        if None in answers or len(answers) != 1:
            raise ValueError(
                f"Baseline answers disagree within query {query_key}: "
                f"{answers}."
            )
        if answers == {"yes"}:
            continue
        if query_key not in evidence_by_key:
            raise ValueError(f"Grounding evidence is missing query {query_key}.")
        evidence_record = evidence_by_key[query_key]
        grounding = evidence_record["grounding"]
        candidates, _ = select_semantic_candidates(
            grounding.get("annotations", []),
            target=str(group["object"]),
            image_width=int(grounding["img_width"]),
            image_height=int(grounding["img_height"]),
            config=config,
        )
        for candidate in candidates:
            annotation_index = int(candidate["annotation_index"])
            jobs.append(
                {
                    "candidate_key": semantic_candidate_key(
                        query_key, annotation_index
                    ),
                    "query_key": query_key,
                    "baseline_ids": list(group["baseline_ids"]),
                    "strategies": list(group["strategies"]),
                    "image": str(group["image"]),
                    "image_id": int(group["image_id"]),
                    "question": str(group["question"]),
                    "object": str(group["object"]),
                    "annotation_index": annotation_index,
                    "grounding_score": candidate["grounding_score"],
                    "mask_score": candidate["mask_score"],
                    "mask_area_ratio": candidate["mask_area_ratio"],
                    "bbox": list(candidate["bbox"]),
                    "semantic_question": semantic_review_question(
                        str(group["object"])
                    ),
                }
            )
    candidate_keys = [str(item["candidate_key"]) for item in jobs]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("Semantic review jobs contain duplicate keys.")
    return jobs


def required_semantic_candidate_keys(
    baseline: Mapping[str, Any],
    evidence_record: Mapping[str, Any],
    *,
    config: SemanticAnswerVerifierConfig,
) -> list[str]:
    """Return the review keys required to materialize one prediction."""
    baseline_answer = parse_yes_no(str(baseline["prediction"]))
    if baseline_answer is None:
        raise ValueError(
            f"Invalid baseline Yes/No answer for {baseline.get('id')}."
        )
    if baseline_answer == "yes":
        return []
    grounding = evidence_record["grounding"]
    candidates, _ = select_semantic_candidates(
        grounding.get("annotations", []),
        target=str(baseline["object"]),
        image_width=int(grounding["img_width"]),
        image_height=int(grounding["img_height"]),
        config=config,
    )
    query_key = str(evidence_record["query_key"])
    return [
        semantic_candidate_key(query_key, int(item["annotation_index"]))
        for item in candidates
    ]


def build_semantic_verified_prediction(
    baseline: Mapping[str, Any],
    evidence_record: Mapping[str, Any],
    *,
    reviews_by_key: Mapping[str, Mapping[str, Any]],
    config: SemanticAnswerVerifierConfig,
) -> dict[str, Any]:
    """Build one paired V2 record from frozen baseline/evidence/reviews."""
    baseline = dict(baseline)
    baseline_evaluation = evaluate_answer(
        str(baseline["prediction"]), str(baseline["gt_answer"])
    )
    if baseline_evaluation != baseline.get("evaluation"):
        raise RuntimeError(
            f"Saved baseline evaluation mismatch for {baseline.get('id')}."
        )
    required_keys = required_semantic_candidate_keys(
        baseline, evidence_record, config=config
    )
    missing = [key for key in required_keys if key not in reviews_by_key]
    if missing:
        raise ValueError(
            f"Semantic reviews are incomplete for {baseline.get('id')}: "
            f"{missing}."
        )
    reviews = [dict(reviews_by_key[key]) for key in required_keys]
    grounding = evidence_record["grounding"]
    verification = verify_binary_answer_v2(
        str(baseline["prediction"]),
        target=str(baseline["object"]),
        annotations=grounding.get("annotations", []),
        image_width=int(grounding["img_width"]),
        image_height=int(grounding["img_height"]),
        semantic_reviews=reviews,
        config=config,
    )
    final_answer = str(verification["final_answer"])
    evaluation = evaluate_answer(final_answer, str(baseline["gt_answer"]))
    baseline_answer = parse_yes_no(str(baseline["prediction"]))
    grounding_latency = (
        float((grounding.get("latency_seconds") or {}).get("total", 0.0))
        if baseline_answer == "no"
        else 0.0
    )
    semantic_latency = sum(
        float(
            item.get(
                "end_to_end_latency_seconds",
                item.get("latency_seconds", 0.0),
            )
        )
        for item in reviews
    )
    semantic_peak = max(
        (
            float(item["cuda_peak_memory_allocated_gb"])
            for item in reviews
            if item.get("cuda_peak_memory_allocated_gb") is not None
        ),
        default=0.0,
    )
    grounding_peak = float(
        evidence_record.get(
            "cuda_peak_memory_allocated_gb",
            grounding.get("cuda_peak_memory_allocated_gb", 0.0),
        )
        or 0.0
    )
    verification_latency = grounding_latency + semantic_latency
    baseline_latency = float(baseline.get("latency_seconds", 0.0))
    return {
        "id": baseline["id"],
        "strategy": baseline["strategy"],
        "question_id": baseline.get("question_id"),
        "image": baseline["image"],
        "image_id": baseline["image_id"],
        "question": baseline["question"],
        "object": baseline["object"],
        "gt_answer": baseline["gt_answer"],
        "model": baseline.get("model"),
        "query_key": evidence_record["query_key"],
        "baseline_prediction": baseline["prediction"],
        "baseline_evaluation": baseline_evaluation,
        "prediction": final_answer,
        "evaluation": evaluation,
        "verification": verification,
        "grounding": compact_grounding_result(grounding),
        "semantic_candidate_keys": required_keys,
        "semantic_review_count": len(reviews),
        "baseline_latency_seconds": round(baseline_latency, 6),
        "grounding_latency_seconds": round(grounding_latency, 6),
        "semantic_review_latency_seconds": round(semantic_latency, 6),
        "verification_latency_seconds": round(
            verification_latency, 6
        ),
        "projected_end_to_end_latency_seconds": round(
            baseline_latency + verification_latency, 6
        ),
        "baseline_generated_tokens": baseline.get("generated_tokens"),
        "semantic_generated_tokens": sum(
            int(item.get("generated_tokens", 0)) for item in reviews
        ),
        "baseline_cuda_peak_memory_allocated_gb": baseline.get(
            "cuda_peak_memory_allocated_gb"
        ),
        "baseline_cuda_memory_reserved_gb": baseline.get(
            "cuda_memory_reserved_gb"
        ),
        "verification_cuda_peak_memory_allocated_gb": round(
            max(grounding_peak, semantic_peak), 6
        ),
    }


def aggregate_pope_semantic_verifier_metrics(
    predictions: Iterable[Mapping[str, Any]],
    *,
    expected_samples: int,
    expected_queries: int,
    completed_queries: int,
    expected_reviews: int,
    completed_reviews: int,
    error_attempts: int = 0,
    status: str = "completed",
) -> dict[str, Any]:
    """Aggregate paired V2 metrics plus crop-review diagnostics."""
    records = [dict(item) for item in predictions]
    metrics = aggregate_pope_verifier_metrics(
        records,
        expected_samples=expected_samples,
        expected_queries=expected_queries,
        completed_queries=completed_queries,
        error_attempts=error_attempts,
        status=status,
        protocol=POPE_SEMANTIC_VERIFIER_BATCH_PROTOCOL,
    )
    unique_reviews: dict[str, dict[str, Any]] = {}
    unique_queries: dict[str, dict[str, Any]] = {}
    for item in records:
        unique_queries.setdefault(str(item["query_key"]), item)
        for review in item["verification"].get("semantic_reviews", []):
            unique_reviews.setdefault(str(review["candidate_key"]), review)
    parsed = Counter(
        str(item.get("parsed_answer") or "invalid")
        for item in unique_reviews.values()
    )
    exact_valid = sum(
        bool(item.get("exact_answer")) for item in unique_reviews.values()
    )
    review_latencies = [
        float(
            item.get(
                "end_to_end_latency_seconds",
                item.get("latency_seconds", 0.0),
            )
        )
        for item in unique_reviews.values()
    ]
    query_statuses = Counter(
        str(item["verification"]["status"])
        for item in unique_queries.values()
    )
    metrics["semantic_review"] = {
        "expected_candidates": expected_reviews,
        "completed_candidates": completed_reviews,
        "remaining_candidates": max(
            expected_reviews - completed_reviews, 0
        ),
        "completion_rate": (
            round(completed_reviews / expected_reviews, 6)
            if expected_reviews
            else 1.0
        ),
        "parsed_answers": dict(sorted(parsed.items())),
        "exact_answer_rate": (
            round(exact_valid / len(unique_reviews), 6)
            if unique_reviews
            else 1.0
        ),
        "latency_total": round(sum(review_latencies), 6),
        "latency_mean": _mean(review_latencies),
        "query_status_counts": dict(sorted(query_statuses.items())),
    }
    metrics["method"] = {
        "grounding_evidence": "Grounding DINO + SAM 2.1 cached evidence",
        "geometry_gate": "maximum mask-area ratio",
        "semantic_gate": "deterministic Qwen candidate-crop Yes/No review",
        "positive_answer_rule": (
            "positive baselines are never demoted by detector silence"
        ),
    }
    return metrics
