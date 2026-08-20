"""Frozen-result failure attribution for grounded answer evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


def _transition(locked_correct: bool, initial_correct: bool | None) -> str:
    if initial_correct is None:
        return "not_compared"
    if locked_correct and initial_correct:
        return "both_correct"
    if locked_correct:
        return "locked_only_correct"
    if initial_correct:
        return "initial_only_correct"
    return "both_wrong"


def analyze_answer_record(
    record: dict[str, Any],
    initial_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attribute one locked prediction without changing its answer policy."""
    task_type = str(record["task_type"])
    policy = record["answer_policy"]
    evaluation = record["evaluation"]
    forced_correct = bool(evaluation["is_correct"])
    abstained = bool(policy["abstained"])
    selective_correct = forced_correct if not abstained else None
    initial_correct = (
        bool(initial_record["evaluation"]["is_correct"])
        if initial_record is not None
        else None
    )
    tags = []
    details: dict[str, Any] = {}

    if task_type == "object_listing":
        target = set(evaluation.get("target_categories", []))
        predicted = set(evaluation.get("predicted_categories", []))
        query = set(record.get("query_plan", {}).get("categories", []))
        missed = target - predicted
        extra = predicted - target
        query_misses = missed - query
        evidence_gate_misses = missed & query
        details.update(
            {
                "target_categories": sorted(target),
                "predicted_categories": sorted(predicted),
                "query_categories": sorted(query),
                "missed_categories": sorted(missed),
                "extra_categories": sorted(extra),
                "query_missed_categories": sorted(query_misses),
                "evidence_gate_missed_categories": sorted(evidence_gate_misses),
            }
        )
        if query_misses:
            tags.append("listing_vlm_query_miss")
        if evidence_gate_misses:
            tags.append("listing_evidence_gate_miss")
        if extra:
            tags.append("listing_supported_false_category")
        if not tags:
            tags.append("correct")

    elif task_type == "object_existence":
        diagnostics = policy.get("diagnostics", {})
        details.update(
            {
                "vlm_answer": diagnostics.get("vlm_answer"),
                "detector_answer": diagnostics.get("detector_answer"),
                "agreement": diagnostics.get("agreement"),
            }
        )
        if abstained and forced_correct:
            tags.append("existence_conservative_disagreement")
        elif abstained and not forced_correct:
            tags.append("existence_error_caught_by_disagreement")
        elif not forced_correct:
            tags.append("existence_consensus_failure")
        else:
            tags.append("correct")

    elif task_type == "spatial_relation":
        status = str(policy["status"])
        details.update(
            {
                "missing_categories": policy.get("diagnostics", {}).get(
                    "missing_categories", []
                ),
                "dx_normalized": policy.get("diagnostics", {}).get(
                    "dx_normalized"
                ),
                "dy_normalized": policy.get("diagnostics", {}).get(
                    "dy_normalized"
                ),
                "dominance": policy.get("diagnostics", {}).get("dominance"),
            }
        )
        if status == "insufficient_evidence":
            tags.append("spatial_missing_evidence")
        elif status == "ambiguous_geometry":
            tags.append("spatial_ambiguous_geometry")
        elif not forced_correct:
            tags.append("spatial_wrong_supported_relation")
        else:
            tags.append("correct")
        if initial_correct and not forced_correct:
            tags.append("spatial_stricter_gate_regression")
        if forced_correct and initial_correct is False:
            tags.append("spatial_locked_policy_gain")
    else:
        tags.append("correct" if forced_correct else "unattributed_error")

    severity = (
        (4 if not forced_correct else 0)
        + (4 if selective_correct is False else 0)
        + (1 if abstained else 0)
        + len(details.get("missed_categories", []))
        + len(details.get("extra_categories", []))
    )
    return {
        "id": record["id"],
        "image": record.get("image"),
        "image_id": int(record["image_id"]),
        "task_type": task_type,
        "question": record.get("question"),
        "gt_answer": record.get("gt_answer"),
        "forced_answer": policy.get("forced_answer"),
        "selective_answer": policy.get("selective_answer"),
        "status": policy.get("status"),
        "forced_correct": forced_correct,
        "abstained": abstained,
        "selective_correct": selective_correct,
        "initial_forced_correct": initial_correct,
        "transition_from_initial": _transition(forced_correct, initial_correct),
        "failure_tags": tags,
        "selected_evidence_count": len(policy.get("selected_evidence", [])),
        "accepted_evidence_count": len(policy.get("accepted_evidence", [])),
        "rejected_evidence_count": len(policy.get("rejected_evidence", [])),
        "severity_score": severity,
        **details,
    }


def aggregate_answer_analysis(
    analyses: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate frozen answer failures and initial-to-locked transitions."""
    analyses = list(analyses)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in analyses:
        groups[item["task_type"]].append(item)

    def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
        answered = [item for item in records if not item["abstained"]]
        return {
            "count": len(records),
            "forced_errors": sum(not item["forced_correct"] for item in records),
            "abstentions": sum(item["abstained"] for item in records),
            "selective_errors": sum(
                item["selective_correct"] is False for item in records
            ),
            "status_counts": dict(
                sorted(Counter(str(item["status"]) for item in records).items())
            ),
            "failure_tag_counts": dict(
                sorted(
                    Counter(
                        tag
                        for item in records
                        for tag in item["failure_tags"]
                        if tag != "correct"
                    ).items()
                )
            ),
            "transition_counts": dict(
                sorted(
                    Counter(
                        item["transition_from_initial"] for item in records
                    ).items()
                )
            ),
            "answered": len(answered),
        }

    listing = groups.get("object_listing", [])
    summary = {
        "overall": summarize(analyses),
        "tasks": {
            task: summarize(records) for task, records in sorted(groups.items())
        },
        "listing_categories": {
            "query_missed": dict(
                sorted(
                    Counter(
                        category
                        for item in listing
                        for category in item.get("query_missed_categories", [])
                    ).items()
                )
            ),
            "evidence_gate_missed": dict(
                sorted(
                    Counter(
                        category
                        for item in listing
                        for category in item.get(
                            "evidence_gate_missed_categories", []
                        )
                    ).items()
                )
            ),
            "extra": dict(
                sorted(
                    Counter(
                        category
                        for item in listing
                        for category in item.get("extra_categories", [])
                    ).items()
                )
            ),
        },
    }
    return summary
