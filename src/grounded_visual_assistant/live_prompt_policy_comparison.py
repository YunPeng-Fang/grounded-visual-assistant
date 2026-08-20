"""Paired Dev comparison for live-pipeline prompt policies."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .relation_prompt_comparison import exact_mcnemar_p_value


ACCEPTANCE_METRICS = {
    "schema_valid_rate_min": (
        "structured_targets",
        "schema_valid_rate",
    ),
    "relation_parse_valid_rate_min": (
        "tasks",
        "spatial_relation",
        "parse_valid_rate",
    ),
    "relation_exact_accuracy_min": (
        "tasks",
        "spatial_relation",
        "exact_accuracy",
    ),
    "listing_macro_f1_min": (
        "tasks",
        "object_listing",
        "macro_f1",
    ),
    "existence_exact_accuracy_min": (
        "tasks",
        "object_existence",
        "exact_accuracy",
    ),
    "target_micro_f1_min": (
        "structured_targets",
        "micro_f1",
    ),
    "box_micro_f1_min": (
        "required_evidence_box_metrics",
        "box_iou_50",
        "micro_f1",
    ),
    "mask_micro_f1_min": (
        "required_evidence_mask_iou_50",
        "micro_f1",
    ),
    "end_to_end_any_success_rate_min": (
        "end_to_end",
        "overall",
        "answer_and_any_evidence_success_rate",
    ),
    "end_to_end_complete_success_rate_min": (
        "end_to_end",
        "overall",
        "answer_and_complete_evidence_success_rate",
    ),
}

COMPARISON_METRICS = {
    "overall_exact_accuracy": ("overall", "exact_accuracy"),
    "schema_valid_rate": ("structured_targets", "schema_valid_rate"),
    "listing_macro_f1": ("tasks", "object_listing", "macro_f1"),
    "existence_exact_accuracy": (
        "tasks",
        "object_existence",
        "exact_accuracy",
    ),
    "relation_exact_accuracy": (
        "tasks",
        "spatial_relation",
        "exact_accuracy",
    ),
    "relation_parse_valid_rate": (
        "tasks",
        "spatial_relation",
        "parse_valid_rate",
    ),
    "target_micro_f1": ("structured_targets", "micro_f1"),
    "box_micro_f1": (
        "required_evidence_box_metrics",
        "box_iou_50",
        "micro_f1",
    ),
    "mask_micro_f1": (
        "required_evidence_mask_iou_50",
        "micro_f1",
    ),
    "end_to_end_any_success_rate": (
        "end_to_end",
        "overall",
        "answer_and_any_evidence_success_rate",
    ),
    "end_to_end_complete_success_rate": (
        "end_to_end",
        "overall",
        "answer_and_complete_evidence_success_rate",
    ),
    "latency_mean_seconds": ("latency_seconds", "mean"),
    "peak_cuda_memory_gb": ("cuda_memory_gb", "peak_allocated_max"),
}


def _nested(payload: Mapping[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError("Missing metric: " + ".".join(path))
        value = value[key]
    return float(value)


def _records_by_id(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    values = [dict(item) for item in records]
    by_id = {str(item["id"]): item for item in values}
    if len(values) != len(by_id):
        raise ValueError("Predictions contain duplicate IDs.")
    return by_id


def _paired_counts(
    ids: list[str],
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    both_correct = 0
    baseline_only = 0
    candidate_only = 0
    for sample_id in ids:
        baseline_correct = bool(
            baseline[sample_id]["evaluation"]["is_correct"]
        )
        candidate_correct = bool(
            candidate[sample_id]["evaluation"]["is_correct"]
        )
        if baseline_correct and candidate_correct:
            both_correct += 1
        elif baseline_correct:
            baseline_only += 1
        elif candidate_correct:
            candidate_only += 1
    both_wrong = len(ids) - both_correct - baseline_only - candidate_only
    return {
        "count": len(ids),
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "both_wrong": both_wrong,
        "net_correct": candidate_only - baseline_only,
        "mcnemar_exact_p_value": exact_mcnemar_p_value(
            baseline_only, candidate_only
        ),
    }


def compare_live_prompt_policies(
    baseline_records: Iterable[Mapping[str, Any]],
    candidate_records: Iterable[Mapping[str, Any]],
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    *,
    baseline_policy: str = "generic-v1",
    candidate_policy: str = "task-aware-coco-v1",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare complete paired Dev predictions and evaluate frozen gates."""
    baseline = _records_by_id(baseline_records)
    candidate = _records_by_id(candidate_records)
    if set(baseline) != set(candidate):
        raise RuntimeError("Baseline and candidate prediction IDs differ.")
    ordered_ids = sorted(baseline)
    if not ordered_ids:
        raise RuntimeError("No paired predictions were found.")
    if {str(item.get("split")) for item in candidate.values()} != {"dev"}:
        raise RuntimeError("Live prompt policy comparison is Dev-only.")

    transitions = []
    for sample_id in ordered_ids:
        left = baseline[sample_id]
        right = candidate[sample_id]
        for field in (
            "image_id",
            "question",
            "task_type",
            "gt_answer",
            "source",
        ):
            if left.get(field) != right.get(field):
                raise RuntimeError(
                    f"Paired field {field} differs for {sample_id}."
                )
        left_correct = bool(left["evaluation"]["is_correct"])
        right_correct = bool(right["evaluation"]["is_correct"])
        transition = (
            "both_correct"
            if left_correct and right_correct
            else "baseline_only_correct"
            if left_correct
            else "candidate_only_correct"
            if right_correct
            else "both_wrong"
        )
        transitions.append(
            {
                "id": sample_id,
                "task_type": left["task_type"],
                "question": left["question"],
                "gt_answer": left["gt_answer"],
                "baseline_prediction": left.get("prediction"),
                "candidate_prediction": right.get("prediction"),
                "baseline_targets": left.get("targets", []),
                "candidate_targets": right.get("targets", []),
                "baseline_correct": left_correct,
                "candidate_correct": right_correct,
                "baseline_schema_valid": bool(
                    left.get("vlm_output", {}).get("schema_valid")
                ),
                "candidate_schema_valid": bool(
                    right.get("vlm_output", {}).get("schema_valid")
                ),
                "baseline_end_to_end_success": bool(
                    left.get("end_to_end_success")
                ),
                "candidate_end_to_end_success": bool(
                    right.get("end_to_end_success")
                ),
                "transition": transition,
            }
        )

    metric_comparison = {}
    for name, path in COMPARISON_METRICS.items():
        baseline_value = _nested(baseline_metrics, path)
        candidate_value = _nested(candidate_metrics, path)
        metric_comparison[name] = {
            "baseline": round(baseline_value, 6),
            "candidate": round(candidate_value, 6),
            "delta": round(candidate_value - baseline_value, 6),
        }

    unknown_gates = set(acceptance) - set(ACCEPTANCE_METRICS)
    missing_gates = set(ACCEPTANCE_METRICS) - set(acceptance)
    if unknown_gates or missing_gates:
        raise ValueError(
            f"Acceptance gates differ from protocol; unknown={sorted(unknown_gates)}, "
            f"missing={sorted(missing_gates)}."
        )
    gates = {}
    for name, path in ACCEPTANCE_METRICS.items():
        threshold = float(acceptance[name])
        observed = _nested(candidate_metrics, path)
        gates[name] = {
            "threshold": threshold,
            "observed": round(observed, 6),
            "passed": observed >= threshold,
        }

    by_task = {}
    for task_type in sorted(
        {str(item["task_type"]) for item in candidate.values()}
    ):
        task_ids = [
            sample_id
            for sample_id in ordered_ids
            if candidate[sample_id]["task_type"] == task_type
        ]
        by_task[task_type] = _paired_counts(
            task_ids, baseline, candidate
        )

    all_gates_passed = all(item["passed"] for item in gates.values())
    return (
        {
            "status": "completed",
            "policies": {
                "baseline": baseline_policy,
                "candidate": candidate_policy,
            },
            "coverage": {
                "paired_questions": len(ordered_ids),
                "split": "dev",
                "tasks": dict(
                    sorted(
                        Counter(
                            str(item["task_type"])
                            for item in candidate.values()
                        ).items()
                    )
                ),
            },
            "metrics": metric_comparison,
            "paired": {
                "overall": _paired_counts(
                    ordered_ids, baseline, candidate
                ),
                "tasks": by_task,
            },
            "acceptance": {
                "all_gates_passed": all_gates_passed,
                "gates": gates,
                "decision": (
                    "accept_" + candidate_policy.replace("-", "_")
                    if all_gates_passed
                    else "reject_or_revise_candidate"
                ),
            },
        },
        transitions,
    )


def render_live_prompt_policy_report(summary: Mapping[str, Any]) -> str:
    """Render a concise Markdown record of the paired experiment."""
    lines = [
        "# Live Pipeline Prompt Policy Comparison",
        "",
        f"Paired Dev questions: {summary['coverage']['paired_questions']}",
        "",
        "## Metrics",
        "",
        (
            f"| Metric | {summary['policies']['baseline']} | "
            f"{summary['policies']['candidate']} | Delta |"
        ),
        "|---|---:|---:|---:|",
    ]
    for name, values in summary["metrics"].items():
        lines.append(
            f"| {name} | {values['baseline']:.6f} | "
            f"{values['candidate']:.6f} | {values['delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Answers",
            "",
            (
                "| Task | Both correct | "
                f"{summary['policies']['baseline']} only | "
                f"{summary['policies']['candidate']} only | "
                "Both wrong | McNemar p |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    paired = {
        "overall": summary["paired"]["overall"],
        **summary["paired"]["tasks"],
    }
    for name, values in paired.items():
        lines.append(
            f"| {name} | {values['both_correct']} | "
            f"{values['baseline_only_correct']} | "
            f"{values['candidate_only_correct']} | "
            f"{values['both_wrong']} | "
            f"{values['mcnemar_exact_p_value']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered Gates",
            "",
            "| Gate | Minimum | Observed | Passed |",
            "|---|---:|---:|:---:|",
        ]
    )
    for name, values in summary["acceptance"]["gates"].items():
        lines.append(
            f"| {name} | {values['threshold']:.6f} | "
            f"{values['observed']:.6f} | "
            f"{'yes' if values['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{summary['acceptance']['decision']}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
