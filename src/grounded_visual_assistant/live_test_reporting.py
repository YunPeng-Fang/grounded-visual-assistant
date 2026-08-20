"""Read-only reporting helpers for the locked live-pipeline Test run."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


GENERALIZATION_METRICS = {
    "overall_exact_accuracy": ("overall", "exact_accuracy"),
    "listing_macro_f1": ("tasks", "object_listing", "macro_f1"),
    "listing_exact_accuracy": (
        "tasks",
        "object_listing",
        "exact_accuracy",
    ),
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
    "relation_balanced_accuracy": (
        "tasks",
        "spatial_relation",
        "balanced_accuracy",
    ),
    "relation_parse_valid_rate": (
        "tasks",
        "spatial_relation",
        "parse_valid_rate",
    ),
    "schema_valid_rate": ("structured_targets", "schema_valid_rate"),
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
    "mean_latency_seconds": ("latency_seconds", "mean"),
}


def _nested(payload: Mapping[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    return float(value)


def build_generalization_rows(
    dev_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return stable Dev-to-Test rows for all primary pipeline layers."""
    rows = []
    for metric, path in GENERALIZATION_METRICS.items():
        dev_value = _nested(dev_metrics, path)
        test_value = _nested(test_metrics, path)
        rows.append(
            {
                "metric": metric,
                "dev": round(dev_value, 6),
                "test": round(test_value, 6),
                "delta_test_minus_dev": round(test_value - dev_value, 6),
            }
        )
    return rows


def analyze_live_test_predictions(
    records: Iterable[Mapping[str, Any]],
    *,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create compact per-sample diagnostics without copying masks or RLE."""
    values = [dict(item) for item in records]
    per_sample = []
    failure_counts: Counter[str] = Counter()
    task_failures: dict[str, Counter[str]] = {}
    for item in values:
        task_type = str(item["task_type"])
        task_failures.setdefault(task_type, Counter())
        evaluation = item["evaluation"]
        target = item["target_evaluation"]
        box = item["evidence_evaluation"]
        mask = item["mask_evaluation"]
        vlm = item.get("vlm_output", {})
        generated_tokens = int(vlm.get("generated_tokens") or 0)
        flags = []
        if not bool(evaluation["is_correct"]):
            flags.append("answer_incorrect")
        if not bool(vlm.get("schema_valid")):
            flags.append("schema_invalid")
        if generated_tokens >= max_new_tokens:
            flags.append("token_limit_hit")
        if int(target.get("fp", 0)) > 0:
            flags.append("target_false_positive")
        if int(target.get("fn", 0)) > 0:
            flags.append("target_miss")
        if int(box.get("fp", 0)) > 0:
            flags.append("box_false_positive")
        if int(box.get("fn", 0)) > 0:
            flags.append("box_miss")
        if int(mask.get("fp", 0)) > 0:
            flags.append("mask_false_positive")
        if int(mask.get("fn", 0)) > 0:
            flags.append("mask_miss")
        if not bool(item.get("evidence_supported")):
            flags.append("evidence_unsupported")
        if not bool(item.get("evidence_complete")):
            flags.append("evidence_incomplete")
        if not bool(item.get("end_to_end_success")):
            flags.append("end_to_end_any_failure")
        if not bool(item.get("end_to_end_complete_success")):
            flags.append("end_to_end_complete_failure")

        severity = (
            3 * int("answer_incorrect" in flags)
            + 3 * int("schema_invalid" in flags)
            + 2 * int("token_limit_hit" in flags)
            + 2 * int("end_to_end_any_failure" in flags)
            + int("target_miss" in flags)
            + int("target_false_positive" in flags)
            + int("box_miss" in flags)
            + int("box_false_positive" in flags)
            + int("mask_miss" in flags)
            + int("mask_false_positive" in flags)
        )
        failure_counts.update(flags)
        task_failures[task_type].update(flags)
        per_sample.append(
            {
                "id": str(item["id"]),
                "image_id": int(item["image_id"]),
                "task_type": task_type,
                "gt_answer": item["gt_answer"],
                "prediction": item["prediction"],
                "answer_score": float(evaluation["score"]),
                "answer_correct": bool(evaluation["is_correct"]),
                "answer_parse_valid": evaluation.get("parse_valid"),
                "schema_valid": bool(vlm.get("schema_valid")),
                "parse_source": vlm.get("parse_source"),
                "generated_tokens": generated_tokens,
                "hit_max_new_tokens": generated_tokens >= max_new_tokens,
                "targets": list(item.get("targets", [])),
                "target_precision": float(target.get("precision", 0.0)),
                "target_recall": float(target.get("recall", 0.0)),
                "target_f1": float(target.get("f1", 0.0)),
                "target_fp": int(target.get("fp", 0)),
                "target_fn": int(target.get("fn", 0)),
                "box_precision": float(box.get("precision", 0.0)),
                "box_recall": float(box.get("recall", 0.0)),
                "box_f1": float(box.get("f1", 0.0)),
                "box_fp": int(box.get("fp", 0)),
                "box_fn": int(box.get("fn", 0)),
                "mask_precision": float(mask.get("precision", 0.0)),
                "mask_recall": float(mask.get("recall", 0.0)),
                "mask_f1": float(mask.get("f1", 0.0)),
                "mask_fp": int(mask.get("fp", 0)),
                "mask_fn": int(mask.get("fn", 0)),
                "evidence_required": bool(item.get("evidence_required")),
                "evidence_supported": bool(item.get("evidence_supported")),
                "evidence_complete": bool(item.get("evidence_complete")),
                "end_to_end_success": bool(item.get("end_to_end_success")),
                "end_to_end_complete_success": bool(
                    item.get("end_to_end_complete_success")
                ),
                "latency_seconds": float(item["latency_seconds"]),
                "flags": flags,
                "severity": severity,
            }
        )
    per_sample.sort(key=lambda item: (-item["severity"], item["id"]))
    summary = {
        "samples": len(values),
        "failure_counts": dict(sorted(failure_counts.items())),
        "task_failure_counts": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(task_failures.items())
        },
        "schema_invalid_count": failure_counts["schema_invalid"],
        "token_limit_hit_count": failure_counts["token_limit_hit"],
        "top_failures": per_sample[:20],
    }
    return summary, per_sample


def relation_confusion_rows(
    metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Flatten the held-out relation confusion matrix."""
    confusion = metrics["tasks"]["spatial_relation"]["confusion"]
    rows = []
    for target, predictions in confusion.items():
        for prediction, count in predictions.items():
            rows.append(
                {
                    "target": target,
                    "prediction": prediction,
                    "count": int(count),
                }
            )
    return rows


def render_live_test_report(summary: Mapping[str, Any]) -> str:
    """Render the immutable held-out result in Markdown."""
    result = summary["test_result"]
    tasks = result["tasks"]
    lines = [
        "# Locked Live-Pipeline Test240 Final Report",
        "",
        "## Integrity",
        "",
        f"- Coverage: {summary['integrity']['coverage']}/240",
        f"- Prediction errors: {summary['integrity']['prediction_errors']}",
        f"- Duplicate IDs: {summary['integrity']['duplicate_ids']}",
        f"- Missing artifacts: {summary['integrity']['missing_artifacts']}",
        f"- Saved metrics replayed: "
        f"`{summary['integrity']['saved_metrics_replayed']}`",
        "",
        "## Held-Out Results",
        "",
        "| Metric | Test |",
        "|---|---:|",
        f"| Overall exact accuracy | {result['overall']['exact_accuracy']:.6f} |",
        f"| Listing macro F1 | {tasks['object_listing']['macro_f1']:.6f} |",
        f"| Existence accuracy | "
        f"{tasks['object_existence']['exact_accuracy']:.6f} |",
        f"| Relation accuracy | "
        f"{tasks['spatial_relation']['exact_accuracy']:.6f} |",
        f"| Relation balanced accuracy | "
        f"{tasks['spatial_relation']['balanced_accuracy']:.6f} |",
        f"| Schema valid rate | "
        f"{result['structured_targets']['schema_valid_rate']:.6f} |",
        f"| Target micro F1 | "
        f"{result['structured_targets']['micro_f1']:.6f} |",
        f"| Box IoU50 micro F1 | {result['box_micro_f1']:.6f} |",
        f"| Mask IoU50 micro F1 | {result['mask_micro_f1']:.6f} |",
        f"| End-to-end complete success | "
        f"{result['end_to_end']['answer_and_complete_evidence_success_rate']:.6f} |",
        f"| Mean latency, seconds | {result['mean_latency_seconds']:.6f} |",
        "",
        "## Dev-to-Test",
        "",
        "| Metric | Dev | Test | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["generalization"]:
        lines.append(
            f"| {row['metric']} | {row['dev']:.6f} | "
            f"{row['test']:.6f} | {row['delta_test_minus_dev']:+.6f} |"
        )
    failures = summary["failure_analysis"]
    lines.extend(
        [
            "",
            "## Failure Audit",
            "",
            f"- Schema-invalid responses: {failures['schema_invalid_count']}",
            f"- Token-limit hits: {failures['token_limit_hit_count']}",
            f"- Answer errors: "
            f"{failures['failure_counts'].get('answer_incorrect', 0)}",
            f"- Target misses: "
            f"{failures['failure_counts'].get('target_miss', 0)}",
            f"- Box misses: {failures['failure_counts'].get('box_miss', 0)}",
            f"- Mask misses: {failures['failure_counts'].get('mask_miss', 0)}",
            "",
            "## Decision",
            "",
            "- The selected v2 policy remains frozen.",
            "- Test-driven prompt or threshold tuning is prohibited.",
            "- The single truncated listing is reported as a held-out limitation.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
