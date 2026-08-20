"""Failure analysis for the frozen cross-dataset VLM benchmark."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .evaluation import aggregate_metrics, score_prediction


REFUSAL_MARKERS = (
    "cannot determine",
    "can't determine",
    "not possible to determine",
    "does not contain",
    "doesn't contain",
    "there is no",
    "there are no",
    "not visible",
    "no visible",
)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def _subset_metrics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "mean_score": _mean(
            float(item["evaluation"]["score"]) for item in records
        ),
        "exact_accuracy": _mean(
            float(item["evaluation"]["is_correct"]) for item in records
        ),
    }


def analyze_hard_prediction(
    sample: Mapping[str, Any], prediction: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify one saved prediction without running a model."""
    task_type = str(sample["task_type"])
    evaluation = prediction["evaluation"]
    answer = str(prediction.get("prediction", ""))
    normalized_answer = answer.lower()
    flags = []
    hit_max = bool(prediction.get("hit_max_new_tokens"))
    if hit_max:
        flags.append("hit_max_new_tokens")
    if not evaluation["is_correct"]:
        flags.append("incorrect")

    missed_categories: list[str] = []
    extra_categories: list[str] = []
    if task_type == "object_listing":
        target = set(evaluation.get("target_categories", []))
        predicted = set(evaluation.get("predicted_categories", []))
        missed_categories = sorted(target - predicted)
        extra_categories = sorted(predicted - target)
        if missed_categories:
            flags.append("listing_missed_category")
        if extra_categories:
            flags.append("listing_extra_category")
    elif task_type == "object_existence" and not evaluation["is_correct"]:
        if evaluation.get("parsed_target") == "no":
            flags.append("existence_false_positive")
        else:
            flags.append("existence_false_negative")
    elif task_type == "spatial_relation":
        if not evaluation.get("parse_valid"):
            flags.append("relation_parse_invalid")
        elif not evaluation["is_correct"]:
            flags.append("relation_wrong_direction")
        if any(marker in normalized_answer for marker in REFUSAL_MARKERS):
            flags.append("relation_refusal_or_absence_claim")

    severity = (
        4 * int("relation_parse_invalid" in flags)
        + 3 * int("existence_false_positive" in flags)
        + 2 * int(hit_max)
        + 2 * len(missed_categories)
        + len(extra_categories)
        + int("relation_wrong_direction" in flags)
        + int("existence_false_negative" in flags)
    )
    return {
        "id": sample["id"],
        "sample_id": sample.get("sample_id", sample.get("image_id")),
        "source": sample.get("source"),
        "split": sample.get("split"),
        "task_type": task_type,
        "question": sample["question"],
        "gt_answer": sample["gt_answer"],
        "prediction": answer,
        "score": float(evaluation["score"]),
        "is_correct": bool(evaluation["is_correct"]),
        "parsed_target": evaluation.get("parsed_target"),
        "parsed_prediction": evaluation.get("parsed_prediction"),
        "parse_valid": evaluation.get("parse_valid"),
        "hit_max_new_tokens": hit_max,
        "generated_tokens": prediction.get("generated_tokens"),
        "missed_categories": missed_categories,
        "extra_categories": extra_categories,
        "flags": flags,
        "severity": severity,
    }


def analyze_hard_vlm_predictions(
    dataset_records: Iterable[Mapping[str, Any]],
    prediction_records: Iterable[Mapping[str, Any]],
    *,
    required_split: str = "dev",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate, rescore, and summarize a complete frozen prediction set."""
    if required_split not in {"dev", "test"}:
        raise ValueError(f"Unsupported analysis split: {required_split}")
    dataset_records = list(dataset_records)
    prediction_records = list(prediction_records)
    dataset_by_id = {str(item["id"]): item for item in dataset_records}
    prediction_by_id = {str(item["id"]): item for item in prediction_records}
    if len(dataset_by_id) != len(dataset_records):
        raise ValueError("Dataset contains duplicate question IDs.")
    if len(prediction_by_id) != len(prediction_records):
        raise ValueError("Predictions contain duplicate question IDs.")
    if set(dataset_by_id) != set(prediction_by_id):
        missing = sorted(set(dataset_by_id) - set(prediction_by_id))
        extra = sorted(set(prediction_by_id) - set(dataset_by_id))
        raise RuntimeError(
            f"Prediction coverage mismatch: missing={len(missing)}, "
            f"extra={len(extra)}."
        )
    splits = {str(item.get("split")) for item in dataset_records}
    if splits != {required_split}:
        raise RuntimeError(
            f"Hard failure analysis requires {required_split!r}, found splits: "
            f"{splits}"
        )

    analyses = []
    ordered_predictions = []
    for sample_id in sorted(dataset_by_id):
        sample = dataset_by_id[sample_id]
        prediction = prediction_by_id[sample_id]
        rescored = score_prediction(sample, str(prediction.get("prediction", "")))
        if rescored != prediction.get("evaluation"):
            raise RuntimeError(f"Saved evaluation does not reproduce for {sample_id}.")
        analyses.append(analyze_hard_prediction(sample, prediction))
        ordered_predictions.append(prediction)

    metrics = aggregate_metrics(
        ordered_predictions,
        expected_samples=len(dataset_records),
        error_attempts=0,
        status="completed",
    )
    flags = Counter(flag for item in analyses for flag in item["flags"])

    existence = [
        item for item in ordered_predictions if item["task_type"] == "object_existence"
    ]
    existence_by_polarity = {}
    for polarity, label in ((True, "positive"), (False, "negative")):
        records = [
            item
            for item in existence
            if bool((item.get("metadata") or {}).get("is_positive")) == polarity
        ]
        existence_by_polarity[label] = _subset_metrics(records)

    listings = [
        item for item in ordered_predictions if item["task_type"] == "object_listing"
    ]
    listing_protocols = {}
    for has_negative, label in (
        (True, "with_verified_negative_distractors"),
        (False, "positive_only_vocabulary"),
    ):
        records = [
            item
            for item in listings
            if bool((item.get("metadata") or {}).get("has_negative_distractor"))
            == has_negative
        ]
        listing_protocols[label] = _subset_metrics(records)

    relation_sources = {}
    for source, source_metrics in metrics["sources"].items():
        relation = source_metrics["tasks"].get("spatial_relation")
        if relation is not None:
            source_analyses = [
                item
                for item in analyses
                if item["source"] == source
                and item["task_type"] == "spatial_relation"
            ]
            relation_sources[source] = {
                **relation,
                "refusal_or_absence_claims": sum(
                    "relation_refusal_or_absence_claim" in item["flags"]
                    for item in source_analyses
                ),
                "hit_max_new_tokens": sum(
                    item["hit_max_new_tokens"] for item in source_analyses
                ),
            }

    warnings = []
    for source, relation in relation_sources.items():
        if relation["exact_accuracy"] <= relation["majority_class_baseline_accuracy"]:
            warnings.append(
                f"{source} relation accuracy does not exceed its majority-class baseline."
            )
        if relation["parse_valid_rate"] < 0.9:
            warnings.append(f"{source} relation parse-valid rate is below 0.90.")
    if flags.get("hit_max_new_tokens", 0):
        warnings.append("Some answers reached the generation token limit.")

    ranked = sorted(analyses, key=lambda item: (-item["severity"], item["id"]))
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "coverage": {
            "expected": len(dataset_records),
            "analyzed": len(analyses),
            "split": required_split,
        },
        "overall": metrics["overall"],
        "tasks": metrics["tasks"],
        "sources": metrics["sources"],
        "existence_by_polarity": existence_by_polarity,
        "listing_protocols": listing_protocols,
        "relation_sources": relation_sources,
        "failure_flags": dict(sorted(flags.items())),
        "warnings": warnings,
        "top_failures": [
            {
                key: item[key]
                for key in (
                    "id",
                    "source",
                    "task_type",
                    "gt_answer",
                    "parsed_prediction",
                    "flags",
                    "severity",
                )
            }
            for item in ranked[:20]
        ],
    }
    return summary, ranked


def render_hard_vlm_report(
    summary: Mapping[str, Any], *, predictions_path: str
) -> str:
    """Render the main diagnostics as a compact Markdown report."""
    lines = [
        "# Hard-Dev Qwen3-VL Failure Analysis",
        "",
        f"Predictions: `{predictions_path}`",
        "",
        "## Result Integrity",
        "",
        f"- Analyzed: {summary['coverage']['analyzed']} / {summary['coverage']['expected']}",
        f"- Split: `{summary['coverage']['split']}`",
        f"- Mean score: {summary['overall']['mean_score']:.4f}",
        f"- Exact accuracy: {summary['overall']['exact_accuracy']:.4f}",
        "",
        "## Key Failures",
        "",
    ]
    for flag, count in summary["failure_flags"].items():
        lines.append(f"- `{flag}`: {count}")

    lines.extend(["", "## Existence By Polarity", "", "| Polarity | Count | Accuracy |", "|---|---:|---:|"])
    for label, metrics in summary["existence_by_polarity"].items():
        lines.append(
            f"| {label} | {metrics['count']} | {metrics['exact_accuracy']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Relation By Source",
            "",
            "| Source | Count | Accuracy | Balanced acc. | Majority baseline | Parse valid | Refusal/absence | Hit max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source, metrics in summary["relation_sources"].items():
        lines.append(
            f"| {source} | {metrics['count']} | {metrics['exact_accuracy']:.4f} | "
            f"{metrics['balanced_accuracy']:.4f} | "
            f"{metrics['majority_class_baseline_accuracy']:.4f} | "
            f"{metrics['parse_valid_rate']:.4f} | "
            f"{metrics['refusal_or_absence_claims']} | "
            f"{metrics['hit_max_new_tokens']} |"
        )

    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary["warnings"])
    lines.extend(
        [
            "",
            "## Highest-Priority Samples",
            "",
            "| Sample | Source | Task | GT | Parsed | Flags |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in summary["top_failures"]:
        lines.append(
            f"| `{item['id']}` | {item['source']} | {item['task_type']} | "
            f"{item['gt_answer']} | {item['parsed_prediction']} | "
            f"{', '.join(item['flags'])} |"
        )
    return "\n".join(lines).rstrip() + "\n"
