"""Metrics for the live answer-to-grounding demo pipeline."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from .evaluation import COCO_CATEGORIES, aggregate_metrics, score_prediction
from .grounding_evaluation import (
    aggregate_grounding_metrics,
    canonicalize_category,
    evaluate_grounding_image,
)
from .vlm_grounding import aggregate_prompt_quality, evaluate_prompt_categories


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def visible_evidence_categories(sample: dict[str, Any]) -> list[str]:
    """Return categories with annotated visible evidence for one question."""
    return sorted(
        {
            str(item["category"])
            for item in sample.get("evidence_boxes", [])
            if str(item.get("category", "")).strip()
        }
    )


def canonicalize_targets(targets: Iterable[str]) -> list[str]:
    """Map generated targets to COCO names when the shared parser can do so."""
    return sorted(
        {
            canonicalize_category(str(target), COCO_CATEGORIES)
            for target in targets
            if str(target).strip()
        }
    )


def evaluate_live_prediction(
    sample: dict[str, Any],
    *,
    answer: str,
    targets: Iterable[str],
    annotations: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score answer, generated targets, and grounded boxes for one question."""
    expected_targets = visible_evidence_categories(sample)
    predicted_targets = canonicalize_targets(targets)
    answer_evaluation = score_prediction(sample, answer)
    target_evaluation = evaluate_prompt_categories(
        predicted_targets,
        expected_targets,
    )
    evidence_evaluation = evaluate_grounding_image(
        list(sample.get("evidence_boxes", [])),
        annotations,
        iou_threshold=iou_threshold,
    )
    evidence_required = evidence_evaluation["gt_count"] > 0
    if evidence_required:
        evidence_supported = evidence_evaluation["tp"] > 0
        evidence_complete = evidence_evaluation["fn"] == 0
    else:
        evidence_supported = evidence_evaluation["prediction_count"] == 0
        evidence_complete = evidence_supported
    return {
        "evaluation": answer_evaluation,
        "target_evaluation": target_evaluation,
        "evidence_evaluation": evidence_evaluation,
        "evidence_required": evidence_required,
        "evidence_supported": evidence_supported,
        "evidence_complete": evidence_complete,
        "end_to_end_success": bool(
            answer_evaluation["is_correct"] and evidence_supported
        ),
        "end_to_end_complete_success": bool(
            answer_evaluation["is_correct"] and evidence_complete
        ),
    }


def _as_pycoco_rle(
    segmentation: Any,
    *,
    height: int,
    width: int,
    mask_util: Any,
) -> dict[str, Any]:
    if isinstance(segmentation, list):
        return mask_util.merge(
            mask_util.frPyObjects(segmentation, height, width)
        )
    if not isinstance(segmentation, dict):
        raise ValueError("Segmentation must be a polygon list or COCO RLE.")
    rle = dict(segmentation)
    if isinstance(rle.get("counts"), list):
        return mask_util.frPyObjects(rle, height, width)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def evaluate_mask_evidence(
    evidence_boxes: list[dict[str, Any]],
    predicted_annotations: list[dict[str, Any]],
    *,
    coco_annotations_by_id: dict[int, dict[str, Any]],
    image_height: int,
    image_width: int,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedily match question-conditioned SAM masks to COCO masks."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1].")
    if not evidence_boxes and not predicted_annotations:
        return {
            "iou_threshold": iou_threshold,
            "gt_count": 0,
            "prediction_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_matched_iou": 0.0,
            "matches": [],
        }

    try:
        from pycocotools import mask as mask_util
    except ImportError as exc:
        raise RuntimeError(
            "Mask IoU evaluation requires pycocotools from "
            "requirements-grounded-sam2.txt."
        ) from exc

    ground_truth = []
    for index, item in enumerate(evidence_boxes):
        annotation_id = int(item["annotation_id"])
        coco_annotation = coco_annotations_by_id.get(annotation_id)
        if coco_annotation is None:
            raise ValueError(
                f"Missing COCO segmentation for annotation {annotation_id}."
            )
        ground_truth.append(
            {
                "index": index,
                "annotation_id": annotation_id,
                "category": str(item["category"]),
                "rle": _as_pycoco_rle(
                    coco_annotation["segmentation"],
                    height=image_height,
                    width=image_width,
                    mask_util=mask_util,
                ),
                "iscrowd": int(coco_annotation.get("iscrowd", 0)),
            }
        )
    target_categories = sorted({item["category"] for item in ground_truth})

    predictions = []
    for index, item in enumerate(predicted_annotations):
        segmentation = item.get("segmentation")
        predictions.append(
            {
                "index": index,
                "category": canonicalize_category(
                    str(item.get("class_name", "unknown")),
                    target_categories,
                ),
                "score": float(item.get("score", 0.0)),
                "rle": (
                    _as_pycoco_rle(
                        segmentation,
                        height=image_height,
                        width=image_width,
                        mask_util=mask_util,
                    )
                    if segmentation is not None
                    else None
                ),
            }
        )

    gt_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ground_truth:
        gt_by_category[item["category"]].append(item)
    for item in predictions:
        predictions_by_category[item["category"]].append(item)

    matches = []
    for category in sorted(set(gt_by_category) | set(predictions_by_category)):
        category_gt = gt_by_category.get(category, [])
        unmatched = {item["index"] for item in category_gt}
        category_predictions = sorted(
            predictions_by_category.get(category, []),
            key=lambda item: (-item["score"], item["index"]),
        )
        for prediction in category_predictions:
            candidates = [item for item in category_gt if item["index"] in unmatched]
            best_target = None
            best_iou = 0.0
            if prediction["rle"] is not None and candidates:
                overlaps = mask_util.iou(
                    [prediction["rle"]],
                    [item["rle"] for item in candidates],
                    [item["iscrowd"] for item in candidates],
                )[0]
                best_position = max(
                    range(len(candidates)),
                    key=lambda position: float(overlaps[position]),
                )
                best_target = candidates[best_position]
                best_iou = float(overlaps[best_position])
            if best_target is not None and best_iou >= iou_threshold:
                unmatched.remove(best_target["index"])
                matches.append(
                    {
                        "category": category,
                        "prediction_index": prediction["index"],
                        "gt_index": best_target["index"],
                        "annotation_id": best_target["annotation_id"],
                        "score": round(prediction["score"], 6),
                        "iou": round(best_iou, 6),
                    }
                )

    gt_count = len(ground_truth)
    prediction_count = len(predictions)
    true_positives = len(matches)
    precision = true_positives / prediction_count if prediction_count else 0.0
    recall = true_positives / gt_count if gt_count else 0.0
    return {
        "iou_threshold": iou_threshold,
        "gt_count": gt_count,
        "prediction_count": prediction_count,
        "tp": true_positives,
        "fp": prediction_count - true_positives,
        "fn": gt_count - true_positives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(_f1(precision, recall), 6),
        "mean_matched_iou": _mean(item["iou"] for item in matches),
        "matches": matches,
    }


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _end_to_end_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    task_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_records[str(record["task_type"])].append(record)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        supported = sum(bool(item["evidence_supported"]) for item in items)
        complete = sum(bool(item["evidence_complete"]) for item in items)
        successes = sum(bool(item["end_to_end_success"]) for item in items)
        complete_successes = sum(
            bool(item["end_to_end_complete_success"]) for item in items
        )
        return {
            "count": len(items),
            "evidence_supported_rate": _rate(supported, len(items)),
            "evidence_complete_rate": _rate(complete, len(items)),
            "answer_and_any_evidence_success_rate": _rate(
                successes, len(items)
            ),
            "answer_and_complete_evidence_success_rate": _rate(
                complete_successes, len(items)
            ),
        }

    return {
        "overall": summarize(records),
        "tasks": {
            task: summarize(items)
            for task, items in sorted(task_records.items())
        },
    }


def _stage_latency_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    stage_names = ("vlm", "grounding", "sam2", "end_to_end")
    result = {}
    for stage in stage_names:
        values = [
            float(item.get("pipeline_latency_seconds", {}).get(stage, 0.0))
            for item in records
        ]
        result[f"{stage}_mean"] = _mean(values)
    total = sum(
        float(item.get("pipeline_latency_seconds", {}).get("end_to_end", 0.0))
        for item in records
    )
    result["throughput_samples_per_second"] = (
        round(len(records) / total, 6) if total else 0.0
    )
    return result


def _grounding_projection(record: dict[str, Any]) -> dict[str, Any]:
    grounding = record.get("grounding", {})
    return {
        "image_id": f"{record.get('image_id')}::{record['id']}",
        "evaluation": record["evidence_evaluation"],
        "latency_seconds": grounding.get(
            "latency_seconds",
            {"grounding": 0.0, "sam2": 0.0, "total": 0.0},
        ),
        "postprocessing": grounding.get("postprocessing", {}),
        **(
            {
                "cuda_peak_memory_allocated_gb": record[
                    "cuda_peak_memory_allocated_gb"
                ]
            }
            if record.get("cuda_peak_memory_allocated_gb") is not None
            else {}
        ),
    }


def _aggregate_mask_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluations = [record["mask_evaluation"] for record in records]
    totals = {
        key: sum(int(item[key]) for item in evaluations)
        for key in ("gt_count", "prediction_count", "tp", "fp", "fn")
    }
    precision = (
        totals["tp"] / totals["prediction_count"]
        if totals["prediction_count"]
        else 0.0
    )
    recall = totals["tp"] / totals["gt_count"] if totals["gt_count"] else 0.0
    return {
        "questions": len(records),
        **totals,
        "micro_precision": round(precision, 6),
        "micro_recall": round(recall, 6),
        "micro_f1": round(_f1(precision, recall), 6),
        "macro_question_precision": _mean(
            item["precision"] for item in evaluations
        ),
        "macro_question_recall": _mean(item["recall"] for item in evaluations),
        "macro_question_f1": _mean(item["f1"] for item in evaluations),
        "mean_matched_iou": _mean(
            match["iou"]
            for item in evaluations
            for match in item["matches"]
        ),
    }


def aggregate_live_pipeline(
    records: list[dict[str, Any]],
    *,
    expected_samples: int,
    expected_required_evidence: int | None = None,
    expected_negative_evidence: int | None = None,
    error_attempts: int = 0,
    status: str = "completed",
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggregate answer, target, evidence, runtime, and end-to-end metrics."""
    metrics = aggregate_metrics(
        records,
        expected_samples=expected_samples,
        error_attempts=error_attempts,
        status=status,
    )
    target_quality = aggregate_prompt_quality(
        [
            {"prompt_evaluation": record["target_evaluation"]}
            for record in records
        ],
        expected_images=expected_samples,
    )
    parse_sources = Counter(
        str(record.get("vlm_output", {}).get("parse_source", "unknown"))
        for record in records
    )
    schema_valid = sum(
        bool(record.get("vlm_output", {}).get("schema_valid"))
        for record in records
    )
    target_quality["schema_valid_count"] = schema_valid
    target_quality["schema_valid_rate"] = _rate(schema_valid, len(records))
    target_quality["parse_sources"] = dict(sorted(parse_sources.items()))
    target_quality["mean_targets_per_question"] = _mean(
        len(record.get("targets", [])) for record in records
    )
    metrics["structured_targets"] = target_quality

    required_records = [
        record for record in records if record["evidence_required"]
    ]
    required_expected = (
        expected_required_evidence
        if expected_required_evidence is not None
        else len(required_records)
    )
    if required_records:
        metrics["required_evidence_box_metrics"] = aggregate_grounding_metrics(
            [_grounding_projection(record) for record in required_records],
            expected_images=required_expected,
            error_attempts=error_attempts,
            status=status,
            iou_threshold=iou_threshold,
        )
        mask_records = [
            record for record in required_records if "mask_evaluation" in record
        ]
        if mask_records:
            metrics["required_evidence_mask_iou_50"] = (
                _aggregate_mask_metrics(mask_records)
            )
    else:
        metrics["required_evidence_box_metrics"] = {
            "status": status,
            "coverage": {
                "expected": required_expected,
                "completed": 0,
                "remaining": required_expected,
                "completion_rate": 0.0,
                "error_attempts": error_attempts,
            },
        }

    negative_records = [
        record for record in records if not record["evidence_required"]
    ]
    negative_expected = (
        expected_negative_evidence
        if expected_negative_evidence is not None
        else len(negative_records)
    )
    negative_empty = sum(
        int(record["evidence_evaluation"]["prediction_count"]) == 0
        for record in negative_records
    )
    metrics["negative_evidence_behavior"] = {
        "expected_questions": negative_expected,
        "completed_questions": len(negative_records),
        "remaining_questions": max(
            negative_expected - len(negative_records), 0
        ),
        "completion_rate": _rate(len(negative_records), negative_expected),
        "correct_empty": negative_empty,
        "false_positive_questions": len(negative_records) - negative_empty,
        "correct_empty_rate": _rate(negative_empty, len(negative_records)),
    }
    metrics["end_to_end"] = _end_to_end_metrics(records)
    metrics["stage_latency_seconds"] = _stage_latency_metrics(records)
    metrics["protocol"] = "live_answer_and_evidence_targets_v1"
    return metrics
