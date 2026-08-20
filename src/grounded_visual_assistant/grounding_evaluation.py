"""Class-aware box metrics for Grounded-SAM-2 experiments."""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from .evaluation import extract_coco_categories, normalize_text


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    """Convert a COCO xywh box to xyxy coordinates."""
    if len(box) != 4:
        raise ValueError(f"Expected four box values, got {len(box)}.")
    x, y, width, height = (float(value) for value in box)
    if width < 0 or height < 0:
        raise ValueError(f"Box width and height must be non-negative: {box}")
    return [x, y, x + width, y + height]


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Return intersection over union for two xyxy boxes."""
    if len(first) != 4 or len(second) != 4:
        raise ValueError("IoU requires two four-value xyxy boxes.")
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection_width = max(min(ax2, bx2) - max(ax1, bx1), 0.0)
    intersection_height = max(min(ay2, by2) - max(ay1, by1), 0.0)
    intersection = intersection_width * intersection_height
    first_area = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    second_area = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def canonicalize_category(label: str, target_categories: Iterable[str] = ()) -> str:
    """Map a detector text label to a canonical COCO category when possible."""
    normalized = normalize_text(str(label))
    target_by_normalized = {
        normalize_text(category): str(category) for category in target_categories
    }
    if normalized in target_by_normalized:
        return target_by_normalized[normalized]

    extracted = extract_coco_categories(normalized)
    target_set = set(target_by_normalized.values())
    target_matches = [category for category in extracted if category in target_set]
    if len(target_matches) == 1:
        return target_matches[0]
    if len(extracted) == 1:
        return extracted[0]
    return normalized or "unknown"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def evaluate_grounding_image(
    evidence_boxes: list[dict[str, Any]],
    predicted_annotations: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedily match scored predictions to same-class COCO boxes."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1].")

    ground_truth = []
    for index, item in enumerate(evidence_boxes):
        ground_truth.append(
            {
                "index": index,
                "category": str(item["category"]),
                "bbox": xywh_to_xyxy(item["bbox_xywh"]),
            }
        )
    target_categories = sorted({item["category"] for item in ground_truth})

    predictions = []
    for index, item in enumerate(predicted_annotations):
        box = item.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"Prediction {index} has an invalid bbox: {box}")
        predictions.append(
            {
                "index": index,
                "category": canonicalize_category(
                    str(item.get("class_name", "unknown")), target_categories
                ),
                "raw_label": str(item.get("class_name", "unknown")),
                "bbox": [float(value) for value in box],
                "score": float(item.get("score", 0.0)),
            }
        )

    gt_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ground_truth:
        gt_by_category[item["category"]].append(item)
    for item in predictions:
        predictions_by_category[item["category"]].append(item)

    prediction_results: dict[int, dict[str, Any]] = {}
    matches = []
    matched_gt_indices: set[int] = set()
    categories = sorted(set(gt_by_category) | set(predictions_by_category))
    category_counts: dict[str, dict[str, int]] = {}

    for category in categories:
        category_gt = gt_by_category.get(category, [])
        unmatched = {item["index"] for item in category_gt}
        category_predictions = sorted(
            predictions_by_category.get(category, []),
            key=lambda item: (-item["score"], item["index"]),
        )
        true_positives = 0
        for prediction in category_predictions:
            candidates = [item for item in category_gt if item["index"] in unmatched]
            best_gt = None
            best_iou = 0.0
            for target in candidates:
                overlap = box_iou(prediction["bbox"], target["bbox"])
                if overlap > best_iou:
                    best_gt = target
                    best_iou = overlap

            matched = best_gt is not None and best_iou >= iou_threshold
            matched_index = best_gt["index"] if matched and best_gt is not None else None
            if matched_index is not None:
                unmatched.remove(matched_index)
                matched_gt_indices.add(matched_index)
                true_positives += 1
                matches.append(
                    {
                        "category": category,
                        "prediction_index": prediction["index"],
                        "gt_index": matched_index,
                        "score": round(prediction["score"], 6),
                        "iou": round(best_iou, 6),
                    }
                )

            prediction_results[prediction["index"]] = {
                **prediction,
                "matched": matched,
                "matched_gt_index": matched_index,
                "matched_iou": round(best_iou, 6) if matched else 0.0,
            }

        prediction_count = len(category_predictions)
        gt_count = len(category_gt)
        category_counts[category] = {
            "gt": gt_count,
            "predicted": prediction_count,
            "tp": true_positives,
            "fp": prediction_count - true_positives,
            "fn": gt_count - true_positives,
        }

    gt_count = len(ground_truth)
    prediction_count = len(predictions)
    true_positives = len(matches)
    false_positives = prediction_count - true_positives
    false_negatives = gt_count - true_positives
    precision = _safe_ratio(true_positives, prediction_count)
    recall = _safe_ratio(true_positives, gt_count)

    return {
        "iou_threshold": iou_threshold,
        "gt_count": gt_count,
        "prediction_count": prediction_count,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(_f1(precision, recall), 6),
        "mean_matched_iou": _mean(match["iou"] for match in matches),
        "matches": matches,
        "prediction_results": [
            prediction_results[index] for index in sorted(prediction_results)
        ],
        "unmatched_gt_indices": sorted(
            set(range(gt_count)) - matched_gt_indices
        ),
        "categories": category_counts,
    }


def _average_precision(
    detection_results: list[dict[str, Any]], total_ground_truth: int
) -> float:
    if total_ground_truth <= 0:
        return 0.0
    ordered = sorted(
        detection_results,
        key=lambda item: (-float(item["score"]), str(item.get("image_id", ""))),
    )
    cumulative_tp = 0
    cumulative_fp = 0
    recalls = []
    precisions = []
    for item in ordered:
        if item["matched"]:
            cumulative_tp += 1
        else:
            cumulative_fp += 1
        recalls.append(cumulative_tp / total_ground_truth)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))

    recalls = [0.0, *recalls, 1.0]
    precisions = [0.0, *precisions, 0.0]
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    average_precision = 0.0
    for index in range(1, len(recalls)):
        if recalls[index] != recalls[index - 1]:
            average_precision += (
                recalls[index] - recalls[index - 1]
            ) * precisions[index]
    return average_precision


def aggregate_grounding_metrics(
    predictions: list[dict[str, Any]],
    *,
    expected_images: int,
    error_attempts: int = 0,
    status: str = "completed",
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggregate image-level grounding results and AP at the selected IoU."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1].")
    observed_thresholds = {
        float(item["evaluation"]["iou_threshold"]) for item in predictions
    }
    if observed_thresholds and observed_thresholds != {float(iou_threshold)}:
        raise ValueError(
            "Prediction IoU thresholds do not match the aggregation threshold: "
            f"{sorted(observed_thresholds)} vs {iou_threshold}."
        )
    threshold_suffix = int(round(iou_threshold * 100))
    ap_name = f"ap{threshold_suffix}"
    map_name = f"map{threshold_suffix}"
    box_metric_name = f"box_iou_{threshold_suffix:02d}"
    totals = {"gt": 0, "predicted": 0, "tp": 0, "fp": 0, "fn": 0}
    category_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"gt": 0, "predicted": 0, "tp": 0, "fp": 0, "fn": 0}
    )
    detections_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matched_ious = []

    for prediction in predictions:
        evaluation = prediction["evaluation"]
        totals["gt"] += int(evaluation["gt_count"])
        totals["predicted"] += int(evaluation["prediction_count"])
        for key in ("tp", "fp", "fn"):
            totals[key] += int(evaluation[key])
        matched_ious.extend(float(item["iou"]) for item in evaluation["matches"])

        for category, counts in evaluation["categories"].items():
            for key in category_totals[category]:
                category_totals[category][key] += int(counts[key])
        for item in evaluation["prediction_results"]:
            detections_by_category[item["category"]].append(
                {**item, "image_id": prediction["image_id"]}
            )

    precision = _safe_ratio(totals["tp"], totals["predicted"])
    recall = _safe_ratio(totals["tp"], totals["gt"])
    per_category = {}
    ap_values = []
    for category in sorted(category_totals):
        counts = category_totals[category]
        category_precision = _safe_ratio(counts["tp"], counts["predicted"])
        category_recall = _safe_ratio(counts["tp"], counts["gt"])
        average_precision = None
        if counts["gt"]:
            average_precision = round(
                _average_precision(detections_by_category[category], counts["gt"]),
                6,
            )
            ap_values.append(average_precision)
        per_category[category] = {
            **counts,
            "precision": round(category_precision, 6),
            "recall": round(category_recall, 6),
            "f1": round(_f1(category_precision, category_recall), 6),
            ap_name: average_precision,
        }

    total_latencies = [
        float(item["latency_seconds"]["total"]) for item in predictions
    ]
    grounding_latencies = [
        float(item["latency_seconds"]["grounding"]) for item in predictions
    ]
    sam2_latencies = [
        float(item["latency_seconds"]["sam2"]) for item in predictions
    ]
    total_latency = sum(total_latencies)
    completed = len(predictions)
    image_precisions = [
        float(item["evaluation"]["precision"]) for item in predictions
    ]
    image_recalls = [float(item["evaluation"]["recall"]) for item in predictions]
    image_f1s = [float(item["evaluation"]["f1"]) for item in predictions]
    candidate_counts = []
    kept_counts = []
    suppressed_counts = []
    for item in predictions:
        fallback_count = int(item["evaluation"]["prediction_count"])
        postprocessing = item.get("postprocessing", {})
        candidate_count = int(postprocessing.get("candidate_count", fallback_count))
        kept_count = int(postprocessing.get("kept_count", fallback_count))
        candidate_counts.append(candidate_count)
        kept_counts.append(kept_count)
        suppressed_counts.append(
            int(postprocessing.get("suppressed_count", candidate_count - kept_count))
        )

    metrics: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "coverage": {
            "expected": expected_images,
            "completed": completed,
            "remaining": max(expected_images - completed, 0),
            "completion_rate": round(completed / expected_images, 6)
            if expected_images
            else 0.0,
            "error_attempts": error_attempts,
        },
        box_metric_name: {
            **totals,
            "micro_precision": round(precision, 6),
            "micro_recall": round(recall, 6),
            "micro_f1": round(_f1(precision, recall), 6),
            "macro_image_precision": _mean(image_precisions),
            "macro_image_recall": _mean(image_recalls),
            "macro_image_f1": _mean(image_f1s),
            "mean_matched_iou": _mean(matched_ious),
            map_name: _mean(ap_values),
        },
        "per_category": per_category,
        "latency_seconds": {
            "total": round(total_latency, 6),
            "mean": _mean(total_latencies),
            "median": round(statistics.median(total_latencies), 6)
            if total_latencies
            else 0.0,
            "p95": _percentile(total_latencies, 0.95),
            "grounding_mean": _mean(grounding_latencies),
            "sam2_mean": _mean(sam2_latencies),
            "throughput_images_per_second": round(completed / total_latency, 6)
            if total_latency
            else 0.0,
        },
        "postprocessing": {
            "candidates_total": sum(candidate_counts),
            "kept_total": sum(kept_counts),
            "suppressed_total": sum(suppressed_counts),
            "candidates_mean_per_image": _mean(candidate_counts),
            "kept_mean_per_image": _mean(kept_counts),
            "suppression_rate": round(
                sum(suppressed_counts) / sum(candidate_counts), 6
            )
            if sum(candidate_counts)
            else 0.0,
        },
    }
    peak_memory = [
        float(item["cuda_peak_memory_allocated_gb"])
        for item in predictions
        if "cuda_peak_memory_allocated_gb" in item
    ]
    if peak_memory:
        metrics["cuda_memory_gb"] = {
            "peak_allocated_mean": _mean(peak_memory),
            "peak_allocated_max": round(max(peak_memory), 6),
        }
    return metrics
