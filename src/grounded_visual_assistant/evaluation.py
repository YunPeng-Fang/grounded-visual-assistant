"""Task-aware scoring helpers for the VLM evaluation pipeline."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable


COCO_CATEGORIES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


CATEGORY_ALIASES = {
    "person": ("person", "people", "man", "woman"),
    "bicycle": ("bicycle", "bike"),
    "motorcycle": ("motorcycle", "motorbike"),
    "airplane": ("airplane", "aeroplane", "plane", "aircraft"),
    "couch": ("couch", "sofa"),
    "potted plant": ("potted plant", "houseplant", "plant"),
    "dining table": ("dining table", "dinner table", "table"),
    "tv": ("tv", "television"),
    "cell phone": ("cell phone", "mobile phone", "smartphone", "phone"),
    "hair drier": ("hair drier", "hair dryer"),
    "refrigerator": ("refrigerator", "fridge"),
    "handbag": ("handbag", "purse"),
    "suitcase": ("suitcase", "luggage"),
    "tennis racket": ("tennis racket", "tennis racquet"),
    "sports ball": ("sports ball", "ball"),
    # Avoid interpreting a color adjective such as "orange shirt" as fruit.
    "orange": ("orange fruit", "oranges", "orange"),
}


RELATION_ALIASES = {
    "to the left of": ("to the left of", "left of", "on the left of"),
    "to the right of": ("to the right of", "right of", "on the right of"),
    "above": ("above", "over", "on top of"),
    "below": ("below", "under", "beneath"),
}


def normalize_text(value: str) -> str:
    """Lowercase text and collapse punctuation and whitespace."""
    normalized = value.lower().replace("_", "-")
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def parse_yes_no(value: str) -> str | None:
    """Extract an unambiguous yes/no answer."""
    tokens = re.findall(r"\b(?:yes|no)\b", normalize_text(value))
    unique = set(tokens)
    if len(unique) == 1:
        return tokens[0]
    return None


def parse_relation(value: str) -> str | None:
    """Map a generated spatial-relation phrase to the benchmark labels."""
    normalized = normalize_text(value)
    matches = set()
    for canonical, aliases in RELATION_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases):
            matches.add(canonical)
    if len(matches) == 1:
        return next(iter(matches))
    return None


def _category_aliases(category: str) -> tuple[str, ...]:
    return CATEGORY_ALIASES.get(category, (category,))


def extract_coco_categories(value: str) -> list[str]:
    """Extract COCO categories from a concise generated object list."""
    return extract_categories_from_vocabulary(value, COCO_CATEGORIES)


def extract_categories_from_vocabulary(
    value: str, vocabulary: Iterable[str]
) -> list[str]:
    """Extract the longest non-overlapping labels from a declared vocabulary."""
    normalized = normalize_text(value)
    matches: list[tuple[int, int, str, str]] = []
    for category in dict.fromkeys(str(item) for item in vocabulary):
        for alias in _category_aliases(category):
            normalized_alias = normalize_text(alias)
            plural_suffix = "" if normalized_alias.endswith("s") else "s?"
            for match in re.finditer(
                rf"\b{re.escape(normalized_alias)}{plural_suffix}\b", normalized
            ):
                matches.append((match.start(), match.end(), category, alias))

    # Prefer longer phrases so "hot dog" does not also become "dog".
    matches.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    occupied: list[tuple[int, int]] = []
    categories = set()
    for start, end, category, _ in matches:
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        occupied.append((start, end))
        categories.add(category)
    return sorted(categories)


def _set_scores(predicted: set[str], target: set[str]) -> dict[str, float | bool]:
    true_positives = len(predicted & target)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(target) if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_match": predicted == target,
    }


def score_prediction(sample: dict[str, Any], prediction: str) -> dict[str, Any]:
    """Score one prediction according to its task type."""
    task_type = sample["task_type"]
    target = str(sample["gt_answer"])

    if task_type == "object_existence":
        parsed = parse_yes_no(prediction)
        expected = normalize_text(target)
        correct = parsed == expected
        return {
            "score": float(correct),
            "is_correct": correct,
            "parsed_prediction": parsed,
            "parsed_target": expected,
            "parse_valid": parsed is not None,
        }

    if task_type == "spatial_relation":
        parsed = parse_relation(prediction)
        expected = parse_relation(target) or normalize_text(target)
        correct = parsed == expected
        return {
            "score": float(correct),
            "is_correct": correct,
            "parsed_prediction": parsed,
            "parsed_target": expected,
            "parse_valid": parsed is not None,
        }

    if task_type == "object_listing":
        metadata = sample.get("metadata") or {}
        allowed_categories = metadata.get("allowed_categories")
        parser_vocabulary = (
            list(allowed_categories) if allowed_categories else list(COCO_CATEGORIES)
        )
        predicted = set(
            extract_categories_from_vocabulary(prediction, parser_vocabulary)
        )
        target_categories = set(sample.get("categories") or [])
        if not target_categories:
            target_categories = set(
                extract_categories_from_vocabulary(target, parser_vocabulary)
            )
        set_scores = _set_scores(predicted, target_categories)
        return {
            "score": set_scores["f1"],
            "is_correct": set_scores["exact_match"],
            "predicted_categories": sorted(predicted),
            "target_categories": sorted(target_categories),
            "parser_vocabulary": (
                "restricted_allowed_categories"
                if allowed_categories
                else "coco80"
            ),
            "parser_vocabulary_size": len(parser_vocabulary),
            **set_scores,
        }

    normalized_prediction = normalize_text(prediction)
    normalized_target = normalize_text(target)
    correct = normalized_prediction == normalized_target
    return {
        "score": float(correct),
        "is_correct": correct,
        "parsed_prediction": normalized_prediction,
        "parsed_target": normalized_target,
    }


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


def _task_metric_groups(
    predictions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        groups[prediction["task_type"]].append(prediction)

    task_metrics: dict[str, Any] = {}
    for task_type, records in sorted(groups.items()):
        evaluations = [record["evaluation"] for record in records]
        metrics: dict[str, Any] = {
            "count": len(records),
            "mean_score": _mean(item["score"] for item in evaluations),
            "exact_accuracy": _mean(
                float(item["is_correct"]) for item in evaluations
            ),
        }
        if task_type == "object_listing":
            metrics.update(
                {
                    "macro_precision": _mean(
                        item["precision"] for item in evaluations
                    ),
                    "macro_recall": _mean(item["recall"] for item in evaluations),
                    "macro_f1": _mean(item["f1"] for item in evaluations),
                }
            )
        if task_type in {"object_existence", "spatial_relation"}:
            metrics["parse_valid_rate"] = _mean(
                float(item["parse_valid"]) for item in evaluations
            )
        if task_type == "spatial_relation":
            labels = sorted(
                {
                    str(item["parsed_target"])
                    for item in evaluations
                    if item.get("parsed_target")
                }
            )
            per_label = {}
            confusion = {}
            for label in labels:
                label_items = [
                    item for item in evaluations if item.get("parsed_target") == label
                ]
                predictions = Counter(
                    str(item.get("parsed_prediction") or "invalid")
                    for item in label_items
                )
                per_label[label] = {
                    "support": len(label_items),
                    "accuracy": _mean(
                        float(item["is_correct"]) for item in label_items
                    ),
                    "parse_valid_rate": _mean(
                        float(item["parse_valid"]) for item in label_items
                    ),
                }
                confusion[label] = dict(sorted(predictions.items()))
            supports = [item["support"] for item in per_label.values()]
            metrics.update(
                {
                    "balanced_accuracy": _mean(
                        item["accuracy"] for item in per_label.values()
                    ),
                    "majority_class_baseline_accuracy": (
                        round(max(supports) / len(evaluations), 6)
                        if supports
                        else 0.0
                    ),
                    "per_label": per_label,
                    "confusion": confusion,
                }
            )
        task_metrics[task_type] = metrics
    return task_metrics


def aggregate_metrics(
    predictions: list[dict[str, Any]],
    *,
    expected_samples: int,
    error_attempts: int = 0,
    status: str = "completed",
) -> dict[str, Any]:
    """Aggregate task and latency metrics from prediction records."""
    task_metrics = _task_metric_groups(predictions)

    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        source_groups[str(prediction.get("source", "unspecified"))].append(
            prediction
        )
    source_metrics = {}
    for source, records in sorted(source_groups.items()):
        source_metrics[source] = {
            "count": len(records),
            "mean_score": _mean(
                float(item["evaluation"]["score"]) for item in records
            ),
            "exact_accuracy": _mean(
                float(item["evaluation"]["is_correct"]) for item in records
            ),
            "tasks": _task_metric_groups(records),
        }

    latencies = [float(item["latency_seconds"]) for item in predictions]
    total_latency = sum(latencies)
    completed = len(predictions)
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "coverage": {
            "expected": expected_samples,
            "completed": completed,
            "remaining": max(expected_samples - completed, 0),
            "completion_rate": round(completed / expected_samples, 6)
            if expected_samples
            else 0.0,
            "error_attempts": error_attempts,
        },
        "overall": {
            "mean_score": _mean(
                float(item["evaluation"]["score"]) for item in predictions
            ),
            "exact_accuracy": _mean(
                float(item["evaluation"]["is_correct"]) for item in predictions
            ),
        },
        "tasks": task_metrics,
        "sources": source_metrics,
        "split_counts": dict(
            sorted(Counter(str(item.get("split", "unspecified")) for item in predictions).items())
        ),
        "latency_seconds": {
            "total": round(total_latency, 6),
            "mean": _mean(latencies),
            "median": round(statistics.median(latencies), 6) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
            "throughput_samples_per_second": round(completed / total_latency, 6)
            if total_latency
            else 0.0,
        },
    }
    peak_memory = [
        float(item["cuda_peak_memory_allocated_gb"])
        for item in predictions
        if "cuda_peak_memory_allocated_gb" in item
    ]
    reserved_memory = [
        float(item["cuda_memory_reserved_gb"])
        for item in predictions
        if "cuda_memory_reserved_gb" in item
    ]
    if peak_memory or reserved_memory:
        metrics["cuda_memory_gb"] = {
            "peak_allocated_mean": _mean(peak_memory),
            "peak_allocated_max": round(max(peak_memory), 6)
            if peak_memory
            else 0.0,
            "reserved_mean": _mean(reserved_memory),
            "reserved_max": round(max(reserved_memory), 6)
            if reserved_memory
            else 0.0,
        }
    return metrics
