"""Bridge saved VLM object predictions into grounding prompts and metrics."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Iterable

from .evaluation import extract_coco_categories


def categories_to_grounding_prompt(categories: Iterable[str]) -> str:
    """Build Grounding DINO's period-separated prompt from canonical labels."""
    normalized = sorted(
        {
            str(category).strip()
            for category in categories
            if str(category).strip()
        }
    )
    return ". ".join(normalized) + "." if normalized else ""


def evaluate_prompt_categories(
    predicted_categories: Iterable[str],
    target_categories: Iterable[str],
) -> dict[str, Any]:
    """Score VLM-generated categories before detector inference."""
    predicted = {str(category) for category in predicted_categories}
    target = {str(category) for category in target_categories}
    true_positives = len(predicted & target)
    false_positives = len(predicted - target)
    false_negatives = len(target - predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(target) if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "target_count": len(target),
        "predicted_count": len(predicted),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_match": predicted == target,
        "empty_prompt": not predicted,
        "hallucinated_categories": sorted(predicted - target),
        "missed_categories": sorted(target - predicted),
    }


def build_vlm_prompt_samples(
    oracle_samples: Iterable[dict[str, Any]],
    vlm_predictions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace oracle prompts with categories parsed from saved VLM answers."""
    listing_by_id: dict[str, dict[str, Any]] = {}
    for record in vlm_predictions:
        if record.get("task_type") != "object_listing":
            continue
        sample_id = str(record.get("id", ""))
        if not sample_id:
            raise ValueError("A VLM object_listing prediction is missing its id.")
        if sample_id in listing_by_id:
            raise ValueError(f"Duplicate VLM object_listing prediction: {sample_id}")
        listing_by_id[sample_id] = record

    samples = []
    for oracle_sample in oracle_samples:
        sample_id = str(oracle_sample["id"])
        vlm_record = listing_by_id.get(sample_id)
        if vlm_record is None:
            raise ValueError(f"Missing VLM object_listing prediction for {sample_id}.")
        if int(vlm_record.get("image_id", -1)) != int(oracle_sample["image_id"]):
            raise ValueError(
                f"VLM image_id mismatch for {sample_id}: "
                f"{vlm_record.get('image_id')} vs {oracle_sample['image_id']}."
            )

        raw_prediction = str(vlm_record.get("prediction", ""))
        structured_output = vlm_record.get("structured_output")
        if (
            isinstance(structured_output, dict)
            and isinstance(structured_output.get("parsed_categories"), list)
        ):
            prompt_categories = sorted(
                {str(value) for value in structured_output["parsed_categories"]}
            )
            prompt_parser = str(
                structured_output.get("parser", "structured_coco_json_v1")
            )
        else:
            prompt_categories = extract_coco_categories(raw_prediction)
            prompt_parser = "extract_coco_categories_v1"
        saved_categories = vlm_record.get("evaluation", {}).get(
            "predicted_categories"
        )
        if isinstance(saved_categories, list):
            saved_categories = sorted({str(value) for value in saved_categories})
        else:
            saved_categories = None

        target_categories = sorted(
            {str(value) for value in oracle_sample["categories"]}
        )
        samples.append(
            {
                **oracle_sample,
                "prompt": categories_to_grounding_prompt(prompt_categories),
                "prompt_categories": prompt_categories,
                "prompt_evaluation": evaluate_prompt_categories(
                    prompt_categories, target_categories
                ),
                "vlm_prediction": {
                    "id": sample_id,
                    "answer": raw_prediction,
                    "model": vlm_record.get("model"),
                    "latency_seconds": float(
                        vlm_record.get("latency_seconds", 0.0)
                    ),
                    "saved_categories": saved_categories,
                    "parser": prompt_parser,
                    "parser_matches_saved": (
                        saved_categories == prompt_categories
                        if saved_categories is not None
                        else None
                    ),
                },
            }
        )
    if not samples:
        raise ValueError("No oracle samples were provided for VLM prompt construction.")
    return samples


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


def aggregate_prompt_quality(
    records: Iterable[dict[str, Any]],
    *,
    expected_images: int,
) -> dict[str, Any]:
    """Aggregate VLM category quality without hiding empty prompts."""
    evaluations = [
        record["prompt_evaluation"]
        for record in records
        if "prompt_evaluation" in record
    ]
    totals = {
        key: sum(int(item[key]) for item in evaluations)
        for key in ("tp", "fp", "fn")
    }
    target_total = sum(int(item["target_count"]) for item in evaluations)
    predicted_total = sum(int(item["predicted_count"]) for item in evaluations)
    precision = totals["tp"] / predicted_total if predicted_total else 0.0
    recall = totals["tp"] / target_total if target_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    hallucinated = Counter(
        category
        for item in evaluations
        for category in item["hallucinated_categories"]
    )
    missed = Counter(
        category for item in evaluations for category in item["missed_categories"]
    )
    completed = len(evaluations)
    return {
        "coverage": {
            "expected": expected_images,
            "completed": completed,
            "remaining": max(expected_images - completed, 0),
            "completion_rate": round(completed / expected_images, 6)
            if expected_images
            else 0.0,
        },
        "counts": {
            "target_categories": target_total,
            "predicted_categories": predicted_total,
            **totals,
            "empty_prompt_images": sum(
                bool(item["empty_prompt"]) for item in evaluations
            ),
        },
        "micro_precision": round(precision, 6),
        "micro_recall": round(recall, 6),
        "micro_f1": round(f1, 6),
        "macro_precision": _mean(float(item["precision"]) for item in evaluations),
        "macro_recall": _mean(float(item["recall"]) for item in evaluations),
        "macro_f1": _mean(float(item["f1"]) for item in evaluations),
        "exact_match_rate": _mean(
            float(item["exact_match"]) for item in evaluations
        ),
        "hallucinated_categories": dict(sorted(hallucinated.items())),
        "missed_categories": dict(sorted(missed.items())),
    }


def aggregate_pipeline_latency(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Aggregate additive VLM, grounding, and SAM2 stage latency."""
    latencies = [
        record["pipeline_latency_seconds"]
        for record in records
        if "pipeline_latency_seconds" in record
    ]
    totals = [float(item["total"]) for item in latencies]
    total = sum(totals)
    return {
        "total": round(total, 6),
        "mean": _mean(totals),
        "median": round(statistics.median(totals), 6) if totals else 0.0,
        "p95": _percentile(totals, 0.95),
        "vlm_mean": _mean(float(item["vlm"]) for item in latencies),
        "grounding_mean": _mean(float(item["grounding"]) for item in latencies),
        "sam2_mean": _mean(float(item["sam2"]) for item in latencies),
        "throughput_images_per_second": round(len(totals) / total, 6)
        if total
        else 0.0,
    }
