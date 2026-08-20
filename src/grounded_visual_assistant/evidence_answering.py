"""Grounding-verified answer policies for the three eval_v0 tasks."""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from .evaluation import COCO_CATEGORIES, normalize_text
from .grounding_evaluation import canonicalize_category
from .vlm_grounding import categories_to_grounding_prompt


@dataclass(frozen=True)
class EvidencePolicyConfig:
    """Thresholds used after Grounded-SAM-2 inference."""

    min_grounding_score: float = 0.3
    min_mask_score: float | None = None
    min_mask_area_ratio: float = 0.0
    relation_margin: float = 0.08

    def __post_init__(self) -> None:
        for name, value in (
            ("min_grounding_score", self.min_grounding_score),
            ("min_mask_area_ratio", self.min_mask_area_ratio),
            ("relation_margin", self.relation_margin),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        if self.min_mask_score is not None and not (
            0.0 <= self.min_mask_score <= 1.0
        ):
            raise ValueError(
                "min_mask_score must be between 0 and 1 or None, got "
                f"{self.min_mask_score}."
            )


def _canonical_question_category(value: str) -> str:
    category = canonicalize_category(value, COCO_CATEGORIES)
    if category not in COCO_CATEGORIES:
        raise ValueError(f"Question entity is not a COCO-80 category: {value!r}")
    return category


def parse_question_entities(question: str, task_type: str) -> list[str]:
    """Parse query entities without consulting ground-truth category metadata."""
    normalized = " ".join(str(question).strip().split())
    if task_type == "object_listing":
        return []
    if task_type == "object_existence":
        match = re.fullmatch(
            r"Is there (?:an?|the) (.+?) in this image\? Answer yes or no\.?",
            normalized,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(f"Unsupported object-existence question: {question!r}")
        return [_canonical_question_category(match.group(1))]
    if task_type == "spatial_relation":
        match = re.fullmatch(
            r"Where is the largest (.+?) relative to the largest (.+?)\?",
            normalized,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(f"Unsupported spatial-relation question: {question!r}")
        categories = [
            _canonical_question_category(match.group(1)),
            _canonical_question_category(match.group(2)),
        ]
        if categories[0] == categories[1]:
            raise ValueError("Spatial-relation entities must be different categories.")
        return categories
    raise ValueError(f"Unsupported task type: {task_type}")


def build_query_plan(
    sample: dict[str, Any],
    structured_categories: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build the task-conditioned Grounding DINO prompt."""
    task_type = str(sample["task_type"])
    if task_type == "object_listing":
        if structured_categories is None:
            raise ValueError("object_listing requires structured VLM categories.")
        categories = sorted(
            {
                _canonical_question_category(str(category))
                for category in structured_categories
            }
        )
        source = "structured_vlm_coco80"
    else:
        categories = parse_question_entities(str(sample["question"]), task_type)
        source = "question_parser"
    return {
        "source": source,
        "categories": categories,
        "prompt": categories_to_grounding_prompt(categories),
    }


def _box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def normalize_evidence(
    annotations: Iterable[dict[str, Any]],
    target_categories: Iterable[str],
    *,
    image_width: int,
    image_height: int,
    config: EvidencePolicyConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Canonicalize detector labels and apply the explicit evidence gate."""
    targets = sorted({_canonical_question_category(value) for value in target_categories})
    target_set = set(targets)
    image_area = max(int(image_width) * int(image_height), 1)
    accepted = []
    rejected = []
    for index, annotation in enumerate(annotations):
        box = annotation.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"Annotation {index} has an invalid bbox: {box}")
        numeric_box = [float(value) for value in box]
        raw_label = str(annotation.get("class_name", "unknown"))
        category = canonicalize_category(raw_label, targets)
        mapping = "canonical_label"
        if category not in target_set and len(targets) == 1:
            # A single-category query is unambiguous even when the processor
            # returns a generic phrase such as "object".
            category = targets[0]
            mapping = "single_query_fallback"
        score = float(annotation.get("score", 0.0))
        raw_mask_score = annotation.get("mask_score")
        mask_score = (
            float(raw_mask_score) if raw_mask_score is not None else None
        )
        mask_area = int(annotation.get("mask_area", 0))
        mask_area_ratio = mask_area / image_area
        reasons = []
        if category not in target_set:
            reasons.append("off_query_category")
        if score < config.min_grounding_score:
            reasons.append("low_grounding_score")
        if (
            config.min_mask_score is not None
            and (mask_score is None or mask_score < config.min_mask_score)
        ):
            reasons.append("low_mask_score")
        if mask_area_ratio < config.min_mask_area_ratio:
            reasons.append("small_mask")
        evidence = {
            "annotation_index": index,
            "category": category,
            "raw_label": raw_label,
            "label_mapping": mapping,
            "bbox": numeric_box,
            "score": round(score, 6),
            "mask_score": round(mask_score, 6) if mask_score is not None else None,
            "mask_area": mask_area,
            "mask_area_ratio": round(mask_area_ratio, 8),
            "estimated_area": round(
                float(mask_area) if mask_area > 0 else _box_area(numeric_box), 3
            ),
        }
        if reasons:
            rejected.append({**evidence, "rejection_reasons": reasons})
        else:
            accepted.append(evidence)
    return accepted, rejected


def _best_instance(
    evidence: Iterable[dict[str, Any]], category: str
) -> dict[str, Any] | None:
    candidates = [item for item in evidence if item["category"] == category]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item["estimated_area"]),
            float(item["score"]),
            -int(item["annotation_index"]),
        ),
    )


def _spatial_relation(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> tuple[str, float, float, float]:
    first_x1, first_y1, first_x2, first_y2 = first["bbox"]
    second_x1, second_y1, second_x2, second_y2 = second["bbox"]
    dx = (
        ((first_x1 + first_x2) / 2) - ((second_x1 + second_x2) / 2)
    ) / max(image_width, 1)
    dy = (
        ((first_y1 + first_y2) / 2) - ((second_y1 + second_y2) / 2)
    ) / max(image_height, 1)
    dominance = max(abs(dx), abs(dy))
    if abs(dx) >= abs(dy):
        relation = "to the right of" if dx > 0 else "to the left of"
    else:
        relation = "below" if dy > 0 else "above"
    return relation, dx, dy, dominance


def answer_with_evidence(
    sample: dict[str, Any],
    query_plan: dict[str, Any],
    annotations: Iterable[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    config: EvidencePolicyConfig,
) -> dict[str, Any]:
    """Produce forced and selective answers from accepted visual evidence."""
    categories = list(query_plan["categories"])
    accepted, rejected = normalize_evidence(
        annotations,
        categories,
        image_width=image_width,
        image_height=image_height,
        config=config,
    )
    task_type = str(sample["task_type"])
    selected_evidence: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    if task_type == "object_listing":
        evidenced_categories = sorted({item["category"] for item in accepted})
        for category in evidenced_categories:
            best = _best_instance(accepted, category)
            if best is not None:
                selected_evidence.append(best)
        forced_answer = ", ".join(evidenced_categories)
        supported = bool(evidenced_categories)
        status = "supported" if supported else "insufficient_evidence"
        claim_count = len(evidenced_categories)
        unsupported_claim_count = 0
    elif task_type == "object_existence":
        if len(categories) != 1:
            raise ValueError("object_existence requires exactly one query category.")
        best = _best_instance(accepted, categories[0])
        if best is not None:
            selected_evidence = [best]
            forced_answer = "yes"
            supported = True
            status = "supported"
            unsupported_claim_count = 0
        else:
            forced_answer = "no"
            supported = False
            status = "insufficient_evidence"
            # Detector silence is not positive visual proof of absence.
            unsupported_claim_count = 1
        claim_count = 1
    elif task_type == "spatial_relation":
        if len(categories) != 2:
            raise ValueError("spatial_relation requires exactly two query categories.")
        first = _best_instance(accepted, categories[0])
        second = _best_instance(accepted, categories[1])
        if first is None or second is None:
            forced_answer = "insufficient evidence"
            supported = False
            status = "insufficient_evidence"
            claim_count = 0
            unsupported_claim_count = 0
            diagnostics["missing_categories"] = [
                category
                for category, instance in zip(categories, (first, second))
                if instance is None
            ]
        else:
            selected_evidence = [first, second]
            relation, dx, dy, dominance = _spatial_relation(
                first,
                second,
                image_width=image_width,
                image_height=image_height,
            )
            forced_answer = relation
            supported = dominance >= config.relation_margin
            status = "supported" if supported else "ambiguous_geometry"
            claim_count = 1
            unsupported_claim_count = 0 if supported else 1
            diagnostics.update(
                {
                    "dx_normalized": round(dx, 6),
                    "dy_normalized": round(dy, 6),
                    "dominance": round(dominance, 6),
                    "relation_margin": config.relation_margin,
                    "instance_rule": "largest predicted mask area, bbox fallback",
                }
            )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    evidence_scores = [float(item["score"]) for item in selected_evidence]
    return {
        "forced_answer": forced_answer,
        "selective_answer": forced_answer if supported else None,
        "abstained": not supported,
        "status": status,
        "claim_supported": supported,
        "claim_count": claim_count,
        "unsupported_claim_count": unsupported_claim_count,
        "confidence": round(min(evidence_scores), 6) if evidence_scores else 0.0,
        "selected_evidence": selected_evidence,
        "accepted_evidence": accepted,
        "rejected_evidence": rejected,
        "diagnostics": diagnostics,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def _answer_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluations = [record["evaluation"] for record in records]
    summary: dict[str, Any] = {
        "count": len(records),
        "mean_score": _mean(float(item["score"]) for item in evaluations),
        "exact_accuracy": _mean(
            float(item["is_correct"]) for item in evaluations
        ),
    }
    if records and {record["task_type"] for record in records} == {
        "object_listing"
    }:
        summary.update(
            {
                "macro_precision": _mean(
                    float(item["precision"]) for item in evaluations
                ),
                "macro_recall": _mean(
                    float(item["recall"]) for item in evaluations
                ),
                "macro_f1": _mean(float(item["f1"]) for item in evaluations),
            }
        )
    return summary


def _evidence_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluations = [record["evidence_evaluation"] for record in records]
    totals = {
        key: sum(int(item[key]) for item in evaluations)
        for key in ("gt_count", "prediction_count", "tp", "fp", "fn")
    }
    precision = totals["tp"] / totals["prediction_count"] if totals["prediction_count"] else 0.0
    recall = totals["tp"] / totals["gt_count"] if totals["gt_count"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_ious = [
        float(match["iou"])
        for evaluation in evaluations
        for match in evaluation.get("matches", [])
    ]
    return {
        "questions": len(records),
        "target_boxes": totals["gt_count"],
        "predicted_boxes": totals["prediction_count"],
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "micro_precision": round(precision, 6),
        "micro_recall": round(recall, 6),
        "micro_f1": round(f1, 6),
        "mean_matched_iou": _mean(matched_ious),
    }


def aggregate_evidence_answering(
    predictions: Iterable[dict[str, Any]],
    *,
    expected_samples: int,
    error_attempts: int = 0,
    status: str = "completed",
) -> dict[str, Any]:
    """Aggregate closed-set, selective, support, evidence, and latency metrics."""
    records = list(predictions)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["task_type"])].append(record)
    selective = [record for record in records if not record["answer_policy"]["abstained"]]
    selective_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selective:
        selective_groups[str(record["task_type"])].append(record)

    claim_count = sum(int(record["answer_policy"]["claim_count"]) for record in records)
    unsupported_claims = sum(
        int(record["answer_policy"]["unsupported_claim_count"])
        for record in records
    )
    selective_claim_count = sum(
        int(record["answer_policy"]["claim_count"]) for record in selective
    )
    selective_unsupported = sum(
        int(record["answer_policy"]["unsupported_claim_count"])
        for record in selective
    )
    latency = [float(record["pipeline_latency_seconds"]["total"]) for record in records]
    latency_total = sum(latency)

    metrics: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "coverage": {
            "expected": expected_samples,
            "completed": len(records),
            "remaining": max(expected_samples - len(records), 0),
            "completion_rate": round(len(records) / expected_samples, 6)
            if expected_samples
            else 0.0,
            "error_attempts": error_attempts,
        },
        "closed_set_answers": {
            "overall": _answer_summary(records) if records else {
                "count": 0,
                "mean_score": 0.0,
                "exact_accuracy": 0.0,
            },
            "tasks": {
                task: _answer_summary(task_records)
                for task, task_records in sorted(groups.items())
            },
        },
        "selective_answers": {
            "answered": len(selective),
            "abstained": len(records) - len(selective),
            "coverage": round(len(selective) / len(records), 6) if records else 0.0,
            "abstention_rate": round(
                (len(records) - len(selective)) / len(records), 6
            )
            if records
            else 0.0,
            "mean_score": _mean(
                float(record["evaluation"]["score"]) for record in selective
            ),
            "exact_accuracy": _mean(
                float(record["evaluation"]["is_correct"]) for record in selective
            ),
            "tasks": {
                task: {
                    **_answer_summary(selective_groups.get(task, [])),
                    "coverage": round(
                        len(selective_groups.get(task, [])) / len(task_records), 6
                    ),
                }
                for task, task_records in sorted(groups.items())
            },
        },
        "evidence_support": {
            "forced_claims": claim_count,
            "forced_unsupported_claims": unsupported_claims,
            "forced_unsupported_claim_rate": round(
                unsupported_claims / claim_count, 6
            )
            if claim_count
            else 0.0,
            "selective_claims": selective_claim_count,
            "selective_unsupported_claims": selective_unsupported,
            "selective_unsupported_claim_rate": round(
                selective_unsupported / selective_claim_count, 6
            )
            if selective_claim_count
            else 0.0,
        },
        "question_conditioned_evidence_iou50": {
            "overall": _evidence_summary(records),
            "tasks": {
                task: _evidence_summary(task_records)
                for task, task_records in sorted(groups.items())
            },
        },
        "latency_seconds": {
            "total": round(latency_total, 6),
            "mean": _mean(latency),
            "throughput_samples_per_second": round(len(records) / latency_total, 6)
            if latency_total
            else 0.0,
            "planning_mean": _mean(
                float(record["pipeline_latency_seconds"].get("planning", 0.0))
                for record in records
            ),
            "grounding_mean": _mean(
                float(record["pipeline_latency_seconds"].get("grounding", 0.0))
                for record in records
            ),
            "sam2_mean": _mean(
                float(record["pipeline_latency_seconds"].get("sam2", 0.0))
                for record in records
            ),
        },
    }
    peak_memory = [
        float(record["cuda_peak_memory_allocated_gb"])
        for record in records
        if "cuda_peak_memory_allocated_gb" in record
    ]
    if peak_memory:
        metrics["cuda_memory_gb"] = {
            "peak_allocated_mean": _mean(peak_memory),
            "peak_allocated_max": round(max(peak_memory), 6),
        }
    return metrics
