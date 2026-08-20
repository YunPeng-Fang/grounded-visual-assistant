"""COCO-format ground truth and prediction helpers for oracle grounding."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from .grounding_evaluation import canonicalize_category


COCO_STAT_NAMES = (
    "ap",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_max_1",
    "ar_max_10",
    "ar_max_100",
    "ar_small",
    "ar_medium",
    "ar_large",
)


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    """Convert an xyxy box to the COCO xywh result format."""
    if len(box) != 4:
        raise ValueError(f"Expected four box values, got {len(box)}.")
    x1, y1, x2, y2 = (float(value) for value in box)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Prediction box has non-positive area: {box}")
    return [x1, y1, width, height]


def oracle_targets_from_questions(
    question_records: Iterable[dict[str, Any]],
) -> tuple[dict[int, set[str]], int]:
    """Return oracle categories by image and the old filtered box count."""
    targets: dict[int, set[str]] = {}
    filtered_box_count = 0
    for record in question_records:
        if record.get("task_type") != "object_listing":
            continue
        image_id = int(record["image_id"])
        if image_id in targets:
            raise ValueError(f"Duplicate object_listing record for image {image_id}.")
        categories = {str(value) for value in record.get("categories", [])}
        if not categories:
            raise ValueError(f"No oracle categories for image {image_id}.")
        targets[image_id] = categories
        filtered_box_count += len(record.get("evidence_boxes", []))
    if not targets:
        raise ValueError("No object_listing records were found.")
    return targets, filtered_box_count


def build_oracle_coco_ground_truth(
    source_coco: dict[str, Any],
    question_records: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore every COCO instance belonging to each image's prompted classes."""
    targets_by_image, filtered_box_count = oracle_targets_from_questions(
        question_records
    )
    images_by_id = {int(item["id"]): item for item in source_coco["images"]}
    categories_by_name = {
        str(item["name"]): item for item in source_coco["categories"]
    }
    missing_images = sorted(set(targets_by_image) - set(images_by_id))
    if missing_images:
        raise ValueError(f"Selected image IDs are absent from COCO: {missing_images[:5]}")

    missing_categories = sorted(
        {
            category
            for categories in targets_by_image.values()
            for category in categories
            if category not in categories_by_name
        }
    )
    if missing_categories:
        raise ValueError(f"Oracle categories are absent from COCO: {missing_categories}")

    category_id_by_name = {
        name: int(item["id"]) for name, item in categories_by_name.items()
    }
    target_category_ids = {
        image_id: {category_id_by_name[name] for name in names}
        for image_id, names in targets_by_image.items()
    }
    annotations = [
        dict(annotation)
        for annotation in source_coco["annotations"]
        if int(annotation["image_id"]) in target_category_ids
        and int(annotation["category_id"])
        in target_category_ids[int(annotation["image_id"])]
    ]
    selected_images = [
        dict(images_by_id[image_id]) for image_id in targets_by_image
    ]
    protocol = {
        "type": "oracle_conditioned",
        "question_task_type": "object_listing",
        "instance_policy": "all_sizes_for_prompted_categories",
        "target_categories_by_image": {
            str(image_id): sorted(categories)
            for image_id, categories in targets_by_image.items()
        },
    }
    ground_truth = {
        "info": dict(source_coco.get("info", {})),
        "licenses": list(source_coco.get("licenses", [])),
        "images": selected_images,
        "annotations": annotations,
        # Keep the original category table so known off-target detections can be
        # represented instead of being silently discarded during conversion.
        "categories": [dict(item) for item in source_coco["categories"]],
        "grounding_protocol": protocol,
    }
    prompted_category_names = {
        category for categories in targets_by_image.values() for category in categories
    }
    report = {
        "selected_images": len(selected_images),
        "prompted_categories": len(prompted_category_names),
        "image_category_prompts": sum(len(value) for value in targets_by_image.values()),
        "filtered_eval_v0_boxes": filtered_box_count,
        "restored_full_instances": len(annotations),
        "additional_instances": len(annotations) - filtered_box_count,
        "crowd_instances": sum(int(item.get("iscrowd", 0)) for item in annotations),
        "missing_segmentations": sum(not item.get("segmentation") for item in annotations),
    }
    return ground_truth, report


def _segmentation_score(annotation: dict[str, Any], mode: str) -> float:
    detector_score = float(annotation.get("score", 0.0))
    mask_score_value = annotation.get("mask_score")
    mask_score = (
        float(mask_score_value) if mask_score_value is not None else detector_score
    )
    if mode == "detector":
        return detector_score
    if mode == "mask":
        return mask_score
    if mode == "product":
        return detector_score * mask_score
    raise ValueError(f"Unsupported segmentation score mode: {mode}")


def filter_predictions_by_detector_score(
    prediction_records: Iterable[dict[str, Any]],
    min_score: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter saved annotations without repeating detector or SAM inference."""
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be between 0 and 1, got {min_score}.")
    filtered_records = []
    annotations_before = 0
    annotations_after = 0
    for record in prediction_records:
        annotations = list(record.get("annotations", []))
        kept = [
            annotation
            for annotation in annotations
            if float(annotation.get("score", 0.0)) >= min_score
        ]
        annotations_before += len(annotations)
        annotations_after += len(kept)
        filtered_records.append({**record, "annotations": kept})
    return filtered_records, {
        "min_detector_score": min_score,
        "prediction_records": len(filtered_records),
        "annotations_before": annotations_before,
        "annotations_after": annotations_after,
        "annotations_removed": annotations_before - annotations_after,
        "retention_rate": round(annotations_after / annotations_before, 6)
        if annotations_before
        else 0.0,
    }


def convert_predictions_to_coco(
    prediction_records: Iterable[dict[str, Any]],
    ground_truth: dict[str, Any],
    *,
    segmentation_score_mode: str = "detector",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Convert project JSONL predictions to COCO bbox and segmentation results."""
    category_id_by_name = {
        str(item["name"]): int(item["id"]) for item in ground_truth["categories"]
    }
    valid_image_ids = {int(item["id"]) for item in ground_truth["images"]}
    protocol_targets = {
        int(image_id): set(categories)
        for image_id, categories in ground_truth.get("grounding_protocol", {})
        .get("target_categories_by_image", {})
        .items()
    }

    bbox_results = []
    segmentation_results = []
    skipped_reasons: Counter[str] = Counter()
    skipped_examples = []
    seen_image_ids = set()
    record_count = 0

    for record in prediction_records:
        record_count += 1
        image_id = int(record["image_id"])
        if image_id not in valid_image_ids:
            raise ValueError(f"Prediction image {image_id} is absent from ground truth.")
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate prediction record for image {image_id}.")
        seen_image_ids.add(image_id)
        targets = set(record.get("target_categories", []))
        expected_targets = protocol_targets.get(image_id)
        if expected_targets is not None and targets != expected_targets:
            raise ValueError(
                f"Oracle targets differ for image {image_id}: "
                f"prediction={sorted(targets)}, gt={sorted(expected_targets)}"
            )

        for index, annotation in enumerate(record.get("annotations", [])):
            raw_label = str(annotation.get("class_name", "unknown"))
            category = canonicalize_category(raw_label, targets)
            category_id = category_id_by_name.get(category)
            if category_id is None:
                skipped_reasons["unmapped_label"] += 1
                if len(skipped_examples) < 20:
                    skipped_examples.append(
                        {
                            "image_id": image_id,
                            "annotation_index": index,
                            "raw_label": raw_label,
                            "canonical_label": category,
                        }
                    )
                continue
            try:
                bbox = xyxy_to_xywh(annotation["bbox"])
            except (KeyError, TypeError, ValueError):
                skipped_reasons["invalid_bbox"] += 1
                continue

            detector_score = float(annotation.get("score", 0.0))
            bbox_results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [round(value, 4) for value in bbox],
                    "score": detector_score,
                }
            )
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, dict) or not {
                "size",
                "counts",
            }.issubset(segmentation):
                skipped_reasons["missing_segmentation"] += 1
                continue
            segmentation_results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "score": _segmentation_score(
                        annotation, segmentation_score_mode
                    ),
                }
            )

    report = {
        "prediction_records": record_count,
        "unique_prediction_images": len(seen_image_ids),
        "bbox_detections": len(bbox_results),
        "segmentation_detections": len(segmentation_results),
        "skipped": dict(sorted(skipped_reasons.items())),
        "skipped_examples": skipped_examples,
        "segmentation_score_mode": segmentation_score_mode,
    }
    return bbox_results, segmentation_results, report


def coco_stats_to_dict(stats: Sequence[float]) -> dict[str, float | None]:
    """Name the 12 summary values emitted by COCOeval."""
    if len(stats) < len(COCO_STAT_NAMES):
        raise ValueError(
            f"Expected {len(COCO_STAT_NAMES)} COCO stats, got {len(stats)}."
        )
    return {
        name: round(float(value), 6) if float(value) >= 0 else None
        for name, value in zip(COCO_STAT_NAMES, stats)
    }
