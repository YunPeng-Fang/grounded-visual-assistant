"""Deterministic image-level splits for grounding experiments."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def coco_size_name(area: float) -> str:
    """Return the COCO small, medium, or large size bucket."""
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def build_image_feature_sets(
    ground_truth: Mapping[str, Any],
) -> dict[int, frozenset[str]]:
    """Represent every image by category-presence and object-size features."""
    features = {
        int(image["id"]): set() for image in ground_truth.get("images", [])
    }
    if not features:
        raise ValueError("Ground truth contains no images.")

    valid_category_ids = {
        int(category["id"]) for category in ground_truth.get("categories", [])
    }
    for annotation in ground_truth.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in features:
            raise ValueError(f"Annotation references unknown image {image_id}.")
        if category_id not in valid_category_ids:
            raise ValueError(f"Annotation references unknown category {category_id}.")
        features[image_id].add(f"category:{category_id}")
        features[image_id].add(
            f"size:{coco_size_name(float(annotation.get('area', 0.0)))}"
        )

    empty_images = sorted(image_id for image_id, value in features.items() if not value)
    if empty_images:
        raise ValueError(
            "Every split image must have at least one annotation; empty images: "
            f"{empty_images[:10]}"
        )
    return {
        image_id: frozenset(value) for image_id, value in features.items()
    }


def multilabel_stratified_split(
    feature_sets: Mapping[int, Iterable[str]],
    *,
    dev_size: int,
    seed: int,
    protect_singletons: bool = True,
) -> tuple[list[int], list[int]]:
    """Greedily approximate image-level multilabel stratification.

    Category presence and COCO size-bucket presence are balanced together. A
    fixed shuffled tie order makes the split deterministic without requiring an
    additional iterative-stratification dependency.
    """
    normalized = {
        int(image_id): frozenset(str(feature) for feature in features)
        for image_id, features in feature_sets.items()
    }
    image_ids = sorted(normalized)
    if not 0 < dev_size < len(image_ids):
        raise ValueError(
            f"dev_size must be between 1 and {len(image_ids) - 1}, got {dev_size}."
        )
    if any(not features for features in normalized.values()):
        raise ValueError("Every image must contain at least one split feature.")

    total_counts = Counter(
        feature for features in normalized.values() for feature in features
    )
    all_features = sorted(total_counts)
    dev_fraction = dev_size / len(image_ids)
    final_targets = {
        feature: (
            max(1.0, count * dev_fraction)
            if feature.startswith("category:") and count >= 2
            else count * dev_fraction
        )
        for feature, count in total_counts.items()
    }
    rng = random.Random(seed)
    tie_order = image_ids.copy()
    rng.shuffle(tie_order)
    tie_rank = {image_id: rank for rank, image_id in enumerate(tie_order)}

    uncovered_categories = {
        feature
        for feature, count in total_counts.items()
        if feature.startswith("category:") and count >= 2
    }
    singleton_categories = {
        feature
        for feature, count in total_counts.items()
        if feature.startswith("category:") and count == 1
    }
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_counts: Counter[str] = Counter()
    for slot in range(1, dev_size + 1):
        progress = slot / dev_size

        def candidate_score(image_id: int) -> tuple[int, float, float, int]:
            candidate_features = normalized[image_id]
            singleton_penalty = (
                len(candidate_features & singleton_categories)
                if protect_singletons
                else 0
            )
            coverage_gain = sum(
                1.0 / total_counts[feature]
                for feature in candidate_features & uncovered_categories
            )
            error = 0.0
            for feature in all_features:
                observed = selected_counts[feature] + (
                    1 if feature in candidate_features else 0
                )
                target = final_targets[feature] * progress
                error += ((observed - target) ** 2) / total_counts[feature]
            return singleton_penalty, -coverage_gain, error, tie_rank[image_id]

        candidate = min(
            (image_id for image_id in image_ids if image_id not in selected_set),
            key=candidate_score,
        )
        selected.append(candidate)
        selected_set.add(candidate)
        selected_counts.update(normalized[candidate])
        uncovered_categories.difference_update(normalized[candidate])

    dev_ids = sorted(selected)
    test_ids = sorted(set(image_ids) - selected_set)
    return dev_ids, test_ids


def split_statistics(
    ground_truth: Mapping[str, Any], image_ids: Iterable[int]
) -> dict[str, Any]:
    """Summarize instances, categories, and size distribution for a split."""
    selected = {int(image_id) for image_id in image_ids}
    category_names = {
        int(category["id"]): str(category["name"])
        for category in ground_truth.get("categories", [])
    }
    annotations = [
        annotation
        for annotation in ground_truth.get("annotations", [])
        if int(annotation["image_id"]) in selected
    ]
    category_instances = Counter(
        category_names[int(annotation["category_id"])] for annotation in annotations
    )
    size_instances = Counter(
        coco_size_name(float(annotation.get("area", 0.0)))
        for annotation in annotations
    )
    size_images: Counter[str] = Counter()
    for image_id in selected:
        present = {
            coco_size_name(float(annotation.get("area", 0.0)))
            for annotation in annotations
            if int(annotation["image_id"]) == image_id
        }
        size_images.update(present)
    return {
        "images": len(selected),
        "instances": len(annotations),
        "categories": len(category_instances),
        "size_instances": {
            name: size_instances.get(name, 0)
            for name in ("small", "medium", "large")
        },
        "images_with_size": {
            name: size_images.get(name, 0)
            for name in ("small", "medium", "large")
        },
        "category_instances": dict(sorted(category_instances.items())),
    }


def load_image_ids(path: str | Path) -> list[int]:
    """Load image IDs from either a JSON list or a split metadata object."""
    split_path = Path(path)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    values = payload.get("image_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"Split file must contain a non-empty image_ids list: {split_path}"
        )
    image_ids = [int(value) for value in values]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"Split file contains duplicate image IDs: {split_path}")
    return image_ids


def image_ids_sha256(image_ids: Iterable[int]) -> str:
    """Hash a split by its sorted semantic contents, independent of JSON layout."""
    canonical = ",".join(str(value) for value in sorted(int(item) for item in image_ids))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()
