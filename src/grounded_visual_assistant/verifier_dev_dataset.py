"""Build and audit the held-out-from-POPE verifier development set."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


VERIFIER_DEV_PROTOCOL = "coco_verifier_dev_supercategory_pairs_v1"


def _stable_rank(seed: int, *parts: object) -> str:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    return value.lower().replace(" ", "-")


def _indefinite_article(value: str) -> str:
    return "an" if value[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def build_verifier_dev_records(
    coco_ground_truth: Mapping[str, Any],
    *,
    dev_image_ids: Iterable[int],
    excluded_image_ids: Iterable[int],
    image_directory: str = "data/eval_v0/images",
    seed: int = 2026,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build balanced positive/hard-negative pairs without POPE images."""
    excluded = {int(value) for value in excluded_image_ids}
    requested = [int(value) for value in dev_image_ids]
    selected_image_ids = [
        image_id for image_id in requested if image_id not in excluded
    ]
    if not selected_image_ids:
        raise ValueError("No verifier development images remain.")

    categories = {
        int(item["id"]): dict(item)
        for item in coco_ground_truth["categories"]
    }
    category_by_name = {
        str(item["name"]): item for item in categories.values()
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw_annotation in coco_ground_truth["annotations"]:
        image_id = int(raw_annotation["image_id"])
        if image_id in selected_image_ids:
            annotations_by_image[image_id].append(dict(raw_annotation))

    missing = [
        image_id
        for image_id in selected_image_ids
        if not annotations_by_image[image_id]
    ]
    if missing:
        raise ValueError(
            f"Selected verifier images have no annotations: {missing}."
        )

    all_category_names = sorted(category_by_name)
    negative_usage: Counter[str] = Counter()
    records = []
    same_supercategory_pairs = 0
    fallback_pairs = 0
    selected_categories: set[str] = set()
    for image_id in selected_image_ids:
        annotations = annotations_by_image[image_id]
        present_names = sorted(
            {
                str(categories[int(item["category_id"])]["name"])
                for item in annotations
            },
            key=lambda name: (
                str(category_by_name[name]["supercategory"]),
                name,
            ),
        )
        present_set = set(present_names)
        used_negative_names: set[str] = set()
        for positive_name in present_names:
            positive_category = category_by_name[positive_name]
            supercategory = str(positive_category["supercategory"])
            same_group = [
                name
                for name in all_category_names
                if name not in present_set
                and name not in used_negative_names
                and str(category_by_name[name]["supercategory"])
                == supercategory
            ]
            if same_group:
                candidates = same_group
                negative_selection = "same_supercategory"
                same_supercategory_pairs += 1
            else:
                candidates = [
                    name
                    for name in all_category_names
                    if name not in present_set
                    and name not in used_negative_names
                ]
                negative_selection = "balanced_fallback"
                fallback_pairs += 1
            if not candidates:
                raise ValueError(
                    f"No negative category remains for image {image_id}."
                )
            negative_name = min(
                candidates,
                key=lambda name: (
                    negative_usage[name],
                    _stable_rank(
                        seed, image_id, positive_name, name
                    ),
                    name,
                ),
            )
            negative_usage[negative_name] += 1
            used_negative_names.add(negative_name)
            selected_categories.update((positive_name, negative_name))

            pair_id = (
                f"verifier_dev_{image_id:012d}_{_slug(positive_name)}"
            )
            image_path = (
                f"{image_directory.rstrip('/')}/{image_id:012d}.jpg"
            )
            evidence_boxes = [
                {
                    "annotation_id": int(item["id"]),
                    "category": positive_name,
                    "bbox_xywh": [
                        round(float(value), 3) for value in item["bbox"]
                    ],
                    "area": round(float(item.get("area", 0.0)), 3),
                    "iscrowd": int(item.get("iscrowd", 0)),
                }
                for item in annotations
                if int(item["category_id"])
                == int(positive_category["id"])
            ]
            common = {
                "pair_id": pair_id,
                "split": "dev",
                "image": image_path,
                "image_id": image_id,
                "source": "COCO val2017 annotations",
                "source_protocol": VERIFIER_DEV_PROTOCOL,
                "positive_object": positive_name,
                "positive_supercategory": supercategory,
            }
            records.append(
                {
                    **common,
                    "id": f"{pair_id}__positive",
                    "pair_role": "positive",
                    "question": (
                        f"Is there {_indefinite_article(positive_name)} "
                        f"{positive_name} in this image?"
                    ),
                    "object": positive_name,
                    "gt_answer": "yes",
                    "supercategory": supercategory,
                    "negative_selection": None,
                    "evidence_boxes": evidence_boxes,
                }
            )
            records.append(
                {
                    **common,
                    "id": (
                        f"{pair_id}__negative__{_slug(negative_name)}"
                    ),
                    "pair_role": "hard_negative",
                    "question": (
                        f"Is there {_indefinite_article(negative_name)} "
                        f"{negative_name} in this image?"
                    ),
                    "object": negative_name,
                    "gt_answer": "no",
                    "supercategory": str(
                        category_by_name[negative_name]["supercategory"]
                    ),
                    "negative_selection": negative_selection,
                    "evidence_boxes": [],
                }
            )

    summary = {
        "images": len(selected_image_ids),
        "questions": len(records),
        "positive_questions": sum(
            item["gt_answer"] == "yes" for item in records
        ),
        "negative_questions": sum(
            item["gt_answer"] == "no" for item in records
        ),
        "pairs": len(records) // 2,
        "categories": len(selected_categories),
        "same_supercategory_negative_pairs": same_supercategory_pairs,
        "fallback_negative_pairs": fallback_pairs,
        "excluded_requested_images": sorted(
            set(requested).intersection(excluded)
        ),
        "selected_image_ids": selected_image_ids,
        "negative_category_usage": dict(sorted(negative_usage.items())),
    }
    validate_verifier_dev_records(
        records,
        coco_ground_truth=coco_ground_truth,
        allowed_image_ids=selected_image_ids,
        excluded_image_ids=excluded,
    )
    return records, summary


def validate_verifier_dev_records(
    records: Iterable[Mapping[str, Any]],
    *,
    coco_ground_truth: Mapping[str, Any],
    allowed_image_ids: Iterable[int],
    excluded_image_ids: Iterable[int],
) -> dict[str, Any]:
    """Validate balance, GT consistency, pairing, and image isolation."""
    records = [dict(item) for item in records]
    allowed = {int(value) for value in allowed_image_ids}
    excluded = {int(value) for value in excluded_image_ids}
    categories = {
        int(item["id"]): str(item["name"])
        for item in coco_ground_truth["categories"]
    }
    present_by_image: dict[int, set[str]] = defaultdict(set)
    for annotation in coco_ground_truth["annotations"]:
        present_by_image[int(annotation["image_id"])].add(
            categories[int(annotation["category_id"])]
        )

    ids = [str(item["id"]) for item in records]
    query_keys = [
        (int(item["image_id"]), str(item["object"])) for item in records
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("Verifier Dev records contain duplicate IDs.")
    if len(query_keys) != len(set(query_keys)):
        raise ValueError(
            "Verifier Dev records contain duplicate image/object queries."
        )
    if any(int(item["image_id"]) not in allowed for item in records):
        raise ValueError("Verifier Dev record is outside the allowed split.")
    if any(int(item["image_id"]) in excluded for item in records):
        raise ValueError("Verifier Dev overlaps an excluded POPE image.")

    labels = Counter(str(item["gt_answer"]) for item in records)
    if labels["yes"] != labels["no"] or sum(labels.values()) != len(records):
        raise ValueError(f"Verifier Dev labels are not balanced: {labels}.")
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        pairs[str(item["pair_id"])].append(item)
        image_id = int(item["image_id"])
        object_name = str(item["object"])
        is_present = object_name in present_by_image[image_id]
        expected_present = str(item["gt_answer"]) == "yes"
        if is_present != expected_present:
            raise ValueError(
                f"Verifier Dev GT mismatch for {item['id']}: "
                f"present={is_present}, label={item['gt_answer']}."
            )
        if expected_present and not item.get("evidence_boxes"):
            raise ValueError(
                f"Positive verifier record has no evidence: {item['id']}."
            )
        if not expected_present and item.get("evidence_boxes"):
            raise ValueError(
                f"Negative verifier record has evidence: {item['id']}."
            )
    for pair_id, items in pairs.items():
        roles = {str(item["pair_role"]) for item in items}
        labels_in_pair = {str(item["gt_answer"]) for item in items}
        if len(items) != 2 or roles != {
            "positive",
            "hard_negative",
        } or labels_in_pair != {"yes", "no"}:
            raise ValueError(f"Invalid verifier pair {pair_id}.")
        if len({int(item["image_id"]) for item in items}) != 1:
            raise ValueError(f"Verifier pair crosses images: {pair_id}.")
    return {
        "questions": len(records),
        "pairs": len(pairs),
        "images": len({int(item["image_id"]) for item in records}),
        "labels": dict(sorted(labels.items())),
        "image_overlap_with_excluded": 0,
    }


def records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(dict(item), ensure_ascii=False, sort_keys=True) + "\n"
        for item in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
