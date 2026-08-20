"""Source-aware question generation for the cross-dataset hard benchmark."""

from __future__ import annotations

import csv
import copy
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .hard_dataset import OPEN_IMAGES_SOURCE, VISUAL_GENOME_SOURCE


def load_verified_image_labels(
    path: str | Path,
    *,
    selected_image_ids: Iterable[str] | None = None,
) -> dict[str, dict[str, set[str]]]:
    """Load human-verified Open Images positive and negative image labels."""
    selected = (
        {str(value) for value in selected_image_ids}
        if selected_image_ids is not None
        else None
    )
    labels: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"positive": set(), "negative": set()}
    )
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "LabelName", "Confidence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Verified image labels are missing columns: {sorted(missing)}"
            )
        for row in reader:
            image_id = str(row["ImageID"]).strip()
            if selected is not None and image_id not in selected:
                continue
            source = str(row.get("Source", "")).strip()
            if source not in {"verification", "crowdsource-verification"}:
                continue
            label_id = str(row["LabelName"]).strip()
            try:
                confidence = float(str(row["Confidence"]).strip())
            except ValueError:
                continue
            if not image_id or not label_id or confidence not in {0.0, 1.0}:
                continue
            key = "positive" if confidence == 1.0 else "negative"
            labels[image_id][key].add(label_id)

    for image_id, item in labels.items():
        conflicts = item["positive"] & item["negative"]
        if conflicts:
            raise ValueError(
                f"Conflicting verified labels for {image_id}: {sorted(conflicts)[:5]}"
            )
    return dict(labels)


def _stable_order(values: Iterable[str], salt: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )


def _objects_by_category(candidate: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidate.get("objects", []):
        grouped[str(item["category"])].append(dict(item))
    return dict(grouped)


def _rank_present_categories(candidate: Mapping[str, Any]) -> list[str]:
    grouped = _objects_by_category(candidate)

    def rank(category: str) -> tuple[Any, ...]:
        objects = grouped[category]
        difficult = any(
            item.get("is_occluded") or item.get("is_truncated")
            for item in objects
        )
        minimum_area = min(float(item["area_ratio"]) for item in objects)
        return (not difficult, minimum_area, category)

    return sorted(grouped, key=rank)


def _verified_negative_categories(
    candidate: Mapping[str, Any],
    verified_labels: Mapping[str, Mapping[str, set[str]]],
    class_names: Mapping[str, str],
) -> dict[str, str]:
    image_id = str(candidate["source_image_id"])
    present = {str(value) for value in candidate.get("categories", [])}
    by_category = {}
    for label_id in verified_labels.get(image_id, {}).get("negative", set()):
        category = class_names.get(label_id)
        if category and category not in present:
            by_category.setdefault(category, label_id)
    return by_category


def _evidence_box(
    item: Mapping[str, Any], width: int, height: int
) -> dict[str, Any]:
    x1, y1, x2, y2 = [
        float(value) for value in item["bbox_xyxy_normalized"]
    ]
    return {
        "annotation_id": item["annotation_id"],
        "category": item["category"],
        "bbox_xyxy_normalized": [x1, y1, x2, y2],
        "bbox_xywh": [
            round(x1 * width, 2),
            round(y1 * height, 2),
            round((x2 - x1) * width, 2),
            round((y2 - y1) * height, 2),
        ],
    }


def _base_question(
    candidate: Mapping[str, Any],
    image: Mapping[str, Any],
    suffix: str,
) -> dict[str, Any]:
    source_short = "oi" if candidate["source"] == OPEN_IMAGES_SOURCE else "vg"
    source_image_id = str(candidate["source_image_id"])
    return {
        "id": f"hard_v1__{source_short}__{source_image_id}__{suffix}",
        "image": image["path"],
        "image_id": candidate["sample_id"],
        "sample_id": candidate["sample_id"],
        "source_image_id": source_image_id,
        "source": candidate["source"],
        "split": candidate["split"],
    }


def _relation_question(
    candidate: Mapping[str, Any], image: Mapping[str, Any]
) -> dict[str, Any]:
    relation = candidate["relations"][0]
    objects = {
        str(item["annotation_id"]): item for item in candidate["objects"]
    }
    subject = objects[str(relation["subject_annotation_id"])]
    object_item = objects[str(relation["object_annotation_id"])]

    if candidate["source"] == OPEN_IMAGES_SOURCE:
        subject_phrase = f"the largest {relation['subject_category']}"
        object_phrase = f"the largest {relation['object_category']}"
        relation_provenance = "derived_from_annotated_box_centers"
        instance_rules = {
            "subject": "largest_annotated_non_group_instance",
            "object": "largest_annotated_non_group_instance",
        }
    else:
        subject_phrase = (
            f"the largest {relation['subject_category']}"
            if relation["subject_instance_rule"] == "largest_instance"
            else f"the {relation['subject_category']}"
        )
        object_phrase = (
            f"the largest {relation['object_category']}"
            if relation["object_instance_rule"] == "largest_instance"
            else f"the {relation['object_category']}"
        )
        relation_provenance = "visual_genome_explicit_relationship"
        instance_rules = {
            "subject": relation["subject_instance_rule"],
            "object": relation["object_instance_rule"],
        }

    question = _base_question(candidate, image, "relation")
    question.update(
        {
            "question": (
                f"Where is {subject_phrase} relative to {object_phrase}? "
                "Answer with above, below, to the left of, or to the right of."
            ),
            "task_type": "spatial_relation",
            "gt_answer": relation["predicate"],
            "categories": [
                relation["subject_category"],
                relation["object_category"],
            ],
            "evidence_boxes": [
                _evidence_box(subject, int(image["width"]), int(image["height"])),
                _evidence_box(
                    object_item, int(image["width"]), int(image["height"])
                ),
            ],
            "metadata": {
                "annotation_scope": candidate["annotation_scope"],
                "relation_provenance": relation_provenance,
                "native_predicate": relation.get("native_predicate"),
                "instance_rules": instance_rules,
                "answer_space": [
                    "above",
                    "below",
                    "to the left of",
                    "to the right of",
                ],
            },
        }
    )
    return question


def build_hard_questions(
    candidates: Iterable[Mapping[str, Any]],
    images: Iterable[Mapping[str, Any]],
    verified_labels: Mapping[str, Mapping[str, set[str]]],
    class_names: Mapping[str, str],
    *,
    seed: int = 2026,
    listing_positive_limit: int = 5,
    listing_negative_limit: int = 4,
) -> list[dict[str, Any]]:
    """Generate restricted Open Images tasks and explicit VG relation tasks."""
    candidates = sorted(candidates, key=lambda item: str(item["sample_id"]))
    image_by_id = {str(item["sample_id"]): item for item in images}
    candidate_ids = {str(item["sample_id"]) for item in candidates}
    if candidate_ids != set(image_by_id):
        raise ValueError("Frozen candidates and images have different sample IDs.")

    open_images = [
        item for item in candidates if item["source"] == OPEN_IMAGES_SOURCE
    ]
    negative_categories = {
        str(item["sample_id"]): _verified_negative_categories(
            item, verified_labels, class_names
        )
        for item in open_images
    }
    negative_eligible = [
        str(item["sample_id"])
        for item in open_images
        if negative_categories[str(item["sample_id"])]
    ]
    negative_target = len(open_images) // 2
    if len(negative_eligible) < negative_target:
        raise RuntimeError(
            "Not enough Open Images samples have human-verified negative labels: "
            f"required={negative_target}, found={len(negative_eligible)}."
        )
    negative_existence_ids = set(
        _stable_order(negative_eligible, f"{seed}:existence")[:negative_target]
    )

    questions = []
    for candidate in candidates:
        sample_id = str(candidate["sample_id"])
        image = image_by_id[sample_id]
        if candidate["source"] == VISUAL_GENOME_SOURCE:
            questions.append(_relation_question(candidate, image))
            continue

        grouped = _objects_by_category(candidate)
        ranked_present = _rank_present_categories(candidate)
        selected_present = ranked_present[:listing_positive_limit]
        negative_by_category = negative_categories[sample_id]
        selected_negative = _stable_order(
            negative_by_category, f"{seed}:{sample_id}:listing"
        )[:listing_negative_limit]
        allowed_categories = _stable_order(
            [*selected_present, *selected_negative],
            f"{seed}:{sample_id}:allowed",
        )
        listing = _base_question(candidate, image, "listing")
        listing.update(
            {
                "question": (
                    "From these categories only, list every category present in "
                    f"the image: {', '.join(allowed_categories)}. Return category "
                    "names only."
                ),
                "task_type": "object_listing",
                "gt_answer": ", ".join(sorted(selected_present)),
                "categories": sorted(selected_present),
                "evidence_boxes": [
                    _evidence_box(item, int(image["width"]), int(image["height"]))
                    for category in selected_present
                    for item in grouped[category]
                ],
                "metadata": {
                    "annotation_scope": candidate["annotation_scope"],
                    "listing_protocol": "restricted_verified_vocabulary",
                    "allowed_categories": allowed_categories,
                    "verified_absent_categories": selected_negative,
                    "has_negative_distractor": bool(selected_negative),
                    "negative_label_provenance": (
                        "open_images_human_verified_confidence_0"
                        if selected_negative
                        else None
                    ),
                },
            }
        )
        questions.append(listing)

        existence = _base_question(candidate, image, "existence")
        if sample_id in negative_existence_ids:
            category = _stable_order(
                negative_by_category, f"{seed}:{sample_id}:negative"
            )[0]
            label_id = negative_by_category[category]
            answer = "no"
            evidence_boxes = []
            is_positive = False
            verification = {
                "source_label_id": label_id,
                "confidence": 0,
                "provenance": "open_images_human_verified_image_label",
            }
        else:
            category = _stable_order(
                selected_present, f"{seed}:{sample_id}:positive"
            )[0]
            answer = "yes"
            evidence_boxes = [
                _evidence_box(item, int(image["width"]), int(image["height"]))
                for item in grouped[category]
            ]
            is_positive = True
            verification = {
                "source_label_ids": sorted(
                    {
                        str(item.get("source_label_id"))
                        for item in grouped[category]
                        if item.get("source_label_id")
                    }
                ),
                "provenance": "open_images_verified_bounding_boxes",
            }
        existence.update(
            {
                "question": (
                    f"Is there a {category} in this image? Answer yes or no."
                ),
                "task_type": "object_existence",
                "gt_answer": answer,
                "categories": [category],
                "evidence_boxes": evidence_boxes,
                "metadata": {
                    "is_positive": is_positive,
                    "annotation_scope": candidate["annotation_scope"],
                    "verification": verification,
                },
            }
        )
        questions.append(existence)
        questions.append(_relation_question(candidate, image))

    validate_hard_questions(questions, candidates)
    return sorted(questions, key=lambda item: item["id"])


def validate_hard_questions(
    questions: Iterable[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    questions = list(questions)
    candidates = list(candidates)
    question_ids = [str(item.get("id", "")) for item in questions]
    if len(question_ids) != len(set(question_ids)) or any(not value for value in question_ids):
        raise ValueError("Question IDs must be present and unique.")

    candidate_by_id = {str(item["sample_id"]): item for item in candidates}
    counts_by_sample = Counter(str(item["sample_id"]) for item in questions)
    for sample_id, candidate in candidate_by_id.items():
        expected = 3 if candidate["source"] == OPEN_IMAGES_SOURCE else 1
        if counts_by_sample[sample_id] != expected:
            raise ValueError(
                f"Expected {expected} questions for {sample_id}, "
                f"found {counts_by_sample[sample_id]}."
            )
    if set(counts_by_sample) != set(candidate_by_id):
        raise ValueError("Questions contain unknown or missing sample IDs.")

    valid_relations = {
        "above",
        "below",
        "to the left of",
        "to the right of",
    }
    for item in questions:
        sample_id = str(item["sample_id"])
        candidate = candidate_by_id[sample_id]
        if item["source"] != candidate["source"] or item["split"] != candidate["split"]:
            raise ValueError(f"Question provenance mismatch for {item['id']}.")
        for box in item.get("evidence_boxes", []):
            normalized = [float(value) for value in box["bbox_xyxy_normalized"]]
            pixel = [float(value) for value in box["bbox_xywh"]]
            if (
                len(normalized) != 4
                or not all(0.0 <= value <= 1.0 for value in normalized)
                or normalized[0] >= normalized[2]
                or normalized[1] >= normalized[3]
                or len(pixel) != 4
                or pixel[2] <= 0
                or pixel[3] <= 0
            ):
                raise ValueError(f"Invalid evidence box in {item['id']}.")

        if item["task_type"] == "object_listing":
            positives = set(item["categories"])
            allowed = set(item["metadata"]["allowed_categories"])
            negatives = set(item["metadata"]["verified_absent_categories"])
            if (
                positives != {value.strip() for value in item["gt_answer"].split(",")}
                or not positives <= allowed
                or not negatives <= allowed
                or positives & negatives
            ):
                raise ValueError(f"Invalid restricted listing labels in {item['id']}.")
            if any(box["category"] not in positives for box in item["evidence_boxes"]):
                raise ValueError(f"Listing evidence exceeds ground truth in {item['id']}.")
        elif item["task_type"] == "object_existence":
            is_positive = bool(item["metadata"]["is_positive"])
            if is_positive != (item["gt_answer"] == "yes"):
                raise ValueError(f"Existence polarity mismatch in {item['id']}.")
            if is_positive and not item["evidence_boxes"]:
                raise ValueError(f"Positive existence lacks evidence in {item['id']}.")
            if not is_positive and item["evidence_boxes"]:
                raise ValueError(f"Negative existence has positive boxes in {item['id']}.")
        elif item["task_type"] == "spatial_relation":
            if item["gt_answer"] not in valid_relations or len(item["evidence_boxes"]) != 2:
                raise ValueError(f"Invalid spatial relation in {item['id']}.")
        else:
            raise ValueError(f"Unknown task type in {item['id']}.")

    existence = [item for item in questions if item["task_type"] == "object_existence"]
    positives = sum(item["metadata"]["is_positive"] for item in existence)
    if positives * 2 != len(existence):
        raise ValueError("Open Images existence questions must be exactly balanced.")
    for item in existence:
        if item["gt_answer"] == "no":
            verification = item["metadata"].get("verification", {})
            if verification.get("confidence") != 0:
                raise ValueError(
                    "Negative existence question lacks a human-verified confidence-0 label."
                )
    for item in questions:
        if (
            item["source"] == VISUAL_GENOME_SOURCE
            and item["task_type"] != "spatial_relation"
        ):
            raise ValueError("Visual Genome may only contribute explicit relation tasks.")

    listings = [item for item in questions if item["task_type"] == "object_listing"]
    listings_with_negatives = sum(
        bool(item["metadata"].get("verified_absent_categories"))
        for item in listings
    )

    return {
        "questions": len(questions),
        "samples": len(candidate_by_id),
        "sources": dict(Counter(item["source"] for item in questions)),
        "tasks": dict(Counter(item["task_type"] for item in questions)),
        "splits": dict(Counter(item["split"] for item in questions)),
        "existence_positive": positives,
        "existence_negative": len(existence) - positives,
        "listing_with_verified_negative_distractors": listings_with_negatives,
        "listing_positive_only_vocabulary": len(listings) - listings_with_negatives,
    }


def apply_relation_prompt_v2(question: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the Dev-designed center-based forced-choice relation prompt."""
    updated = copy.deepcopy(dict(question))
    if updated.get("task_type") != "spatial_relation":
        return updated
    categories = list(updated.get("categories") or [])
    if len(categories) != 2:
        raise ValueError(f"Relation question has invalid categories: {updated.get('id')}")
    metadata = dict(updated.get("metadata") or {})
    rules = metadata.get("instance_rules") or {}

    def instance_phrase(category: str, rule: str | None) -> str:
        if rule and "largest" in rule:
            return f"the largest visible {category}"
        return f"the visible {category}"

    subject = instance_phrase(categories[0], rules.get("subject"))
    object_item = instance_phrase(categories[1], rules.get("object"))
    updated["question"] = (
        "Treat both named instances as present in this image, even if a category "
        "label is broad. Compare their visual center positions. Where is the "
        f"center of {subject} relative to the center of {object_item}? Reply with "
        "exactly one label: above; below; to the left of; to the right of."
    )
    metadata.update(
        {
            "prompt_version": "relation_center_forced_choice_v2",
            "forced_choice": True,
            "relation_geometry": "visual_instance_center",
        }
    )
    updated["metadata"] = metadata
    return updated


def apply_visual_genome_relation_prompt_v3(
    question: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the Dev-designed semantic forced-choice prompt to a VG relation."""
    updated = copy.deepcopy(dict(question))
    if updated.get("task_type") != "spatial_relation":
        raise ValueError(
            "Visual Genome prompt v3 only accepts spatial relation questions: "
            f"{updated.get('id')}"
        )
    if updated.get("source") != VISUAL_GENOME_SOURCE:
        raise ValueError(
            "Visual Genome prompt v3 received a different source: "
            f"{updated.get('id')}"
        )
    categories = list(updated.get("categories") or [])
    if len(categories) != 2:
        raise ValueError(
            f"Relation question has invalid categories: {updated.get('id')}"
        )
    metadata = dict(updated.get("metadata") or {})
    rules = metadata.get("instance_rules") or {}

    def instance_phrase(category: str, rule: str | None) -> str:
        if rule and "largest" in rule:
            return f"the largest visible {category}"
        return f"the visible {category}"

    subject = instance_phrase(categories[0], rules.get("subject"))
    object_item = instance_phrase(categories[1], rules.get("object"))
    updated["question"] = (
        "Treat both named object instances as present in this image, even if a "
        "category label is broad or unusual. Judge their depicted spatial "
        "relationship, not category definitions or real-world size. Where is "
        f"{subject} relative to {object_item}? Reply with exactly one label: "
        "above; below; to the left of; to the right of."
    )
    metadata.update(
        {
            "prompt_version": "visual_genome_semantic_forced_choice_v3",
            "forced_choice": True,
            "relation_geometry": "depicted_semantic_spatial_relationship",
        }
    )
    updated["metadata"] = metadata
    return updated


def apply_locked_source_aware_relation_prompt(
    question: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the locked source-aware relation policy without changing other tasks."""
    updated = copy.deepcopy(dict(question))
    if updated.get("task_type") != "spatial_relation":
        return updated
    if (
        policy.get("protocol") != "hard_relation_source_aware_prompt_policy_v1"
        or policy.get("status") != "locked"
        or policy.get("selected_on_split") != "dev"
        or not policy.get("immutable")
    ):
        raise ValueError("The source-aware relation prompt policy is not locked.")

    source = str(updated.get("source"))
    source_policy = (policy.get("sources") or {}).get(source)
    if not isinstance(source_policy, Mapping) or not source_policy.get(
        "selection_passed"
    ):
        raise ValueError(f"No accepted relation prompt for source: {source}")
    selected_variant = source_policy.get("selected_variant")
    if source == OPEN_IMAGES_SOURCE and selected_variant == "v2":
        return apply_relation_prompt_v2(updated)
    if source == VISUAL_GENOME_SOURCE and selected_variant == "v3":
        return apply_visual_genome_relation_prompt_v3(updated)
    raise ValueError(
        f"Unsupported locked relation prompt selection: {source}={selected_variant}"
    )
