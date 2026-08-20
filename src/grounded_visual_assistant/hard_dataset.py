"""Cross-dataset hard-case indexing, scoring, selection, and splitting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .dataset_splits import multilabel_stratified_split


OPEN_IMAGES_SOURCE = "open_images_v7_validation"
VISUAL_GENOME_SOURCE = "visual_genome_v1_4"
OPEN_IMAGES_VALIDATION_URL = (
    "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
)

RELATION_ALIASES = {
    "above": "above",
    "below": "below",
    "left": "to the left of",
    "left of": "to the left of",
    "on left of": "to the left of",
    "right": "to the right of",
    "right of": "to the right of",
    "on right of": "to the right of",
    "to the left of": "to the left of",
    "to the right of": "to the right of",
}

DIFFICULTY_WEIGHTS = {
    "tiny_object": 2.0,
    "small_object": 1.0,
    "occluded": 1.5,
    "truncated": 1.0,
    "group_of": 1.0,
    "dense_instances": 1.5,
    "repeated_category": 1.0,
    "category_diversity": 1.0,
    "long_tail_category": 2.0,
    "spatial_relation": 1.5,
    "dense_relations": 1.0,
}


def canonical_text(value: Any) -> str:
    """Normalize a source label without inventing semantic aliases."""
    return " ".join(str(value or "").replace("_", " ").strip().lower().split())


def parse_flag(value: Any) -> bool:
    return str(value or "0").strip().lower() in {"1", "true", "yes"}


def iter_json_array(path: str | Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Stream a top-level JSON array without loading the full file into RAM."""
    source = Path(path)
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    finished = False

    with source.open("r", encoding="utf-8") as handle:
        while not finished:
            chunk = handle.read(chunk_size)
            eof = chunk == ""
            buffer += chunk

            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if not buffer.startswith("["):
                        raise ValueError(f"Expected a JSON array in {source}.")
                    buffer = buffer[1:]
                    started = True
                    continue

                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    buffer = buffer[1:]
                    finished = True
                    break
                if not buffer:
                    break

                try:
                    item, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                yield item
                buffer = buffer[end:]

            if eof:
                if not finished:
                    raise ValueError(f"Incomplete JSON array in {source}.")
                if buffer.strip():
                    raise ValueError(f"Unexpected trailing JSON content in {source}.")
                break


def load_open_images_classes(path: str | Path) -> dict[str, str]:
    """Load the headerless MID-to-name table used by Open Images."""
    classes: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            label_id = row[0].strip()
            if label_id.lower() in {"labelname", "label_id"}:
                continue
            classes[label_id] = canonical_text(row[1])
    if not classes:
        raise ValueError(f"No Open Images classes found in {path}.")
    return classes


def _area_ratio(box: Iterable[float]) -> float:
    xmin, ymin, xmax, ymax = [float(value) for value in box]
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def _tail_threshold(category_image_counts: Mapping[str, int]) -> int:
    frequencies = sorted(int(value) for value in category_image_counts.values())
    if not frequencies:
        return 0
    index = max(0, math.ceil(len(frequencies) * 0.2) - 1)
    return frequencies[index]


def _difficulty(
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    category_image_counts: Mapping[str, int],
) -> dict[str, Any]:
    categories = [str(item["category"]) for item in objects]
    category_counts = Counter(categories)
    areas = [float(item.get("area_ratio", 0.0)) for item in objects]
    tags: list[str] = []

    if areas and min(areas) <= 0.005:
        tags.append("tiny_object")
    elif areas and min(areas) <= 0.02:
        tags.append("small_object")
    if any(item.get("is_occluded") for item in objects):
        tags.append("occluded")
    if any(item.get("is_truncated") for item in objects):
        tags.append("truncated")
    if any(item.get("is_group_of") for item in objects):
        tags.append("group_of")
    if len(objects) >= 8:
        tags.append("dense_instances")
    if category_counts and max(category_counts.values()) >= 3:
        tags.append("repeated_category")
    if len(category_counts) >= 5:
        tags.append("category_diversity")

    tail_threshold = _tail_threshold(category_image_counts)
    tail_categories = sorted(
        category
        for category in category_counts
        if category_image_counts.get(category, 0) <= tail_threshold
    )
    if tail_categories:
        tags.append("long_tail_category")
    if relations:
        tags.append("spatial_relation")
    if len(relations) >= 5:
        tags.append("dense_relations")

    score = round(sum(DIFFICULTY_WEIGHTS[tag] for tag in tags), 3)
    return {
        "score": score,
        "tags": tags,
        "statistics": {
            "instances": len(objects),
            "categories": len(category_counts),
            "relations": len(relations),
            "minimum_area_ratio": round(min(areas), 8) if areas else None,
            "occluded_instances": sum(
                bool(item.get("is_occluded")) for item in objects
            ),
            "truncated_instances": sum(
                bool(item.get("is_truncated")) for item in objects
            ),
            "tail_frequency_threshold": tail_threshold,
            "tail_categories": tail_categories,
        },
    }


def derive_geometric_relation(
    objects: list[dict[str, Any]],
    *,
    minimum_separation: float = 0.12,
    minimum_axis_gap: float = 0.04,
) -> dict[str, Any] | None:
    """Choose an unambiguous four-way relation from normalized boxes."""
    largest_by_category: dict[str, dict[str, Any]] = {}
    for item in objects:
        category = str(item["category"])
        previous = largest_by_category.get(category)
        if previous is None or item["area_ratio"] > previous["area_ratio"]:
            largest_by_category[category] = item

    ranked = sorted(
        largest_by_category.values(),
        key=lambda item: float(item["area_ratio"]),
        reverse=True,
    )[:8]
    candidates: list[tuple[float, dict[str, Any]]] = []
    for index, subject in enumerate(ranked):
        for object_item in ranked[index + 1 :]:
            sx1, sy1, sx2, sy2 = subject["bbox_xyxy_normalized"]
            ox1, oy1, ox2, oy2 = object_item["bbox_xyxy_normalized"]
            dx = ((sx1 + sx2) - (ox1 + ox2)) / 2
            dy = ((sy1 + sy2) - (oy1 + oy2)) / 2
            separation = max(abs(dx), abs(dy))
            axis_gap = abs(abs(dx) - abs(dy))
            if separation < minimum_separation or axis_gap < minimum_axis_gap:
                continue
            if abs(dx) > abs(dy):
                predicate = "to the right of" if dx > 0 else "to the left of"
            else:
                predicate = "below" if dy > 0 else "above"
            relation = {
                "subject_annotation_id": subject["annotation_id"],
                "subject_category": subject["category"],
                "object_annotation_id": object_item["annotation_id"],
                "object_category": object_item["category"],
                "predicate": predicate,
                "native_predicate": "derived_from_box_centers",
                "separation": round(separation, 6),
                "axis_gap": round(axis_gap, 6),
            }
            candidates.append((separation + axis_gap, relation))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def load_open_images_candidates(
    boxes_path: str | Path,
    classes_path: str | Path,
) -> list[dict[str, Any]]:
    """Index eligible Open Images validation examples from official CSV files."""
    class_names = load_open_images_classes(classes_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    category_images: dict[str, set[str]] = defaultdict(set)

    with Path(boxes_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Open Images boxes are missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            image_id = str(row["ImageID"]).strip()
            label_id = str(row["LabelName"]).strip()
            category = class_names.get(label_id)
            if not image_id or not category:
                continue
            box = [
                float(row["XMin"]),
                float(row["YMin"]),
                float(row["XMax"]),
                float(row["YMax"]),
            ]
            if not all(0.0 <= value <= 1.0 for value in box):
                continue
            area_ratio = _area_ratio(box)
            if area_ratio <= 0:
                continue
            grouped[image_id].append(
                {
                    "annotation_id": f"{image_id}:{row_number}",
                    "category": category,
                    "source_label_id": label_id,
                    "bbox_xyxy_normalized": [round(value, 8) for value in box],
                    "area_ratio": round(area_ratio, 8),
                    "is_occluded": parse_flag(row.get("IsOccluded")),
                    "is_truncated": parse_flag(row.get("IsTruncated")),
                    "is_group_of": parse_flag(row.get("IsGroupOf")),
                }
            )
            category_images[category].add(image_id)

    category_image_counts = {
        category: len(image_ids) for category, image_ids in category_images.items()
    }
    candidates = []
    for image_id, objects in grouped.items():
        relation = derive_geometric_relation(
            [item for item in objects if not item.get("is_group_of")]
        )
        if relation is None:
            continue
        relations = [relation]
        categories = sorted({item["category"] for item in objects})
        candidates.append(
            {
                "schema_version": 1,
                "sample_id": f"open_images:{image_id}",
                "source": OPEN_IMAGES_SOURCE,
                "source_image_id": image_id,
                "image": {
                    "url": OPEN_IMAGES_VALIDATION_URL.format(image_id=image_id),
                    "relative_path": f"images/open_images/{image_id}.jpg",
                    "width": None,
                    "height": None,
                },
                "categories": categories,
                "objects": objects,
                "relations": relations,
                "annotation_scope": "exhaustive_for_verified_boxable_classes",
                "difficulty": _difficulty(
                    objects, relations, category_image_counts
                ),
            }
        )
    return sorted(candidates, key=lambda item: item["sample_id"])


def _entity_surface_name(entity: Mapping[str, Any]) -> str:
    names = entity.get("names")
    if isinstance(names, list) and names:
        return canonical_text(names[0])
    return canonical_text(entity.get("name"))


def _entity_name(entity: Mapping[str, Any]) -> str:
    synsets = entity.get("synsets")
    if isinstance(synsets, list) and synsets:
        lemma = str(synsets[0]).split(".", 1)[0]
        if lemma:
            return canonical_text(lemma)
    name = _entity_surface_name(entity)
    return name[:-2].strip() if name.endswith("'s") else name


def _visual_genome_box(
    entity: Mapping[str, Any], width: int, height: int
) -> list[float] | None:
    try:
        x = float(entity["x"])
        y = float(entity["y"])
        w = float(entity["w"])
        h = float(entity["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None
    return [
        max(0.0, min(1.0, x / width)),
        max(0.0, min(1.0, y / height)),
        max(0.0, min(1.0, (x + w) / width)),
        max(0.0, min(1.0, (y + h) / height)),
    ]


def load_visual_genome_image_data(path: str | Path) -> dict[str, dict[str, Any]]:
    metadata = {}
    for item in iter_json_array(path):
        image_id = str(item.get("image_id", "")).strip()
        if image_id:
            metadata[image_id] = item
    if not metadata:
        raise ValueError(f"No Visual Genome image metadata found in {path}.")
    return metadata


def load_visual_genome_candidates(
    relationships_path: str | Path,
    image_data_path: str | Path,
    *,
    excluded_coco_ids: Iterable[int] = (),
) -> list[dict[str, Any]]:
    """Index explicit four-way spatial relations from Visual Genome."""
    image_data = load_visual_genome_image_data(image_data_path)
    excluded = {int(value) for value in excluded_coco_ids}
    raw_candidates: list[dict[str, Any]] = []
    category_images: dict[str, set[str]] = defaultdict(set)

    for image_entry in iter_json_array(relationships_path):
        image_id = str(image_entry.get("image_id", "")).strip()
        metadata = image_data.get(image_id)
        if not metadata:
            continue
        coco_id = metadata.get("coco_id")
        if coco_id is not None and int(coco_id) in excluded:
            continue
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
        if width <= 0 or height <= 0:
            continue

        image_relationships = image_entry.get("relationships", [])
        all_objects_by_id: dict[str, dict[str, Any]] = {}
        for relation_index, relation in enumerate(image_relationships):
            for role in ("subject", "object"):
                entity = relation.get(role) or {}
                entity_name = _entity_name(entity)
                entity_box = _visual_genome_box(entity, width, height)
                if not entity_name or entity_box is None:
                    continue
                entity_id = str(
                    entity.get("object_id")
                    or f"relationship:{relation_index}:{role}"
                )
                all_objects_by_id.setdefault(
                    entity_id,
                    {
                        "annotation_id": entity_id,
                        "category": entity_name,
                        "bbox_xyxy_normalized": [
                            round(value, 8) for value in entity_box
                        ],
                        "area_ratio": round(_area_ratio(entity_box), 8),
                        "is_occluded": False,
                        "is_truncated": False,
                        "is_group_of": False,
                    },
                )

        objects_by_id: dict[str, dict[str, Any]] = {}
        relations: list[dict[str, Any]] = []
        seen_relations: set[tuple[str, str, str]] = set()
        for relation_index, relation in enumerate(image_relationships):
            predicate_native = canonical_text(relation.get("predicate"))
            predicate = RELATION_ALIASES.get(predicate_native)
            subject = relation.get("subject") or {}
            object_item = relation.get("object") or {}
            subject_id = str(
                subject.get("object_id")
                or f"relationship:{relation_index}:subject"
            )
            object_id = str(
                object_item.get("object_id")
                or f"relationship:{relation_index}:object"
            )
            subject_record = all_objects_by_id.get(subject_id)
            object_record = all_objects_by_id.get(object_id)
            if not predicate or not subject_record or not object_record:
                continue
            subject_name = subject_record["category"]
            object_name = object_record["category"]
            if subject_name == object_name:
                continue
            relation_key = (subject_id, predicate, object_id)
            if relation_key in seen_relations:
                continue
            seen_relations.add(relation_key)

            objects_by_id.setdefault(subject_id, subject_record)
            objects_by_id.setdefault(object_id, object_record)
            relations.append(
                {
                    "relationship_id": relation.get("relationship_id"),
                    "subject_annotation_id": subject_id,
                    "subject_category": subject_name,
                    "native_subject_category": _entity_surface_name(subject),
                    "object_annotation_id": object_id,
                    "object_category": object_name,
                    "native_object_category": _entity_surface_name(object_item),
                    "predicate": predicate,
                    "native_predicate": predicate_native,
                }
            )

        if len(objects_by_id) < 2 or not relations:
            continue

        objects_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in all_objects_by_id.values():
            objects_by_category[item["category"]].append(item)
        largest_ids: dict[str, str] = {}
        for category, category_objects in objects_by_category.items():
            ranked = sorted(
                category_objects,
                key=lambda item: (-item["area_ratio"], item["annotation_id"]),
            )
            if len(ranked) == 1 or (
                ranked[0]["area_ratio"] > ranked[1]["area_ratio"] + 1e-8
            ):
                largest_ids[category] = ranked[0]["annotation_id"]

        safe_relations = []
        for relation in relations:
            endpoint_is_safe = True
            for role in ("subject", "object"):
                category = relation[f"{role}_category"]
                annotation_id = relation[f"{role}_annotation_id"]
                category_objects = objects_by_category[category]
                if len(category_objects) == 1:
                    relation[f"{role}_instance_rule"] = "unique_category"
                elif largest_ids.get(category) == annotation_id:
                    relation[f"{role}_instance_rule"] = "largest_instance"
                else:
                    endpoint_is_safe = False
                    break
            if endpoint_is_safe:
                safe_relations.append(relation)
        if not safe_relations:
            continue

        safe_object_ids = {
            relation[f"{role}_annotation_id"]
            for relation in safe_relations
            for role in ("subject", "object")
        }
        objects = [
            item
            for annotation_id, item in objects_by_id.items()
            if annotation_id in safe_object_ids
        ]
        categories = sorted({item["category"] for item in objects})
        for category in categories:
            category_images[category].add(image_id)
        raw_candidates.append(
            {
                "schema_version": 1,
                "sample_id": f"visual_genome:{image_id}",
                "source": VISUAL_GENOME_SOURCE,
                "source_image_id": image_id,
                "image": {
                    "url": metadata.get("url"),
                    "relative_path": f"images/visual_genome/{image_id}.jpg",
                    "width": width,
                    "height": height,
                },
                "categories": categories,
                "objects": objects,
                "relations": safe_relations,
                "annotation_scope": (
                    "unambiguous_explicit_spatial_relationship_endpoints"
                ),
                "provenance": {"coco_id": coco_id},
            }
        )

    category_image_counts = {
        category: len(image_ids) for category, image_ids in category_images.items()
    }
    for candidate in raw_candidates:
        candidate["difficulty"] = _difficulty(
            candidate["objects"],
            candidate["relations"],
            category_image_counts,
        )
    return sorted(raw_candidates, key=lambda item: item["sample_id"])


def select_hard_candidates(
    candidates: Iterable[dict[str, Any]],
    source_quotas: Mapping[str, int],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Select difficult examples while rewarding underrepresented tags/classes."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["source"])].append(candidate)

    selected: list[dict[str, Any]] = []
    for source, quota in source_quotas.items():
        pool = list(grouped.get(source, []))
        if quota < 1:
            continue
        if len(pool) < quota:
            raise ValueError(
                f"Source {source} has {len(pool)} eligible candidates; requested {quota}."
            )
        rng = random.Random(f"{seed}:{source}")
        tie_rank = {item["sample_id"]: rng.random() for item in pool}
        tag_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()

        for _ in range(quota):
            def rank(item: Mapping[str, Any]) -> tuple[float, float]:
                tags = item["difficulty"]["tags"]
                categories = item["categories"]
                diversity_bonus = sum(1 / (1 + tag_counts[tag]) for tag in tags)
                category_bonus = 0.2 * sum(
                    1 / (1 + category_counts[category])
                    for category in categories[:10]
                )
                score = float(item["difficulty"]["score"])
                return score + diversity_bonus + category_bonus, tie_rank[item["sample_id"]]

            winner = max(pool, key=rank)
            pool.remove(winner)
            selected.append(winner)
            tag_counts.update(winner["difficulty"]["tags"])
            category_counts.update(winner["categories"])
    return sorted(selected, key=lambda item: item["sample_id"])


def split_hard_candidates(
    selected: Iterable[dict[str, Any]],
    *,
    dev_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Create source-balanced, multilabel-stratified Dev/Test sample IDs."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be strictly between 0 and 1.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        grouped[str(item["source"])].append(item)

    dev_ids: list[str] = []
    test_ids: list[str] = []
    for source, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item["sample_id"])
        if len(items) < 2:
            raise ValueError(f"Source {source} needs at least two selected samples.")
        dev_size = min(len(items) - 1, max(1, round(len(items) * dev_fraction)))
        features = {
            index: {
                *(f"tag:{tag}" for tag in item["difficulty"]["tags"]),
                *(f"category:{category}" for category in item["categories"]),
            }
            for index, item in enumerate(items)
        }
        source_seed = int(
            hashlib.sha256(f"{seed}:{source}".encode("utf-8")).hexdigest()[:8],
            16,
        )
        dev_indices, test_indices = multilabel_stratified_split(
            features,
            dev_size=dev_size,
            seed=source_seed,
            protect_singletons=False,
        )
        dev_ids.extend(items[index]["sample_id"] for index in dev_indices)
        test_ids.extend(items[index]["sample_id"] for index in test_indices)
    return sorted(dev_ids), sorted(test_ids)


def hard_dataset_statistics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    sources = Counter(str(item["source"]) for item in records)
    tags = Counter(
        tag for item in records for tag in item["difficulty"].get("tags", [])
    )
    categories = {category for item in records for category in item["categories"]}
    return {
        "samples": len(records),
        "sources": dict(sorted(sources.items())),
        "categories": len(categories),
        "difficulty_tags": dict(sorted(tags.items())),
        "mean_difficulty_score": round(
            sum(float(item["difficulty"]["score"]) for item in records)
            / len(records),
            6,
        )
        if records
        else 0.0,
    }
