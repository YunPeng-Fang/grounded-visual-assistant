"""Parsing and integrity helpers for the official COCO POPE benchmark."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


POPE_STRATEGIES = ("random", "popular", "adversarial")
POPE_IMAGE_PATTERN = re.compile(r"^COCO_val2014_(\d{12})\.jpg$")
POPE_QUESTION_PATTERN = re.compile(
    r"^\s*is there (?:a|an|any) (.+?) in the image\?\s*$",
    flags=re.IGNORECASE,
)


def sha256sum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Read either an ordinary JSON array or the JSONL used by POPE."""
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"POPE metadata is empty: {source}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"POPE JSON array must contain objects: {source}")
        return payload
    if isinstance(payload, dict):
        return [payload]

    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON at {source}:{line_number}"
            ) from error
        if not isinstance(item, dict):
            raise ValueError(
                f"POPE record must be an object at {source}:{line_number}"
            )
        records.append(item)
    if not records:
        raise ValueError(f"POPE metadata has no records: {source}")
    return records


def extract_object_phrase(question: str) -> str:
    match = POPE_QUESTION_PATTERN.fullmatch(str(question))
    if not match:
        raise ValueError(f"Unsupported POPE question template: {question!r}")
    phrase = " ".join(match.group(1).strip().lower().split())
    if not phrase:
        raise ValueError("POPE question contains an empty object phrase.")
    return phrase


def normalize_questions(
    records: Iterable[Mapping[str, Any]], strategy: str
) -> list[dict[str, Any]]:
    """Validate official records and convert them to the project schema."""
    if strategy not in POPE_STRATEGIES:
        raise ValueError(f"Unsupported POPE strategy: {strategy}")
    normalized = []
    seen_ids = set()
    for index, item in enumerate(records, start=1):
        missing = {
            key
            for key in ("question_id", "image", "text", "label")
            if key not in item
        }
        if missing:
            raise ValueError(
                f"POPE {strategy} record {index} is missing: {sorted(missing)}"
            )
        question_id = str(item["question_id"])
        if question_id in seen_ids:
            raise ValueError(
                f"Duplicate POPE {strategy} question_id: {question_id}"
            )
        seen_ids.add(question_id)

        image = str(item["image"])
        image_match = POPE_IMAGE_PATTERN.fullmatch(image)
        if not image_match:
            raise ValueError(f"Unsafe or unexpected POPE image name: {image}")
        question = " ".join(str(item["text"]).strip().split())
        label = str(item["label"]).strip().lower()
        if label not in {"yes", "no"}:
            raise ValueError(
                f"POPE label must be yes/no, found {item['label']!r}"
            )
        normalized.append(
            {
                "id": f"pope_coco_{strategy}_{question_id}",
                "benchmark": "POPE",
                "dataset": "COCO val2014",
                "strategy": strategy,
                "question_id": item["question_id"],
                "image": f"data/pope/images/{image}",
                "image_file": image,
                "image_id": int(image_match.group(1)),
                "question": question,
                "object": extract_object_phrase(question),
                "gt_answer": label,
                "task_type": "object_existence",
            }
        )
    return normalized


def question_statistics(
    questions_by_strategy: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    per_strategy = {}
    all_ids = set()
    all_images = set()
    for strategy in POPE_STRATEGIES:
        questions = questions_by_strategy[strategy]
        ids = [str(item["id"]) for item in questions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate normalized IDs in {strategy}.")
        overlap = all_ids.intersection(ids)
        if overlap:
            raise ValueError(f"Cross-strategy duplicate ID: {sorted(overlap)[0]}")
        all_ids.update(ids)
        images = {str(item["image_file"]) for item in questions}
        all_images.update(images)
        labels = Counter(str(item["gt_answer"]) for item in questions)
        per_strategy[strategy] = {
            "questions": len(questions),
            "images": len(images),
            "yes": labels["yes"],
            "no": labels["no"],
        }
    return {
        "questions": len(all_ids),
        "images": len(all_images),
        "per_strategy": per_strategy,
    }


def build_image_manifest(
    questions_by_strategy: Mapping[str, list[Mapping[str, Any]]],
    *,
    image_base_url: str = "https://images.cocodataset.org/val2014",
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for strategy in POPE_STRATEGIES:
        for item in questions_by_strategy[strategy]:
            file_name = str(item["image_file"])
            record = grouped.setdefault(
                file_name,
                {
                    "image_file": file_name,
                    "image_id": int(item["image_id"]),
                    "relative_path": f"images/{file_name}",
                    "url": f"{image_base_url.rstrip('/')}/{file_name}",
                    "strategies": set(),
                    "questions": 0,
                },
            )
            record["strategies"].add(strategy)
            record["questions"] += 1
    return [
        {
            **item,
            "strategies": [
                strategy
                for strategy in POPE_STRATEGIES
                if strategy in item["strategies"]
            ],
        }
        for _, item in sorted(grouped.items())
    ]
