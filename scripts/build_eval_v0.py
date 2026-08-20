"""Download and build a compact COCO-based VLM evaluation set.

The generated dataset contains three questions per image:

1. object listing
2. balanced object existence
3. coarse spatial relation

COCO bounding boxes are kept as evidence so the same records can later be used
for grounding and segmentation experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


COCO_HOST = "images.cocodataset.org"
ANNOTATIONS_PATH = "/annotations/annotations_trainval2017.zip"
IMAGE_PATH_TEMPLATE = "/val2017/{file_name}"
ANNOTATION_MEMBER = "annotations/instances_val2017.json"
ANNOTATIONS_MD5 = "f4bbac642086de4f52a3fdda2de5fa2c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact, automatically scored COCO VLM benchmark."
    )
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--relation-margin", type=float, default=0.08)
    parser.add_argument(
        "--transport",
        choices=("https", "http"),
        default="https",
        help="Use http only when a local proxy breaks COCO HTTPS certificates.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/coco"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval_v0"))
    return parser.parse_args()


def download_file(url: str, destination: Path, label: str) -> None:
    """Download a URL unless a non-empty destination already exists."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing {label}: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    curl = shutil.which("curl")
    if curl:
        command = [
            curl,
            "--location",
            "--fail",
            "--retry",
            "3",
            "--connect-timeout",
            "30",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ]
        print(f"Downloading {label} with curl")
        subprocess.run(command, check=True)
        partial.replace(destination)
        return

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        total = int(response.headers.get("Content-Length", 0))
        with partial.open("wb") as output, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=label,
        ) as progress:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                progress.update(len(chunk))
    partial.replace(destination)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_annotations(raw_dir: Path, transport: str) -> Path:
    archive = raw_dir / "annotations_trainval2017.zip"
    annotation_path = raw_dir / ANNOTATION_MEMBER
    annotations_url = f"{transport}://{COCO_HOST}{ANNOTATIONS_PATH}"

    if archive.exists() and md5sum(archive) != ANNOTATIONS_MD5:
        print(f"Removing incomplete or corrupt archive: {archive}")
        archive.unlink()
    download_file(annotations_url, archive, "COCO annotations")

    actual_md5 = md5sum(archive)
    if actual_md5 != ANNOTATIONS_MD5:
        raise RuntimeError(
            "COCO annotation archive checksum mismatch: "
            f"expected {ANNOTATIONS_MD5}, got {actual_md5}. "
            f"Delete the incomplete file and retry: {archive}"
        )

    if not annotation_path.exists():
        print(f"Extracting {ANNOTATION_MEMBER}")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extract(ANNOTATION_MEMBER, raw_dir)
    return annotation_path


def load_coco(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evidence_for(
    annotations: list[dict[str, Any]],
    category_names: dict[int, str],
    category_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    evidence = []
    for annotation in annotations:
        category_id = annotation["category_id"]
        if category_ids is not None and category_id not in category_ids:
            continue
        evidence.append(
            {
                "annotation_id": annotation["id"],
                "category": category_names[category_id],
                "bbox_xywh": [round(value, 2) for value in annotation["bbox"]],
            }
        )
    return evidence


def choose_relation(
    annotations: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    category_names: dict[int, str],
    margin: float,
) -> dict[str, Any] | None:
    largest_by_category: dict[int, dict[str, Any]] = {}
    for annotation in annotations:
        category_id = annotation["category_id"]
        previous = largest_by_category.get(category_id)
        if previous is None or annotation["area"] > previous["area"]:
            largest_by_category[category_id] = annotation

    ranked = sorted(
        largest_by_category.values(), key=lambda item: item["area"], reverse=True
    )[:6]
    candidates = []
    for first_index, first in enumerate(ranked):
        for second in ranked[first_index + 1 :]:
            first_x, first_y, first_w, first_h = first["bbox"]
            second_x, second_y, second_w, second_h = second["bbox"]
            dx = ((first_x + first_w / 2) - (second_x + second_w / 2)) / image_width
            dy = ((first_y + first_h / 2) - (second_y + second_h / 2)) / image_height
            dominance = max(abs(dx), abs(dy))
            if dominance < margin:
                continue

            if abs(dx) >= abs(dy):
                answer = "to the right of" if dx > 0 else "to the left of"
            else:
                answer = "below" if dy > 0 else "above"
            candidates.append((dominance, first, second, answer))

    if not candidates:
        return None

    _, first, second, answer = max(candidates, key=lambda item: item[0])
    return {
        "first": first,
        "second": second,
        "first_name": category_names[first["category_id"]],
        "second_name": category_names[second["category_id"]],
        "answer": answer,
    }


def build_candidates(
    coco: dict[str, Any], min_area_ratio: float, relation_margin: float
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    images = {image["id"]: image for image in coco["images"]}
    category_names = {
        category["id"]: category["name"] for category in coco["categories"]
    }
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for annotation in coco["annotations"]:
        image = images[annotation["image_id"]]
        image_area = image["width"] * image["height"]
        if annotation.get("iscrowd", 0):
            continue
        if annotation["area"] / image_area < min_area_ratio:
            continue
        grouped[annotation["image_id"]].append(annotation)

    candidates = []
    for image_id, annotations in grouped.items():
        image = images[image_id]
        relation = choose_relation(
            annotations,
            image["width"],
            image["height"],
            category_names,
            relation_margin,
        )
        if relation is None:
            continue
        candidates.append(
            {"image": image, "annotations": annotations, "relation": relation}
        )
    return candidates, category_names


def record(
    *,
    question_id: str,
    relative_image_path: str,
    image_id: int,
    question: str,
    task_type: str,
    answer: str,
    categories: list[str],
    evidence: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "id": question_id,
        "image": relative_image_path,
        "image_id": image_id,
        "question": question,
        "task_type": task_type,
        "gt_answer": answer,
        "source": "COCO val2017",
        "categories": categories,
        "evidence_boxes": evidence,
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def build_dataset(args: argparse.Namespace) -> None:
    annotation_path = prepare_annotations(args.raw_dir, args.transport)
    coco = load_coco(annotation_path)
    candidates, category_names = build_candidates(
        coco, args.min_area_ratio, args.relation_margin
    )
    if len(candidates) < args.num_images:
        raise RuntimeError(
            f"Only {len(candidates)} eligible images were found; "
            f"requested {args.num_images}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.num_images]
    all_category_ids = set(category_names)

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    questions = []

    for index, item in enumerate(tqdm(selected, desc="Building eval_v0")):
        image = item["image"]
        annotations = item["annotations"]
        relation = item["relation"]
        file_name = image["file_name"]
        destination = image_dir / file_name
        download_file(
            (
                f"{args.transport}://{COCO_HOST}"
                f"{IMAGE_PATH_TEMPLATE.format(file_name=file_name)}"
            ),
            destination,
            file_name,
        )

        relative_image_path = destination.as_posix()
        present_ids = {annotation["category_id"] for annotation in annotations}
        present_names = sorted(category_names[item] for item in present_ids)
        image_prefix = f"coco_{image['id']:012d}"

        questions.append(
            record(
                question_id=f"{image_prefix}_listing",
                relative_image_path=relative_image_path,
                image_id=image["id"],
                question="List the main visible object categories in this image.",
                task_type="object_listing",
                answer=", ".join(present_names),
                categories=present_names,
                evidence=evidence_for(annotations, category_names),
            )
        )

        is_positive = index % 2 == 0
        if is_positive:
            target_id = rng.choice(sorted(present_ids))
            existence_answer = "yes"
        else:
            target_id = rng.choice(sorted(all_category_ids - present_ids))
            existence_answer = "no"
        target_name = category_names[target_id]
        questions.append(
            record(
                question_id=f"{image_prefix}_existence",
                relative_image_path=relative_image_path,
                image_id=image["id"],
                question=f"Is there a {target_name} in this image? Answer yes or no.",
                task_type="object_existence",
                answer=existence_answer,
                categories=[target_name],
                evidence=evidence_for(
                    annotations, category_names, {target_id}
                ),
                metadata={"is_positive": is_positive},
            )
        )

        relation_ids = {
            relation["first"]["category_id"],
            relation["second"]["category_id"],
        }
        questions.append(
            record(
                question_id=f"{image_prefix}_relation",
                relative_image_path=relative_image_path,
                image_id=image["id"],
                question=(
                    f"Where is the largest {relation['first_name']} relative to "
                    f"the largest {relation['second_name']}?"
                ),
                task_type="spatial_relation",
                answer=relation["answer"],
                categories=[relation["first_name"], relation["second_name"]],
                evidence=evidence_for(
                    [relation["first"], relation["second"]],
                    category_names,
                    relation_ids,
                ),
                metadata={
                    "instance_rule": "largest annotated instance by area"
                },
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "questions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in questions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "name": "eval_v0",
        "source": "COCO val2017",
        "seed": args.seed,
        "num_images": len(selected),
        "num_questions": len(questions),
        "questions_per_image": 3,
        "task_counts": {
            "object_listing": len(selected),
            "object_existence": len(selected),
            "spatial_relation": len(selected),
        },
        "existence_balance": {
            "yes": sum(
                item["gt_answer"] == "yes"
                for item in questions
                if item["task_type"] == "object_existence"
            ),
            "no": sum(
                item["gt_answer"] == "no"
                for item in questions
                if item["task_type"] == "object_existence"
            ),
        },
        "generation": {
            "min_area_ratio": args.min_area_ratio,
            "relation_margin": args.relation_margin,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Created {len(questions)} questions from {len(selected)} images.")
    print(f"Questions: {jsonl_path}")
    print(f"Manifest:  {manifest_path}")


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
