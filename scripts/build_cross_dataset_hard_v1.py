"""Build a frozen cross-dataset hard-case candidate manifest and split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.hard_dataset import (
    OPEN_IMAGES_SOURCE,
    VISUAL_GENOME_SOURCE,
    hard_dataset_statistics,
    load_open_images_candidates,
    load_visual_genome_candidates,
    select_hard_candidates,
    split_hard_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and split Open Images and Visual Genome hard cases without "
            "downloading image pixels or generating benchmark questions."
        )
    )
    parser.add_argument(
        "--open-images-boxes",
        default="data/raw/open_images/validation-annotations-bbox.csv",
    )
    parser.add_argument(
        "--open-images-classes",
        default="data/raw/open_images/class-descriptions-boxable.csv",
    )
    parser.add_argument(
        "--visual-genome-relationships",
        default="data/raw/visual_genome/relationships.json",
    )
    parser.add_argument(
        "--visual-genome-image-data",
        default="data/raw/visual_genome/image_data.json",
    )
    parser.add_argument(
        "--exclude-questions",
        default="data/eval_v0/questions.jsonl",
        help="Existing COCO questions whose image IDs must be excluded.",
    )
    parser.add_argument("--open-images-count", type=int, default=200)
    parser.add_argument("--visual-genome-count", type=int, default=200)
    parser.add_argument("--dev-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", default="data/cross_dataset_hard_v1"
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_excluded_coco_ids(path: Path) -> set[int]:
    image_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                image_ids.add(int(item["image_id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid question record at {path}:{line_number}."
                ) from error
    if not image_ids:
        raise ValueError(f"No excluded COCO image IDs found in {path}.")
    return image_ids


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    boxes_path = project_path(args.open_images_boxes)
    classes_path = project_path(args.open_images_classes)
    relationships_path = project_path(args.visual_genome_relationships)
    image_data_path = project_path(args.visual_genome_image_data)
    excluded_questions_path = project_path(args.exclude_questions)
    output_dir = project_path(args.output_dir)

    input_paths = {
        "open_images_boxes": boxes_path,
        "open_images_classes": classes_path,
        "visual_genome_relationships": relationships_path,
        "visual_genome_image_data": image_data_path,
        "excluded_questions": excluded_questions_path,
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required metadata files:\n- " + "\n- ".join(missing)
        )

    excluded_coco_ids = load_excluded_coco_ids(excluded_questions_path)
    print("Indexing Open Images annotations...")
    open_images = load_open_images_candidates(boxes_path, classes_path)
    print(f"Eligible Open Images candidates: {len(open_images)}")
    print("Indexing Visual Genome relationships...")
    visual_genome = load_visual_genome_candidates(
        relationships_path,
        image_data_path,
        excluded_coco_ids=excluded_coco_ids,
    )
    print(f"Eligible Visual Genome candidates: {len(visual_genome)}")

    quotas = {
        OPEN_IMAGES_SOURCE: args.open_images_count,
        VISUAL_GENOME_SOURCE: args.visual_genome_count,
    }
    selected = select_hard_candidates(
        [*open_images, *visual_genome], quotas, seed=args.seed
    )
    dev_ids, test_ids = split_hard_candidates(
        selected, dev_fraction=args.dev_fraction, seed=args.seed
    )
    dev_set = set(dev_ids)
    test_set = set(test_ids)
    if dev_set & test_set or dev_set | test_set != {
        item["sample_id"] for item in selected
    }:
        raise RuntimeError("Generated hard-case splits are not disjoint and complete.")

    records = []
    for item in selected:
        record = dict(item)
        record["split"] = "dev" if item["sample_id"] in dev_set else "test"
        records.append(record)
    write_jsonl(output_dir / "candidates.jsonl", records)
    write_json(
        output_dir / "splits" / "dev_sample_ids.json",
        {"name": "hard_dev", "sample_ids": dev_ids},
    )
    write_json(
        output_dir / "splits" / "test_sample_ids.json",
        {"name": "hard_test", "sample_ids": test_ids},
    )
    write_jsonl(
        output_dir / "download_manifest.jsonl",
        [
            {
                "sample_id": item["sample_id"],
                "source": item["source"],
                "url": item["image"]["url"],
                "relative_path": item["image"]["relative_path"],
                "split": "dev" if item["sample_id"] in dev_set else "test",
            }
            for item in selected
        ],
    )

    manifest = {
        "name": "cross_dataset_hard_v1",
        "schema_version": 1,
        "seed": args.seed,
        "selection": {
            "source_quotas": quotas,
            "dev_fraction": args.dev_fraction,
            "difficulty_selection": "weighted_tags_with_diversity_bonus",
            "split_singletons": "deterministic_balancing_without_test_bias",
            "group_boxes_allowed_as_relation_endpoints": False,
        },
        "exclusions": {
            "coco_image_ids": len(excluded_coco_ids),
            "visual_genome_coco_overlap_removed": True,
            "content_hash_dedup_pending_image_download": True,
        },
        "statistics": {
            "all": hard_dataset_statistics(records),
            "dev": hard_dataset_statistics(
                item for item in records if item["split"] == "dev"
            ),
            "test": hard_dataset_statistics(
                item for item in records if item["split"] == "test"
            ),
        },
        "input_sha256": {
            name: sha256sum(path) for name, path in input_paths.items()
        },
        "next_required_step": (
            "Download only selected images, verify dimensions and licenses, "
            "then remove exact and perceptual duplicates before question generation."
        ),
    }
    write_json(output_dir / "manifest.json", manifest)

    print(f"Selected: {len(records)}")
    print(f"Dev/Test: {len(dev_ids)}/{len(test_ids)}")
    print(f"Output:   {output_dir}")
    print(json.dumps(manifest["statistics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
