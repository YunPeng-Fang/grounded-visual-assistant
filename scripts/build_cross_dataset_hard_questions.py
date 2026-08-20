"""Generate immutable source-aware questions for the frozen hard benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_benchmark import read_jsonl
from grounded_visual_assistant.hard_dataset import load_open_images_classes
from grounded_visual_assistant.hard_questions import (
    build_hard_questions,
    load_verified_image_labels,
    validate_hard_questions,
)
from grounded_visual_assistant.image_dedup import sha256sum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate restricted-vocabulary Open Images questions and explicit "
            "Visual Genome relation questions from the frozen benchmark."
        )
    )
    parser.add_argument(
        "--frozen-dir", default="data/cross_dataset_hard_v1/frozen"
    )
    parser.add_argument(
        "--verified-image-labels",
        default=(
            "data/raw/open_images/"
            "oidv7-val-annotations-human-imagelabels.csv"
        ),
    )
    parser.add_argument(
        "--open-images-classes",
        default="data/raw/open_images/oidv7-class-descriptions.csv",
    )
    parser.add_argument(
        "--output-dir", default="data/cross_dataset_hard_v1/questions_v1"
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in records
    ).encode("utf-8")


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def verify_frozen_artifacts(frozen_dir: Path) -> dict[str, Any]:
    manifest_path = frozen_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Frozen benchmark manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_path, expected_hash in manifest["artifact_sha256"].items():
        path = frozen_dir / relative_path
        if not path.is_file() or sha256sum(path) != expected_hash:
            raise RuntimeError(f"Frozen artifact is missing or modified: {path}")
    return manifest


def main() -> None:
    args = parse_args()
    frozen_dir = project_path(args.frozen_dir)
    verified_path = project_path(args.verified_image_labels)
    classes_path = project_path(args.open_images_classes)
    output_dir = project_path(args.output_dir)
    for path in (verified_path, classes_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Open Images metadata is missing: {path}")

    frozen_manifest = verify_frozen_artifacts(frozen_dir)
    candidates = read_jsonl(frozen_dir / "candidates.jsonl")
    images = read_jsonl(frozen_dir / "images.jsonl")
    open_image_ids = {
        str(item["source_image_id"])
        for item in candidates
        if str(item["source"]).startswith("open_images")
    }
    verified_labels = load_verified_image_labels(
        verified_path, selected_image_ids=open_image_ids
    )
    class_names = load_open_images_classes(classes_path)
    questions = build_hard_questions(
        candidates,
        images,
        verified_labels,
        class_names,
        seed=args.seed,
    )
    statistics = validate_hard_questions(questions, candidates)
    dev_questions = [item for item in questions if item["split"] == "dev"]
    test_questions = [item for item in questions if item["split"] == "test"]
    artifacts = {
        "questions.jsonl": jsonl_bytes(questions),
        "dev_questions.jsonl": jsonl_bytes(dev_questions),
        "test_questions.jsonl": jsonl_bytes(test_questions),
    }
    manifest = {
        "name": "cross_dataset_hard_questions_v1",
        "schema_version": 1,
        "immutable": True,
        "seed": args.seed,
        "statistics": statistics,
        "protocols": {
            "open_images_listing": "restricted_human_verified_vocabulary",
            "open_images_existence": "balanced_boxes_positive_human_verified_negative",
            "open_images_relation": "largest_non_group_boxes_center_geometry",
            "visual_genome_relation": "explicit_unambiguous_relationship",
            "visual_genome_existence_or_listing": "prohibited",
        },
        "input_sha256": {
            "frozen_manifest": sha256sum(frozen_dir / "manifest.json"),
            "frozen_candidates": sha256sum(frozen_dir / "candidates.jsonl"),
            "frozen_images": sha256sum(frozen_dir / "images.jsonl"),
            "verified_image_labels": sha256sum(verified_path),
            "open_images_classes": sha256sum(classes_path),
        },
        "frozen_pixel_set_sha256": frozen_manifest["pixel_set_sha256"],
        "artifact_sha256": {
            name: bytes_sha256(payload) for name, payload in artifacts.items()
        },
    }
    manifest_payload = json_bytes(manifest)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_payload:
            raise RuntimeError(
                "Question set already exists but current inputs differ. Create a "
                "new version instead of overwriting it."
            )
        for relative_path, payload in artifacts.items():
            path = output_dir / relative_path
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"Question artifact was modified: {path}")
        status = "verified"
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to populate non-empty output directory: {output_dir}"
            )
        for relative_path, payload in artifacts.items():
            write_atomic(output_dir / relative_path, payload)
        write_atomic(manifest_path, manifest_payload)
        status = "created"

    print(f"Question set status: {status}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
