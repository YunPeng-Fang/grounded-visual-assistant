"""Create deterministic development and test image splits for eval_v0."""

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

from grounded_visual_assistant.dataset_splits import (
    build_image_feature_sets,
    multilabel_stratified_split,
    split_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed image-level dev/test splits for grounding experiments."
    )
    parser.add_argument(
        "--ground-truth", default="data/eval_v0/coco_grounding_gt.json"
    )
    parser.add_argument("--output-dir", default="data/eval_v0/splits")
    parser.add_argument("--dev-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
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


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    ground_truth_path = project_path(args.ground_truth)
    output_dir = project_path(args.output_dir)
    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"Ground truth not found: {ground_truth_path}")

    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    feature_sets = build_image_feature_sets(ground_truth)
    dev_ids, test_ids = multilabel_stratified_split(
        feature_sets,
        dev_size=args.dev_size,
        seed=args.seed,
    )
    if set(dev_ids) & set(test_ids):
        raise RuntimeError("Generated dev and test splits overlap.")
    if set(dev_ids) | set(test_ids) != set(feature_sets):
        raise RuntimeError("Generated splits do not cover every image.")

    try:
        source_ground_truth = ground_truth_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source_ground_truth = str(ground_truth_path)
    common = {
        "source_ground_truth": source_ground_truth,
        "source_sha256": sha256sum(ground_truth_path),
        "seed": args.seed,
        "strategy": "greedy_multilabel_category_and_coco_size",
    }
    dev_payload = {
        **common,
        "name": "dev",
        "image_ids": dev_ids,
        "statistics": split_statistics(ground_truth, dev_ids),
    }
    test_payload = {
        **common,
        "name": "test",
        "image_ids": test_ids,
        "statistics": split_statistics(ground_truth, test_ids),
    }
    dev_path = output_dir / "dev_image_ids.json"
    test_path = output_dir / "test_image_ids.json"
    write_json_atomic(dev_path, dev_payload)
    write_json_atomic(test_path, test_payload)

    print(f"Dev:  {dev_path} ({len(dev_ids)} images)")
    print(f"Test: {test_path} ({len(test_ids)} images)")
    print(json.dumps({"dev": dev_payload["statistics"], "test": test_payload["statistics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
