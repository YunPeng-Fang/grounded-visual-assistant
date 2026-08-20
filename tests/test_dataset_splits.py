from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.dataset_splits import (
    build_image_feature_sets,
    coco_size_name,
    image_ids_sha256,
    load_image_ids,
    multilabel_stratified_split,
)


class DatasetSplitsTest(unittest.TestCase):
    def test_coco_size_name(self) -> None:
        self.assertEqual(coco_size_name(100), "small")
        self.assertEqual(coco_size_name(32**2), "medium")
        self.assertEqual(coco_size_name(96**2), "large")

    def test_split_is_deterministic_disjoint_and_complete(self) -> None:
        features = {
            image_id: {
                f"category:{image_id % 4}",
                "size:small" if image_id % 2 else "size:large",
            }
            for image_id in range(1, 21)
        }
        first = multilabel_stratified_split(features, dev_size=4, seed=2026)
        second = multilabel_stratified_split(features, dev_size=4, seed=2026)
        self.assertEqual(first, second)
        dev_ids, test_ids = first
        self.assertEqual(len(dev_ids), 4)
        self.assertEqual(len(test_ids), 16)
        self.assertFalse(set(dev_ids) & set(test_ids))
        self.assertEqual(set(dev_ids) | set(test_ids), set(features))

    def test_ground_truth_features_include_category_and_size(self) -> None:
        ground_truth = {
            "images": [{"id": 7}],
            "categories": [{"id": 3, "name": "car"}],
            "annotations": [
                {"image_id": 7, "category_id": 3, "area": 400}
            ],
        }
        self.assertEqual(
            build_image_feature_sets(ground_truth)[7],
            frozenset({"category:3", "size:small"}),
        )

    def test_load_image_ids_accepts_metadata_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(
                json.dumps({"name": "dev", "image_ids": [3, 1]}),
                encoding="utf-8",
            )
            self.assertEqual(load_image_ids(path), [3, 1])

    def test_image_id_hash_ignores_order(self) -> None:
        self.assertEqual(image_ids_sha256([3, 1]), image_ids_sha256([1, 3]))


if __name__ == "__main__":
    unittest.main()
