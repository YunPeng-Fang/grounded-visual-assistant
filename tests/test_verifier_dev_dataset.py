from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.verifier_dev_dataset import (
    build_verifier_dev_records,
    records_sha256,
    validate_verifier_dev_records,
)


def synthetic_coco() -> dict:
    return {
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
            {"id": 3, "name": "truck", "supercategory": "vehicle"},
            {"id": 4, "name": "chair", "supercategory": "furniture"},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 100,
                "category_id": 1,
                "bbox": [1, 2, 10, 20],
                "area": 200,
                "iscrowd": 0,
            },
            {
                "id": 11,
                "image_id": 100,
                "category_id": 2,
                "bbox": [5, 6, 30, 20],
                "area": 600,
                "iscrowd": 0,
            },
            {
                "id": 12,
                "image_id": 200,
                "category_id": 4,
                "bbox": [2, 3, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
        ],
    }


class VerifierDevDatasetTest(unittest.TestCase):
    def test_builds_balanced_deterministic_pairs(self) -> None:
        first, summary = build_verifier_dev_records(
            synthetic_coco(),
            dev_image_ids=[100, 200],
            excluded_image_ids=[200],
            seed=2026,
        )
        second, _ = build_verifier_dev_records(
            synthetic_coco(),
            dev_image_ids=[100, 200],
            excluded_image_ids=[200],
            seed=2026,
        )

        self.assertEqual(len(first), 4)
        self.assertEqual(summary["images"], 1)
        self.assertEqual(summary["positive_questions"], 2)
        self.assertEqual(summary["negative_questions"], 2)
        self.assertEqual(summary["excluded_requested_images"], [200])
        self.assertEqual(records_sha256(first), records_sha256(second))
        self.assertNotIn(200, {item["image_id"] for item in first})
        self.assertEqual(
            len({(item["image_id"], item["object"]) for item in first}),
            4,
        )

    def test_validation_rejects_negative_present_category(self) -> None:
        records, _ = build_verifier_dev_records(
            synthetic_coco(),
            dev_image_ids=[100],
            excluded_image_ids=[],
        )
        negative = next(
            item for item in records if item["gt_answer"] == "no"
        )
        negative["object"] = "car"

        with self.assertRaisesRegex(ValueError, "duplicate|GT mismatch"):
            validate_verifier_dev_records(
                records,
                coco_ground_truth=synthetic_coco(),
                allowed_image_ids=[100],
                excluded_image_ids=[],
            )


if __name__ == "__main__":
    unittest.main()
