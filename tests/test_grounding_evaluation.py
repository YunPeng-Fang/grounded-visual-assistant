from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.grounding_evaluation import (
    aggregate_grounding_metrics,
    box_iou,
    canonicalize_category,
    evaluate_grounding_image,
    xywh_to_xyxy,
)


class GroundingEvaluationTest(unittest.TestCase):
    def test_box_conversion_and_iou(self) -> None:
        self.assertEqual(xywh_to_xyxy([2, 3, 5, 7]), [2.0, 3.0, 7.0, 10.0])
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_category_alias(self) -> None:
        self.assertEqual(canonicalize_category("sofa", ["couch"]), "couch")

    def test_class_aware_matching(self) -> None:
        ground_truth = [
            {"category": "person", "bbox_xywh": [0, 0, 10, 10]},
            {"category": "dog", "bbox_xywh": [20, 20, 10, 10]},
        ]
        predictions = [
            {"class_name": "person", "bbox": [0, 0, 10, 10], "score": 0.9},
            {"class_name": "person", "bbox": [0, 0, 9, 9], "score": 0.8},
            {"class_name": "cat", "bbox": [20, 20, 30, 30], "score": 0.7},
        ]
        result = evaluate_grounding_image(ground_truth, predictions)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 2)
        self.assertEqual(result["fn"], 1)
        self.assertAlmostEqual(result["precision"], 1 / 3, places=6)
        self.assertEqual(result["recall"], 0.5)

    def test_map50_includes_missed_categories(self) -> None:
        perfect = evaluate_grounding_image(
            [{"category": "person", "bbox_xywh": [0, 0, 10, 10]}],
            [{"class_name": "person", "bbox": [0, 0, 10, 10], "score": 0.9}],
        )
        missed = evaluate_grounding_image(
            [{"category": "dog", "bbox_xywh": [0, 0, 10, 10]}],
            [],
        )
        records = [
            {
                "image_id": 1,
                "evaluation": perfect,
                "latency_seconds": {"total": 1.0, "grounding": 0.4, "sam2": 0.6},
                "postprocessing": {
                    "candidate_count": 2,
                    "kept_count": 1,
                    "suppressed_count": 1,
                },
            },
            {
                "image_id": 2,
                "evaluation": missed,
                "latency_seconds": {"total": 1.0, "grounding": 0.4, "sam2": 0.6},
            },
        ]
        metrics = aggregate_grounding_metrics(records, expected_images=2)
        self.assertEqual(metrics["box_iou_50"]["map50"], 0.5)
        self.assertEqual(metrics["box_iou_50"]["micro_recall"], 0.5)
        self.assertEqual(metrics["postprocessing"]["candidates_total"], 2)
        self.assertEqual(metrics["postprocessing"]["suppressed_total"], 1)


if __name__ == "__main__":
    unittest.main()
