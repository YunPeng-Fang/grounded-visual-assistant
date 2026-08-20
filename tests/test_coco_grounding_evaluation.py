from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.coco_grounding_evaluation import (
    build_oracle_coco_ground_truth,
    coco_stats_to_dict,
    convert_predictions_to_coco,
    filter_predictions_by_detector_score,
    xyxy_to_xywh,
)


class CocoGroundingEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source_coco = {
            "info": {"description": "test"},
            "licenses": [],
            "images": [
                {"id": 1, "file_name": "1.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "2.jpg", "width": 100, "height": 100},
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "person"},
                {"id": 18, "name": "dog", "supercategory": "animal"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                    "segmentation": [[0, 0, 20, 0, 20, 20, 0, 20]],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [50, 50, 2, 2],
                    "area": 4,
                    "iscrowd": 0,
                    "segmentation": [[50, 50, 52, 50, 52, 52, 50, 52]],
                },
                {
                    "id": 3,
                    "image_id": 1,
                    "category_id": 18,
                    "bbox": [20, 20, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                    "segmentation": [[20, 20, 30, 20, 30, 30, 20, 30]],
                },
            ],
        }
        self.questions = [
            {
                "id": "1_listing",
                "image_id": 1,
                "task_type": "object_listing",
                "categories": ["person"],
                "evidence_boxes": [{"category": "person", "bbox_xywh": [0, 0, 20, 20]}],
            },
            {
                "id": "2_listing",
                "image_id": 2,
                "task_type": "object_listing",
                "categories": ["dog"],
                "evidence_boxes": [],
            },
        ]

    def test_box_conversion(self) -> None:
        self.assertEqual(xyxy_to_xywh([2, 3, 7, 11]), [2.0, 3.0, 5.0, 8.0])

    def test_ground_truth_restores_small_prompted_instances(self) -> None:
        ground_truth, report = build_oracle_coco_ground_truth(
            self.source_coco, self.questions
        )
        annotation_ids = {item["id"] for item in ground_truth["annotations"]}
        self.assertEqual(annotation_ids, {1, 2})
        self.assertEqual(report["filtered_eval_v0_boxes"], 1)
        self.assertEqual(report["restored_full_instances"], 2)
        self.assertEqual(report["additional_instances"], 1)

    def test_prediction_conversion_keeps_bbox_and_mask(self) -> None:
        ground_truth, _ = build_oracle_coco_ground_truth(
            self.source_coco, self.questions
        )
        predictions = [
            {
                "image_id": 1,
                "target_categories": ["person"],
                "annotations": [
                    {
                        "class_name": "person",
                        "bbox": [0, 0, 20, 20],
                        "score": 0.8,
                        "mask_score": 0.9,
                        "segmentation": {"size": [100, 100], "counts": "encoded"},
                    },
                    {
                        "class_name": "mystery object",
                        "bbox": [0, 0, 10, 10],
                        "score": 0.5,
                        "segmentation": {"size": [100, 100], "counts": "encoded"},
                    },
                ],
            }
        ]
        bbox, segmentation, report = convert_predictions_to_coco(
            predictions, ground_truth, segmentation_score_mode="product"
        )
        self.assertEqual(len(bbox), 1)
        self.assertEqual(bbox[0]["bbox"], [0.0, 0.0, 20.0, 20.0])
        self.assertEqual(len(segmentation), 1)
        self.assertAlmostEqual(segmentation[0]["score"], 0.72)
        self.assertEqual(report["skipped"]["unmapped_label"], 1)

    def test_coco_stats_names(self) -> None:
        metrics = coco_stats_to_dict([value / 10 for value in range(12)])
        self.assertEqual(metrics["ap50"], 0.1)
        self.assertEqual(metrics["ar_large"], 1.1)

    def test_prediction_score_filter_is_non_mutating(self) -> None:
        predictions = [
            {
                "image_id": 1,
                "annotations": [
                    {"class_name": "person", "score": 0.8},
                    {"class_name": "person", "score": 0.3},
                ],
            }
        ]
        filtered, report = filter_predictions_by_detector_score(
            predictions, min_score=0.5
        )
        self.assertEqual(len(predictions[0]["annotations"]), 2)
        self.assertEqual(len(filtered[0]["annotations"]), 1)
        self.assertEqual(report["annotations_removed"], 1)
        self.assertEqual(report["retention_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
