from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.live_pipeline_evaluation import (
    aggregate_live_pipeline,
    canonicalize_targets,
    evaluate_live_prediction,
    evaluate_mask_evidence,
    visible_evidence_categories,
)


def positive_sample() -> dict:
    return {
        "id": "positive",
        "image_id": 1,
        "task_type": "object_listing",
        "question": "List the main visible object categories.",
        "gt_answer": "bottle",
        "categories": ["bottle"],
        "source": "test",
        "split": "dev",
        "evidence_boxes": [
            {
                "annotation_id": 1,
                "category": "bottle",
                "bbox_xywh": [10, 10, 20, 20],
            }
        ],
    }


class LivePipelineEvaluationTest(unittest.TestCase):
    def test_visible_categories_and_target_canonicalization(self) -> None:
        sample = positive_sample()
        self.assertEqual(visible_evidence_categories(sample), ["bottle"])
        self.assertEqual(
            canonicalize_targets(["Person", "mobile phone", "person"]),
            ["cell phone", "person"],
        )

    def test_positive_answer_target_and_box_are_jointly_scored(self) -> None:
        scored = evaluate_live_prediction(
            positive_sample(),
            answer="bottle",
            targets=["bottle"],
            annotations=[
                {
                    "class_name": "bottle",
                    "bbox": [10, 10, 30, 30],
                    "score": 0.9,
                }
            ],
        )
        self.assertTrue(scored["evaluation"]["is_correct"])
        self.assertEqual(scored["target_evaluation"]["f1"], 1.0)
        self.assertEqual(scored["evidence_evaluation"]["f1"], 1.0)
        self.assertTrue(scored["end_to_end_success"])
        self.assertTrue(scored["end_to_end_complete_success"])

    def test_negative_question_rewards_empty_evidence(self) -> None:
        sample = {
            "id": "negative",
            "image_id": 2,
            "task_type": "object_existence",
            "question": "Is there a cat? Answer yes or no.",
            "gt_answer": "no",
            "categories": ["cat"],
            "evidence_boxes": [],
        }
        scored = evaluate_live_prediction(
            sample,
            answer="No.",
            targets=[],
            annotations=[],
        )
        self.assertFalse(scored["evidence_required"])
        self.assertTrue(scored["evidence_supported"])
        self.assertTrue(scored["end_to_end_success"])
        mask = evaluate_mask_evidence(
            [],
            [],
            coco_annotations_by_id={},
            image_height=10,
            image_width=10,
        )
        self.assertEqual(mask["gt_count"], 0)
        self.assertEqual(mask["prediction_count"], 0)

    def test_aggregate_reports_all_pipeline_layers(self) -> None:
        sample = positive_sample()
        scored = evaluate_live_prediction(
            sample,
            answer="bottle",
            targets=["bottle"],
            annotations=[
                {
                    "class_name": "bottle",
                    "bbox": [10, 10, 30, 30],
                    "score": 0.9,
                }
            ],
        )
        record = {
            **sample,
            "prediction": "bottle",
            "targets": ["bottle"],
            **scored,
            "vlm_output": {
                "schema_valid": True,
                "parse_source": "direct_json",
            },
            "latency_seconds": 1.5,
            "pipeline_latency_seconds": {
                "vlm": 1.0,
                "grounding": 0.3,
                "sam2": 0.1,
                "end_to_end": 1.5,
            },
            "grounding": {
                "latency_seconds": {
                    "grounding": 0.3,
                    "sam2": 0.1,
                    "total": 0.4,
                },
                "postprocessing": {
                    "candidate_count": 1,
                    "kept_count": 1,
                    "suppressed_count": 0,
                },
            },
        }
        metrics = aggregate_live_pipeline(
            [record],
            expected_samples=2,
            expected_required_evidence=2,
            expected_negative_evidence=1,
            status="running",
        )
        self.assertEqual(metrics["overall"]["exact_accuracy"], 1.0)
        self.assertEqual(
            metrics["structured_targets"]["schema_valid_rate"], 1.0
        )
        self.assertEqual(
            metrics["required_evidence_box_metrics"]["box_iou_50"][
                "micro_f1"
            ],
            1.0,
        )
        self.assertEqual(
            metrics["required_evidence_box_metrics"]["coverage"]["remaining"],
            1,
        )
        self.assertEqual(
            metrics["negative_evidence_behavior"]["remaining_questions"],
            1,
        )
        self.assertEqual(
            metrics["end_to_end"]["overall"][
                "answer_and_any_evidence_success_rate"
            ],
            1.0,
        )
        self.assertEqual(metrics["stage_latency_seconds"]["vlm_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
