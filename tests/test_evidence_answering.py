from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.evaluation import score_prediction
from grounded_visual_assistant.evidence_answering import (
    EvidencePolicyConfig,
    aggregate_evidence_answering,
    answer_with_evidence,
    build_query_plan,
    normalize_evidence,
    parse_question_entities,
)


def annotation(
    category: str,
    box: list[float],
    *,
    score: float = 0.8,
    mask_area: int = 100,
) -> dict:
    return {
        "class_name": category,
        "bbox": box,
        "score": score,
        "mask_score": 0.9,
        "mask_area": mask_area,
    }


class EvidenceAnsweringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EvidencePolicyConfig(
            min_grounding_score=0.3,
            relation_margin=0.08,
        )

    def test_parse_question_entities(self) -> None:
        self.assertEqual(
            parse_question_entities(
                "Is there a dining table in this image? Answer yes or no.",
                "object_existence",
            ),
            ["dining table"],
        )
        self.assertEqual(
            parse_question_entities(
                "Where is the largest person relative to the largest sports ball?",
                "spatial_relation",
            ),
            ["person", "sports ball"],
        )

    def test_query_plan_uses_structured_categories_only_for_listing(self) -> None:
        listing = {
            "task_type": "object_listing",
            "question": "List the objects.",
        }
        plan = build_query_plan(listing, ["person", "bottle", "person"])
        self.assertEqual(plan["categories"], ["bottle", "person"])
        self.assertEqual(plan["prompt"], "bottle. person.")
        self.assertEqual(plan["source"], "structured_vlm_coco80")

    def test_single_query_fallback_is_explicit(self) -> None:
        accepted, rejected = normalize_evidence(
            [annotation("object", [0, 0, 10, 10])],
            ["bottle"],
            image_width=100,
            image_height=100,
            config=self.config,
        )
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["category"], "bottle")
        self.assertEqual(accepted[0]["label_mapping"], "single_query_fallback")

    def test_listing_removes_categories_without_evidence(self) -> None:
        sample = {"task_type": "object_listing"}
        plan = build_query_plan(sample, ["bottle", "cup"])
        result = answer_with_evidence(
            sample,
            plan,
            [annotation("bottle", [10, 10, 30, 40])],
            image_width=100,
            image_height=100,
            config=self.config,
        )
        self.assertEqual(result["forced_answer"], "bottle")
        self.assertEqual(result["selective_answer"], "bottle")
        self.assertTrue(result["claim_supported"])

    def test_negative_existence_is_forced_but_selectively_abstained(self) -> None:
        sample = {
            "task_type": "object_existence",
            "question": "Is there a bottle in this image? Answer yes or no.",
        }
        plan = build_query_plan(sample)
        result = answer_with_evidence(
            sample,
            plan,
            [],
            image_width=100,
            image_height=100,
            config=self.config,
        )
        self.assertEqual(result["forced_answer"], "no")
        self.assertIsNone(result["selective_answer"])
        self.assertTrue(result["abstained"])
        self.assertEqual(result["unsupported_claim_count"], 1)

    def test_spatial_relation_uses_largest_instances_and_dominant_axis(self) -> None:
        sample = {
            "task_type": "spatial_relation",
            "question": (
                "Where is the largest person relative to the largest dog?"
            ),
        }
        plan = build_query_plan(sample)
        result = answer_with_evidence(
            sample,
            plan,
            [
                annotation("person", [10, 20, 20, 40], mask_area=100),
                annotation("person", [5, 20, 25, 50], mask_area=500),
                annotation("dog", [70, 20, 90, 50], mask_area=400),
            ],
            image_width=100,
            image_height=100,
            config=self.config,
        )
        self.assertEqual(result["forced_answer"], "to the left of")
        self.assertFalse(result["abstained"])
        self.assertEqual(
            result["selected_evidence"][0]["annotation_index"], 1
        )

    def test_spatial_relation_abstains_on_ambiguous_geometry(self) -> None:
        sample = {
            "task_type": "spatial_relation",
            "question": "Where is the largest cup relative to the largest bowl?",
        }
        result = answer_with_evidence(
            sample,
            build_query_plan(sample),
            [
                annotation("cup", [45, 45, 55, 55]),
                annotation("bowl", [48, 48, 58, 58]),
            ],
            image_width=100,
            image_height=100,
            config=self.config,
        )
        self.assertEqual(result["status"], "ambiguous_geometry")
        self.assertTrue(result["abstained"])

    def test_mixed_task_aggregation(self) -> None:
        specs = [
            (
                {
                    "task_type": "object_listing",
                    "gt_answer": "bottle",
                    "categories": ["bottle"],
                },
                "bottle",
                False,
                1,
                0,
            ),
            (
                {"task_type": "object_existence", "gt_answer": "no"},
                "no",
                True,
                1,
                1,
            ),
            (
                {"task_type": "spatial_relation", "gt_answer": "to the left of"},
                "to the left of",
                False,
                1,
                0,
            ),
        ]
        records = []
        for index, (sample, answer, abstained, claims, unsupported) in enumerate(specs):
            records.append(
                {
                    "id": str(index),
                    "image_id": index,
                    "task_type": sample["task_type"],
                    "evaluation": score_prediction(sample, answer),
                    "answer_policy": {
                        "abstained": abstained,
                        "claim_count": claims,
                        "unsupported_claim_count": unsupported,
                    },
                    "evidence_evaluation": {
                        "gt_count": 1,
                        "prediction_count": 1,
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                        "matches": [{"iou": 0.75}],
                    },
                    "pipeline_latency_seconds": {
                        "total": 1.0,
                        "planning": 0.2,
                        "grounding": 0.6,
                        "sam2": 0.2,
                    },
                }
            )
        metrics = aggregate_evidence_answering(records, expected_samples=3)
        self.assertEqual(metrics["closed_set_answers"]["overall"]["exact_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["selective_answers"]["coverage"], 2 / 3, places=5)
        self.assertEqual(metrics["selective_answers"]["exact_accuracy"], 1.0)
        self.assertAlmostEqual(
            metrics["evidence_support"]["forced_unsupported_claim_rate"],
            1 / 3,
            places=5,
        )
        self.assertEqual(
            metrics["question_conditioned_evidence_iou50"]["overall"]["micro_f1"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
