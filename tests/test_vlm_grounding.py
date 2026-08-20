from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.vlm_grounding import (
    aggregate_pipeline_latency,
    aggregate_prompt_quality,
    build_vlm_prompt_samples,
    categories_to_grounding_prompt,
    evaluate_prompt_categories,
)


class VlmGroundingTest(unittest.TestCase):
    def test_categories_to_grounding_prompt(self) -> None:
        self.assertEqual(
            categories_to_grounding_prompt(["person", "dog", "person"]),
            "dog. person.",
        )
        self.assertEqual(categories_to_grounding_prompt([]), "")

    def test_prompt_category_score_includes_hallucinations_and_misses(self) -> None:
        result = evaluate_prompt_categories(
            ["person", "cat"],
            ["person", "dog"],
        )
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["hallucinated_categories"], ["cat"])
        self.assertEqual(result["missed_categories"], ["dog"])

    def test_build_vlm_prompt_samples_uses_only_generated_answer(self) -> None:
        oracle = [
            {
                "id": "image_1_listing",
                "image": "image.jpg",
                "image_id": 1,
                "categories": ["dog", "person"],
                "evidence_boxes": [{"category": "dog", "bbox_xywh": [0, 0, 1, 1]}],
                "prompt": "dog. person.",
            }
        ]
        vlm = [
            {
                "id": "image_1_listing",
                "image_id": 1,
                "task_type": "object_listing",
                "prediction": "A person and a cat are visible.",
                # This GT-bearing field must not be used to build the prompt.
                "categories": ["dog", "person"],
                "model": "vlm",
                "latency_seconds": 2.5,
                "evaluation": {"predicted_categories": ["cat", "person"]},
            }
        ]
        sample = build_vlm_prompt_samples(oracle, vlm)[0]
        self.assertEqual(sample["prompt_categories"], ["cat", "person"])
        self.assertEqual(sample["prompt"], "cat. person.")
        self.assertEqual(sample["categories"], ["dog", "person"])
        self.assertEqual(sample["prompt_evaluation"]["f1"], 0.5)
        self.assertTrue(sample["vlm_prediction"]["parser_matches_saved"])

    def test_empty_vlm_answer_remains_an_evaluated_sample(self) -> None:
        oracle = [
            {
                "id": "image_1_listing",
                "image": "image.jpg",
                "image_id": 1,
                "categories": ["bowl"],
                "evidence_boxes": [{"category": "bowl", "bbox_xywh": [0, 0, 1, 1]}],
                "prompt": "bowl.",
            }
        ]
        vlm = [
            {
                "id": "image_1_listing",
                "image_id": 1,
                "task_type": "object_listing",
                "prediction": "Cookware and food are visible.",
            }
        ]
        sample = build_vlm_prompt_samples(oracle, vlm)[0]
        self.assertEqual(sample["prompt"], "")
        self.assertTrue(sample["prompt_evaluation"]["empty_prompt"])
        self.assertEqual(sample["prompt_evaluation"]["fn"], 1)

    def test_structured_categories_are_not_reparsed_as_free_text(self) -> None:
        oracle = [
            {
                "id": "image_1_listing",
                "image": "image.jpg",
                "image_id": 1,
                "categories": ["person"],
                "evidence_boxes": [],
                "prompt": "person.",
            }
        ]
        vlm = [
            {
                "id": "image_1_listing",
                "image_id": 1,
                "task_type": "object_listing",
                "prediction": '["person"] but a cat may also be visible',
                "structured_output": {
                    "parser": "structured_coco_json_v1",
                    "parsed_categories": ["person"],
                },
                "evaluation": {"predicted_categories": ["person"]},
            }
        ]
        sample = build_vlm_prompt_samples(oracle, vlm)[0]
        self.assertEqual(sample["prompt_categories"], ["person"])
        self.assertEqual(
            sample["vlm_prediction"]["parser"], "structured_coco_json_v1"
        )

    def test_missing_vlm_prediction_is_rejected(self) -> None:
        oracle = [{"id": "missing", "image_id": 1, "categories": ["person"]}]
        with self.assertRaisesRegex(ValueError, "Missing VLM"):
            build_vlm_prompt_samples(oracle, [])

    def test_aggregate_prompt_and_pipeline_metrics(self) -> None:
        records = [
            {
                "prompt_evaluation": evaluate_prompt_categories(
                    ["person", "cat"], ["person", "dog"]
                ),
                "pipeline_latency_seconds": {
                    "vlm": 2.0,
                    "grounding": 0.2,
                    "sam2": 0.1,
                    "total": 2.3,
                },
            },
            {
                "prompt_evaluation": evaluate_prompt_categories([], ["bowl"]),
                "pipeline_latency_seconds": {
                    "vlm": 1.0,
                    "grounding": 0.0,
                    "sam2": 0.0,
                    "total": 1.0,
                },
            },
        ]
        quality = aggregate_prompt_quality(records, expected_images=2)
        self.assertEqual(quality["counts"]["target_categories"], 3)
        self.assertEqual(quality["counts"]["predicted_categories"], 2)
        self.assertEqual(quality["counts"]["empty_prompt_images"], 1)
        self.assertAlmostEqual(quality["micro_precision"], 0.5)
        self.assertAlmostEqual(quality["micro_recall"], 1 / 3, places=6)

        latency = aggregate_pipeline_latency(records)
        self.assertEqual(latency["mean"], 1.65)
        self.assertEqual(latency["vlm_mean"], 1.5)
        self.assertEqual(latency["grounding_mean"], 0.1)


if __name__ == "__main__":
    unittest.main()
