from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.evaluation import (
    aggregate_metrics,
    extract_categories_from_vocabulary,
    extract_coco_categories,
    parse_relation,
    parse_yes_no,
    score_prediction,
)


class EvaluationTest(unittest.TestCase):
    def test_parse_yes_no(self) -> None:
        self.assertEqual(parse_yes_no("Yes, there is one."), "yes")
        self.assertEqual(parse_yes_no("No. I cannot see one."), "no")
        self.assertIsNone(parse_yes_no("Yes and no"))

    def test_parse_relation(self) -> None:
        self.assertEqual(parse_relation("It is on the left of the chair."), "to the left of")
        self.assertEqual(parse_relation("The laptop is beneath the bottle."), "below")

    def test_extract_categories_prefers_long_phrases(self) -> None:
        self.assertEqual(extract_coco_categories("hot dog"), ["hot dog"])
        self.assertEqual(
            extract_coco_categories("A person, sofa, and television."),
            ["couch", "person", "tv"],
        )

    def test_extract_categories_uses_restricted_open_images_vocabulary(self) -> None:
        vocabulary = ["human arm", "sports equipment", "helmet", "cat"]
        self.assertEqual(
            extract_categories_from_vocabulary(
                "Human arms, a helmet, and sports equipment.", vocabulary
            ),
            ["helmet", "human arm", "sports equipment"],
        )

    def test_object_listing_f1(self) -> None:
        sample = {
            "task_type": "object_listing",
            "gt_answer": "person, couch",
            "categories": ["person", "couch"],
        }
        result = score_prediction(sample, "A person, sofa, and television.")
        self.assertAlmostEqual(result["precision"], 2 / 3, places=5)
        self.assertEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 0.8, places=5)
        self.assertFalse(result["exact_match"])

    def test_object_listing_scores_restricted_vocabulary(self) -> None:
        sample = {
            "task_type": "object_listing",
            "gt_answer": "human arm, sports equipment",
            "categories": ["human arm", "sports equipment"],
            "metadata": {
                "allowed_categories": [
                    "human arm",
                    "sports equipment",
                    "alpaca",
                ]
            },
        }
        result = score_prediction(
            sample, "human arms, sports equipment, and alpaca"
        )
        self.assertEqual(
            result["predicted_categories"],
            ["alpaca", "human arm", "sports equipment"],
        )
        self.assertAlmostEqual(result["precision"], 2 / 3, places=5)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(
            result["parser_vocabulary"], "restricted_allowed_categories"
        )

    def test_aggregate_metrics(self) -> None:
        predictions = [
            {
                "source": "open_images",
                "split": "dev",
                "task_type": "object_existence",
                "latency_seconds": 2.0,
                "cuda_peak_memory_allocated_gb": 16.0,
                "cuda_memory_reserved_gb": 17.0,
                "evaluation": {
                    "score": 1.0,
                    "is_correct": True,
                    "parse_valid": True,
                },
            },
            {
                "source": "visual_genome",
                "split": "dev",
                "task_type": "object_existence",
                "latency_seconds": 4.0,
                "cuda_peak_memory_allocated_gb": 18.0,
                "cuda_memory_reserved_gb": 19.0,
                "evaluation": {
                    "score": 0.0,
                    "is_correct": False,
                    "parse_valid": True,
                },
            },
        ]
        metrics = aggregate_metrics(predictions, expected_samples=4)
        self.assertEqual(metrics["coverage"]["completed"], 2)
        self.assertEqual(metrics["coverage"]["completion_rate"], 0.5)
        self.assertEqual(metrics["tasks"]["object_existence"]["exact_accuracy"], 0.5)
        self.assertEqual(metrics["latency_seconds"]["mean"], 3.0)
        self.assertEqual(metrics["cuda_memory_gb"]["peak_allocated_max"], 18.0)
        self.assertEqual(metrics["sources"]["open_images"]["exact_accuracy"], 1.0)
        self.assertEqual(metrics["sources"]["visual_genome"]["exact_accuracy"], 0.0)
        self.assertEqual(metrics["split_counts"], {"dev": 2})

    def test_relation_metrics_report_balance_and_majority_baseline(self) -> None:
        predictions = []
        for target, parsed, correct in (
            ("above", "above", True),
            ("above", "below", False),
            ("above", None, False),
            ("below", "below", True),
        ):
            predictions.append(
                {
                    "source": "visual_genome",
                    "split": "dev",
                    "task_type": "spatial_relation",
                    "latency_seconds": 1.0,
                    "evaluation": {
                        "score": float(correct),
                        "is_correct": correct,
                        "parse_valid": parsed is not None,
                        "parsed_target": target,
                        "parsed_prediction": parsed,
                    },
                }
            )
        metrics = aggregate_metrics(predictions, expected_samples=4)
        relation = metrics["tasks"]["spatial_relation"]
        self.assertEqual(relation["exact_accuracy"], 0.5)
        self.assertEqual(relation["balanced_accuracy"], 0.666667)
        self.assertEqual(relation["majority_class_baseline_accuracy"], 0.75)
        self.assertEqual(relation["confusion"]["above"]["invalid"], 1)


if __name__ == "__main__":
    unittest.main()
