from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.evaluation import score_prediction
from grounded_visual_assistant.hard_vlm_analysis import (
    analyze_hard_vlm_predictions,
    render_hard_vlm_report,
)


class HardVlmAnalysisTest(unittest.TestCase):
    def test_analysis_attributes_failures_and_relation_bias(self) -> None:
        samples = [
            {
                "id": "existence",
                "sample_id": "open_images:one",
                "source": "open_images_v7_validation",
                "split": "dev",
                "task_type": "object_existence",
                "question": "Is there a cat?",
                "gt_answer": "no",
                "metadata": {"is_positive": False},
            },
            {
                "id": "relation",
                "sample_id": "visual_genome:two",
                "source": "visual_genome_v1_4",
                "split": "dev",
                "task_type": "spatial_relation",
                "question": "Where is the person relative to the chair?",
                "gt_answer": "above",
            },
            {
                "id": "listing",
                "sample_id": "open_images:three",
                "source": "open_images_v7_validation",
                "split": "dev",
                "task_type": "object_listing",
                "question": "List the categories.",
                "gt_answer": "human arm",
                "categories": ["human arm"],
                "metadata": {
                    "allowed_categories": ["human arm", "alpaca"],
                    "has_negative_distractor": True,
                },
            },
        ]
        answers = {
            "existence": "Yes.",
            "relation": "It is not possible to determine this relation.",
            "listing": "human arm",
        }
        predictions = []
        for sample in samples:
            answer = answers[sample["id"]]
            predictions.append(
                {
                    **{key: sample[key] for key in ("id", "source", "split", "task_type")},
                    "prediction": answer,
                    "evaluation": score_prediction(sample, answer),
                    "latency_seconds": 1.0,
                    "hit_max_new_tokens": sample["id"] == "relation",
                    "metadata": sample.get("metadata", {}),
                }
            )

        summary, analyses = analyze_hard_vlm_predictions(samples, predictions)
        by_id = {item["id"]: item for item in analyses}
        self.assertIn("existence_false_positive", by_id["existence"]["flags"])
        self.assertIn("relation_parse_invalid", by_id["relation"]["flags"])
        self.assertIn(
            "relation_refusal_or_absence_claim", by_id["relation"]["flags"]
        )
        relation = summary["relation_sources"]["visual_genome_v1_4"]
        self.assertEqual(relation["majority_class_baseline_accuracy"], 1.0)
        self.assertTrue(summary["warnings"])
        report = render_hard_vlm_report(summary, predictions_path="predictions.jsonl")
        self.assertIn("Relation By Source", report)

    def test_analysis_requires_an_explicit_test_split(self) -> None:
        sample = {
            "id": "one",
            "source": "open_images_v7_validation",
            "split": "test",
            "task_type": "object_existence",
            "question": "Is there a cat?",
            "gt_answer": "yes",
            "metadata": {"is_positive": True},
        }
        prediction = {
            "id": "one",
            "source": sample["source"],
            "split": "test",
            "task_type": sample["task_type"],
            "prediction": "yes",
            "evaluation": score_prediction(sample, "yes"),
            "latency_seconds": 1.0,
            "metadata": sample["metadata"],
        }
        with self.assertRaisesRegex(RuntimeError, "requires 'dev'"):
            analyze_hard_vlm_predictions([sample], [prediction])
        summary, analyses = analyze_hard_vlm_predictions(
            [sample], [prediction], required_split="test"
        )
        self.assertEqual(summary["coverage"]["split"], "test")
        self.assertEqual(len(analyses), 1)


if __name__ == "__main__":
    unittest.main()
