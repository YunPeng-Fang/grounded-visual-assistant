from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.failure_analysis import (
    aggregate_failure_analysis,
    analyze_prediction,
    explicitly_mentioned_categories,
    has_non_terminal_ending,
    render_failure_report,
)


def sample_prediction() -> dict:
    return {
        "id": "image_1_listing",
        "image_id": 1,
        "image": "image.jpg",
        "target_categories": ["person", "sandwich"],
        "prompt_categories": ["person"],
        "target_evidence_boxes": [
            {"category": "person", "bbox_xywh": [0, 0, 1, 1]},
            {"category": "sandwich", "bbox_xywh": [1, 1, 1, 1]},
            {"category": "sandwich", "bbox_xywh": [2, 2, 1, 1]},
        ],
        "prompt_evaluation": {
            "precision": 1.0,
            "recall": 0.5,
            "f1": 2 / 3,
            "missed_categories": ["sandwich"],
            "hallucinated_categories": [],
        },
        "vlm_prediction": {
            "answer": "A person is serving sandwiches and"
        },
        "evaluation": {
            "gt_count": 3,
            "prediction_count": 2,
            "tp": 1,
            "fp": 1,
            "fn": 2,
            "f1": 0.4,
            "categories": {
                "person": {"gt": 1, "predicted": 2, "tp": 1, "fp": 1, "fn": 0},
                "sandwich": {"gt": 2, "predicted": 0, "tp": 0, "fp": 0, "fn": 2},
            },
        },
    }


class FailureAnalysisTest(unittest.TestCase):
    def test_irregular_plural_is_a_literal_parser_diagnostic(self) -> None:
        self.assertEqual(
            explicitly_mentioned_categories(
                "Several sandwiches are visible.", ["sandwich", "person"]
            ),
            ["sandwich"],
        )

    def test_non_terminal_ending_is_only_a_heuristic(self) -> None:
        self.assertTrue(has_non_terminal_ending("The image contains a person and"))
        self.assertFalse(has_non_terminal_ending("The image contains a person."))

    def test_prediction_attribution_separates_prompt_and_grounding(self) -> None:
        result = analyze_prediction(sample_prediction())
        self.assertEqual(result["prompt_missed_gt_boxes"], 2)
        self.assertEqual(result["grounding_missed_prompted_gt_boxes"], 0)
        self.assertEqual(result["prompted_target_false_positive_boxes"], 1)
        self.assertEqual(result["parser_recoverable_categories"], ["sandwich"])
        self.assertIn("parser_miss", result["flags"])
        self.assertIn("non_terminal_answer", result["flags"])

    def test_aggregate_and_markdown_report(self) -> None:
        summary, analyses = aggregate_failure_analysis([sample_prediction()])
        attribution = summary["grounding_attribution"]
        self.assertEqual(attribution["false_negative_boxes"], 2)
        self.assertEqual(
            attribution["false_negatives_from_missing_prompt_categories"], 2
        )
        self.assertEqual(attribution["prompt_stage_share_of_false_negatives"], 1.0)
        report = render_failure_report(
            summary,
            analyses,
            predictions_path="predictions.jsonl",
        )
        self.assertIn("VLM-Prompt Grounding Failure Analysis", report)
        self.assertIn("image_1_listing", report)
        self.assertIn("sandwich", report)


if __name__ == "__main__":
    unittest.main()
