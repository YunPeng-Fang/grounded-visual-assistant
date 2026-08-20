from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.relation_prompt_comparison import (
    compare_relation_prompts,
    exact_mcnemar_p_value,
)


def prediction(question_id: str, source: str, correct: bool) -> dict:
    target = "above"
    parsed = "above" if correct else "below"
    return {
        "id": question_id,
        "source": source,
        "split": "dev",
        "task_type": "spatial_relation",
        "gt_answer": target,
        "latency_seconds": 1.0,
        "generated_tokens": 3,
        "evaluation": {
            "score": float(correct),
            "is_correct": correct,
            "parse_valid": True,
            "parsed_target": target,
            "parsed_prediction": parsed,
        },
    }


class RelationPromptComparisonTest(unittest.TestCase):
    def test_exact_mcnemar_and_paired_transitions(self) -> None:
        self.assertEqual(exact_mcnemar_p_value(0, 4), 0.125)
        baseline = [
            prediction("one", "open_images_v7_validation", False),
            prediction("two", "open_images_v7_validation", True),
        ]
        candidate = [
            prediction("one", "open_images_v7_validation", True),
            prediction("two", "open_images_v7_validation", True),
        ]
        summary = compare_relation_prompts(baseline, candidate)
        paired = summary["overall"]["paired"]
        self.assertEqual(paired["candidate_only_correct"], 1)
        self.assertEqual(paired["baseline_only_correct"], 0)
        self.assertEqual(summary["overall"]["delta"]["exact_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
