from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_dataset import (
    OPEN_IMAGES_SOURCE,
    VISUAL_GENOME_SOURCE,
)
from grounded_visual_assistant.relation_prompt_selection import (
    build_relation_prompt_selection,
)


def prediction(
    question_id: str,
    source: str,
    correct: bool,
    *,
    hit_max: bool = False,
) -> dict:
    return {
        "id": question_id,
        "source": source,
        "split": "dev",
        "task_type": "spatial_relation",
        "gt_answer": "above",
        "latency_seconds": 0.2,
        "generated_tokens": 2,
        "hit_max_new_tokens": hit_max,
        "evaluation": {
            "score": float(correct),
            "is_correct": correct,
            "parse_valid": True,
            "parsed_target": "above",
            "parsed_prediction": "above" if correct else "below",
        },
    }


class RelationPromptSelectionTest(unittest.TestCase):
    def test_locks_source_aware_policy_after_frozen_gates_pass(self) -> None:
        baseline = []
        prompt_v2 = []
        prompt_v3 = []
        for index in range(100):
            baseline.append(
                prediction(f"oi-{index}", OPEN_IMAGES_SOURCE, False)
            )
            prompt_v2.append(
                prediction(f"oi-{index}", OPEN_IMAGES_SOURCE, True)
            )
            baseline.append(
                prediction(
                    f"vg-{index}", VISUAL_GENOME_SOURCE, index < 58
                )
            )
            prompt_v2.append(
                prediction(
                    f"vg-{index}", VISUAL_GENOME_SOURCE, index < 53
                )
            )
            prompt_v3.append(
                prediction(
                    f"vg-{index}", VISUAL_GENOME_SOURCE, index < 62
                )
            )
        manifest = {
            "split": "dev",
            "source": VISUAL_GENOME_SOURCE,
            "questions": 100,
            "acceptance_criteria": {
                "parse_valid_rate_min": 0.98,
                "hit_max_new_tokens_max": 0,
                "balanced_accuracy_min": 0.50,
                "exact_accuracy_min": 0.58,
            },
        }

        summary, policy, transitions = build_relation_prompt_selection(
            baseline, prompt_v2, prompt_v3, manifest
        )

        self.assertEqual(summary["status"], "accepted")
        self.assertTrue(summary["acceptance"]["open_images_v2"]["passed"])
        self.assertTrue(summary["acceptance"]["visual_genome_v3"]["passed"])
        self.assertEqual(policy["status"], "locked")
        self.assertEqual(
            policy["sources"][OPEN_IMAGES_SOURCE]["selected_variant"], "v2"
        )
        self.assertEqual(
            policy["sources"][VISUAL_GENOME_SOURCE]["selected_variant"], "v3"
        )
        self.assertEqual(len(transitions), 100)


if __name__ == "__main__":
    unittest.main()
