from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.demo import GROUNDED_DEMO_SYSTEM_PROMPT
from grounded_visual_assistant.live_pipeline_prompting import (
    GENERIC_PROMPT_POLICY,
    TASK_AWARE_COCO_POLICY,
    TASK_AWARE_COCO_V2_POLICY,
    build_live_pipeline_system_prompt,
    evidence_target_limit,
)


class LivePipelinePromptingTest(unittest.TestCase):
    def test_generic_policy_preserves_demo_prompt_and_limit(self) -> None:
        sample = {
            "task_type": "object_listing",
            "question": "List the main visible object categories in this image.",
        }
        self.assertEqual(
            build_live_pipeline_system_prompt(sample, GENERIC_PROMPT_POLICY),
            GROUNDED_DEMO_SYSTEM_PROMPT,
        )
        self.assertEqual(evidence_target_limit(GENERIC_PROMPT_POLICY), 6)

    def test_listing_prompt_is_coco_constrained_and_ignores_ground_truth(
        self,
    ) -> None:
        sample = {
            "task_type": "object_listing",
            "question": "List the main visible object categories in this image.",
            "gt_answer": "secret answer",
            "categories": ["secret category"],
            "evidence_boxes": [{"category": "secret evidence"}],
        }
        changed_ground_truth = {
            **sample,
            "gt_answer": "different",
            "categories": ["different"],
            "evidence_boxes": [],
        }
        first = build_live_pipeline_system_prompt(
            sample, TASK_AWARE_COCO_POLICY
        )
        second = build_live_pipeline_system_prompt(
            changed_ground_truth, TASK_AWARE_COCO_POLICY
        )
        self.assertEqual(first, second)
        self.assertIn("person, bicycle, car", first)
        self.assertIn("hair drier, toothbrush", first)
        self.assertNotIn("secret", first)
        self.assertEqual(evidence_target_limit(TASK_AWARE_COCO_POLICY), 80)

    def test_existence_and_relation_prompts_use_question_entities(self) -> None:
        existence_sample = {
            "task_type": "object_existence",
            "question": (
                "Is there a mobile phone in this image? Answer yes or no."
            ),
        }
        existence = build_live_pipeline_system_prompt(
            existence_sample,
            TASK_AWARE_COCO_POLICY,
        )
        self.assertIn('["cell phone"]', existence)
        self.assertIn('exactly "yes" or "no"', existence)
        self.assertEqual(
            existence,
            build_live_pipeline_system_prompt(
                existence_sample, TASK_AWARE_COCO_V2_POLICY
            ),
        )

        relation_sample = {
            "task_type": "spatial_relation",
            "question": (
                "Where is the largest umbrella relative to the largest "
                "person?"
            ),
        }
        relation = build_live_pipeline_system_prompt(
            relation_sample,
            TASK_AWARE_COCO_POLICY,
        )
        self.assertIn('["umbrella", "person"]', relation)
        self.assertIn('"to the left of"', relation)
        self.assertIn("bounding-box centers", relation)
        self.assertEqual(
            relation,
            build_live_pipeline_system_prompt(
                relation_sample, TASK_AWARE_COCO_V2_POLICY
            ),
        )

    def test_v2_listing_is_compact_and_does_not_use_ground_truth(self) -> None:
        sample = {
            "task_type": "object_listing",
            "question": "List the main visible object categories in this image.",
            "gt_answer": "secret",
            "categories": ["secret"],
        }
        prompt = build_live_pipeline_system_prompt(
            sample, TASK_AWARE_COCO_V2_POLICY
        )
        without_ground_truth = build_live_pipeline_system_prompt(
            {
                "task_type": sample["task_type"],
                "question": sample["question"],
            },
            TASK_AWARE_COCO_V2_POLICY,
        )
        self.assertEqual(prompt, without_ground_truth)
        self.assertIn("at most eight unique categories", prompt)
        self.assertIn("Never copy, continue, or enumerate", prompt)
        self.assertIn("If uncertain about a category, omit it", prompt)
        self.assertNotIn("secret", prompt)
        self.assertEqual(evidence_target_limit(TASK_AWARE_COCO_V2_POLICY), 8)


if __name__ == "__main__":
    unittest.main()
