from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.evidence_answering import EvidencePolicyConfig
from grounded_visual_assistant.policy_calibration import (
    aggregate_policy_records,
    apply_locked_policy,
    fuse_existence_consensus,
    replay_grounded_policy,
    structured_listing_policy,
    validate_locked_policy,
)


class PolicyCalibrationTest(unittest.TestCase):
    def locked_policy(self) -> dict:
        return {
            "protocol": "task_aware_evidence_fusion_v1",
            "development_split_only": True,
            "tasks": {
                "object_listing": {
                    "candidate_id": "listing",
                    "mode": "grounded_evidence_gate",
                    "config": {
                        "min_grounding_score": 0.4,
                        "min_mask_score": None,
                        "min_mask_area_ratio": 0.005,
                        "relation_margin": 0.08,
                    },
                },
                "object_existence": {
                    "candidate_id": "existence",
                    "mode": "vlm_grounding_consensus",
                    "config": {
                        "min_grounding_score": 0.3,
                        "min_mask_score": None,
                        "min_mask_area_ratio": 0.0,
                        "relation_margin": 0.08,
                    },
                },
                "spatial_relation": {
                    "candidate_id": "spatial",
                    "mode": "grounded_geometry",
                    "config": {
                        "min_grounding_score": 0.45,
                        "min_mask_score": None,
                        "min_mask_area_ratio": 0.0,
                        "relation_margin": 0.08,
                    },
                },
            },
        }

    def existence_record(self, gt_answer: str = "no") -> dict:
        return {
            "id": "existence",
            "image": "image.jpg",
            "image_id": 1,
            "question": "Is there a bottle in this image? Answer yes or no.",
            "task_type": "object_existence",
            "gt_answer": gt_answer,
            "target_categories": ["bottle"],
            "query_plan": {
                "source": "question_parser",
                "categories": ["bottle"],
                "prompt": "bottle.",
            },
            "annotations": [],
        }

    def test_replay_applies_a_stricter_saved_evidence_gate(self) -> None:
        source = self.existence_record(gt_answer="yes")
        source["annotations"] = [
            {
                "class_name": "bottle",
                "bbox": [0, 0, 20, 20],
                "score": 0.35,
                "mask_score": 0.95,
                "mask_area": 400,
            }
        ]
        replay = replay_grounded_policy(
            source,
            EvidencePolicyConfig(min_grounding_score=0.4),
            image_width=100,
            image_height=100,
        )
        self.assertEqual(replay["answer_policy"]["forced_answer"], "no")
        self.assertTrue(replay["answer_policy"]["abstained"])
        self.assertFalse(replay["evaluation"]["is_correct"])

    def test_existence_consensus_abstains_on_disagreement(self) -> None:
        source = self.existence_record(gt_answer="no")
        source["annotations"] = [
            {
                "class_name": "bottle",
                "bbox": [0, 0, 20, 20],
                "score": 0.8,
                "mask_score": 0.95,
                "mask_area": 400,
            }
        ]
        detector = replay_grounded_policy(
            source,
            EvidencePolicyConfig(min_grounding_score=0.3),
            image_width=100,
            image_height=100,
        )
        fused = fuse_existence_consensus(detector, {"prediction": "no"})
        self.assertEqual(fused["answer_policy"]["forced_answer"], "no")
        self.assertTrue(fused["answer_policy"]["abstained"])
        self.assertEqual(
            fused["answer_policy"]["status"], "vlm_grounding_disagreement"
        )
        self.assertTrue(fused["evaluation"]["is_correct"])

    def test_negative_consensus_is_marked_as_cross_model_support(self) -> None:
        detector = replay_grounded_policy(
            self.existence_record(gt_answer="no"),
            EvidencePolicyConfig(min_grounding_score=0.3),
            image_width=100,
            image_height=100,
        )
        fused = fuse_existence_consensus(detector, {"prediction": "No."})
        self.assertFalse(fused["answer_policy"]["abstained"])
        self.assertEqual(
            fused["answer_policy"]["support_type"], "cross_model_agreement"
        )
        self.assertTrue(fused["selective_evaluation"]["is_correct"])

    def test_structured_listing_and_aggregate(self) -> None:
        source = {
            "id": "listing",
            "image": "image.jpg",
            "image_id": 1,
            "question": "List objects.",
            "task_type": "object_listing",
            "gt_answer": "bottle, cup",
            "target_categories": ["bottle", "cup"],
            "categories": ["bottle", "cup"],
            "query_plan": {
                "categories": ["cup", "bottle"],
                "prompt": "bottle. cup.",
            },
        }
        record = structured_listing_policy(source)
        metrics = aggregate_policy_records([record])
        self.assertEqual(record["answer_policy"]["forced_answer"], "bottle, cup")
        self.assertEqual(metrics["overall"]["forced_macro_f1"], 1.0)
        self.assertEqual(metrics["overall"]["selective_coverage"], 1.0)

    def test_locked_policy_validation_rejects_non_dev_protocol(self) -> None:
        payload = self.locked_policy()
        payload["development_split_only"] = False
        with self.assertRaises(ValueError):
            validate_locked_policy(payload)

    def test_apply_locked_existence_policy_uses_consensus(self) -> None:
        source = self.existence_record(gt_answer="yes")
        source["annotations"] = [
            {
                "class_name": "bottle",
                "bbox": [0, 0, 20, 20],
                "score": 0.8,
                "mask_score": 0.95,
                "mask_area": 400,
            }
        ]
        applied = apply_locked_policy(
            source,
            self.locked_policy(),
            image_width=100,
            image_height=100,
            vlm_record={"prediction": "yes"},
        )
        self.assertEqual(applied["answer_policy"]["selective_answer"], "yes")
        self.assertEqual(
            applied["policy_config"]["candidate_id"], "existence"
        )
        self.assertTrue(applied["evaluation"]["is_correct"])

    def test_apply_locked_existence_policy_requires_vlm_record(self) -> None:
        with self.assertRaises(ValueError):
            apply_locked_policy(
                self.existence_record(),
                self.locked_policy(),
                image_width=100,
                image_height=100,
            )


if __name__ == "__main__":
    unittest.main()
