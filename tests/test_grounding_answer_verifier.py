from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.grounding_answer_verifier import (
    GROUNDING_ANSWER_VERIFIER_PROTOCOL,
    GroundingAnswerVerifierConfig,
    compact_grounding_result,
    verify_binary_answer,
)


def annotation(score: float, *, mask_area: int = 100) -> dict:
    return {
        "class_name": "dog",
        "bbox": [10, 20, 50, 80],
        "score": score,
        "mask_score": 0.95,
        "mask_area": mask_area,
        "segmentation": {"size": [100, 100], "counts": "rle"},
    }


class GroundingAnswerVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GroundingAnswerVerifierConfig(
            evidence_score_threshold=0.3,
            promotion_score_threshold=0.45,
        )

    def verify(self, answer: str, annotations: list[dict]) -> dict:
        return verify_binary_answer(
            answer,
            target="dog",
            annotations=annotations,
            image_width=100,
            image_height=100,
            config=self.config,
        )

    def test_promotes_negative_with_high_confidence_localized_evidence(
        self,
    ) -> None:
        result = self.verify("No", [annotation(0.6)])

        self.assertEqual(
            result["protocol"], GROUNDING_ANSWER_VERIFIER_PROTOCOL
        )
        self.assertEqual(result["final_answer"], "yes")
        self.assertTrue(result["changed"])
        self.assertEqual(result["correction_direction"], "no_to_yes")
        self.assertEqual(
            result["status"], "promoted_by_localized_evidence"
        )

    def test_preserves_negative_below_promotion_threshold(self) -> None:
        result = self.verify("No", [annotation(0.4)])

        self.assertEqual(result["final_answer"], "no")
        self.assertFalse(result["changed"])
        self.assertEqual(result["evidence_level"], "accepted")

    def test_never_demotes_positive_when_detector_is_silent(self) -> None:
        result = self.verify("Yes", [])

        self.assertEqual(result["final_answer"], "yes")
        self.assertFalse(result["changed"])
        self.assertEqual(
            result["status"], "positive_preserved_without_evidence"
        )

    def test_applies_mask_and_area_gates(self) -> None:
        config = GroundingAnswerVerifierConfig(
            evidence_score_threshold=0.3,
            promotion_score_threshold=0.45,
            min_mask_score=0.9,
            min_mask_area_ratio=0.02,
        )
        result = verify_binary_answer(
            "No",
            target="dog",
            annotations=[annotation(0.8, mask_area=100)],
            image_width=100,
            image_height=100,
            config=config,
        )

        self.assertEqual(result["final_answer"], "no")
        self.assertEqual(result["accepted_evidence_count"], 0)
        self.assertIn(
            "small_mask",
            result["rejected_evidence"][0]["rejection_reasons"],
        )

    def test_requires_unambiguous_binary_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            self.verify("Maybe", [annotation(0.8)])

    def test_compacts_segmentation_payloads(self) -> None:
        result = compact_grounding_result(
            {
                "annotations": [annotation(0.8)],
                "img_width": 100,
                "img_height": 100,
            }
        )

        self.assertNotIn("segmentation", result["annotations"][0])
        self.assertEqual(result["img_width"], 100)


if __name__ == "__main__":
    unittest.main()
