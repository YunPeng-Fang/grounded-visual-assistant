from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.semantic_answer_verifier import (
    SEMANTIC_ANSWER_VERIFIER_PROTOCOL,
    SemanticAnswerVerifierConfig,
    context_crop_box,
    select_semantic_candidates,
    verify_binary_answer_v2,
    write_semantic_crop,
)


def annotation(
    index: int,
    *,
    score: float = 0.6,
    mask_score: float = 0.95,
    mask_area: int = 1000,
) -> dict:
    return {
        "class_name": "bus",
        "bbox": [10 + index, 20, 50 + index, 80],
        "score": score,
        "mask_score": mask_score,
        "mask_area": mask_area,
    }


def review(index: int, answer: str) -> dict:
    return {
        "candidate_key": f"query__annotation-{index:03d}",
        "annotation_index": index,
        "answer": answer,
    }


class SemanticAnswerVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SemanticAnswerVerifierConfig(
            evidence_score_threshold=0.3,
            min_mask_score=0.5,
            max_mask_area_ratio=0.9,
            max_candidates_per_query=2,
            min_crop_size=32,
        )

    def verify(self, answer: str, reviews: list[dict]) -> dict:
        return verify_binary_answer_v2(
            answer,
            target="bus",
            annotations=[annotation(0)],
            image_width=100,
            image_height=100,
            semantic_reviews=reviews,
            config=self.config,
        )

    def test_promotes_only_after_exact_semantic_confirmation(self) -> None:
        result = self.verify("No", [review(0, "Yes")])

        self.assertEqual(
            result["protocol"], SEMANTIC_ANSWER_VERIFIER_PROTOCOL
        )
        self.assertEqual(result["final_answer"], "yes")
        self.assertTrue(result["changed"])
        self.assertEqual(
            result["status"], "promoted_by_semantic_confirmation"
        )

    def test_preserves_negative_after_semantic_rejection(self) -> None:
        result = self.verify("No", [review(0, "No")])

        self.assertEqual(result["final_answer"], "no")
        self.assertFalse(result["changed"])
        self.assertEqual(
            result["status"], "negative_preserved_by_semantic_rejection"
        )

    def test_requires_exact_semantic_answer(self) -> None:
        result = self.verify("No", [review(0, "Yes, it is visible")])

        self.assertEqual(result["final_answer"], "no")
        self.assertEqual(
            result["status"],
            "negative_preserved_non_exact_semantic_review",
        )

    def test_filters_near_full_frame_mask(self) -> None:
        candidates, rejected = select_semantic_candidates(
            [annotation(0, mask_area=9500)],
            target="bus",
            image_width=100,
            image_height=100,
            config=self.config,
        )

        self.assertEqual(candidates, [])
        self.assertIn("large_mask", rejected[0]["rejection_reasons"])
        result = verify_binary_answer_v2(
            "No",
            target="bus",
            annotations=[annotation(0, mask_area=9500)],
            image_width=100,
            image_height=100,
            semantic_reviews=[],
            config=self.config,
        )
        self.assertEqual(
            result["status"], "negative_preserved_by_geometry_gate"
        )

    def test_limits_review_candidates_by_grounding_score(self) -> None:
        candidates, rejected = select_semantic_candidates(
            [
                annotation(0, score=0.4),
                annotation(1, score=0.8),
                annotation(2, score=0.6),
            ],
            target="bus",
            image_width=100,
            image_height=100,
            config=self.config,
        )

        self.assertEqual(
            [item["annotation_index"] for item in candidates], [1, 2]
        )
        self.assertIn("candidate_limit", rejected[0]["rejection_reasons"])

    def test_positive_baseline_never_requires_review(self) -> None:
        result = self.verify("Yes", [])

        self.assertEqual(result["final_answer"], "yes")
        self.assertEqual(
            result["status"],
            "positive_preserved_without_negative_recheck",
        )

    def test_missing_candidate_review_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing candidate"):
            self.verify("No", [])

    def test_context_crop_is_square_and_clamped(self) -> None:
        box = context_crop_box(
            [0, 0, 10, 20],
            image_width=100,
            image_height=80,
            padding_ratio=0.25,
            min_crop_size=40,
        )

        self.assertEqual(box, [0, 0, 40, 40])

    def test_writes_decodable_rgb_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            output = Path(directory) / "crop.jpg"
            Image.new("RGB", (100, 80), "white").save(source)

            metadata = write_semantic_crop(
                source,
                output,
                box=[0, 0, 10, 20],
                config=self.config,
            )

            self.assertTrue(output.is_file())
            with Image.open(output) as crop:
                self.assertEqual(crop.mode, "RGB")
                self.assertEqual(crop.size, (32, 32))
            self.assertEqual(metadata["crop_width"], 32)


if __name__ == "__main__":
    unittest.main()
