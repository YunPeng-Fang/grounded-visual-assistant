from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_evaluation import evaluate_answer
from grounded_visual_assistant.verifier_dev_contrastive_review import (
    build_contrastive_review_jobs,
    contrastive_options,
    evaluate_contrastive_cascade,
    ordered_v3_keys_sha256,
    parse_contrastive_answer,
    validate_coco_ontology,
    write_marked_candidate_crop,
)


def semantic_review(
    *,
    query: str,
    baseline_id: str,
    target: str,
    score: float,
    answer: str,
    annotation_index: int = 0,
) -> dict:
    return {
        "candidate_key": (
            f"{query}__annotation-{annotation_index:03d}"
        ),
        "query_key": query,
        "baseline_id": baseline_id,
        "image": "image.jpg",
        "image_id": 1,
        "question": f"Is there a {target}?",
        "object": target,
        "annotation_index": annotation_index,
        "grounding_score": score,
        "mask_score": 0.9,
        "mask_area_ratio": 0.1,
        "crop_image": "crop.jpg",
        "crop_sha256": "hash",
        "source_box_xyxy": [25, 25, 75, 75],
        "crop_box_xyxy": [0, 0, 100, 100],
        "answer": answer,
    }


def baseline(
    sample_id: str,
    *,
    target: str,
    prediction: str,
    role: str,
) -> dict:
    return {
        "id": sample_id,
        "pair_id": f"pair-{sample_id}",
        "pair_role": role,
        "image": "image.jpg",
        "image_id": 1,
        "question": f"Is there a {sample_id}?",
        "object": sample_id,
        "gt_answer": target,
        "prediction": prediction,
        "evaluation": evaluate_answer(prediction, target),
    }


class VerifierDevContrastiveReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = yaml.safe_load(
            (
                PROJECT_ROOT
                / "configs"
                / "coco_80_supercategories_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        cls.groups, cls.category_to_group = validate_coco_ontology(
            payload
        )

    def test_ontology_covers_coco_and_builds_vehicle_options(self) -> None:
        options = contrastive_options(
            "truck",
            groups=self.groups,
            category_to_group=self.category_to_group,
        )

        self.assertIn("bus", options)
        self.assertIn("truck", options)
        self.assertIn("none", options)
        self.assertEqual(options, tuple(sorted(options)))

    def test_jobs_use_only_top1_exact_v2_yes_without_gt(self) -> None:
        reviews = [
            semantic_review(
                query="query-a",
                baseline_id="chair",
                target="chair",
                score=0.6,
                answer="No",
            ),
            semantic_review(
                query="query-a",
                baseline_id="chair",
                target="chair",
                score=0.5,
                answer="Yes",
                annotation_index=1,
            ),
            semantic_review(
                query="query-b",
                baseline_id="truck",
                target="truck",
                score=0.7,
                answer="Yes",
            ),
            semantic_review(
                query="query-c",
                baseline_id="book",
                target="book",
                score=0.7,
                answer="Yes.",
            ),
        ]

        jobs, selected = build_contrastive_review_jobs(
            reviews,
            groups=self.groups,
            category_to_group=self.category_to_group,
            evidence_score_threshold=0.3,
            min_mask_score=0.5,
            min_mask_area_ratio=0.0,
            max_mask_area_ratio=0.9,
            max_candidates_per_query=1,
            none_label="none",
            require_exact_v2_yes=True,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual([item["object"] for item in jobs], ["truck"])
        self.assertTrue(ordered_v3_keys_sha256(jobs))
        self.assertTrue(all("gt_answer" not in item for item in jobs))
        self.assertTrue(all("pair_role" not in item for item in jobs))

    def test_writes_visible_marker_and_parses_exact_label(self) -> None:
        job = semantic_review(
            query="query-a",
            baseline_id="truck",
            target="truck",
            score=0.7,
            answer="Yes",
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jpg"
            output = Path(directory) / "marked.jpg"
            Image.new("RGB", (100, 100), "white").save(source)

            metadata = write_marked_candidate_crop(
                source,
                output,
                job=job,
                marker_color=[255, 0, 0],
                marker_width=4,
            )
            with Image.open(output) as image:
                red_pixels = sum(
                    red > 180 and green < 100 and blue < 100
                    for red, green, blue in image.convert("RGB").getdata()
                )

        self.assertGreater(red_pixels, 100)
        self.assertEqual(metadata["marker_box_xyxy"], [25, 25, 75, 75])
        self.assertEqual(
            parse_contrastive_answer(
                "bus", ["bus", "truck", "none"]
            )["selected_label"],
            "bus",
        )
        self.assertFalse(
            parse_contrastive_answer(
                "bus.", ["bus", "truck", "none"]
            )["valid"]
        )

    def test_cascade_locks_one_beneficial_correction(self) -> None:
        records = [
            baseline(
                "book",
                target="yes",
                prediction="No",
                role="positive",
            ),
            baseline(
                "truck",
                target="no",
                prediction="No",
                role="hard_negative",
            ),
            baseline(
                "person",
                target="yes",
                prediction="Yes",
                role="positive",
            ),
        ]
        raw_reviews = [
            semantic_review(
                query="query-book",
                baseline_id="book",
                target="book",
                score=0.6,
                answer="Yes",
            ),
            semantic_review(
                query="query-truck",
                baseline_id="truck",
                target="truck",
                score=0.7,
                answer="Yes",
            ),
        ]
        jobs, _ = build_contrastive_review_jobs(
            raw_reviews,
            groups=self.groups,
            category_to_group=self.category_to_group,
            evidence_score_threshold=0.3,
            min_mask_score=0.5,
            min_mask_area_ratio=0.0,
            max_mask_area_ratio=0.9,
            max_candidates_per_query=1,
            none_label="none",
            require_exact_v2_yes=True,
        )
        reviews = [
            {"v3_key": jobs[0]["v3_key"], "answer": "book"},
            {"v3_key": jobs[1]["v3_key"], "answer": "bus"},
        ]

        predictions, metrics = evaluate_contrastive_cascade(
            records,
            jobs=jobs,
            contrastive_reviews=reviews,
            require_strict_accuracy_improvement=True,
            require_non_decreasing_f1=True,
            require_positive_net_corrections=True,
        )

        self.assertEqual(metrics["v3"]["accuracy"], 1.0)
        self.assertEqual(metrics["corrections"]["beneficial"], 1)
        self.assertEqual(metrics["corrections"]["harmful"], 0)
        self.assertTrue(metrics["selection"]["eligible"])
        self.assertEqual(
            next(item for item in predictions if item["id"] == "truck")[
                "prediction"
            ],
            "no",
        )


if __name__ == "__main__":
    unittest.main()
