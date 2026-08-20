from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.verifier_dev_semantic_review import (
    VERIFIER_DEV_SEMANTIC_REVIEW_PROTOCOL,
    aggregate_dev_semantic_review_metrics,
    build_dev_semantic_review_jobs,
    ordered_candidate_keys_sha256,
)
from grounded_visual_assistant.semantic_answer_verifier import (
    SemanticAnswerVerifierConfig,
)


def evidence(
    query_key: str,
    baseline_id: str,
    *,
    scores: list[float],
) -> dict:
    return {
        "query_key": query_key,
        "baseline_id": baseline_id,
        "image": "image.jpg",
        "image_id": 1,
        "question": f"Is there a {baseline_id}?",
        "object": baseline_id,
        "grounding": {
            "img_width": 100,
            "img_height": 100,
            "annotations": [
                {
                    "class_name": baseline_id,
                    "bbox": [10 + index, 10, 50 + index, 50],
                    "score": score,
                    "mask_score": 0.9,
                    "mask_area": 1000,
                }
                for index, score in enumerate(scores)
            ],
        },
    }


class VerifierDevSemanticReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SemanticAnswerVerifierConfig(
            max_mask_area_ratio=1.0,
            max_candidates_per_query=2,
            min_crop_size=32,
        )

    def test_builds_candidate_union_without_gt(self) -> None:
        records = [
            evidence("query-a", "book", scores=[0.6, 0.4, 0.35]),
            evidence("query-b", "chair", scores=[]),
        ]

        jobs = build_dev_semantic_review_jobs(
            records, config=self.config
        )

        self.assertEqual(len(jobs), 2)
        self.assertEqual(
            [item["annotation_index"] for item in jobs], [0, 1]
        )
        self.assertTrue(ordered_candidate_keys_sha256(jobs))
        self.assertTrue(all("gt_answer" not in item for item in jobs))
        self.assertTrue(all("pair_role" not in item for item in jobs))

    def test_aggregates_post_inference_diagnostics(self) -> None:
        records = [
            evidence("query-a", "book", scores=[0.6]),
            evidence("query-b", "chair", scores=[0.5]),
        ]
        jobs = build_dev_semantic_review_jobs(
            records, config=self.config
        )
        reviews = [
            {
                **jobs[0],
                "answer": "Yes",
                "latency_seconds": 0.2,
                "generated_tokens": 2,
                "hit_max_new_tokens": False,
                "cuda_peak_memory_allocated_gb": 16.0,
            },
            {
                **jobs[1],
                "answer": "No",
                "latency_seconds": 0.2,
                "generated_tokens": 2,
                "hit_max_new_tokens": False,
                "cuda_peak_memory_allocated_gb": 16.0,
            },
        ]
        baseline = {
            "book": {"gt_answer": "yes"},
            "chair": {"gt_answer": "no"},
        }

        metrics = aggregate_dev_semantic_review_metrics(
            reviews,
            jobs=jobs,
            evidence_records=records,
            baseline_by_id=baseline,
        )

        self.assertEqual(
            metrics["protocol"], VERIFIER_DEV_SEMANTIC_REVIEW_PROTOCOL
        )
        self.assertEqual(metrics["coverage"]["completed_candidates"], 2)
        self.assertEqual(metrics["answers"]["parsed"], {"no": 1, "yes": 1})
        self.assertEqual(
            metrics["dev_diagnostics"]["false_negative"][
                "semantic_yes_queries"
            ],
            1,
        )
        self.assertEqual(
            metrics["dev_diagnostics"]["true_negative"][
                "semantic_yes_queries"
            ],
            0,
        )
        self.assertFalse(
            metrics["methodology"]["inference_jobs_use_ground_truth"]
        )


if __name__ == "__main__":
    unittest.main()
