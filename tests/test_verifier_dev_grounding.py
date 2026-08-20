from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_evaluation import evaluate_answer
from grounded_visual_assistant.verifier_dev_grounding import (
    VERIFIER_DEV_GROUNDING_PROTOCOL,
    aggregate_verifier_dev_grounding_metrics,
    build_negative_grounding_jobs,
    ordered_query_keys_sha256,
)


def baseline(
    sample_id: str,
    *,
    prediction: str,
    target: str,
) -> dict:
    return {
        "id": sample_id,
        "image": "image.jpg",
        "image_id": 1,
        "question": f"Is there a {sample_id}?",
        "object": sample_id,
        "gt_answer": target,
        "prediction": prediction,
        "evaluation": evaluate_answer(prediction, target),
        "pair_role": "positive" if target == "yes" else "hard_negative",
    }


def evidence(
    query_key: str,
    baseline_id: str,
    *,
    score: float | None,
) -> dict:
    annotations = (
        []
        if score is None
        else [
            {
                "class_name": baseline_id,
                "bbox": [1, 2, 10, 20],
                "score": score,
                "mask_score": 0.9,
                "mask_area": 100,
            }
        ]
    )
    return {
        "query_key": query_key,
        "baseline_id": baseline_id,
        "grounding": {
            "latency_seconds": {"total": 0.2},
            "cuda_peak_memory_allocated_gb": 3.0,
            "annotations": annotations,
        },
    }


class VerifierDevGroundingTest(unittest.TestCase):
    def test_selects_only_negative_answers_without_gt_fields(self) -> None:
        records = [
            baseline("book", prediction="No", target="yes"),
            baseline("remote", prediction="Yes", target="no"),
        ]

        jobs = build_negative_grounding_jobs(records)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["baseline_id"], "book")
        self.assertNotIn("gt_answer", jobs[0])
        self.assertNotIn("pair_role", jobs[0])
        self.assertEqual(
            ordered_query_keys_sha256(jobs),
            ordered_query_keys_sha256(jobs),
        )

    def test_aggregates_fn_and_tn_candidate_diagnostics(self) -> None:
        records = [
            baseline("book", prediction="No", target="yes"),
            baseline("chair", prediction="No", target="no"),
        ]
        jobs = build_negative_grounding_jobs(records)
        evidence_records = [
            evidence(jobs[0]["query_key"], "book", score=0.6),
            evidence(jobs[1]["query_key"], "chair", score=None),
        ]

        metrics = aggregate_verifier_dev_grounding_metrics(
            evidence_records,
            baseline_by_id={item["id"]: item for item in records},
            expected_queries=2,
        )

        self.assertEqual(metrics["protocol"], VERIFIER_DEV_GROUNDING_PROTOCOL)
        self.assertEqual(metrics["coverage"]["completed_queries"], 2)
        self.assertEqual(metrics["candidates"]["queries_with_candidates"], 1)
        self.assertEqual(
            metrics["dev_diagnostics"]["false_negative"][
                "candidate_presence_rate"
            ],
            1.0,
        )
        self.assertEqual(
            metrics["dev_diagnostics"]["true_negative"][
                "candidate_presence_rate"
            ],
            0.0,
        )
        self.assertFalse(
            metrics["methodology"]["inference_selection_uses_ground_truth"]
        )


if __name__ == "__main__":
    unittest.main()
