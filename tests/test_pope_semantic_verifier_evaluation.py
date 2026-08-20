from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_evaluation import evaluate_answer
from grounded_visual_assistant.pope_semantic_verifier_evaluation import (
    POPE_SEMANTIC_VERIFIER_BATCH_PROTOCOL,
    aggregate_pope_semantic_verifier_metrics,
    build_semantic_review_jobs,
    build_semantic_verified_prediction,
)
from grounded_visual_assistant.pope_verifier_evaluation import (
    group_verification_queries,
)
from grounded_visual_assistant.semantic_answer_verifier import (
    SemanticAnswerVerifierConfig,
)


def baseline(
    sample_id: str,
    *,
    strategy: str,
    prediction: str = "No",
    target: str = "yes",
) -> dict:
    return {
        "id": sample_id,
        "strategy": strategy,
        "question_id": 1,
        "image": "image.jpg",
        "image_id": 1,
        "question": "Is there a bus?",
        "object": "bus",
        "gt_answer": target,
        "prediction": prediction,
        "evaluation": evaluate_answer(prediction, target),
        "model": "qwen",
        "latency_seconds": 1.0,
        "generated_tokens": 1,
        "cuda_peak_memory_allocated_gb": 16.0,
    }


def evidence(query_key: str) -> dict:
    return {
        "query_key": query_key,
        "image": "image.jpg",
        "image_id": 1,
        "question": "Is there a bus?",
        "object": "bus",
        "cuda_peak_memory_allocated_gb": 3.0,
        "grounding": {
            "img_width": 100,
            "img_height": 100,
            "latency_seconds": {"total": 0.2},
            "annotations": [
                {
                    "class_name": "bus",
                    "bbox": [10, 10, 60, 60],
                    "score": 0.7,
                    "mask_score": 0.9,
                    "mask_area": 2500,
                }
            ],
        },
    }


class PopeSemanticVerifierEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SemanticAnswerVerifierConfig(min_crop_size=32)

    def test_deduplicates_semantic_jobs_across_strategies(self) -> None:
        records = [
            baseline("random-1", strategy="random"),
            baseline("popular-1", strategy="popular"),
            baseline("adversarial-1", strategy="adversarial"),
        ]
        groups = group_verification_queries(records)
        source = evidence(groups[0]["query_key"])

        jobs = build_semantic_review_jobs(
            groups,
            records_by_id={item["id"]: item for item in records},
            evidence_by_key={source["query_key"]: source},
            config=self.config,
        )

        self.assertEqual(len(jobs), 1)
        self.assertEqual(len(jobs[0]["baseline_ids"]), 3)
        self.assertNotIn("gt_answer", jobs[0])

    def test_builds_beneficial_semantic_correction(self) -> None:
        record = baseline("random-1", strategy="random")
        group = group_verification_queries([record])[0]
        source = evidence(group["query_key"])
        jobs = build_semantic_review_jobs(
            [group],
            records_by_id={record["id"]: record},
            evidence_by_key={source["query_key"]: source},
            config=self.config,
        )
        review = {
            **jobs[0],
            "answer": "Yes",
            "latency_seconds": 0.5,
            "end_to_end_latency_seconds": 0.6,
            "generated_tokens": 1,
            "cuda_peak_memory_allocated_gb": 16.0,
        }

        prediction = build_semantic_verified_prediction(
            record,
            source,
            reviews_by_key={review["candidate_key"]: review},
            config=self.config,
        )
        metrics = aggregate_pope_semantic_verifier_metrics(
            [prediction],
            expected_samples=1,
            expected_queries=1,
            completed_queries=1,
            expected_reviews=1,
            completed_reviews=1,
        )

        self.assertEqual(prediction["prediction"], "yes")
        self.assertEqual(
            metrics["protocol"], POPE_SEMANTIC_VERIFIER_BATCH_PROTOCOL
        )
        self.assertEqual(
            metrics["paired_outcomes"]["beneficial_corrections"], 1
        )
        self.assertEqual(metrics["semantic_review"]["parsed_answers"]["yes"], 1)
        self.assertEqual(
            metrics["paired_outcomes"][
                "mcnemar_exact_two_sided_p_value"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
