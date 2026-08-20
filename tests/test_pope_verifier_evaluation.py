from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.grounding_answer_verifier import (
    GroundingAnswerVerifierConfig,
)
from grounded_visual_assistant.pope_evaluation import evaluate_answer
from grounded_visual_assistant.pope_verifier_evaluation import (
    aggregate_pope_verifier_metrics,
    build_verified_prediction,
    group_verification_queries,
    verification_query_key,
)


def baseline(
    sample_id: str,
    *,
    strategy: str,
    target: str,
    answer: str,
    object_name: str = "dog",
) -> dict:
    return {
        "id": sample_id,
        "strategy": strategy,
        "question_id": 1,
        "image": "image.jpg",
        "image_id": 7,
        "question": f"Is there a {object_name} in the image?",
        "object": object_name,
        "gt_answer": target,
        "prediction": answer,
        "evaluation": evaluate_answer(answer, target),
        "latency_seconds": 0.2,
        "cuda_peak_memory_allocated_gb": 16.0,
    }


def evidence(record: dict, score: float | None) -> dict:
    annotations = []
    if score is not None:
        annotations.append(
            {
                "class_name": record["object"],
                "bbox": [0, 0, 20, 20],
                "score": score,
                "mask_score": 0.95,
                "mask_area": 300,
                "segmentation": {"size": [100, 100], "counts": "rle"},
            }
        )
    return {
        "query_key": verification_query_key(record),
        "cuda_peak_memory_allocated_gb": 2.5,
        "grounding": {
            "annotations": annotations,
            "img_width": 100,
            "img_height": 100,
            "latency_seconds": {"total": 0.4},
        },
    }


class PopeVerifierEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GroundingAnswerVerifierConfig(
            evidence_score_threshold=0.3,
            promotion_score_threshold=0.45,
        )

    def test_query_key_excludes_strategy_and_ground_truth(self) -> None:
        first = baseline(
            "random", strategy="random", target="yes", answer="No"
        )
        second = baseline(
            "popular", strategy="popular", target="yes", answer="No"
        )
        self.assertEqual(
            verification_query_key(first),
            verification_query_key(second),
        )

        groups = group_verification_queries([first, second])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["baseline_ids"], ["random", "popular"])

    def test_conflicting_labels_for_same_query_are_rejected(self) -> None:
        first = baseline(
            "random", strategy="random", target="yes", answer="No"
        )
        second = baseline(
            "popular", strategy="popular", target="no", answer="No"
        )
        with self.assertRaisesRegex(RuntimeError, "conflicting"):
            group_verification_queries([first, second])

    def test_builds_beneficial_and_harmful_corrections(self) -> None:
        missed = baseline(
            "missed", strategy="random", target="yes", answer="No"
        )
        correct_negative = baseline(
            "negative",
            strategy="popular",
            target="no",
            answer="No",
            object_name="cat",
        )
        rescued = build_verified_prediction(
            missed,
            evidence(missed, 0.8),
            config=self.config,
        )
        harmed = build_verified_prediction(
            correct_negative,
            evidence(correct_negative, 0.8),
            config=self.config,
        )

        metrics = aggregate_pope_verifier_metrics(
            [rescued, harmed],
            expected_samples=2,
            expected_queries=2,
            completed_queries=2,
        )
        paired = metrics["paired_outcomes"]
        self.assertEqual(paired["beneficial_corrections"], 1)
        self.assertEqual(paired["harmful_corrections"], 1)
        self.assertEqual(paired["net_correct_corrections"], 0)
        self.assertEqual(metrics["corrections"]["changed_answers"], 2)
        self.assertEqual(
            metrics["latency_seconds"][
                "uncached_projected_end_to_end_mean_per_question"
            ],
            0.6,
        )

    def test_reports_deduplicated_grounding_latency(self) -> None:
        random = baseline(
            "random", strategy="random", target="yes", answer="No"
        )
        popular = baseline(
            "popular", strategy="popular", target="yes", answer="No"
        )
        shared_evidence = evidence(random, 0.8)
        predictions = [
            build_verified_prediction(
                item,
                shared_evidence,
                config=self.config,
            )
            for item in (random, popular)
        ]

        metrics = aggregate_pope_verifier_metrics(
            predictions,
            expected_samples=2,
            expected_queries=1,
            completed_queries=1,
        )
        self.assertEqual(
            metrics["coverage"][
                "grounding_queries_saved_by_deduplication"
            ],
            1,
        )
        self.assertEqual(
            metrics["latency_seconds"]["verification_unique_query_total"],
            0.4,
        )
        self.assertEqual(
            metrics["latency_seconds"][
                "uncached_projected_end_to_end_total"
            ],
            1.2,
        )


if __name__ == "__main__":
    unittest.main()
