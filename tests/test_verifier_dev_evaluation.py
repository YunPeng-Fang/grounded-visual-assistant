from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_evaluation import evaluate_answer
from grounded_visual_assistant.verifier_dev_evaluation import (
    VERIFIER_DEV_BASELINE_PROTOCOL,
    aggregate_verifier_dev_metrics,
)


def prediction(
    sample_id: str,
    *,
    role: str,
    target: str,
    answer: str,
) -> dict:
    return {
        "id": sample_id,
        "pair_id": "pair-1",
        "pair_role": role,
        "supercategory": "vehicle",
        "prediction": answer,
        "evaluation": evaluate_answer(answer, target),
        "latency_seconds": 1.0,
        "generated_tokens": 2,
        "hit_max_new_tokens": False,
        "cuda_peak_memory_allocated_gb": 16.0,
    }


class VerifierDevEvaluationTest(unittest.TestCase):
    def test_reports_binary_and_pair_metrics(self) -> None:
        records = [
            prediction(
                "positive",
                role="positive",
                target="yes",
                answer="Yes",
            ),
            prediction(
                "negative",
                role="hard_negative",
                target="no",
                answer="No",
            ),
        ]

        metrics = aggregate_verifier_dev_metrics(
            records, expected_samples=2
        )

        self.assertEqual(
            metrics["protocol"], VERIFIER_DEV_BASELINE_PROTOCOL
        )
        self.assertEqual(metrics["overall"]["accuracy"], 1.0)
        self.assertEqual(metrics["pairs"]["both_correct"], 1)
        self.assertEqual(metrics["pairs"]["both_correct_rate"], 1.0)
        self.assertEqual(metrics["generation"]["token_limit_hits"], 0)


if __name__ == "__main__":
    unittest.main()
