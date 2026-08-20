from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_evaluation import (
    aggregate_metrics,
    binary_metrics,
    evaluate_answer,
    official_parse_answer,
    select_records,
    selected_ids_sha256,
)


def sample(
    sample_id: str,
    strategy: str,
    target: str,
    answer: str,
) -> dict:
    return {
        "id": sample_id,
        "strategy": strategy,
        "evaluation": evaluate_answer(answer, target),
        "latency_seconds": 1.0,
        "generated_tokens": 2,
        "hit_max_new_tokens": False,
    }


class PopeEvaluationTest(unittest.TestCase):
    def test_reproduces_official_answer_conversion(self) -> None:
        self.assertEqual(official_parse_answer("No, there is not."), "no")
        self.assertEqual(official_parse_answer("There is no dog."), "no")
        self.assertEqual(official_parse_answer("Yes."), "yes")
        # The official evaluator maps any answer without no/not to yes.
        self.assertEqual(official_parse_answer("I cannot tell."), "yes")

    def test_tracks_strict_parse_separately(self) -> None:
        valid = evaluate_answer("Yes.", "yes")
        invalid = evaluate_answer("I cannot tell.", "no")

        self.assertTrue(valid["strict_parse_valid"])
        self.assertIsNone(invalid["strict_prediction"])
        self.assertEqual(invalid["official_prediction"], "yes")
        self.assertFalse(invalid["is_correct"])

    def test_computes_official_binary_metrics(self) -> None:
        predictions = [
            sample("tp", "random", "yes", "Yes."),
            sample("fp", "random", "no", "Yes."),
            sample("tn", "random", "no", "No."),
            sample("fn", "random", "yes", "No."),
        ]

        metrics = binary_metrics(predictions)

        self.assertEqual(
            metrics["confusion"],
            {"tp": 1, "fp": 1, "tn": 1, "fn": 1},
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["yes_ratio"], 0.5)

    def test_selects_equal_prefix_from_each_strategy(self) -> None:
        records = [
            {"id": f"{strategy}-{index}", "strategy": strategy}
            for strategy in ("random", "popular", "adversarial")
            for index in range(3)
        ]

        selected = select_records(records, samples_per_strategy=2)

        self.assertEqual(
            [item["id"] for item in selected],
            [
                "random-0",
                "random-1",
                "popular-0",
                "popular-1",
                "adversarial-0",
                "adversarial-1",
            ],
        )
        self.assertEqual(
            selected_ids_sha256(selected),
            selected_ids_sha256(selected),
        )

    def test_aggregates_metrics_by_strategy(self) -> None:
        predictions = [
            sample("r1", "random", "yes", "Yes."),
            sample("r2", "random", "no", "No."),
            sample("p1", "popular", "yes", "No."),
        ]

        metrics = aggregate_metrics(predictions, expected_samples=3)

        self.assertEqual(metrics["coverage"]["completed"], 3)
        self.assertEqual(metrics["overall"]["accuracy"], 0.666667)
        self.assertEqual(metrics["strategies"]["random"]["accuracy"], 1.0)
        self.assertEqual(metrics["strategies"]["popular"]["recall"], 0.0)
        self.assertEqual(metrics["generation"]["token_limit_hits"], 0)


if __name__ == "__main__":
    unittest.main()

