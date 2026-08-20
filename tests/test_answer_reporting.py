from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.answer_reporting import (
    aggregate_answer_analysis,
    analyze_answer_record,
)


def base_record(task_type: str) -> dict:
    return {
        "id": task_type,
        "image": "image.jpg",
        "image_id": 1,
        "question": "question",
        "task_type": task_type,
        "gt_answer": "target",
        "query_plan": {"categories": []},
        "answer_policy": {
            "forced_answer": "prediction",
            "selective_answer": "prediction",
            "abstained": False,
            "status": "supported",
            "selected_evidence": [],
            "accepted_evidence": [],
            "rejected_evidence": [],
            "diagnostics": {},
        },
        "evaluation": {"is_correct": False},
    }


class AnswerReportingTest(unittest.TestCase):
    def test_listing_separates_query_and_gate_misses(self) -> None:
        record = base_record("object_listing")
        record["query_plan"]["categories"] = ["cup", "vase", "book"]
        record["evaluation"].update(
            {
                "target_categories": ["cup", "vase", "person"],
                "predicted_categories": ["cup", "book"],
            }
        )
        result = analyze_answer_record(record)
        self.assertEqual(result["query_missed_categories"], ["person"])
        self.assertEqual(result["evidence_gate_missed_categories"], ["vase"])
        self.assertEqual(result["extra_categories"], ["book"])
        self.assertIn("listing_vlm_query_miss", result["failure_tags"])
        self.assertIn("listing_evidence_gate_miss", result["failure_tags"])

    def test_existence_disagreement_can_catch_a_forced_error(self) -> None:
        record = base_record("object_existence")
        record["answer_policy"].update(
            {
                "selective_answer": None,
                "abstained": True,
                "status": "vlm_grounding_disagreement",
                "diagnostics": {
                    "vlm_answer": "yes",
                    "detector_answer": "no",
                    "agreement": False,
                },
            }
        )
        result = analyze_answer_record(record)
        self.assertEqual(
            result["failure_tags"], ["existence_error_caught_by_disagreement"]
        )
        self.assertIsNone(result["selective_correct"])

    def test_spatial_marks_stricter_gate_regression(self) -> None:
        record = base_record("spatial_relation")
        record["answer_policy"].update(
            {
                "forced_answer": "insufficient evidence",
                "selective_answer": None,
                "abstained": True,
                "status": "insufficient_evidence",
                "diagnostics": {"missing_categories": ["person"]},
            }
        )
        initial = {"evaluation": {"is_correct": True}}
        result = analyze_answer_record(record, initial)
        self.assertIn("spatial_missing_evidence", result["failure_tags"])
        self.assertIn("spatial_stricter_gate_regression", result["failure_tags"])
        self.assertEqual(result["transition_from_initial"], "initial_only_correct")

    def test_aggregate_counts_failures_and_abstentions(self) -> None:
        listing = base_record("object_listing")
        listing["evaluation"].update(
            {"target_categories": ["cup"], "predicted_categories": []}
        )
        listing["query_plan"]["categories"] = []
        existence = base_record("object_existence")
        existence["evaluation"]["is_correct"] = True
        existence["answer_policy"].update(
            {
                "selective_answer": None,
                "abstained": True,
                "status": "vlm_grounding_disagreement",
            }
        )
        analyses = [
            analyze_answer_record(listing),
            analyze_answer_record(existence),
        ]
        summary = aggregate_answer_analysis(analyses)
        self.assertEqual(summary["overall"]["forced_errors"], 1)
        self.assertEqual(summary["overall"]["abstentions"], 1)
        self.assertEqual(
            summary["tasks"]["object_existence"]["selective_errors"], 0
        )


if __name__ == "__main__":
    unittest.main()
