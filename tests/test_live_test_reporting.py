from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.live_test_reporting import (
    analyze_live_test_predictions,
    build_generalization_rows,
    relation_confusion_rows,
)


def compact_metrics(value: float) -> dict:
    return {
        "overall": {"exact_accuracy": value},
        "tasks": {
            "object_listing": {
                "macro_f1": value,
                "exact_accuracy": value,
            },
            "object_existence": {"exact_accuracy": value},
            "spatial_relation": {
                "exact_accuracy": value,
                "balanced_accuracy": value,
                "parse_valid_rate": value,
                "confusion": {"above": {"above": 1}},
            },
        },
        "structured_targets": {
            "schema_valid_rate": value,
            "micro_f1": value,
        },
        "required_evidence_box_metrics": {
            "box_iou_50": {"micro_f1": value}
        },
        "required_evidence_mask_iou_50": {"micro_f1": value},
        "end_to_end": {
            "overall": {
                "answer_and_any_evidence_success_rate": value,
                "answer_and_complete_evidence_success_rate": value,
            }
        },
        "latency_seconds": {"mean": value},
    }


def failed_record() -> dict:
    return {
        "id": "failed",
        "image_id": 1,
        "task_type": "object_listing",
        "gt_answer": "person",
        "prediction": '{"answer":"person"',
        "evaluation": {"score": 0.0, "is_correct": False},
        "vlm_output": {
            "schema_valid": False,
            "parse_source": "unparseable",
            "generated_tokens": 192,
        },
        "targets": [],
        "target_evaluation": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fp": 0,
            "fn": 1,
        },
        "evidence_evaluation": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fp": 0,
            "fn": 1,
        },
        "mask_evaluation": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fp": 0,
            "fn": 1,
        },
        "evidence_required": True,
        "evidence_supported": False,
        "evidence_complete": False,
        "end_to_end_success": False,
        "end_to_end_complete_success": False,
        "latency_seconds": 3.0,
    }


class LiveTestReportingTest(unittest.TestCase):
    def test_generalization_rows_compute_test_minus_dev(self) -> None:
        rows = build_generalization_rows(
            compact_metrics(0.5), compact_metrics(0.75)
        )
        self.assertTrue(rows)
        self.assertTrue(
            all(item["delta_test_minus_dev"] == 0.25 for item in rows)
        )

    def test_failure_analysis_marks_truncation_and_evidence_misses(self) -> None:
        summary, per_sample = analyze_live_test_predictions(
            [failed_record()], max_new_tokens=192
        )
        self.assertEqual(summary["schema_invalid_count"], 1)
        self.assertEqual(summary["token_limit_hit_count"], 1)
        self.assertIn("token_limit_hit", per_sample[0]["flags"])
        self.assertIn("target_miss", per_sample[0]["flags"])
        self.assertIn("box_miss", per_sample[0]["flags"])
        self.assertIn("mask_miss", per_sample[0]["flags"])
        self.assertGreater(per_sample[0]["severity"], 0)

    def test_relation_confusion_is_flattened(self) -> None:
        rows = relation_confusion_rows(compact_metrics(1.0))
        self.assertEqual(
            rows, [{"target": "above", "prediction": "above", "count": 1}]
        )


if __name__ == "__main__":
    unittest.main()
