from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.live_prompt_policy_comparison import (
    ACCEPTANCE_METRICS,
    compare_live_prompt_policies,
    render_live_prompt_policy_report,
)


def prediction(
    sample_id: str,
    task_type: str,
    *,
    correct: bool,
) -> dict:
    return {
        "id": sample_id,
        "image_id": 1,
        "question": f"Question {sample_id}",
        "task_type": task_type,
        "gt_answer": "yes",
        "source": "COCO val2017",
        "split": "dev",
        "prediction": "yes" if correct else "no",
        "targets": ["person"],
        "evaluation": {"is_correct": correct},
        "vlm_output": {"schema_valid": True},
        "end_to_end_success": correct,
    }


def metrics(value: float) -> dict:
    return {
        "overall": {"exact_accuracy": value},
        "tasks": {
            "object_listing": {"macro_f1": value},
            "object_existence": {"exact_accuracy": value},
            "spatial_relation": {
                "exact_accuracy": value,
                "parse_valid_rate": value,
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
        "cuda_memory_gb": {"peak_allocated_max": value},
    }


class LivePromptPolicyComparisonTest(unittest.TestCase):
    def test_paired_transitions_and_gates(self) -> None:
        baseline = [
            prediction("listing", "object_listing", correct=True),
            prediction("existence", "object_existence", correct=False),
            prediction("relation", "spatial_relation", correct=False),
        ]
        candidate = [
            prediction("listing", "object_listing", correct=True),
            prediction("existence", "object_existence", correct=True),
            prediction("relation", "spatial_relation", correct=True),
        ]
        acceptance = {name: 0.7 for name in ACCEPTANCE_METRICS}
        summary, transitions = compare_live_prompt_policies(
            baseline,
            candidate,
            metrics(0.5),
            metrics(0.8),
            acceptance,
        )
        self.assertTrue(summary["acceptance"]["all_gates_passed"])
        self.assertEqual(
            summary["paired"]["overall"]["candidate_only_correct"], 2
        )
        self.assertEqual(
            summary["paired"]["tasks"]["spatial_relation"]["net_correct"], 1
        )
        self.assertEqual(
            {item["transition"] for item in transitions},
            {"both_correct", "candidate_only_correct"},
        )
        report = render_live_prompt_policy_report(summary)
        self.assertIn("accept_task_aware_coco_v1", report)
        self.assertIn("task-aware-coco-v1", report)

    def test_report_uses_explicit_policy_names(self) -> None:
        baseline = [prediction("x", "object_existence", correct=True)]
        candidate = [prediction("x", "object_existence", correct=True)]
        summary, _ = compare_live_prompt_policies(
            baseline,
            candidate,
            metrics(1.0),
            metrics(1.0),
            {name: 0.0 for name in ACCEPTANCE_METRICS},
            baseline_policy="task-aware-coco-v1",
            candidate_policy="task-aware-coco-v2",
        )
        report = render_live_prompt_policy_report(summary)
        self.assertIn("task-aware-coco-v1", report)
        self.assertIn("task-aware-coco-v2", report)
        self.assertIn("task-aware-coco-v1 only", report)
        self.assertIn("task-aware-coco-v2 only", report)
        self.assertEqual(
            summary["acceptance"]["decision"],
            "accept_task_aware_coco_v2",
        )

    def test_rejects_unpaired_ground_truth(self) -> None:
        baseline = [prediction("x", "object_existence", correct=True)]
        candidate = [prediction("x", "object_existence", correct=True)]
        candidate[0]["gt_answer"] = "no"
        with self.assertRaisesRegex(RuntimeError, "gt_answer"):
            compare_live_prompt_policies(
                baseline,
                candidate,
                metrics(1.0),
                metrics(1.0),
                {name: 0.0 for name in ACCEPTANCE_METRICS},
            )


if __name__ == "__main__":
    unittest.main()
