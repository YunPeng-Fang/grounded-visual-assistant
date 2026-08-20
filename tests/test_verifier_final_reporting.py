from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_evaluation import evaluate_answer
from grounded_visual_assistant.verifier_final_reporting import (
    build_failure_analysis,
    build_variant_summary,
    markdown_report,
    validate_final_freeze,
)


def metric(accuracy: float, f1: float) -> dict:
    return {
        "count": 2,
        "confusion": {"tp": 1, "fp": 0, "tn": 1, "fn": 0},
        "accuracy": accuracy,
        "precision": 1.0,
        "recall": 1.0,
        "f1": f1,
        "yes_ratio": 0.5,
        "strict_parse_valid_rate": 1.0,
    }


def policy_row(
    policy_id: str,
    *,
    accuracy: float,
    f1: float,
    changed: int,
    beneficial: int,
    harmful: int,
    reviews: int,
) -> dict:
    return {
        "policy_id": policy_id,
        "module": "baseline" if policy_id == "baseline" else "semantic",
        "score_threshold": None if policy_id == "baseline" else 0.3,
        "accuracy": accuracy,
        "precision": 1.0,
        "recall": 1.0,
        "f1": f1,
        "changed_answers": changed,
        "beneficial": beneficial,
        "harmful": harmful,
        "net_correct": beneficial - harmful,
        "grounding_queries": 0 if policy_id == "baseline" else 2,
        "semantic_reviews": reviews,
        "incremental_latency_seconds": float(reviews),
        "eligible": False,
        "rejection_reasons": (
            "reference_policy"
            if policy_id == "baseline"
            else "no_strict_accuracy_improvement"
        ),
    }


class VerifierFinalReportingTest(unittest.TestCase):
    def setUp(self) -> None:
        baseline_metrics = metric(1.0, 1.0)
        self.stage38 = {
            "protocol": "verifier_dev_offline_ablation_v1",
            "status": "completed",
            "coverage": {"grounding_queries": 2},
            "baseline": baseline_metrics,
            "selection": {
                "decision": "retain_baseline_no_eligible_verifier",
                "selected_policy_id": "baseline",
                "eligible_policy_ids": [],
            },
            "policy_table": [
                policy_row(
                    "baseline",
                    accuracy=1.0,
                    f1=1.0,
                    changed=0,
                    beneficial=0,
                    harmful=0,
                    reviews=0,
                ),
                policy_row(
                    "v1",
                    accuracy=0.5,
                    f1=0.5,
                    changed=1,
                    beneficial=0,
                    harmful=1,
                    reviews=0,
                ),
                policy_row(
                    "v2-rescue",
                    accuracy=0.5,
                    f1=0.5,
                    changed=2,
                    beneficial=1,
                    harmful=1,
                    reviews=2,
                ),
                policy_row(
                    "v2-noop",
                    accuracy=1.0,
                    f1=1.0,
                    changed=0,
                    beneficial=0,
                    harmful=0,
                    reviews=1,
                ),
            ],
        }
        self.stage38_policy = {
            "selected_policy": {"policy_id": "baseline"},
            "held_out_evaluation_pending": False,
        }
        self.stage39 = {
            "protocol": "verifier_dev_contrastive_review_v3",
            "status": "completed",
            "coverage": {
                "v2_top1_candidate_queries": 2,
                "completed_candidates": 1,
            },
            "evaluation": {
                "baseline": baseline_metrics,
                "v3": metric(0.5, 0.5),
                "corrections": {
                    "changed_answers": 1,
                    "beneficial": 0,
                    "harmful": 1,
                    "net_correct": -1,
                },
                "selection": {
                    "decision": "reject_v3_on_dev",
                    "eligible": False,
                    "rejection_reasons": [
                        "no_strict_accuracy_improvement"
                    ],
                },
            },
            "runtime_projection": {"incremental_latency_seconds": 3.0},
        }
        self.stage39_decision = {
            "decision": "reject_v3_on_dev",
            "held_out_evaluation_pending": False,
        }

    def test_requires_both_stages_to_reject_verifier(self) -> None:
        decision = validate_final_freeze(
            self.stage38,
            self.stage38_policy,
            self.stage39,
            self.stage39_decision,
        )

        self.assertEqual(
            decision["decision"],
            "retain_qwen_baseline_disable_answer_rewrite",
        )
        self.assertFalse(decision["held_out_verifier_run_permitted"])

    def test_rejects_an_eligible_v3(self) -> None:
        self.stage39["evaluation"]["selection"]["eligible"] = True

        with self.assertRaisesRegex(RuntimeError, "eligible"):
            validate_final_freeze(
                self.stage38,
                self.stage38_policy,
                self.stage39,
                self.stage39_decision,
            )

    def test_builds_five_variant_rows(self) -> None:
        rows = build_variant_summary(
            self.stage38,
            self.stage39,
            v1_policy_id="v1",
            v2_rescue_policy_id="v2-rescue",
            v2_noop_policy_id="v2-noop",
        )

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["decision"], "retained")
        self.assertEqual(rows[-1]["model_reviews"], 3)
        self.assertEqual(rows[-1]["decision"], "rejected")

    def test_traces_detector_miss_and_v2_regression(self) -> None:
        baseline = [
            {
                "id": "stop-sign",
                "pair_id": "a",
                "pair_role": "positive",
                "image_id": 1,
                "image": "a.jpg",
                "question": "Is there a stop sign?",
                "object": "stop sign",
                "gt_answer": "yes",
                "prediction": "No",
                "evaluation": evaluate_answer("No", "yes"),
            },
            {
                "id": "chair",
                "pair_id": "b",
                "pair_role": "hard_negative",
                "image_id": 2,
                "image": "b.jpg",
                "question": "Is there a chair?",
                "object": "chair",
                "gt_answer": "no",
                "prediction": "No",
                "evaluation": evaluate_answer("No", "no"),
            },
        ]
        evidence = [
            {
                "baseline_id": "stop-sign",
                "grounding": {"annotations": []},
            },
            {
                "baseline_id": "chair",
                "grounding": {"annotations": [{"score": 0.4}]},
            },
        ]
        semantic = [
            {
                "baseline_id": "chair",
                "candidate_key": "chair-0",
                "annotation_index": 0,
                "grounding_score": 0.4,
                "parsed_answer": "yes",
            }
        ]
        corrections = [
            {
                "policy_id": "v2",
                "id": "chair",
                "prediction": "yes",
                "correction": "harmful",
            }
        ]
        v3_reviews = [
            {"baseline_id": "chair", "selected_label": "chair"}
        ]
        v3_predictions = [
            {
                "id": "stop-sign",
                "prediction": "no",
                "changed": False,
                "correction": "unchanged",
            },
            {
                "id": "chair",
                "prediction": "yes",
                "changed": True,
                "correction": "harmful",
                "contrastive_selected_label": "chair",
            },
        ]

        analysis = build_failure_analysis(
            baseline,
            evidence,
            semantic,
            corrections,
            v3_reviews,
            v3_predictions,
            v2_policy_id="v2",
        )

        taxonomies = {item["id"]: item["taxonomy"] for item in analysis["cases"]}
        self.assertEqual(taxonomies["stop-sign"], "grounding_recall_miss")
        self.assertEqual(
            taxonomies["chair"], "category_ambiguity_false_accept"
        )
        report = markdown_report(
            {"decision": "retain"},
            build_variant_summary(
                self.stage38,
                self.stage39,
                v1_policy_id="v1",
                v2_rescue_policy_id="v2-rescue",
                v2_noop_policy_id="v2-noop",
            ),
            analysis,
        )
        self.assertIn("Do not claim", report)


if __name__ == "__main__":
    unittest.main()
