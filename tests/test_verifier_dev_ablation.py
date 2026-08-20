from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_evaluation import evaluate_answer
from grounded_visual_assistant.semantic_answer_verifier import (
    semantic_candidate_key,
)
from grounded_visual_assistant.verifier_dev_ablation import (
    VerifierDevPolicy,
    build_policy_grid,
    evaluate_dev_policy,
    ordered_policy_ids_sha256,
    select_dev_policy,
)


def baseline(
    sample_id: str,
    *,
    prediction: str,
    target: str,
    role: str,
) -> dict:
    return {
        "id": sample_id,
        "pair_id": f"pair-{sample_id}",
        "pair_role": role,
        "image_id": 1,
        "image": "image.jpg",
        "question": f"Is there a {sample_id}?",
        "object": sample_id,
        "gt_answer": target,
        "prediction": prediction,
        "evaluation": evaluate_answer(prediction, target),
        "latency_seconds": 0.1,
        "cuda_peak_memory_allocated_gb": 4.0,
    }


def evidence(sample_id: str, score: float | None) -> dict:
    annotations = []
    if score is not None:
        annotations.append(
            {
                "class_name": sample_id,
                "bbox": [10, 10, 50, 50],
                "score": score,
                "mask_score": 0.9,
                "mask_area": 1000,
            }
        )
    return {
        "query_key": f"query-{sample_id}",
        "baseline_id": sample_id,
        "image": "image.jpg",
        "image_id": 1,
        "question": f"Is there a {sample_id}?",
        "object": sample_id,
        "grounding": {
            "img_width": 100,
            "img_height": 100,
            "annotations": annotations,
            "latency_seconds": {"total": 0.2},
        },
        "cuda_peak_memory_allocated_gb": 2.0,
    }


class VerifierDevAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            baseline(
                "book", prediction="No", target="yes", role="positive"
            ),
            baseline(
                "truck",
                prediction="No",
                target="no",
                role="hard_negative",
            ),
            baseline(
                "person", prediction="Yes", target="yes", role="positive"
            ),
            baseline(
                "chair",
                prediction="No",
                target="no",
                role="hard_negative",
            ),
        ]
        evidence_records = [
            evidence("book", 0.6),
            evidence("truck", 0.7),
            evidence("chair", None),
        ]
        self.evidence_by_id = {
            item["baseline_id"]: item for item in evidence_records
        }
        self.reviews = {
            semantic_candidate_key("query-book", 0): {
                "candidate_key": semantic_candidate_key("query-book", 0),
                "annotation_index": 0,
                "answer": "Yes",
                "latency_seconds": 0.05,
                "cuda_peak_memory_allocated_gb": 4.0,
            },
            semantic_candidate_key("query-truck", 0): {
                "candidate_key": semantic_candidate_key("query-truck", 0),
                "annotation_index": 0,
                "answer": "No",
                "latency_seconds": 0.05,
                "cuda_peak_memory_allocated_gb": 4.0,
            },
        }

    def test_builds_stable_unique_policy_grid(self) -> None:
        policies = build_policy_grid(
            score_thresholds=[0.3, 0.45],
            templates=[
                {
                    "family": "v1",
                    "module": "grounding",
                    "semantic_gate": False,
                    "max_mask_area_ratio": 1.0,
                },
                {
                    "family": "v2",
                    "module": "semantic",
                    "semantic_gate": True,
                    "max_mask_area_ratio": 0.9,
                    "max_candidates_per_query": 2,
                },
            ],
            min_mask_score=0.5,
            min_mask_area_ratio=0.0,
        )

        self.assertEqual(len(policies), 5)
        self.assertEqual(policies[0].policy_id, "baseline")
        self.assertEqual(len({item.policy_id for item in policies}), 5)
        self.assertTrue(ordered_policy_ids_sha256(policies))

    def test_semantic_gate_removes_grounding_harm(self) -> None:
        grounding_policy = VerifierDevPolicy(
            policy_id="v1",
            family="v1",
            module="grounding",
            score_threshold=0.3,
            min_mask_score=0.5,
            min_mask_area_ratio=0.0,
            max_mask_area_ratio=1.0,
            max_candidates_per_query=None,
            semantic_gate=False,
        )
        semantic_policy = VerifierDevPolicy(
            policy_id="v2",
            family="v2",
            module="semantic",
            score_threshold=0.3,
            min_mask_score=0.5,
            min_mask_area_ratio=0.0,
            max_mask_area_ratio=0.9,
            max_candidates_per_query=2,
            semantic_gate=True,
        )

        _, grounding = evaluate_dev_policy(
            grounding_policy,
            baseline_records=self.records,
            evidence_by_baseline_id=self.evidence_by_id,
            reviews_by_key=self.reviews,
        )
        predictions, semantic = evaluate_dev_policy(
            semantic_policy,
            baseline_records=self.records,
            evidence_by_baseline_id=self.evidence_by_id,
            reviews_by_key=self.reviews,
        )

        self.assertEqual(grounding["corrections"]["beneficial"], 1)
        self.assertEqual(grounding["corrections"]["harmful"], 1)
        self.assertEqual(semantic["corrections"]["beneficial"], 1)
        self.assertEqual(semantic["corrections"]["harmful"], 0)
        self.assertEqual(semantic["metrics"]["accuracy"], 1.0)
        self.assertEqual(
            next(item for item in predictions if item["id"] == "truck")[
                "prediction"
            ],
            "no",
        )

    def test_selection_locks_only_strict_improvement(self) -> None:
        baseline_summary = self._summary(
            "baseline", accuracy=0.75, f1=0.8, net=0, harmful=0
        )
        harmful = self._summary(
            "harmful", accuracy=0.75, f1=0.8, net=0, harmful=1
        )
        improved = self._summary(
            "improved", accuracy=1.0, f1=1.0, net=1, harmful=0
        )

        rows, decision = select_dev_policy(
            [baseline_summary, harmful, improved],
            require_strict_accuracy_improvement=True,
            require_non_decreasing_f1=True,
            require_positive_net_corrections=True,
        )

        self.assertEqual(
            decision["decision"], "lock_dev_selected_verifier"
        )
        self.assertEqual(decision["selected_policy_id"], "improved")
        self.assertFalse(rows[1]["selection"]["eligible"])
        self.assertTrue(rows[2]["selection"]["eligible"])

    def test_selection_falls_back_to_baseline(self) -> None:
        baseline_summary = self._summary(
            "baseline", accuracy=0.75, f1=0.8, net=0, harmful=0
        )
        candidate = self._summary(
            "candidate", accuracy=0.5, f1=0.6, net=-1, harmful=1
        )

        _, decision = select_dev_policy(
            [baseline_summary, candidate],
            require_strict_accuracy_improvement=True,
            require_non_decreasing_f1=True,
            require_positive_net_corrections=True,
        )

        self.assertEqual(
            decision["decision"],
            "retain_baseline_no_eligible_verifier",
        )
        self.assertEqual(decision["selected_policy_id"], "baseline")

    @staticmethod
    def _summary(
        policy_id: str,
        *,
        accuracy: float,
        f1: float,
        net: int,
        harmful: int,
    ) -> dict:
        return {
            "policy": {
                "policy_id": policy_id,
                "family": policy_id,
                "module": (
                    "baseline" if policy_id == "baseline" else "semantic"
                ),
            },
            "metrics": {
                "accuracy": accuracy,
                "f1": f1,
                "precision": f1,
            },
            "corrections": {
                "net_correct": net,
                "harmful": harmful,
            },
            "runtime_projection": {
                "incremental_latency_seconds": 0.0,
            },
        }


if __name__ == "__main__":
    unittest.main()
