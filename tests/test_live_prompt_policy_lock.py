from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.live_prompt_policy_lock import (
    LOCK_PROTOCOL,
    TEST_PROTOCOL,
    build_locked_policy,
    build_test_protocol,
)


def comparison(*, accepted: bool = True) -> dict:
    return {
        "status": "completed",
        "policies": {
            "baseline": "task-aware-coco-v1",
            "candidate": "task-aware-coco-v2",
        },
        "coverage": {"split": "dev", "paired_questions": 60},
        "acceptance": {
            "all_gates_passed": accepted,
            "decision": (
                "accept_task_aware_coco_v2"
                if accepted
                else "reject_or_revise_candidate"
            ),
            "gates": {"schema_valid_rate_min": {"passed": accepted}},
        },
    }


def manifest() -> dict:
    return {
        "protocol": "live_prompt_policy_selection_v2",
        "sample_count": 60,
        "baseline": {"prompt_policy": "task-aware-coco-v1"},
        "candidate": {
            "prompt_policy": "task-aware-coco-v2",
            "prompt_template_sha256": "template-hash",
        },
    }


class LivePromptPolicyLockTest(unittest.TestCase):
    def test_builds_lock_and_complete_test_protocol(self) -> None:
        locked = build_locked_policy(
            comparison(), manifest(), {"candidate_predictions": "abc"}
        )
        self.assertEqual(locked["protocol"], LOCK_PROTOCOL)
        self.assertEqual(
            locked["selected_prompt_policy"], "task-aware-coco-v2"
        )
        protocol = build_test_protocol(
            locked,
            locked_policy_path="outputs/selected_policy.json",
            locked_policy_sha256="lock-hash",
            dataset_path="data/questions.jsonl",
            dataset_sha256="dataset-hash",
            split_path="data/test_ids.json",
            split_sha256="split-hash",
            selected_sample_ids_sha256="sample-hash",
            coco_ground_truth_path="data/coco.json",
            coco_ground_truth_sha256="coco-hash",
            expected_images=80,
            expected_samples=240,
            run_name="locked-test-run",
            runtime_files_sha256={"scripts/runner.py": "runner-hash"},
        )
        self.assertEqual(protocol["protocol"], TEST_PROTOCOL)
        self.assertFalse(protocol["allow_partial"])
        self.assertEqual(protocol["expected_samples"], 240)
        self.assertEqual(protocol["run_name"], "locked-test-run")

    def test_rejects_failed_selection(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "failed acceptance"):
            build_locked_policy(
                comparison(accepted=False),
                manifest(),
                {"candidate_predictions": "abc"},
            )

    def test_rejects_unlocked_policy_for_test(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not locked"):
            build_test_protocol(
                {
                    "protocol": LOCK_PROTOCOL,
                    "status": "draft",
                    "selected_prompt_policy": "task-aware-coco-v2",
                    "prompt_template_sha256": "hash",
                },
                locked_policy_path="selected.json",
                locked_policy_sha256="hash",
                dataset_path="questions.jsonl",
                dataset_sha256="hash",
                split_path="test.json",
                split_sha256="hash",
                selected_sample_ids_sha256="hash",
                coco_ground_truth_path="coco.json",
                coco_ground_truth_sha256="hash",
                expected_images=80,
                expected_samples=240,
                run_name="test",
                runtime_files_sha256={"scripts/runner.py": "runner-hash"},
            )


if __name__ == "__main__":
    unittest.main()
