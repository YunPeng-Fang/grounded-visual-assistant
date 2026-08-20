"""Immutable lock records for the selected live-pipeline prompt policy."""

from __future__ import annotations

from typing import Any, Mapping


LOCK_PROTOCOL = "live_prompt_policy_lock_v1"
TEST_PROTOCOL = "live_pipeline_held_out_test_v1"


def build_locked_policy(
    comparison: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build the deterministic policy lock after all frozen gates pass."""
    if comparison.get("status") != "completed":
        raise RuntimeError("Prompt policy comparison is not completed.")
    coverage = comparison.get("coverage", {})
    if coverage.get("split") != "dev" or int(
        coverage.get("paired_questions", 0)
    ) != int(selection_manifest["sample_count"]):
        raise RuntimeError("Prompt policy lock requires complete paired Dev.")
    acceptance = comparison.get("acceptance", {})
    if not acceptance.get("all_gates_passed"):
        raise RuntimeError("Cannot lock a policy that failed acceptance gates.")

    selected_policy = str(selection_manifest["candidate"]["prompt_policy"])
    expected_decision = "accept_" + selected_policy.replace("-", "_")
    if acceptance.get("decision") != expected_decision:
        raise RuntimeError("Comparison decision does not select the candidate.")
    policies = comparison.get("policies", {})
    if (
        policies.get("baseline")
        != selection_manifest["baseline"]["prompt_policy"]
        or policies.get("candidate") != selected_policy
    ):
        raise RuntimeError("Comparison policies differ from the manifest.")

    return {
        "protocol": LOCK_PROTOCOL,
        "status": "locked",
        "selection_split": "dev",
        "selection_questions": int(selection_manifest["sample_count"]),
        "selected_prompt_policy": selected_policy,
        "prompt_template_sha256": selection_manifest["candidate"][
            "prompt_template_sha256"
        ],
        "selection_manifest_protocol": selection_manifest["protocol"],
        "acceptance": {
            "all_gates_passed": True,
            "decision": acceptance["decision"],
            "gates": acceptance["gates"],
        },
        "input_sha256": dict(sorted(input_sha256.items())),
    }


def build_test_protocol(
    locked_policy: Mapping[str, Any],
    *,
    locked_policy_path: str,
    locked_policy_sha256: str,
    dataset_path: str,
    dataset_sha256: str,
    split_path: str,
    split_sha256: str,
    selected_sample_ids_sha256: str,
    coco_ground_truth_path: str,
    coco_ground_truth_sha256: str,
    expected_images: int,
    expected_samples: int,
    run_name: str,
    runtime_files_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build the complete, no-partial-run held-out Test protocol."""
    if locked_policy.get("protocol") != LOCK_PROTOCOL:
        raise RuntimeError("Unsupported locked policy protocol.")
    if locked_policy.get("status") != "locked":
        raise RuntimeError("Prompt policy is not locked.")
    return {
        "protocol": TEST_PROTOCOL,
        "status": "locked",
        "split": "test",
        "allow_partial": False,
        "selected_prompt_policy": locked_policy["selected_prompt_policy"],
        "prompt_template_sha256": locked_policy["prompt_template_sha256"],
        "locked_policy": locked_policy_path,
        "locked_policy_sha256": locked_policy_sha256,
        "dataset": dataset_path,
        "dataset_sha256": dataset_sha256,
        "split_image_ids": split_path,
        "split_image_ids_sha256": split_sha256,
        "selected_sample_ids_sha256": selected_sample_ids_sha256,
        "coco_ground_truth": coco_ground_truth_path,
        "coco_ground_truth_sha256": coco_ground_truth_sha256,
        "expected_images": int(expected_images),
        "expected_samples": int(expected_samples),
        "run_name": run_name,
        "runtime_files_sha256": dict(sorted(runtime_files_sha256.items())),
    }


def render_locked_policy_report(
    locked_policy: Mapping[str, Any],
    test_protocol: Mapping[str, Any],
) -> str:
    """Render the policy decision and one-shot held-out protocol."""
    lines = [
        "# Locked Live-Pipeline Prompt Policy",
        "",
        f"Selected policy: `{locked_policy['selected_prompt_policy']}`",
        f"Selection split: `{locked_policy['selection_split']}`",
        f"Paired selection questions: {locked_policy['selection_questions']}",
        f"All pre-registered gates passed: "
        f"`{locked_policy['acceptance']['all_gates_passed']}`",
        "",
        "## Held-Out Test",
        "",
        f"Expected images: {test_protocol['expected_images']}",
        f"Expected questions: {test_protocol['expected_samples']}",
        f"Partial runs allowed: `{test_protocol['allow_partial']}`",
        f"Frozen run name: `{test_protocol['run_name']}`",
        "",
        "The Test run must use the locked protocol and may be resumed only with "
        "the identical command. Do not inspect partial metrics or change the "
        "prompt, model, thresholds, decoding, dataset, or evaluator.",
    ]
    return "\n".join(lines).rstrip() + "\n"
