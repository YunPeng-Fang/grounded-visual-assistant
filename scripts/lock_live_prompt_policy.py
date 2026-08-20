"""Freeze accepted task-aware COCO v2 and build the held-out Test protocol."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.live_prompt_policy_lock import (
    build_locked_policy,
    build_test_protocol,
    render_locked_policy_report,
)
from grounded_visual_assistant.live_pipeline_evaluation import (
    aggregate_live_pipeline,
)
from grounded_visual_assistant.live_prompt_policy_comparison import (
    compare_live_prompt_policies,
)


MANIFEST_PATH = PROJECT_ROOT / "configs/live_prompt_policy_v2.yaml"
COMPARISON_DIR = (
    PROJECT_ROOT
    / "outputs/eval_live_pipeline_v0/prompt_policy_v1_vs_v2_dev"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/eval_live_pipeline_v0/locked_policy_v1"
)
DATASET_PATH = PROJECT_ROOT / "data/eval_v0/questions.jsonl"
TEST_SPLIT_PATH = PROJECT_ROOT / "data/eval_v0/splits/test_image_ids.json"
COCO_GT_PATH = PROJECT_ROOT / "data/eval_v0/coco_grounding_gt.json"
PROMPT_TEMPLATE_PATH = (
    PROJECT_ROOT
    / "src/grounded_visual_assistant/live_pipeline_prompting.py"
)
TEST_RUNTIME_PATHS = {
    "scripts/batch_eval_live_pipeline.py": (
        PROJECT_ROOT / "scripts/batch_eval_live_pipeline.py"
    ),
    "src/grounded_visual_assistant/artifact_paths.py": (
        PROJECT_ROOT / "src/grounded_visual_assistant/artifact_paths.py"
    ),
    "src/grounded_visual_assistant/demo.py": (
        PROJECT_ROOT / "src/grounded_visual_assistant/demo.py"
    ),
    "src/grounded_visual_assistant/live_pipeline_evaluation.py": (
        PROJECT_ROOT
        / "src/grounded_visual_assistant/live_pipeline_evaluation.py"
    ),
}
TASK_TYPES = ("object_listing", "object_existence", "spatial_relation")


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_ids_sha256(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return sha256bytes(payload.encode("utf-8"))


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def without_generated_timestamps(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_generated_timestamps(item)
            for key, item in value.items()
            if key != "generated_at_utc"
        }
    if isinstance(value, list):
        return [without_generated_timestamps(item) for item in value]
    return value


def verify_metrics_replay(
    records: list[dict[str, Any]],
    saved_metrics: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    required = sum(bool(item.get("evidence_required")) for item in records)
    replayed = aggregate_live_pipeline(
        records,
        expected_samples=len(records),
        expected_required_evidence=required,
        expected_negative_evidence=len(records) - required,
        error_attempts=0,
        status="completed",
        iou_threshold=float(run_config["iou_threshold"]),
    )
    if without_generated_timestamps(replayed) != without_generated_timestamps(
        saved_metrics
    ):
        raise RuntimeError("Saved metrics differ from the independent replay.")


def selected_test_ids() -> tuple[list[str], int]:
    split = json.loads(TEST_SPLIT_PATH.read_text(encoding="utf-8"))
    image_ids = [int(value) for value in split["image_ids"]]
    records = load_jsonl(DATASET_PATH)
    by_image_task = {
        (int(item["image_id"]), str(item["task_type"])): str(item["id"])
        for item in records
    }
    selected = []
    for image_id in image_ids:
        for task_type in TASK_TYPES:
            key = (image_id, task_type)
            if key not in by_image_task:
                raise RuntimeError(f"Test split misses sample {key}.")
            selected.append(by_image_task[key])
    if len(selected) != len(set(selected)):
        raise RuntimeError("Held-out Test contains duplicate sample IDs.")
    return selected, len(image_ids)


def validate_comparison(
    manifest: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Path]:
    if comparison.get("acceptance", {}).get("all_gates_passed") is not True:
        raise RuntimeError("The v2 comparison did not pass every gate.")
    baseline_dir = PROJECT_ROOT / manifest["baseline"]["run_dir"]
    candidate_dir = PROJECT_ROOT / manifest["candidate"]["run_dir"]
    paths = {
        "selection_manifest": MANIFEST_PATH,
        "comparison_summary": COMPARISON_DIR / "summary.json",
        "comparison_transitions": COMPARISON_DIR
        / "paired_transitions.jsonl",
        "baseline_predictions": baseline_dir / "predictions.jsonl",
        "baseline_metrics": baseline_dir / "metrics.json",
        "baseline_run_config": baseline_dir / "run_config.json",
        "baseline_errors": baseline_dir / "errors.jsonl",
        "candidate_predictions": candidate_dir / "predictions.jsonl",
        "candidate_metrics": candidate_dir / "metrics.json",
        "candidate_run_config": candidate_dir / "run_config.json",
        "candidate_errors": candidate_dir / "errors.jsonl",
        "prompt_template": PROMPT_TEMPLATE_PATH,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing lock input: {path}")
    if sha256sum(paths["baseline_predictions"]) != manifest["baseline"][
        "predictions_sha256"
    ]:
        raise RuntimeError("Frozen v1 predictions hash changed.")
    if sha256sum(paths["baseline_metrics"]) != manifest["baseline"][
        "metrics_sha256"
    ]:
        raise RuntimeError("Frozen v1 metrics hash changed.")
    if sha256sum(PROMPT_TEMPLATE_PATH) != manifest["candidate"][
        "prompt_template_sha256"
    ]:
        raise RuntimeError("Selected prompt template hash changed.")
    if (
        paths["baseline_errors"].read_text(encoding="utf-8").strip()
        or paths["candidate_errors"].read_text(encoding="utf-8").strip()
    ):
        raise RuntimeError("A policy-selection run contains errors.")
    baseline_records = load_jsonl(paths["baseline_predictions"])
    candidate_records = load_jsonl(paths["candidate_predictions"])
    baseline_metrics = json.loads(
        paths["baseline_metrics"].read_text(encoding="utf-8")
    )
    candidate_metrics = json.loads(
        paths["candidate_metrics"].read_text(encoding="utf-8")
    )
    baseline_config = json.loads(
        paths["baseline_run_config"].read_text(encoding="utf-8")
    )
    candidate_config = json.loads(
        paths["candidate_run_config"].read_text(encoding="utf-8")
    )
    if (
        candidate_config.get("prompt_policy")
        != manifest["candidate"]["prompt_policy"]
        or candidate_config.get("prompt_policy_manifest_sha256")
        != sha256sum(MANIFEST_PATH)
    ):
        raise RuntimeError("Candidate run does not match the v2 manifest.")
    coverage = candidate_metrics.get("coverage", {})
    if (
        candidate_metrics.get("status") != "completed"
        or int(coverage.get("expected", -1)) != int(manifest["sample_count"])
        or int(coverage.get("completed", -1)) != int(manifest["sample_count"])
        or int(coverage.get("error_attempts", -1)) != 0
    ):
        raise RuntimeError("Selected v2 run is incomplete or has errors.")
    verify_metrics_replay(
        baseline_records, baseline_metrics, baseline_config
    )
    verify_metrics_replay(
        candidate_records, candidate_metrics, candidate_config
    )
    replayed_comparison, _ = compare_live_prompt_policies(
        baseline_records,
        candidate_records,
        baseline_metrics,
        candidate_metrics,
        manifest["acceptance"],
        baseline_policy=manifest["baseline"]["prompt_policy"],
        candidate_policy=manifest["candidate"]["prompt_policy"],
    )
    saved_core = {
        key: value
        for key, value in comparison.items()
        if key != "artifacts"
    }
    if replayed_comparison != saved_core:
        raise RuntimeError("Saved comparison differs from paired replay.")
    return paths


def main() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    comparison = json.loads(
        (COMPARISON_DIR / "summary.json").read_text(encoding="utf-8")
    )
    input_paths = validate_comparison(manifest, comparison)
    input_sha256 = {
        name: sha256sum(path) for name, path in sorted(input_paths.items())
    }
    locked_policy = build_locked_policy(
        comparison, manifest, input_sha256
    )
    locked_policy_payload = json_bytes(locked_policy)
    selected_ids, image_count = selected_test_ids()
    test_protocol = build_test_protocol(
        locked_policy,
        locked_policy_path=(
            "outputs/eval_live_pipeline_v0/locked_policy_v1/"
            "selected_policy.json"
        ),
        locked_policy_sha256=sha256bytes(locked_policy_payload),
        dataset_path="data/eval_v0/questions.jsonl",
        dataset_sha256=sha256sum(DATASET_PATH),
        split_path="data/eval_v0/splits/test_image_ids.json",
        split_sha256=sha256sum(TEST_SPLIT_PATH),
        selected_sample_ids_sha256=stable_ids_sha256(selected_ids),
        coco_ground_truth_path="data/eval_v0/coco_grounding_gt.json",
        coco_ground_truth_sha256=sha256sum(COCO_GT_PATH),
        expected_images=image_count,
        expected_samples=len(selected_ids),
        run_name=(
            "test__live-answer-grounding-v1__qwen3-vl-8b-instruct"
            "__box-0.30__text-0.30__task-aware-coco-v2__locked-v1"
        ),
        runtime_files_sha256={
            name: sha256sum(path)
            for name, path in TEST_RUNTIME_PATHS.items()
        },
    )
    artifacts = {
        "selected_policy.json": locked_policy_payload,
        "test_protocol.json": json_bytes(test_protocol),
        "report.md": render_locked_policy_report(
            locked_policy, test_protocol
        ).encode("utf-8"),
    }
    selected_policy_path = OUTPUT_DIR / "selected_policy.json"
    if selected_policy_path.exists():
        for filename, payload in artifacts.items():
            path = OUTPUT_DIR / filename
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"Locked artifact differs: {path}")
        status = "verified"
    else:
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            raise RuntimeError(f"Refusing non-empty lock directory: {OUTPUT_DIR}")
        for filename, payload in artifacts.items():
            write_atomic(OUTPUT_DIR / filename, payload)
        status = "created"
    print(f"Live prompt policy lock: {status}")
    print(json.dumps(test_protocol, ensure_ascii=False, indent=2))
    print(f"Report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
