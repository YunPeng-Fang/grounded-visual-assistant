"""Freeze the complete live-pipeline Test240 result and failure audit."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.artifact_paths import resolve_project_path
from grounded_visual_assistant.live_pipeline_evaluation import (
    aggregate_live_pipeline,
)
from grounded_visual_assistant.live_test_reporting import (
    analyze_live_test_predictions,
    build_generalization_rows,
    relation_confusion_rows,
    render_live_test_report,
)


RUN_NAME = (
    "test__live-answer-grounding-v1__qwen3-vl-8b-instruct"
    "__box-0.30__text-0.30__task-aware-coco-v2__locked-v1"
)
TEST_RUN = PROJECT_ROOT / "outputs/eval_live_pipeline_v0" / RUN_NAME
OUTPUT_DIR = TEST_RUN / "final_report"
LOCK_DIR = PROJECT_ROOT / "outputs/eval_live_pipeline_v0/locked_policy_v1"
TEST_PROTOCOL_PATH = LOCK_DIR / "test_protocol.json"
SELECTED_POLICY_PATH = LOCK_DIR / "selected_policy.json"
DATASET_PATH = PROJECT_ROOT / "data/eval_v0/questions.jsonl"
TEST_SPLIT_PATH = PROJECT_ROOT / "data/eval_v0/splits/test_image_ids.json"
COCO_GT_PATH = PROJECT_ROOT / "data/eval_v0/coco_grounding_gt.json"
DEV_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs/eval_live_pipeline_v0/"
    "dev__live-answer-grounding-v1__qwen3-vl-8b-instruct"
    "__box-0.30__text-0.30__task-aware-coco-v2/metrics.json"
)
DEMO_CONFIG_PATH = PROJECT_ROOT / "configs/demo.yaml"
TASK_TYPES = ("object_listing", "object_existence", "spatial_relation")


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in records
    ).encode("utf-8")


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in rows:
        row = {key: item.get(key) for key in fields}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def expected_test_ids() -> list[str]:
    split = load_json(TEST_SPLIT_PATH)
    image_ids = [int(value) for value in split["image_ids"]]
    dataset = load_jsonl(DATASET_PATH)
    by_image_task = {
        (int(item["image_id"]), str(item["task_type"])): str(item["id"])
        for item in dataset
    }
    return [
        by_image_task[(image_id, task_type)]
        for image_id in image_ids
        for task_type in TASK_TYPES
    ]


def validate_and_replay() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    test_protocol = load_json(TEST_PROTOCOL_PATH)
    selected_policy = load_json(SELECTED_POLICY_PATH)
    run_config = load_json(TEST_RUN / "run_config.json")
    saved_metrics = load_json(TEST_RUN / "metrics.json")
    predictions = load_jsonl(TEST_RUN / "predictions.jsonl")
    errors = (TEST_RUN / "errors.jsonl").read_text(encoding="utf-8").strip()

    if (
        test_protocol.get("status") != "locked"
        or test_protocol.get("split") != "test"
        or test_protocol.get("allow_partial") is not False
        or test_protocol.get("expected_samples") != 240
        or test_protocol.get("expected_images") != 80
        or test_protocol.get("run_name") != RUN_NAME
    ):
        raise RuntimeError("Held-out Test protocol is not the locked Test240.")
    if (
        run_config.get("split") != "test"
        or run_config.get("prompt_policy")
        != test_protocol["selected_prompt_policy"]
        or run_config.get("test_protocol_sha256")
        != sha256sum(TEST_PROTOCOL_PATH)
        or run_config.get("locked_policy_sha256")
        != sha256sum(SELECTED_POLICY_PATH)
        or run_config.get("prompt_template_sha256")
        != test_protocol["prompt_template_sha256"]
    ):
        raise RuntimeError("Test run config differs from the locked protocol.")
    if (
        sha256sum(DATASET_PATH) != test_protocol["dataset_sha256"]
        or sha256sum(TEST_SPLIT_PATH)
        != test_protocol["split_image_ids_sha256"]
        or sha256sum(COCO_GT_PATH)
        != test_protocol["coco_ground_truth_sha256"]
    ):
        raise RuntimeError("A held-out Test input hash changed.")
    for relative_path, expected_hash in test_protocol[
        "runtime_files_sha256"
    ].items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file() or sha256sum(path) != expected_hash:
            raise RuntimeError(f"Locked runtime source changed: {path}")
    if (
        selected_policy.get("status") != "locked"
        or selected_policy.get("selected_prompt_policy")
        != test_protocol["selected_prompt_policy"]
    ):
        raise RuntimeError("Selected prompt policy is not locked.")
    coverage = saved_metrics.get("coverage", {})
    if (
        saved_metrics.get("status") != "completed"
        or int(coverage.get("expected", -1)) != 240
        or int(coverage.get("completed", -1)) != 240
        or int(coverage.get("remaining", -1)) != 0
        or int(coverage.get("error_attempts", -1)) != 0
        or errors
    ):
        raise RuntimeError("Locked Test run is incomplete or contains errors.")
    ids = [str(item["id"]) for item in predictions]
    if (
        len(ids) != 240
        or len(set(ids)) != 240
        or ids != expected_test_ids()
    ):
        raise RuntimeError("Locked Test prediction IDs differ from Test240.")

    missing_artifacts = []
    for item in predictions:
        for artifact_path, _ in item.get("grounding", {}).get(
            "artifacts", []
        ):
            path = resolve_project_path(artifact_path, PROJECT_ROOT)
            if not path.is_file():
                missing_artifacts.append(str(artifact_path))
    if missing_artifacts:
        raise RuntimeError(
            f"Missing Test artifacts; first: {missing_artifacts[0]}"
        )

    required = sum(bool(item.get("evidence_required")) for item in predictions)
    replayed = aggregate_live_pipeline(
        predictions,
        expected_samples=240,
        expected_required_evidence=required,
        expected_negative_evidence=240 - required,
        error_attempts=0,
        status="completed",
        iou_threshold=float(run_config["iou_threshold"]),
    )
    if without_generated_timestamps(replayed) != without_generated_timestamps(
        saved_metrics
    ):
        raise RuntimeError("Saved Test metrics differ from independent replay.")
    return predictions, replayed, saved_metrics, run_config


def main() -> None:
    predictions, replayed, saved_metrics, run_config = validate_and_replay()
    demo_config = yaml.safe_load(DEMO_CONFIG_PATH.read_text(encoding="utf-8"))
    max_new_tokens = int(demo_config["vlm"]["max_new_tokens"])
    failure_analysis, per_sample = analyze_live_test_predictions(
        predictions, max_new_tokens=max_new_tokens
    )
    dev_metrics = load_json(DEV_METRICS_PATH)
    generalization = build_generalization_rows(dev_metrics, replayed)
    box_metrics = replayed["required_evidence_box_metrics"]["box_iou_50"]
    mask_metrics = replayed["required_evidence_mask_iou_50"]
    result = {
        "overall": replayed["overall"],
        "tasks": replayed["tasks"],
        "structured_targets": replayed["structured_targets"],
        "box_micro_f1": box_metrics["micro_f1"],
        "mask_micro_f1": mask_metrics["micro_f1"],
        "negative_evidence_behavior": replayed[
            "negative_evidence_behavior"
        ],
        "end_to_end": replayed["end_to_end"]["overall"],
        "mean_latency_seconds": replayed["latency_seconds"]["mean"],
        "throughput_samples_per_second": replayed["latency_seconds"][
            "throughput_samples_per_second"
        ],
        "peak_cuda_memory_gb": replayed["cuda_memory_gb"][
            "peak_allocated_max"
        ],
    }
    summary = {
        "protocol": "locked_live_pipeline_test240_final_report_v1",
        "status": "finalized",
        "test_run_completed_at_utc": saved_metrics["generated_at_utc"],
        "integrity": {
            "coverage": 240,
            "prediction_errors": 0,
            "duplicate_ids": 0,
            "missing_artifacts": 0,
            "test_protocol_hash_verified": True,
            "selected_policy_hash_verified": True,
            "dataset_hash_verified": True,
            "split_hash_verified": True,
            "coco_ground_truth_hash_verified": True,
            "runtime_source_hashes_verified": True,
            "saved_metrics_replayed": True,
            "post_test_tuning": "prohibited",
        },
        "test_result": result,
        "generalization": generalization,
        "failure_analysis": failure_analysis,
        "decision": {
            "policy_remains_frozen": True,
            "test_driven_retuning_allowed": False,
            "reported_limitation": (
                "one object-listing response reached the generation limit "
                "and produced truncated JSON"
            ),
        },
    }

    per_sample_fields = [
        "id",
        "image_id",
        "task_type",
        "gt_answer",
        "prediction",
        "answer_score",
        "answer_correct",
        "answer_parse_valid",
        "schema_valid",
        "parse_source",
        "generated_tokens",
        "hit_max_new_tokens",
        "targets",
        "target_precision",
        "target_recall",
        "target_f1",
        "target_fp",
        "target_fn",
        "box_precision",
        "box_recall",
        "box_f1",
        "box_fp",
        "box_fn",
        "mask_precision",
        "mask_recall",
        "mask_f1",
        "mask_fp",
        "mask_fn",
        "evidence_required",
        "evidence_supported",
        "evidence_complete",
        "end_to_end_success",
        "end_to_end_complete_success",
        "latency_seconds",
        "flags",
        "severity",
    ]
    artifacts = {
        "summary.json": json_bytes(summary),
        "report.md": render_live_test_report(summary).encode("utf-8"),
        "per_sample_analysis.jsonl": jsonl_bytes(per_sample),
        "per_sample_analysis.csv": csv_bytes(
            per_sample, per_sample_fields
        ),
        "generalization.csv": csv_bytes(
            generalization,
            ["metric", "dev", "test", "delta_test_minus_dev"],
        ),
        "relation_confusion.csv": csv_bytes(
            relation_confusion_rows(replayed),
            ["target", "prediction", "count"],
        ),
    }
    input_paths = {
        "test_protocol": TEST_PROTOCOL_PATH,
        "selected_policy": SELECTED_POLICY_PATH,
        "dataset": DATASET_PATH,
        "test_split": TEST_SPLIT_PATH,
        "coco_ground_truth": COCO_GT_PATH,
        "dev_metrics": DEV_METRICS_PATH,
        "demo_config": DEMO_CONFIG_PATH,
        "test_predictions": TEST_RUN / "predictions.jsonl",
        "test_metrics": TEST_RUN / "metrics.json",
        "test_run_config": TEST_RUN / "run_config.json",
        "test_errors": TEST_RUN / "errors.jsonl",
    }
    manifest = {
        "protocol": summary["protocol"],
        "immutable": True,
        "input_sha256": {
            name: sha256sum(path)
            for name, path in sorted(input_paths.items())
        },
        "artifact_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(artifacts.items())
        },
    }
    artifacts["manifest.json"] = json_bytes(manifest)

    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists():
        for filename, payload in artifacts.items():
            path = OUTPUT_DIR / filename
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"Final Test report differs: {path}")
        status = "verified"
    else:
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            raise RuntimeError(f"Refusing non-empty output directory: {OUTPUT_DIR}")
        for filename, payload in artifacts.items():
            write_atomic(OUTPUT_DIR / filename, payload)
        status = "created"
    print(f"Locked live-pipeline Test240 report: {status}")
    print(json.dumps(summary["integrity"], ensure_ascii=False, indent=2))
    print(f"Report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
