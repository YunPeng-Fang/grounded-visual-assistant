"""Freeze the complete Hard-Test400 result and read-only failure analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_benchmark import read_jsonl
from grounded_visual_assistant.hard_dataset import (
    OPEN_IMAGES_SOURCE,
    VISUAL_GENOME_SOURCE,
)
from grounded_visual_assistant.hard_test_reporting import (
    build_generalization_rows,
    render_locked_hard_test_report,
)
from grounded_visual_assistant.hard_vlm_analysis import (
    analyze_hard_vlm_predictions,
)


DATASET = (
    PROJECT_ROOT
    / "data/cross_dataset_hard_v1/questions_locked_test_v1/"
    "test_questions.jsonl"
)
DATASET_MANIFEST = DATASET.parent / "manifest.json"
POLICY_DIR = (
    PROJECT_ROOT
    / "outputs/cross_dataset_hard_v1/relation_prompt_policy_dev_v1"
)
POLICY = POLICY_DIR / "selected_policy.json"
POLICY_SUMMARY = POLICY_DIR / "summary.json"
DEV_V1_RUN = (
    PROJECT_ROOT
    / "outputs/cross_dataset_hard_v1/vlm/"
    "hard-dev400__qwen3-vl-8b-instruct"
)
DEV_V2_RUN = (
    PROJECT_ROOT
    / "outputs/cross_dataset_hard_v1/vlm/"
    "hard-dev200-relation__qwen3-vl-8b-instruct__prompt-v2"
)
DEV_V3_RUN = (
    PROJECT_ROOT
    / "outputs/cross_dataset_hard_v1/vlm/"
    "hard-dev100-vg-relation__qwen3-vl-8b-instruct__prompt-v3"
)
TEST_RUN = (
    PROJECT_ROOT
    / "outputs/cross_dataset_hard_v1/vlm/"
    "hard-test400__qwen3-vl-8b-instruct__source-aware-policy-v1"
)
OUTPUT_DIR = TEST_RUN / "final_report"


def sha256sum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in records
    ).encode("utf-8")


def csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for item in rows:
        row = {key: item.get(key) for key in fieldnames}
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


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_manifest = load_json(DATASET_MANIFEST)
    policy = load_json(POLICY)
    run_config = load_json(TEST_RUN / "run_config.json")
    saved_metrics = load_json(TEST_RUN / "metrics.json")
    errors = (TEST_RUN / "errors.jsonl").read_text(encoding="utf-8").strip()
    dataset_hash = sha256sum(DATASET)
    manifest_hash = sha256sum(DATASET_MANIFEST)
    policy_hash = sha256sum(POLICY)
    if dataset_manifest.get("split") != "test":
        raise RuntimeError("Locked dataset manifest is not Test.")
    if dataset_manifest["artifact_sha256"]["test_questions.jsonl"] != dataset_hash:
        raise RuntimeError("Locked Test dataset hash differs from its manifest.")
    if dataset_manifest["input_sha256"]["selected_policy"] != policy_hash:
        raise RuntimeError("Locked Test policy hash differs from its manifest.")
    if (
        policy.get("status") != "locked"
        or policy.get("selected_on_split") != "dev"
        or not policy.get("immutable")
    ):
        raise RuntimeError("Selected relation policy is not locked from Dev.")
    if (
        run_config.get("dataset_sha256") != dataset_hash
        or run_config.get("dataset_manifest_sha256") != manifest_hash
        or run_config.get("required_split") != "test"
        or run_config.get("task_type") != "all"
        or int(run_config.get("max_new_tokens", 0)) != 64
        or bool(run_config.get("do_sample"))
    ):
        raise RuntimeError("Test run config differs from the locked protocol.")
    coverage = saved_metrics.get("coverage", {})
    if (
        saved_metrics.get("status") != "completed"
        or int(coverage.get("expected", 0)) != 400
        or int(coverage.get("completed", 0)) != 400
        or int(coverage.get("remaining", -1)) != 0
        or int(coverage.get("error_attempts", -1)) != 0
        or errors
    ):
        raise RuntimeError("Test run is incomplete or contains errors.")

    test_model_keys = (
        "model_id",
        "torch_dtype",
        "max_new_tokens",
        "do_sample",
    )
    for dev_run in (DEV_V1_RUN, DEV_V2_RUN, DEV_V3_RUN):
        dev_config = load_json(dev_run / "run_config.json")
        if any(
            dev_config.get(key) != run_config.get(key)
            for key in test_model_keys
        ):
            raise RuntimeError(f"Dev/Test model config differs: {dev_run}")
    return dataset_manifest, policy, saved_metrics


def metric_replay_matches(
    replay: dict[str, Any], saved: dict[str, Any]
) -> bool:
    return all(
        replay[key] == saved[key]
        for key in ("overall", "tasks", "sources", "split_counts")
    )


def dev_reference() -> dict[str, Any]:
    baseline = load_json(DEV_V1_RUN / "metrics.json")
    selection = load_json(POLICY_SUMMARY)
    comparisons = selection["comparisons"]
    open_images = comparisons["v1_vs_v2_all_sources"]["sources"][
        OPEN_IMAGES_SOURCE
    ]["candidate"]
    visual_genome = comparisons["visual_genome_v1_vs_v3"]["overall"][
        "candidate"
    ]
    return {
        "object_existence": {
            "exact_accuracy": baseline["tasks"]["object_existence"][
                "exact_accuracy"
            ]
        },
        "object_listing": {
            "macro_f1": baseline["tasks"]["object_listing"]["macro_f1"]
        },
        "relations": {
            OPEN_IMAGES_SOURCE: open_images,
            VISUAL_GENOME_SOURCE: visual_genome,
        },
    }


def confusion_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in (OPEN_IMAGES_SOURCE, VISUAL_GENOME_SOURCE):
        confusion = metrics["sources"][source]["tasks"]["spatial_relation"][
            "confusion"
        ]
        for target, predictions in confusion.items():
            for prediction, count in predictions.items():
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "prediction": prediction,
                        "count": count,
                    }
                )
    return rows


def main() -> None:
    _, _, saved_metrics = validate_inputs()
    dataset = read_jsonl(DATASET)
    predictions = read_jsonl(TEST_RUN / "predictions.jsonl")
    analysis, per_sample = analyze_hard_vlm_predictions(
        dataset, predictions, required_split="test"
    )
    analysis.pop("generated_at_utc", None)
    replay_metrics = {
        "overall": analysis["overall"],
        "tasks": analysis["tasks"],
        "sources": analysis["sources"],
        "split_counts": {"test": 400},
    }
    if not metric_replay_matches(replay_metrics, saved_metrics):
        raise RuntimeError("Saved Test metrics do not reproduce from predictions.")
    token_hits = sum(bool(item["hit_max_new_tokens"]) for item in per_sample)
    if token_hits:
        raise RuntimeError("Locked Test predictions contain token-limit hits.")

    generalization = build_generalization_rows(
        dev_reference(), replay_metrics
    )
    summary = {
        "protocol": "locked_hard_test400_final_report_v1",
        "status": "finalized",
        "test_run_completed_at_utc": saved_metrics["generated_at_utc"],
        "integrity": {
            "coverage": 400,
            "prediction_errors": 0,
            "duplicate_ids": 0,
            "token_limit_hits": 0,
            "dataset_hash_verified": True,
            "manifest_hash_verified": True,
            "policy_hash_verified": True,
            "model_config_matches_dev": True,
            "saved_metrics_replayed": True,
            "post_test_tuning": "prohibited",
        },
        "test_result": {
            "overall": replay_metrics["overall"],
            "tasks": replay_metrics["tasks"],
            "sources": replay_metrics["sources"],
            "latency_seconds": saved_metrics["latency_seconds"],
            "cuda_memory_gb": saved_metrics["cuda_memory_gb"],
        },
        "generalization": generalization,
        "failure_analysis": {
            "existence_by_polarity": analysis["existence_by_polarity"],
            "listing_protocols": analysis["listing_protocols"],
            "relation_sources": analysis["relation_sources"],
            "failure_flags": analysis["failure_flags"],
            "warnings": analysis["warnings"],
            "top_failures": analysis["top_failures"],
        },
        "decision": {
            "policy_remains_frozen": True,
            "test_driven_retuning_allowed": False,
            "primary_held_out_weakness": (
                "visual_genome_semantic_spatial_relation_generalization"
            ),
        },
    }

    per_sample_fields = [
        "id",
        "sample_id",
        "source",
        "split",
        "task_type",
        "gt_answer",
        "parsed_prediction",
        "score",
        "is_correct",
        "parse_valid",
        "hit_max_new_tokens",
        "generated_tokens",
        "missed_categories",
        "extra_categories",
        "flags",
        "severity",
    ]
    artifacts = {
        "summary.json": json_bytes(summary),
        "report.md": render_locked_hard_test_report(summary).encode("utf-8"),
        "per_sample_analysis.jsonl": jsonl_bytes(per_sample),
        "per_sample_analysis.csv": csv_bytes(
            per_sample, per_sample_fields
        ),
        "generalization.csv": csv_bytes(
            generalization,
            ["scope", "metric", "dev", "test", "delta_test_minus_dev"],
        ),
        "relation_confusion.csv": csv_bytes(
            confusion_rows(replay_metrics),
            ["source", "target", "prediction", "count"],
        ),
    }
    input_paths = {
        "dataset": DATASET,
        "dataset_manifest": DATASET_MANIFEST,
        "selected_policy": POLICY,
        "policy_summary": POLICY_SUMMARY,
        "dev_v1_metrics": DEV_V1_RUN / "metrics.json",
        "dev_v2_metrics": DEV_V2_RUN / "metrics.json",
        "dev_v3_metrics": DEV_V3_RUN / "metrics.json",
        "test_predictions": TEST_RUN / "predictions.jsonl",
        "test_run_config": TEST_RUN / "run_config.json",
        "test_metrics": TEST_RUN / "metrics.json",
        "test_errors": TEST_RUN / "errors.jsonl",
    }
    report_manifest = {
        "protocol": summary["protocol"],
        "immutable": True,
        "input_sha256": {
            key: sha256sum(path) for key, path in sorted(input_paths.items())
        },
        "artifact_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(artifacts.items())
        },
    }
    artifacts["manifest.json"] = json_bytes(report_manifest)

    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists():
        for relative_path, payload in artifacts.items():
            path = OUTPUT_DIR / relative_path
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"Final Test report differs: {path}")
        status = "verified"
    else:
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            raise RuntimeError(f"Refusing non-empty output directory: {OUTPUT_DIR}")
        for relative_path, payload in artifacts.items():
            write_atomic(OUTPUT_DIR / relative_path, payload)
        status = "created"

    print(f"Locked Hard-Test final report: {status}")
    print(json.dumps(summary["integrity"], ensure_ascii=False, indent=2))
    print(f"Report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
