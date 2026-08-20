"""Compare generic and task-aware live-pipeline prompt policies offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.live_prompt_policy_comparison import (
    compare_live_prompt_policies,
    render_live_prompt_policy_report,
)
from grounded_visual_assistant.live_pipeline_evaluation import (
    aggregate_live_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen generic and task-aware Dev runs."
    )
    parser.add_argument(
        "--manifest", default="configs/live_prompt_policy_v1.yaml"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


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
    predictions: list[dict[str, Any]],
    saved_metrics: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    expected_required = sum(
        bool(item.get("evidence_required")) for item in predictions
    )
    replayed = aggregate_live_pipeline(
        predictions,
        expected_samples=len(predictions),
        expected_required_evidence=expected_required,
        expected_negative_evidence=len(predictions) - expected_required,
        error_attempts=0,
        status="completed",
        iou_threshold=float(run_config["iou_threshold"]),
    )
    if without_generated_timestamps(replayed) != without_generated_timestamps(
        saved_metrics
    ):
        raise RuntimeError("Saved metrics do not match an independent replay.")


def validate_run(
    run_dir: Path,
    expected_policy: str,
    expected_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "run_config.json"
    for path in (predictions_path, metrics_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing run artifact: {path}")
    predictions = read_jsonl(predictions_path)
    metrics = read_json(metrics_path)
    run_config = read_json(config_path)
    policy = str(run_config.get("prompt_policy") or "generic-v1")
    if policy != expected_policy:
        raise RuntimeError(
            f"{run_dir.name} uses {policy}, expected {expected_policy}."
        )
    coverage = metrics.get("coverage", {})
    if (
        metrics.get("status") != "completed"
        or int(coverage.get("expected", -1)) != expected_samples
        or int(coverage.get("completed", -1)) != expected_samples
        or int(coverage.get("error_attempts", -1)) != 0
        or len(predictions) != expected_samples
    ):
        raise RuntimeError(f"Run is not complete and error-free: {run_dir}")
    return predictions, metrics, run_config


def main() -> None:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not str(manifest.get("protocol", "")).startswith(
        "live_prompt_policy_selection_v"
    ):
        raise RuntimeError(f"Unsupported manifest: {manifest_path}")
    expected_samples = int(manifest["sample_count"])
    baseline_spec = manifest["baseline"]
    candidate_spec = manifest["candidate"]
    baseline_dir = project_path(baseline_spec["run_dir"])
    candidate_dir = project_path(candidate_spec["run_dir"])

    baseline_predictions_path = baseline_dir / "predictions.jsonl"
    baseline_metrics_path = baseline_dir / "metrics.json"
    if sha256sum(baseline_predictions_path) != str(
        baseline_spec["predictions_sha256"]
    ):
        raise RuntimeError("Frozen baseline predictions hash changed.")
    if sha256sum(baseline_metrics_path) != str(
        baseline_spec["metrics_sha256"]
    ):
        raise RuntimeError("Frozen baseline metrics hash changed.")

    baseline, baseline_metrics, baseline_config = validate_run(
        baseline_dir,
        str(baseline_spec["prompt_policy"]),
        expected_samples,
    )
    candidate, candidate_metrics, candidate_config = validate_run(
        candidate_dir,
        str(candidate_spec["prompt_policy"]),
        expected_samples,
    )
    shared_fields = (
        "dataset_sha256",
        "split_image_ids_sha256",
        "selected_sample_ids_sha256",
        "coco_ground_truth_sha256",
        "demo_config_sha256",
        "vlm_config_sha256",
        "grounding_config_sha256",
        "model_id",
        "grounding_model_id",
        "sam2_checkpoint",
        "box_threshold",
        "text_threshold",
        "nms_iou_threshold",
        "iou_threshold",
    )
    differences = {
        field: {
            "baseline": baseline_config.get(field),
            "candidate": candidate_config.get(field),
        }
        for field in shared_fields
        if baseline_config.get(field) != candidate_config.get(field)
    }
    if differences:
        raise RuntimeError(f"Run configurations differ: {differences}")
    verify_metrics_replay(
        baseline, baseline_metrics, baseline_config
    )
    verify_metrics_replay(
        candidate, candidate_metrics, candidate_config
    )
    manifest_hash = sha256sum(manifest_path)
    if candidate_config.get("prompt_policy_manifest_sha256") != manifest_hash:
        raise RuntimeError("Candidate was not run from the current manifest.")
    expected_template_hash = candidate_spec.get("prompt_template_sha256")
    if (
        expected_template_hash is not None
        and candidate_config.get("prompt_template_sha256")
        != expected_template_hash
    ):
        raise RuntimeError(
            "Candidate run used a different prompt template hash."
        )

    summary, transitions = compare_live_prompt_policies(
        baseline,
        candidate,
        baseline_metrics,
        candidate_metrics,
        manifest["acceptance"],
        baseline_policy=str(baseline_spec["prompt_policy"]),
        candidate_policy=str(candidate_spec["prompt_policy"]),
    )
    summary["artifacts"] = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "baseline_run": str(baseline_dir),
        "baseline_predictions_sha256": sha256sum(
            baseline_dir / "predictions.jsonl"
        ),
        "candidate_run": str(candidate_dir),
        "candidate_predictions_sha256": sha256sum(
            candidate_dir / "predictions.jsonl"
        ),
        "candidate_metrics_sha256": sha256sum(
            candidate_dir / "metrics.json"
        ),
    }

    output_dir = project_path(
        args.output_dir
        or manifest.get("comparison_output_dir")
        or "outputs/eval_live_pipeline_v0/prompt_policy_dev_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "summary.json", summary)
    transition_path = output_dir / "paired_transitions.jsonl"
    temporary = transition_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in transitions
        ),
        encoding="utf-8",
    )
    temporary.replace(transition_path)
    (output_dir / "report.md").write_text(
        render_live_prompt_policy_report(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
