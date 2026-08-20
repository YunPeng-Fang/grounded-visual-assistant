"""Cache Grounded-SAM-2 evidence for frozen negative Dev110 answers."""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_eval_pope_verifier import (
    append_jsonl,
    count_jsonl,
    load_evidence,
    project_path,
    utc_now,
    write_json_atomic,
)
from grounded_visual_assistant.grounding_answer_verifier import (
    compact_grounding_result,
)
from grounded_visual_assistant.pope_dataset import (
    read_json_records,
    sha256sum,
)
from grounded_visual_assistant.verifier_dev_evaluation import (
    VERIFIER_DEV_BASELINE_PROTOCOL,
)
from grounded_visual_assistant.verifier_dev_grounding import (
    VERIFIER_DEV_GROUNDING_PROTOCOL,
    aggregate_verifier_dev_grounding_metrics,
    build_negative_grounding_jobs,
    ordered_query_keys_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Grounding DINO + SAM 2.1 only for strict No answers from "
            "the frozen Verifier Dev110 Qwen baseline."
        )
    )
    parser.add_argument(
        "--config", default="configs/verifier_dev_grounding_v1.yaml"
    )
    parser.add_argument("--baseline-predictions", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--grounding-model-id", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-config", default=None)
    parser.add_argument("--box-threshold", type=float, default=None)
    parser.add_argument("--text-threshold", type=float, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default=None,
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--visualize-limit", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.save_every <= 0:
        parser.error("--save-every must be positive.")
    if args.max_errors < 0 or args.visualize_limit < 0:
        parser.error(
            "--max-errors and --visualize-limit must be zero or greater."
        )
    return args


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")
    return slug.lower() or "run"


def validate_baseline_source(
    predictions_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics_path = predictions_path.parent / "metrics.json"
    config_path = predictions_path.parent / "run_config.json"
    for path in (predictions_path, metrics_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Dev baseline artifact missing: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    if metrics.get("protocol") != VERIFIER_DEV_BASELINE_PROTOCOL:
        raise RuntimeError("Dev baseline metrics protocol mismatch.")
    if metrics.get("status") != "completed":
        raise RuntimeError("Dev baseline metrics are not completed.")
    coverage = metrics.get("coverage") or {}
    if coverage.get("completed") != 110 or coverage.get("remaining") != 0:
        raise RuntimeError("Dev baseline must contain all 110 predictions.")
    if run_config.get("protocol") != VERIFIER_DEV_BASELINE_PROTOCOL:
        raise RuntimeError("Dev baseline run config protocol mismatch.")
    records = read_json_records(predictions_path)
    if len(records) != 110:
        raise RuntimeError(
            f"Dev baseline predictions must contain 110 rows, found "
            f"{len(records)}."
        )
    return records, {
        "baseline_predictions": str(predictions_path),
        "baseline_predictions_sha256": sha256sum(predictions_path),
        "baseline_metrics": str(metrics_path),
        "baseline_metrics_sha256": sha256sum(metrics_path),
        "baseline_run_config": str(config_path),
        "baseline_run_config_sha256": sha256sum(config_path),
        "baseline_model_id": run_config.get("model_id"),
    }


def load_settings(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    config_path = project_path(args.config)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != VERIFIER_DEV_GROUNDING_PROTOCOL:
        raise ValueError(
            f"Unsupported Dev grounding protocol: {payload.get('protocol')}"
        )
    inputs = dict(payload["inputs"])
    grounding_entry = dict(payload["grounding"])
    runtime_entry = dict(payload["runtime"])
    grounding_config_path = project_path(grounding_entry["config"])
    grounding_yaml = yaml.safe_load(
        grounding_config_path.read_text(encoding="utf-8")
    )
    grounding = dict(grounding_yaml["grounding"])
    sam2 = dict(grounding_yaml["sam2"])
    runtime = dict(grounding_yaml["runtime"])
    baseline_path = project_path(
        args.baseline_predictions or inputs["baseline_predictions"]
    )
    output_dir = project_path(
        args.output_dir or runtime_entry["output_dir"]
    )
    settings = {
        "grounding_model_id": (
            args.grounding_model_id or grounding["model_id"]
        ),
        "sam2_checkpoint": args.sam2_checkpoint or sam2["checkpoint"],
        "sam2_model_config": (
            args.sam2_model_config or sam2["model_config"]
        ),
        "box_threshold": (
            args.box_threshold
            if args.box_threshold is not None
            else float(grounding_entry["box_threshold"])
        ),
        "text_threshold": (
            args.text_threshold
            if args.text_threshold is not None
            else float(grounding_entry["text_threshold"])
        ),
        "nms_iou_threshold": (
            args.nms_iou_threshold
            if args.nms_iou_threshold is not None
            else grounding_entry.get("nms_iou_threshold")
        ),
        "device": args.device or runtime.get("device", "cuda"),
        "dtype": args.dtype or runtime.get("dtype", "float16"),
        "local_files_only": bool(
            args.local_files_only
            or grounding.get("local_files_only", False)
        ),
    }
    for name in ("box_threshold", "text_threshold"):
        value = float(settings[name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    sources = {
        "dev_grounding_config": str(config_path),
        "dev_grounding_config_sha256": sha256sum(config_path),
        "grounded_sam2_config": str(grounding_config_path),
        "grounded_sam2_config_sha256": sha256sum(
            grounding_config_path
        ),
    }
    return baseline_path, output_dir, settings, sources


def preflight(
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    images = set()
    objects = set()
    for job in jobs:
        image_path = project_path(job["image"])
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Dev grounding image is missing: {image_path}"
            )
        images.add(str(image_path.resolve()))
        objects.add(str(job["object"]))
    return {
        "queries": len(jobs),
        "images": len(images),
        "objects": len(objects),
        "ordered_query_keys_sha256": ordered_query_keys_sha256(jobs),
        "selection_uses_ground_truth": False,
        "selection_rule": "strict No from frozen Dev110 Qwen baseline",
    }


def validate_or_create_run_config(
    path: Path, current: Mapping[str, Any]
) -> None:
    immutable = (
        "protocol",
        "baseline_predictions_sha256",
        "baseline_metrics_sha256",
        "baseline_run_config_sha256",
        "dev_grounding_config_sha256",
        "grounded_sam2_config_sha256",
        "ordered_query_keys_sha256",
        "grounding_model_id",
        "sam2_checkpoint",
        "sam2_model_config",
        "box_threshold",
        "text_threshold",
        "nms_iou_threshold",
        "device",
        "dtype",
    )
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        differences = {
            key: {"existing": existing.get(key), "current": current.get(key)}
            for key in immutable
            if existing.get(key) != current.get(key)
        }
        if differences:
            raise RuntimeError(
                "The Dev grounding run is incompatible. Choose a new "
                f"--run-name. Differences: {differences}"
            )
        return
    write_json_atomic(path, current)


def add_runtime_metadata(
    path: Path,
    runner: Any,
    torch_module: Any,
    transformers_version: str,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "runtime" in payload:
        return
    cuda_available = torch_module.cuda.is_available()
    payload["runtime"] = {
        "python": sys.version.split()[0],
        "torch": torch_module.__version__,
        "torch_cuda": torch_module.version.cuda,
        "transformers": transformers_version,
        "cuda_available": cuda_available,
        "visible_gpu_count": torch_module.cuda.device_count(),
        "gpu_0": (
            torch_module.cuda.get_device_name(0)
            if cuda_available
            else None
        ),
        "grounding_model_class": type(runner.grounding_model).__name__,
        "sam2_model_class": type(runner.sam2_predictor.model).__name__,
    }
    write_json_atomic(path, payload)


def save_metrics(
    path: Path,
    *,
    evidence_by_key: Mapping[str, Mapping[str, Any]],
    selected_keys: set[str],
    baseline_by_id: Mapping[str, Mapping[str, Any]],
    expected_queries: int,
    error_attempts: int,
    status: str,
) -> dict[str, Any]:
    evidence = [
        dict(item)
        for key, item in evidence_by_key.items()
        if key in selected_keys
    ]
    metrics = aggregate_verifier_dev_grounding_metrics(
        evidence,
        baseline_by_id=baseline_by_id,
        expected_queries=expected_queries,
        error_attempts=error_attempts,
        status=status,
    )
    write_json_atomic(path, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    baseline_path, output_dir, settings, config_sources = load_settings(args)
    baseline_records, baseline_source = validate_baseline_source(
        baseline_path
    )
    baseline_by_id = {
        str(item["id"]): item for item in baseline_records
    }
    jobs = build_negative_grounding_jobs(baseline_records)
    summary = preflight(jobs)
    print(f"Baseline:  {baseline_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        "Grounding: "
        f"box={settings['box_threshold']:.2f}, "
        f"text={settings['text_threshold']:.2f}"
    )
    if args.preflight_only:
        print("Preflight complete: no model was loaded.")
        return

    model_name = Path(settings["grounding_model_id"]).name
    run_name = args.run_name or (
        f"grounding-dev{len(jobs)}__{model_name}__sam2.1-base-plus"
    )
    run_dir = output_dir / slugify(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = run_dir / "evidence.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_config_path = run_dir / "run_config.json"
    visual_dir = run_dir / "visualizations"
    run_config = {
        "created_at_utc": utc_now(),
        "protocol": VERIFIER_DEV_GROUNDING_PROTOCOL,
        **baseline_source,
        **config_sources,
        "expected_queries": len(jobs),
        "ordered_query_keys_sha256": summary[
            "ordered_query_keys_sha256"
        ],
        **settings,
    }
    validate_or_create_run_config(run_config_path, run_config)

    evidence_by_key = load_evidence(evidence_path)
    selected_keys = {str(item["query_key"]) for item in jobs}
    completed_keys = selected_keys.intersection(evidence_by_key)
    pending = [
        item for item in jobs if item["query_key"] not in completed_keys
    ]
    historical_errors = count_jsonl(errors_path)
    print(f"Run dir:   {run_dir}")
    print(f"Selected:  {len(jobs)}")
    print(f"Completed: {len(completed_keys)}")
    print(f"Pending:   {len(pending)}")
    if not pending:
        metrics = save_metrics(
            metrics_path,
            evidence_by_key=evidence_by_key,
            selected_keys=selected_keys,
            baseline_by_id=baseline_by_id,
            expected_queries=len(jobs),
            error_attempts=historical_errors,
            status="completed",
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    try:
        import torch
        import transformers

        from grounded_visual_assistant.grounded_sam2 import (
            GroundedSam2,
            GroundedSam2Config,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Grounded-SAM-2 dependencies are missing for Dev evidence."
        ) from error
    runner = GroundedSam2(GroundedSam2Config(**settings))
    add_runtime_metadata(
        run_config_path, runner, torch, transformers.__version__
    )

    invocation_errors = 0
    successes_since_save = 0
    visualized = 0
    status = "completed"
    fatal_error: Exception | None = None
    with evidence_path.open(
        "a", encoding="utf-8", buffering=1
    ) as evidence_file, errors_path.open(
        "a", encoding="utf-8", buffering=1
    ) as errors_file:
        try:
            for job in tqdm(
                pending, desc="Verifier Dev grounding", unit="query"
            ):
                try:
                    artifact_dir = None
                    if visualized < args.visualize_limit:
                        artifact_dir = visual_dir / str(job["query_key"])
                        visualized += 1
                    result = runner.predict(
                        project_path(job["image"]),
                        f"{job['object']}.",
                        output_dir=artifact_dir,
                    )
                    evidence_record = {
                        **dict(job),
                        "grounding": compact_grounding_result(result),
                        "cuda_peak_memory_allocated_gb": result.get(
                            "cuda_peak_memory_allocated_gb"
                        ),
                        "evaluated_at_utc": utc_now(),
                    }
                    append_jsonl(evidence_file, evidence_record)
                    evidence_by_key[str(job["query_key"])] = evidence_record
                    successes_since_save += 1
                    if successes_since_save >= args.save_every:
                        save_metrics(
                            metrics_path,
                            evidence_by_key=evidence_by_key,
                            selected_keys=selected_keys,
                            baseline_by_id=baseline_by_id,
                            expected_queries=len(jobs),
                            error_attempts=(
                                historical_errors + invocation_errors
                            ),
                            status="running",
                        )
                        successes_since_save = 0
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    invocation_errors += 1
                    append_jsonl(
                        errors_file,
                        {
                            "query_key": job["query_key"],
                            "baseline_id": job["baseline_id"],
                            "image": job["image"],
                            "object": job["object"],
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(limit=5),
                            "attempted_at_utc": utc_now(),
                        },
                    )
                    tqdm.write(
                        f"ERROR {job['query_key']}: "
                        f"{type(error).__name__}: {error}"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if args.fail_fast or (
                        args.max_errors
                        and invocation_errors >= args.max_errors
                    ):
                        status = "stopped_on_error_limit"
                        break
        except KeyboardInterrupt as error:
            status = "interrupted"
            fatal_error = error

    completed_keys = selected_keys.intersection(evidence_by_key)
    if len(completed_keys) != len(selected_keys) and status == "completed":
        status = "incomplete"
    metrics = save_metrics(
        metrics_path,
        evidence_by_key=evidence_by_key,
        selected_keys=selected_keys,
        baseline_by_id=baseline_by_id,
        expected_queries=len(jobs),
        error_attempts=historical_errors + invocation_errors,
        status=status,
    )
    print(f"Evidence: {evidence_path}")
    print(f"Errors:   {errors_path}")
    print(f"Metrics:  {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if fatal_error is not None:
        raise fatal_error
    if status != "completed":
        raise RuntimeError(
            f"Dev grounding ended with status={status}; repeat the identical "
            "command to resume."
        )


if __name__ == "__main__":
    main()
