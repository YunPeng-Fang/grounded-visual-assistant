"""Run resumable oracle- or VLM-prompt Grounded-SAM-2 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.grounding_evaluation import (
    aggregate_grounding_metrics,
    evaluate_grounding_image,
)
from grounded_visual_assistant.dataset_splits import image_ids_sha256, load_image_ids
from grounded_visual_assistant.vlm_grounding import (
    aggregate_pipeline_latency,
    aggregate_prompt_quality,
    build_vlm_prompt_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate local Grounded-SAM-2 using oracle or saved VLM prompts."
        )
    )
    parser.add_argument("--dataset", default="data/eval_v0/questions.jsonl")
    parser.add_argument("--config", default="configs/grounded_sam2.yaml")
    parser.add_argument("--grounding-model-id", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-config", default=None)
    parser.add_argument("--box-threshold", type=float, default=None)
    parser.add_argument("--text-threshold", type=float, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    parser.add_argument(
        "--prompt-source",
        choices=("oracle", "vlm"),
        default="oracle",
        help="Oracle uses GT categories; vlm parses saved object_listing answers.",
    )
    parser.add_argument(
        "--vlm-predictions",
        default=None,
        help="VLM predictions.jsonl; required when --prompt-source vlm.",
    )
    parser.add_argument(
        "--image-ids",
        default=None,
        help="JSON list or split metadata file selecting image IDs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval_grounding_v0",
        help="Root output directory, relative to the project root by default.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output subdirectory name. Use a new name for changed experiments.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Evaluate the first N unique images for a smoke test.",
    )
    parser.add_argument(
        "--visualize-limit",
        type=int,
        default=10,
        help="Save overlays and individual masks for the first N selected images.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Refresh metrics.json after this many successful images.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Stop after N errors in this invocation; use 0 for no limit.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be greater than zero.")
    if args.visualize_limit < 0:
        parser.error("--visualize-limit must be zero or greater.")
    if args.save_every <= 0:
        parser.error("--save-every must be greater than zero.")
    if args.max_errors < 0:
        parser.error("--max-errors must be zero or greater.")
    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be in (0, 1].")
    if args.prompt_source == "vlm" and not args.vlm_predictions:
        parser.error("--vlm-predictions is required when --prompt-source vlm.")
    return args


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = project_path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            missing = {"id", "image", "image_id", "task_type"} - record.keys()
            if missing:
                raise ValueError(
                    f"Missing fields on {path}:{line_number}: {sorted(missing)}"
                )
            if record["id"] in seen_ids:
                raise ValueError(f"Duplicate sample id: {record['id']}")
            seen_ids.add(record["id"])
            records.append(record)
    return records


def build_oracle_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one object-listing record per image as the oracle protocol."""
    samples = []
    seen_image_ids = set()
    for record in records:
        if record["task_type"] != "object_listing":
            continue
        image_id = record["image_id"]
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate object_listing record for image {image_id}.")
        categories = sorted({str(value) for value in record.get("categories", [])})
        evidence_boxes = record.get("evidence_boxes", [])
        if not categories:
            raise ValueError(f"No oracle categories for image {image_id}.")
        if not evidence_boxes:
            raise ValueError(f"No evidence boxes for image {image_id}.")
        seen_image_ids.add(image_id)
        samples.append(
            {
                "id": record["id"],
                "image": record["image"],
                "image_id": image_id,
                "source": record.get("source"),
                "categories": categories,
                "evidence_boxes": evidence_boxes,
                "prompt": ". ".join(categories) + ".",
            }
        )
    if not samples:
        raise RuntimeError("The dataset contains no object_listing records.")
    return samples


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    predictions = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Corrupt prediction JSON on {path}:{line_number}: {exc}"
                ) from exc
            predictions[record["id"]] = record
    return predictions


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    """Load an external prediction JSONL without dataset-specific fields."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on {path}:{line_number}: {exc}"
                ) from exc
            records.append(record)
    if not records:
        raise ValueError(f"Prediction file is empty: {path}")
    return records


def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def resolve_image_path(image: str, dataset_path: Path) -> Path:
    path = Path(image)
    if path.is_absolute():
        return path
    candidates = (PROJECT_ROOT / path, dataset_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")
    return slug.lower() or "run"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def validate_or_create_run_config(path: Path, current: dict[str, Any]) -> None:
    immutable_keys = (
        "dataset_sha256",
        "prompt_source",
        "vlm_predictions_sha256",
        "vlm_prompt_parser",
        "grounding_model_id",
        "sam2_checkpoint",
        "sam2_model_config",
        "box_threshold",
        "text_threshold",
        "nms_iou_threshold",
        "iou_threshold",
        "image_ids_sha256",
        "device",
        "dtype",
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        differences = {
            key: {"existing": existing.get(key), "current": current.get(key)}
            for key in immutable_keys
            if existing.get(key) != current.get(key)
        }
        if differences:
            raise RuntimeError(
                "The output directory belongs to an incompatible run. "
                f"Choose a new --run-name. Differences: {differences}"
            )
        return
    write_json_atomic(path, current)


def add_runtime_metadata(
    path: Path, runner: Any, torch_module: Any, transformers_version: str
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
        "gpu_0": torch_module.cuda.get_device_name(0) if cuda_available else None,
        "grounding_model_class": type(runner.grounding_model).__name__,
        "sam2_model_class": type(runner.sam2_predictor.model).__name__,
    }
    write_json_atomic(path, payload)


def save_metrics(
    path: Path,
    predictions_by_id: dict[str, dict[str, Any]],
    selected_ids: set[str],
    expected_images: int,
    error_attempts: int,
    status: str,
    iou_threshold: float,
    prompt_source: str,
) -> dict[str, Any]:
    predictions = [
        prediction
        for sample_id, prediction in predictions_by_id.items()
        if sample_id in selected_ids
    ]
    metrics = aggregate_grounding_metrics(
        predictions,
        expected_images=expected_images,
        error_attempts=error_attempts,
        status=status,
        iou_threshold=iou_threshold,
    )
    if prompt_source == "vlm":
        metrics["prompt_quality"] = aggregate_prompt_quality(
            predictions,
            expected_images=expected_images,
        )
        metrics["end_to_end_latency_seconds"] = aggregate_pipeline_latency(
            predictions
        )
    write_json_atomic(path, metrics)
    return metrics


def empty_prompt_result(
    *,
    grounding_model_id: str,
    sam2_checkpoint: str,
    sam2_model_config: str,
    box_threshold: float,
    text_threshold: float,
    nms_iou_threshold: float | None,
    device: str,
) -> dict[str, Any]:
    """Represent an empty VLM prompt as a valid all-missed prediction."""
    return {
        "text_prompt": "",
        "annotations": [],
        "models": {
            "grounding": grounding_model_id,
            "sam2_checkpoint": sam2_checkpoint,
            "sam2_config": sam2_model_config,
        },
        "thresholds": {
            "box": box_threshold,
            "text": text_threshold,
            "nms_iou": nms_iou_threshold,
        },
        "postprocessing": {
            "candidate_count": 0,
            "kept_count": 0,
            "suppressed_count": 0,
        },
        "latency_seconds": {"grounding": 0.0, "sam2": 0.0, "total": 0.0},
        "device": device,
        "skipped_reason": "empty_vlm_prompt",
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset_path = project_path(args.dataset)
    samples = build_oracle_samples(load_jsonl(dataset_path))
    vlm_predictions_path = (
        project_path(args.vlm_predictions) if args.vlm_predictions else None
    )
    image_ids_path = project_path(args.image_ids) if args.image_ids else None
    requested_image_ids: set[int] | None = None
    if image_ids_path is not None:
        requested_image_ids = set(load_image_ids(image_ids_path))
        available_image_ids = {int(sample["image_id"]) for sample in samples}
        unknown_image_ids = sorted(requested_image_ids - available_image_ids)
        if unknown_image_ids:
            raise ValueError(
                "Split contains image IDs absent from the dataset: "
                f"{unknown_image_ids[:10]}"
            )
        samples = [
            sample
            for sample in samples
            if int(sample["image_id"]) in requested_image_ids
        ]
    if args.max_images is not None:
        samples = samples[: args.max_images]
    if args.prompt_source == "vlm":
        assert vlm_predictions_path is not None
        samples = build_vlm_prompt_samples(
            samples,
            load_prediction_records(vlm_predictions_path),
        )
        vlm_prompt_parsers = {
            str(sample["vlm_prediction"]["parser"]) for sample in samples
        }
        if len(vlm_prompt_parsers) != 1:
            raise ValueError(
                "VLM predictions mix prompt parsers: "
                f"{sorted(vlm_prompt_parsers)}"
            )
        vlm_prompt_parser = next(iter(vlm_prompt_parsers))
    else:
        vlm_prompt_parser = None

    grounding_cfg = dict(cfg["grounding"])
    sam2_cfg = dict(cfg["sam2"])
    runtime_cfg = dict(cfg["runtime"])
    grounding_model_id = args.grounding_model_id or grounding_cfg["model_id"]
    sam2_checkpoint = args.sam2_checkpoint or sam2_cfg["checkpoint"]
    sam2_model_config = args.sam2_model_config or sam2_cfg["model_config"]
    box_threshold = (
        args.box_threshold
        if args.box_threshold is not None
        else float(grounding_cfg.get("box_threshold", 0.4))
    )
    text_threshold = (
        args.text_threshold
        if args.text_threshold is not None
        else float(grounding_cfg.get("text_threshold", 0.3))
    )
    nms_iou_threshold = (
        args.nms_iou_threshold
        if args.nms_iou_threshold is not None
        else grounding_cfg.get("nms_iou_threshold")
    )
    if nms_iou_threshold is not None:
        nms_iou_threshold = float(nms_iou_threshold)
    device = args.device or runtime_cfg.get("device", "cuda")
    dtype = args.dtype or runtime_cfg.get("dtype", "float16")
    local_files_only = bool(
        args.local_files_only or grounding_cfg.get("local_files_only", False)
    )

    dataset_name = dataset_path.parent.name or dataset_path.stem
    grounding_name = Path(grounding_model_id).name
    sam2_name = Path(sam2_checkpoint).stem
    run_name = args.run_name or (
        f"{slugify(dataset_name)}__{slugify(grounding_name)}"
        f"__{slugify(sam2_name)}__{args.prompt_source}"
        + (
            f"__{slugify(image_ids_path.stem)}"
            if image_ids_path is not None
            else ""
        )
        + (
            f"__nms-{nms_iou_threshold:.2f}"
            if nms_iou_threshold is not None
            else ""
        )
    )
    run_dir = project_path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    visualization_root = run_dir / "visualizations"

    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_config_path = run_dir / "run_config.json"
    run_config = {
        "created_at_utc": utc_now(),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256sum(dataset_path),
        "prompt_source": args.prompt_source,
        "vlm_predictions": (
            str(vlm_predictions_path) if vlm_predictions_path is not None else None
        ),
        "vlm_predictions_sha256": (
            sha256sum(vlm_predictions_path)
            if vlm_predictions_path is not None
            else None
        ),
        "vlm_prompt_parser": (
            vlm_prompt_parser
        ),
        "grounding_model_id": grounding_model_id,
        "sam2_checkpoint": sam2_checkpoint,
        "sam2_model_config": sam2_model_config,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "iou_threshold": args.iou_threshold,
        "image_ids": str(image_ids_path) if image_ids_path is not None else None,
        "image_ids_sha256": (
            image_ids_sha256(requested_image_ids)
            if requested_image_ids is not None
            else None
        ),
        "device": device,
        "dtype": dtype,
        "local_files_only": local_files_only,
    }
    validate_or_create_run_config(run_config_path, run_config)

    predictions_by_id = load_existing_predictions(predictions_path)
    selected_ids = {sample["id"] for sample in samples}
    completed_ids = set(predictions_by_id) & selected_ids
    pending = [sample for sample in samples if sample["id"] not in completed_ids]
    sample_positions = {sample["id"]: index for index, sample in enumerate(samples)}
    historical_errors = count_jsonl_records(errors_path)

    print(f"Dataset:   {dataset_path}")
    print(f"Run dir:   {run_dir}")
    print(f"Selected:  {len(samples)} unique images")
    print(f"Completed: {len(completed_ids)}")
    print(f"Pending:   {len(pending)}")
    print(f"Protocol:  {args.prompt_source}")
    print(f"Split:     {image_ids_path or 'all images'}")
    if args.prompt_source == "vlm":
        prompt_quality = aggregate_prompt_quality(
            samples,
            expected_images=len(samples),
        )
        prompt_counts = prompt_quality["counts"]
        print(f"VLM file:  {vlm_predictions_path}")
        print(
            "Prompts:   "
            f"P={prompt_quality['micro_precision']:.4f}, "
            f"R={prompt_quality['micro_recall']:.4f}, "
            f"F1={prompt_quality['micro_f1']:.4f}, "
            f"empty={prompt_counts['empty_prompt_images']}"
        )

    if not pending:
        metrics = save_metrics(
            metrics_path,
            predictions_by_id,
            selected_ids,
            len(samples),
            historical_errors,
            "completed",
            args.iou_threshold,
            args.prompt_source,
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
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Grounded-SAM-2 runtime dependencies are missing. Run "
            "scripts/install_grounded_sam2.sh first."
        ) from exc

    runner = GroundedSam2(
        GroundedSam2Config(
            grounding_model_id=grounding_model_id,
            sam2_checkpoint=sam2_checkpoint,
            sam2_model_config=sam2_model_config,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    )
    add_runtime_metadata(run_config_path, runner, torch, transformers.__version__)

    invocation_errors = 0
    successes_since_save = 0
    status = "completed"
    fatal_error: Exception | None = None
    with predictions_path.open("a", encoding="utf-8", buffering=1) as predictions_file, \
        errors_path.open("a", encoding="utf-8", buffering=1) as errors_file:
        try:
            for sample in tqdm(pending, desc="Grounding", unit="image"):
                image_path = resolve_image_path(sample["image"], dataset_path)
                try:
                    if not image_path.exists():
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    sample_position = sample_positions[sample["id"]]
                    artifact_dir = (
                        visualization_root / f"{int(sample['image_id']):012d}"
                        if sample_position < args.visualize_limit
                        else None
                    )
                    if sample["prompt"]:
                        result = runner.predict(
                            image_path,
                            sample["prompt"],
                            output_dir=artifact_dir,
                        )
                    else:
                        result = empty_prompt_result(
                            grounding_model_id=grounding_model_id,
                            sam2_checkpoint=sam2_checkpoint,
                            sam2_model_config=sam2_model_config,
                            box_threshold=box_threshold,
                            text_threshold=text_threshold,
                            nms_iou_threshold=nms_iou_threshold,
                            device=device,
                        )
                    evaluation = evaluate_grounding_image(
                        sample["evidence_boxes"],
                        result["annotations"],
                        iou_threshold=args.iou_threshold,
                    )
                    prediction: dict[str, Any] = {
                        "id": sample["id"],
                        "image": sample["image"],
                        "image_id": sample["image_id"],
                        "source": sample["source"],
                        "prompt_source": args.prompt_source,
                        "prompt": result["text_prompt"],
                        "prompt_categories": sample.get(
                            "prompt_categories", sample["categories"]
                        ),
                        "target_categories": sample["categories"],
                        "target_evidence_boxes": sample["evidence_boxes"],
                        "annotations": result["annotations"],
                        "evaluation": evaluation,
                        "models": result["models"],
                        "thresholds": result["thresholds"],
                        "postprocessing": result["postprocessing"],
                        "latency_seconds": result["latency_seconds"],
                        "device": result["device"],
                        "evaluated_at_utc": utc_now(),
                    }
                    if "skipped_reason" in result:
                        prediction["skipped_reason"] = result["skipped_reason"]
                    if args.prompt_source == "vlm":
                        prediction["prompt_evaluation"] = sample[
                            "prompt_evaluation"
                        ]
                        prediction["vlm_prediction"] = sample["vlm_prediction"]
                        vlm_latency = float(
                            sample["vlm_prediction"]["latency_seconds"]
                        )
                        prediction["pipeline_latency_seconds"] = {
                            "vlm": round(vlm_latency, 6),
                            "grounding": float(
                                result["latency_seconds"]["grounding"]
                            ),
                            "sam2": float(result["latency_seconds"]["sam2"]),
                            "total": round(
                                vlm_latency
                                + float(result["latency_seconds"]["total"]),
                                6,
                            ),
                        }
                    if "cuda_peak_memory_allocated_gb" in result:
                        prediction["cuda_peak_memory_allocated_gb"] = result[
                            "cuda_peak_memory_allocated_gb"
                        ]
                    append_jsonl(predictions_file, prediction)
                    predictions_by_id[sample["id"]] = prediction
                    successes_since_save += 1

                    if successes_since_save >= args.save_every:
                        save_metrics(
                            metrics_path,
                            predictions_by_id,
                            selected_ids,
                            len(samples),
                            historical_errors + invocation_errors,
                            "running",
                            args.iou_threshold,
                            args.prompt_source,
                        )
                        successes_since_save = 0
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    invocation_errors += 1
                    error = {
                        "id": sample["id"],
                        "image": sample["image"],
                        "image_id": sample["image_id"],
                        "prompt": sample["prompt"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=5),
                        "attempted_at_utc": utc_now(),
                    }
                    append_jsonl(errors_file, error)
                    tqdm.write(f"ERROR {sample['id']}: {type(exc).__name__}: {exc}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if args.fail_fast:
                        status = "failed"
                        fatal_error = exc
                        break
                    if args.max_errors and invocation_errors >= args.max_errors:
                        status = "stopped_after_errors"
                        tqdm.write(
                            f"Stopping after {invocation_errors} errors. "
                            "Fix the issue and rerun to resume."
                        )
                        break
        except KeyboardInterrupt:
            status = "interrupted"
            print("\nInterrupted. Completed predictions are preserved.")

    if status == "completed":
        completed_count = len(set(predictions_by_id) & selected_ids)
        if completed_count < len(samples):
            status = "completed_with_errors"

    metrics = save_metrics(
        metrics_path,
        predictions_by_id,
        selected_ids,
        len(samples),
        historical_errors + invocation_errors,
        status,
        args.iou_threshold,
        args.prompt_source,
    )
    print(f"Predictions: {predictions_path}")
    print(f"Errors:      {errors_path}")
    print(f"Metrics:     {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if fatal_error is not None:
        raise RuntimeError(
            f"Evaluation stopped by --fail-fast after: {fatal_error}"
        ) from fatal_error


if __name__ == "__main__":
    main()
