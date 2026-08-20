"""Run a resumable, task-aware VLM evaluation over a JSONL dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.evaluation import aggregate_metrics, score_prediction


TASK_TYPES = ("all", "object_listing", "object_existence", "spatial_relation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate a local VLM with resumable JSONL outputs."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval_v0/questions.jsonl",
        help="Evaluation JSONL path, relative to the project root by default.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-id", default=None, help="Override config model id.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory. Defaults to runtime.eval_output_dir.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output subdirectory name. Use a new name for a fresh experiment.",
    )
    parser.add_argument("--task-type", choices=TASK_TYPES, default="all")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override generation length; 64 is recommended for eval_v0.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N selected records for a smoke test.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Refresh metrics.json after this many successful predictions.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Stop after N errors in this invocation; use 0 for no limit.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--required-split",
        choices=("dev", "test"),
        default=None,
        help="Reject records outside this frozen split.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate dataset hashes and images without loading a model.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never download model or processor files.",
    )
    args = parser.parse_args()
    if args.save_every <= 0:
        parser.error("--save-every must be greater than zero.")
    if args.max_errors < 0:
        parser.error("--max-errors must be zero or greater.")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero.")
    if args.required_split == "test" and args.max_samples is not None:
        parser.error(
            "Held-out Test evaluation must be complete; --max-samples is prohibited."
        )
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
            missing = {
                "id",
                "image",
                "question",
                "task_type",
                "gt_answer",
            } - record.keys()
            if missing:
                raise ValueError(
                    f"Missing fields on {path}:{line_number}: {sorted(missing)}"
                )
            if record["id"] in seen_ids:
                raise ValueError(f"Duplicate sample id: {record['id']}")
            seen_ids.add(record["id"])
            records.append(record)
    return records


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


def verify_dataset_manifest(dataset_path: Path) -> dict[str, Any] | None:
    manifest_path = dataset_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = (manifest.get("artifact_sha256") or {}).get(dataset_path.name)
    if expected_hash is None:
        return {
            "path": str(manifest_path),
            "sha256": sha256sum(manifest_path),
            "artifact_verified": False,
        }
    actual_hash = sha256sum(dataset_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Dataset hash does not match its immutable manifest: {dataset_path}"
        )
    return {
        "path": str(manifest_path),
        "sha256": sha256sum(manifest_path),
        "artifact_verified": True,
        "artifact_sha256": actual_hash,
    }


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


def validate_or_create_run_config(
    path: Path, current: dict[str, Any]
) -> None:
    immutable_keys = (
        "dataset_sha256",
        "dataset_manifest_sha256",
        "model_id",
        "torch_dtype",
        "device_map",
        "max_new_tokens",
        "do_sample",
        "task_type",
        "required_split",
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        differences = {
            key: {"existing": existing.get(key), "current": current.get(key)}
            for key in immutable_keys
            if key in existing and existing.get(key) != current.get(key)
        }
        if differences:
            raise RuntimeError(
                "The output directory belongs to an incompatible run. "
                f"Choose a new --run-name. Differences: {differences}"
            )
        migrated = False
        for key in immutable_keys:
            if key not in existing:
                existing[key] = current.get(key)
                migrated = True
        if migrated:
            write_json_atomic(path, existing)
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
        "model_class": type(runner.model).__name__,
    }
    write_json_atomic(path, payload)


def selected_records(
    records: list[dict[str, Any]], task_type: str, max_samples: int | None
) -> list[dict[str, Any]]:
    if task_type != "all":
        records = [record for record in records if record["task_type"] == task_type]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be greater than zero.")
        records = records[:max_samples]
    return records


def preflight_dataset(
    records: list[dict[str, Any]],
    dataset_path: Path,
    required_split: str | None,
) -> dict[str, Any]:
    missing_images = []
    resolved_images = set()
    for record in records:
        split = record.get("split")
        if required_split is not None and split != required_split:
            raise RuntimeError(
                f"Record {record['id']} belongs to split {split!r}, not "
                f"the required {required_split!r} split."
            )
        image_path = resolve_image_path(str(record["image"]), dataset_path)
        if not image_path.is_file():
            missing_images.append(str(image_path))
        else:
            resolved_images.add(str(image_path.resolve()))
    if missing_images:
        raise FileNotFoundError(
            f"Dataset references {len(missing_images)} missing images; first: "
            f"{missing_images[0]}"
        )
    return {
        "questions": len(records),
        "images": len(resolved_images),
        "tasks": dict(sorted(Counter(item["task_type"] for item in records).items())),
        "sources": dict(
            sorted(Counter(str(item.get("source", "unknown")) for item in records).items())
        ),
        "splits": dict(
            sorted(Counter(str(item.get("split", "unspecified")) for item in records).items())
        ),
    }


def save_metrics(
    path: Path,
    predictions_by_id: dict[str, dict[str, Any]],
    selected_ids: set[str],
    expected_samples: int,
    error_attempts: int,
    status: str,
) -> dict[str, Any]:
    predictions = [
        prediction
        for sample_id, prediction in predictions_by_id.items()
        if sample_id in selected_ids
    ]
    metrics = aggregate_metrics(
        predictions,
        expected_samples=expected_samples,
        error_attempts=error_attempts,
        status=status,
    )
    write_json_atomic(path, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset_path = project_path(args.dataset)
    records = selected_records(load_jsonl(dataset_path), args.task_type, args.max_samples)
    if not records:
        raise RuntimeError("No evaluation records matched the requested filters.")

    manifest_verification = verify_dataset_manifest(dataset_path)
    dataset_summary = preflight_dataset(records, dataset_path, args.required_split)

    model_cfg = dict(cfg["model"])
    if args.model_id:
        model_cfg["model_id"] = args.model_id
    if args.max_new_tokens is not None:
        model_cfg["max_new_tokens"] = args.max_new_tokens

    dataset_name = dataset_path.parent.name or dataset_path.stem
    model_name = Path(model_cfg["model_id"]).name
    run_name = args.run_name or f"{slugify(dataset_name)}__{slugify(model_name)}"
    output_root = project_path(
        args.output_dir
        or cfg.get("runtime", {}).get("eval_output_dir", "outputs/eval_v0")
    )
    run_dir = output_root / run_name

    print(f"Dataset:   {dataset_path}")
    print(f"Run dir:   {run_dir}")
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    if manifest_verification is not None:
        print(
            "Manifest:  "
            + (
                "artifact hash verified"
                if manifest_verification["artifact_verified"]
                else "found (dataset hash not declared)"
            )
        )
    if args.preflight_only:
        print("Preflight complete: no model was loaded and no run was created.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_config_path = run_dir / "run_config.json"
    run_config = {
        "created_at_utc": utc_now(),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256sum(dataset_path),
        "dataset_manifest_sha256": (
            manifest_verification["sha256"]
            if manifest_verification is not None
            else None
        ),
        "model_id": model_cfg["model_id"],
        "torch_dtype": model_cfg.get("torch_dtype", "auto"),
        "device_map": model_cfg.get("device_map", "auto"),
        "max_new_tokens": int(model_cfg.get("max_new_tokens", 256)),
        "do_sample": bool(model_cfg.get("do_sample", False)),
        "task_type": args.task_type,
        "required_split": args.required_split,
    }
    validate_or_create_run_config(run_config_path, run_config)

    predictions_by_id = load_existing_predictions(predictions_path)
    selected_ids = {record["id"] for record in records}
    completed_ids = set(predictions_by_id) & selected_ids
    pending = [record for record in records if record["id"] not in completed_ids]
    historical_errors = count_jsonl_records(errors_path)

    print(f"Selected:  {len(records)}")
    print(f"Completed: {len(completed_ids)}")
    print(f"Pending:   {len(pending)}")

    if not pending:
        metrics = save_metrics(
            metrics_path,
            predictions_by_id,
            selected_ids,
            len(records),
            historical_errors,
            "completed",
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    try:
        import torch
        import transformers

        from grounded_visual_assistant.vlm_baseline import (
            VlmBaseline,
            VlmBaselineConfig,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "VLM runtime dependencies are missing. Install "
            "requirements-torch-cu124.txt and requirements.txt first."
        ) from exc

    runner = VlmBaseline(
        VlmBaselineConfig(
            model_id=model_cfg["model_id"],
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            device_map=model_cfg.get("device_map", "auto"),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 256)),
            do_sample=bool(model_cfg.get("do_sample", False)),
            local_files_only=bool(
                args.local_files_only or model_cfg.get("local_files_only", False)
            ),
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
            for sample in tqdm(pending, desc="Evaluating", unit="sample"):
                image_path = resolve_image_path(sample["image"], dataset_path)
                try:
                    if not image_path.exists():
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    result = runner.answer(image_path, sample["question"])
                    evaluation = score_prediction(sample, result["answer"])
                    prediction = {
                        "id": sample["id"],
                        "image": sample["image"],
                        "question": sample["question"],
                        "task_type": sample["task_type"],
                        "gt_answer": sample["gt_answer"],
                        "prediction": result["answer"],
                        "evaluation": evaluation,
                        "model": result["model"],
                        "latency_seconds": result.get(
                            "end_to_end_latency_seconds", result["latency_seconds"]
                        ),
                        "generation_latency_seconds": result["latency_seconds"],
                        "device": result["device"],
                        "cuda_available": result["cuda_available"],
                        "evaluated_at_utc": utc_now(),
                    }
                    for key in (
                        "image_id",
                        "sample_id",
                        "source_image_id",
                        "source",
                        "split",
                        "categories",
                        "evidence_boxes",
                        "metadata",
                    ):
                        if key in sample:
                            prediction[key] = sample[key]
                    for key in (
                        "generated_tokens",
                        "max_new_tokens",
                        "hit_max_new_tokens",
                        "cuda_memory_allocated_gb",
                        "cuda_peak_memory_allocated_gb",
                        "cuda_memory_reserved_gb",
                    ):
                        if key in result:
                            prediction[key] = result[key]
                    append_jsonl(predictions_file, prediction)
                    predictions_by_id[sample["id"]] = prediction
                    successes_since_save += 1

                    if successes_since_save >= max(args.save_every, 1):
                        save_metrics(
                            metrics_path,
                            predictions_by_id,
                            selected_ids,
                            len(records),
                            historical_errors + invocation_errors,
                            "running",
                        )
                        successes_since_save = 0
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # continue past isolated corrupt samples
                    invocation_errors += 1
                    error = {
                        "id": sample["id"],
                        "image": sample["image"],
                        "question": sample["question"],
                        "task_type": sample["task_type"],
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

    metrics = save_metrics(
        metrics_path,
        predictions_by_id,
        selected_ids,
        len(records),
        historical_errors + invocation_errors,
        status,
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
