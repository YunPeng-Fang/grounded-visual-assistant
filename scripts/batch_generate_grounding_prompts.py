"""Generate resumable ontology-constrained JSON prompts with a local VLM."""

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

from grounded_visual_assistant.dataset_splits import image_ids_sha256, load_image_ids
from grounded_visual_assistant.evaluation import aggregate_metrics
from grounded_visual_assistant.structured_prompting import (
    STRUCTURED_PROMPT_PARSER,
    STRUCTURED_PROMPT_VERSION,
    STRUCTURED_SYSTEM_PROMPT,
    aggregate_structured_output,
    build_structured_category_question,
    evaluate_structured_category_answer,
)
from grounded_visual_assistant.vlm_grounding import aggregate_prompt_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured COCO-80 category prompts with a local VLM."
    )
    parser.add_argument("--dataset", default="data/eval_v0/questions.jsonl")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--image-ids",
        default=None,
        help="JSON list or split metadata file selecting image IDs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to runtime.eval_output_dir from the VLM config.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Generate only the first N selected images for a resumable smoke test.",
    )
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero.")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be greater than zero.")
    if args.save_every <= 0:
        parser.error("--save-every must be greater than zero.")
    if args.max_errors < 0:
        parser.error("--max-errors must be zero or greater.")
    return args


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(project_path(path).read_text(encoding="utf-8"))


def load_listing_samples(path: Path) -> list[dict[str, Any]]:
    samples = []
    seen_ids = set()
    seen_image_ids = set()
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
            if record.get("task_type") != "object_listing":
                continue
            missing = {
                "id",
                "image",
                "image_id",
                "gt_answer",
                "categories",
            } - record.keys()
            if missing:
                raise ValueError(
                    f"Missing fields on {path}:{line_number}: {sorted(missing)}"
                )
            sample_id = str(record["id"])
            image_id = int(record["image_id"])
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id: {sample_id}")
            if image_id in seen_image_ids:
                raise ValueError(f"Duplicate object_listing image_id: {image_id}")
            seen_ids.add(sample_id)
            seen_image_ids.add(image_id)
            samples.append(record)
    if not samples:
        raise ValueError(f"No object_listing samples found in {path}.")
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
            sample_id = str(record["id"])
            if sample_id in predictions:
                raise ValueError(f"Duplicate saved prediction id: {sample_id}")
            predictions[sample_id] = record
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


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        "image_ids_sha256",
        "model_id",
        "torch_dtype",
        "device_map",
        "max_new_tokens",
        "do_sample",
        "prompt_version",
        "prompt_parser",
        "system_prompt_sha256",
        "question_sha256",
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
        "model_class": type(runner.model).__name__,
    }
    write_json_atomic(path, payload)


def save_metrics(
    path: Path,
    predictions_by_id: dict[str, dict[str, Any]],
    selected_ids: set[str],
    expected_images: int,
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
        expected_samples=expected_images,
        error_attempts=error_attempts,
        status=status,
    )
    metrics["structured_output"] = aggregate_structured_output(predictions)
    metrics["prompt_quality"] = aggregate_prompt_quality(
        predictions,
        expected_images=expected_images,
    )
    write_json_atomic(path, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset_path = project_path(args.dataset)
    samples = load_listing_samples(dataset_path)
    image_ids_path = project_path(args.image_ids) if args.image_ids else None
    requested_image_ids: set[int] | None = None
    if image_ids_path is not None:
        requested_image_ids = set(load_image_ids(image_ids_path))
        available = {int(sample["image_id"]) for sample in samples}
        unknown = sorted(requested_image_ids - available)
        if unknown:
            raise ValueError(
                f"Split image IDs are absent from the dataset: {unknown[:10]}"
            )
        samples = [
            sample
            for sample in samples
            if int(sample["image_id"]) in requested_image_ids
        ]
    if args.max_images is not None:
        samples = samples[: args.max_images]

    model_cfg = dict(cfg["model"])
    if args.model_id:
        model_cfg["model_id"] = args.model_id
    model_cfg["max_new_tokens"] = args.max_new_tokens
    question = build_structured_category_question()

    model_name = Path(model_cfg["model_id"]).name
    split_name = image_ids_path.stem if image_ids_path is not None else "all"
    run_name = args.run_name or (
        f"structured-prompts__{slugify(model_name)}__{slugify(split_name)}"
    )
    output_root = project_path(
        args.output_dir
        or cfg.get("runtime", {}).get("eval_output_dir", "outputs/eval_v0")
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_config_path = run_dir / "run_config.json"
    run_config = {
        "created_at_utc": utc_now(),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256sum(dataset_path),
        "image_ids": str(image_ids_path) if image_ids_path is not None else None,
        "image_ids_sha256": (
            image_ids_sha256(requested_image_ids)
            if requested_image_ids is not None
            else None
        ),
        "model_id": model_cfg["model_id"],
        "torch_dtype": model_cfg.get("torch_dtype", "auto"),
        "device_map": model_cfg.get("device_map", "auto"),
        "max_new_tokens": int(model_cfg["max_new_tokens"]),
        "do_sample": bool(model_cfg.get("do_sample", False)),
        "local_files_only": bool(
            args.local_files_only or model_cfg.get("local_files_only", False)
        ),
        "prompt_version": STRUCTURED_PROMPT_VERSION,
        "prompt_parser": STRUCTURED_PROMPT_PARSER,
        "system_prompt_sha256": text_sha256(STRUCTURED_SYSTEM_PROMPT),
        "question_sha256": text_sha256(question),
    }
    validate_or_create_run_config(run_config_path, run_config)

    predictions_by_id = load_existing_predictions(predictions_path)
    selected_ids = {str(sample["id"]) for sample in samples}
    completed_ids = set(predictions_by_id) & selected_ids
    pending = [sample for sample in samples if sample["id"] not in completed_ids]
    historical_errors = count_jsonl_records(errors_path)
    print(f"Dataset:   {dataset_path}")
    print(f"Run dir:   {run_dir}")
    print(f"Selected:  {len(samples)} unique images")
    print(f"Completed: {len(completed_ids)}")
    print(f"Pending:   {len(pending)}")
    print(f"Split:     {image_ids_path or 'all images'}")
    print(f"Prompt:    {STRUCTURED_PROMPT_VERSION}")

    if not pending:
        metrics = save_metrics(
            metrics_path,
            predictions_by_id,
            selected_ids,
            len(samples),
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
            "VLM runtime dependencies are missing. Install the project "
            "requirements in the grounded-vlm environment."
        ) from exc

    runner = VlmBaseline(
        VlmBaselineConfig(
            model_id=model_cfg["model_id"],
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            device_map=model_cfg.get("device_map", "auto"),
            max_new_tokens=int(model_cfg["max_new_tokens"]),
            do_sample=bool(model_cfg.get("do_sample", False)),
            local_files_only=run_config["local_files_only"],
        )
    )
    add_runtime_metadata(run_config_path, runner, torch, transformers.__version__)

    invocation_errors = 0
    successes_since_save = 0
    status = "completed"
    fatal_error: Exception | None = None
    with predictions_path.open("a", encoding="utf-8", buffering=1) as prediction_file, \
        errors_path.open("a", encoding="utf-8", buffering=1) as error_file:
        try:
            for sample in tqdm(pending, desc="Structured prompts", unit="image"):
                image_path = resolve_image_path(sample["image"], dataset_path)
                try:
                    if not image_path.exists():
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    result = runner.answer(
                        image_path,
                        question,
                        system_prompt=STRUCTURED_SYSTEM_PROMPT,
                    )
                    structured, evaluation = evaluate_structured_category_answer(
                        result["answer"], sample["categories"]
                    )
                    prediction: dict[str, Any] = {
                        "id": sample["id"],
                        "image": sample["image"],
                        "image_id": sample["image_id"],
                        "source": sample.get("source"),
                        "question": question,
                        "task_type": "object_listing",
                        "gt_answer": sample["gt_answer"],
                        "categories": sample["categories"],
                        "prediction": result["answer"],
                        "structured_output": structured,
                        "evaluation": evaluation,
                        "prompt_evaluation": evaluation,
                        "model": result["model"],
                        "latency_seconds": result.get(
                            "end_to_end_latency_seconds", result["latency_seconds"]
                        ),
                        "generation_latency_seconds": result["latency_seconds"],
                        "generated_tokens": result.get("generated_tokens"),
                        "max_new_tokens": result.get("max_new_tokens"),
                        "hit_max_new_tokens": result.get(
                            "hit_max_new_tokens", False
                        ),
                        "device": result["device"],
                        "cuda_available": result["cuda_available"],
                        "prompt_version": STRUCTURED_PROMPT_VERSION,
                        "evaluated_at_utc": utc_now(),
                    }
                    for key in (
                        "cuda_memory_allocated_gb",
                        "cuda_peak_memory_allocated_gb",
                        "cuda_memory_reserved_gb",
                    ):
                        if key in result:
                            prediction[key] = result[key]
                    append_jsonl(prediction_file, prediction)
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
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=5),
                        "attempted_at_utc": utc_now(),
                    }
                    append_jsonl(error_file, error)
                    tqdm.write(f"ERROR {sample['id']}: {type(exc).__name__}: {exc}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if args.fail_fast:
                        status = "failed"
                        fatal_error = exc
                        break
                    if args.max_errors and invocation_errors >= args.max_errors:
                        status = "stopped_after_errors"
                        break
        except KeyboardInterrupt:
            status = "interrupted"
            print("\nInterrupted. Completed predictions are preserved.")

    completed_count = len(set(predictions_by_id) & selected_ids)
    if status == "completed" and completed_count < len(samples):
        status = "completed_with_errors"
    metrics = save_metrics(
        metrics_path,
        predictions_by_id,
        selected_ids,
        len(samples),
        historical_errors + invocation_errors,
        status,
    )
    print(f"Predictions: {predictions_path}")
    print(f"Errors:      {errors_path}")
    print(f"Metrics:     {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if fatal_error is not None:
        raise RuntimeError(
            f"Generation stopped by --fail-fast after: {fatal_error}"
        ) from fatal_error


if __name__ == "__main__":
    main()
