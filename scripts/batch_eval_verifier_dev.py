"""Run the resumable Qwen baseline on the POPE-isolated Verifier Dev110."""

from __future__ import annotations

import argparse
import hashlib
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

from grounded_visual_assistant.pope_dataset import (
    read_json_records,
    sha256sum,
)
from grounded_visual_assistant.pope_evaluation import evaluate_answer
from grounded_visual_assistant.verifier_dev_dataset import (
    VERIFIER_DEV_PROTOCOL,
)
from grounded_visual_assistant.verifier_dev_evaluation import (
    VERIFIER_DEV_BASELINE_PROTOCOL,
    VERIFIER_DEV_SYSTEM_PROMPT,
    aggregate_verifier_dev_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen on the locked Verifier Dev110 protocol."
    )
    parser.add_argument(
        "--dataset", default="data/verifier_dev_v1/questions.jsonl"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--output-dir", default="outputs/eval_verifier_dev_v1"
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.save_every <= 0:
        parser.error("--max-new-tokens and --save-every must be positive.")
    if args.max_errors < 0:
        parser.error("--max-errors must be zero or greater.")
    return args


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")
    return slug.lower() or "run"


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
    handle.flush()


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    predictions = {}
    for item in read_json_records(path):
        sample_id = str(item["id"])
        if sample_id in predictions:
            raise ValueError(f"Duplicate Verifier Dev prediction: {sample_id}")
        predictions[sample_id] = item
    return predictions


def validate_dataset(
    dataset_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = dataset_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Verifier Dev manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != VERIFIER_DEV_PROTOCOL:
        raise RuntimeError("Verifier Dev dataset protocol mismatch.")
    expected_hash = (
        manifest.get("artifact_sha256") or {}
    ).get("questions.jsonl")
    actual_hash = sha256sum(dataset_path)
    if expected_hash != actual_hash:
        raise RuntimeError("Verifier Dev questions hash mismatch.")
    records = read_json_records(dataset_path)
    required = {
        "id",
        "pair_id",
        "pair_role",
        "image",
        "image_id",
        "question",
        "object",
        "gt_answer",
        "supercategory",
    }
    ids = []
    labels = {"yes": 0, "no": 0}
    images = set()
    for index, item in enumerate(records, start=1):
        missing = required - item.keys()
        if missing:
            raise ValueError(
                f"Verifier Dev record {index} misses {sorted(missing)}."
            )
        label = str(item["gt_answer"])
        if label not in labels:
            raise ValueError(f"Invalid Verifier Dev label: {label}")
        image_path = project_path(str(item["image"]))
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Verifier Dev image is missing: {image_path}"
            )
        ids.append(str(item["id"]))
        labels[label] += 1
        images.add(int(item["image_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("Verifier Dev contains duplicate IDs.")
    if labels != {"yes": 55, "no": 55} or len(records) != 110:
        raise RuntimeError(
            f"Verifier Dev110 balance mismatch: {labels}, n={len(records)}."
        )
    return records, {
        "questions": len(records),
        "images": len(images),
        "labels": labels,
        "pairs": len({str(item["pair_id"]) for item in records}),
        "categories": len({str(item["object"]) for item in records}),
        "dataset_sha256": actual_hash,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256sum(manifest_path),
        "selected_ids_sha256": hashlib.sha256(
            (
                "\n".join(str(item["id"]) for item in records) + "\n"
            ).encode("utf-8")
        ).hexdigest(),
    }


def validate_or_create_run_config(
    path: Path, current: Mapping[str, Any]
) -> None:
    immutable = (
        "protocol",
        "dataset_sha256",
        "dataset_manifest_sha256",
        "selected_ids_sha256",
        "model_id",
        "torch_dtype",
        "device_map",
        "max_new_tokens",
        "do_sample",
        "system_prompt_sha256",
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
                "The Verifier Dev run directory is incompatible. Choose a "
                f"new --run-name. Differences: {differences}"
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
        "gpu_0": (
            torch_module.cuda.get_device_name(0)
            if cuda_available
            else None
        ),
        "model_class": type(runner.model).__name__,
    }
    write_json_atomic(path, payload)


def save_metrics(
    path: Path,
    predictions_by_id: Mapping[str, Mapping[str, Any]],
    selected_ids: set[str],
    *,
    expected_samples: int,
    error_attempts: int,
    status: str,
) -> dict[str, Any]:
    predictions = [
        dict(item)
        for sample_id, item in predictions_by_id.items()
        if sample_id in selected_ids
    ]
    metrics = aggregate_verifier_dev_metrics(
        predictions,
        expected_samples=expected_samples,
        error_attempts=error_attempts,
        status=status,
    )
    write_json_atomic(path, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    dataset_path = project_path(args.dataset)
    records, summary = validate_dataset(dataset_path)
    print(f"Dataset:   {dataset_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.preflight_only:
        print("Preflight complete: no model was loaded.")
        return

    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = dict(config["model"])
    if args.model_id is not None:
        model_config["model_id"] = args.model_id
    model_config["max_new_tokens"] = args.max_new_tokens
    model_config["local_files_only"] = bool(
        args.local_files_only
        or model_config.get("local_files_only", False)
    )
    model_name = Path(model_config["model_id"]).name
    run_name = args.run_name or (
        f"qwen-baseline-dev110__{slugify(model_name)}"
    )
    run_dir = project_path(args.output_dir) / slugify(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_config_path = run_dir / "run_config.json"
    run_config = {
        "created_at_utc": utc_now(),
        "protocol": VERIFIER_DEV_BASELINE_PROTOCOL,
        "dataset": str(dataset_path),
        "dataset_sha256": summary["dataset_sha256"],
        "dataset_manifest": summary["manifest"],
        "dataset_manifest_sha256": summary["manifest_sha256"],
        "selected_ids_sha256": summary["selected_ids_sha256"],
        "model_config": str(config_path),
        "model_config_sha256": sha256sum(config_path),
        "model_id": model_config["model_id"],
        "torch_dtype": model_config.get("torch_dtype", "auto"),
        "device_map": model_config.get("device_map", "auto"),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(model_config.get("do_sample", False)),
        "local_files_only": model_config["local_files_only"],
        "system_prompt": VERIFIER_DEV_SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(
            VERIFIER_DEV_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    if run_config["do_sample"]:
        raise RuntimeError("Verifier Dev baseline requires deterministic decoding.")
    validate_or_create_run_config(run_config_path, run_config)

    predictions_by_id = load_predictions(predictions_path)
    selected_ids = {str(item["id"]) for item in records}
    completed_ids = selected_ids.intersection(predictions_by_id)
    pending = [
        item for item in records if str(item["id"]) not in completed_ids
    ]
    historical_errors = count_jsonl(errors_path)
    print(f"Run dir:   {run_dir}")
    print(f"Selected:  {len(records)}")
    print(f"Completed: {len(completed_ids)}")
    print(f"Pending:   {len(pending)}")
    if not pending:
        metrics = save_metrics(
            metrics_path,
            predictions_by_id,
            selected_ids,
            expected_samples=len(records),
            error_attempts=historical_errors,
            status="completed",
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
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "VLM runtime dependencies are missing for Verifier Dev."
        ) from error
    runner = VlmBaseline(
        VlmBaselineConfig(
            model_id=model_config["model_id"],
            torch_dtype=model_config.get("torch_dtype", "auto"),
            device_map=model_config.get("device_map", "auto"),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            local_files_only=model_config["local_files_only"],
        )
    )
    add_runtime_metadata(
        run_config_path, runner, torch, transformers.__version__
    )

    invocation_errors = 0
    successes_since_save = 0
    status = "completed"
    fatal_error: Exception | None = None
    with predictions_path.open(
        "a", encoding="utf-8", buffering=1
    ) as predictions_file, errors_path.open(
        "a", encoding="utf-8", buffering=1
    ) as errors_file:
        try:
            for sample in tqdm(
                pending, desc="Verifier Dev baseline", unit="sample"
            ):
                try:
                    result = runner.answer(
                        project_path(str(sample["image"])),
                        str(sample["question"]),
                        system_prompt=VERIFIER_DEV_SYSTEM_PROMPT,
                    )
                    evaluation = evaluate_answer(
                        result["answer"], str(sample["gt_answer"])
                    )
                    prediction = {
                        **dict(sample),
                        "prediction": result["answer"],
                        "evaluation": evaluation,
                        "model": result["model"],
                        "latency_seconds": result.get(
                            "end_to_end_latency_seconds",
                            result["latency_seconds"],
                        ),
                        "generation_latency_seconds": result[
                            "latency_seconds"
                        ],
                        "device": result["device"],
                        "cuda_available": result["cuda_available"],
                        "evaluated_at_utc": utc_now(),
                    }
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
                    predictions_by_id[str(sample["id"])] = prediction
                    successes_since_save += 1
                    if successes_since_save >= args.save_every:
                        save_metrics(
                            metrics_path,
                            predictions_by_id,
                            selected_ids,
                            expected_samples=len(records),
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
                            "id": sample["id"],
                            "image": sample["image"],
                            "object": sample["object"],
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(limit=5),
                            "attempted_at_utc": utc_now(),
                        },
                    )
                    tqdm.write(
                        f"ERROR {sample['id']}: "
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

    completed = len(selected_ids.intersection(predictions_by_id))
    if completed != len(records) and status == "completed":
        status = "incomplete"
    metrics = save_metrics(
        metrics_path,
        predictions_by_id,
        selected_ids,
        expected_samples=len(records),
        error_attempts=historical_errors + invocation_errors,
        status=status,
    )
    print(f"Predictions: {predictions_path}")
    print(f"Errors:      {errors_path}")
    print(f"Metrics:     {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if fatal_error is not None:
        raise fatal_error
    if status != "completed":
        raise RuntimeError(
            f"Verifier Dev ended with status={status}; repeat the identical "
            "command to resume."
        )


if __name__ == "__main__":
    main()
