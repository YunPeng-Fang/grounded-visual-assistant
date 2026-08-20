"""Run an official-compatible, resumable POPE baseline evaluation."""

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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_dataset import (
    POPE_STRATEGIES,
    read_json_records,
    sha256sum,
)
from grounded_visual_assistant.pope_evaluation import (
    POPE_PROTOCOL,
    POPE_SYSTEM_PROMPT,
    aggregate_metrics,
    evaluate_answer,
    select_records,
    selected_ids_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local VLM on the official COCO POPE benchmark."
    )
    parser.add_argument("--dataset", default="data/pope/questions.jsonl")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--output-dir", default="outputs/eval_pope_v0")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--strategy",
        choices=("all", *POPE_STRATEGIES),
        default="all",
    )
    parser.add_argument(
        "--samples-per-strategy",
        type=int,
        default=None,
        help="Select the first N questions from every requested strategy.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=30)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require all 3000 questions for every requested strategy.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.samples_per_strategy is not None and args.samples_per_strategy <= 0:
        parser.error("--samples-per-strategy must be positive.")
    if args.max_new_tokens <= 0 or args.save_every <= 0:
        parser.error("--max-new-tokens and --save-every must be positive.")
    if args.max_errors < 0:
        parser.error("--max-errors must be zero or greater.")
    if args.require_complete and args.samples_per_strategy is not None:
        parser.error(
            "--require-complete cannot be combined with "
            "--samples-per-strategy."
        )
    return args


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")
    return slug.lower() or "run"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    predictions = {}
    for item in read_json_records(path):
        sample_id = str(item["id"])
        if sample_id in predictions:
            raise ValueError(f"Duplicate saved POPE prediction: {sample_id}")
        predictions[sample_id] = item
    return predictions


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def verify_dataset_manifest(dataset_path: Path) -> dict[str, Any]:
    manifest_path = dataset_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"POPE manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = (manifest.get("artifact_sha256") or {}).get(dataset_path.name)
    actual = sha256sum(dataset_path)
    if expected != actual:
        raise RuntimeError(
            "POPE questions hash does not match manifest; rerun the offline "
            "data audit before evaluation."
        )
    return {
        "path": str(manifest_path),
        "sha256": sha256sum(manifest_path),
        "questions_sha256": actual,
    }


def preflight(
    records: list[dict[str, Any]],
    *,
    require_complete: bool,
    requested_strategy: str,
) -> dict[str, Any]:
    required_fields = {
        "id",
        "strategy",
        "image",
        "question",
        "object",
        "gt_answer",
    }
    missing_images = []
    images = set()
    labels: dict[str, Counter] = {
        strategy: Counter() for strategy in POPE_STRATEGIES
    }
    strategies = Counter()
    for index, item in enumerate(records, start=1):
        missing = required_fields - item.keys()
        if missing:
            raise ValueError(
                f"Selected POPE record {index} is missing {sorted(missing)}."
            )
        strategy = str(item["strategy"])
        if strategy not in POPE_STRATEGIES:
            raise ValueError(f"Unsupported POPE strategy: {strategy}")
        label = str(item["gt_answer"]).lower()
        if label not in {"yes", "no"}:
            raise ValueError(f"Unsupported POPE label: {label}")
        image_path = project_path(str(item["image"]))
        if not image_path.is_file():
            missing_images.append(str(image_path))
        else:
            images.add(str(image_path.resolve()))
        labels[strategy][label] += 1
        strategies[strategy] += 1
    if missing_images:
        raise FileNotFoundError(
            f"POPE references {len(missing_images)} missing images; first: "
            f"{missing_images[0]}"
        )
    if require_complete:
        requested = (
            POPE_STRATEGIES
            if requested_strategy == "all"
            else (requested_strategy,)
        )
        for strategy in requested:
            if strategies[strategy] != 3000:
                raise RuntimeError(
                    f"Complete POPE {strategy} requires 3000 questions, "
                    f"found {strategies[strategy]}."
                )
            if labels[strategy] != Counter({"yes": 1500, "no": 1500}):
                raise RuntimeError(
                    f"Complete POPE {strategy} labels are not balanced: "
                    f"{dict(labels[strategy])}."
                )
    return {
        "questions": len(records),
        "images": len(images),
        "strategies": dict(sorted(strategies.items())),
        "labels": {
            strategy: dict(sorted(counts.items()))
            for strategy, counts in labels.items()
            if counts
        },
        "selected_ids_sha256": selected_ids_sha256(records),
        "complete_protocol_required": require_complete,
    }


def validate_or_create_run_config(
    path: Path, current: dict[str, Any]
) -> None:
    immutable_keys = (
        "protocol",
        "dataset_sha256",
        "dataset_manifest_sha256",
        "selected_ids_sha256",
        "model_id",
        "torch_dtype",
        "device_map",
        "max_new_tokens",
        "do_sample",
        "strategy",
        "samples_per_strategy",
        "require_complete",
        "system_prompt_sha256",
    )
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        differences = {
            key: {"existing": existing.get(key), "current": current.get(key)}
            for key in immutable_keys
            if existing.get(key) != current.get(key)
        }
        if differences:
            raise RuntimeError(
                "The POPE run directory is incompatible. Choose a new "
                f"--run-name. Differences: {differences}"
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
    predictions_by_id: dict[str, dict[str, Any]],
    selected_ids: set[str],
    expected_samples: int,
    error_attempts: int,
    status: str,
) -> dict[str, Any]:
    predictions = [
        item
        for sample_id, item in predictions_by_id.items()
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
    dataset_path = project_path(args.dataset)
    records = select_records(
        read_json_records(dataset_path),
        strategy=args.strategy,
        samples_per_strategy=args.samples_per_strategy,
    )
    manifest = verify_dataset_manifest(dataset_path)
    dataset_summary = preflight(
        records,
        require_complete=args.require_complete,
        requested_strategy=args.strategy,
    )
    print(f"Dataset:   {dataset_path}")
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    print("Manifest:  artifact hash verified")
    if args.preflight_only:
        print("Preflight complete: no model was loaded.")
        return

    config = yaml.safe_load(
        project_path(args.config).read_text(encoding="utf-8")
    )
    model_config = dict(config["model"])
    if args.model_id is not None:
        model_config["model_id"] = args.model_id
    model_config["max_new_tokens"] = args.max_new_tokens
    model_name = Path(model_config["model_id"]).name
    selection_name = (
        "full"
        if args.samples_per_strategy is None
        else f"smoke-{args.samples_per_strategy}-per-strategy"
    )
    run_name = args.run_name or (
        f"pope-{args.strategy}-{selection_name}__{slugify(model_name)}"
    )
    run_dir = project_path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_config_path = run_dir / "run_config.json"
    run_config = {
        "created_at_utc": utc_now(),
        "protocol": POPE_PROTOCOL,
        "dataset": str(dataset_path),
        "dataset_sha256": manifest["questions_sha256"],
        "dataset_manifest_sha256": manifest["sha256"],
        "selected_ids_sha256": dataset_summary["selected_ids_sha256"],
        "model_id": model_config["model_id"],
        "torch_dtype": model_config.get("torch_dtype", "auto"),
        "device_map": model_config.get("device_map", "auto"),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(model_config.get("do_sample", False)),
        "strategy": args.strategy,
        "samples_per_strategy": args.samples_per_strategy,
        "require_complete": args.require_complete,
        "system_prompt": POPE_SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(
            POPE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    validate_or_create_run_config(run_config_path, run_config)

    predictions_by_id = load_predictions(predictions_path)
    selected_ids = {str(item["id"]) for item in records}
    completed_ids = set(predictions_by_id).intersection(selected_ids)
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
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "VLM runtime dependencies are missing. Install the project "
            "requirements before POPE evaluation."
        ) from error

    runner = VlmBaseline(
        VlmBaselineConfig(
            model_id=model_config["model_id"],
            torch_dtype=model_config.get("torch_dtype", "auto"),
            device_map=model_config.get("device_map", "auto"),
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(model_config.get("do_sample", False)),
            local_files_only=bool(
                args.local_files_only
                or model_config.get("local_files_only", False)
            ),
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
            for sample in tqdm(pending, desc="POPE", unit="sample"):
                try:
                    image_path = project_path(str(sample["image"]))
                    result = runner.answer(
                        image_path,
                        str(sample["question"]),
                        system_prompt=POPE_SYSTEM_PROMPT,
                    )
                    evaluation = evaluate_answer(
                        result["answer"], str(sample["gt_answer"])
                    )
                    prediction = {
                        "id": sample["id"],
                        "strategy": sample["strategy"],
                        "question_id": sample["question_id"],
                        "image": sample["image"],
                        "image_id": sample["image_id"],
                        "question": sample["question"],
                        "object": sample["object"],
                        "gt_answer": sample["gt_answer"],
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
                            len(records),
                            historical_errors + invocation_errors,
                            "running",
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
                            "strategy": sample["strategy"],
                            "image": sample["image"],
                            "question": sample["question"],
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
        except Exception as error:
            status = "failed"
            fatal_error = error

    completed = len(set(predictions_by_id).intersection(selected_ids))
    if status == "completed" and completed < len(records):
        status = "incomplete"
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
        raise fatal_error
    if status != "completed":
        raise RuntimeError(f"POPE evaluation ended with status: {status}")


if __name__ == "__main__":
    main()
