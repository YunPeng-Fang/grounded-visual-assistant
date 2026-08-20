"""Batch-evaluate Grounding-aware answer verification on saved POPE outputs."""

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
from typing import Any, Iterable, Mapping

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.grounding_answer_verifier import (
    GROUNDING_ANSWER_VERIFIER_PROTOCOL,
    GroundingAnswerVerifierConfig,
    compact_grounding_result,
)
from grounded_visual_assistant.pope_dataset import (
    POPE_STRATEGIES,
    read_json_records,
    sha256sum,
)
from grounded_visual_assistant.pope_evaluation import (
    POPE_PROTOCOL,
    evaluate_answer,
    select_records,
    selected_ids_sha256,
)
from grounded_visual_assistant.pope_verifier_evaluation import (
    POPE_VERIFIER_BATCH_PROTOCOL,
    aggregate_pope_verifier_metrics,
    build_verified_prediction,
    group_verification_queries,
    verification_query_key,
)


DEFAULT_BASELINE_PREDICTIONS = (
    "outputs/eval_pope_v0/pope-full9000__qwen3-vl-8b-instruct/"
    "predictions.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cached Grounding DINO + SAM 2.1 verification over saved POPE "
            "Qwen predictions."
        )
    )
    parser.add_argument(
        "--baseline-predictions",
        default=DEFAULT_BASELINE_PREDICTIONS,
    )
    parser.add_argument(
        "--config",
        default="configs/grounding_answer_verifier_v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval_pope_verifier_v1",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--strategy",
        choices=("all", *POPE_STRATEGIES),
        default="all",
    )
    parser.add_argument("--samples-per-strategy", type=int, default=None)
    parser.add_argument("--grounding-model-id", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-config", default=None)
    parser.add_argument("--box-threshold", type=float, default=None)
    parser.add_argument("--text-threshold", type=float, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    parser.add_argument("--evidence-score-threshold", type=float, default=None)
    parser.add_argument("--promotion-score-threshold", type=float, default=None)
    parser.add_argument("--min-mask-score", type=float, default=None)
    parser.add_argument("--min-mask-area-ratio", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default=None,
    )
    parser.add_argument("--save-every", type=int, default=30)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--visualize-limit", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.samples_per_strategy is not None and args.samples_per_strategy <= 0:
        parser.error("--samples-per-strategy must be positive.")
    if args.require_complete and args.samples_per_strategy is not None:
        parser.error(
            "--require-complete cannot be combined with "
            "--samples-per-strategy."
        )
    if args.save_every <= 0:
        parser.error("--save-every must be positive.")
    if args.max_errors < 0 or args.visualize_limit < 0:
        parser.error("--max-errors and --visualize-limit must be non-negative.")
    for name in (
        "box_threshold",
        "text_threshold",
        "nms_iou_threshold",
        "evidence_score_threshold",
        "promotion_score_threshold",
        "min_mask_score",
        "min_mask_area_ratio",
    ):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1].")
    return args


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def slugify(value: str) -> str:
    return (
        re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-_").lower()
        or "run"
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(
    path: Path, records: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def load_evidence(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    evidence = {}
    for record in read_json_records(path):
        key = str(record["query_key"])
        if key in evidence:
            raise ValueError(f"Duplicate saved verification query: {key}")
        evidence[key] = record
    return evidence


def query_keys_sha256(groups: Iterable[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(group["query_key"]) for group in groups) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_baseline_source(path: Path) -> dict[str, Any]:
    metrics_path = path.parent / "metrics.json"
    config_path = path.parent / "run_config.json"
    if not metrics_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "Baseline metrics.json and run_config.json must be beside "
            f"{path}."
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    if metrics.get("status") != "completed":
        raise RuntimeError("POPE baseline metrics are not completed.")
    coverage = metrics.get("coverage") or {}
    if coverage.get("completed") != coverage.get("expected"):
        raise RuntimeError("POPE baseline coverage is incomplete.")
    if run_config.get("protocol") != POPE_PROTOCOL:
        raise RuntimeError("Baseline run uses an unsupported POPE protocol.")
    return {
        "baseline_predictions": str(path),
        "baseline_predictions_sha256": sha256sum(path),
        "baseline_metrics": str(metrics_path),
        "baseline_metrics_sha256": sha256sum(metrics_path),
        "baseline_run_config": str(config_path),
        "baseline_run_config_sha256": sha256sum(config_path),
        "baseline_model_id": run_config.get("model_id"),
        "baseline_completed": coverage.get("completed"),
    }


def preflight(
    records: list[dict[str, Any]],
    *,
    require_complete: bool,
    requested_strategy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {
        "id",
        "strategy",
        "image",
        "image_id",
        "question",
        "object",
        "gt_answer",
        "prediction",
        "evaluation",
    }
    ids = []
    images = set()
    strategies = Counter()
    labels: dict[str, Counter[str]] = {
        strategy: Counter() for strategy in POPE_STRATEGIES
    }
    for index, record in enumerate(records, start=1):
        missing = required - record.keys()
        if missing:
            raise ValueError(
                f"Baseline record {index} misses {sorted(missing)}."
            )
        sample_id = str(record["id"])
        ids.append(sample_id)
        strategy = str(record["strategy"])
        if strategy not in POPE_STRATEGIES:
            raise ValueError(f"Unsupported POPE strategy: {strategy}")
        target = str(record["gt_answer"]).strip().lower()
        expected = evaluate_answer(str(record["prediction"]), target)
        for field, value in expected.items():
            if record["evaluation"].get(field) != value:
                raise RuntimeError(
                    f"Baseline evaluation does not reproduce for "
                    f"{sample_id} field {field!r}."
                )
        if not expected["strict_parse_valid"]:
            raise RuntimeError(
                f"Baseline answer is not strict Yes/No for {sample_id}."
            )
        image_path = project_path(str(record["image"]))
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Baseline image is missing: {image_path}"
            )
        images.add(str(image_path.resolve()))
        strategies[strategy] += 1
        labels[strategy][target] += 1
    if len(ids) != len(set(ids)):
        raise ValueError("Selected baseline predictions contain duplicate IDs.")

    if require_complete:
        requested = (
            POPE_STRATEGIES
            if requested_strategy == "all"
            else (requested_strategy,)
        )
        for strategy in requested:
            if strategies[strategy] != 3000:
                raise RuntimeError(
                    f"Complete {strategy} verification requires 3000 "
                    f"questions, found {strategies[strategy]}."
                )
            if labels[strategy] != Counter({"yes": 1500, "no": 1500}):
                raise RuntimeError(
                    f"Complete {strategy} labels are not balanced: "
                    f"{dict(labels[strategy])}."
                )
    groups = group_verification_queries(records)
    return (
        {
            "questions": len(records),
            "unique_queries": len(groups),
            "grounding_queries_saved": len(records) - len(groups),
            "images": len(images),
            "strategies": dict(sorted(strategies.items())),
            "labels": {
                strategy: dict(sorted(counts.items()))
                for strategy, counts in labels.items()
                if counts
            },
            "selected_ids_sha256": selected_ids_sha256(records),
            "selected_query_keys_sha256": query_keys_sha256(groups),
            "complete_protocol_required": require_complete,
        },
        groups,
    )


def load_settings(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], GroundingAnswerVerifierConfig, dict[str, Any]]:
    verifier_path = project_path(args.config)
    verifier_yaml = yaml.safe_load(
        verifier_path.read_text(encoding="utf-8")
    )
    if verifier_yaml.get("protocol") != GROUNDING_ANSWER_VERIFIER_PROTOCOL:
        raise ValueError(
            f"Unsupported verifier protocol: {verifier_yaml.get('protocol')}"
        )
    grounding_entry = dict(verifier_yaml["grounding"])
    grounding_path = project_path(grounding_entry["config"])
    grounding_yaml = yaml.safe_load(
        grounding_path.read_text(encoding="utf-8")
    )
    grounding = dict(grounding_yaml["grounding"])
    sam2 = dict(grounding_yaml["sam2"])
    runtime = dict(grounding_yaml["runtime"])
    verification = dict(verifier_yaml["verification"])

    grounding_settings = {
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
    verifier_config = GroundingAnswerVerifierConfig(
        evidence_score_threshold=(
            args.evidence_score_threshold
            if args.evidence_score_threshold is not None
            else float(verification["evidence_score_threshold"])
        ),
        promotion_score_threshold=(
            args.promotion_score_threshold
            if args.promotion_score_threshold is not None
            else float(verification["promotion_score_threshold"])
        ),
        min_mask_score=(
            args.min_mask_score
            if args.min_mask_score is not None
            else verification.get("min_mask_score")
        ),
        min_mask_area_ratio=(
            args.min_mask_area_ratio
            if args.min_mask_area_ratio is not None
            else float(verification.get("min_mask_area_ratio", 0.0))
        ),
    )
    if (
        grounding_settings["box_threshold"]
        > verifier_config.evidence_score_threshold
    ):
        raise ValueError(
            "Detector box_threshold cannot exceed the evidence threshold."
        )
    sources = {
        "verifier_config": str(verifier_path),
        "verifier_config_sha256": sha256sum(verifier_path),
        "grounding_config": str(grounding_path),
        "grounding_config_sha256": sha256sum(grounding_path),
    }
    return grounding_settings, verifier_config, sources


def validate_or_create_run_config(
    path: Path, current: Mapping[str, Any]
) -> None:
    immutable = (
        "protocol",
        "verifier_protocol",
        "baseline_predictions_sha256",
        "baseline_metrics_sha256",
        "baseline_run_config_sha256",
        "verifier_config_sha256",
        "grounding_config_sha256",
        "selected_ids_sha256",
        "selected_query_keys_sha256",
        "grounding_model_id",
        "sam2_checkpoint",
        "sam2_model_config",
        "box_threshold",
        "text_threshold",
        "nms_iou_threshold",
        "evidence_score_threshold",
        "promotion_score_threshold",
        "min_mask_score",
        "min_mask_area_ratio",
        "strategy",
        "samples_per_strategy",
        "require_complete",
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
                "The POPE verifier run is incompatible. Choose a new "
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


def materialize_outputs(
    *,
    records: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    evidence_by_key: Mapping[str, dict[str, Any]],
    verifier_config: GroundingAnswerVerifierConfig,
    predictions_path: Path,
    metrics_path: Path,
    error_attempts: int,
    status: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_keys = {str(group["query_key"]) for group in groups}
    available_keys = selected_keys.intersection(evidence_by_key)
    predictions = [
        build_verified_prediction(
            record,
            evidence_by_key[verification_query_key(record)],
            config=verifier_config,
        )
        for record in records
        if verification_query_key(record) in available_keys
    ]
    metrics = aggregate_pope_verifier_metrics(
        predictions,
        expected_samples=len(records),
        expected_queries=len(groups),
        completed_queries=len(available_keys),
        error_attempts=error_attempts,
        status=status,
    )
    write_jsonl_atomic(predictions_path, predictions)
    write_json_atomic(metrics_path, metrics)
    return metrics, predictions


def main() -> None:
    args = parse_args()
    baseline_path = project_path(args.baseline_predictions)
    baseline_source = validate_baseline_source(baseline_path)
    records = select_records(
        read_json_records(baseline_path),
        strategy=args.strategy,
        samples_per_strategy=args.samples_per_strategy,
    )
    summary, groups = preflight(
        records,
        require_complete=args.require_complete,
        requested_strategy=args.strategy,
    )
    grounding_settings, verifier_config, config_sources = load_settings(args)
    print(f"Baseline:  {baseline_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        "Verifier:  "
        f"evidence={verifier_config.evidence_score_threshold:.2f}, "
        f"promotion={verifier_config.promotion_score_threshold:.2f}"
    )
    if args.preflight_only:
        print("Preflight complete: no model was loaded.")
        return

    selection = (
        "full"
        if args.samples_per_strategy is None
        else f"smoke{args.samples_per_strategy * len(summary['strategies'])}"
    )
    run_name = args.run_name or (
        f"pope-{selection}__positive-rescue-v1"
        f"__promotion-{verifier_config.promotion_score_threshold:.2f}"
    )
    run_dir = project_path(args.output_dir) / slugify(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = run_dir / "evidence.jsonl"
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_config_path = run_dir / "run_config.json"
    visual_dir = run_dir / "visualizations"
    run_config = {
        "created_at_utc": utc_now(),
        "protocol": POPE_VERIFIER_BATCH_PROTOCOL,
        "verifier_protocol": GROUNDING_ANSWER_VERIFIER_PROTOCOL,
        **baseline_source,
        **config_sources,
        "selected_ids_sha256": summary["selected_ids_sha256"],
        "selected_query_keys_sha256": (
            summary["selected_query_keys_sha256"]
        ),
        "grounding_model_id": grounding_settings["grounding_model_id"],
        "sam2_checkpoint": grounding_settings["sam2_checkpoint"],
        "sam2_model_config": grounding_settings["sam2_model_config"],
        "box_threshold": grounding_settings["box_threshold"],
        "text_threshold": grounding_settings["text_threshold"],
        "nms_iou_threshold": grounding_settings["nms_iou_threshold"],
        "evidence_score_threshold": (
            verifier_config.evidence_score_threshold
        ),
        "promotion_score_threshold": (
            verifier_config.promotion_score_threshold
        ),
        "min_mask_score": verifier_config.min_mask_score,
        "min_mask_area_ratio": verifier_config.min_mask_area_ratio,
        "strategy": args.strategy,
        "samples_per_strategy": args.samples_per_strategy,
        "require_complete": args.require_complete,
        "device": grounding_settings["device"],
        "dtype": grounding_settings["dtype"],
        "local_files_only": grounding_settings["local_files_only"],
    }
    validate_or_create_run_config(run_config_path, run_config)

    evidence_by_key = load_evidence(evidence_path)
    selected_keys = {str(group["query_key"]) for group in groups}
    completed_keys = selected_keys.intersection(evidence_by_key)
    pending = [
        group for group in groups if group["query_key"] not in completed_keys
    ]
    historical_errors = count_jsonl(errors_path)
    print(f"Run dir:   {run_dir}")
    print(f"Questions: {len(records)}")
    print(f"Queries:   {len(groups)}")
    print(f"Completed: {len(completed_keys)}")
    print(f"Pending:   {len(pending)}")

    if not pending:
        metrics, _ = materialize_outputs(
            records=records,
            groups=groups,
            evidence_by_key=evidence_by_key,
            verifier_config=verifier_config,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
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
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Grounded-SAM-2 runtime dependencies are missing."
        ) from exc

    runner = GroundedSam2(GroundedSam2Config(**grounding_settings))
    add_runtime_metadata(
        run_config_path,
        runner,
        torch,
        transformers.__version__,
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
            for group in tqdm(
                pending, desc="POPE verifier", unit="query"
            ):
                try:
                    image_path = project_path(group["image"])
                    artifact_dir = None
                    if visualized < args.visualize_limit:
                        artifact_dir = visual_dir / str(group["query_key"])
                        visualized += 1
                    grounded = runner.predict(
                        image_path,
                        f"{group['object']}.",
                        output_dir=artifact_dir,
                    )
                    compact = compact_grounding_result(grounded)
                    evidence_record = {
                        "query_key": group["query_key"],
                        "baseline_ids": group["baseline_ids"],
                        "strategies": group["strategies"],
                        "image": group["image"],
                        "image_id": group["image_id"],
                        "question": group["question"],
                        "object": group["object"],
                        "grounding": compact,
                        "cuda_peak_memory_allocated_gb": grounded.get(
                            "cuda_peak_memory_allocated_gb"
                        ),
                        "evaluated_at_utc": utc_now(),
                    }
                    append_jsonl(evidence_file, evidence_record)
                    evidence_by_key[str(group["query_key"])] = evidence_record
                    successes_since_save += 1
                    if successes_since_save >= args.save_every:
                        materialize_outputs(
                            records=records,
                            groups=groups,
                            evidence_by_key=evidence_by_key,
                            verifier_config=verifier_config,
                            predictions_path=predictions_path,
                            metrics_path=metrics_path,
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
                            "query_key": group["query_key"],
                            "baseline_ids": group["baseline_ids"],
                            "image": group["image"],
                            "object": group["object"],
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(limit=5),
                            "attempted_at_utc": utc_now(),
                        },
                    )
                    tqdm.write(
                        f"ERROR {group['query_key']}: "
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

    completed_keys = selected_keys.intersection(evidence_by_key)
    if status == "completed" and len(completed_keys) < len(groups):
        status = "incomplete"
    metrics, _ = materialize_outputs(
        records=records,
        groups=groups,
        evidence_by_key=evidence_by_key,
        verifier_config=verifier_config,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        error_attempts=historical_errors + invocation_errors,
        status=status,
    )
    print(f"Evidence:    {evidence_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Errors:      {errors_path}")
    print(f"Metrics:     {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if fatal_error is not None:
        raise fatal_error
    if status != "completed":
        raise RuntimeError(
            f"POPE verifier evaluation ended with status: {status}"
        )


if __name__ == "__main__":
    main()
