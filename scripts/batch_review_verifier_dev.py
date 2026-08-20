"""Run resumable Qwen semantic review over the Dev candidate union."""

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
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_eval_pope_verifier import (
    append_jsonl,
    count_jsonl,
    project_path,
    utc_now,
    write_json_atomic,
)
from scripts.batch_ground_verifier_dev import (
    validate_baseline_source,
)
from grounded_visual_assistant.evaluation import parse_yes_no
from grounded_visual_assistant.pope_dataset import (
    read_json_records,
    sha256sum,
)
from grounded_visual_assistant.pope_verifier_evaluation import (
    verification_query_key,
)
from grounded_visual_assistant.semantic_answer_verifier import (
    SEMANTIC_REVIEW_SYSTEM_PROMPT,
    SemanticAnswerVerifierConfig,
    write_semantic_crop,
)
from grounded_visual_assistant.verifier_dev_grounding import (
    VERIFIER_DEV_GROUNDING_PROTOCOL,
)
from grounded_visual_assistant.verifier_dev_semantic_review import (
    VERIFIER_DEV_SEMANTIC_REVIEW_PROTOCOL,
    aggregate_dev_semantic_review_metrics,
    build_dev_semantic_review_jobs,
    ordered_candidate_keys_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review the complete Verifier Dev grounding-candidate union "
            "with deterministic Qwen crop Yes/No inference."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/verifier_dev_semantic_review_v1.yaml",
    )
    parser.add_argument("--baseline-predictions", default=None)
    parser.add_argument("--grounding-run-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.save_every <= 0:
        parser.error("--save-every must be positive.")
    if args.max_errors < 0:
        parser.error("--max-errors must be zero or greater.")
    return args


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")
    return slug.lower() or "run"


def load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    reviews = {}
    for item in read_json_records(path):
        key = str(item["candidate_key"])
        if key in reviews:
            raise ValueError(f"Duplicate Dev semantic review: {key}")
        reviews[key] = item
    return reviews


def load_settings(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
    Path,
    SemanticAnswerVerifierConfig,
    dict[str, Any],
    dict[str, Any],
]:
    config_path = project_path(args.config)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != VERIFIER_DEV_SEMANTIC_REVIEW_PROTOCOL:
        raise ValueError(
            f"Unsupported Dev semantic protocol: {payload.get('protocol')}"
        )
    inputs = dict(payload["inputs"])
    union = dict(payload["candidate_union"])
    semantic_model = dict(payload["semantic_model"])
    runtime = dict(payload["runtime"])
    baseline_path = project_path(
        args.baseline_predictions or inputs["baseline_predictions"]
    )
    grounding_run_dir = project_path(
        args.grounding_run_dir or inputs["grounding_run_dir"]
    )
    output_dir = project_path(args.output_dir or runtime["output_dir"])
    verifier_config = SemanticAnswerVerifierConfig(
        evidence_score_threshold=float(
            union["evidence_score_threshold"]
        ),
        min_mask_score=union.get("min_mask_score"),
        min_mask_area_ratio=float(union["min_mask_area_ratio"]),
        max_mask_area_ratio=float(union["max_mask_area_ratio"]),
        max_candidates_per_query=int(
            union["max_candidates_per_query"]
        ),
        crop_padding_ratio=float(union["crop_padding_ratio"]),
        min_crop_size=int(union["min_crop_size"]),
        require_exact_semantic_answer=bool(
            union.get("require_exact_semantic_answer", True)
        ),
    )
    model_config_path = project_path(semantic_model["config"])
    model_yaml = yaml.safe_load(
        model_config_path.read_text(encoding="utf-8")
    )
    model = dict(model_yaml["model"])
    if args.model_id is not None:
        model["model_id"] = args.model_id
    if args.torch_dtype is not None:
        model["torch_dtype"] = args.torch_dtype
    if args.device_map is not None:
        model["device_map"] = args.device_map
    model["max_new_tokens"] = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(semantic_model["max_new_tokens"])
    )
    model["do_sample"] = bool(semantic_model.get("do_sample", False))
    model["local_files_only"] = bool(
        args.local_files_only or model.get("local_files_only", False)
    )
    if model["max_new_tokens"] <= 0:
        raise ValueError("Semantic max_new_tokens must be positive.")
    if model["do_sample"]:
        raise ValueError("Dev semantic review requires deterministic decoding.")
    sources = {
        "dev_semantic_config": str(config_path),
        "dev_semantic_config_sha256": sha256sum(config_path),
        "semantic_model_config": str(model_config_path),
        "semantic_model_config_sha256": sha256sum(model_config_path),
    }
    return (
        baseline_path,
        grounding_run_dir,
        output_dir,
        verifier_config,
        model,
        sources,
    )


def validate_grounding_source(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence_path = run_dir / "evidence.jsonl"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "run_config.json"
    for path in (evidence_path, metrics_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Dev grounding artifact missing: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    if metrics.get("protocol") != VERIFIER_DEV_GROUNDING_PROTOCOL:
        raise RuntimeError("Dev grounding metrics protocol mismatch.")
    if metrics.get("status") != "completed":
        raise RuntimeError("Dev grounding metrics are not completed.")
    coverage = metrics.get("coverage") or {}
    if (
        coverage.get("completed_queries") != 57
        or coverage.get("remaining_queries") != 0
    ):
        raise RuntimeError("Dev grounding must contain all 57 queries.")
    if run_config.get("protocol") != VERIFIER_DEV_GROUNDING_PROTOCOL:
        raise RuntimeError("Dev grounding run config protocol mismatch.")
    records = read_json_records(evidence_path)
    if len(records) != 57:
        raise RuntimeError(
            f"Dev grounding evidence must contain 57 rows, found "
            f"{len(records)}."
        )
    return records, {
        "grounding_run_dir": str(run_dir),
        "grounding_evidence": str(evidence_path),
        "grounding_evidence_sha256": sha256sum(evidence_path),
        "grounding_metrics": str(metrics_path),
        "grounding_metrics_sha256": sha256sum(metrics_path),
        "grounding_run_config": str(config_path),
        "grounding_run_config_sha256": sha256sum(config_path),
    }


def validate_alignment(
    evidence: list[dict[str, Any]],
    baseline_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    for record in evidence:
        baseline_id = str(record["baseline_id"])
        if baseline_id not in baseline_by_id:
            raise RuntimeError(
                f"Dev evidence baseline ID is missing: {baseline_id}."
            )
        baseline = baseline_by_id[baseline_id]
        if parse_yes_no(str(baseline["prediction"])) != "no":
            raise RuntimeError(
                f"Dev evidence is not a baseline-No query: {baseline_id}."
            )
        expected = {
            "query_key": verification_query_key(baseline),
            "image": str(baseline["image"]),
            "image_id": int(baseline["image_id"]),
            "question": str(baseline["question"]),
            "object": str(baseline["object"]),
        }
        observed = {
            key: (
                int(record[key]) if key == "image_id" else str(record[key])
            )
            for key in expected
        }
        if observed != expected:
            raise RuntimeError(
                f"Dev evidence alignment mismatch for {baseline_id}."
            )


def validate_or_create_run_config(
    path: Path, current: Mapping[str, Any]
) -> None:
    immutable = (
        "protocol",
        "baseline_predictions_sha256",
        "baseline_metrics_sha256",
        "baseline_run_config_sha256",
        "grounding_evidence_sha256",
        "grounding_metrics_sha256",
        "grounding_run_config_sha256",
        "dev_semantic_config_sha256",
        "semantic_model_config_sha256",
        "ordered_candidate_keys_sha256",
        "semantic_review_system_prompt_sha256",
        "model_id",
        "torch_dtype",
        "device_map",
        "max_new_tokens",
        "do_sample",
        "evidence_score_threshold",
        "min_mask_score",
        "min_mask_area_ratio",
        "max_mask_area_ratio",
        "max_candidates_per_query",
        "crop_padding_ratio",
        "min_crop_size",
        "require_exact_semantic_answer",
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
                "The Dev semantic run is incompatible. Choose a new "
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
        "semantic_model_class": type(runner.model).__name__,
    }
    write_json_atomic(path, payload)


def prepare_crop(
    job: Mapping[str, Any],
    *,
    crop_dir: Path,
    config: SemanticAnswerVerifierConfig,
) -> dict[str, Any]:
    crop_path = crop_dir / f"{job['candidate_key']}.jpg"
    metadata = write_semantic_crop(
        project_path(str(job["image"])),
        crop_path,
        box=job["bbox"],
        config=config,
    )
    metadata["crop_sha256"] = sha256sum(crop_path)
    return metadata


def save_metrics(
    path: Path,
    *,
    reviews_by_key: Mapping[str, Mapping[str, Any]],
    jobs: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    baseline_by_id: Mapping[str, Mapping[str, Any]],
    error_attempts: int,
    status: str,
) -> dict[str, Any]:
    selected_keys = {str(item["candidate_key"]) for item in jobs}
    reviews = [
        dict(item)
        for key, item in reviews_by_key.items()
        if key in selected_keys
    ]
    metrics = aggregate_dev_semantic_review_metrics(
        reviews,
        jobs=jobs,
        evidence_records=evidence,
        baseline_by_id=baseline_by_id,
        error_attempts=error_attempts,
        status=status,
    )
    write_json_atomic(path, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    (
        baseline_path,
        grounding_run_dir,
        output_dir,
        verifier_config,
        model_config,
        config_sources,
    ) = load_settings(args)
    baseline_records, baseline_source = validate_baseline_source(
        baseline_path
    )
    baseline_by_id = {
        str(item["id"]): item for item in baseline_records
    }
    evidence, grounding_source = validate_grounding_source(
        grounding_run_dir
    )
    validate_alignment(evidence, baseline_by_id)
    jobs = build_dev_semantic_review_jobs(
        evidence, config=verifier_config
    )
    summary = {
        "grounding_queries": len(evidence),
        "candidate_queries": len(
            {str(item["query_key"]) for item in jobs}
        ),
        "candidate_reviews": len(jobs),
        "images": len({str(item["image"]) for item in jobs}),
        "objects": len({str(item["object"]) for item in jobs}),
        "ordered_candidate_keys_sha256": (
            ordered_candidate_keys_sha256(jobs)
        ),
        "inference_jobs_use_ground_truth": False,
    }
    for job in jobs:
        image_path = project_path(str(job["image"]))
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Dev semantic image is missing: {image_path}"
            )
    print(f"Baseline:  {baseline_path}")
    print(f"Evidence:  {grounding_source['grounding_evidence']}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.preflight_only:
        print("Preflight complete: no model was loaded and no crop was written.")
        return

    model_name = Path(model_config["model_id"]).name
    run_name = args.run_name or (
        f"semantic-review-dev{len(jobs)}__{model_name}"
    )
    run_dir = output_dir / slugify(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = run_dir / "crops"
    reviews_path = run_dir / "semantic_reviews.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_config_path = run_dir / "run_config.json"
    run_config = {
        "created_at_utc": utc_now(),
        "protocol": VERIFIER_DEV_SEMANTIC_REVIEW_PROTOCOL,
        **baseline_source,
        **grounding_source,
        **config_sources,
        "expected_candidates": len(jobs),
        "ordered_candidate_keys_sha256": summary[
            "ordered_candidate_keys_sha256"
        ],
        "semantic_review_system_prompt": SEMANTIC_REVIEW_SYSTEM_PROMPT,
        "semantic_review_system_prompt_sha256": hashlib.sha256(
            SEMANTIC_REVIEW_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "model_id": model_config["model_id"],
        "torch_dtype": model_config.get("torch_dtype", "auto"),
        "device_map": model_config.get("device_map", "auto"),
        "max_new_tokens": model_config["max_new_tokens"],
        "do_sample": model_config["do_sample"],
        "local_files_only": model_config["local_files_only"],
        "evidence_score_threshold": (
            verifier_config.evidence_score_threshold
        ),
        "min_mask_score": verifier_config.min_mask_score,
        "min_mask_area_ratio": verifier_config.min_mask_area_ratio,
        "max_mask_area_ratio": verifier_config.max_mask_area_ratio,
        "max_candidates_per_query": (
            verifier_config.max_candidates_per_query
        ),
        "crop_padding_ratio": verifier_config.crop_padding_ratio,
        "min_crop_size": verifier_config.min_crop_size,
        "require_exact_semantic_answer": (
            verifier_config.require_exact_semantic_answer
        ),
    }
    validate_or_create_run_config(run_config_path, run_config)

    reviews_by_key = load_reviews(reviews_path)
    expected_keys = {str(item["candidate_key"]) for item in jobs}
    completed_keys = expected_keys.intersection(reviews_by_key)
    pending = [
        item
        for item in jobs
        if str(item["candidate_key"]) not in completed_keys
    ]
    historical_errors = count_jsonl(errors_path)
    print(f"Run dir:   {run_dir}")
    print(f"Reviews:   {len(jobs)}")
    print(f"Completed: {len(completed_keys)}")
    print(f"Pending:   {len(pending)}")
    if not pending:
        metrics = save_metrics(
            metrics_path,
            reviews_by_key=reviews_by_key,
            jobs=jobs,
            evidence=evidence,
            baseline_by_id=baseline_by_id,
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
            "VLM runtime dependencies are missing for Dev semantic review."
        ) from error

    crop_metadata = {
        str(job["candidate_key"]): prepare_crop(
            job, crop_dir=crop_dir, config=verifier_config
        )
        for job in pending
    }
    runner = VlmBaseline(
        VlmBaselineConfig(
            model_id=model_config["model_id"],
            torch_dtype=model_config.get("torch_dtype", "auto"),
            device_map=model_config.get("device_map", "auto"),
            max_new_tokens=model_config["max_new_tokens"],
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
    with reviews_path.open(
        "a", encoding="utf-8", buffering=1
    ) as reviews_file, errors_path.open(
        "a", encoding="utf-8", buffering=1
    ) as errors_file:
        try:
            for job in tqdm(
                pending, desc="Verifier Dev semantic", unit="crop"
            ):
                try:
                    crop = crop_metadata[str(job["candidate_key"])]
                    result = runner.answer(
                        crop["crop_image"],
                        str(job["semantic_question"]),
                        system_prompt=SEMANTIC_REVIEW_SYSTEM_PROMPT,
                    )
                    review = {
                        **dict(job),
                        **crop,
                        "answer": result["answer"],
                        "parsed_answer": parse_yes_no(
                            str(result["answer"])
                        ),
                        "model": result["model"],
                        "latency_seconds": result["latency_seconds"],
                        "end_to_end_latency_seconds": result.get(
                            "end_to_end_latency_seconds",
                            result["latency_seconds"],
                        ),
                        "generated_tokens": result.get("generated_tokens"),
                        "max_new_tokens": result.get("max_new_tokens"),
                        "hit_max_new_tokens": result.get(
                            "hit_max_new_tokens"
                        ),
                        "cuda_peak_memory_allocated_gb": result.get(
                            "cuda_peak_memory_allocated_gb"
                        ),
                        "cuda_memory_reserved_gb": result.get(
                            "cuda_memory_reserved_gb"
                        ),
                        "reviewed_at_utc": utc_now(),
                    }
                    append_jsonl(reviews_file, review)
                    reviews_by_key[str(job["candidate_key"])] = review
                    successes_since_save += 1
                    if successes_since_save >= args.save_every:
                        save_metrics(
                            metrics_path,
                            reviews_by_key=reviews_by_key,
                            jobs=jobs,
                            evidence=evidence,
                            baseline_by_id=baseline_by_id,
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
                            "candidate_key": job["candidate_key"],
                            "query_key": job["query_key"],
                            "image": job["image"],
                            "object": job["object"],
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(limit=5),
                            "attempted_at_utc": utc_now(),
                        },
                    )
                    tqdm.write(
                        f"ERROR {job['candidate_key']}: "
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

    completed_keys = expected_keys.intersection(reviews_by_key)
    if len(completed_keys) != len(expected_keys) and status == "completed":
        status = "incomplete"
    metrics = save_metrics(
        metrics_path,
        reviews_by_key=reviews_by_key,
        jobs=jobs,
        evidence=evidence,
        baseline_by_id=baseline_by_id,
        error_attempts=historical_errors + invocation_errors,
        status=status,
    )
    print(f"Reviews: {reviews_path}")
    print(f"Errors:  {errors_path}")
    print(f"Metrics: {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if fatal_error is not None:
        raise fatal_error
    if status != "completed":
        raise RuntimeError(
            f"Dev semantic review ended with status={status}; repeat the "
            "identical command to resume."
        )


if __name__ == "__main__":
    main()
