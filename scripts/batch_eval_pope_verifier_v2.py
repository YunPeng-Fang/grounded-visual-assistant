"""Run resumable V2 semantic crop verification over frozen POPE evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_eval_pope_verifier import (
    append_jsonl,
    count_jsonl,
    load_evidence,
    preflight,
    project_path,
    slugify,
    utc_now,
    validate_baseline_source,
    write_json_atomic,
    write_jsonl_atomic,
)
from grounded_visual_assistant.evaluation import parse_yes_no
from grounded_visual_assistant.pope_dataset import (
    read_json_records,
    sha256sum,
)
from grounded_visual_assistant.pope_evaluation import select_records
from grounded_visual_assistant.pope_semantic_verifier_evaluation import (
    POPE_SEMANTIC_VERIFIER_BATCH_PROTOCOL,
    aggregate_pope_semantic_verifier_metrics,
    build_semantic_review_jobs,
    build_semantic_verified_prediction,
    required_semantic_candidate_keys,
)
from grounded_visual_assistant.pope_verifier_evaluation import (
    POPE_VERIFIER_BATCH_PROTOCOL,
    group_verification_queries,
    verification_query_key,
)
from grounded_visual_assistant.semantic_answer_verifier import (
    SEMANTIC_ANSWER_VERIFIER_PROTOCOL,
    SEMANTIC_REVIEW_SYSTEM_PROMPT,
    SemanticAnswerVerifierConfig,
    select_semantic_candidates,
    write_semantic_crop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm cached Grounded-SAM-2 candidates with deterministic "
            "Qwen crop-level Yes/No review."
        )
    )
    parser.add_argument(
        "--config", default="configs/grounding_answer_verifier_v2.yaml"
    )
    parser.add_argument("--baseline-predictions", default=None)
    parser.add_argument("--grounding-run-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--strategy",
        choices=("all", "random", "popular", "adversarial"),
        default="all",
    )
    parser.add_argument("--samples-per-strategy", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--evidence-score-threshold", type=float, default=None)
    parser.add_argument("--min-mask-score", type=float, default=None)
    parser.add_argument("--min-mask-area-ratio", type=float, default=None)
    parser.add_argument("--max-mask-area-ratio", type=float, default=None)
    parser.add_argument("--max-candidates-per-query", type=int, default=None)
    parser.add_argument("--crop-padding-ratio", type=float, default=None)
    parser.add_argument("--min-crop-size", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-errors", type=int, default=10)
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
    if args.max_errors < 0:
        parser.error("--max-errors must be zero or greater.")
    return args


def load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    reviews = {}
    for item in read_json_records(path):
        key = str(item["candidate_key"])
        if key in reviews:
            raise ValueError(f"Duplicate semantic review: {key}")
        reviews[key] = item
    return reviews


def hash_ordered_keys(items: Iterable[Mapping[str, Any]], key: str) -> str:
    payload = "\n".join(str(item[key]) for item in items) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    if payload.get("protocol") != SEMANTIC_ANSWER_VERIFIER_PROTOCOL:
        raise ValueError(
            f"Unsupported V2 protocol: {payload.get('protocol')}"
        )
    inputs = dict(payload["inputs"])
    verification = dict(payload["verification"])
    semantic_model = dict(payload["semantic_model"])
    runtime = dict(payload["runtime"])

    baseline_path = project_path(
        args.baseline_predictions or inputs["baseline_predictions"]
    )
    grounding_run_dir = project_path(
        args.grounding_run_dir or inputs["grounding_run_dir"]
    )
    output_dir = project_path(
        args.output_dir or runtime["output_dir"]
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
        raise ValueError("semantic max_new_tokens must be positive.")
    if model["do_sample"]:
        raise ValueError("V2 semantic review requires deterministic decoding.")

    verifier = SemanticAnswerVerifierConfig(
        evidence_score_threshold=(
            args.evidence_score_threshold
            if args.evidence_score_threshold is not None
            else float(verification["evidence_score_threshold"])
        ),
        min_mask_score=(
            args.min_mask_score
            if args.min_mask_score is not None
            else verification.get("min_mask_score")
        ),
        min_mask_area_ratio=(
            args.min_mask_area_ratio
            if args.min_mask_area_ratio is not None
            else float(verification["min_mask_area_ratio"])
        ),
        max_mask_area_ratio=(
            args.max_mask_area_ratio
            if args.max_mask_area_ratio is not None
            else float(verification["max_mask_area_ratio"])
        ),
        max_candidates_per_query=(
            args.max_candidates_per_query
            if args.max_candidates_per_query is not None
            else int(verification["max_candidates_per_query"])
        ),
        crop_padding_ratio=(
            args.crop_padding_ratio
            if args.crop_padding_ratio is not None
            else float(verification["crop_padding_ratio"])
        ),
        min_crop_size=(
            args.min_crop_size
            if args.min_crop_size is not None
            else int(verification["min_crop_size"])
        ),
        require_exact_semantic_answer=bool(
            verification.get("require_exact_semantic_answer", True)
        ),
    )
    sources = {
        "v2_config": str(config_path),
        "v2_config_sha256": sha256sum(config_path),
        "semantic_model_config": str(model_config_path),
        "semantic_model_config_sha256": sha256sum(model_config_path),
    }
    return (
        baseline_path,
        grounding_run_dir,
        output_dir,
        verifier,
        model,
        sources,
    )


def validate_grounding_source(run_dir: Path) -> dict[str, Any]:
    evidence_path = run_dir / "evidence.jsonl"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "run_config.json"
    for path in (evidence_path, metrics_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"V1 grounding artifact missing: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    if metrics.get("protocol") != POPE_VERIFIER_BATCH_PROTOCOL:
        raise RuntimeError("Grounding source uses an unsupported protocol.")
    if metrics.get("status") != "completed":
        raise RuntimeError("Grounding source is not completed.")
    coverage = metrics.get("coverage") or {}
    if (
        coverage.get("completed_unique_queries")
        != coverage.get("expected_unique_queries")
    ):
        raise RuntimeError("Grounding source query coverage is incomplete.")
    if run_config.get("protocol") != POPE_VERIFIER_BATCH_PROTOCOL:
        raise RuntimeError("Grounding run config protocol is incompatible.")
    return {
        "grounding_run_dir": str(run_dir),
        "grounding_evidence": str(evidence_path),
        "grounding_evidence_sha256": sha256sum(evidence_path),
        "grounding_metrics": str(metrics_path),
        "grounding_metrics_sha256": sha256sum(metrics_path),
        "grounding_run_config": str(config_path),
        "grounding_run_config_sha256": sha256sum(config_path),
    }


def validate_evidence_alignment(
    groups: Iterable[Mapping[str, Any]],
    evidence_by_key: Mapping[str, Mapping[str, Any]],
) -> None:
    for group in groups:
        key = str(group["query_key"])
        if key not in evidence_by_key:
            raise RuntimeError(f"V1 evidence is missing query {key}.")
        evidence = evidence_by_key[key]
        expected = {
            "image": str(group["image"]),
            "image_id": int(group["image_id"]),
            "question": str(group["question"]),
            "object": str(group["object"]),
        }
        observed = {
            "image": str(evidence["image"]),
            "image_id": int(evidence["image_id"]),
            "question": str(evidence["question"]),
            "object": str(evidence["object"]),
        }
        if observed != expected:
            raise RuntimeError(
                f"V1 evidence query mismatch for {key}: "
                f"{observed} != {expected}."
            )


def v2_preflight_summary(
    records: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    evidence_by_key: Mapping[str, Mapping[str, Any]],
    jobs: list[dict[str, Any]],
    config: SemanticAnswerVerifierConfig,
) -> dict[str, Any]:
    records_by_id = {str(item["id"]): item for item in records}
    negative_queries = 0
    no_evidence_queries = 0
    geometry_gated_queries = 0
    candidate_counts = Counter()
    for group in groups:
        group_records = [
            records_by_id[str(item)] for item in group["baseline_ids"]
        ]
        answers = {
            parse_yes_no(str(item["prediction"])) for item in group_records
        }
        if answers != {"no"}:
            continue
        negative_queries += 1
        grounding = evidence_by_key[str(group["query_key"])]["grounding"]
        candidates, rejected = select_semantic_candidates(
            grounding.get("annotations", []),
            target=str(group["object"]),
            image_width=int(grounding["img_width"]),
            image_height=int(grounding["img_height"]),
            config=config,
        )
        candidate_counts[len(candidates)] += 1
        if not candidates:
            reasons = {
                reason
                for item in rejected
                for reason in item.get("rejection_reasons", [])
            }
            if "large_mask" in reasons:
                geometry_gated_queries += 1
            else:
                no_evidence_queries += 1
    return {
        "negative_baseline_queries": negative_queries,
        "candidate_queries": len(
            {str(item["query_key"]) for item in jobs}
        ),
        "semantic_candidate_reviews": len(jobs),
        "semantic_candidate_keys_sha256": hash_ordered_keys(
            jobs, "candidate_key"
        ),
        "no_evidence_queries": no_evidence_queries,
        "geometry_gated_queries": geometry_gated_queries,
        "candidates_per_negative_query": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
    }


def validate_or_create_run_config(
    path: Path, current: Mapping[str, Any]
) -> None:
    immutable = (
        "protocol",
        "verifier_protocol",
        "baseline_predictions_sha256",
        "baseline_metrics_sha256",
        "baseline_run_config_sha256",
        "grounding_evidence_sha256",
        "grounding_metrics_sha256",
        "grounding_run_config_sha256",
        "v2_config_sha256",
        "semantic_model_config_sha256",
        "selected_ids_sha256",
        "selected_query_keys_sha256",
        "semantic_candidate_keys_sha256",
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
                "The V2 run directory is incompatible. Choose a new "
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


def materialize_outputs(
    *,
    records: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    evidence_by_key: Mapping[str, Mapping[str, Any]],
    jobs: list[dict[str, Any]],
    reviews_by_key: Mapping[str, Mapping[str, Any]],
    verifier_config: SemanticAnswerVerifierConfig,
    predictions_path: Path,
    metrics_path: Path,
    error_attempts: int,
    status: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = []
    for baseline in records:
        query_key = verification_query_key(baseline)
        evidence_record = evidence_by_key[query_key]
        required = required_semantic_candidate_keys(
            baseline, evidence_record, config=verifier_config
        )
        if any(key not in reviews_by_key for key in required):
            continue
        predictions.append(
            build_semantic_verified_prediction(
                baseline,
                evidence_record,
                reviews_by_key=reviews_by_key,
                config=verifier_config,
            )
        )
    completed_query_keys = {
        str(item["query_key"]) for item in predictions
    }
    expected_review_keys = {
        str(item["candidate_key"]) for item in jobs
    }
    completed_review_keys = expected_review_keys.intersection(reviews_by_key)
    metrics = aggregate_pope_semantic_verifier_metrics(
        predictions,
        expected_samples=len(records),
        expected_queries=len(groups),
        completed_queries=len(completed_query_keys),
        expected_reviews=len(expected_review_keys),
        completed_reviews=len(completed_review_keys),
        error_attempts=error_attempts,
        status=status,
    )
    write_jsonl_atomic(predictions_path, predictions)
    write_json_atomic(metrics_path, metrics)
    return metrics, predictions


def prepare_crop(
    job: Mapping[str, Any],
    *,
    crop_dir: Path,
    config: SemanticAnswerVerifierConfig,
) -> dict[str, Any]:
    crop_path = crop_dir / f"{job['candidate_key']}.jpg"
    if not crop_path.is_file():
        crop = write_semantic_crop(
            project_path(str(job["image"])),
            crop_path,
            box=job["bbox"],
            config=config,
        )
    else:
        crop = {
            "source_image": str(project_path(str(job["image"]))),
            "crop_image": str(crop_path),
            "source_box_xyxy": [
                round(float(value), 3) for value in job["bbox"]
            ],
        }
    crop["crop_sha256"] = sha256sum(crop_path)
    return crop


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
    baseline_source = validate_baseline_source(baseline_path)
    records = select_records(
        read_json_records(baseline_path),
        strategy=args.strategy,
        samples_per_strategy=args.samples_per_strategy,
    )
    base_summary, groups = preflight(
        records,
        require_complete=args.require_complete,
        requested_strategy=args.strategy,
    )
    grounding_source = validate_grounding_source(grounding_run_dir)
    evidence_by_key = load_evidence(
        Path(grounding_source["grounding_evidence"])
    )
    validate_evidence_alignment(groups, evidence_by_key)
    records_by_id = {str(item["id"]): item for item in records}
    jobs = build_semantic_review_jobs(
        groups,
        records_by_id=records_by_id,
        evidence_by_key=evidence_by_key,
        config=verifier_config,
    )
    semantic_summary = v2_preflight_summary(
        records,
        groups,
        evidence_by_key,
        jobs,
        verifier_config,
    )
    summary = {**base_summary, **semantic_summary}
    print(f"Baseline:  {baseline_path}")
    print(f"Evidence:  {grounding_source['grounding_evidence']}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        "V2 gate:   "
        f"evidence={verifier_config.evidence_score_threshold:.2f}, "
        f"max-area={verifier_config.max_mask_area_ratio:.2f}, "
        f"candidates={verifier_config.max_candidates_per_query}"
    )
    if args.preflight_only:
        print("Preflight complete: no model was loaded and no crop was written.")
        return

    selection = (
        "full"
        if args.samples_per_strategy is None
        else f"smoke{args.samples_per_strategy * len(summary['strategies'])}"
    )
    run_name = args.run_name or f"pope-{selection}__semantic-rescue-v2"
    run_dir = output_dir / slugify(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = run_dir / "crops"
    reviews_path = run_dir / "semantic_reviews.jsonl"
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_config_path = run_dir / "run_config.json"
    run_config = {
        "created_at_utc": utc_now(),
        "protocol": POPE_SEMANTIC_VERIFIER_BATCH_PROTOCOL,
        "verifier_protocol": SEMANTIC_ANSWER_VERIFIER_PROTOCOL,
        **baseline_source,
        **grounding_source,
        **config_sources,
        "selected_ids_sha256": summary["selected_ids_sha256"],
        "selected_query_keys_sha256": (
            summary["selected_query_keys_sha256"]
        ),
        "semantic_candidate_keys_sha256": (
            summary["semantic_candidate_keys_sha256"]
        ),
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
        "strategy": args.strategy,
        "samples_per_strategy": args.samples_per_strategy,
        "require_complete": args.require_complete,
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
    print(f"Questions: {len(records)}")
    print(f"Reviews:   {len(jobs)}")
    print(f"Completed: {len(completed_keys)}")
    print(f"Pending:   {len(pending)}")

    if not pending:
        metrics, _ = materialize_outputs(
            records=records,
            groups=groups,
            evidence_by_key=evidence_by_key,
            jobs=jobs,
            reviews_by_key=reviews_by_key,
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

        from grounded_visual_assistant.vlm_baseline import (
            VlmBaseline,
            VlmBaselineConfig,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "VLM runtime dependencies are missing for V2 review."
        ) from error

    for job in pending:
        prepare_crop(job, crop_dir=crop_dir, config=verifier_config)
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
            for job in tqdm(pending, desc="V2 semantic review", unit="crop"):
                try:
                    crop = prepare_crop(
                        job, crop_dir=crop_dir, config=verifier_config
                    )
                    result = runner.answer(
                        crop["crop_image"],
                        str(job["semantic_question"]),
                        system_prompt=SEMANTIC_REVIEW_SYSTEM_PROMPT,
                    )
                    parsed = parse_yes_no(str(result["answer"]))
                    review = {
                        **dict(job),
                        **crop,
                        "answer": result["answer"],
                        "parsed_answer": parsed,
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
                        materialize_outputs(
                            records=records,
                            groups=groups,
                            evidence_by_key=evidence_by_key,
                            jobs=jobs,
                            reviews_by_key=reviews_by_key,
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
    metrics, _ = materialize_outputs(
        records=records,
        groups=groups,
        evidence_by_key=evidence_by_key,
        jobs=jobs,
        reviews_by_key=reviews_by_key,
        verifier_config=verifier_config,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        error_attempts=historical_errors + invocation_errors,
        status=status,
    )
    print(f"Reviews:     {reviews_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Errors:      {errors_path}")
    print(f"Metrics:     {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if fatal_error is not None:
        raise fatal_error
    if status != "completed":
        raise RuntimeError(
            f"V2 run ended with status={status}; repeat the identical "
            "command to resume."
        )


if __name__ == "__main__":
    main()
