"""Run the Dev-only V3 contrastive candidate review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from collections import Counter
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
from scripts.batch_ground_verifier_dev import validate_baseline_source
from scripts.batch_review_verifier_dev import (
    validate_alignment,
    validate_grounding_source,
)
from scripts.compare_verifier_dev_ablations import validate_semantic_source
from grounded_visual_assistant.pope_dataset import (
    read_json_records,
    sha256sum,
)
from grounded_visual_assistant.verifier_dev_contrastive_review import (
    CONTRASTIVE_REVIEW_SYSTEM_PROMPT,
    VERIFIER_DEV_CONTRASTIVE_REVIEW_PROTOCOL,
    build_contrastive_review_jobs,
    evaluate_contrastive_cascade,
    ordered_v3_keys_sha256,
    parse_contrastive_answer,
    validate_coco_ontology,
    write_marked_candidate_crop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review V2-confirmed Dev candidates with a red-box, "
            "same-supercategory forced-choice prompt."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/verifier_dev_contrastive_review_v3.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=3)
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write GT-free jobs and marked crops without loading Qwen.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and args.prepare_only:
        parser.error("--preflight-only and --prepare-only are exclusive.")
    if args.save_every <= 0 or (
        args.max_new_tokens is not None and args.max_new_tokens <= 0
    ):
        parser.error("--save-every and --max-new-tokens must be positive.")
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
        key = str(item["v3_key"])
        if key in reviews:
            raise ValueError(f"Duplicate V3 contrastive review: {key}.")
        reviews[key] = item
    return reviews


def load_settings(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    config_path = project_path(args.config)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != VERIFIER_DEV_CONTRASTIVE_REVIEW_PROTOCOL:
        raise ValueError(
            f"Unsupported V3 protocol: {payload.get('protocol')}."
        )
    inputs = dict(payload["inputs"])
    cascade = dict(payload["v2_cascade"])
    contrastive = dict(payload["contrastive_review"])
    semantic_model = dict(payload["semantic_model"])
    selection = dict(payload["selection"])
    runtime = dict(payload["runtime"])
    if bool(selection.get("held_out_data_used_for_selection", True)):
        raise ValueError("V3 Dev selection cannot access held-out data.")

    ontology_path = project_path(inputs["ontology"])
    ontology_payload = yaml.safe_load(
        ontology_path.read_text(encoding="utf-8")
    )
    groups, category_to_group = validate_coco_ontology(ontology_payload)
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
        raise ValueError("V3 max_new_tokens must be positive.")
    if model["do_sample"]:
        raise ValueError("V3 requires deterministic decoding.")
    output_dir = project_path(args.output_dir or runtime["output_dir"])
    sources = {
        "v3_config": str(config_path),
        "v3_config_sha256": sha256sum(config_path),
        "ontology": str(ontology_path),
        "ontology_sha256": sha256sum(ontology_path),
        "semantic_model_config": str(model_config_path),
        "semantic_model_config_sha256": sha256sum(model_config_path),
    }
    ontology = {
        "groups": groups,
        "category_to_group": category_to_group,
    }
    return (
        config_path,
        project_path(inputs["baseline_predictions"]),
        project_path(inputs["grounding_run_dir"]),
        project_path(inputs["semantic_run_dir"]),
        ontology_path,
        output_dir,
        cascade,
        contrastive,
        model,
        {"selection": selection, "sources": sources, "ontology": ontology},
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
        "semantic_reviews_sha256",
        "semantic_metrics_sha256",
        "semantic_run_config_sha256",
        "v3_config_sha256",
        "ontology_sha256",
        "semantic_model_config_sha256",
        "ordered_v3_keys_sha256",
        "contrastive_review_system_prompt_sha256",
        "expected_reviews",
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
        "require_exact_v2_yes",
        "none_label",
        "marker_color",
        "marker_width",
        "require_exact_label",
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
                "The V3 run is incompatible. Choose a new --run-name. "
                f"Differences: {differences}"
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


def resolve_source_crop(
    semantic_run_dir: Path, job: Mapping[str, Any]
) -> Path:
    return semantic_run_dir / "crops" / Path(
        str(job["source_crop_image"])
    ).name


def prepare_marked_crop(
    job: Mapping[str, Any],
    *,
    semantic_run_dir: Path,
    crop_dir: Path,
    marker_color: list[int],
    marker_width: int,
) -> dict[str, Any]:
    source_crop = resolve_source_crop(semantic_run_dir, job)
    if not source_crop.is_file():
        raise FileNotFoundError(f"V3 source crop missing: {source_crop}.")
    if sha256sum(source_crop) != job["source_crop_sha256"]:
        raise RuntimeError(
            f"V3 source crop hash mismatch: {job['candidate_key']}."
        )
    output_path = crop_dir / f"{job['v3_key']}.jpg"
    metadata = write_marked_candidate_crop(
        source_crop,
        output_path,
        job=job,
        marker_color=marker_color,
        marker_width=marker_width,
    )
    metadata["marked_crop_sha256"] = sha256sum(output_path)
    return metadata


def save_outputs(
    *,
    run_dir: Path,
    jobs: list[dict[str, Any]],
    selected_v2: list[dict[str, Any]],
    reviews_by_key: Mapping[str, Mapping[str, Any]],
    baseline_records: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    selection: Mapping[str, Any],
    error_attempts: int,
    status: str,
) -> dict[str, Any]:
    expected_keys = {str(item["v3_key"]) for item in jobs}
    reviews = [
        dict(item)
        for key, item in reviews_by_key.items()
        if key in expected_keys
    ]
    parsed = Counter()
    exact = 0
    for job in jobs:
        key = str(job["v3_key"])
        if key not in reviews_by_key:
            continue
        result = parse_contrastive_answer(
            reviews_by_key[key].get("answer"), job["allowed_labels"]
        )
        parsed[str(result["selected_label"] or "invalid")] += 1
        exact += int(bool(result["exact_allowed_label"]))
    metrics: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "protocol": VERIFIER_DEV_CONTRASTIVE_REVIEW_PROTOCOL,
        "status": status,
        "coverage": {
            "expected_candidates": len(jobs),
            "completed_candidates": len(reviews),
            "remaining_candidates": max(len(jobs) - len(reviews), 0),
            "completion_rate": (
                round(len(reviews) / len(jobs), 6) if jobs else 1.0
            ),
            "v2_top1_candidate_queries": len(selected_v2),
            "error_attempts": error_attempts,
        },
        "answers": {
            "parsed_labels": dict(sorted(parsed.items())),
            "exact_allowed_label_rate": (
                round(exact / len(reviews), 6) if reviews else 0.0
            ),
            "token_limit_hits": sum(
                bool(item.get("hit_max_new_tokens")) for item in reviews
            ),
        },
        "methodology": {
            "inference_jobs_use_ground_truth": False,
            "candidate_selection_uses_ground_truth": False,
            "options_source": "standalone official COCO-80 ontology",
            "held_out_data_used_for_selection": False,
        },
    }
    if status == "completed" and len(reviews) == len(jobs):
        predictions, evaluation = evaluate_contrastive_cascade(
            baseline_records,
            jobs=jobs,
            contrastive_reviews=reviews,
            require_strict_accuracy_improvement=bool(
                selection["require_strict_accuracy_improvement"]
            ),
            require_non_decreasing_f1=bool(
                selection["require_non_decreasing_f1"]
            ),
            require_positive_net_corrections=bool(
                selection["require_positive_net_corrections"]
            ),
        )
        baseline_latency = sum(
            float(item.get("latency_seconds", 0.0))
            for item in baseline_records
        )
        grounding_latency = sum(
            float(
                (item["grounding"].get("latency_seconds") or {}).get(
                    "total", 0.0
                )
            )
            for item in evidence
        )
        v2_latency = sum(
            float(
                item.get(
                    "end_to_end_latency_seconds",
                    item.get("latency_seconds", 0.0),
                )
            )
            for item in selected_v2
        )
        v3_latency = sum(
            float(
                item.get(
                    "end_to_end_latency_seconds",
                    item.get("latency_seconds", 0.0),
                )
            )
            for item in reviews
        )
        metrics["evaluation"] = evaluation
        metrics["runtime_projection"] = {
            "baseline_latency_seconds": round(baseline_latency, 6),
            "grounding_latency_seconds": round(grounding_latency, 6),
            "v2_top1_review_latency_seconds": round(v2_latency, 6),
            "v3_contrastive_review_latency_seconds": round(v3_latency, 6),
            "incremental_latency_seconds": round(
                grounding_latency + v2_latency + v3_latency, 6
            ),
            "projected_end_to_end_total": round(
                baseline_latency
                + grounding_latency
                + v2_latency
                + v3_latency,
                6,
            ),
        }
        write_jsonl_atomic(run_dir / "predictions.jsonl", predictions)
        decision = {
            "created_at_utc": utc_now(),
            "protocol": VERIFIER_DEV_CONTRASTIVE_REVIEW_PROTOCOL,
            **evaluation["selection"],
            "policy": {
                "cascade": (
                    "Qwen No -> Grounding/SAM -> V2 exact Yes -> "
                    "V3 exact target label"
                ),
                "score_threshold": 0.30,
                "max_mask_area_ratio": 0.90,
                "max_candidates_per_query": 1,
                "contrastive_options": (
                    "target supercategory plus none"
                ),
            },
            "dev_metrics": evaluation["v3"],
            "dev_delta": evaluation["delta"],
            "dev_corrections": evaluation["corrections"],
            "held_out_evaluation_pending": bool(
                evaluation["selection"]["eligible"]
            ),
        }
        write_json_atomic(run_dir / "v3_decision.json", decision)
    write_json_atomic(run_dir / "metrics.json", metrics)
    return metrics


def write_jsonl_atomic(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            handle.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    (
        config_path,
        baseline_path,
        grounding_run_dir,
        semantic_run_dir,
        ontology_path,
        output_dir,
        cascade,
        contrastive,
        model,
        extra,
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
    semantic_reviews, semantic_source = validate_semantic_source(
        semantic_run_dir,
        baseline_source=baseline_source,
        grounding_source=grounding_source,
    )
    jobs, selected_v2 = build_contrastive_review_jobs(
        semantic_reviews,
        groups=extra["ontology"]["groups"],
        category_to_group=extra["ontology"]["category_to_group"],
        evidence_score_threshold=float(
            cascade["evidence_score_threshold"]
        ),
        min_mask_score=cascade.get("min_mask_score"),
        min_mask_area_ratio=float(cascade["min_mask_area_ratio"]),
        max_mask_area_ratio=float(cascade["max_mask_area_ratio"]),
        max_candidates_per_query=int(
            cascade["max_candidates_per_query"]
        ),
        none_label=str(contrastive["none_label"]),
        require_exact_v2_yes=bool(cascade["require_exact_v2_yes"]),
    )
    summary = {
        "baseline_predictions": len(baseline_records),
        "grounding_queries": len(evidence),
        "stage37_reviews": len(semantic_reviews),
        "v2_top1_candidates": len(selected_v2),
        "v3_contrastive_reviews": len(jobs),
        "images": len({str(item["image"]) for item in jobs}),
        "objects": sorted({str(item["object"]) for item in jobs}),
        "ordered_v3_keys_sha256": ordered_v3_keys_sha256(jobs),
        "inference_jobs_use_ground_truth": False,
        "model_inference_required": not (
            args.preflight_only or args.prepare_only
        ),
    }
    for job in jobs:
        source_crop = resolve_source_crop(semantic_run_dir, job)
        if not source_crop.is_file():
            raise FileNotFoundError(f"V3 source crop missing: {source_crop}.")
        if sha256sum(source_crop) != job["source_crop_sha256"]:
            raise RuntimeError(
                f"V3 source crop hash mismatch: {job['candidate_key']}."
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.preflight_only:
        print("Preflight complete: no model or marked crop was created.")
        return

    model_name = Path(model["model_id"]).name
    run_name = args.run_name or (
        f"contrastive-review-dev{len(jobs)}__{model_name}"
    )
    run_dir = output_dir / slugify(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = run_dir / "contrastive_reviews.jsonl"
    errors_path = run_dir / "errors.jsonl"
    run_config_path = run_dir / "run_config.json"
    run_config = {
        "created_at_utc": utc_now(),
        "protocol": VERIFIER_DEV_CONTRASTIVE_REVIEW_PROTOCOL,
        **baseline_source,
        **grounding_source,
        **semantic_source,
        **extra["sources"],
        "expected_reviews": len(jobs),
        "ordered_v3_keys_sha256": ordered_v3_keys_sha256(jobs),
        "contrastive_review_system_prompt": (
            CONTRASTIVE_REVIEW_SYSTEM_PROMPT
        ),
        "contrastive_review_system_prompt_sha256": hashlib.sha256(
            CONTRASTIVE_REVIEW_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "model_id": model["model_id"],
        "torch_dtype": model.get("torch_dtype", "auto"),
        "device_map": model.get("device_map", "auto"),
        "max_new_tokens": model["max_new_tokens"],
        "do_sample": model["do_sample"],
        "local_files_only": model["local_files_only"],
        "evidence_score_threshold": float(
            cascade["evidence_score_threshold"]
        ),
        "min_mask_score": cascade.get("min_mask_score"),
        "min_mask_area_ratio": float(cascade["min_mask_area_ratio"]),
        "max_mask_area_ratio": float(cascade["max_mask_area_ratio"]),
        "max_candidates_per_query": int(
            cascade["max_candidates_per_query"]
        ),
        "require_exact_v2_yes": bool(cascade["require_exact_v2_yes"]),
        "none_label": str(contrastive["none_label"]),
        "marker_color": list(contrastive["marker_color"]),
        "marker_width": int(contrastive["marker_width"]),
        "require_exact_label": bool(contrastive["require_exact_label"]),
        "held_out_data_used_for_selection": False,
    }
    validate_or_create_run_config(run_config_path, run_config)
    write_jsonl_atomic(run_dir / "jobs.jsonl", jobs)

    reviews_by_key = load_reviews(reviews_path)
    expected_keys = {str(item["v3_key"]) for item in jobs}
    completed_keys = expected_keys.intersection(reviews_by_key)
    pending = [
        item
        for item in jobs
        if str(item["v3_key"]) not in completed_keys
    ]
    historical_errors = count_jsonl(errors_path)
    print(f"Run dir:   {run_dir}")
    print(f"Reviews:   {len(jobs)}")
    print(f"Completed: {len(completed_keys)}")
    print(f"Pending:   {len(pending)}")
    if not pending:
        metrics = save_outputs(
            run_dir=run_dir,
            jobs=jobs,
            selected_v2=selected_v2,
            reviews_by_key=reviews_by_key,
            baseline_records=baseline_records,
            evidence=evidence,
            selection=extra["selection"],
            error_attempts=historical_errors,
            status="completed",
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    crop_dir = run_dir / "marked_crops"
    crop_metadata = {
        str(job["v3_key"]): prepare_marked_crop(
            job,
            semantic_run_dir=semantic_run_dir,
            crop_dir=crop_dir,
            marker_color=list(contrastive["marker_color"]),
            marker_width=int(contrastive["marker_width"]),
        )
        for job in pending
    }
    if args.prepare_only:
        metrics = save_outputs(
            run_dir=run_dir,
            jobs=jobs,
            selected_v2=selected_v2,
            reviews_by_key=reviews_by_key,
            baseline_records=baseline_records,
            evidence=evidence,
            selection=extra["selection"],
            error_attempts=historical_errors,
            status="prepared",
        )
        print(f"Jobs:         {run_dir / 'jobs.jsonl'}")
        print(f"Marked crops: {crop_dir}")
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
            "VLM runtime dependencies are missing for V3 review."
        ) from error

    runner = VlmBaseline(
        VlmBaselineConfig(
            model_id=model["model_id"],
            torch_dtype=model.get("torch_dtype", "auto"),
            device_map=model.get("device_map", "auto"),
            max_new_tokens=model["max_new_tokens"],
            do_sample=False,
            local_files_only=model["local_files_only"],
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
            for job in tqdm(pending, desc="Verifier Dev V3", unit="crop"):
                try:
                    crop = crop_metadata[str(job["v3_key"])]
                    result = runner.answer(
                        crop["marked_crop_image"],
                        str(job["contrastive_question"]),
                        system_prompt=CONTRASTIVE_REVIEW_SYSTEM_PROMPT,
                    )
                    parsed = parse_contrastive_answer(
                        result["answer"], job["allowed_labels"]
                    )
                    review = {
                        **dict(job),
                        **crop,
                        "answer": result["answer"],
                        **parsed,
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
                    reviews_by_key[str(job["v3_key"])] = review
                    successes_since_save += 1
                    if successes_since_save >= args.save_every:
                        save_outputs(
                            run_dir=run_dir,
                            jobs=jobs,
                            selected_v2=selected_v2,
                            reviews_by_key=reviews_by_key,
                            baseline_records=baseline_records,
                            evidence=evidence,
                            selection=extra["selection"],
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
                            "v3_key": job["v3_key"],
                            "candidate_key": job["candidate_key"],
                            "image": job["image"],
                            "object": job["object"],
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(limit=5),
                            "attempted_at_utc": utc_now(),
                        },
                    )
                    tqdm.write(
                        f"ERROR {job['v3_key']}: "
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
    metrics = save_outputs(
        run_dir=run_dir,
        jobs=jobs,
        selected_v2=selected_v2,
        reviews_by_key=reviews_by_key,
        baseline_records=baseline_records,
        evidence=evidence,
        selection=extra["selection"],
        error_attempts=historical_errors + invocation_errors,
        status=status,
    )
    print(f"Reviews: {reviews_path}")
    print(f"Errors:  {errors_path}")
    print(f"Metrics: {run_dir / 'metrics.json'}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if fatal_error is not None:
        raise fatal_error
    if status != "completed":
        raise RuntimeError(
            f"V3 contrastive review ended with status={status}; repeat the "
            "identical command to resume."
        )


if __name__ == "__main__":
    main()
