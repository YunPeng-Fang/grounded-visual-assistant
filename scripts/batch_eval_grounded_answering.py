"""Run resumable grounding-verified answering on eval_v0."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.dataset_splits import image_ids_sha256, load_image_ids
from grounded_visual_assistant.evaluation import score_prediction
from grounded_visual_assistant.evidence_answering import (
    EvidencePolicyConfig,
    aggregate_evidence_answering,
    answer_with_evidence,
    build_query_plan,
)
from grounded_visual_assistant.grounding_evaluation import evaluate_grounding_image
from grounded_visual_assistant.policy_calibration import (
    apply_locked_policy,
    validate_locked_policy,
)
from grounded_visual_assistant.structured_prompting import (
    parse_structured_category_answer,
)


TASK_TYPES = ("object_listing", "object_existence", "spatial_relation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate evidence-verified answers using structured Qwen categories, "
            "Grounding DINO, and SAM 2.1."
        )
    )
    parser.add_argument("--dataset", default="data/eval_v0/questions.jsonl")
    parser.add_argument("--config", default="configs/grounded_sam2.yaml")
    parser.add_argument(
        "--structured-predictions",
        required=True,
        help="Structured Qwen object-listing predictions.jsonl.",
    )
    parser.add_argument(
        "--policy-file",
        default=None,
        help="Immutable selected_policy.json from Dev20 calibration.",
    )
    parser.add_argument(
        "--answer-vlm-predictions",
        default=None,
        help="Original three-task Qwen predictions for locked existence fusion.",
    )
    parser.add_argument("--image-ids", default=None)
    parser.add_argument("--grounding-model-id", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-config", default=None)
    parser.add_argument("--box-threshold", type=float, default=None)
    parser.add_argument("--text-threshold", type=float, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    parser.add_argument("--evidence-score-threshold", type=float, default=None)
    parser.add_argument("--evidence-mask-score-threshold", type=float, default=None)
    parser.add_argument("--evidence-min-mask-area-ratio", type=float, default=0.0)
    parser.add_argument("--relation-margin", type=float, default=0.08)
    parser.add_argument("--evidence-iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    parser.add_argument(
        "--output-dir",
        default="outputs/eval_answering_v0",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Select the first N images and all three questions per image.",
    )
    parser.add_argument("--visualize-limit", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all inputs and locked policy without loading models.",
    )
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
    for name in (
        "evidence_score_threshold",
        "evidence_mask_score_threshold",
        "evidence_min_mask_area_ratio",
        "relation_margin",
    ):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1].")
    if not 0.0 < args.evidence_iou_threshold <= 1.0:
        parser.error("--evidence-iou-threshold must be in (0, 1].")
    if bool(args.policy_file) != bool(args.answer_vlm_predictions):
        parser.error(
            "--policy-file and --answer-vlm-predictions must be provided together."
        )
    return args


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
            sample_id = str(record.get("id", ""))
            if not sample_id:
                raise ValueError(f"Missing id on {path}:{line_number}.")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id on {path}: {sample_id}")
            seen_ids.add(sample_id)
            records.append(record)
    if not records:
        raise ValueError(f"JSONL file is empty: {path}")
    return records


def select_samples(
    records: list[dict[str, Any]],
    requested_image_ids: set[int] | None,
    max_images: int | None,
) -> list[dict[str, Any]]:
    required = {"id", "image", "image_id", "question", "task_type", "gt_answer"}
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Dataset sample {record.get('id')} misses {sorted(missing)}")
        if record["task_type"] not in TASK_TYPES:
            raise ValueError(f"Unsupported task type: {record['task_type']}")

    available = {int(record["image_id"]) for record in records}
    if requested_image_ids is not None:
        unknown = sorted(requested_image_ids - available)
        if unknown:
            raise ValueError(f"Split image IDs are absent from dataset: {unknown[:10]}")
        records = [
            record
            for record in records
            if int(record["image_id"]) in requested_image_ids
        ]

    image_order = list(dict.fromkeys(int(record["image_id"]) for record in records))
    if max_images is not None:
        keep = set(image_order[:max_images])
        records = [record for record in records if int(record["image_id"]) in keep]

    counts = Counter(
        (int(record["image_id"]), str(record["task_type"])) for record in records
    )
    invalid = [key for key, count in counts.items() if count != 1]
    if invalid:
        raise ValueError(f"Expected one record per image/task, invalid keys: {invalid[:5]}")
    selected_images = {int(record["image_id"]) for record in records}
    missing_tasks = {
        image_id: sorted(set(TASK_TYPES) - {task for (current, task) in counts if current == image_id})
        for image_id in selected_images
    }
    missing_tasks = {key: value for key, value in missing_tasks.items() if value}
    if missing_tasks:
        raise ValueError(f"Selected images are missing tasks: {missing_tasks}")
    return records


def load_structured_by_image(path: Path) -> dict[int, dict[str, Any]]:
    by_image = {}
    for record in load_jsonl(path):
        if record.get("task_type") != "object_listing":
            continue
        image_id = int(record["image_id"])
        if image_id in by_image:
            raise ValueError(f"Duplicate structured prediction for image {image_id}.")
        structured = record.get("structured_output")
        if not isinstance(structured, dict):
            structured = parse_structured_category_answer(str(record.get("prediction", "")))
        categories = structured.get("parsed_categories")
        if not isinstance(categories, list):
            raise ValueError(f"Structured prediction {record['id']} has no category list.")
        by_image[image_id] = {
            "id": record["id"],
            "categories": sorted({str(category) for category in categories}),
            "structured_output": structured,
            "prediction": record.get("prediction"),
            "model": record.get("model"),
            "latency_seconds": float(record.get("latency_seconds", 0.0)),
            "prompt_version": record.get("prompt_version"),
        }
    if not by_image:
        raise ValueError(f"No object_listing predictions found in {path}.")
    return by_image


def load_answer_vlm_by_id(path: Path) -> dict[str, dict[str, Any]]:
    """Load original task predictions used by the locked consensus policy."""
    return {str(record["id"]): record for record in load_jsonl(path)}


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
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_").lower() or "run"


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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return {str(record["id"]): record for record in load_jsonl(path)}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def validate_or_create_run_config(path: Path, current: dict[str, Any]) -> None:
    immutable_keys = (
        "dataset_sha256",
        "structured_predictions_sha256",
        "policy_file_sha256",
        "answer_vlm_predictions_sha256",
        "answer_policy_protocol",
        "image_ids_sha256",
        "grounding_model_id",
        "sam2_checkpoint",
        "sam2_model_config",
        "box_threshold",
        "text_threshold",
        "nms_iou_threshold",
        "evidence_score_threshold",
        "evidence_mask_score_threshold",
        "evidence_min_mask_area_ratio",
        "relation_margin",
        "evidence_iou_threshold",
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
                "Output directory belongs to an incompatible run. Choose a new "
                f"--run-name. Differences: {differences}"
            )
        return
    write_json_atomic(path, current)


def add_runtime_metadata(path: Path, runner: Any, torch: Any, transformers: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "runtime" in payload:
        return
    cuda_available = torch.cuda.is_available()
    payload["runtime"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_available": cuda_available,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_0": torch.cuda.get_device_name(0) if cuda_available else None,
        "grounding_model_class": type(runner.grounding_model).__name__,
        "sam2_model_class": type(runner.sam2_predictor.model).__name__,
    }
    write_json_atomic(path, payload)


def empty_result(image_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "image_path": str(image_path),
        "text_prompt": "",
        "annotations": [],
        "img_width": width,
        "img_height": height,
        "models": {
            "grounding": config["grounding_model_id"],
            "sam2_checkpoint": config["sam2_checkpoint"],
            "sam2_config": config["sam2_model_config"],
        },
        "thresholds": {
            "box": config["box_threshold"],
            "text": config["text_threshold"],
            "nms_iou": config["nms_iou_threshold"],
        },
        "postprocessing": {"candidate_count": 0, "kept_count": 0, "suppressed_count": 0},
        "latency_seconds": {"grounding": 0.0, "sam2": 0.0, "total": 0.0},
        "device": config["device"],
        "skipped_reason": "empty_structured_prompt",
    }


def save_metrics(
    path: Path,
    predictions_by_id: dict[str, dict[str, Any]],
    selected_ids: set[str],
    expected_samples: int,
    error_attempts: int,
    status: str,
) -> dict[str, Any]:
    records = [
        prediction
        for sample_id, prediction in predictions_by_id.items()
        if sample_id in selected_ids
    ]
    metrics = aggregate_evidence_answering(
        records,
        expected_samples=expected_samples,
        error_attempts=error_attempts,
        status=status,
    )
    write_json_atomic(path, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    dataset_path = project_path(args.dataset)
    structured_path = project_path(args.structured_predictions)
    policy_path = project_path(args.policy_file) if args.policy_file else None
    answer_vlm_path = (
        project_path(args.answer_vlm_predictions)
        if args.answer_vlm_predictions
        else None
    )
    locked_policy = None
    answer_vlm_by_id: dict[str, dict[str, Any]] = {}
    if policy_path is not None:
        locked_policy = validate_locked_policy(
            json.loads(policy_path.read_text(encoding="utf-8"))
        )
        assert answer_vlm_path is not None
        answer_vlm_by_id = load_answer_vlm_by_id(answer_vlm_path)
    image_ids_path = project_path(args.image_ids) if args.image_ids else None
    if (
        locked_policy is not None
        and image_ids_path is not None
        and "test" in image_ids_path.stem.lower()
        and args.max_images is not None
    ):
        raise ValueError(
            "Locked Test evaluation must run the complete split; --max-images is "
            "not allowed. Resume the same full run after an interruption."
        )
    requested_image_ids = (
        set(load_image_ids(image_ids_path)) if image_ids_path is not None else None
    )
    samples = select_samples(
        load_jsonl(dataset_path), requested_image_ids, args.max_images
    )
    structured_by_image = load_structured_by_image(structured_path)
    selected_image_ids = {int(sample["image_id"]) for sample in samples}
    missing_structured = sorted(selected_image_ids - set(structured_by_image))
    if missing_structured:
        raise ValueError(
            "Structured predictions are missing selected images: "
            f"{missing_structured[:10]}"
        )
    if locked_policy is not None:
        existence_ids = {
            str(sample["id"])
            for sample in samples
            if sample["task_type"] == "object_existence"
        }
        missing_answer_vlm = sorted(existence_ids - set(answer_vlm_by_id))
        if missing_answer_vlm:
            raise ValueError(
                "Answer VLM predictions are missing selected existence IDs: "
                f"{missing_answer_vlm[:10]}"
            )

    cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
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
    evidence_score_threshold = (
        args.evidence_score_threshold
        if args.evidence_score_threshold is not None
        else box_threshold
    )
    if evidence_score_threshold < box_threshold:
        raise ValueError(
            "--evidence-score-threshold cannot be lower than --box-threshold "
            "because those detector candidates were already discarded."
        )
    device = args.device or runtime_cfg.get("device", "cuda")
    dtype = args.dtype or runtime_cfg.get("dtype", "float16")
    local_files_only = bool(
        args.local_files_only or grounding_cfg.get("local_files_only", False)
    )
    policy = EvidencePolicyConfig(
        min_grounding_score=evidence_score_threshold,
        min_mask_score=args.evidence_mask_score_threshold,
        min_mask_area_ratio=args.evidence_min_mask_area_ratio,
        relation_margin=args.relation_margin,
    )
    if locked_policy is not None:
        for task_type, entry in locked_policy["tasks"].items():
            if entry["mode"] == "structured_vlm_only":
                continue
            task_score = float(entry["config"]["min_grounding_score"])
            if task_score < box_threshold:
                raise ValueError(
                    f"Locked {task_type} score {task_score} is below detector box "
                    f"threshold {box_threshold}; raw candidates would be missing."
                )

    split_name = image_ids_path.stem if image_ids_path is not None else "all"
    run_name = args.run_name or (
        f"{slugify(split_name)}__evidence-answering__structured-coco80-v1"
        f"__box-{box_threshold:.2f}__text-{text_threshold:.2f}__nms-"
        + (f"{nms_iou_threshold:.2f}" if nms_iou_threshold is not None else "none")
        + ("__locked-task-aware-v1" if locked_policy is not None else "")
    )
    run_dir = project_path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_config_path = run_dir / "run_config.json"
    visualizations_dir = run_dir / "visualizations"
    run_config = {
        "created_at_utc": utc_now(),
        "protocol": (
            "locked_task_aware_evidence_answering_v1"
            if locked_policy is not None
            else "grounding_verified_answering_v1"
        ),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256sum(dataset_path),
        "structured_predictions": str(structured_path),
        "structured_predictions_sha256": sha256sum(structured_path),
        "policy_file": str(policy_path) if policy_path is not None else None,
        "policy_file_sha256": (
            sha256sum(policy_path) if policy_path is not None else None
        ),
        "answer_policy_protocol": (
            locked_policy["protocol"] if locked_policy is not None else None
        ),
        "answer_vlm_predictions": (
            str(answer_vlm_path) if answer_vlm_path is not None else None
        ),
        "answer_vlm_predictions_sha256": (
            sha256sum(answer_vlm_path) if answer_vlm_path is not None else None
        ),
        "image_ids": str(image_ids_path) if image_ids_path is not None else None,
        "image_ids_sha256": (
            image_ids_sha256(requested_image_ids)
            if requested_image_ids is not None
            else None
        ),
        "grounding_model_id": grounding_model_id,
        "sam2_checkpoint": sam2_checkpoint,
        "sam2_model_config": sam2_model_config,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "evidence_score_threshold": evidence_score_threshold,
        "evidence_mask_score_threshold": args.evidence_mask_score_threshold,
        "evidence_min_mask_area_ratio": args.evidence_min_mask_area_ratio,
        "relation_margin": args.relation_margin,
        "evidence_iou_threshold": args.evidence_iou_threshold,
        "device": device,
        "dtype": dtype,
        "local_files_only": local_files_only,
    }
    validate_or_create_run_config(run_config_path, run_config)

    predictions_by_id = load_existing_predictions(predictions_path)
    selected_ids = {str(sample["id"]) for sample in samples}
    completed_ids = set(predictions_by_id) & selected_ids
    pending = [sample for sample in samples if str(sample["id"]) not in completed_ids]
    positions = {str(sample["id"]): index for index, sample in enumerate(samples)}
    historical_errors = count_jsonl(errors_path)

    print(f"Dataset:    {dataset_path}")
    print(f"Structured: {structured_path}")
    print(f"Policy:     {policy_path or 'initial shared policy'}")
    if answer_vlm_path is not None:
        print(f"Answer VLM: {answer_vlm_path}")
    print(f"Split:      {image_ids_path or 'all images'}")
    print(f"Run dir:    {run_dir}")
    print(f"Selected:   {len(samples)} questions / {len(selected_image_ids)} images")
    print(f"Completed:  {len(completed_ids)}")
    print(f"Pending:    {len(pending)}")

    if args.preflight_only:
        print("Preflight:  passed; no model was loaded and no prediction was made.")
        return

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

        from grounded_visual_assistant.grounded_sam2 import (
            GroundedSam2,
            GroundedSam2Config,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Grounded-SAM-2 dependencies are missing. Run "
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
    add_runtime_metadata(run_config_path, runner, torch, transformers)

    invocation_errors = 0
    successes_since_save = 0
    status = "completed"
    fatal_error: Exception | None = None
    with predictions_path.open("a", encoding="utf-8", buffering=1) as prediction_file, \
        errors_path.open("a", encoding="utf-8", buffering=1) as error_file:
        try:
            for sample in tqdm(pending, desc="Evidence answering", unit="question"):
                sample_id = str(sample["id"])
                structured = structured_by_image[int(sample["image_id"])]
                planning_start = time.perf_counter()
                try:
                    query_plan = build_query_plan(
                        sample,
                        structured["categories"]
                        if sample["task_type"] == "object_listing"
                        else None,
                    )
                    parser_latency = time.perf_counter() - planning_start
                    answer_vlm_record = answer_vlm_by_id.get(sample_id)
                    if sample["task_type"] == "object_listing":
                        cached_vlm_latency = float(structured["latency_seconds"])
                    elif (
                        sample["task_type"] == "object_existence"
                        and answer_vlm_record is not None
                    ):
                        cached_vlm_latency = float(
                            answer_vlm_record.get("latency_seconds", 0.0)
                        )
                    else:
                        cached_vlm_latency = 0.0
                    planning_latency = parser_latency + cached_vlm_latency
                    image_path = resolve_image_path(str(sample["image"]), dataset_path)
                    if not image_path.is_file():
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    artifact_dir = (
                        visualizations_dir / sample_id
                        if positions[sample_id] < args.visualize_limit
                        else None
                    )
                    if query_plan["prompt"]:
                        result = runner.predict(
                            image_path,
                            query_plan["prompt"],
                            output_dir=artifact_dir,
                        )
                    else:
                        result = empty_result(image_path, run_config)
                    policy_source = {
                        **sample,
                        "target_categories": sample.get("categories", []),
                        "query_plan": query_plan,
                        "annotations": result["annotations"],
                    }
                    if locked_policy is not None:
                        applied_policy = apply_locked_policy(
                            policy_source,
                            locked_policy,
                            image_width=int(result["img_width"]),
                            image_height=int(result["img_height"]),
                            vlm_record=answer_vlm_record,
                        )
                        answer_policy = applied_policy["answer_policy"]
                        evaluation = applied_policy["evaluation"]
                        policy_metadata = applied_policy["policy_config"]
                    else:
                        answer_policy = answer_with_evidence(
                            sample,
                            query_plan,
                            result["annotations"],
                            image_width=int(result["img_width"]),
                            image_height=int(result["img_height"]),
                            config=policy,
                        )
                        forced_answer = str(answer_policy["forced_answer"])
                        evaluation = score_prediction(sample, forced_answer)
                        policy_metadata = {
                            "protocol": "shared_evidence_policy_v1",
                            "config": {
                                "min_grounding_score": evidence_score_threshold,
                                "min_mask_score": args.evidence_mask_score_threshold,
                                "min_mask_area_ratio": (
                                    args.evidence_min_mask_area_ratio
                                ),
                                "relation_margin": args.relation_margin,
                            },
                        }
                    policy_annotations = [
                        {
                            "class_name": item["category"],
                            "bbox": item["bbox"],
                            "score": item["score"],
                        }
                        for item in answer_policy["accepted_evidence"]
                    ]
                    evidence_evaluation = evaluate_grounding_image(
                        list(sample.get("evidence_boxes", [])),
                        policy_annotations,
                        iou_threshold=args.evidence_iou_threshold,
                    )
                    pipeline_latency = {
                        "planning": round(planning_latency, 6),
                        "cached_vlm": round(cached_vlm_latency, 6),
                        "query_parser": round(parser_latency, 6),
                        "grounding": float(result["latency_seconds"]["grounding"]),
                        "sam2": float(result["latency_seconds"]["sam2"]),
                        "total": round(
                            planning_latency + float(result["latency_seconds"]["total"]),
                            6,
                        ),
                    }
                    prediction: dict[str, Any] = {
                        "id": sample_id,
                        "image": sample["image"],
                        "image_id": sample["image_id"],
                        "source": sample.get("source"),
                        "question": sample["question"],
                        "task_type": sample["task_type"],
                        "gt_answer": sample["gt_answer"],
                        "target_categories": sample.get("categories", []),
                        "target_evidence_boxes": sample.get("evidence_boxes", []),
                        "query_plan": query_plan,
                        "structured_vlm": structured
                        if sample["task_type"] == "object_listing"
                        else None,
                        "answer_vlm": (
                            {
                                "id": answer_vlm_record.get("id"),
                                "prediction": answer_vlm_record.get("prediction"),
                                "model": answer_vlm_record.get("model"),
                                "latency_seconds": answer_vlm_record.get(
                                    "latency_seconds", 0.0
                                ),
                            }
                            if sample["task_type"] == "object_existence"
                            and answer_vlm_record is not None
                            else None
                        ),
                        "annotations": result["annotations"],
                        "answer_policy": answer_policy,
                        "applied_policy": policy_metadata,
                        "evaluation": evaluation,
                        "evidence_evaluation": evidence_evaluation,
                        "models": result["models"],
                        "thresholds": {
                            **result["thresholds"],
                            "evidence_score": policy_metadata["config"].get(
                                "min_grounding_score", evidence_score_threshold
                            ),
                            "evidence_mask_score": policy_metadata["config"].get(
                                "min_mask_score",
                                args.evidence_mask_score_threshold,
                            ),
                            "evidence_min_mask_area_ratio": policy_metadata[
                                "config"
                            ].get(
                                "min_mask_area_ratio",
                                args.evidence_min_mask_area_ratio,
                            ),
                            "relation_margin": policy_metadata["config"].get(
                                "relation_margin", args.relation_margin
                            ),
                            "locked_task_policy": (
                                policy_metadata if locked_policy is not None else None
                            ),
                        },
                        "postprocessing": result["postprocessing"],
                        "latency_seconds": result["latency_seconds"],
                        "pipeline_latency_seconds": pipeline_latency,
                        "device": result["device"],
                        "evaluated_at_utc": utc_now(),
                    }
                    if "skipped_reason" in result:
                        prediction["skipped_reason"] = result["skipped_reason"]
                    if "cuda_peak_memory_allocated_gb" in result:
                        prediction["cuda_peak_memory_allocated_gb"] = result[
                            "cuda_peak_memory_allocated_gb"
                        ]
                    append_jsonl(prediction_file, prediction)
                    predictions_by_id[sample_id] = prediction
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
                    append_jsonl(
                        error_file,
                        {
                            "id": sample_id,
                            "image": sample.get("image"),
                            "image_id": sample.get("image_id"),
                            "task_type": sample.get("task_type"),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=5),
                            "attempted_at_utc": utc_now(),
                        },
                    )
                    tqdm.write(f"ERROR {sample_id}: {type(exc).__name__}: {exc}")
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

    if status == "completed" and len(set(predictions_by_id) & selected_ids) < len(samples):
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
        raise RuntimeError(f"Evaluation stopped after: {fatal_error}") from fatal_error


if __name__ == "__main__":
    main()
