"""Batch-evaluate the live answer-to-grounding demo pipeline on eval_v0."""

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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.demo import DemoRuntime, EVIDENCE_MODE
from grounded_visual_assistant.artifact_paths import portable_gallery
from grounded_visual_assistant.live_pipeline_evaluation import (
    aggregate_live_pipeline,
    evaluate_live_prediction,
    evaluate_mask_evidence,
)
from grounded_visual_assistant.live_pipeline_prompting import (
    GENERIC_PROMPT_POLICY,
    PROMPT_POLICIES,
    TASK_AWARE_COCO_POLICY,
    TASK_AWARE_COCO_V2_POLICY,
    build_live_pipeline_system_prompt,
    evidence_target_limit,
)
from grounded_visual_assistant.live_prompt_policy_lock import (
    LOCK_PROTOCOL,
    TEST_PROTOCOL,
)


TASK_TYPES = ("object_listing", "object_existence", "spatial_relation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate per-question Qwen answer/evidence targets followed by "
            "Grounding DINO and SAM 2.1."
        )
    )
    parser.add_argument("--dataset", default="data/eval_v0/questions.jsonl")
    parser.add_argument(
        "--split-image-ids",
        default="data/eval_v0/splits/dev_image_ids.json",
    )
    parser.add_argument(
        "--coco-ground-truth",
        default="data/eval_v0/coco_grounding_gt.json",
    )
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument(
        "--prompt-policy",
        choices=PROMPT_POLICIES,
        default=GENERIC_PROMPT_POLICY,
    )
    parser.add_argument(
        "--policy-manifest",
        default=None,
        help="Pre-registered acceptance criteria for the task-aware candidate.",
    )
    parser.add_argument(
        "--test-protocol",
        default=(
            "outputs/eval_live_pipeline_v0/locked_policy_v1/"
            "test_protocol.json"
        ),
        help="Immutable protocol required for a complete held-out Test run.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval_live_pipeline_v0",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Smoke-test the first N Dev images and all three tasks per image.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly unlock a complete Test split run.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate data and protocol without loading either model.",
    )
    args = parser.parse_args()
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be greater than zero.")
    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be in (0, 1].")
    if args.save_every <= 0:
        parser.error("--save-every must be greater than zero.")
    if args.max_errors < 0:
        parser.error("--max-errors must be zero or greater.")
    return args


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_ids_sha256(values: list[str | int]) -> str:
    payload = json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def slugify(value: str) -> str:
    return (
        re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-_").lower()
        or "run"
    )


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
                raise ValueError(
                    f"Invalid JSON on {path}:{line_number}: {exc}"
                ) from exc
            sample_id = str(record.get("id", ""))
            if not sample_id:
                raise ValueError(f"Missing id on {path}:{line_number}.")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id: {sample_id}")
            seen_ids.add(sample_id)
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def load_split(path: Path) -> tuple[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        split_name = str(payload.get("name") or path.stem)
        values = payload.get("image_ids")
    else:
        split_name = path.stem.replace("_image_ids", "")
        values = payload
    if not isinstance(values, list) or not values:
        raise ValueError(f"Split has no image_ids list: {path}")
    image_ids = [int(value) for value in values]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"Split contains duplicate image IDs: {path}")
    return split_name.lower(), image_ids


def resolve_image_path(image: str, dataset_path: Path) -> Path:
    path = Path(image)
    if path.is_absolute():
        return path
    candidates = (PROJECT_ROOT / path, dataset_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def select_samples(
    records: list[dict[str, Any]],
    split_name: str,
    split_image_ids: list[int],
    max_images: int | None,
) -> list[dict[str, Any]]:
    required = {
        "id",
        "image",
        "image_id",
        "question",
        "task_type",
        "gt_answer",
        "evidence_boxes",
    }
    by_image_task: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(
                f"Sample {record.get('id')} misses fields: {sorted(missing)}"
            )
        task_type = str(record["task_type"])
        if task_type not in TASK_TYPES:
            raise ValueError(f"Unsupported task type: {task_type}")
        key = (int(record["image_id"]), task_type)
        if key in by_image_task:
            raise ValueError(f"Duplicate image/task sample: {key}")
        by_image_task[key] = record

    selected_image_ids = (
        split_image_ids[:max_images]
        if max_images is not None
        else split_image_ids
    )
    selected = []
    for image_id in selected_image_ids:
        for task_type in TASK_TYPES:
            key = (image_id, task_type)
            if key not in by_image_task:
                raise ValueError(f"Split image is missing task sample: {key}")
            selected.append(
                {
                    **by_image_task[key],
                    "split": split_name,
                }
            )
    return selected


def load_coco_ground_truth(
    path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, tuple[int, int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = {
        int(item["id"]): item for item in payload.get("annotations", [])
    }
    image_sizes = {
        int(item["id"]): (int(item["height"]), int(item["width"]))
        for item in payload.get("images", [])
    }
    if not annotations or not image_sizes:
        raise ValueError(f"Invalid COCO ground truth: {path}")
    return annotations, image_sizes


def preflight(
    samples: list[dict[str, Any]],
    dataset_path: Path,
    coco_annotations: dict[int, dict[str, Any]],
    image_sizes: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    missing_images = []
    missing_annotations = []
    for sample in samples:
        image_path = resolve_image_path(str(sample["image"]), dataset_path)
        if not image_path.is_file():
            missing_images.append(str(image_path))
        image_id = int(sample["image_id"])
        if image_id not in image_sizes:
            raise ValueError(f"COCO ground truth misses image {image_id}.")
        for item in sample.get("evidence_boxes", []):
            annotation_id = int(item["annotation_id"])
            if annotation_id not in coco_annotations:
                missing_annotations.append(annotation_id)
    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} images; first: {missing_images[0]}"
        )
    if missing_annotations:
        raise ValueError(
            "COCO ground truth misses evidence annotations; first: "
            f"{missing_annotations[0]}"
        )
    return {
        "samples": len(samples),
        "images": len({int(item["image_id"]) for item in samples}),
        "tasks": dict(
            sorted(Counter(str(item["task_type"]) for item in samples).items())
        ),
        "splits": dict(
            sorted(Counter(str(item["split"]) for item in samples).items())
        ),
        "required_evidence_questions": sum(
            bool(item.get("evidence_boxes")) for item in samples
        ),
        "negative_evidence_questions": sum(
            not bool(item.get("evidence_boxes")) for item in samples
        ),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    return {
        str(record["id"]): record
        for record in load_jsonl(path)
    }


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def validate_or_create_run_config(
    path: Path,
    current: dict[str, Any],
) -> None:
    immutable_keys = (
        "protocol",
        "dataset_sha256",
        "split_image_ids_sha256",
        "selected_sample_ids_sha256",
        "coco_ground_truth_sha256",
        "demo_config_sha256",
        "vlm_config_sha256",
        "grounding_config_sha256",
        "split",
        "model_id",
        "grounding_model_id",
        "sam2_checkpoint",
        "box_threshold",
        "text_threshold",
        "nms_iou_threshold",
        "iou_threshold",
        "prompt_policy",
        "prompt_policy_manifest_sha256",
        "prompt_template_sha256",
        "test_protocol_sha256",
        "locked_policy_sha256",
    )
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        differences = {}
        for key in immutable_keys:
            existing_value = existing.get(key)
            current_value = current.get(key)
            if key == "prompt_policy" and existing_value is None:
                existing_value = GENERIC_PROMPT_POLICY
            if (
                current["prompt_policy"] == GENERIC_PROMPT_POLICY
                and key in {
                    "prompt_policy_manifest_sha256",
                    "prompt_template_sha256",
                }
                and existing_value is None
            ):
                continue
            if existing_value != current_value:
                differences[key] = {
                    "existing": existing_value,
                    "current": current_value,
                }
        if differences:
            raise RuntimeError(
                "Output directory belongs to an incompatible run. Choose a "
                f"new --run-name. Differences: {differences}"
            )
        return
    write_json_atomic(path, current)


def load_protocol_config(config_path: Path) -> dict[str, Any]:
    demo_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    vlm_config_path = project_path(demo_config["vlm"]["config"])
    grounding_config_path = project_path(demo_config["grounding"]["config"])
    vlm_source = yaml.safe_load(vlm_config_path.read_text(encoding="utf-8"))
    grounding_source = yaml.safe_load(
        grounding_config_path.read_text(encoding="utf-8")
    )
    return {
        "demo": demo_config,
        "vlm_config_path": vlm_config_path,
        "grounding_config_path": grounding_config_path,
        "model_id": vlm_source["model"]["model_id"],
        "grounding_model_id": grounding_source["grounding"]["model_id"],
        "sam2_checkpoint": grounding_source["sam2"]["checkpoint"],
        "box_threshold": float(
            demo_config["grounding"].get(
                "box_threshold",
                grounding_source["grounding"].get("box_threshold", 0.4),
            )
        ),
        "text_threshold": float(
            demo_config["grounding"].get(
                "text_threshold",
                grounding_source["grounding"].get("text_threshold", 0.3),
            )
        ),
        "nms_iou_threshold": demo_config["grounding"].get(
            "nms_iou_threshold",
            grounding_source["grounding"].get("nms_iou_threshold"),
        ),
    }


def load_policy_manifest(
    path: Path,
    prompt_policy: str,
    split_name: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if prompt_policy == GENERIC_PROMPT_POLICY:
        return None, None
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not str(manifest.get("protocol", "")).startswith(
        "live_prompt_policy_selection_v"
    ):
        raise ValueError(f"Unsupported prompt policy manifest: {path}")
    if str(manifest.get("split", "")).lower() != split_name:
        raise ValueError(
            f"Prompt policy manifest is for {manifest.get('split')}, "
            f"not {split_name}."
        )
    candidate_policy = str(manifest.get("candidate", {}).get("prompt_policy"))
    if candidate_policy != prompt_policy:
        raise ValueError(
            f"Manifest candidate is {candidate_policy}, not {prompt_policy}."
        )
    if not isinstance(manifest.get("acceptance"), dict):
        raise ValueError(f"Manifest has no acceptance criteria: {path}")
    expected_template_hash = manifest.get("candidate", {}).get(
        "prompt_template_sha256"
    )
    if expected_template_hash is not None:
        prompt_template_path = (
            SRC_ROOT
            / "grounded_visual_assistant"
            / "live_pipeline_prompting.py"
        )
        actual_template_hash = sha256sum(prompt_template_path)
        if str(expected_template_hash) != actual_template_hash:
            raise ValueError(
                "Prompt template does not match the pre-registered candidate: "
                f"{actual_template_hash}"
            )
    return manifest, sha256sum(path)


def default_policy_manifest(prompt_policy: str) -> str:
    if prompt_policy == TASK_AWARE_COCO_V2_POLICY:
        return "configs/live_prompt_policy_v2.yaml"
    return "configs/live_prompt_policy_v1.yaml"


def load_locked_test_protocol(
    path: Path,
    *,
    prompt_policy: str,
    dataset_path: Path,
    split_path: Path,
    coco_gt_path: Path,
    selected_sample_ids: list[str],
    image_count: int,
) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol") != TEST_PROTOCOL
        or payload.get("status") != "locked"
        or payload.get("split") != "test"
        or payload.get("allow_partial") is not False
    ):
        raise RuntimeError(f"Invalid locked Test protocol: {path}")
    if payload.get("selected_prompt_policy") != prompt_policy:
        raise RuntimeError(
            "Test prompt policy differs from the locked selection."
        )
    prompt_template_path = (
        SRC_ROOT
        / "grounded_visual_assistant"
        / "live_pipeline_prompting.py"
    )
    if payload.get("prompt_template_sha256") != sha256sum(
        prompt_template_path
    ):
        raise RuntimeError("Locked Test prompt template hash changed.")
    runtime_hashes = payload.get("runtime_files_sha256")
    if not isinstance(runtime_hashes, dict) or not runtime_hashes:
        raise RuntimeError("Locked Test protocol has no runtime source hashes.")
    runtime_differences = {
        relative_path: {
            "locked": expected_hash,
            "current": (
                sha256sum(project_path(relative_path))
                if project_path(relative_path).is_file()
                else None
            ),
        }
        for relative_path, expected_hash in runtime_hashes.items()
        if (
            not project_path(relative_path).is_file()
            or sha256sum(project_path(relative_path)) != expected_hash
        )
    }
    if runtime_differences:
        raise RuntimeError(
            f"Held-out Test runtime differs from the lock: "
            f"{runtime_differences}"
        )
    locked_policy_path = project_path(payload["locked_policy"])
    if (
        not locked_policy_path.is_file()
        or payload.get("locked_policy_sha256")
        != sha256sum(locked_policy_path)
    ):
        raise RuntimeError("Locked policy file is missing or changed.")
    locked_policy = json.loads(
        locked_policy_path.read_text(encoding="utf-8")
    )
    if (
        locked_policy.get("protocol") != LOCK_PROTOCOL
        or locked_policy.get("status") != "locked"
        or locked_policy.get("selected_prompt_policy") != prompt_policy
        or locked_policy.get("prompt_template_sha256")
        != payload.get("prompt_template_sha256")
    ):
        raise RuntimeError("Locked policy contents are incompatible.")
    expected_hashes = {
        "dataset_sha256": sha256sum(dataset_path),
        "split_image_ids_sha256": sha256sum(split_path),
        "selected_sample_ids_sha256": stable_ids_sha256(
            selected_sample_ids
        ),
        "coco_ground_truth_sha256": sha256sum(coco_gt_path),
    }
    differences = {
        key: {
            "locked": payload.get(key),
            "current": current,
        }
        for key, current in expected_hashes.items()
        if payload.get(key) != current
    }
    if differences:
        raise RuntimeError(
            f"Held-out Test inputs differ from the lock: {differences}"
        )
    if (
        int(payload.get("expected_images", -1)) != image_count
        or int(payload.get("expected_samples", -1))
        != len(selected_sample_ids)
    ):
        raise RuntimeError("Held-out Test coverage differs from the lock.")
    if not str(payload.get("run_name", "")).strip():
        raise RuntimeError("Locked Test protocol has no run name.")
    return payload, sha256sum(path)


def save_metrics(
    path: Path,
    predictions_by_id: dict[str, dict[str, Any]],
    selected_ids: set[str],
    expected_samples: int,
    expected_required_evidence: int,
    expected_negative_evidence: int,
    error_attempts: int,
    status: str,
    iou_threshold: float,
) -> dict[str, Any]:
    records = [
        prediction
        for sample_id, prediction in predictions_by_id.items()
        if sample_id in selected_ids
    ]
    metrics = aggregate_live_pipeline(
        records,
        expected_samples=expected_samples,
        expected_required_evidence=expected_required_evidence,
        expected_negative_evidence=expected_negative_evidence,
        error_attempts=error_attempts,
        status=status,
        iou_threshold=iou_threshold,
    )
    write_json_atomic(path, metrics)
    return metrics


def compact_grounding_annotations(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "class_name": item.get("class_name"),
            "bbox": item.get("bbox"),
            "score": item.get("score"),
            "mask_score": item.get("mask_score"),
            "mask_area": item.get("mask_area"),
            "segmentation": item.get("segmentation"),
        }
        for item in annotations
    ]


def main() -> None:
    args = parse_args()
    dataset_path = project_path(args.dataset)
    split_path = project_path(args.split_image_ids)
    coco_gt_path = project_path(args.coco_ground_truth)
    config_path = project_path(args.config)
    policy_manifest_path = project_path(
        args.policy_manifest or default_policy_manifest(args.prompt_policy)
    )
    test_protocol_path = project_path(args.test_protocol)
    split_name, split_image_ids = load_split(split_path)
    if split_name == "test" and not args.allow_test:
        raise RuntimeError(
            "Test is locked by default. Run the complete Dev protocol first; "
            "then pass --allow-test for one complete Test run."
        )
    if split_name == "test" and args.max_images is not None:
        raise RuntimeError(
            "Partial Test evaluation is prohibited; remove --max-images."
        )

    samples = select_samples(
        load_jsonl(dataset_path),
        split_name,
        split_image_ids,
        args.max_images,
    )
    coco_annotations, image_sizes = load_coco_ground_truth(coco_gt_path)
    preflight_summary = preflight(
        samples,
        dataset_path,
        coco_annotations,
        image_sizes,
    )
    protocol = load_protocol_config(config_path)
    selected_sample_ids = [str(item["id"]) for item in samples]
    locked_test_protocol = None
    test_protocol_sha256 = None
    if split_name == "test":
        policy_manifest = None
        policy_manifest_sha256 = None
        (
            locked_test_protocol,
            test_protocol_sha256,
        ) = load_locked_test_protocol(
            test_protocol_path,
            prompt_policy=args.prompt_policy,
            dataset_path=dataset_path,
            split_path=split_path,
            coco_gt_path=coco_gt_path,
            selected_sample_ids=selected_sample_ids,
            image_count=preflight_summary["images"],
        )
    else:
        policy_manifest, policy_manifest_sha256 = load_policy_manifest(
            policy_manifest_path,
            args.prompt_policy,
            split_name,
        )
    for sample in samples:
        build_live_pipeline_system_prompt(sample, args.prompt_policy)
    model_name = Path(str(protocol["model_id"])).name
    automatic_run_name = (
        f"{slugify(split_name)}__live-answer-grounding-v1"
        f"__{slugify(model_name)}"
        f"__box-{protocol['box_threshold']:.2f}"
        f"__text-{protocol['text_threshold']:.2f}"
        + (
            f"__{slugify(args.prompt_policy)}"
            if args.prompt_policy != GENERIC_PROMPT_POLICY
            else ""
        )
        + (
            f"__smoke-{args.max_images}"
            if args.max_images is not None
            else ""
        )
    )
    if locked_test_protocol is not None:
        locked_run_name = str(locked_test_protocol["run_name"])
        if args.run_name is not None and args.run_name != locked_run_name:
            raise RuntimeError(
                f"Test run name is locked to {locked_run_name}."
            )
        run_name = locked_run_name
    else:
        run_name = args.run_name or automatic_run_name
    run_dir = project_path(args.output_dir) / run_name
    print(f"Dataset:   {dataset_path}")
    print(f"Split:     {split_path} ({split_name})")
    print(f"Policy:    {args.prompt_policy}")
    if locked_test_protocol is not None:
        print(f"Test lock: {test_protocol_path}")
    print(f"Run dir:   {run_dir}")
    print(json.dumps(preflight_summary, ensure_ascii=False, indent=2))
    if args.preflight_only:
        print("Preflight completed without loading models.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    metrics_path = run_dir / "metrics.json"
    run_config_path = run_dir / "run_config.json"
    run_config = {
        "created_at_utc": utc_now(),
        "protocol": "live_answer_and_evidence_targets_v1",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256sum(dataset_path),
        "split_image_ids": str(split_path),
        "split_image_ids_sha256": sha256sum(split_path),
        "selected_sample_ids_sha256": stable_ids_sha256(
            selected_sample_ids
        ),
        "coco_ground_truth": str(coco_gt_path),
        "coco_ground_truth_sha256": sha256sum(coco_gt_path),
        "demo_config": str(config_path),
        "demo_config_sha256": sha256sum(config_path),
        "vlm_config_sha256": sha256sum(protocol["vlm_config_path"]),
        "grounding_config_sha256": sha256sum(
            protocol["grounding_config_path"]
        ),
        "split": split_name,
        "model_id": protocol["model_id"],
        "grounding_model_id": protocol["grounding_model_id"],
        "sam2_checkpoint": protocol["sam2_checkpoint"],
        "box_threshold": protocol["box_threshold"],
        "text_threshold": protocol["text_threshold"],
        "nms_iou_threshold": protocol["nms_iou_threshold"],
        "iou_threshold": args.iou_threshold,
        "prompt_policy": args.prompt_policy,
        "prompt_policy_manifest": (
            str(policy_manifest_path) if policy_manifest is not None else None
        ),
        "prompt_policy_manifest_sha256": policy_manifest_sha256,
        "prompt_template_sha256": sha256sum(
            SRC_ROOT
            / "grounded_visual_assistant"
            / "live_pipeline_prompting.py"
        ),
        "test_protocol": (
            str(test_protocol_path)
            if locked_test_protocol is not None
            else None
        ),
        "test_protocol_sha256": test_protocol_sha256,
        "locked_policy_sha256": (
            locked_test_protocol.get("locked_policy_sha256")
            if locked_test_protocol is not None
            else None
        ),
        "sample_count": len(samples),
        "image_count": preflight_summary["images"],
    }
    validate_or_create_run_config(run_config_path, run_config)

    predictions_by_id = load_existing_predictions(predictions_path)
    selected_ids = set(selected_sample_ids)
    completed_ids = set(predictions_by_id) & selected_ids
    pending = [item for item in samples if str(item["id"]) not in completed_ids]
    historical_errors = count_jsonl(errors_path)
    print(f"Selected:  {len(samples)} questions")
    print(f"Completed: {len(completed_ids)}")
    print(f"Pending:   {len(pending)}")
    if not pending:
        metrics = save_metrics(
            metrics_path,
            predictions_by_id,
            selected_ids,
            len(samples),
            preflight_summary["required_evidence_questions"],
            preflight_summary["negative_evidence_questions"],
            historical_errors,
            "completed",
            args.iou_threshold,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    runtime = DemoRuntime(PROJECT_ROOT, config_path)
    runtime.config["runtime"]["output_dir"] = str(run_dir / "artifacts")
    invocation_errors = 0
    successes_since_save = 0
    fatal_error: Exception | None = None
    with predictions_path.open(
        "a", encoding="utf-8", buffering=1
    ) as predictions_file, errors_path.open(
        "a", encoding="utf-8", buffering=1
    ) as errors_file:
        for sample in tqdm(pending, desc="Live pipeline", unit="question"):
            sample_id = str(sample["id"])
            image_path = resolve_image_path(str(sample["image"]), dataset_path)
            try:
                started = time.perf_counter()
                result = runtime.run(
                    image_path,
                    str(sample["question"]),
                    EVIDENCE_MODE,
                    system_prompt=build_live_pipeline_system_prompt(
                        sample, args.prompt_policy
                    ),
                    evidence_target_limit=evidence_target_limit(
                        args.prompt_policy
                    ),
                )
                end_to_end_latency = time.perf_counter() - started
                annotations = compact_grounding_annotations(
                    result["raw_annotations"]
                )
                scored = evaluate_live_prediction(
                    sample,
                    answer=result["answer"],
                    targets=result["targets"],
                    annotations=annotations,
                    iou_threshold=args.iou_threshold,
                )
                image_height, image_width = image_sizes[
                    int(sample["image_id"])
                ]
                mask_evaluation = evaluate_mask_evidence(
                    list(sample.get("evidence_boxes", [])),
                    annotations,
                    coco_annotations_by_id=coco_annotations,
                    image_height=image_height,
                    image_width=image_width,
                    iou_threshold=args.iou_threshold,
                )
                diagnostics = result["diagnostics"]
                grounding_latency = diagnostics.get(
                    "grounding_latency_seconds",
                    {"grounding": 0.0, "sam2": 0.0, "total": 0.0},
                )
                peak_memory = diagnostics.get(
                    "cuda_peak_memory_allocated_gb"
                )
                record = {
                    "id": sample_id,
                    "image": str(sample["image"]),
                    "image_id": int(sample["image_id"]),
                    "question": sample["question"],
                    "task_type": sample["task_type"],
                    "gt_answer": sample["gt_answer"],
                    "source": sample.get("source"),
                    "split": split_name,
                    "categories": sample.get("categories", []),
                    "evidence_boxes": sample.get("evidence_boxes", []),
                    "prediction": result["answer"],
                    "targets": result["targets"],
                    "target_source": result["target_source"],
                    "prompt_policy": args.prompt_policy,
                    "vlm_output": {
                        "raw_answer": result["vlm_raw_answer"],
                        "parse_source": diagnostics["vlm_parse_source"],
                        "schema_valid": diagnostics["vlm_schema_valid"],
                        "model": diagnostics.get("vlm_model"),
                        "generated_tokens": diagnostics.get(
                            "vlm_generated_tokens"
                        ),
                    },
                    "grounding": {
                        "prompt": diagnostics.get("grounding_prompt", ""),
                        "annotations": annotations,
                        "models": diagnostics.get("grounding_models"),
                        "thresholds": diagnostics.get("thresholds"),
                        "postprocessing": diagnostics.get(
                            "grounding_postprocessing", {}
                        ),
                        "latency_seconds": grounding_latency,
                        "artifacts": portable_gallery(
                            result["gallery"], PROJECT_ROOT
                        ),
                    },
                    **scored,
                    "mask_evaluation": mask_evaluation,
                    "latency_seconds": round(end_to_end_latency, 6),
                    "pipeline_latency_seconds": {
                        "vlm": float(
                            diagnostics.get("vlm_latency_seconds") or 0.0
                        ),
                        "grounding": float(
                            grounding_latency.get("grounding", 0.0)
                        ),
                        "sam2": float(grounding_latency.get("sam2", 0.0)),
                        "end_to_end": round(end_to_end_latency, 6),
                    },
                    **(
                        {"cuda_peak_memory_allocated_gb": peak_memory}
                        if peak_memory is not None
                        else {}
                    ),
                }
                append_jsonl(predictions_file, record)
                predictions_by_id[sample_id] = record
                successes_since_save += 1
                if successes_since_save >= args.save_every:
                    save_metrics(
                        metrics_path,
                        predictions_by_id,
                        selected_ids,
                        len(samples),
                        preflight_summary["required_evidence_questions"],
                        preflight_summary["negative_evidence_questions"],
                        historical_errors + invocation_errors,
                        "running",
                        args.iou_threshold,
                    )
                    successes_since_save = 0
            except Exception as exc:
                invocation_errors += 1
                append_jsonl(
                    errors_file,
                    {
                        "timestamp_utc": utc_now(),
                        "id": sample_id,
                        "image": str(image_path),
                        "task_type": sample.get("task_type"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                if args.fail_fast or (
                    args.max_errors
                    and invocation_errors >= args.max_errors
                ):
                    fatal_error = exc
                    break

    completed = len(set(predictions_by_id) & selected_ids)
    status = "completed" if completed == len(samples) else "incomplete"
    metrics = save_metrics(
        metrics_path,
        predictions_by_id,
        selected_ids,
        len(samples),
        preflight_summary["required_evidence_questions"],
        preflight_summary["negative_evidence_questions"],
        historical_errors + invocation_errors,
        status,
        args.iou_threshold,
    )
    print(f"Predictions: {predictions_path}")
    print(f"Errors:      {errors_path}")
    print(f"Metrics:     {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if fatal_error is not None:
        raise RuntimeError(
            f"Live pipeline stopped after {invocation_errors} errors."
        ) from fatal_error


if __name__ == "__main__":
    main()
