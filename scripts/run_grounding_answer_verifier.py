"""Run one Grounding-aware binary answer verification request."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.grounding_answer_verifier import (
    GROUNDING_ANSWER_VERIFIER_PROTOCOL,
    GroundingAnswerVerifierConfig,
    compact_grounding_result,
    verify_binary_answer,
)
from grounded_visual_assistant.pope_evaluation import evaluate_answer


DEFAULT_BASELINE_PREDICTIONS = (
    "outputs/eval_pope_v0/pope-full9000__qwen3-vl-8b-instruct/"
    "predictions.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one saved or manually supplied binary VLM answer with "
            "Grounding DINO and SAM 2.1."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/grounding_answer_verifier_v1.yaml",
    )
    parser.add_argument("--sample-id", default=None)
    parser.add_argument(
        "--baseline-predictions",
        default=DEFAULT_BASELINE_PREDICTIONS,
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--object", dest="target", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--baseline-answer", default=None)
    parser.add_argument(
        "--gt-answer",
        choices=("yes", "no"),
        default=None,
        help="Optional label used only to report before/after correctness.",
    )
    parser.add_argument("--grounding-model-id", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-config", default=None)
    parser.add_argument("--box-threshold", type=float, default=None)
    parser.add_argument("--text-threshold", type=float, default=None)
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.sample_id is None:
        missing = [
            name
            for name, value in (
                ("--image", args.image),
                ("--object", args.target),
                ("--baseline-answer", args.baseline_answer),
            )
            if not value
        ]
        if missing:
            parser.error(
                "Manual mode requires " + ", ".join(missing) + "."
            )
    for name in (
        "box_threshold",
        "text_threshold",
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
        or "verification"
    )


def load_prediction(path: Path, sample_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline predictions not found: {path}")
    match = None
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
            if str(record.get("id")) == sample_id:
                if match is not None:
                    raise ValueError(
                        f"Duplicate baseline prediction ID: {sample_id}"
                    )
                match = record
    if match is None:
        raise KeyError(f"Baseline sample ID not found: {sample_id}")
    required = {
        "image",
        "object",
        "question",
        "prediction",
        "gt_answer",
    }
    missing = required - match.keys()
    if missing:
        raise ValueError(
            f"Baseline prediction {sample_id} misses {sorted(missing)}."
        )
    return match


def resolve_request(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_id is not None:
        baseline_path = project_path(args.baseline_predictions)
        record = load_prediction(baseline_path, args.sample_id)
        return {
            "sample_id": args.sample_id,
            "source": "saved_pope_baseline",
            "baseline_predictions": str(baseline_path),
            "image": str(project_path(record["image"])),
            "target": str(record["object"]),
            "question": str(record["question"]),
            "baseline_answer": str(record["prediction"]),
            "gt_answer": str(record["gt_answer"]).lower(),
            "strategy": record.get("strategy"),
        }
    return {
        "sample_id": None,
        "source": "manual",
        "baseline_predictions": None,
        "image": str(project_path(args.image)),
        "target": str(args.target),
        "question": args.question,
        "baseline_answer": str(args.baseline_answer),
        "gt_answer": args.gt_answer,
        "strategy": None,
    }


def load_settings(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], GroundingAnswerVerifierConfig]:
    verifier_yaml = yaml.safe_load(
        project_path(args.config).read_text(encoding="utf-8")
    )
    if verifier_yaml.get("protocol") != GROUNDING_ANSWER_VERIFIER_PROTOCOL:
        raise ValueError(
            f"Unsupported verifier protocol: {verifier_yaml.get('protocol')}"
        )
    grounding_entry = dict(verifier_yaml["grounding"])
    grounding_yaml = yaml.safe_load(
        project_path(grounding_entry["config"]).read_text(encoding="utf-8")
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
        "nms_iou_threshold": grounding_entry.get("nms_iou_threshold"),
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
            "Detector box_threshold cannot exceed the verifier "
            "evidence_score_threshold because required candidates would be "
            "discarded before verification."
        )
    return grounding_settings, verifier_config


def main() -> None:
    args = parse_args()
    request = resolve_request(args)
    image_path = Path(request["image"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    grounding_settings, verifier_config = load_settings(args)
    output_root = (
        project_path(args.output_dir)
        if args.output_dir
        else project_path(
            yaml.safe_load(
                project_path(args.config).read_text(encoding="utf-8")
            )["runtime"]["output_dir"]
        )
    )
    run_name = slugify(
        request["sample_id"]
        or f"{image_path.stem}-{request['target']}"
    )
    output_dir = output_root / run_name
    preflight = {
        "protocol": GROUNDING_ANSWER_VERIFIER_PROTOCOL,
        "request": request,
        "grounding": grounding_settings,
        "verification": {
            "evidence_score_threshold": (
                verifier_config.evidence_score_threshold
            ),
            "promotion_score_threshold": (
                verifier_config.promotion_score_threshold
            ),
            "min_mask_score": verifier_config.min_mask_score,
            "min_mask_area_ratio": verifier_config.min_mask_area_ratio,
        },
        "output_dir": str(output_dir),
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.preflight_only:
        print("Preflight complete: no model was loaded.")
        return

    from grounded_visual_assistant.grounded_sam2 import (
        GroundedSam2,
        GroundedSam2Config,
    )

    grounder = GroundedSam2(
        GroundedSam2Config(**grounding_settings)
    )
    grounded = grounder.predict(
        image_path,
        f"{request['target']}.",
        output_dir=output_dir,
    )
    verification = verify_binary_answer(
        request["baseline_answer"],
        target=request["target"],
        annotations=grounded["annotations"],
        image_width=int(grounded["img_width"]),
        image_height=int(grounded["img_height"]),
        config=verifier_config,
    )
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": GROUNDING_ANSWER_VERIFIER_PROTOCOL,
        "request": request,
        "verification": verification,
        "grounding": compact_grounding_result(grounded),
    }
    if request["gt_answer"] is not None:
        payload["evaluation"] = {
            "baseline": evaluate_answer(
                request["baseline_answer"], request["gt_answer"]
            ),
            "verified": evaluate_answer(
                verification["final_answer"], request["gt_answer"]
            ),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "verification.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Answer: {verification['baseline_answer']} -> "
        f"{verification['final_answer']}"
    )
    print(f"Status: {verification['status']}")
    print(f"Result: {result_path}")


if __name__ == "__main__":
    main()
