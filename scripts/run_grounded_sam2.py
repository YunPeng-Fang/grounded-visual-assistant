"""Run the official Grounded-SAM-2 image pipeline through the project adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-ground objects with Grounding DINO and segment with SAM 2.1."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--prompt",
        required=True,
        help='Period-separated targets, for example "person. umbrella."',
    )
    parser.add_argument("--config", default="configs/grounded_sam2.yaml")
    parser.add_argument("--grounding-model-id", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-config", default=None)
    parser.add_argument("--box-threshold", type=float, default=None)
    parser.add_argument("--text-threshold", type=float, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    grounding_cfg = cfg["grounding"]
    sam2_cfg = cfg["sam2"]
    runtime_cfg = cfg["runtime"]

    image_path = project_path(args.image)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else project_path(runtime_cfg["output_dir"]) / image_path.stem
    )

    from grounded_visual_assistant.grounded_sam2 import (
        GroundedSam2,
        GroundedSam2Config,
    )

    runner = GroundedSam2(
        GroundedSam2Config(
            grounding_model_id=(
                args.grounding_model_id or grounding_cfg["model_id"]
            ),
            sam2_checkpoint=args.sam2_checkpoint or sam2_cfg["checkpoint"],
            sam2_model_config=(
                args.sam2_model_config or sam2_cfg["model_config"]
            ),
            box_threshold=(
                args.box_threshold
                if args.box_threshold is not None
                else float(grounding_cfg.get("box_threshold", 0.4))
            ),
            text_threshold=(
                args.text_threshold
                if args.text_threshold is not None
                else float(grounding_cfg.get("text_threshold", 0.3))
            ),
            nms_iou_threshold=(
                args.nms_iou_threshold
                if args.nms_iou_threshold is not None
                else grounding_cfg.get("nms_iou_threshold")
            ),
            device=args.device or runtime_cfg.get("device", "cuda"),
            dtype=args.dtype or runtime_cfg.get("dtype", "float16"),
            local_files_only=bool(
                args.local_files_only or grounding_cfg.get("local_files_only", False)
            ),
        )
    )
    result = runner.predict(image_path, args.prompt, output_dir=output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved Grounded-SAM-2 artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
