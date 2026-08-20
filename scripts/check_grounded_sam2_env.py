"""Check Grounded-SAM-2 imports, versions, and configured local files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grounded_sam2.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    report = {
        "config": str(config_path),
        "grounding_model": cfg["grounding"]["model_id"],
        "grounding_model_exists": Path(cfg["grounding"]["model_id"]).is_dir(),
        "sam2_checkpoint": cfg["sam2"]["checkpoint"],
        "sam2_checkpoint_exists": Path(cfg["sam2"]["checkpoint"]).is_file(),
    }
    try:
        import torch
        import transformers
        from sam2.build_sam import build_sam2  # noqa: F401
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
        from transformers import AutoModelForZeroShotObjectDetection  # noqa: F401

        report.update(
            {
                "imports_ok": True,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "transformers": transformers.__version__,
                "sam2_import": "OK",
            }
        )
    except Exception as exc:
        report.update(
            {
                "imports_ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

    report["ready"] = bool(
        report.get("imports_ok")
        and report["grounding_model_exists"]
        and report["sam2_checkpoint_exists"]
        and report.get("cuda_available")
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
