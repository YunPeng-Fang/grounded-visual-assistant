"""Run the single-image VLM baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.io_utils import save_json
from grounded_visual_assistant.vlm_baseline import VlmBaseline, VlmBaselineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-image VLM baseline.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--question", default=None, help="Question to ask about the image.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--model-id", default=None, help="Override model id.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/processor only from local files or cache; never download.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    question = args.question or cfg["prompts"]["default_question"]
    model_cfg = cfg["model"]
    if args.model_id:
        model_cfg["model_id"] = args.model_id

    runner = VlmBaseline(
        VlmBaselineConfig(
            model_id=model_cfg["model_id"],
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            device_map=model_cfg.get("device_map", "auto"),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 256)),
            do_sample=bool(model_cfg.get("do_sample", False)),
            local_files_only=bool(args.local_files_only or model_cfg.get("local_files_only", False)),
        )
    )
    result = runner.answer(args.image, question)
    output_dir = args.output_dir or cfg["runtime"]["output_dir"]
    output_path = save_json(result, PROJECT_ROOT / output_dir, prefix="vlm_baseline")

    print(result["answer"])
    print(f"\nSaved result to: {output_path}")


if __name__ == "__main__":
    main()
