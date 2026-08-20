"""Compare Hard-Dev relation prompt v1 and v2 predictions offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_benchmark import read_jsonl
from grounded_visual_assistant.relation_prompt_comparison import (
    compare_relation_prompts,
    render_relation_prompt_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare relation prompts offline.")
    parser.add_argument(
        "--baseline",
        default=(
            "outputs/cross_dataset_hard_v1/vlm/"
            "hard-dev400__qwen3-vl-8b-instruct/predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--candidate",
        default=(
            "outputs/cross_dataset_hard_v1/vlm/"
            "hard-dev200-relation__qwen3-vl-8b-instruct__prompt-v2/"
            "predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cross_dataset_hard_v1/relation_prompt_v1_vs_v2",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    baseline_path = project_path(args.baseline)
    candidate_path = project_path(args.candidate)
    output_dir = project_path(args.output_dir)
    summary = compare_relation_prompts(
        read_jsonl(baseline_path), read_jsonl(candidate_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        render_relation_prompt_comparison(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
