"""Generate an offline failure report for a complete Hard-Dev VLM run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_benchmark import read_jsonl
from grounded_visual_assistant.hard_vlm_analysis import (
    analyze_hard_vlm_predictions,
    render_hard_vlm_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a complete frozen Hard-Dev VLM run offline."
    )
    parser.add_argument(
        "--dataset",
        default="data/cross_dataset_hard_v1/questions_v1/dev_questions.jsonl",
    )
    parser.add_argument(
        "--predictions",
        default=(
            "outputs/cross_dataset_hard_v1/vlm/"
            "hard-dev400__qwen3-vl-8b-instruct/predictions.jsonl"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset_path = project_path(args.dataset)
    predictions_path = project_path(args.predictions)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else predictions_path.parent / "failure_analysis"
    )
    dataset = read_jsonl(dataset_path)
    predictions = read_jsonl(predictions_path)
    summary, analyses = analyze_hard_vlm_predictions(dataset, predictions)

    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_sample.jsonl", analyses)
    fieldnames = [
        "id",
        "sample_id",
        "source",
        "split",
        "task_type",
        "gt_answer",
        "parsed_prediction",
        "score",
        "is_correct",
        "parse_valid",
        "hit_max_new_tokens",
        "generated_tokens",
        "missed_categories",
        "extra_categories",
        "flags",
        "severity",
    ]
    with (output_dir / "per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in analyses:
            row = {key: item.get(key) for key in fieldnames}
            for key in ("missed_categories", "extra_categories", "flags"):
                row[key] = ";".join(row[key] or [])
            writer.writerow(row)
    (output_dir / "report.md").write_text(
        render_hard_vlm_report(summary, predictions_path=str(predictions_path)),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
