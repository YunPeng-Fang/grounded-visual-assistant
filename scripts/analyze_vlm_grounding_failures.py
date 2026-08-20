"""Generate a reproducible failure report from VLM-prompt grounding outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.failure_analysis import (
    aggregate_failure_analysis,
    render_failure_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute VLM-prompt grounding failures by pipeline stage."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="VLM-prompt Grounded-SAM-2 predictions.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <prediction-run>/failure_analysis.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions not found: {path}")
    records = []
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
            missing = {
                "id",
                "image_id",
                "prompt_source",
                "prompt_categories",
                "target_categories",
                "prompt_evaluation",
                "evaluation",
                "vlm_prediction",
            } - record.keys()
            if missing:
                raise ValueError(
                    f"Missing fields on {path}:{line_number}: {sorted(missing)}"
                )
            if record["prompt_source"] != "vlm":
                raise ValueError(
                    f"Expected prompt_source=vlm on {path}:{line_number}."
                )
            records.append(record)
    if not records:
        raise ValueError(f"Predictions file is empty: {path}")
    return records


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    list_fields = {
        "target_categories",
        "prompt_categories",
        "missed_categories",
        "off_target_categories",
        "parser_recoverable_categories",
        "generation_omitted_categories",
        "flags",
    }
    excluded_fields = {"vlm_answer"}
    fieldnames = [
        key for key in records[0] if key not in excluded_fields
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames}
            for key in list_fields:
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    predictions_path = project_path(args.predictions)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else predictions_path.parent / "failure_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, analyses = aggregate_failure_analysis(
        load_jsonl(predictions_path)
    )
    summary["predictions"] = str(predictions_path)
    summary_path = output_dir / "summary.json"
    per_image_path = output_dir / "per_image.jsonl"
    csv_path = output_dir / "per_image.csv"
    report_path = output_dir / "report.md"
    write_json_atomic(summary_path, summary)
    write_jsonl_atomic(per_image_path, analyses)
    write_csv(csv_path, analyses)
    report_path.write_text(
        render_failure_report(
            summary,
            analyses,
            predictions_path=str(predictions_path),
        ),
        encoding="utf-8",
    )

    attribution = summary["grounding_attribution"]
    print(f"Images:       {summary['coverage']['images']}")
    print(
        "Prompt FN:    "
        f"{attribution['false_negatives_from_missing_prompt_categories']} boxes"
    )
    print(
        "Grounding FN: "
        f"{attribution['false_negatives_after_category_was_prompted']} boxes"
    )
    print(f"Summary:      {summary_path}")
    print(f"Per-image:    {per_image_path}")
    print(f"CSV:          {csv_path}")
    print(f"Report:       {report_path}")


if __name__ == "__main__":
    main()
