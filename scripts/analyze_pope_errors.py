"""Generate reproducible error-attribution artifacts for a POPE run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.pope_error_analysis import (
    analyze_pope_predictions,
    render_case_sheet,
    render_pope_error_report,
    validate_pope_analysis_sources,
)


DEFAULT_PREDICTIONS = (
    "outputs/eval_pope_v0/pope-full9000__qwen3-vl-8b-instruct/"
    "predictions.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a complete saved POPE evaluation offline."
    )
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--metrics",
        default=None,
        help="Defaults to metrics.json beside predictions.",
    )
    parser.add_argument(
        "--run-config",
        default=None,
        help="Defaults to run_config.json beside predictions.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <prediction-run>/error_analysis.",
    )
    parser.add_argument("--top-objects", type=int, default=15)
    parser.add_argument("--representative-cases", type=int, default=12)
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Do not render representative JPEG contact sheets.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on {path}:{line_number}: {exc}"
                ) from exc
    if not records:
        raise ValueError(f"JSONL file is empty: {path}")
    return records


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(
    path: Path, records: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv_atomic(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError(f"Cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(records[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = ";".join(str(item) for item in value)
            writer.writerow(row)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    predictions_path = project_path(args.predictions)
    metrics_path = (
        project_path(args.metrics)
        if args.metrics
        else predictions_path.parent / "metrics.json"
    )
    run_config_path = (
        project_path(args.run_config)
        if args.run_config
        else predictions_path.parent / "run_config.json"
    )
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else predictions_path.parent / "error_analysis"
    )

    analysis = analyze_pope_predictions(
        read_jsonl(predictions_path),
        representative_limit=args.representative_cases,
        top_n=args.top_objects,
    )
    validate_pope_analysis_sources(
        analysis,
        metrics=read_json(metrics_path),
        run_config=read_json(run_config_path),
    )
    analysis.summary["sources"] = {
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "run_config": str(run_config_path),
    }

    visual_paths = {}
    if not args.skip_visuals:
        for error_type, filename, title in (
            (
                "false_negative",
                "false_negative_cases.jpg",
                "Representative POPE False Negatives",
            ),
            (
                "false_positive",
                "false_positive_cases.jpg",
                "Representative POPE False Positives",
            ),
        ):
            cases = [
                item
                for item in analysis.representative_cases
                if item["error_type"] == error_type
            ]
            render_result = render_case_sheet(
                cases,
                project_root=PROJECT_ROOT,
                output_path=output_dir / filename,
                title=title,
            )
            if render_result["rendered"]:
                visual_paths[error_type] = filename
            analysis.summary.setdefault("visuals", {})[error_type] = (
                render_result
            )

    write_json_atomic(output_dir / "summary.json", analysis.summary)
    write_jsonl_atomic(output_dir / "errors.jsonl", analysis.errors)
    write_jsonl_atomic(
        output_dir / "representative_cases.jsonl",
        analysis.representative_cases,
    )
    write_csv_atomic(output_dir / "per_object.csv", analysis.per_object)
    write_csv_atomic(output_dir / "per_image.csv", analysis.per_image)
    (output_dir / "report.md").write_text(
        render_pope_error_report(
            analysis,
            predictions_path=str(predictions_path),
            visual_paths=visual_paths,
        ),
        encoding="utf-8",
    )

    attribution = analysis.summary["error_attribution"]
    print(f"Predictions:   {analysis.summary['coverage']['predictions']}")
    print(f"Raw errors:    {attribution['raw_error_questions']}")
    print(
        "Unique FN/FP: "
        f"{attribution['unique_false_negative_queries']} / "
        f"{attribution['unique_false_positive_queries']}"
    )
    print(f"Output:        {output_dir}")
    print(f"Report:        {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
