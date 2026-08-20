"""Combine box-score sweeps from several Grounding DINO text thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare completed box-score sweeps across text thresholds."
    )
    parser.add_argument(
        "--summaries",
        nargs="*",
        default=None,
        help="Explicit box_score_sweep/summary.json files.",
    )
    parser.add_argument(
        "--root", default="outputs/eval_grounding_v0"
    )
    parser.add_argument(
        "--pattern",
        default=(
            "dev__box-0.25__text-*__nms-none/"
            "coco_eval/box_score_sweep/summary.json"
        ),
    )
    parser.add_argument(
        "--require-text-thresholds",
        type=float,
        nargs="+",
        default=(0.20, 0.30, 0.40),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval_grounding_v0/dev__box-text-threshold-sweep",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def enriched_rows(summary_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    text_threshold = summary.get("text_threshold")
    if text_threshold is None:
        raise ValueError(f"Summary has no text_threshold: {summary_path}")
    rows = []
    for source_row in summary.get("results", []):
        row = dict(source_row)
        box_threshold = float(row["box_threshold"])
        metrics_path = summary_path.parent / f"box-{box_threshold:.2f}" / "coco_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Metrics not found: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        conversion = metrics["conversion"]
        retained = int(metrics["score_filter"]["annotations_after"])
        mapped = int(conversion["bbox_detections"])
        unmapped = int(conversion.get("skipped", {}).get("unmapped_label", 0))
        row.update(
            {
                "text_threshold": float(text_threshold),
                "box_threshold": box_threshold,
                "mapped_detections": mapped,
                "unmapped_detections": unmapped,
                "label_valid_rate": round(mapped / retained, 6)
                if retained
                else 0.0,
                "evaluated_images": int(metrics["coverage"]["evaluated_images"]),
                "source_summary": str(summary_path),
            }
        )
        rows.append(row)
    return summary, rows


def main() -> None:
    args = parse_args()
    root = project_path(args.root)
    if args.summaries:
        summary_paths = [project_path(value) for value in args.summaries]
    else:
        summary_paths = sorted(root.glob(args.pattern))
    if not summary_paths:
        raise FileNotFoundError(
            f"No score-sweep summaries matched {root / args.pattern}"
        )

    summaries = []
    rows = []
    observed_text_thresholds = set()
    for summary_path in summary_paths:
        summary, summary_rows = enriched_rows(summary_path)
        text_threshold = float(summary["text_threshold"])
        if text_threshold in observed_text_thresholds:
            raise ValueError(f"Duplicate text threshold {text_threshold}.")
        observed_text_thresholds.add(text_threshold)
        summaries.append(summary)
        rows.extend(summary_rows)

    required = {float(value) for value in args.require_text_thresholds}
    missing = sorted(required - observed_text_thresholds)
    if missing:
        raise RuntimeError(
            f"Missing required text-threshold sweeps: {missing}; "
            f"observed {sorted(observed_text_thresholds)}"
        )
    rows.sort(key=lambda row: (row["text_threshold"], row["box_threshold"]))

    zero_unmapped_rows = [row for row in rows if row["unmapped_detections"] == 0]
    selection_pool = zero_unmapped_rows or rows
    recommended = max(
        selection_pool,
        key=lambda row: (
            row["mask_ap"],
            row["bbox_ap"],
            row["mask_ap_small"],
            -row["detections"],
            row["box_threshold"],
        ),
    )

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "dev_box_and_text_threshold_sweep",
        "source_summaries": [str(path) for path in summary_paths],
        "selection_rule": (
            "prefer zero unmapped labels; then max mask AP, bbox AP, small "
            "mask AP, and fewer detections"
        ),
        "recommended": {
            "box_threshold": recommended["box_threshold"],
            "text_threshold": recommended["text_threshold"],
        },
        "results": rows,
    }
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    write_json(summary_json, payload)
    write_csv(summary_csv, rows)

    columns = (
        "text_threshold",
        "box_threshold",
        "detections",
        "unmapped_detections",
        "bbox_ap",
        "mask_ap",
        "mask_ap_small",
    )
    print("  ".join(f"{name:>14}" for name in columns))
    for row in rows:
        values = []
        for name in columns:
            value = row[name]
            values.append(
                f"{value:14.6f}" if isinstance(value, float) else f"{value:14}"
            )
        print("  ".join(values))
    print(
        "\nProvisional recommendation: "
        f"box={recommended['box_threshold']:.2f}, "
        f"text={recommended['text_threshold']:.2f}"
    )
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
