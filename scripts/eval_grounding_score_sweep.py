"""Evaluate several detector-score cutoffs from one low-threshold run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT_ROOT / "scripts" / "eval_grounded_sam2_coco.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-sweep Grounding DINO score thresholds using one saved "
            "low-threshold prediction run."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--image-ids", default="data/eval_v0/splits/dev_image_ids.json"
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.25, 0.30, 0.35, 0.40, 0.45),
    )
    parser.add_argument(
        "--segmentation-score",
        choices=("detector", "mask", "product"),
        default="detector",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <prediction-run>/coco_eval/box_score_sweep.",
    )
    args = parser.parse_args()
    if any(not 0.0 <= value <= 1.0 for value in args.thresholds):
        parser.error("Every threshold must be between 0 and 1.")
    if len(args.thresholds) != len(set(args.thresholds)):
        parser.error("Threshold values must be unique.")
    return args


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(threshold: float, metrics: dict[str, Any]) -> dict[str, Any]:
    bbox = metrics["bbox"]["summary"]
    mask = metrics["segmentation"]["summary"]
    score_filter = metrics["score_filter"]
    conversion = metrics["conversion"]
    unmapped = int(conversion.get("skipped", {}).get("unmapped_label", 0))
    retained = int(score_filter["annotations_after"])
    mapped = int(conversion["bbox_detections"])
    return {
        "box_threshold": threshold,
        "detections": retained,
        "mapped_detections": mapped,
        "unmapped_detections": unmapped,
        "label_valid_rate": round(mapped / retained, 6) if retained else 0.0,
        "removed": score_filter["annotations_removed"],
        "retention_rate": score_filter["retention_rate"],
        "bbox_ap": bbox["ap"],
        "bbox_ap50": bbox["ap50"],
        "bbox_ap75": bbox["ap75"],
        "bbox_ap_small": bbox["ap_small"],
        "bbox_ar100": bbox["ar_max_100"],
        "mask_ap": mask["ap"],
        "mask_ap50": mask["ap50"],
        "mask_ap75": mask["ap75"],
        "mask_ap_small": mask["ap_small"],
        "mask_ar100": mask["ar_max_100"],
    }


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


def main() -> None:
    args = parse_args()
    predictions_path = project_path(args.predictions)
    image_ids_path = project_path(args.image_ids)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")
    if not image_ids_path.is_file():
        raise FileNotFoundError(f"Image split not found: {image_ids_path}")

    run_config_path = predictions_path.parent / "run_config.json"
    run_config = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.is_file()
        else {}
    )
    source_threshold = run_config.get("box_threshold")
    if source_threshold is not None:
        invalid = [
            value
            for value in args.thresholds
            if value + 1e-12 < float(source_threshold)
        ]
        if invalid:
            raise ValueError(
                "Offline thresholds cannot be lower than the source inference "
                f"threshold {source_threshold}: {invalid}"
            )

    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else predictions_path.parent / "coco_eval" / "box_score_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for threshold in sorted(args.thresholds):
        threshold_dir = output_dir / f"box-{threshold:.2f}"
        command = [
            sys.executable,
            str(EVALUATOR),
            "--predictions",
            str(predictions_path),
            "--image-ids",
            str(image_ids_path),
            "--min-detector-score",
            f"{threshold:.8f}",
            "--segmentation-score",
            args.segmentation_score,
            "--output-dir",
            str(threshold_dir),
            "--require-complete",
        ]
        print(f"\n=== Box score >= {threshold:.2f} ===", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        metrics_path = threshold_dir / "coco_metrics.json"
        rows.append(
            metric_row(
                threshold,
                json.loads(metrics_path.read_text(encoding="utf-8")),
            )
        )

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
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "offline_detector_score_sweep",
        "predictions": str(predictions_path),
        "predictions_sha256": sha256sum(predictions_path),
        "image_ids": str(image_ids_path),
        "source_box_threshold": source_threshold,
        "text_threshold": run_config.get("text_threshold"),
        "nms_iou_threshold": run_config.get("nms_iou_threshold"),
        "segmentation_score": args.segmentation_score,
        "selection_rule": (
            "prefer zero unmapped labels; then max mask AP, bbox AP, small "
            "mask AP, and fewer detections"
        ),
        "recommended_box_threshold": recommended["box_threshold"],
        "results": rows,
    }
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    write_json(summary_json, summary)
    write_csv(summary_csv, rows)

    columns = (
        "box_threshold",
        "detections",
        "unmapped_detections",
        "bbox_ap",
        "bbox_ap50",
        "mask_ap",
        "mask_ap50",
        "mask_ap_small",
    )
    print("\n" + "  ".join(f"{name:>14}" for name in columns))
    for row in rows:
        values = []
        for name in columns:
            value = row[name]
            values.append(
                f"{value:14.6f}" if isinstance(value, float) else f"{value:14}"
            )
        print("  ".join(values))
    print(f"\nProvisional recommendation: box={recommended['box_threshold']:.2f}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
