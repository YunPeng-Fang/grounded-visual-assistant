"""Evaluate Grounded-SAM-2 JSONL predictions with standard COCOeval."""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.coco_grounding_evaluation import (
    coco_stats_to_dict,
    convert_predictions_to_coco,
    filter_predictions_by_detector_score,
)
from grounded_visual_assistant.dataset_splits import load_image_ids
from grounded_visual_assistant.vlm_grounding import (
    aggregate_pipeline_latency,
    aggregate_prompt_quality,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standard COCO bbox and segmentation evaluation."
    )
    parser.add_argument(
        "--ground-truth", default="data/eval_v0/coco_grounding_gt.json"
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help=(
            "Grounded-SAM-2 predictions.jsonl. If omitted, exactly one file "
            "must exist below outputs/eval_grounding_v0."
        ),
    )
    parser.add_argument(
        "--prediction-root", default="outputs/eval_grounding_v0"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to a coco_eval directory beside predictions.jsonl.",
    )
    parser.add_argument(
        "--segmentation-score",
        choices=("detector", "mask", "product"),
        default="detector",
    )
    parser.add_argument(
        "--image-ids",
        default=None,
        help="JSON list or split metadata file selecting image IDs.",
    )
    parser.add_argument(
        "--min-detector-score",
        type=float,
        default=None,
        help="Offline-filter saved annotations by Grounding DINO score.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless predictions cover every ground-truth image.",
    )
    args = parser.parse_args()
    if args.min_detector_score is not None and not (
        0.0 <= args.min_detector_score <= 1.0
    ):
        parser.error("--min-detector-score must be between 0 and 1.")
    return args


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def discover_predictions(root: Path) -> Path:
    candidates = sorted(root.glob("**/predictions.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No predictions.jsonl found below: {root}")
    if len(candidates) > 1:
        choices = "\n".join(f"  {path}" for path in candidates)
        raise RuntimeError(
            "Multiple prediction files were found; pass --predictions explicitly:\n"
            f"{choices}"
        )
    return candidates[0]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return records


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _mean_valid(values: Any) -> float | None:
    valid = values[values > -1]
    return round(float(valid.mean()), 6) if valid.size else None


def per_category_metrics(evaluator: Any, coco_gt: Any) -> dict[str, Any]:
    """Extract per-category AP, AP50, and AP75 from COCOeval precision."""
    import numpy as np

    precision = evaluator.eval["precision"]
    iou_thresholds = evaluator.params.iouThrs
    iou_50 = np.where(np.isclose(iou_thresholds, 0.5))[0]
    iou_75 = np.where(np.isclose(iou_thresholds, 0.75))[0]
    category_names = {
        int(item["id"]): str(item["name"])
        for item in coco_gt.dataset["categories"]
    }
    results = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        category_precision = precision[:, :, category_index, 0, -1]
        results[category_names[int(category_id)]] = {
            "ap": _mean_valid(category_precision),
            "ap50": _mean_valid(category_precision[iou_50, :]),
            "ap75": _mean_valid(category_precision[iou_75, :]),
        }
    return results


def run_coco_eval(
    coco_gt: Any,
    results: list[dict[str, Any]],
    *,
    iou_type: str,
    image_ids: list[int],
    category_ids: list[int],
) -> tuple[dict[str, Any], str]:
    from pycocotools.cocoeval import COCOeval

    if not results:
        raise RuntimeError(f"No {iou_type} detections are available for COCOeval.")
    coco_results = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_results, iou_type)
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = category_ids
    captured = io.StringIO()
    with redirect_stdout(captured):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    summary = captured.getvalue()
    print(f"\n[{iou_type}]\n{summary}")
    return {
        "summary": coco_stats_to_dict(evaluator.stats),
        "per_category": per_category_metrics(evaluator, coco_gt),
    }, summary


def main() -> None:
    args = parse_args()
    ground_truth_path = project_path(args.ground_truth)
    if args.predictions:
        predictions_path = project_path(args.predictions)
    else:
        predictions_path = discover_predictions(project_path(args.prediction_root))
    if args.output_dir:
        output_dir = project_path(args.output_dir)
    else:
        output_dir = predictions_path.parent / "coco_eval"
        if args.image_ids:
            output_dir /= project_path(args.image_ids).stem
        if args.min_detector_score is not None:
            output_dir /= f"score-{args.min_detector_score:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Ground truth not found: {ground_truth_path}. Run "
            "scripts/build_coco_grounding_gt.py first."
        )
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")

    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    all_ground_truth_image_ids = {
        int(item["id"]) for item in ground_truth["images"]
    }
    image_ids_path = project_path(args.image_ids) if args.image_ids else None
    expected_image_ids = (
        set(load_image_ids(image_ids_path))
        if image_ids_path is not None
        else all_ground_truth_image_ids
    )
    unknown_image_ids = sorted(expected_image_ids - all_ground_truth_image_ids)
    if unknown_image_ids:
        raise ValueError(
            "Split contains image IDs absent from ground truth: "
            f"{unknown_image_ids[:10]}"
        )
    all_prediction_records = load_jsonl(predictions_path)
    prediction_records = [
        item
        for item in all_prediction_records
        if int(item["image_id"]) in expected_image_ids
    ]
    if args.min_detector_score is not None:
        prediction_records, score_filter_report = (
            filter_predictions_by_detector_score(
                prediction_records, args.min_detector_score
            )
        )
    else:
        annotation_count = sum(
            len(item.get("annotations", [])) for item in prediction_records
        )
        score_filter_report = {
            "min_detector_score": None,
            "prediction_records": len(prediction_records),
            "annotations_before": annotation_count,
            "annotations_after": annotation_count,
            "annotations_removed": 0,
            "retention_rate": 1.0 if annotation_count else 0.0,
        }
    completed_image_ids = {int(item["image_id"]) for item in prediction_records}
    missing_image_ids = sorted(expected_image_ids - completed_image_ids)
    if args.require_complete and missing_image_ids:
        raise RuntimeError(
            f"Predictions are missing {len(missing_image_ids)} images: "
            f"{missing_image_ids[:10]}"
        )
    evaluation_image_ids = sorted(expected_image_ids & completed_image_ids)
    prompt_sources = {
        str(item.get("prompt_source", "oracle")) for item in prediction_records
    }
    if len(prompt_sources) > 1:
        raise ValueError(
            f"Predictions mix multiple prompt sources: {sorted(prompt_sources)}"
        )
    prompt_source = next(iter(prompt_sources), "oracle")
    bbox_results, segmentation_results, conversion_report = (
        convert_predictions_to_coco(
            prediction_records,
            ground_truth,
            segmentation_score_mode=args.segmentation_score,
        )
    )
    bbox_results_path = output_dir / "coco_bbox_results.json"
    segmentation_results_path = output_dir / "coco_segm_results.json"
    write_json_atomic(bbox_results_path, bbox_results)
    write_json_atomic(segmentation_results_path, segmentation_results)

    category_ids = sorted(
        {
            int(item["category_id"])
            for item in ground_truth["annotations"]
            if int(item["image_id"]) in evaluation_image_ids
        }
    )

    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for standard COCO evaluation. Install "
            "requirements-grounded-sam2.txt in the server environment."
        ) from exc

    coco_gt = COCO(str(ground_truth_path))
    bbox_metrics, bbox_summary = run_coco_eval(
        coco_gt,
        bbox_results,
        iou_type="bbox",
        image_ids=evaluation_image_ids,
        category_ids=category_ids,
    )
    segmentation_metrics, segmentation_summary = run_coco_eval(
        coco_gt,
        segmentation_results,
        iou_type="segm",
        image_ids=evaluation_image_ids,
        category_ids=category_ids,
    )

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "oracle_conditioned_all_instances"
            if prompt_source == "oracle"
            else "vlm_prompted_against_oracle_target_instances"
        ),
        "prompt_source": prompt_source,
        "ground_truth": str(ground_truth_path),
        "predictions": str(predictions_path),
        "image_ids": str(image_ids_path) if image_ids_path is not None else None,
        "selected_image_ids": sorted(expected_image_ids),
        "coverage": {
            "expected_images": len(expected_image_ids),
            "completed_images": len(completed_image_ids & expected_image_ids),
            "evaluated_images": len(evaluation_image_ids),
            "missing_images": len(missing_image_ids),
            "missing_image_ids": missing_image_ids,
        },
        "ground_truth_instances": sum(
            int(item["image_id"]) in expected_image_ids
            for item in ground_truth["annotations"]
        ),
        "evaluated_ground_truth_instances": sum(
            int(item["image_id"]) in evaluation_image_ids
            for item in ground_truth["annotations"]
        ),
        "conversion": conversion_report,
        "score_filter": score_filter_report,
        "bbox": bbox_metrics,
        "segmentation": segmentation_metrics,
        "console_summaries": {
            "bbox": bbox_summary,
            "segmentation": segmentation_summary,
        },
    }
    if prompt_source == "vlm":
        metrics["prompt_quality"] = aggregate_prompt_quality(
            prediction_records,
            expected_images=len(expected_image_ids),
        )
        metrics["end_to_end_latency_seconds"] = aggregate_pipeline_latency(
            prediction_records
        )
    metrics_path = output_dir / "coco_metrics.json"
    write_json_atomic(metrics_path, metrics)
    print(f"BBox results: {bbox_results_path}")
    print(f"Segm results: {segmentation_results_path}")
    print(f"Metrics:      {metrics_path}")


if __name__ == "__main__":
    main()
