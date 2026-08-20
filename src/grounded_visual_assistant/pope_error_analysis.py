"""Offline error attribution for completed POPE evaluations."""

from __future__ import annotations

import hashlib
import statistics
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .pope_dataset import POPE_STRATEGIES
from .pope_evaluation import binary_metrics, evaluate_answer


@dataclass(frozen=True)
class PopeErrorAnalysis:
    """Structured outputs produced by one POPE error-analysis run."""

    summary: dict[str, Any]
    errors: list[dict[str, Any]]
    per_object: list[dict[str, Any]]
    per_image: list[dict[str, Any]]
    representative_cases: list[dict[str, Any]]


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _semantic_query_key(record: Mapping[str, Any]) -> str:
    payload = "\n".join(
        (
            str(record["image_id"]),
            str(record["object"]).strip().lower(),
            str(record["question"]).strip().lower(),
            str(record["gt_answer"]).strip().lower(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _analyze_prediction(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "id",
        "strategy",
        "image_id",
        "image",
        "question",
        "object",
        "gt_answer",
        "prediction",
        "evaluation",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(
            f"POPE prediction {record.get('id', '<unknown>')} is missing "
            f"fields: {sorted(missing)}"
        )

    strategy = str(record["strategy"])
    if strategy not in POPE_STRATEGIES:
        raise ValueError(
            f"Unsupported POPE strategy for {record['id']}: {strategy!r}"
        )
    target = str(record["gt_answer"]).strip().lower()
    recomputed = evaluate_answer(str(record["prediction"]), target)
    saved = record["evaluation"]
    for field, expected in recomputed.items():
        if saved.get(field) != expected:
            raise RuntimeError(
                f"Saved POPE evaluation does not reproduce for "
                f"{record['id']} field {field!r}."
            )

    prediction = recomputed["official_prediction"]
    if target == "yes":
        error_type = "true_positive" if prediction == "yes" else "false_negative"
    else:
        error_type = "false_positive" if prediction == "yes" else "true_negative"
    return {
        "id": str(record["id"]),
        "semantic_query_key": _semantic_query_key(record),
        "strategy": strategy,
        "image_id": int(record["image_id"]),
        "image": str(record["image"]),
        "object": str(record["object"]).strip().lower(),
        "question": str(record["question"]),
        "gt_answer": target,
        "prediction": str(record["prediction"]),
        "parsed_prediction": prediction,
        "error_type": error_type,
        "is_error": error_type in {"false_positive", "false_negative"},
        "latency_seconds": float(record.get("latency_seconds", 0.0)),
        "generated_tokens": record.get("generated_tokens"),
        "strict_parse_valid": bool(recomputed["strict_parse_valid"]),
    }


def _object_rows(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "counts": Counter(),
            "positive_keys": set(),
            "false_negative_keys": set(),
            "negative_keys": set(),
            "false_positive_keys": set(),
            "strategies": set(),
        }
    )
    for item in analyses:
        bucket = buckets[item["object"]]
        bucket["counts"][item["error_type"]] += 1
        bucket["strategies"].add(item["strategy"])
        if item["gt_answer"] == "yes":
            bucket["positive_keys"].add(item["semantic_query_key"])
            if item["error_type"] == "false_negative":
                bucket["false_negative_keys"].add(item["semantic_query_key"])
        else:
            bucket["negative_keys"].add(item["semantic_query_key"])
            if item["error_type"] == "false_positive":
                bucket["false_positive_keys"].add(item["semantic_query_key"])

    rows = []
    for object_name, bucket in buckets.items():
        counts = bucket["counts"]
        tp = counts["true_positive"]
        fp = counts["false_positive"]
        tn = counts["true_negative"]
        fn = counts["false_negative"]
        total = tp + fp + tn + fn
        positive = tp + fn
        negative = tn + fp
        rows.append(
            {
                "object": object_name,
                "total_questions": total,
                "errors": fp + fn,
                "accuracy": _ratio(tp + tn, total),
                "positive_questions": positive,
                "true_positives": tp,
                "false_negatives": fn,
                "recall": _ratio(tp, positive),
                "unique_positive_queries": len(bucket["positive_keys"]),
                "unique_false_negative_queries": len(
                    bucket["false_negative_keys"]
                ),
                "negative_questions": negative,
                "true_negatives": tn,
                "false_positives": fp,
                "false_positive_rate": _ratio(fp, negative),
                "unique_negative_queries": len(bucket["negative_keys"]),
                "unique_false_positive_queries": len(
                    bucket["false_positive_keys"]
                ),
                "strategies": sorted(bucket["strategies"]),
            }
        )
    return sorted(
        rows,
        key=lambda item: (-item["errors"], item["object"]),
    )


def _image_rows(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "image": "",
            "counts": Counter(),
            "error_keys": set(),
            "false_negative_keys": set(),
            "false_positive_keys": set(),
            "objects": set(),
            "strategies": set(),
        }
    )
    for item in analyses:
        bucket = buckets[item["image_id"]]
        bucket["image"] = item["image"]
        bucket["counts"][item["error_type"]] += 1
        if item["is_error"]:
            bucket["error_keys"].add(item["semantic_query_key"])
            bucket["objects"].add(item["object"])
            bucket["strategies"].add(item["strategy"])
        if item["error_type"] == "false_negative":
            bucket["false_negative_keys"].add(item["semantic_query_key"])
        if item["error_type"] == "false_positive":
            bucket["false_positive_keys"].add(item["semantic_query_key"])

    rows = []
    for image_id, bucket in buckets.items():
        counts = bucket["counts"]
        raw_errors = counts["false_negative"] + counts["false_positive"]
        rows.append(
            {
                "image_id": image_id,
                "image": bucket["image"],
                "questions": sum(bucket["counts"].values()),
                "raw_errors": raw_errors,
                "unique_error_queries": len(bucket["error_keys"]),
                "false_negatives": counts["false_negative"],
                "unique_false_negative_queries": len(
                    bucket["false_negative_keys"]
                ),
                "false_positives": counts["false_positive"],
                "unique_false_positive_queries": len(
                    bucket["false_positive_keys"]
                ),
                "error_objects": sorted(bucket["objects"]),
                "error_strategies": sorted(bucket["strategies"]),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            -item["unique_error_queries"],
            -item["raw_errors"],
            item["image_id"],
        ),
    )


def _select_representatives(
    errors: list[dict[str, Any]],
    *,
    error_type: str,
    limit: int,
) -> list[dict[str, Any]]:
    candidates = [item for item in errors if item["error_type"] == error_type]
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_key[item["semantic_query_key"]].append(item)

    deduplicated = []
    strategy_order = {name: index for index, name in enumerate(POPE_STRATEGIES)}
    for records in by_key.values():
        records = sorted(
            records,
            key=lambda item: (strategy_order[item["strategy"]], item["id"]),
        )
        representative = dict(records[0])
        representative["observed_strategies"] = sorted(
            {item["strategy"] for item in records},
            key=strategy_order.get,
        )
        deduplicated.append(representative)

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(
        deduplicated,
        key=lambda value: (value["object"], value["id"]),
    ):
        by_object[item["object"]].append(item)
    object_order = sorted(
        by_object,
        key=lambda name: (-len(by_object[name]), name),
    )

    selected = []
    depth = 0
    while len(selected) < limit:
        added = False
        for object_name in object_order:
            records = by_object[object_name]
            if depth < len(records):
                selected.append(records[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def analyze_pope_predictions(
    prediction_records: Iterable[Mapping[str, Any]],
    *,
    representative_limit: int = 12,
    top_n: int = 15,
) -> PopeErrorAnalysis:
    """Validate predictions and summarize POPE failures without model inference."""
    if representative_limit < 0:
        raise ValueError("representative_limit must be non-negative.")
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    prediction_records = [dict(item) for item in prediction_records]
    if not prediction_records:
        raise ValueError("No POPE prediction records were provided.")
    ids = [str(item.get("id")) for item in prediction_records]
    if len(ids) != len(set(ids)):
        raise ValueError("POPE predictions contain duplicate IDs.")

    analyses = [_analyze_prediction(item) for item in prediction_records]
    errors = [item for item in analyses if item["is_error"]]
    objects = _object_rows(analyses)
    images = _image_rows(analyses)

    strategy_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_item in prediction_records:
        strategy_records[str(raw_item["strategy"])].append(raw_item)
    strategy_metrics = {
        strategy: binary_metrics(strategy_records[strategy])
        for strategy in POPE_STRATEGIES
        if strategy_records[strategy]
    }
    overall = binary_metrics(prediction_records)

    positive_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in analyses:
        if item["gt_answer"] == "yes":
            positive_groups[item["semantic_query_key"]].append(item)
    expected_strategies = set(POPE_STRATEGIES)
    complete_positive_groups = sum(
        {item["strategy"] for item in group} == expected_strategies
        for group in positive_groups.values()
    )
    positive_disagreements = [
        {
            "semantic_query_key": key,
            "image_id": group[0]["image_id"],
            "object": group[0]["object"],
            "predictions": {
                item["strategy"]: item["parsed_prediction"] for item in group
            },
        }
        for key, group in positive_groups.items()
        if len({item["parsed_prediction"] for item in group}) > 1
    ]

    fn_keys = {
        item["semantic_query_key"]
        for item in errors
        if item["error_type"] == "false_negative"
    }
    fp_keys = {
        item["semantic_query_key"]
        for item in errors
        if item["error_type"] == "false_positive"
    }
    false_negatives = overall["confusion"]["fn"]
    false_positives = overall["confusion"]["fp"]
    unique_error_keys = fn_keys | fp_keys
    latencies = [item["latency_seconds"] for item in analyses]

    top_false_negative_objects = sorted(
        (item for item in objects if item["false_negatives"]),
        key=lambda item: (
            -item["unique_false_negative_queries"],
            -item["false_negatives"],
            item["object"],
        ),
    )[:top_n]
    top_false_positive_objects = sorted(
        (item for item in objects if item["false_positives"]),
        key=lambda item: (
            -item["false_positives"],
            -item["false_positive_rate"],
            item["object"],
        ),
    )[:top_n]

    representatives = [
        *_select_representatives(
            errors,
            error_type="false_negative",
            limit=representative_limit,
        ),
        *_select_representatives(
            errors,
            error_type="false_positive",
            limit=representative_limit,
        ),
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "coverage": {
            "predictions": len(analyses),
            "unique_ids": len(ids),
            "images": len({item["image_id"] for item in analyses}),
            "objects": len(objects),
            "strategies": {
                strategy: len(strategy_records[strategy])
                for strategy in POPE_STRATEGIES
                if strategy_records[strategy]
            },
            "strict_parse_valid": sum(
                item["strict_parse_valid"] for item in analyses
            ),
        },
        "overall": overall,
        "strategies": strategy_metrics,
        "error_attribution": {
            "raw_error_questions": len(errors),
            "raw_error_rate": _ratio(len(errors), len(analyses)),
            "false_negative_questions": false_negatives,
            "false_positive_questions": false_positives,
            "false_negative_share": _ratio(false_negatives, len(errors)),
            "false_positive_share": _ratio(false_positives, len(errors)),
            "unique_error_queries": len(unique_error_keys),
            "unique_false_negative_queries": len(fn_keys),
            "unique_false_positive_queries": len(fp_keys),
        },
        "positive_query_repetition": {
            "unique_positive_queries": len(positive_groups),
            "complete_three_strategy_groups": complete_positive_groups,
            "unanimous_prediction_groups": (
                len(positive_groups) - len(positive_disagreements)
            ),
            "cross_strategy_disagreements": len(positive_disagreements),
            "disagreement_examples": positive_disagreements[:top_n],
        },
        "latency_seconds": {
            "mean": round(statistics.fmean(latencies), 6),
            "median": round(statistics.median(latencies), 6),
        },
        "top_false_negative_objects": top_false_negative_objects,
        "top_false_positive_objects": top_false_positive_objects,
        "hardest_images": images[:top_n],
    }
    return PopeErrorAnalysis(
        summary=summary,
        errors=sorted(
            errors,
            key=lambda item: (
                item["error_type"],
                item["object"],
                item["id"],
            ),
        ),
        per_object=objects,
        per_image=images,
        representative_cases=representatives,
    )


def validate_pope_analysis_sources(
    analysis: PopeErrorAnalysis,
    *,
    metrics: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> None:
    """Ensure the derived analysis belongs to the completed saved run."""
    if metrics.get("status") != "completed":
        raise RuntimeError("POPE metrics do not describe a completed run.")
    coverage = metrics.get("coverage") or {}
    analyzed = analysis.summary["coverage"]["predictions"]
    if coverage.get("completed") != analyzed or coverage.get("expected") != analyzed:
        raise RuntimeError(
            "POPE metrics coverage does not match the analyzed predictions."
        )
    if (metrics.get("overall") or {}).get("confusion") != (
        analysis.summary["overall"]["confusion"]
    ):
        raise RuntimeError(
            "POPE metrics confusion matrix does not reproduce from predictions."
        )
    if metrics.get("protocol") != run_config.get("protocol"):
        raise RuntimeError("POPE metrics and run config protocols do not match.")
    if run_config.get("strategy") != "all":
        raise RuntimeError("POPE final analysis requires strategy='all'.")
    if not run_config.get("require_complete"):
        raise RuntimeError("POPE final analysis requires require_complete=true.")


def _load_font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    candidates = (
        (
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "DejaVuSans-Bold.ttf",
        )
        if bold
        else (
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "DejaVuSans.ttf",
        )
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_case_sheet(
    cases: Iterable[Mapping[str, Any]],
    *,
    project_root: Path,
    output_path: Path,
    title: str,
    columns: int = 3,
) -> dict[str, Any]:
    """Render a compact visual sheet for representative POPE errors."""
    from PIL import Image, ImageDraw, ImageOps

    cases = list(cases)
    if not cases:
        return {"rendered": 0, "missing_images": []}
    if columns <= 0:
        raise ValueError("columns must be positive.")

    panel_width = 420
    panel_height = 390
    image_width = 396
    image_height = 248
    margin = 18
    header_height = 58
    rows = (len(cases) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + columns * panel_width,
            header_height + margin + rows * panel_height,
        ),
        "#F4F6F8",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(26, bold=True)
    label_font = _load_font(17, bold=True)
    body_font = _load_font(15)
    draw.text((margin, 14), title, fill="#18212B", font=title_font)

    missing_images = []
    for index, case in enumerate(cases):
        row, column = divmod(index, columns)
        x = margin + column * panel_width
        y = header_height + row * panel_height
        draw.rounded_rectangle(
            (x, y, x + panel_width - 12, y + panel_height - 12),
            radius=6,
            fill="#FFFFFF",
            outline="#CDD5DF",
            width=1,
        )
        image_path = Path(str(case["image"]))
        if not image_path.is_absolute():
            image_path = project_root / image_path
        image_box = (
            x + 6,
            y + 6,
            x + 6 + image_width,
            y + 6 + image_height,
        )
        if image_path.is_file():
            with Image.open(image_path) as source:
                fitted = ImageOps.contain(
                    source.convert("RGB"),
                    (image_width, image_height),
                    method=Image.Resampling.LANCZOS,
                )
            background = Image.new("RGB", (image_width, image_height), "#E9EDF2")
            background.paste(
                fitted,
                (
                    (image_width - fitted.width) // 2,
                    (image_height - fitted.height) // 2,
                ),
            )
            canvas.paste(background, (image_box[0], image_box[1]))
        else:
            missing_images.append(str(image_path))
            draw.rectangle(image_box, fill="#E9EDF2")
            draw.text(
                (x + 20, y + 112),
                "Image unavailable",
                fill="#647181",
                font=body_font,
            )

        error_label = (
            "FALSE NEGATIVE"
            if case["error_type"] == "false_negative"
            else "FALSE POSITIVE"
        )
        accent = "#B93838" if case["error_type"] == "false_negative" else "#A45B00"
        text_y = y + image_height + 18
        draw.text(
            (x + 12, text_y),
            f"{error_label} | {case['object']}",
            fill=accent,
            font=label_font,
        )
        draw.text(
            (x + 12, text_y + 25),
            (
                f"GT {case['gt_answer'].title()} -> "
                f"Pred {case['parsed_prediction'].title()} | "
                f"{case['strategy']}"
            ),
            fill="#25313D",
            font=body_font,
        )
        question_lines = textwrap.wrap(str(case["question"]), width=49)[:3]
        draw.multiline_text(
            (x + 12, text_y + 49),
            "\n".join(question_lines),
            fill="#465464",
            font=body_font,
            spacing=3,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, optimize=True)
    return {
        "rendered": len(cases),
        "missing_images": missing_images,
        "path": str(output_path),
    }


def render_pope_error_report(
    analysis: PopeErrorAnalysis,
    *,
    predictions_path: str,
    visual_paths: Mapping[str, str] | None = None,
) -> str:
    """Render the POPE error attribution as a compact Markdown report."""
    summary = analysis.summary
    coverage = summary["coverage"]
    overall = summary["overall"]
    attribution = summary["error_attribution"]
    repetition = summary["positive_query_repetition"]
    lines = [
        "# POPE Error Attribution Report",
        "",
        f"Predictions: `{predictions_path}`",
        "",
        "## Integrity",
        "",
        f"- Predictions: {coverage['predictions']}",
        f"- Images: {coverage['images']}",
        f"- Queried objects: {coverage['objects']}",
        (
            "- Strict Yes/No parses: "
            f"{coverage['strict_parse_valid']} / {coverage['predictions']}"
        ),
        "",
        "## Headline",
        "",
        f"- Accuracy: {overall['accuracy']:.4f}",
        f"- Precision: {overall['precision']:.4f}",
        f"- Recall: {overall['recall']:.4f}",
        f"- F1: {overall['f1']:.4f}",
        f"- Raw error questions: {attribution['raw_error_questions']}",
        (
            "- False negatives / false positives: "
            f"{attribution['false_negative_questions']} / "
            f"{attribution['false_positive_questions']}"
        ),
        (
            "- Unique false-negative / false-positive queries: "
            f"{attribution['unique_false_negative_queries']} / "
            f"{attribution['unique_false_positive_queries']}"
        ),
        "",
        "## Strategy Breakdown",
        "",
        "| Strategy | Count | Accuracy | Precision | Recall | F1 | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in POPE_STRATEGIES:
        metrics = summary["strategies"].get(strategy)
        if metrics is None:
            continue
        confusion = metrics["confusion"]
        lines.append(
            f"| {strategy} | {metrics['count']} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {confusion['fp']} | {confusion['fn']} |"
        )

    lines.extend(
        [
            "",
            "## Positive-Query Repetition Audit",
            "",
            (
                "- Unique positive queries: "
                f"{repetition['unique_positive_queries']}"
            ),
            (
                "- Complete three-strategy groups: "
                f"{repetition['complete_three_strategy_groups']}"
            ),
            (
                "- Cross-strategy prediction disagreements: "
                f"{repetition['cross_strategy_disagreements']}"
            ),
            "",
            (
                "POPE reuses each positive query across random, popular, and "
                "adversarial variants. Raw false-negative counts therefore "
                "include protocol-level repetition; unique-query counts are "
                "the appropriate number for qualitative failure review."
            ),
            "",
            "## Most-Missed Objects",
            "",
            "| Object | Unique FN | Raw FN | Unique positives | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["top_false_negative_objects"]:
        lines.append(
            f"| {item['object']} | "
            f"{item['unique_false_negative_queries']} | "
            f"{item['false_negatives']} | "
            f"{item['unique_positive_queries']} | {item['recall']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Most-Frequent False Positives",
            "",
            "| Object | FP | Negative queries | False-positive rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in summary["top_false_positive_objects"]:
        lines.append(
            f"| {item['object']} | {item['false_positives']} | "
            f"{item['negative_questions']} | "
            f"{item['false_positive_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Hardest Images",
            "",
            "| Image ID | Unique errors | Raw errors | FN | FP | Objects |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in summary["hardest_images"][:10]:
        lines.append(
            f"| {item['image_id']} | {item['unique_error_queries']} | "
            f"{item['raw_errors']} | {item['false_negatives']} | "
            f"{item['false_positives']} | "
            f"{', '.join(item['error_objects'])} |"
        )

    if visual_paths:
        lines.extend(["", "## Representative Cases", ""])
        for label, path in visual_paths.items():
            lines.extend(
                [
                    f"### {label.replace('_', ' ').title()}",
                    "",
                    f"![{label}]({path})",
                    "",
                ]
            )

    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            (
                "- POPE provides image-level Yes/No supervision, not boxes or "
                "masks. This report attributes errors to queried objects and "
                "sampling strategies; it does not prove whether an error was "
                "caused by scale, occlusion, language bias, or localization."
            ),
            (
                "- POPE negatives are derived from COCO annotations. A false "
                "positive is a benchmark disagreement, not automatically a "
                "proven hallucination: incomplete annotations, synonyms, and "
                "ontology boundaries can produce visually ambiguous cases."
            ),
            (
                "- Object-level rates should be read together with support. "
                "A high error rate on a rare object is less stable than a high "
                "error count on a well-supported object."
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
