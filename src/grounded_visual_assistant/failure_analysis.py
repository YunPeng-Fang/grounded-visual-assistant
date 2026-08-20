"""Failure attribution for the saved VLM-to-grounding pipeline."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

from .evaluation import CATEGORY_ALIASES, normalize_text


def _pluralize_phrase(value: str) -> str:
    words = value.split()
    if not words:
        return value
    word = words[-1]
    if word.endswith(("s", "x", "z", "ch", "sh")):
        plural = word + "es"
    elif len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        plural = word[:-1] + "ies"
    else:
        plural = word + "s"
    return " ".join([*words[:-1], plural])


def category_surface_forms(category: str) -> set[str]:
    """Return conservative literal forms used only for parser diagnostics."""
    aliases = CATEGORY_ALIASES.get(category, (category,))
    forms = {normalize_text(alias) for alias in aliases}
    forms.update(_pluralize_phrase(form) for form in list(forms))
    return {form for form in forms if form}


def explicitly_mentioned_categories(
    answer: str, categories: Iterable[str]
) -> list[str]:
    """Find target names stated literally in an answer, including plurals."""
    normalized = normalize_text(answer)
    mentioned = []
    for category in categories:
        if any(
            re.search(rf"\b{re.escape(form)}\b", normalized)
            for form in category_surface_forms(str(category))
        ):
            mentioned.append(str(category))
    return sorted(set(mentioned))


def has_non_terminal_ending(answer: str) -> bool:
    """Flag answers that may have stopped at the generation token limit."""
    stripped = answer.rstrip()
    return bool(stripped) and not bool(re.search(r"[.!?\])}]$", stripped))


def analyze_prediction(record: dict[str, Any]) -> dict[str, Any]:
    """Attribute one image's misses and false positives to pipeline stages."""
    target_categories = set(record.get("target_categories", []))
    prompt_categories = set(record.get("prompt_categories", []))
    prompt_evaluation = record.get("prompt_evaluation", {})
    missed_categories = set(
        prompt_evaluation.get("missed_categories", target_categories - prompt_categories)
    )
    off_target_categories = set(
        prompt_evaluation.get(
            "hallucinated_categories", prompt_categories - target_categories
        )
    )
    answer = str(record.get("vlm_prediction", {}).get("answer", ""))
    literal_mentions = set(
        explicitly_mentioned_categories(answer, missed_categories)
    )
    parser_recoverable = missed_categories & literal_mentions
    generation_omissions = missed_categories - parser_recoverable

    target_box_counts = Counter(
        str(item["category"])
        for item in record.get("target_evidence_boxes", [])
    )
    category_metrics = record.get("evaluation", {}).get("categories", {})
    prompted_targets = target_categories & prompt_categories
    prompt_missed_gt_boxes = sum(
        target_box_counts[category] for category in missed_categories
    )
    grounding_missed_prompted_gt_boxes = sum(
        int(category_metrics.get(category, {}).get("fn", 0))
        for category in prompted_targets
    )
    prompted_target_false_positive_boxes = sum(
        int(category_metrics.get(category, {}).get("fp", 0))
        for category in prompted_targets
    )
    off_target_false_positive_boxes = sum(
        int(category_metrics.get(category, {}).get("fp", 0))
        for category in off_target_categories
    )
    grounding = record.get("evaluation", {})
    total_false_positive_boxes = int(grounding.get("fp", 0))
    unattributed_false_positive_boxes = max(
        total_false_positive_boxes
        - prompted_target_false_positive_boxes
        - off_target_false_positive_boxes,
        0,
    )

    flags = []
    if not prompt_categories:
        flags.append("empty_prompt")
    if generation_omissions:
        flags.append("vlm_omission")
    if parser_recoverable:
        flags.append("parser_miss")
    if off_target_categories:
        flags.append("off_target_prompt")
    if grounding_missed_prompted_gt_boxes:
        flags.append("grounding_miss")
    if total_false_positive_boxes:
        flags.append("grounding_false_positive")
    non_terminal_ending = has_non_terminal_ending(answer)
    if non_terminal_ending:
        flags.append("non_terminal_answer")

    severity_score = (
        4 * prompt_missed_gt_boxes
        + 3 * grounding_missed_prompted_gt_boxes
        + total_false_positive_boxes
        + 2 * len(off_target_categories)
        + (5 if not prompt_categories else 0)
    )
    return {
        "id": record["id"],
        "image_id": int(record["image_id"]),
        "image": record.get("image"),
        "target_categories": sorted(target_categories),
        "prompt_categories": sorted(prompt_categories),
        "missed_categories": sorted(missed_categories),
        "off_target_categories": sorted(off_target_categories),
        "parser_recoverable_categories": sorted(parser_recoverable),
        "generation_omitted_categories": sorted(generation_omissions),
        "empty_prompt": not prompt_categories,
        "non_terminal_answer": non_terminal_ending,
        "answer_character_count": len(answer),
        "vlm_answer": answer,
        "prompt_precision": float(prompt_evaluation.get("precision", 0.0)),
        "prompt_recall": float(prompt_evaluation.get("recall", 0.0)),
        "prompt_f1": float(prompt_evaluation.get("f1", 0.0)),
        "grounding_gt_boxes": int(grounding.get("gt_count", 0)),
        "grounding_predicted_boxes": int(grounding.get("prediction_count", 0)),
        "grounding_tp": int(grounding.get("tp", 0)),
        "grounding_fp": total_false_positive_boxes,
        "grounding_fn": int(grounding.get("fn", 0)),
        "grounding_f1": float(grounding.get("f1", 0.0)),
        "prompt_missed_gt_boxes": prompt_missed_gt_boxes,
        "grounding_missed_prompted_gt_boxes": (
            grounding_missed_prompted_gt_boxes
        ),
        "prompted_target_false_positive_boxes": (
            prompted_target_false_positive_boxes
        ),
        "off_target_false_positive_boxes": off_target_false_positive_boxes,
        "unattributed_false_positive_boxes": unattributed_false_positive_boxes,
        "flags": flags,
        "severity_score": severity_score,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def aggregate_failure_analysis(
    prediction_records: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return an experiment summary and severity-ranked image diagnostics."""
    analyses = [analyze_prediction(record) for record in prediction_records]
    if not analyses:
        raise ValueError("No prediction records were provided.")
    ids = [item["id"] for item in analyses]
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction records contain duplicate sample IDs.")

    missed_categories = Counter(
        category for item in analyses for category in item["missed_categories"]
    )
    off_target_categories = Counter(
        category
        for item in analyses
        for category in item["off_target_categories"]
    )
    parser_recoverable = Counter(
        category
        for item in analyses
        for category in item["parser_recoverable_categories"]
    )
    generation_omitted = Counter(
        category
        for item in analyses
        for category in item["generation_omitted_categories"]
    )
    flag_counts = Counter(flag for item in analyses for flag in item["flags"])
    grounding_fn = sum(item["grounding_fn"] for item in analyses)
    prompt_missed_gt_boxes = sum(
        item["prompt_missed_gt_boxes"] for item in analyses
    )
    grounding_missed_prompted = sum(
        item["grounding_missed_prompted_gt_boxes"] for item in analyses
    )
    grounding_fp = sum(item["grounding_fp"] for item in analyses)
    prompted_target_fp = sum(
        item["prompted_target_false_positive_boxes"] for item in analyses
    )
    off_target_fp = sum(
        item["off_target_false_positive_boxes"] for item in analyses
    )

    ranked = sorted(
        analyses,
        key=lambda item: (-item["severity_score"], item["id"]),
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "images": len(analyses),
            "images_with_any_failure": sum(
                bool(item["missed_categories"])
                or bool(item["off_target_categories"])
                or bool(item["grounding_fn"])
                or bool(item["grounding_fp"])
                for item in analyses
            ),
            "empty_prompt_images": sum(item["empty_prompt"] for item in analyses),
            "non_terminal_answer_images": sum(
                item["non_terminal_answer"] for item in analyses
            ),
        },
        "prompt": {
            "mean_precision": _mean(item["prompt_precision"] for item in analyses),
            "mean_recall": _mean(item["prompt_recall"] for item in analyses),
            "mean_f1": _mean(item["prompt_f1"] for item in analyses),
            "missed_category_mentions": sum(missed_categories.values()),
            "off_target_category_mentions": sum(off_target_categories.values()),
            "parser_recoverable_mentions": sum(parser_recoverable.values()),
            "generation_omitted_mentions": sum(generation_omitted.values()),
            "missed_categories": dict(sorted(missed_categories.items())),
            "off_target_categories": dict(sorted(off_target_categories.items())),
            "parser_recoverable_categories": dict(
                sorted(parser_recoverable.items())
            ),
            "generation_omitted_categories": dict(
                sorted(generation_omitted.items())
            ),
        },
        "grounding_attribution": {
            "ground_truth_boxes": sum(
                item["grounding_gt_boxes"] for item in analyses
            ),
            "false_negative_boxes": grounding_fn,
            "false_negatives_from_missing_prompt_categories": (
                prompt_missed_gt_boxes
            ),
            "false_negatives_after_category_was_prompted": (
                grounding_missed_prompted
            ),
            "prompt_stage_share_of_false_negatives": round(
                prompt_missed_gt_boxes / grounding_fn, 6
            )
            if grounding_fn
            else 0.0,
            "false_positive_boxes": grounding_fp,
            "false_positives_for_prompted_target_categories": prompted_target_fp,
            "false_positives_for_off_target_categories": off_target_fp,
            "unattributed_false_positive_boxes": max(
                grounding_fp - prompted_target_fp - off_target_fp, 0
            ),
        },
        "flagged_image_counts": dict(sorted(flag_counts.items())),
        "top_failures": [
            {
                key: item[key]
                for key in (
                    "id",
                    "image_id",
                    "severity_score",
                    "missed_categories",
                    "off_target_categories",
                    "prompt_missed_gt_boxes",
                    "grounding_missed_prompted_gt_boxes",
                    "grounding_fp",
                    "flags",
                )
            }
            for item in ranked[:10]
        ],
    }
    return summary, ranked


def _category_text(values: Iterable[str]) -> str:
    values = list(values)
    return ", ".join(values) if values else "-"


def render_failure_report(
    summary: dict[str, Any],
    analyses: list[dict[str, Any]],
    *,
    predictions_path: str,
) -> str:
    """Render a compact Markdown report with ranked per-image diagnostics."""
    coverage = summary["coverage"]
    prompt = summary["prompt"]
    attribution = summary["grounding_attribution"]
    lines = [
        "# VLM-Prompt Grounding Failure Analysis",
        "",
        f"Predictions: `{predictions_path}`",
        "",
        "## Summary",
        "",
        f"- Images: {coverage['images']}",
        f"- Images with any recorded failure: {coverage['images_with_any_failure']}",
        f"- Empty prompts: {coverage['empty_prompt_images']}",
        (
            "- Non-terminal answer endings: "
            f"{coverage['non_terminal_answer_images']} (truncation heuristic)"
        ),
        (
            "- Prompt macro P/R/F1: "
            f"{prompt['mean_precision']:.4f} / {prompt['mean_recall']:.4f} / "
            f"{prompt['mean_f1']:.4f}"
        ),
        "",
        "## Error Attribution",
        "",
        "| Error source | Count |",
        "|---|---:|",
        (
            "| GT boxes missed because the category was absent from the prompt | "
            f"{attribution['false_negatives_from_missing_prompt_categories']} |"
        ),
        (
            "| GT boxes missed after the category was prompted | "
            f"{attribution['false_negatives_after_category_was_prompted']} |"
        ),
        f"| Total false-negative boxes | {attribution['false_negative_boxes']} |",
        (
            "| FP boxes for prompted target categories | "
            f"{attribution['false_positives_for_prompted_target_categories']} |"
        ),
        (
            "| FP boxes for benchmark off-target categories | "
            f"{attribution['false_positives_for_off_target_categories']} |"
        ),
        f"| Total false-positive boxes | {attribution['false_positive_boxes']} |",
        "",
        (
            "The prompt stage accounts for "
            f"{100 * attribution['prompt_stage_share_of_false_negatives']:.1f}% "
            "of all matched-box false negatives. Off-target means absent from the "
            "benchmark target set; it does not prove the object is visually absent."
        ),
        "",
        "## Prompt Errors",
        "",
        f"- Missed category mentions: {prompt['missed_category_mentions']}",
        f"- Benchmark off-target mentions: {prompt['off_target_category_mentions']}",
        f"- Literal parser-recoverable misses: {prompt['parser_recoverable_mentions']}",
        f"- Generation omissions: {prompt['generation_omitted_mentions']}",
        f"- Missed categories: `{prompt['missed_categories']}`",
        f"- Off-target categories: `{prompt['off_target_categories']}`",
        "",
        "## Ranked Images",
        "",
        "| Rank | Sample | Missed | Off-target | Prompt FN boxes | Grounding FN boxes | FP boxes | Flags |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(analyses, start=1):
        lines.append(
            f"| {rank} | `{item['id']}` | "
            f"{_category_text(item['missed_categories'])} | "
            f"{_category_text(item['off_target_categories'])} | "
            f"{item['prompt_missed_gt_boxes']} | "
            f"{item['grounding_missed_prompted_gt_boxes']} | "
            f"{item['grounding_fp']} | "
            f"{_category_text(item['flags'])} |"
        )

    lines.extend(["", "## Priority Cases", ""])
    for item in analyses[:10]:
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- Target: {_category_text(item['target_categories'])}",
                f"- Prompt: {_category_text(item['prompt_categories'])}",
                f"- Missed: {_category_text(item['missed_categories'])}",
                f"- Off-target: {_category_text(item['off_target_categories'])}",
                (
                    "- Parser-recoverable: "
                    f"{_category_text(item['parser_recoverable_categories'])}"
                ),
                f"- Grounding TP/FP/FN: {item['grounding_tp']} / "
                f"{item['grounding_fp']} / {item['grounding_fn']}",
                "- VLM answer:",
                "",
                "> " + item["vlm_answer"].replace("\n", "\n> "),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
