"""Paired comparison for Hard-Dev relation prompt variants."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Iterable, Mapping

from .evaluation import aggregate_metrics


def exact_mcnemar_p_value(baseline_only: int, candidate_only: int) -> float:
    """Return the two-sided exact binomial McNemar p-value."""
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    lower = min(baseline_only, candidate_only)
    probability = 2 * sum(
        math.comb(discordant, index) for index in range(lower + 1)
    ) / (2**discordant)
    return round(min(1.0, probability), 6)


def _relation_records(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    selected = [
        dict(item) for item in records if item["task_type"] == "spatial_relation"
    ]
    by_id = {str(item["id"]): item for item in selected}
    if len(by_id) != len(selected):
        raise ValueError("Relation predictions contain duplicate IDs.")
    return by_id


def compare_relation_prompts(
    baseline_records: Iterable[Mapping[str, Any]],
    candidate_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare relation predictions with exact paired transitions by source."""
    baseline = _relation_records(baseline_records)
    candidate = _relation_records(candidate_records)
    if set(baseline) != set(candidate):
        raise RuntimeError("Baseline and candidate relation IDs differ.")
    if {str(item.get("split")) for item in candidate.values()} != {"dev"}:
        raise RuntimeError("Relation prompt comparison is Dev-only.")
    for question_id in baseline:
        if (
            baseline[question_id].get("gt_answer")
            != candidate[question_id].get("gt_answer")
            or baseline[question_id].get("source")
            != candidate[question_id].get("source")
        ):
            raise RuntimeError(f"Paired ground truth differs for {question_id}.")

    ordered_ids = sorted(baseline)
    baseline_list = [baseline[question_id] for question_id in ordered_ids]
    candidate_list = [candidate[question_id] for question_id in ordered_ids]
    baseline_metrics = aggregate_metrics(
        baseline_list, expected_samples=len(ordered_ids)
    )
    candidate_metrics = aggregate_metrics(
        candidate_list, expected_samples=len(ordered_ids)
    )

    def paired_summary(ids: list[str]) -> dict[str, Any]:
        both_correct = sum(
            baseline[item]["evaluation"]["is_correct"]
            and candidate[item]["evaluation"]["is_correct"]
            for item in ids
        )
        baseline_only = sum(
            baseline[item]["evaluation"]["is_correct"]
            and not candidate[item]["evaluation"]["is_correct"]
            for item in ids
        )
        candidate_only = sum(
            not baseline[item]["evaluation"]["is_correct"]
            and candidate[item]["evaluation"]["is_correct"]
            for item in ids
        )
        both_wrong = len(ids) - both_correct - baseline_only - candidate_only
        invalid_to_valid = sum(
            not baseline[item]["evaluation"]["parse_valid"]
            and candidate[item]["evaluation"]["parse_valid"]
            for item in ids
        )
        return {
            "count": len(ids),
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
            "both_wrong": both_wrong,
            "net_correct": candidate_only - baseline_only,
            "invalid_to_valid": invalid_to_valid,
            "mcnemar_exact_p_value": exact_mcnemar_p_value(
                baseline_only, candidate_only
            ),
        }

    sources = sorted({str(item["source"]) for item in candidate.values()})
    source_comparisons = {}
    for source in sources:
        ids = [item for item in ordered_ids if candidate[item]["source"] == source]
        baseline_relation = baseline_metrics["sources"][source]["tasks"][
            "spatial_relation"
        ]
        candidate_relation = candidate_metrics["sources"][source]["tasks"][
            "spatial_relation"
        ]
        source_comparisons[source] = {
            "baseline": baseline_relation,
            "candidate": candidate_relation,
            "delta": {
                "exact_accuracy": round(
                    candidate_relation["exact_accuracy"]
                    - baseline_relation["exact_accuracy"],
                    6,
                ),
                "balanced_accuracy": round(
                    candidate_relation["balanced_accuracy"]
                    - baseline_relation["balanced_accuracy"],
                    6,
                ),
                "parse_valid_rate": round(
                    candidate_relation["parse_valid_rate"]
                    - baseline_relation["parse_valid_rate"],
                    6,
                ),
            },
            "paired": paired_summary(ids),
        }

    baseline_relation = baseline_metrics["tasks"]["spatial_relation"]
    candidate_relation = candidate_metrics["tasks"]["spatial_relation"]
    baseline_hit_max = sum(
        bool(item.get("hit_max_new_tokens")) for item in baseline_list
    )
    candidate_hit_max = sum(
        bool(item.get("hit_max_new_tokens")) for item in candidate_list
    )
    baseline_latencies = [float(item["latency_seconds"]) for item in baseline_list]
    candidate_latencies = [float(item["latency_seconds"]) for item in candidate_list]
    candidate_tokens = [
        int(item["generated_tokens"])
        for item in candidate_list
        if "generated_tokens" in item
    ]
    open_images = source_comparisons.get("open_images_v7_validation", {})
    visual_genome = source_comparisons.get("visual_genome_v1_4", {})
    open_images_accept = bool(
        open_images
        and open_images["delta"]["exact_accuracy"] > 0
        and open_images["delta"]["balanced_accuracy"] > 0
        and open_images["paired"]["mcnemar_exact_p_value"] < 0.05
    )
    visual_genome_accept = bool(
        visual_genome
        and visual_genome["delta"]["exact_accuracy"] >= 0
        and visual_genome["delta"]["balanced_accuracy"] > 0
        and visual_genome["paired"]["mcnemar_exact_p_value"] < 0.05
    )
    return {
        "status": "completed",
        "coverage": {"paired_relation_questions": len(ordered_ids), "split": "dev"},
        "overall": {
            "baseline": baseline_relation,
            "candidate": candidate_relation,
            "delta": {
                "exact_accuracy": round(
                    candidate_relation["exact_accuracy"]
                    - baseline_relation["exact_accuracy"],
                    6,
                ),
                "balanced_accuracy": round(
                    candidate_relation["balanced_accuracy"]
                    - baseline_relation["balanced_accuracy"],
                    6,
                ),
                "parse_valid_rate": round(
                    candidate_relation["parse_valid_rate"]
                    - baseline_relation["parse_valid_rate"],
                    6,
                ),
            },
            "paired": paired_summary(ordered_ids),
        },
        "sources": source_comparisons,
        "efficiency": {
            "baseline_hit_max_new_tokens": baseline_hit_max,
            "candidate_hit_max_new_tokens": candidate_hit_max,
            "baseline_mean_latency_seconds": round(
                statistics.fmean(baseline_latencies), 6
            ),
            "candidate_mean_latency_seconds": round(
                statistics.fmean(candidate_latencies), 6
            ),
            "candidate_generated_tokens_mean": round(
                statistics.fmean(candidate_tokens), 6
            )
            if candidate_tokens
            else None,
            "candidate_generated_tokens_max": max(candidate_tokens)
            if candidate_tokens
            else None,
        },
        "candidate_prediction_distribution": dict(
            sorted(
                Counter(
                    str(item["evaluation"].get("parsed_prediction") or "invalid")
                    for item in candidate_list
                ).items()
            )
        ),
        "decision": {
            "global_lock": False,
            "open_images_v2_accept": open_images_accept,
            "visual_genome_v2_accept": visual_genome_accept,
            "recommendation": (
                "Use prompt v2 as the Open Images relation candidate. Retain v1 "
                "for Visual Genome until a separately validated prompt or policy "
                "is selected; do not generate a Test variant yet."
            ),
        },
    }


def render_relation_prompt_comparison(summary: Mapping[str, Any]) -> str:
    """Render the paired comparison and source-aware decision."""
    overall = summary["overall"]
    lines = [
        "# Hard-Dev Relation Prompt V1 vs V2",
        "",
        f"Paired questions: {summary['coverage']['paired_relation_questions']}",
        "",
        "## Overall",
        "",
        "| Metric | V1 | V2 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in ("exact_accuracy", "balanced_accuracy", "parse_valid_rate"):
        lines.append(
            f"| {key} | {overall['baseline'][key]:.4f} | "
            f"{overall['candidate'][key]:.4f} | {overall['delta'][key]:+.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "Paired transitions: "
                f"V1-only={overall['paired']['baseline_only_correct']}, "
                f"V2-only={overall['paired']['candidate_only_correct']}, "
                f"McNemar p={overall['paired']['mcnemar_exact_p_value']:.6f}."
            ),
            "",
            "## By Source",
            "",
            "| Source | V1 acc. | V2 acc. | Delta | V1 bal. | V2 bal. | McNemar p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source, item in summary["sources"].items():
        lines.append(
            f"| {source} | {item['baseline']['exact_accuracy']:.4f} | "
            f"{item['candidate']['exact_accuracy']:.4f} | "
            f"{item['delta']['exact_accuracy']:+.4f} | "
            f"{item['baseline']['balanced_accuracy']:.4f} | "
            f"{item['candidate']['balanced_accuracy']:.4f} | "
            f"{item['paired']['mcnemar_exact_p_value']:.6f} |"
        )
    efficiency = summary["efficiency"]
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            f"- Token-limit hits: {efficiency['baseline_hit_max_new_tokens']} -> "
            f"{efficiency['candidate_hit_max_new_tokens']}",
            f"- Mean latency: {efficiency['baseline_mean_latency_seconds']:.4f}s -> "
            f"{efficiency['candidate_mean_latency_seconds']:.4f}s",
            f"- V2 generated tokens, mean/max: "
            f"{efficiency['candidate_generated_tokens_mean']:.2f} / "
            f"{efficiency['candidate_generated_tokens_max']}",
            "",
            "## Decision",
            "",
            f"- Global lock: `{summary['decision']['global_lock']}`",
            f"- Open Images v2 accepted: `{summary['decision']['open_images_v2_accept']}`",
            f"- Visual Genome v2 accepted: `{summary['decision']['visual_genome_v2_accept']}`",
            f"- Recommendation: {summary['decision']['recommendation']}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
