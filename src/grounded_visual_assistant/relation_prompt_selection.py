"""Select and report a source-aware Hard-Dev relation prompt policy."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .hard_dataset import OPEN_IMAGES_SOURCE, VISUAL_GENOME_SOURCE
from .relation_prompt_comparison import compare_relation_prompts


def _source_records(
    records: Iterable[Mapping[str, Any]], source: str
) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in records
        if item.get("task_type") == "spatial_relation"
        and item.get("source") == source
    ]


def _prediction_by_id(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected = {
        str(item["id"]): dict(item)
        for item in records
        if item.get("task_type") == "spatial_relation"
    }
    if not selected:
        raise ValueError("No relation predictions were provided.")
    return selected


def _metric_gate(
    *,
    candidate_metrics: Mapping[str, Any],
    candidate_hit_max: int,
    criteria: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    values = {
        "parse_valid_rate": float(candidate_metrics["parse_valid_rate"]),
        "hit_max_new_tokens": int(candidate_hit_max),
        "balanced_accuracy": float(candidate_metrics["balanced_accuracy"]),
        "exact_accuracy": float(candidate_metrics["exact_accuracy"]),
    }
    checks = {
        "parse_valid_rate": {
            "value": values["parse_valid_rate"],
            "operator": ">=",
            "threshold": float(criteria["parse_valid_rate_min"]),
            "passed": (
                values["parse_valid_rate"]
                >= float(criteria["parse_valid_rate_min"])
            ),
        },
        "hit_max_new_tokens": {
            "value": values["hit_max_new_tokens"],
            "operator": "<=",
            "threshold": int(criteria["hit_max_new_tokens_max"]),
            "passed": (
                values["hit_max_new_tokens"]
                <= int(criteria["hit_max_new_tokens_max"])
            ),
        },
        "balanced_accuracy": {
            "value": values["balanced_accuracy"],
            "operator": ">=",
            "threshold": float(criteria["balanced_accuracy_min"]),
            "passed": (
                values["balanced_accuracy"]
                >= float(criteria["balanced_accuracy_min"])
            ),
        },
        "exact_accuracy": {
            "value": values["exact_accuracy"],
            "operator": ">=",
            "threshold": float(criteria["exact_accuracy_min"]),
            "passed": (
                values["exact_accuracy"]
                >= float(criteria["exact_accuracy_min"])
            ),
        },
    }
    return checks


def build_relation_prompt_selection(
    baseline_records: Iterable[Mapping[str, Any]],
    prompt_v2_records: Iterable[Mapping[str, Any]],
    prompt_v3_records: Iterable[Mapping[str, Any]],
    prompt_v3_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Compare v1/v2/v3 and lock a source-aware Dev-selected policy."""
    baseline_records = [dict(item) for item in baseline_records]
    prompt_v2_records = [dict(item) for item in prompt_v2_records]
    prompt_v3_records = [dict(item) for item in prompt_v3_records]
    criteria = dict(prompt_v3_manifest.get("acceptance_criteria") or {})
    required_criteria = {
        "parse_valid_rate_min",
        "hit_max_new_tokens_max",
        "balanced_accuracy_min",
        "exact_accuracy_min",
    }
    if required_criteria - set(criteria):
        raise ValueError("Prompt-v3 manifest is missing acceptance criteria.")
    if (
        prompt_v3_manifest.get("split") != "dev"
        or prompt_v3_manifest.get("source") != VISUAL_GENOME_SOURCE
        or int(prompt_v3_manifest.get("questions", 0)) != 100
    ):
        raise ValueError("Prompt-v3 manifest is not the frozen VG Dev100 artifact.")

    v1_v2 = compare_relation_prompts(baseline_records, prompt_v2_records)
    v1_vg = _source_records(baseline_records, VISUAL_GENOME_SOURCE)
    v2_vg = _source_records(prompt_v2_records, VISUAL_GENOME_SOURCE)
    v3_vg = _source_records(prompt_v3_records, VISUAL_GENOME_SOURCE)
    if len(v3_vg) != 100 or len(v1_vg) != 100 or len(v2_vg) != 100:
        raise RuntimeError(
            "Visual Genome prompt comparison requires 100 paired records per variant."
        )
    if len(prompt_v3_records) != len(v3_vg):
        raise RuntimeError("Prompt v3 contains non-Visual-Genome relation records.")

    v1_v3 = compare_relation_prompts(v1_vg, v3_vg)
    v2_v3 = compare_relation_prompts(v2_vg, v3_vg)
    open_images = v1_v2["sources"][OPEN_IMAGES_SOURCE]
    open_images_gate = {
        "exact_accuracy_improved": (
            open_images["delta"]["exact_accuracy"] > 0
        ),
        "balanced_accuracy_improved": (
            open_images["delta"]["balanced_accuracy"] > 0
        ),
        "mcnemar_p_below_0_05": (
            open_images["paired"]["mcnemar_exact_p_value"] < 0.05
        ),
    }
    open_images_passed = all(open_images_gate.values())

    v3_metrics = v1_v3["overall"]["candidate"]
    v3_hit_max = int(v1_v3["efficiency"]["candidate_hit_max_new_tokens"])
    visual_genome_gate = _metric_gate(
        candidate_metrics=v3_metrics,
        candidate_hit_max=v3_hit_max,
        criteria=criteria,
    )
    visual_genome_passed = all(
        item["passed"] for item in visual_genome_gate.values()
    )
    all_passed = open_images_passed and visual_genome_passed

    summary = {
        "status": "accepted" if all_passed else "rejected",
        "selection_split": "dev",
        "coverage": {
            "relation_questions": 200,
            "open_images": 100,
            "visual_genome": 100,
        },
        "comparisons": {
            "v1_vs_v2_all_sources": v1_v2,
            "visual_genome_v1_vs_v3": v1_v3,
            "visual_genome_v2_vs_v3": v2_v3,
        },
        "acceptance": {
            "open_images_v2": {
                "passed": open_images_passed,
                "checks": open_images_gate,
            },
            "visual_genome_v3": {
                "passed": visual_genome_passed,
                "checks": visual_genome_gate,
            },
            "source_aware_policy": all_passed,
        },
        "test_status": "not_generated_not_evaluated",
    }

    selected_policy = {
        "protocol": "hard_relation_source_aware_prompt_policy_v1",
        "immutable": True,
        "selected_on_split": "dev",
        "status": "locked" if all_passed else "not_locked",
        "sources": {
            OPEN_IMAGES_SOURCE: {
                "selected_variant": "v2",
                "prompt_version": "relation_center_forced_choice_v2",
                "question_transform": "apply_relation_prompt_v2",
                "selection_passed": open_images_passed,
            },
            VISUAL_GENOME_SOURCE: {
                "selected_variant": "v3",
                "prompt_version": "visual_genome_semantic_forced_choice_v3",
                "question_transform": "apply_visual_genome_relation_prompt_v3",
                "selection_passed": visual_genome_passed,
            },
        },
        "acceptance": summary["acceptance"],
        "test_status": "not_generated_not_evaluated",
    }

    v1_by_id = _prediction_by_id(v1_vg)
    v2_by_id = _prediction_by_id(v2_vg)
    v3_by_id = _prediction_by_id(v3_vg)
    if set(v1_by_id) != set(v2_by_id) or set(v1_by_id) != set(v3_by_id):
        raise RuntimeError("Visual Genome v1/v2/v3 prediction IDs differ.")
    transitions = []
    for question_id in sorted(v1_by_id):
        v1 = v1_by_id[question_id]
        v2 = v2_by_id[question_id]
        v3 = v3_by_id[question_id]
        if not (
            v1.get("gt_answer") == v2.get("gt_answer") == v3.get("gt_answer")
        ):
            raise RuntimeError(f"Ground truth differs for {question_id}.")
        transitions.append(
            {
                "id": question_id,
                "source": VISUAL_GENOME_SOURCE,
                "gt_answer": v1["gt_answer"],
                "v1_prediction": v1["evaluation"].get("parsed_prediction"),
                "v1_correct": bool(v1["evaluation"]["is_correct"]),
                "v2_prediction": v2["evaluation"].get("parsed_prediction"),
                "v2_correct": bool(v2["evaluation"]["is_correct"]),
                "v3_prediction": v3["evaluation"].get("parsed_prediction"),
                "v3_correct": bool(v3["evaluation"]["is_correct"]),
            }
        )
    return summary, selected_policy, transitions


def render_relation_prompt_selection(summary: Mapping[str, Any]) -> str:
    """Render the source-aware prompt selection report."""
    oi = summary["comparisons"]["v1_vs_v2_all_sources"]["sources"][
        OPEN_IMAGES_SOURCE
    ]
    vg13 = summary["comparisons"]["visual_genome_v1_vs_v3"]["overall"]
    vg23 = summary["comparisons"]["visual_genome_v2_vs_v3"]["overall"]
    vg_gate = summary["acceptance"]["visual_genome_v3"]["checks"]
    lines = [
        "# Hard-Dev Source-Aware Relation Prompt Selection",
        "",
        f"Decision: **{summary['status']}**",
        "",
        "## Paired Results",
        "",
        "| Comparison | Baseline acc. | Candidate acc. | Baseline bal. | Candidate bal. | Baseline-only | Candidate-only | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Open Images v1 vs v2 | {oi['baseline']['exact_accuracy']:.4f} | "
            f"{oi['candidate']['exact_accuracy']:.4f} | "
            f"{oi['baseline']['balanced_accuracy']:.4f} | "
            f"{oi['candidate']['balanced_accuracy']:.4f} | "
            f"{oi['paired']['baseline_only_correct']} | "
            f"{oi['paired']['candidate_only_correct']} | "
            f"{oi['paired']['mcnemar_exact_p_value']:.6f} |"
        ),
        (
            f"| Visual Genome v1 vs v3 | "
            f"{vg13['baseline']['exact_accuracy']:.4f} | "
            f"{vg13['candidate']['exact_accuracy']:.4f} | "
            f"{vg13['baseline']['balanced_accuracy']:.4f} | "
            f"{vg13['candidate']['balanced_accuracy']:.4f} | "
            f"{vg13['paired']['baseline_only_correct']} | "
            f"{vg13['paired']['candidate_only_correct']} | "
            f"{vg13['paired']['mcnemar_exact_p_value']:.6f} |"
        ),
        (
            f"| Visual Genome v2 vs v3 | "
            f"{vg23['baseline']['exact_accuracy']:.4f} | "
            f"{vg23['candidate']['exact_accuracy']:.4f} | "
            f"{vg23['baseline']['balanced_accuracy']:.4f} | "
            f"{vg23['candidate']['balanced_accuracy']:.4f} | "
            f"{vg23['paired']['baseline_only_correct']} | "
            f"{vg23['paired']['candidate_only_correct']} | "
            f"{vg23['paired']['mcnemar_exact_p_value']:.6f} |"
        ),
        "",
        "## Frozen V3 Gate",
        "",
        "| Metric | Value | Rule | Threshold | Passed |",
        "|---|---:|:---:|---:|:---:|",
    ]
    for key in (
        "parse_valid_rate",
        "hit_max_new_tokens",
        "balanced_accuracy",
        "exact_accuracy",
    ):
        item = vg_gate[key]
        lines.append(
            f"| {key} | {item['value']:.6f} | {item['operator']} | "
            f"{item['threshold']:.6f} | {item['passed']} |"
        )
    lines.extend(
        [
            "",
            "## Locked Policy",
            "",
            "- Open Images relations: prompt v2, center-based forced choice.",
            "- Visual Genome relations: prompt v3, semantic forced choice.",
            "- Selection data: Hard-Dev only.",
            "- Hard-Test status: not generated and not evaluated.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
