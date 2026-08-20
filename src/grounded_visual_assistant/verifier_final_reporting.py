"""Deterministic reporting for the frozen Verifier Dev decision."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .evaluation import normalize_text, parse_yes_no


VERIFIER_FINAL_FREEZE_PROTOCOL = "verifier_dev_final_freeze_v1"
STAGE38_PROTOCOL = "verifier_dev_offline_ablation_v1"
STAGE39_PROTOCOL = "verifier_dev_contrastive_review_v3"


def _policy_row(
    metrics: Mapping[str, Any], policy_id: str
) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in metrics.get("policy_table", [])
        if item.get("policy_id") == policy_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Stage 38 row for {policy_id!r}, "
            f"found {len(matches)}."
        )
    return matches[0]


def validate_final_freeze(
    stage38_metrics: Mapping[str, Any],
    stage38_policy: Mapping[str, Any],
    stage39_metrics: Mapping[str, Any],
    stage39_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Require both Dev selection stages to reject answer rewriting."""
    if stage38_metrics.get("protocol") != STAGE38_PROTOCOL:
        raise ValueError("Stage 38 metrics protocol mismatch.")
    if stage38_metrics.get("status") != "completed":
        raise RuntimeError("Stage 38 metrics are incomplete.")
    selection38 = stage38_metrics.get("selection") or {}
    if selection38.get("decision") != (
        "retain_baseline_no_eligible_verifier"
    ):
        raise RuntimeError("Stage 38 did not retain the baseline.")
    if selection38.get("selected_policy_id") != "baseline":
        raise RuntimeError("Stage 38 selected a non-baseline policy.")
    if selection38.get("eligible_policy_ids"):
        raise RuntimeError("Stage 38 still contains an eligible verifier.")
    if stage38_policy.get("held_out_evaluation_pending") is not False:
        raise RuntimeError("Stage 38 still requests held-out evaluation.")
    if stage38_policy.get("selected_policy", {}).get(
        "policy_id"
    ) != "baseline":
        raise RuntimeError("Stage 38 policy artifact is not the baseline.")

    if stage39_metrics.get("protocol") != STAGE39_PROTOCOL:
        raise ValueError("Stage 39 metrics protocol mismatch.")
    if stage39_metrics.get("status") != "completed":
        raise RuntimeError("Stage 39 metrics are incomplete.")
    evaluation39 = stage39_metrics.get("evaluation") or {}
    selection39 = evaluation39.get("selection") or {}
    if selection39.get("decision") != "reject_v3_on_dev":
        raise RuntimeError("Stage 39 did not reject V3 on Dev.")
    if selection39.get("eligible") is not False:
        raise RuntimeError("Stage 39 still marks V3 as eligible.")
    if stage39_decision.get("decision") != "reject_v3_on_dev":
        raise RuntimeError("Stage 39 decision artifact disagrees.")
    if stage39_decision.get("held_out_evaluation_pending") is not False:
        raise RuntimeError("Stage 39 still requests held-out evaluation.")

    baseline38 = stage38_metrics.get("baseline")
    baseline39 = evaluation39.get("baseline")
    if baseline38 != baseline39:
        raise RuntimeError("Stage 38 and Stage 39 baseline metrics differ.")
    return {
        "decision": "retain_qwen_baseline_disable_answer_rewrite",
        "stage38_decision": selection38["decision"],
        "stage39_decision": selection39["decision"],
        "baseline_metrics": baseline38,
        "held_out_verifier_run_permitted": False,
    }


def build_variant_summary(
    stage38_metrics: Mapping[str, Any],
    stage39_metrics: Mapping[str, Any],
    *,
    v1_policy_id: str,
    v2_rescue_policy_id: str,
    v2_noop_policy_id: str,
) -> list[dict[str, Any]]:
    """Build the compact, interview-facing V1/V2/V3 comparison."""
    configured = [
        ("baseline", "Frozen Qwen baseline", "stage35"),
        (v1_policy_id, "V1 grounding-only", "stage38"),
        (v2_rescue_policy_id, "V2 semantic rescue", "stage38"),
        (v2_noop_policy_id, "V2 conservative no-op", "stage38"),
    ]
    rows = []
    for policy_id, label, stage in configured:
        item = _policy_row(stage38_metrics, policy_id)
        rows.append(
            {
                "variant_id": policy_id,
                "label": label,
                "stage": stage,
                "module": item["module"],
                "score_threshold": item.get("score_threshold"),
                "accuracy": item["accuracy"],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1"],
                "changed_answers": item["changed_answers"],
                "beneficial": item["beneficial"],
                "harmful": item["harmful"],
                "net_correct": item["net_correct"],
                "grounding_queries": (
                    0
                    if policy_id == "baseline"
                    else stage38_metrics["coverage"]["grounding_queries"]
                ),
                "model_reviews": item["semantic_reviews"],
                "incremental_latency_seconds": item[
                    "incremental_latency_seconds"
                ],
                "eligible": bool(item.get("eligible", False)),
                "decision": (
                    "retained"
                    if policy_id == "baseline"
                    else "rejected"
                ),
                "rejection_reasons": item.get(
                    "rejection_reasons", ""
                ),
            }
        )

    evaluation = stage39_metrics["evaluation"]
    runtime = stage39_metrics["runtime_projection"]
    v3_metrics = evaluation["v3"]
    corrections = evaluation["corrections"]
    rows.append(
        {
            "variant_id": "v3-contrastive-category-review",
            "label": "V3 contrastive category review",
            "stage": "stage39",
            "module": "semantic_contrastive",
            "score_threshold": 0.3,
            "accuracy": v3_metrics["accuracy"],
            "precision": v3_metrics["precision"],
            "recall": v3_metrics["recall"],
            "f1": v3_metrics["f1"],
            "changed_answers": corrections["changed_answers"],
            "beneficial": corrections["beneficial"],
            "harmful": corrections["harmful"],
            "net_correct": corrections["net_correct"],
            "grounding_queries": stage38_metrics["coverage"][
                "grounding_queries"
            ],
            "model_reviews": (
                stage39_metrics["coverage"][
                    "v2_top1_candidate_queries"
                ]
                + stage39_metrics["coverage"]["completed_candidates"]
            ),
            "incremental_latency_seconds": runtime[
                "incremental_latency_seconds"
            ],
            "eligible": bool(evaluation["selection"]["eligible"]),
            "decision": "rejected",
            "rejection_reasons": ";".join(
                evaluation["selection"]["rejection_reasons"]
            ),
        }
    )
    return rows


def _taxonomy(
    baseline: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    semantic: list[Mapping[str, Any]],
    v2_correction: Mapping[str, Any] | None,
    v3_prediction: Mapping[str, Any] | None,
) -> tuple[str, str]:
    gt = normalize_text(str(baseline["gt_answer"]))
    prediction = parse_yes_no(str(baseline["prediction"]))
    annotations = (
        list((evidence.get("grounding") or {}).get("annotations", []))
        if evidence
        else []
    )
    semantic_yes = any(
        normalize_text(str(item.get("parsed_answer", ""))) == "yes"
        for item in semantic
    )
    selected_label = (
        normalize_text(
            str(v3_prediction.get("contrastive_selected_label") or "")
        )
        if v3_prediction
        else ""
    )
    target = normalize_text(str(baseline["object"]))

    if gt == "no" and prediction == "yes":
        return (
            "baseline_false_positive_not_rechecked",
            "The asymmetric verifier only rechecks baseline No answers, so "
            "this baseline false positive remains outside its recall path.",
        )
    if gt == "yes" and prediction == "no":
        if not annotations:
            return (
                "grounding_recall_miss",
                "Grounding produced no candidate, so later semantic stages "
                "had no evidence to review.",
            )
        if v2_correction and v2_correction.get("correction") == "beneficial":
            if selected_label and selected_label != target:
                return (
                    "contrastive_false_rejection",
                    "V2 rescued the false negative, but V3 selected a "
                    "non-target label and removed the rescue.",
                )
        if not semantic_yes:
            return (
                "semantic_false_rejection",
                "Grounding found a candidate, but the semantic crop review "
                "rejected it, leaving the false negative unresolved.",
            )
        return (
            "unresolved_false_negative",
            "The cascade did not retain a valid target confirmation.",
        )
    if v2_correction and v2_correction.get("correction") == "harmful":
        if selected_label == target:
            return (
                "category_ambiguity_false_accept",
                "Both V2 and V3 accepted the target label for an absent "
                "hard-negative category, creating a false positive.",
            )
        if selected_label and selected_label != target:
            return (
                "cross_category_confusion_caught_by_v3",
                "V2 confused a related category with the target; V3 named "
                "a non-target class and blocked that promotion.",
            )
        return (
            "semantic_false_accept",
            "The semantic gate promoted a correct baseline negative.",
        )
    return "diagnostic_case", "Included for deterministic error auditing."


def build_failure_analysis(
    baseline_records: Iterable[Mapping[str, Any]],
    evidence_records: Iterable[Mapping[str, Any]],
    semantic_reviews: Iterable[Mapping[str, Any]],
    v2_corrections: Iterable[Mapping[str, Any]],
    v3_reviews: Iterable[Mapping[str, Any]],
    v3_predictions: Iterable[Mapping[str, Any]],
    *,
    v2_policy_id: str,
) -> dict[str, Any]:
    """Trace baseline errors and representative verifier regressions."""
    baseline = [dict(item) for item in baseline_records]
    evidence_by_id = {
        str(item["baseline_id"]): dict(item) for item in evidence_records
    }
    semantic_by_id: dict[str, list[dict[str, Any]]] = {}
    for raw_item in semantic_reviews:
        item = dict(raw_item)
        semantic_by_id.setdefault(str(item["baseline_id"]), []).append(item)
    for items in semantic_by_id.values():
        items.sort(
            key=lambda item: (
                -float(item["grounding_score"]),
                int(item["annotation_index"]),
            )
        )
    v2_by_id = {
        str(item["id"]): dict(item)
        for item in v2_corrections
        if item.get("policy_id") == v2_policy_id
    }
    v3_reviews_by_id = {
        str(item["baseline_id"]): dict(item) for item in v3_reviews
    }
    v3_predictions_by_id = {
        str(item["id"]): dict(item) for item in v3_predictions
    }
    interesting_ids = {
        str(item["id"])
        for item in baseline
        if not bool((item.get("evaluation") or {}).get("is_correct"))
    }
    interesting_ids.update(v2_by_id)

    cases = []
    for item in baseline:
        baseline_id = str(item["id"])
        if baseline_id not in interesting_ids:
            continue
        evidence = evidence_by_id.get(baseline_id)
        semantic = semantic_by_id.get(baseline_id, [])
        v2 = v2_by_id.get(baseline_id)
        v3_review = v3_reviews_by_id.get(baseline_id)
        v3_prediction = v3_predictions_by_id.get(baseline_id)
        annotations = (
            list((evidence.get("grounding") or {}).get("annotations", []))
            if evidence
            else []
        )
        taxonomy, explanation = _taxonomy(
            item, evidence, semantic, v2, v3_prediction
        )
        cases.append(
            {
                "id": baseline_id,
                "scope": (
                    "baseline_error"
                    if not item["evaluation"]["is_correct"]
                    else "verifier_regression_risk"
                ),
                "pair_role": item["pair_role"],
                "image_id": item["image_id"],
                "object": item["object"],
                "gt_answer": item["gt_answer"],
                "baseline_prediction": parse_yes_no(
                    str(item["prediction"])
                ),
                "baseline_correct": bool(
                    item["evaluation"]["is_correct"]
                ),
                "grounding_candidate_count": len(annotations),
                "max_grounding_score": (
                    round(
                        max(float(value["score"]) for value in annotations),
                        6,
                    )
                    if annotations
                    else None
                ),
                "semantic_reviews": [
                    {
                        "candidate_key": review["candidate_key"],
                        "grounding_score": review["grounding_score"],
                        "answer": review.get("parsed_answer"),
                    }
                    for review in semantic
                ],
                "v2_prediction": (
                    v2["prediction"]
                    if v2
                    else parse_yes_no(str(item["prediction"]))
                ),
                "v2_correction": (
                    v2["correction"] if v2 else "unchanged"
                ),
                "v3_selected_label": (
                    v3_review.get("selected_label")
                    if v3_review
                    else None
                ),
                "v3_prediction": (
                    v3_prediction.get("prediction")
                    if v3_prediction
                    else parse_yes_no(str(item["prediction"]))
                ),
                "final_frozen_prediction": parse_yes_no(
                    str(item["prediction"])
                ),
                "taxonomy": taxonomy,
                "explanation": explanation,
            }
        )

    baseline_errors = [
        item for item in baseline if not item["evaluation"]["is_correct"]
    ]
    v2_changes = list(v2_by_id.values())
    v3_changes = [
        item for item in v3_predictions_by_id.values() if item["changed"]
    ]
    return {
        "summary": {
            "baseline_errors": len(baseline_errors),
            "baseline_false_positives": sum(
                normalize_text(str(item["gt_answer"])) == "no"
                for item in baseline_errors
            ),
            "baseline_false_negatives": sum(
                normalize_text(str(item["gt_answer"])) == "yes"
                for item in baseline_errors
            ),
            "v2_changed_answers": len(v2_changes),
            "v2_beneficial": sum(
                item["correction"] == "beneficial" for item in v2_changes
            ),
            "v2_harmful": sum(
                item["correction"] == "harmful" for item in v2_changes
            ),
            "v3_changed_answers": len(v3_changes),
            "v3_beneficial": sum(
                item["correction"] == "beneficial" for item in v3_changes
            ),
            "v3_harmful": sum(
                item["correction"] == "harmful" for item in v3_changes
            ),
            "taxonomy_counts": dict(
                sorted(Counter(item["taxonomy"] for item in cases).items())
            ),
        },
        "cases": cases,
        "conclusions": [
            "Detector confidence alone cannot separate beneficial rescues "
            "from related-category false positives.",
            "Target-biased binary crop review improves precision over V1 "
            "but still creates more errors than it fixes.",
            "Contrastive review catches the truck confusion but rejects the "
            "only V2 book rescue and still accepts chair.",
            "The evidence stack remains useful for localization and failure "
            "auditing, but not for answer rewriting under the frozen gates.",
        ],
    }


def markdown_report(
    final_policy: Mapping[str, Any],
    variants: Iterable[Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> str:
    """Render a concise, auditable report suitable for interview review."""
    rows = [dict(item) for item in variants]
    summary = analysis["summary"]
    lines = [
        "# Verifier Dev110 Final Freeze",
        "",
        f"- Final decision: `{final_policy['decision']}`",
        "- Answer rewriting: disabled",
        "- Grounded evidence: retained for localization and auditing only",
        "- Held-out verifier evaluation: not permitted after Dev rejection",
        "- Model inference in this stage: none",
        "",
        "## Controlled Comparison",
        "",
        "| Variant | Accuracy | F1 | Good | Harm | Net | Reviews | Extra latency | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item['label']} | {item['accuracy']:.6f} | "
            f"{item['f1']:.6f} | {item['beneficial']} | "
            f"{item['harmful']} | {item['net_correct']} | "
            f"{item['model_reviews']} | "
            f"{item['incremental_latency_seconds']:.4f}s | "
            f"{item['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Failure Audit",
            "",
            f"The frozen baseline has {summary['baseline_errors']} errors: "
            f"{summary['baseline_false_negatives']} false negatives and "
            f"{summary['baseline_false_positives']} false positive. The "
            "representative score-0.30 V2 path makes one beneficial and two "
            "harmful changes; V3 makes zero beneficial and one harmful "
            "change.",
            "",
            "| Target | Scope | Base | V2 | V3 label | Final | Taxonomy |",
            "|---|---|---:|---:|---|---:|---|",
        ]
    )
    for item in analysis["cases"]:
        lines.append(
            f"| `{item['object']}` | {item['scope']} | "
            f"{item['baseline_prediction']} | {item['v2_prediction']} | "
            f"{item['v3_selected_label'] or '-'} | "
            f"{item['final_frozen_prediction']} | "
            f"`{item['taxonomy']}` |"
        )
    lines.extend(
        [
            "",
            "## Engineering Conclusion",
            "",
            "All answer-rewriting variants fail the pre-registered Dev gates: "
            "strictly higher accuracy, non-decreasing F1, and positive net "
            "corrections. The system therefore keeps the strongest measured "
            "policy, the frozen Qwen baseline, and exposes Grounding DINO plus "
            "SAM 2.1 output as inspectable visual evidence instead of silently "
            "overwriting answers.",
            "",
            "## Interview Claim",
            "",
            "Designed and audited three evidence-based answer-verification "
            "strategies on an isolated 110-question development protocol; "
            "enforced immutable source hashes and pre-registered acceptance "
            "gates, then rejected all three when none improved the frozen "
            "baseline without regressions.",
            "",
            "Do not claim that the verifier reduced hallucination: the Dev "
            "evidence does not support that statement.",
            "",
        ]
    )
    return "\n".join(lines)
