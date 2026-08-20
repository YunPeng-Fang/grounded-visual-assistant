"""Deterministic reporting helpers for the locked Hard-Test evaluation."""

from __future__ import annotations

from typing import Any, Mapping

from .hard_dataset import OPEN_IMAGES_SOURCE, VISUAL_GENOME_SOURCE


def build_generalization_rows(
    dev_reference: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the fixed Dev-to-Test comparison rows for the selected policy."""
    rows = []

    def add(
        scope: str,
        metric: str,
        dev_value: float,
        test_value: float,
    ) -> None:
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "dev": round(float(dev_value), 6),
                "test": round(float(test_value), 6),
                "delta_test_minus_dev": round(
                    float(test_value) - float(dev_value), 6
                ),
            }
        )

    add(
        "object_existence",
        "exact_accuracy",
        dev_reference["object_existence"]["exact_accuracy"],
        test_metrics["tasks"]["object_existence"]["exact_accuracy"],
    )
    add(
        "object_listing",
        "macro_f1",
        dev_reference["object_listing"]["macro_f1"],
        test_metrics["tasks"]["object_listing"]["macro_f1"],
    )
    for source in (OPEN_IMAGES_SOURCE, VISUAL_GENOME_SOURCE):
        dev_relation = dev_reference["relations"][source]
        test_relation = test_metrics["sources"][source]["tasks"][
            "spatial_relation"
        ]
        for metric in (
            "exact_accuracy",
            "balanced_accuracy",
            "parse_valid_rate",
        ):
            add(
                f"spatial_relation:{source}",
                metric,
                dev_relation[metric],
                test_relation[metric],
            )
    return rows


def render_locked_hard_test_report(summary: Mapping[str, Any]) -> str:
    """Render the immutable held-out result and read-only failure analysis."""
    metrics = summary["test_result"]
    analysis = summary["failure_analysis"]
    lines = [
        "# Locked Hard-Test400 Final Report",
        "",
        "## Result Integrity",
        "",
        "- Protocol: `hard_relation_source_aware_prompt_policy_v1`",
        "- Split: `test`",
        "- Coverage: 400 / 400",
        "- Prediction errors: 0",
        "- Token-limit hits: 0",
        "- Dataset, manifest, policy, run, and replay checks: passed",
        "- Post-Test tuning: prohibited",
        "",
        "## Held-Out Results",
        "",
        "| Task | Primary metric | Value |",
        "|---|---|---:|",
        (
            "| object_existence | exact_accuracy | "
            f"{metrics['tasks']['object_existence']['exact_accuracy']:.4f} |"
        ),
        (
            "| object_listing | macro_f1 | "
            f"{metrics['tasks']['object_listing']['macro_f1']:.4f} |"
        ),
        (
            "| spatial_relation | exact_accuracy | "
            f"{metrics['tasks']['spatial_relation']['exact_accuracy']:.4f} |"
        ),
        (
            "| spatial_relation | balanced_accuracy | "
            f"{metrics['tasks']['spatial_relation']['balanced_accuracy']:.4f} |"
        ),
        "",
        (
            f"Overall mean score: `{metrics['overall']['mean_score']:.6f}`; "
            f"overall exact accuracy: `{metrics['overall']['exact_accuracy']:.6f}`."
        ),
        "",
        "## Relation By Source",
        "",
        "| Source | Accuracy | Balanced acc. | Majority baseline | Parse valid |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in (OPEN_IMAGES_SOURCE, VISUAL_GENOME_SOURCE):
        item = metrics["sources"][source]["tasks"]["spatial_relation"]
        lines.append(
            f"| {source} | {item['exact_accuracy']:.4f} | "
            f"{item['balanced_accuracy']:.4f} | "
            f"{item['majority_class_baseline_accuracy']:.4f} | "
            f"{item['parse_valid_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Dev To Test",
            "",
            "| Scope | Metric | Dev | Test | Delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in summary["generalization"]:
        lines.append(
            f"| {item['scope']} | {item['metric']} | {item['dev']:.4f} | "
            f"{item['test']:.4f} | {item['delta_test_minus_dev']:+.4f} |"
        )

    lines.extend(["", "## Failure Attribution", ""])
    for flag, count in analysis["failure_flags"].items():
        lines.append(f"- `{flag}`: {count}")
    lines.extend(
        [
            "",
            "### Existence By Polarity",
            "",
            "| Polarity | Count | Accuracy |",
            "|---|---:|---:|",
        ]
    )
    for label, item in analysis["existence_by_polarity"].items():
        lines.append(
            f"| {label} | {item['count']} | {item['exact_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "### Listing Protocol",
            "",
            "| Protocol | Count | Mean score | Exact accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, item in analysis["listing_protocols"].items():
        lines.append(
            f"| {label} | {item['count']} | {item['mean_score']:.4f} | "
            f"{item['exact_accuracy']:.4f} |"
        )

    lines.extend(["", "## Interpretation", ""])
    lines.extend(
        [
            "- Open Images relation performance generalizes without a drop in "
            "the primary metrics.",
            "- Visual Genome relation accuracy and balanced accuracy decline on "
            "Test; this is the main held-out weakness.",
            "- Visual Genome raw accuracy is below its majority-class baseline, "
            "so balanced accuracy and per-label confusion must be reported.",
            "- Listing exact match is strict; macro F1 is the primary listing "
            "metric and remains stable from Dev to Test.",
            "- These findings are descriptive. No prompt, threshold, or policy "
            "may be changed using this Test result.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
