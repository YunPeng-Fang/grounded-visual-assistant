"""Generate immutable Test80 comparisons, failure analysis, and figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.answer_reporting import (
    aggregate_answer_analysis,
    analyze_answer_record,
)
from grounded_visual_assistant.dataset_splits import load_image_ids
from grounded_visual_assistant.evaluation import aggregate_metrics
from grounded_visual_assistant.evidence_answering import EvidencePolicyConfig
from grounded_visual_assistant.policy_calibration import (
    aggregate_policy_records,
    build_policy_record,
    replay_grounded_policy,
)


TASK_TYPES = ("object_listing", "object_existence", "spatial_relation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize the frozen locked-policy Test80 evaluation."
    )
    parser.add_argument(
        "--test-predictions",
        default=(
            "outputs/eval_answering_v0/"
            "test__locked-task-aware__box-0.30__text-0.30/predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--original-vlm-predictions",
        default="outputs/eval_v0/eval_v0__qwen3-vl-8b-instruct/predictions.jsonl",
    )
    parser.add_argument(
        "--structured-listing-metrics",
        default="outputs/eval_v0/test__qwen3-vl__coco80-json-v1/metrics.json",
    )
    parser.add_argument(
        "--dev-calibration-summary",
        default=(
            "outputs/eval_answering_v0/"
            "dev__evidence-answering__coco80-json-v1__box-0.30__text-0.30/"
            "policy_calibration/summary.json"
        ),
    )
    parser.add_argument(
        "--test-image-ids",
        default="data/eval_v0/splits/test_image_ids.json",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            sample_id = str(record.get("id", ""))
            if not sample_id or sample_id in seen:
                raise ValueError(f"Missing or duplicate ID on {path}:{line_number}.")
            seen.add(sample_id)
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def sha256sum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_image_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_sizes(records: list[dict[str, Any]]) -> dict[int, tuple[int, int]]:
    sizes = {}
    for record in records:
        image_id = int(record["image_id"])
        if image_id in sizes:
            continue
        with Image.open(resolve_image_path(str(record["image"]))) as image:
            sizes[image_id] = image.size
    return sizes


def uniform_original_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    tasks = {}
    for task, item in metrics["tasks"].items():
        task_result = {
            "count": item["count"],
            "forced_mean_score": item["mean_score"],
            "forced_exact_accuracy": item["exact_accuracy"],
            "selective_answered": item["count"],
            "selective_abstained": 0,
            "selective_coverage": 1.0,
            "selective_mean_score": item["mean_score"],
            "selective_exact_accuracy": item["exact_accuracy"],
        }
        if task == "object_listing":
            task_result.update(
                {
                    "forced_macro_precision": item["macro_precision"],
                    "forced_macro_recall": item["macro_recall"],
                    "forced_macro_f1": item["macro_f1"],
                    "selective_macro_precision": item["macro_precision"],
                    "selective_macro_recall": item["macro_recall"],
                    "selective_macro_f1": item["macro_f1"],
                }
            )
        tasks[task] = task_result
    return {
        "overall": {
            "count": metrics["coverage"]["completed"],
            "forced_mean_score": overall["mean_score"],
            "forced_exact_accuracy": overall["exact_accuracy"],
            "selective_answered": metrics["coverage"]["completed"],
            "selective_abstained": 0,
            "selective_coverage": 1.0,
            "selective_mean_score": overall["mean_score"],
            "selective_exact_accuracy": overall["exact_accuracy"],
        },
        "tasks": tasks,
    }


def policy_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for policy_name, metrics in comparison.items():
        item = metrics["overall"]
        rows.append(
            {
                "policy": policy_name,
                "forced_mean_score": item["forced_mean_score"],
                "forced_exact_accuracy": item["forced_exact_accuracy"],
                "selective_coverage": item["selective_coverage"],
                "selective_mean_score": item["selective_mean_score"],
                "selective_exact_accuracy": item["selective_exact_accuracy"],
            }
        )
    return rows


def task_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for policy_name, metrics in comparison.items():
        for task, item in metrics["tasks"].items():
            rows.append(
                {
                    "policy": policy_name,
                    "task_type": task,
                    "forced_mean_score": item["forced_mean_score"],
                    "forced_exact_accuracy": item["forced_exact_accuracy"],
                    "selective_coverage": item["selective_coverage"],
                    "selective_mean_score": item["selective_mean_score"],
                    "selective_exact_accuracy": item[
                        "selective_exact_accuracy"
                    ],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_failure_csv(path: Path, analyses: list[dict[str, Any]]) -> None:
    rows = []
    for item in analyses:
        rows.append(
            {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in item.items()
            }
        )
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_bar_labels(axis, bars) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def generate_figures(
    output_dir: Path,
    comparison: dict[str, Any],
    failure_summary: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    names = ["Original VLM", "Shared evidence", "Locked task-aware"]
    keys = ["original_vlm", "initial_shared_evidence", "locked_task_aware"]
    forced = [comparison[key]["overall"]["forced_exact_accuracy"] for key in keys]
    selective = [
        comparison[key]["overall"]["selective_exact_accuracy"] for key in keys
    ]
    coverage = [comparison[key]["overall"]["selective_coverage"] for key in keys]
    x = np.arange(len(names))
    width = 0.24
    fig, axis = plt.subplots(figsize=(10, 5.6))
    bars1 = axis.bar(x - width, forced, width, label="Forced exact", color="#176B87")
    bars2 = axis.bar(x, selective, width, label="Selective exact", color="#D1495B")
    bars3 = axis.bar(x + width, coverage, width, label="Selective coverage", color="#EDAE49")
    add_bar_labels(axis, bars1)
    add_bar_labels(axis, bars2)
    add_bar_labels(axis, bars3)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Rate")
    axis.set_title("Frozen Test80 Policy Comparison")
    axis.set_xticks(x, names)
    axis.legend(loc="upper left", ncols=3)
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(figure_dir / "policy_comparison.png", dpi=180)
    plt.close(fig)

    locked = comparison["locked_task_aware"]["tasks"]
    tasks = list(TASK_TYPES)
    labels = ["Listing F1", "Existence", "Spatial"]
    forced_task = [
        locked["object_listing"]["forced_mean_score"],
        locked["object_existence"]["forced_exact_accuracy"],
        locked["spatial_relation"]["forced_exact_accuracy"],
    ]
    selective_task = [
        locked["object_listing"]["selective_mean_score"],
        locked["object_existence"]["selective_exact_accuracy"],
        locked["spatial_relation"]["selective_exact_accuracy"],
    ]
    task_coverage = [locked[task]["selective_coverage"] for task in tasks]
    fig, axis = plt.subplots(figsize=(10, 5.6))
    bars1 = axis.bar(x - width, forced_task, width, label="Forced metric", color="#4C78A8")
    bars2 = axis.bar(x, selective_task, width, label="Selective metric", color="#D1495B")
    bars3 = axis.bar(x + width, task_coverage, width, label="Coverage", color="#59A14F")
    add_bar_labels(axis, bars1)
    add_bar_labels(axis, bars2)
    add_bar_labels(axis, bars3)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Rate")
    axis.set_title("Locked Task-Aware Test80 Results")
    axis.set_xticks(x, labels)
    axis.legend(loc="upper left", ncols=3)
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(figure_dir / "task_performance.png", dpi=180)
    plt.close(fig)

    failure_tasks = ["object_listing", "object_existence", "spatial_relation"]
    failure_labels = ["Listing", "Existence", "Spatial"]
    forced_errors = [
        failure_summary["tasks"][task]["forced_errors"] for task in failure_tasks
    ]
    abstentions = [
        failure_summary["tasks"][task]["abstentions"] for task in failure_tasks
    ]
    selective_errors = [
        failure_summary["tasks"][task]["selective_errors"]
        for task in failure_tasks
    ]
    fig, axis = plt.subplots(figsize=(9, 5.6))
    bars1 = axis.bar(x - width, forced_errors, width, label="Forced errors", color="#D1495B")
    bars2 = axis.bar(x, abstentions, width, label="Abstentions", color="#EDAE49")
    bars3 = axis.bar(x + width, selective_errors, width, label="Selective errors", color="#7A5195")
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(int(bar.get_height())),
                ha="center",
                fontsize=9,
            )
    axis.set_ylim(0, max(forced_errors + abstentions + selective_errors) + 8)
    axis.set_ylabel("Questions")
    axis.set_title("Frozen Test80 Error and Abstention Counts")
    axis.set_xticks(x, failure_labels)
    axis.legend(loc="upper left", ncols=3)
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(figure_dir / "failure_breakdown.png", dpi=180)
    plt.close(fig)


def render_report(summary: dict[str, Any], top_failures: list[dict[str, Any]]) -> str:
    comparison = summary["comparison"]
    locked = comparison["locked_task_aware"]
    dev = summary["dev_locked_metrics"]
    failures = summary["failure_summary"]
    latency = summary["locked_metrics"]["latency_seconds"]
    memory = summary["locked_metrics"].get("cuda_memory_gb", {})
    lines = [
        "# Frozen Test80 Final Report",
        "",
        "The task-aware policy was selected on Dev20 and applied once to Test80. "
        "This report performs no threshold selection and does not alter any saved "
        "prediction.",
        "",
        "## Integrity",
        "",
        f"- Coverage: `{summary['integrity']['completed']}/{summary['integrity']['expected']}` questions across `80` images.",
        f"- Error attempts: `{summary['integrity']['error_attempts']}`.",
        f"- Input and policy hashes matched: `{summary['integrity']['all_hashes_match']}`.",
        f"- Metrics recomputation matched: `{summary['integrity']['metrics_recompute_match']}`.",
        "",
        "## Policy Comparison",
        "",
        "| Policy | Forced mean score | Forced exact | Selective coverage | Selective exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("original_vlm", "initial_shared_evidence", "locked_task_aware"):
        item = comparison[name]["overall"]
        lines.append(
            f"| {name} | {item['forced_mean_score']:.6f} | "
            f"{item['forced_exact_accuracy']:.6f} | "
            f"{item['selective_coverage']:.6f} | "
            f"{item['selective_exact_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The locked system improves forced exact accuracy by "
            f"`{summary['deltas']['locked_vs_original_forced_exact_pp']:.2f}` "
            "percentage points over the original VLM. Its selective output is "
            f"`{locked['overall']['selective_exact_accuracy']:.4f}` accurate at "
            f"`{locked['overall']['selective_coverage']:.4f}` coverage.",
            "",
            "![Policy comparison](figures/policy_comparison.png)",
            "",
            "## Task Results",
            "",
            "| Task | Forced metric | Forced exact | Selective metric | Selective exact | Coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task in TASK_TYPES:
        item = locked["tasks"][task]
        lines.append(
            f"| {task} | {item['forced_mean_score']:.6f} | "
            f"{item['forced_exact_accuracy']:.6f} | "
            f"{item['selective_mean_score']:.6f} | "
            f"{item['selective_exact_accuracy']:.6f} | "
            f"{item['selective_coverage']:.6f} |"
        )
    lines.extend(
        [
            "",
            "![Task performance](figures/task_performance.png)",
            "",
            "## Dev-to-Test Generalization",
            "",
            "| Task | Dev forced | Test forced | Dev selective | Test selective | Dev coverage | Test coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task in TASK_TYPES:
        dev_item = dev["tasks"][task]
        test_item = locked["tasks"][task]
        lines.append(
            f"| {task} | {dev_item['forced_mean_score']:.6f} | "
            f"{test_item['forced_mean_score']:.6f} | "
            f"{dev_item['selective_mean_score']:.6f} | "
            f"{test_item['selective_mean_score']:.6f} | "
            f"{dev_item['selective_coverage']:.6f} | "
            f"{test_item['selective_coverage']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The spatial score threshold selected on Dev20 is conservative on "
            "Test80: locked spatial forced accuracy is `0.6750` and coverage is "
            "`0.7250`, while the pre-defined shared score-0.30 baseline reaches "
            "`0.8625` forced accuracy and `0.9375` coverage. The locked result "
            "remains the official held-out result; the shared policy is reported "
            "only as a pre-defined ablation and must not replace it post hoc.",
            "",
            "## Failure Attribution",
            "",
            f"- Listing: `{failures['tasks']['object_listing']['forced_errors']}` non-exact answers; tags are `{failures['tasks']['object_listing']['failure_tag_counts']}`.",
            f"- Existence: `{failures['tasks']['object_existence']['abstentions']}` disagreements and `{failures['tasks']['object_existence']['selective_errors']}` selective errors.",
            f"- Spatial: `{failures['tasks']['spatial_relation']['forced_errors']}` forced errors, `{failures['tasks']['spatial_relation']['abstentions']}` abstentions, and `{failures['tasks']['spatial_relation']['selective_errors']}` selective errors.",
            "",
            "![Failure breakdown](figures/failure_breakdown.png)",
            "",
            "### Highest-Severity Samples",
            "",
            "| ID | Task | GT | Forced answer | Status | Tags |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in top_failures[:12]:
        lines.append(
            f"| {item['id']} | {item['task_type']} | {item['gt_answer']} | "
            f"{item['forced_answer']} | {item['status']} | "
            f"{', '.join(item['failure_tags'])} |"
        )
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            f"- Mean end-to-end latency: `{latency['mean']:.6f}` seconds per question.",
            f"- Throughput: `{latency['throughput_samples_per_second']:.6f}` questions/second.",
            f"- Peak allocated CUDA memory: `{memory.get('peak_allocated_max', 0.0):.4f}` GiB.",
            "",
            "## Resume-Ready Summary",
            "",
            "Built a task-aware grounded multimodal answering pipeline using "
            "Qwen3-VL, Grounding DINO, and SAM 2.1; improved held-out exact "
            "accuracy from 56.7% to 73.3%, and achieved 80.2% selective accuracy "
            "at 84.2% coverage with auditable box/mask evidence.",
            "",
            "## Frozen Decision",
            "",
            "Do not change policy thresholds or rerun calibration on Test80. "
            "Future spatial-policy work requires a new validation split or nested "
            "cross-validation, followed by evaluation on a new held-out set.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    test_path = project_path(args.test_predictions)
    run_dir = test_path.parent
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "run_config.json"
    errors_path = run_dir / "errors.jsonl"
    original_path = project_path(args.original_vlm_predictions)
    structured_path = project_path(args.structured_listing_metrics)
    dev_path = project_path(args.dev_calibration_summary)
    image_ids_path = project_path(args.test_image_ids)
    output_dir = project_path(args.output_dir) if args.output_dir else run_dir / "final_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(test_path)
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    if run_config.get("protocol") != "locked_task_aware_evidence_answering_v1":
        raise ValueError("Test run is not a locked task-aware protocol.")
    if len(records) != 240 or len({int(item["image_id"]) for item in records}) != 80:
        raise ValueError("Frozen Test80 report requires 240 questions from 80 images.")
    if errors_path.exists() and errors_path.stat().st_size:
        raise RuntimeError("Frozen Test80 run contains error records.")

    test_ids = set(load_image_ids(image_ids_path))
    if {int(item["image_id"]) for item in records} != test_ids:
        raise ValueError("Prediction image IDs do not exactly match Test80.")
    task_counts = {
        task: sum(item["task_type"] == task for item in records)
        for task in TASK_TYPES
    }
    if set(task_counts.values()) != {80}:
        raise ValueError(f"Expected 80 records per task, got {task_counts}.")

    original_records = [
        item
        for item in load_jsonl(original_path)
        if int(item["image_id"]) in test_ids
    ]
    original_metrics = aggregate_metrics(original_records, expected_samples=240)
    original_uniform = uniform_original_metrics(original_metrics)
    structured_metrics = json.loads(structured_path.read_text(encoding="utf-8"))
    dev_summary = json.loads(dev_path.read_text(encoding="utf-8"))
    dev_locked = dev_summary["selected_metrics"]

    initial_config = EvidencePolicyConfig(
        min_grounding_score=float(run_config["evidence_score_threshold"]),
        min_mask_score=run_config.get("evidence_mask_score_threshold"),
        min_mask_area_ratio=float(
            run_config.get("evidence_min_mask_area_ratio", 0.0)
        ),
        relation_margin=float(run_config.get("relation_margin", 0.08)),
    )
    sizes = load_sizes(records)
    initial_records = []
    locked_records = []
    for record in records:
        width, height = sizes[int(record["image_id"])]
        initial_records.append(
            replay_grounded_policy(
                record,
                initial_config,
                image_width=width,
                image_height=height,
            )
        )
        locked_records.append(
            build_policy_record(
                record,
                record["answer_policy"],
                policy_name="locked_task_aware",
                policy_config=record["applied_policy"],
            )
        )
    initial_by_id = {item["id"]: item for item in initial_records}
    locked_aggregate = aggregate_policy_records(locked_records)
    initial_aggregate = aggregate_policy_records(initial_records)

    rebuilt_metrics = {
        "closed_set_answers": saved_metrics["closed_set_answers"],
        "selective_answers": saved_metrics["selective_answers"],
    }
    aggregate_projection = {
        "closed_set_answers": {
            "overall": {
                "count": locked_aggregate["overall"]["count"],
                "mean_score": locked_aggregate["overall"]["forced_mean_score"],
                "exact_accuracy": locked_aggregate["overall"][
                    "forced_exact_accuracy"
                ],
            }
        },
        "selective_answers": {
            "answered": locked_aggregate["overall"]["selective_answered"],
            "abstained": locked_aggregate["overall"]["selective_abstained"],
            "coverage": locked_aggregate["overall"]["selective_coverage"],
            "mean_score": locked_aggregate["overall"]["selective_mean_score"],
            "exact_accuracy": locked_aggregate["overall"][
                "selective_exact_accuracy"
            ],
        },
    }
    metrics_recompute_match = (
        aggregate_projection["closed_set_answers"]["overall"]
        == rebuilt_metrics["closed_set_answers"]["overall"]
        and aggregate_projection["selective_answers"]
        == {
            key: rebuilt_metrics["selective_answers"][key]
            for key in aggregate_projection["selective_answers"]
        }
    )

    policy_path = dev_path.parent / "selected_policy.json"
    structured_predictions_path = structured_path.parent / "predictions.jsonl"
    hash_checks = {
        "dataset": sha256sum(project_path("data/eval_v0/questions.jsonl"))
        == run_config["dataset_sha256"],
        "structured_predictions": sha256sum(
            structured_predictions_path
        )
        == run_config["structured_predictions_sha256"],
        "policy": sha256sum(policy_path) == run_config["policy_file_sha256"],
        "answer_vlm": sha256sum(original_path)
        == run_config["answer_vlm_predictions_sha256"],
    }

    analyses = [
        analyze_answer_record(record, initial_by_id.get(record["id"]))
        for record in records
    ]
    analyses.sort(key=lambda item: (-item["severity_score"], item["id"]))
    failure_summary = aggregate_answer_analysis(analyses)
    comparison = {
        "original_vlm": original_uniform,
        "initial_shared_evidence": initial_aggregate,
        "locked_task_aware": locked_aggregate,
    }
    locked_overall = locked_aggregate["overall"]
    original_overall = original_uniform["overall"]
    summary = {
        "protocol": "frozen_test80_final_report_v1",
        "integrity": {
            "expected": 240,
            "completed": len(records),
            "error_attempts": saved_metrics["coverage"]["error_attempts"],
            "task_counts": task_counts,
            "hash_checks": hash_checks,
            "all_hashes_match": all(hash_checks.values()),
            "metrics_recompute_match": metrics_recompute_match,
        },
        "comparison": comparison,
        "structured_listing_baseline": structured_metrics["tasks"][
            "object_listing"
        ],
        "dev_locked_metrics": dev_locked,
        "locked_metrics": saved_metrics,
        "deltas": {
            "locked_vs_original_forced_mean_score": round(
                locked_overall["forced_mean_score"]
                - original_overall["forced_mean_score"],
                6,
            ),
            "locked_vs_original_forced_exact_pp": round(
                100
                * (
                    locked_overall["forced_exact_accuracy"]
                    - original_overall["forced_exact_accuracy"]
                ),
                4,
            ),
            "locked_vs_initial_selective_exact_pp": round(
                100
                * (
                    locked_overall["selective_exact_accuracy"]
                    - initial_aggregate["overall"]["selective_exact_accuracy"]
                ),
                4,
            ),
            "listing_f1_vs_structured_only": round(
                locked_aggregate["tasks"]["object_listing"][
                    "forced_mean_score"
                ]
                - structured_metrics["tasks"]["object_listing"]["mean_score"],
                6,
            ),
        },
        "failure_summary": failure_summary,
    }

    write_json(output_dir / "final_summary.json", summary)
    write_json(output_dir / "failure_summary.json", failure_summary)
    write_jsonl(output_dir / "failure_analysis.jsonl", analyses)
    write_failure_csv(output_dir / "failure_analysis.csv", analyses)
    write_csv(output_dir / "policy_comparison.csv", policy_rows(comparison))
    write_csv(output_dir / "task_metrics.csv", task_rows(comparison))
    generate_figures(output_dir, comparison, failure_summary)
    (output_dir / "final_report.md").write_text(
        render_report(summary, analyses), encoding="utf-8"
    )

    print(f"Output:  {output_dir}")
    print(f"Report:  {output_dir / 'final_report.md'}")
    print(f"Figures: {output_dir / 'figures'}")
    print(json.dumps(summary["integrity"], ensure_ascii=False, indent=2))
    print(json.dumps(comparison["locked_task_aware"]["overall"], indent=2))


if __name__ == "__main__":
    main()
