"""Compare frozen Verifier Dev policies without running any model."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_eval_pope_verifier import (
    project_path,
    utc_now,
    write_json_atomic,
)
from scripts.batch_ground_verifier_dev import validate_baseline_source
from scripts.batch_review_verifier_dev import (
    validate_alignment,
    validate_grounding_source,
)
from grounded_visual_assistant.pope_dataset import (
    read_json_records,
    sha256sum,
)
from grounded_visual_assistant.verifier_dev_ablation import (
    VERIFIER_DEV_ABLATION_PROTOCOL,
    build_policy_grid,
    evaluate_dev_policy,
    flatten_policy_summary,
    ordered_policy_ids_sha256,
    select_dev_policy,
)
from grounded_visual_assistant.verifier_dev_semantic_review import (
    VERIFIER_DEV_SEMANTIC_REVIEW_PROTOCOL,
    ordered_candidate_keys_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pre-registered V1/V2 policy grid over frozen Dev110, "
            "Dev57, and Dev23 artifacts."
        )
    )
    parser.add_argument(
        "--config", default="configs/verifier_dev_ablation_v1.yaml"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")
    return slug.lower() or "run"


def write_jsonl_atomic(
    path: Path, records: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            handle.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv_atomic(
    path: Path, records: list[Mapping[str, Any]]
) -> None:
    if not records:
        raise ValueError("Cannot write an empty ablation table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def load_settings(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    str,
    dict[str, Any],
    list[Any],
    dict[str, Any],
]:
    config_path = project_path(args.config)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != VERIFIER_DEV_ABLATION_PROTOCOL:
        raise ValueError(
            f"Unsupported Dev ablation protocol: {payload.get('protocol')}"
        )
    inputs = dict(payload["inputs"])
    universe = dict(payload["candidate_universe"])
    selection = dict(payload["selection"])
    runtime = dict(payload["runtime"])
    policies = build_policy_grid(
        score_thresholds=universe["score_thresholds"],
        templates=payload["policy_templates"],
        min_mask_score=universe.get("min_mask_score"),
        min_mask_area_ratio=float(universe["min_mask_area_ratio"]),
    )
    if selection.get("primary_metric") != "accuracy":
        raise ValueError("Stage 38 requires accuracy as the primary metric.")
    if selection.get("fallback_policy") != "baseline":
        raise ValueError("Stage 38 fallback must remain the frozen baseline.")
    if bool(selection.get("held_out_data_used_for_selection", True)):
        raise ValueError("Held-out data cannot be used for Dev selection.")
    output_dir = project_path(args.output_dir or runtime["output_dir"])
    run_name = args.run_name or str(runtime["run_name"])
    return (
        config_path,
        project_path(inputs["baseline_predictions"]),
        project_path(inputs["grounding_run_dir"]),
        project_path(inputs["semantic_run_dir"]),
        run_name,
        selection,
        policies,
        {
            "output_dir": output_dir,
            "config_sha256": sha256sum(config_path),
        },
    )


def validate_semantic_source(
    run_dir: Path,
    *,
    baseline_source: Mapping[str, Any],
    grounding_source: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reviews_path = run_dir / "semantic_reviews.jsonl"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "run_config.json"
    crop_dir = run_dir / "crops"
    for path in (reviews_path, metrics_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Dev semantic artifact missing: {path}"
            )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    if metrics.get("protocol") != VERIFIER_DEV_SEMANTIC_REVIEW_PROTOCOL:
        raise RuntimeError("Dev semantic metrics protocol mismatch.")
    if metrics.get("status") != "completed":
        raise RuntimeError("Dev semantic review is not completed.")
    coverage = metrics.get("coverage") or {}
    if (
        coverage.get("expected_candidates") != 23
        or coverage.get("completed_candidates") != 23
        or coverage.get("remaining_candidates") != 0
    ):
        raise RuntimeError("Dev semantic review must contain all 23 crops.")
    if run_config.get("protocol") != VERIFIER_DEV_SEMANTIC_REVIEW_PROTOCOL:
        raise RuntimeError("Dev semantic run config protocol mismatch.")
    source_links = {
        "baseline_predictions_sha256": baseline_source[
            "baseline_predictions_sha256"
        ],
        "grounding_evidence_sha256": grounding_source[
            "grounding_evidence_sha256"
        ],
    }
    for key, expected in source_links.items():
        if run_config.get(key) != expected:
            raise RuntimeError(
                f"Dev semantic source-chain hash mismatch: {key}."
            )

    reviews = read_json_records(reviews_path)
    if len(reviews) != 23:
        raise RuntimeError(
            f"Dev semantic review must contain 23 rows, found {len(reviews)}."
        )
    keys = [str(item["candidate_key"]) for item in reviews]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Dev semantic reviews contain duplicate keys.")
    observed_hash = ordered_candidate_keys_sha256(reviews)
    if observed_hash != run_config.get("ordered_candidate_keys_sha256"):
        raise RuntimeError("Dev semantic candidate-key hash mismatch.")
    forbidden = {"gt_answer", "pair_role", "expected_answer"}
    for item in reviews:
        leaked = forbidden.intersection(item)
        if leaked:
            raise RuntimeError(
                f"Semantic review contains diagnostic labels: {leaked}."
            )
        if str(item.get("answer", "")).strip().lower() not in {"yes", "no"}:
            raise RuntimeError(
                f"Semantic answer is not exact Yes/No: "
                f"{item.get('candidate_key')}."
            )
        crop_name = Path(str(item["crop_image"])).name
        crop_path = crop_dir / crop_name
        if not crop_path.is_file():
            raise FileNotFoundError(f"Semantic crop missing: {crop_path}")
        if sha256sum(crop_path) != item.get("crop_sha256"):
            raise RuntimeError(f"Semantic crop hash mismatch: {crop_name}.")
    return reviews, {
        "semantic_run_dir": str(run_dir),
        "semantic_reviews": str(reviews_path),
        "semantic_reviews_sha256": sha256sum(reviews_path),
        "semantic_metrics": str(metrics_path),
        "semantic_metrics_sha256": sha256sum(metrics_path),
        "semantic_run_config": str(config_path),
        "semantic_run_config_sha256": sha256sum(config_path),
        "ordered_candidate_keys_sha256": observed_hash,
    }


def validate_or_create_run_config(
    path: Path, current: Mapping[str, Any]
) -> None:
    immutable = (
        "protocol",
        "ablation_config_sha256",
        "baseline_predictions_sha256",
        "baseline_metrics_sha256",
        "baseline_run_config_sha256",
        "grounding_evidence_sha256",
        "grounding_metrics_sha256",
        "grounding_run_config_sha256",
        "semantic_reviews_sha256",
        "semantic_metrics_sha256",
        "semantic_run_config_sha256",
        "ordered_candidate_keys_sha256",
        "ordered_policy_ids_sha256",
        "policy_count",
    )
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        differences = {
            key: {"existing": existing.get(key), "current": current.get(key)}
            for key in immutable
            if existing.get(key) != current.get(key)
        }
        if differences:
            raise RuntimeError(
                "The Dev ablation output is incompatible. Choose a new "
                f"--run-name. Differences: {differences}"
            )
        return
    write_json_atomic(path, current)


def markdown_report(
    rows: list[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> str:
    selected_id = str(selection["selected_policy_id"])
    lines = [
        "# Verifier Dev110 Offline Ablation",
        "",
        f"- Decision: `{selection['decision']}`",
        f"- Selected policy: `{selected_id}`",
        f"- Eligible verifier policies: "
        f"{len(selection['eligible_policy_ids'])}",
        "- Selection data: Dev110 only; held-out POPE was not accessed.",
        "",
        "## Policy Table",
        "",
        "| Policy | Acc. | F1 | Beneficial | Harmful | Net | Reviews | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        marker = "yes" if row["eligible"] else "no"
        lines.append(
            f"| `{row['policy_id']}` | {row['accuracy']:.6f} | "
            f"{row['f1']:.6f} | {row['beneficial']} | "
            f"{row['harmful']} | {row['net_correct']} | "
            f"{row['semantic_reviews']} | {marker} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "A verifier is lockable only when it strictly improves "
                "accuracy, does not reduce F1, and produces positive net "
                "corrections. If no candidate passes every gate, the "
                "baseline remains locked and the verifier is rejected on "
                "development data."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    (
        config_path,
        baseline_path,
        grounding_run_dir,
        semantic_run_dir,
        run_name,
        selection_config,
        policies,
        runtime,
    ) = load_settings(args)
    baseline_records, baseline_source = validate_baseline_source(
        baseline_path
    )
    baseline_by_id = {
        str(item["id"]): item for item in baseline_records
    }
    evidence, grounding_source = validate_grounding_source(
        grounding_run_dir
    )
    validate_alignment(evidence, baseline_by_id)
    reviews, semantic_source = validate_semantic_source(
        semantic_run_dir,
        baseline_source=baseline_source,
        grounding_source=grounding_source,
    )
    evidence_by_id = {
        str(item["baseline_id"]): item for item in evidence
    }
    reviews_by_key = {
        str(item["candidate_key"]): item for item in reviews
    }
    policy_hash = ordered_policy_ids_sha256(policies)
    audit = {
        "baseline_predictions": len(baseline_records),
        "grounding_queries": len(evidence),
        "semantic_reviews": len(reviews),
        "policy_count": len(policies),
        "ordered_policy_ids_sha256": policy_hash,
        "held_out_data_used_for_selection": False,
        "model_inference_required": False,
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.audit_only:
        print("Audit complete: no model was loaded and no output was written.")
        return

    run_dir = runtime["output_dir"] / slugify(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "created_at_utc": utc_now(),
        "protocol": VERIFIER_DEV_ABLATION_PROTOCOL,
        "ablation_config": str(config_path),
        "ablation_config_sha256": runtime["config_sha256"],
        **baseline_source,
        **grounding_source,
        **semantic_source,
        "policy_count": len(policies),
        "ordered_policy_ids_sha256": policy_hash,
        "held_out_data_used_for_selection": False,
        "model_inference_required": False,
    }
    validate_or_create_run_config(run_dir / "run_config.json", run_config)

    summaries = []
    predictions_by_policy = {}
    for policy in policies:
        predictions, summary = evaluate_dev_policy(
            policy,
            baseline_records=baseline_records,
            evidence_by_baseline_id=evidence_by_id,
            reviews_by_key=reviews_by_key,
        )
        predictions_by_policy[policy.policy_id] = predictions
        summaries.append(summary)
    summaries, selection = select_dev_policy(
        summaries,
        require_strict_accuracy_improvement=bool(
            selection_config["require_strict_accuracy_improvement"]
        ),
        require_non_decreasing_f1=bool(
            selection_config["require_non_decreasing_f1"]
        ),
        require_positive_net_corrections=bool(
            selection_config["require_positive_net_corrections"]
        ),
    )
    rows = [flatten_policy_summary(item) for item in summaries]
    selected_id = str(selection["selected_policy_id"])
    selected_summary = next(
        item
        for item in summaries
        if item["policy"]["policy_id"] == selected_id
    )
    corrections = [
        {
            "policy_id": policy_id,
            **item,
        }
        for policy_id, predictions in predictions_by_policy.items()
        for item in predictions
        if item["changed"]
    ]
    metrics = {
        "generated_at_utc": utc_now(),
        "protocol": VERIFIER_DEV_ABLATION_PROTOCOL,
        "status": "completed",
        "coverage": {
            "dev_questions": len(baseline_records),
            "grounding_queries": len(evidence),
            "semantic_candidate_reviews": len(reviews),
            "policies": len(policies),
        },
        "baseline": summaries[0]["metrics"],
        "selection": selection,
        "selected_policy": selected_summary,
        "policy_table": rows,
        "methodology": {
            "model_inference_run": False,
            "ground_truth_used_for_inference": False,
            "ground_truth_used_for_offline_evaluation": True,
            "held_out_data_used_for_selection": False,
            "selection_fallback": "baseline",
        },
    }
    selected_policy = {
        "created_at_utc": utc_now(),
        "protocol": VERIFIER_DEV_ABLATION_PROTOCOL,
        **selection,
        "selected_policy": selected_summary["policy"],
        "dev_metrics": selected_summary["metrics"],
        "dev_corrections": selected_summary["corrections"],
        "dev_only_selection": True,
        "held_out_evaluation_pending": (
            selection["decision"] == "lock_dev_selected_verifier"
        ),
        "source_run_config_sha256": sha256sum(
            run_dir / "run_config.json"
        ),
    }

    write_json_atomic(run_dir / "metrics.json", metrics)
    write_json_atomic(run_dir / "selected_policy.json", selected_policy)
    write_jsonl_atomic(run_dir / "ablation_table.jsonl", rows)
    write_csv_atomic(run_dir / "ablation_table.csv", rows)
    write_jsonl_atomic(run_dir / "corrections.jsonl", corrections)
    write_jsonl_atomic(
        run_dir / "selected_predictions.jsonl",
        predictions_by_policy[selected_id],
    )
    write_text_atomic(
        run_dir / "report.md", markdown_report(rows, selection)
    )

    print(f"Run dir:          {run_dir}")
    print(f"Metrics:          {run_dir / 'metrics.json'}")
    print(f"Selected policy:  {run_dir / 'selected_policy.json'}")
    print(f"Report:           {run_dir / 'report.md'}")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
