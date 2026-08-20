"""Freeze the final Verifier Dev decision without running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

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
from scripts.compare_verifier_dev_ablations import (
    slugify,
    validate_semantic_source,
    write_csv_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from grounded_visual_assistant.evaluation import parse_yes_no
from grounded_visual_assistant.pope_dataset import (
    read_json_records,
    sha256sum,
)
from grounded_visual_assistant.pope_evaluation import binary_metrics
from grounded_visual_assistant.verifier_final_reporting import (
    VERIFIER_FINAL_FREEZE_PROTOCOL,
    build_failure_analysis,
    build_variant_summary,
    markdown_report,
    validate_final_freeze,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the rejected Dev verifier branch and generate an "
            "auditable V1/V2/V3 failure report without model inference."
        )
    )
    parser.add_argument(
        "--config", default="configs/verifier_dev_final_report_v1.yaml"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required verifier artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _source_entry(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required verifier artifact missing: {path}")
    return {"path": _relative(path), "sha256": sha256sum(path)}


def _source_bundle_sha256(sources: Mapping[str, Mapping[str, str]]) -> str:
    payload = "".join(
        f"{name}\t{item['path']}\t{item['sha256']}\n"
        for name, item in sorted(sources.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_source_hash(
    run_config: Mapping[str, Any], key: str, expected: str, *, stage: str
) -> None:
    if run_config.get(key) != expected:
        raise RuntimeError(f"{stage} source-chain hash mismatch: {key}.")


def _validate_selected_baseline(
    baseline_records: list[Mapping[str, Any]],
    selected_records: list[Mapping[str, Any]],
) -> None:
    if len(selected_records) != len(baseline_records):
        raise RuntimeError("Stage 38 selected-prediction coverage mismatch.")
    selected_by_id = {str(item["id"]): item for item in selected_records}
    if len(selected_by_id) != len(selected_records):
        raise RuntimeError("Stage 38 selected predictions contain duplicates.")
    for item in baseline_records:
        baseline_id = str(item["id"])
        selected = selected_by_id.get(baseline_id)
        if selected is None:
            raise RuntimeError(
                f"Stage 38 selected prediction missing: {baseline_id}."
            )
        expected = parse_yes_no(str(item["prediction"]))
        if selected.get("prediction") != expected:
            raise RuntimeError(
                f"Stage 38 selected prediction is not baseline: {baseline_id}."
            )


def _validate_v3_predictions(
    baseline_records: list[Mapping[str, Any]],
    v3_predictions: list[Mapping[str, Any]],
) -> None:
    baseline_ids = [str(item["id"]) for item in baseline_records]
    v3_ids = [str(item["id"]) for item in v3_predictions]
    if len(v3_ids) != len(set(v3_ids)):
        raise RuntimeError("Stage 39 predictions contain duplicate IDs.")
    if v3_ids != baseline_ids:
        raise RuntimeError("Stage 39 prediction order/coverage mismatch.")


def _load_settings(args: argparse.Namespace) -> dict[str, Any]:
    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != VERIFIER_FINAL_FREEZE_PROTOCOL:
        raise ValueError(
            f"Unsupported final-freeze protocol: {config.get('protocol')}"
        )
    final_policy = dict(config["final_policy"])
    if final_policy.get("answer_rewrite_enabled") is not False:
        raise ValueError("Final answer rewriting must be disabled.")
    if final_policy.get("held_out_verifier_run_permitted") is not False:
        raise ValueError("Rejected Dev verifiers cannot advance to held-out.")
    inputs = {
        key: project_path(value)
        for key, value in dict(config["inputs"]).items()
    }
    runtime = dict(config["runtime"])
    return {
        "config_path": config_path,
        "config": config,
        "inputs": inputs,
        "variants": dict(config["representative_variants"]),
        "final_policy": final_policy,
        "output_dir": project_path(
            args.output_dir or runtime["output_dir"]
        ),
        "run_name": args.run_name or str(runtime["run_name"]),
    }


def _case_csv_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in analysis["cases"]:
        semantic = ";".join(
            f"{review['grounding_score']:.6f}:{review['answer']}"
            for review in item["semantic_reviews"]
        )
        rows.append(
            {
                "id": item["id"],
                "scope": item["scope"],
                "pair_role": item["pair_role"],
                "image_id": item["image_id"],
                "object": item["object"],
                "gt_answer": item["gt_answer"],
                "baseline_prediction": item["baseline_prediction"],
                "grounding_candidate_count": item[
                    "grounding_candidate_count"
                ],
                "max_grounding_score": item["max_grounding_score"],
                "semantic_reviews": semantic,
                "v2_prediction": item["v2_prediction"],
                "v2_correction": item["v2_correction"],
                "v3_selected_label": item["v3_selected_label"],
                "v3_prediction": item["v3_prediction"],
                "final_frozen_prediction": item[
                    "final_frozen_prediction"
                ],
                "taxonomy": item["taxonomy"],
                "explanation": item["explanation"],
            }
        )
    return rows


def _validate_or_create_run_config(
    path: Path, current: Mapping[str, Any]
) -> None:
    if path.is_file():
        existing = _load_json(path)
        for key in (
            "protocol",
            "config_sha256",
            "source_bundle_sha256",
            "held_out_data_used_for_selection",
            "model_inference_required",
        ):
            if existing.get(key) != current.get(key):
                raise RuntimeError(
                    "Final verifier output is source-incompatible; choose "
                    f"a new --run-name. Mismatch: {key}."
                )
        return
    write_json_atomic(path, current)


def main() -> None:
    args = parse_args()
    settings = _load_settings(args)
    inputs = settings["inputs"]

    baseline_path = inputs["baseline_predictions"]
    baseline_records, baseline_source = validate_baseline_source(
        baseline_path
    )
    evidence_path = inputs["grounding_evidence"]
    evidence, grounding_source = validate_grounding_source(
        evidence_path.parent
    )
    baseline_by_id = {str(item["id"]): item for item in baseline_records}
    validate_alignment(evidence, baseline_by_id)
    semantic_path = inputs["semantic_reviews"]
    semantic_reviews, semantic_source = validate_semantic_source(
        semantic_path.parent,
        baseline_source=baseline_source,
        grounding_source=grounding_source,
    )

    stage38_dir = inputs["stage38_run_dir"]
    stage39_dir = inputs["stage39_run_dir"]
    stage38_paths = {
        "stage38_metrics": stage38_dir / "metrics.json",
        "stage38_selected_policy": stage38_dir / "selected_policy.json",
        "stage38_run_config": stage38_dir / "run_config.json",
        "stage38_corrections": stage38_dir / "corrections.jsonl",
        "stage38_selected_predictions": (
            stage38_dir / "selected_predictions.jsonl"
        ),
    }
    stage39_paths = {
        "stage39_metrics": stage39_dir / "metrics.json",
        "stage39_decision": stage39_dir / "v3_decision.json",
        "stage39_run_config": stage39_dir / "run_config.json",
        "stage39_reviews": stage39_dir / "contrastive_reviews.jsonl",
        "stage39_predictions": stage39_dir / "predictions.jsonl",
    }
    stage38_metrics = _load_json(stage38_paths["stage38_metrics"])
    stage38_policy = _load_json(
        stage38_paths["stage38_selected_policy"]
    )
    stage38_config = _load_json(stage38_paths["stage38_run_config"])
    stage38_corrections = read_json_records(
        stage38_paths["stage38_corrections"]
    )
    selected_predictions = read_json_records(
        stage38_paths["stage38_selected_predictions"]
    )
    stage39_metrics = _load_json(stage39_paths["stage39_metrics"])
    stage39_decision = _load_json(stage39_paths["stage39_decision"])
    stage39_config = _load_json(stage39_paths["stage39_run_config"])
    stage39_reviews = read_json_records(stage39_paths["stage39_reviews"])
    stage39_predictions = read_json_records(
        stage39_paths["stage39_predictions"]
    )

    for stage, run_config in (
        ("Stage 38", stage38_config),
        ("Stage 39", stage39_config),
    ):
        _require_source_hash(
            run_config,
            "baseline_predictions_sha256",
            baseline_source["baseline_predictions_sha256"],
            stage=stage,
        )
        _require_source_hash(
            run_config,
            "grounding_evidence_sha256",
            grounding_source["grounding_evidence_sha256"],
            stage=stage,
        )
        _require_source_hash(
            run_config,
            "semantic_reviews_sha256",
            semantic_source["semantic_reviews_sha256"],
            stage=stage,
        )
        if run_config.get("held_out_data_used_for_selection") is not False:
            raise RuntimeError(f"{stage} used held-out data for selection.")

    freeze = validate_final_freeze(
        stage38_metrics,
        stage38_policy,
        stage39_metrics,
        stage39_decision,
    )
    _validate_selected_baseline(baseline_records, selected_predictions)
    _validate_v3_predictions(baseline_records, stage39_predictions)
    if binary_metrics(baseline_records) != freeze["baseline_metrics"]:
        raise RuntimeError("Recomputed baseline metrics differ from freeze.")
    if binary_metrics(stage39_predictions) != stage39_metrics[
        "evaluation"
    ]["v3"]:
        raise RuntimeError("Recomputed V3 metrics differ from Stage 39.")

    variant_ids = settings["variants"]
    variants = build_variant_summary(
        stage38_metrics,
        stage39_metrics,
        v1_policy_id=variant_ids["v1_best_policy_id"],
        v2_rescue_policy_id=variant_ids["v2_rescue_policy_id"],
        v2_noop_policy_id=variant_ids["v2_noop_policy_id"],
    )
    analysis = build_failure_analysis(
        baseline_records,
        evidence,
        semantic_reviews,
        stage38_corrections,
        stage39_reviews,
        stage39_predictions,
        v2_policy_id=variant_ids["v2_rescue_policy_id"],
    )

    config_path = settings["config_path"]
    sources = {
        "config": _source_entry(config_path),
        "baseline_predictions": _source_entry(baseline_path),
        "grounding_evidence": _source_entry(evidence_path),
        "semantic_reviews": _source_entry(semantic_path),
        **{
            name: _source_entry(path)
            for name, path in {**stage38_paths, **stage39_paths}.items()
        },
    }
    source_bundle_hash = _source_bundle_sha256(sources)
    audit = {
        "protocol": VERIFIER_FINAL_FREEZE_PROTOCOL,
        "baseline_questions": len(baseline_records),
        "grounding_queries": len(evidence),
        "semantic_reviews": len(semantic_reviews),
        "stage38_policies": stage38_metrics["coverage"]["policies"],
        "stage39_reviews": len(stage39_reviews),
        "failure_cases": len(analysis["cases"]),
        "final_decision": freeze["decision"],
        "source_bundle_sha256": source_bundle_hash,
        "held_out_data_used_for_selection": False,
        "model_inference_required": False,
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.audit_only:
        print("Audit complete: no model was loaded and no output was written.")
        return

    run_dir = settings["output_dir"] / slugify(settings["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "created_at_utc": utc_now(),
        "protocol": VERIFIER_FINAL_FREEZE_PROTOCOL,
        "config": _relative(config_path),
        "config_sha256": sha256sum(config_path),
        "source_bundle_sha256": source_bundle_hash,
        "sources": sources,
        "held_out_data_used_for_selection": False,
        "model_inference_required": False,
    }
    _validate_or_create_run_config(run_dir / "run_config.json", run_config)

    final_policy = {
        "generated_at_utc": utc_now(),
        "protocol": VERIFIER_FINAL_FREEZE_PROTOCOL,
        **freeze,
        **settings["final_policy"],
        "development_protocol": {
            "questions": len(baseline_records),
            "selection_stages": ["stage38_v1_v2", "stage39_v3"],
            "selection_gates": stage38_metrics["selection"][
                "selection_gates"
            ],
        },
        "rejected_variants": [
            {
                "variant_id": item["variant_id"],
                "accuracy": item["accuracy"],
                "f1": item["f1"],
                "net_correct": item["net_correct"],
                "rejection_reasons": item["rejection_reasons"],
            }
            for item in variants
            if item["variant_id"] != "baseline"
        ],
        "source_bundle_sha256": source_bundle_hash,
        "scientific_claim_boundary": (
            "The evidence modules improve inspectability but did not improve "
            "Dev answer accuracy under the frozen acceptance gates."
        ),
    }
    write_json_atomic(run_dir / "final_policy.json", final_policy)
    write_json_atomic(run_dir / "failure_analysis.json", analysis)
    write_jsonl_atomic(run_dir / "failure_cases.jsonl", analysis["cases"])
    write_csv_atomic(run_dir / "failure_cases.csv", _case_csv_rows(analysis))
    write_jsonl_atomic(run_dir / "variant_summary.jsonl", variants)
    write_csv_atomic(run_dir / "variant_summary.csv", variants)
    write_text_atomic(
        run_dir / "report.md",
        markdown_report(final_policy, variants, analysis),
    )
    generated = {
        name: _source_entry(run_dir / name)
        for name in (
            "run_config.json",
            "final_policy.json",
            "variant_summary.jsonl",
            "variant_summary.csv",
            "failure_analysis.json",
            "failure_cases.jsonl",
            "failure_cases.csv",
            "report.md",
        )
    }
    write_json_atomic(
        run_dir / "artifact_manifest.json",
        {
            "generated_at_utc": utc_now(),
            "protocol": VERIFIER_FINAL_FREEZE_PROTOCOL,
            "source_bundle_sha256": source_bundle_hash,
            "artifacts": generated,
        },
    )

    print(f"Run dir:       {run_dir}")
    print(f"Final policy:  {run_dir / 'final_policy.json'}")
    print(f"Report:        {run_dir / 'report.md'}")
    print(f"Manifest:      {run_dir / 'artifact_manifest.json'}")


if __name__ == "__main__":
    main()
