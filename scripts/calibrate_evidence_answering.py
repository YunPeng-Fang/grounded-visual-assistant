"""Calibrate task-aware answer policies offline on the sealed Dev20 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.evidence_answering import EvidencePolicyConfig
from grounded_visual_assistant.policy_calibration import (
    aggregate_policy_records,
    build_policy_record,
    fuse_existence_consensus,
    replay_grounded_policy,
    structured_listing_policy,
)


TASK_TYPES = ("object_listing", "object_existence", "spatial_relation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep saved Dev20 evidence and lock a task-aware answer policy."
    )
    parser.add_argument(
        "--evidence-predictions",
        default=(
            "outputs/eval_answering_v0/"
            "dev__evidence-answering__coco80-json-v1__box-0.30__text-0.30/"
            "predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--vlm-predictions",
        default="outputs/eval_v0/eval_v0__qwen3-vl-8b-instruct/predictions.jsonl",
        help="Original three-task Qwen predictions used for existence fusion.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--score-thresholds",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60",
    )
    parser.add_argument("--mask-score-thresholds", default="none,0.90,0.95")
    parser.add_argument("--mask-area-ratios", default="0,0.001,0.003,0.005")
    parser.add_argument(
        "--relation-margins",
        default="0.04,0.08,0.12,0.16,0.20,0.24",
    )
    parser.add_argument("--min-selective-coverage", type=float, default=0.80)
    args = parser.parse_args()
    if not 0.0 < args.min_selective_coverage <= 1.0:
        parser.error("--min-selective-coverage must be in (0, 1].")
    return args


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
            if not sample_id:
                raise ValueError(f"Missing id on {path}:{line_number}.")
            if sample_id in seen:
                raise ValueError(f"Duplicate id on {path}: {sample_id}")
            seen.add(sample_id)
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def parse_float_grid(value: str, *, allow_none: bool = False) -> list[float | None]:
    parsed: list[float | None] = []
    for item in value.split(","):
        stripped = item.strip().lower()
        if allow_none and stripped in {"none", "null"}:
            parsed.append(None)
            continue
        number = float(stripped)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"Grid value must be in [0, 1], got {number}.")
        parsed.append(number)
    if not parsed:
        raise ValueError("A calibration grid cannot be empty.")
    return list(dict.fromkeys(parsed))


def sha256sum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_image_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_image_sizes(records: list[dict[str, Any]]) -> dict[int, tuple[int, int]]:
    sizes = {}
    for record in records:
        image_id = int(record["image_id"])
        if image_id in sizes:
            continue
        image_path = resolve_image_path(str(record["image"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Calibration image not found: {image_path}")
        with Image.open(image_path) as image:
            sizes[image_id] = image.size
    return sizes


def config_payload(config: EvidencePolicyConfig) -> dict[str, Any]:
    return {
        "min_grounding_score": config.min_grounding_score,
        "min_mask_score": config.min_mask_score,
        "min_mask_area_ratio": config.min_mask_area_ratio,
        "relation_margin": config.relation_margin,
    }


def replay_task(
    records: list[dict[str, Any]],
    config: EvidencePolicyConfig,
    sizes: dict[int, tuple[int, int]],
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        width, height = sizes[int(record["image_id"])]
        output.append(
            replay_grounded_policy(
                record,
                config,
                image_width=width,
                image_height=height,
            )
        )
    return output


def candidate(
    *,
    candidate_id: str,
    task_type: str,
    mode: str,
    config: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = aggregate_policy_records(records)["overall"]
    return {
        "candidate_id": candidate_id,
        "task_type": task_type,
        "mode": mode,
        "config": config,
        "metrics": metrics,
    }


def config_id(prefix: str, config: EvidencePolicyConfig) -> str:
    mask = "none" if config.min_mask_score is None else f"{config.min_mask_score:.3f}"
    return (
        f"{prefix}__score-{config.min_grounding_score:.3f}"
        f"__mask-{mask}__area-{config.min_mask_area_ratio:.4f}"
        f"__margin-{config.relation_margin:.3f}"
    )


def simplicity_key(item: dict[str, Any], base_score: float) -> tuple[float, ...]:
    config = item["config"]
    score = float(config.get("min_grounding_score", base_score))
    mask = config.get("min_mask_score")
    area = float(config.get("min_mask_area_ratio", 0.0))
    margin = float(config.get("relation_margin", 0.08))
    return (
        -abs(score - base_score),
        float(mask is None),
        -area,
        -abs(margin - 0.08),
    )


def select_listing(
    candidates: list[dict[str, Any]], base_score: float
) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda item: (
            float(item["metrics"]["forced_mean_score"]),
            float(item["metrics"]["forced_exact_accuracy"]),
            float(item["metrics"]["selective_coverage"]),
            *simplicity_key(item, base_score),
        ),
    )


def select_selective(
    candidates: list[dict[str, Any]],
    *,
    minimum_coverage: float,
    base_score: float,
) -> dict[str, Any]:
    eligible = [
        item
        for item in candidates
        if float(item["metrics"]["selective_coverage"]) >= minimum_coverage
    ]
    if not eligible:
        raise RuntimeError(
            f"No candidate reaches selective coverage {minimum_coverage:.3f}."
        )
    return max(
        eligible,
        key=lambda item: (
            float(item["metrics"]["selective_exact_accuracy"]),
            float(item["metrics"]["selective_coverage"]),
            float(item["metrics"]["forced_exact_accuracy"]),
            float(item["metrics"]["selective_mean_score"]),
            *simplicity_key(item, base_score),
        ),
    )


def flatten_candidate(item: dict[str, Any], selected_id: str) -> dict[str, Any]:
    config = item["config"]
    metrics = item["metrics"]
    return {
        "selected": item["candidate_id"] == selected_id,
        "task_type": item["task_type"],
        "candidate_id": item["candidate_id"],
        "mode": item["mode"],
        "min_grounding_score": config.get("min_grounding_score"),
        "min_mask_score": config.get("min_mask_score"),
        "min_mask_area_ratio": config.get("min_mask_area_ratio"),
        "relation_margin": config.get("relation_margin"),
        "forced_mean_score": metrics["forced_mean_score"],
        "forced_exact_accuracy": metrics["forced_exact_accuracy"],
        "selective_coverage": metrics["selective_coverage"],
        "selective_mean_score": metrics["selective_mean_score"],
        "selective_exact_accuracy": metrics["selective_exact_accuracy"],
    }


def baseline_vlm_policy(record: dict[str, Any]) -> dict[str, Any]:
    answer = str(record.get("prediction", ""))
    output = {
        "forced_answer": answer,
        "selective_answer": answer,
        "abstained": False,
        "status": "vlm_baseline",
        "claim_supported": False,
        "support_type": "vlm_only",
        "claim_count": 1,
        "unsupported_claim_count": 1,
        "confidence": None,
        "selected_evidence": [],
        "accepted_evidence": [],
        "rejected_evidence": [],
        "diagnostics": {},
    }
    return build_policy_record(
        record,
        output,
        policy_name="original_vlm_baseline",
        policy_config={},
    )


def current_policy_record(record: dict[str, Any]) -> dict[str, Any]:
    return build_policy_record(
        record,
        record["answer_policy"],
        policy_name="current_evidence_policy",
        policy_config={
            "min_grounding_score": record["thresholds"]["evidence_score"],
            "min_mask_score": record["thresholds"]["evidence_mask_score"],
            "min_mask_area_ratio": record["thresholds"][
                "evidence_min_mask_area_ratio"
            ],
            "relation_margin": record["thresholds"]["relation_margin"],
        },
    )


def render_report(summary: dict[str, Any]) -> str:
    selected = summary["selected_policy"]["tasks"]
    metrics = summary["selected_metrics"]
    comparison = summary["comparison"]
    lines = [
        "# Dev20 Evidence Policy Calibration",
        "",
        "This report is derived entirely from saved Dev20 predictions. No model "
        "inference was repeated and no Test80 record contributed to selection.",
        "",
        "## Selection Rule",
        "",
        f"- Minimum selective coverage: {summary['minimum_selective_coverage']:.2f}",
        "- Listing: maximize forced macro F1, then exact accuracy.",
        "- Existence and spatial relation: maximize selective exact accuracy among "
        "candidates meeting the coverage constraint, then coverage.",
        "- Ties prefer fewer added gates and values nearest the original policy.",
        "",
        "## Selected Policies",
        "",
        "| Task | Mode | Score | Mask score | Min mask area ratio | Relation margin |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in TASK_TYPES:
        item = selected[task]
        config = item["config"]
        lines.append(
            f"| {task} | {item['mode']} | "
            f"{config.get('min_grounding_score', '-')} | "
            f"{config.get('min_mask_score', '-')} | "
            f"{config.get('min_mask_area_ratio', '-')} | "
            f"{config.get('relation_margin', '-')} |"
        )
    lines.extend(
        [
            "",
            "## Overall Comparison",
            "",
            "| Policy | Forced mean score | Forced exact | Selective coverage | "
            "Selective exact |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("original_vlm", "current_evidence", "selected_task_aware"):
        item = comparison[name]["overall"]
        lines.append(
            f"| {name} | {item['forced_mean_score']:.6f} | "
            f"{item['forced_exact_accuracy']:.6f} | "
            f"{item['selective_coverage']:.6f} | "
            f"{item['selective_exact_accuracy']:.6f} |"
        )
    lines.extend(["", "## Selected Task Metrics", ""])
    for task, item in metrics["tasks"].items():
        lines.append(
            f"- `{task}`: forced score `{item['forced_mean_score']:.6f}`, "
            f"forced exact `{item['forced_exact_accuracy']:.6f}`, selective "
            f"coverage `{item['selective_coverage']:.6f}`, selective exact "
            f"`{item['selective_exact_accuracy']:.6f}`."
        )
    lines.extend(
        [
            "",
            "## Protocol Decision",
            "",
            "Treat `selected_policy.json` as immutable before Test80. Re-running "
            "this calibration on Test80 or changing task thresholds after viewing "
            "test metrics would invalidate the held-out comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    evidence_path = project_path(args.evidence_predictions)
    vlm_path = project_path(args.vlm_predictions)
    evidence_run_dir = evidence_path.parent
    evidence_run_config_path = evidence_run_dir / "run_config.json"
    if not evidence_run_config_path.is_file():
        raise FileNotFoundError(
            f"Evidence run_config.json not found: {evidence_run_config_path}"
        )
    evidence_run_config = json.loads(
        evidence_run_config_path.read_text(encoding="utf-8")
    )
    split_path = Path(str(evidence_run_config.get("image_ids", "")))
    if "dev" not in split_path.stem.lower():
        raise RuntimeError(
            "Policy calibration is Dev-only; the evidence run is not marked as a "
            f"dev split: {split_path}"
        )

    records = load_jsonl(evidence_path)
    task_records = {
        task: [record for record in records if record["task_type"] == task]
        for task in TASK_TYPES
    }
    counts = {task: len(values) for task, values in task_records.items()}
    if len(set(counts.values())) != 1 or not all(counts.values()):
        raise ValueError(f"Expected balanced non-empty task records, got {counts}.")
    vlm_by_id = {record["id"]: record for record in load_jsonl(vlm_path)}
    missing_vlm = sorted(
        record["id"]
        for record in task_records["object_existence"]
        if record["id"] not in vlm_by_id
    )
    if missing_vlm:
        raise ValueError(f"VLM predictions miss existence IDs: {missing_vlm[:10]}")

    raw_box_threshold = float(evidence_run_config["box_threshold"])
    base_score = float(evidence_run_config["evidence_score_threshold"])
    scores = [float(value) for value in parse_float_grid(args.score_thresholds)]
    if min(scores) < raw_box_threshold:
        raise ValueError(
            "A score grid value is below the saved detector threshold; discarded "
            "candidates cannot be recovered offline."
        )
    mask_scores = parse_float_grid(args.mask_score_thresholds, allow_none=True)
    area_ratios = [float(value) for value in parse_float_grid(args.mask_area_ratios)]
    relation_margins = [
        float(value) for value in parse_float_grid(args.relation_margins)
    ]
    sizes = load_image_sizes(records)

    listing_candidates = [
        candidate(
            candidate_id="listing__structured-vlm-only",
            task_type="object_listing",
            mode="structured_vlm_only",
            config={},
            records=[
                structured_listing_policy(record)
                for record in task_records["object_listing"]
            ],
        )
    ]
    existence_candidates = []
    spatial_candidates = []

    base_margin = float(evidence_run_config["relation_margin"])
    for score, mask_score, area_ratio in product(scores, mask_scores, area_ratios):
        common_config = EvidencePolicyConfig(
            min_grounding_score=score,
            min_mask_score=mask_score,
            min_mask_area_ratio=area_ratio,
            relation_margin=base_margin,
        )
        listing_replay = replay_task(
            task_records["object_listing"], common_config, sizes
        )
        listing_candidates.append(
            candidate(
                candidate_id=config_id("listing", common_config),
                task_type="object_listing",
                mode="grounded_evidence_gate",
                config=config_payload(common_config),
                records=listing_replay,
            )
        )

        existence_detector = replay_task(
            task_records["object_existence"], common_config, sizes
        )
        existence_fused = [
            fuse_existence_consensus(record, vlm_by_id[record["id"]])
            for record in existence_detector
        ]
        existence_candidates.append(
            candidate(
                candidate_id=config_id("existence-consensus", common_config),
                task_type="object_existence",
                mode="vlm_grounding_consensus",
                config=config_payload(common_config),
                records=existence_fused,
            )
        )

        for relation_margin in relation_margins:
            spatial_config = EvidencePolicyConfig(
                min_grounding_score=score,
                min_mask_score=mask_score,
                min_mask_area_ratio=area_ratio,
                relation_margin=relation_margin,
            )
            spatial_candidates.append(
                candidate(
                    candidate_id=config_id("spatial", spatial_config),
                    task_type="spatial_relation",
                    mode="grounded_geometry",
                    config=config_payload(spatial_config),
                    records=replay_task(
                        task_records["spatial_relation"], spatial_config, sizes
                    ),
                )
            )

    selected_listing = select_listing(listing_candidates, base_score)
    selected_existence = select_selective(
        existence_candidates,
        minimum_coverage=args.min_selective_coverage,
        base_score=base_score,
    )
    selected_spatial = select_selective(
        spatial_candidates,
        minimum_coverage=args.min_selective_coverage,
        base_score=base_score,
    )
    selected = {
        "object_listing": selected_listing,
        "object_existence": selected_existence,
        "spatial_relation": selected_spatial,
    }

    if selected_listing["mode"] == "structured_vlm_only":
        selected_listing_records = [
            structured_listing_policy(record)
            for record in task_records["object_listing"]
        ]
    else:
        listing_config = EvidencePolicyConfig(**selected_listing["config"])
        selected_listing_records = replay_task(
            task_records["object_listing"], listing_config, sizes
        )
    existence_config = EvidencePolicyConfig(**selected_existence["config"])
    selected_existence_detector = replay_task(
        task_records["object_existence"], existence_config, sizes
    )
    selected_existence_records = [
        fuse_existence_consensus(record, vlm_by_id[record["id"]])
        for record in selected_existence_detector
    ]
    spatial_config = EvidencePolicyConfig(**selected_spatial["config"])
    selected_spatial_records = replay_task(
        task_records["spatial_relation"], spatial_config, sizes
    )
    selected_by_id = {
        record["id"]: record
        for record in (
            selected_listing_records
            + selected_existence_records
            + selected_spatial_records
        )
    }
    selected_records = [selected_by_id[record["id"]] for record in records]

    original_vlm_records = [
        baseline_vlm_policy(vlm_by_id[record["id"]])
        for record in records
        if record["id"] in vlm_by_id
    ]
    if len(original_vlm_records) != len(records):
        raise ValueError("Original VLM predictions do not cover every Dev20 question.")
    current_records = [current_policy_record(record) for record in records]
    selected_metrics = aggregate_policy_records(selected_records)
    comparison = {
        "original_vlm": aggregate_policy_records(original_vlm_records),
        "current_evidence": aggregate_policy_records(current_records),
        "selected_task_aware": selected_metrics,
    }

    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else evidence_run_dir / "policy_calibration"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_policy = {
        "created_at_utc": utc_now(),
        "protocol": "task_aware_evidence_fusion_v1",
        "development_split_only": True,
        "minimum_selective_coverage": args.min_selective_coverage,
        "source_evidence_predictions": str(evidence_path),
        "source_evidence_sha256": sha256sum(evidence_path),
        "source_vlm_predictions": str(vlm_path),
        "source_vlm_sha256": sha256sum(vlm_path),
        "selection_rules": {
            "object_listing": (
                "maximize forced mean score, exact accuracy, then coverage"
            ),
            "object_existence": (
                "maximize selective exact accuracy subject to coverage floor, "
                "then coverage and forced accuracy"
            ),
            "spatial_relation": (
                "maximize selective exact accuracy subject to coverage floor, "
                "then coverage and forced accuracy"
            ),
            "tie_break": "prefer simpler gates nearest the original policy",
        },
        "tasks": {
            task: {
                "candidate_id": item["candidate_id"],
                "mode": item["mode"],
                "config": item["config"],
                "dev_metrics": item["metrics"],
            }
            for task, item in selected.items()
        },
    }
    summary = {
        "generated_at_utc": utc_now(),
        "status": "completed",
        "split": str(split_path),
        "sample_counts": counts,
        "candidate_counts": {
            "object_listing": len(listing_candidates),
            "object_existence": len(existence_candidates),
            "spatial_relation": len(spatial_candidates),
        },
        "minimum_selective_coverage": args.min_selective_coverage,
        "selected_policy": selected_policy,
        "selected_metrics": selected_metrics,
        "comparison": comparison,
    }
    write_json(output_dir / "selected_policy.json", selected_policy)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "selected_predictions.jsonl", selected_records)
    (output_dir / "report.md").write_text(
        render_report(summary), encoding="utf-8"
    )

    all_candidates = {
        "object_listing": listing_candidates,
        "object_existence": existence_candidates,
        "spatial_relation": spatial_candidates,
    }
    write_json(output_dir / "candidates.json", all_candidates)
    rows = []
    for task, task_candidates in all_candidates.items():
        selected_id = selected[task]["candidate_id"]
        rows.extend(flatten_candidate(item, selected_id) for item in task_candidates)
    with (output_dir / "candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Output: {output_dir}")
    print("Selected policies:")
    for task in TASK_TYPES:
        item = selected[task]
        print(f"  {task}: {item['candidate_id']}")
    print(json.dumps(selected_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
