"""Offline calibration helpers for task-aware evidence answer policies."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from .evaluation import parse_yes_no, score_prediction
from .evidence_answering import EvidencePolicyConfig, answer_with_evidence


LOCKED_POLICY_PROTOCOL = "task_aware_evidence_fusion_v1"
LOCKED_POLICY_MODES = {
    "object_listing": {"grounded_evidence_gate", "structured_vlm_only"},
    "object_existence": {"vlm_grounding_consensus"},
    "spatial_relation": {"grounded_geometry"},
}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def build_policy_record(
    source: dict[str, Any],
    policy_output: dict[str, Any],
    *,
    policy_name: str,
    policy_config: dict[str, Any],
) -> dict[str, Any]:
    """Attach task scoring to one offline policy output."""
    forced_answer = str(policy_output["forced_answer"])
    selective_answer = policy_output.get("selective_answer")
    return {
        "id": source["id"],
        "image": source.get("image"),
        "image_id": source["image_id"],
        "question": source["question"],
        "task_type": source["task_type"],
        "gt_answer": source["gt_answer"],
        "target_categories": source.get("target_categories", []),
        "query_plan": source.get("query_plan"),
        "policy_name": policy_name,
        "policy_config": policy_config,
        "answer_policy": policy_output,
        "evaluation": score_prediction(source, forced_answer),
        "selective_evaluation": (
            score_prediction(source, str(selective_answer))
            if selective_answer is not None
            else None
        ),
    }


def replay_grounded_policy(
    record: dict[str, Any],
    config: EvidencePolicyConfig,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Reapply an evidence gate to saved raw Grounded-SAM-2 annotations."""
    output = answer_with_evidence(
        record,
        record["query_plan"],
        record.get("annotations", []),
        image_width=image_width,
        image_height=image_height,
        config=config,
    )
    policy_config = {
        "min_grounding_score": config.min_grounding_score,
        "min_mask_score": config.min_mask_score,
        "min_mask_area_ratio": config.min_mask_area_ratio,
        "relation_margin": config.relation_margin,
    }
    return build_policy_record(
        record,
        output,
        policy_name="grounded_evidence_gate",
        policy_config=policy_config,
    )


def structured_listing_policy(record: dict[str, Any]) -> dict[str, Any]:
    """Use ontology-constrained VLM categories without a grounding gate."""
    categories = sorted({str(value) for value in record["query_plan"]["categories"]})
    answer = ", ".join(categories)
    supported = bool(categories)
    output = {
        "forced_answer": answer,
        "selective_answer": answer if supported else None,
        "abstained": not supported,
        "status": "structured_vlm_output" if supported else "empty_vlm_output",
        "claim_supported": supported,
        "support_type": "structured_vlm",
        "claim_count": len(categories),
        "unsupported_claim_count": 0,
        "confidence": None,
        "selected_evidence": [],
        "accepted_evidence": [],
        "rejected_evidence": [],
        "diagnostics": {},
    }
    return build_policy_record(
        record,
        output,
        policy_name="structured_vlm_only",
        policy_config={},
    )


def fuse_existence_consensus(
    detector_record: dict[str, Any],
    vlm_record: dict[str, Any],
) -> dict[str, Any]:
    """Use VLM for the forced answer and abstain on VLM/detector disagreement."""
    detector_output = detector_record["answer_policy"]
    detector_answer = parse_yes_no(str(detector_output["forced_answer"]))
    vlm_raw_answer = str(vlm_record.get("prediction", ""))
    vlm_answer = parse_yes_no(vlm_raw_answer)

    forced_answer = vlm_answer or detector_answer or "insufficient evidence"
    agreement = (
        vlm_answer is not None
        and detector_answer is not None
        and vlm_answer == detector_answer
    )
    if agreement:
        status = (
            "grounded_positive_agreement"
            if vlm_answer == "yes"
            else "cross_model_negative_agreement"
        )
    elif vlm_answer is None:
        status = "invalid_vlm_answer"
    else:
        status = "vlm_grounding_disagreement"

    output = {
        "forced_answer": forced_answer,
        "selective_answer": forced_answer if agreement else None,
        "abstained": not agreement,
        "status": status,
        "claim_supported": agreement,
        "support_type": (
            "localized_grounding_and_vlm"
            if agreement and forced_answer == "yes"
            else "cross_model_agreement"
            if agreement
            else "none"
        ),
        "claim_count": 1,
        "unsupported_claim_count": 0 if agreement else 1,
        "confidence": detector_output.get("confidence", 0.0),
        "selected_evidence": detector_output.get("selected_evidence", []),
        "accepted_evidence": detector_output.get("accepted_evidence", []),
        "rejected_evidence": detector_output.get("rejected_evidence", []),
        "diagnostics": {
            "vlm_raw_answer": vlm_raw_answer,
            "vlm_answer": vlm_answer,
            "detector_answer": detector_answer,
            "agreement": agreement,
        },
    }
    return build_policy_record(
        detector_record,
        output,
        policy_name="vlm_grounding_consensus",
        policy_config={
            "forced_answer_source": "vlm_with_detector_fallback",
            "selective_rule": "answer_only_on_binary_agreement",
            "detector_policy": detector_record["policy_config"],
        },
    )


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    forced = [record["evaluation"] for record in records]
    selective_records = [
        record for record in records if record["selective_evaluation"] is not None
    ]
    selective = [record["selective_evaluation"] for record in selective_records]
    result: dict[str, Any] = {
        "count": len(records),
        "forced_mean_score": _mean(float(item["score"]) for item in forced),
        "forced_exact_accuracy": _mean(
            float(item["is_correct"]) for item in forced
        ),
        "selective_answered": len(selective),
        "selective_abstained": len(records) - len(selective),
        "selective_coverage": round(len(selective) / len(records), 6)
        if records
        else 0.0,
        "selective_mean_score": _mean(
            float(item["score"]) for item in selective
        ),
        "selective_exact_accuracy": _mean(
            float(item["is_correct"]) for item in selective
        ),
        "status_counts": dict(
            sorted(Counter(record["answer_policy"]["status"] for record in records).items())
        ),
    }
    task_types = {record["task_type"] for record in records}
    if task_types == {"object_listing"}:
        result.update(
            {
                "forced_macro_precision": _mean(
                    float(item["precision"]) for item in forced
                ),
                "forced_macro_recall": _mean(
                    float(item["recall"]) for item in forced
                ),
                "forced_macro_f1": _mean(float(item["f1"]) for item in forced),
                "selective_macro_precision": _mean(
                    float(item["precision"]) for item in selective
                ),
                "selective_macro_recall": _mean(
                    float(item["recall"]) for item in selective
                ),
                "selective_macro_f1": _mean(
                    float(item["f1"]) for item in selective
                ),
            }
        )
    return result


def aggregate_policy_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate forced and selective metrics for one or several tasks."""
    records = list(records)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["task_type"])].append(record)
    return {
        "overall": _summary(records),
        "tasks": {
            task: _summary(task_records)
            for task, task_records in sorted(groups.items())
        },
    }


def validate_locked_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the Dev-selected policy before held-out application."""
    if payload.get("protocol") != LOCKED_POLICY_PROTOCOL:
        raise ValueError(
            "Unsupported locked policy protocol: "
            f"{payload.get('protocol')!r}."
        )
    if payload.get("development_split_only") is not True:
        raise ValueError("Locked policy must be selected on the development split.")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("Locked policy is missing its task map.")
    missing = set(LOCKED_POLICY_MODES) - set(tasks)
    extra = set(tasks) - set(LOCKED_POLICY_MODES)
    if missing or extra:
        raise ValueError(
            f"Locked policy task mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}."
        )
    for task_type, allowed_modes in LOCKED_POLICY_MODES.items():
        entry = tasks[task_type]
        if not isinstance(entry, dict):
            raise ValueError(f"Locked policy entry for {task_type} is invalid.")
        mode = entry.get("mode")
        if mode not in allowed_modes:
            raise ValueError(
                f"Unsupported {task_type} policy mode: {mode!r}."
            )
        config = entry.get("config", {})
        if not isinstance(config, dict):
            raise ValueError(f"Locked policy config for {task_type} is invalid.")
        if mode != "structured_vlm_only":
            EvidencePolicyConfig(**config)
        if not str(entry.get("candidate_id", "")):
            raise ValueError(f"Locked policy {task_type} has no candidate_id.")
    return payload


def apply_locked_policy(
    record: dict[str, Any],
    locked_policy: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    vlm_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one immutable task policy to saved or freshly generated evidence."""
    validate_locked_policy(locked_policy)
    task_type = str(record["task_type"])
    entry = locked_policy["tasks"][task_type]
    mode = str(entry["mode"])
    if mode == "structured_vlm_only":
        applied = structured_listing_policy(record)
    else:
        config = EvidencePolicyConfig(**entry["config"])
        applied = replay_grounded_policy(
            record,
            config,
            image_width=image_width,
            image_height=image_height,
        )
        if mode == "vlm_grounding_consensus":
            if vlm_record is None:
                raise ValueError(
                    "Existence consensus requires the matching VLM prediction."
                )
            applied = fuse_existence_consensus(applied, vlm_record)
    applied["policy_name"] = "locked_task_aware_evidence_fusion"
    applied["policy_config"] = {
        "protocol": locked_policy["protocol"],
        "candidate_id": entry["candidate_id"],
        "mode": mode,
        "config": entry.get("config", {}),
    }
    return applied
