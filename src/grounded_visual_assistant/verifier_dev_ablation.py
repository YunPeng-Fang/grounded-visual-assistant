"""Offline policy ablations for the frozen Verifier Dev protocol."""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .evaluation import parse_yes_no
from .grounding_answer_verifier import (
    GroundingAnswerVerifierConfig,
    normalize_verification_evidence,
)
from .pope_evaluation import binary_metrics, evaluate_answer
from .pope_verifier_evaluation import exact_mcnemar_p_value
from .semantic_answer_verifier import (
    SemanticAnswerVerifierConfig,
    normalize_semantic_review,
    select_semantic_candidates,
    semantic_candidate_key,
)


VERIFIER_DEV_ABLATION_PROTOCOL = "verifier_dev_offline_ablation_v1"


@dataclass(frozen=True)
class VerifierDevPolicy:
    """One deterministic policy in the frozen Dev ablation grid."""

    policy_id: str
    family: str
    module: str
    score_threshold: float | None
    min_mask_score: float | None
    min_mask_area_ratio: float
    max_mask_area_ratio: float
    max_candidates_per_query: int | None
    semantic_gate: bool

    def __post_init__(self) -> None:
        if self.module not in {"baseline", "grounding", "semantic"}:
            raise ValueError(f"Unsupported policy module: {self.module}")
        if self.module == "baseline":
            if self.score_threshold is not None:
                raise ValueError("Baseline policy cannot have a score gate.")
            return
        if self.score_threshold is None or not (
            0.0 <= self.score_threshold <= 1.0
        ):
            raise ValueError("Verifier score threshold must be in [0, 1].")
        if not 0.0 <= self.min_mask_area_ratio <= 1.0:
            raise ValueError("Minimum mask-area ratio must be in [0, 1].")
        if not 0.0 < self.max_mask_area_ratio <= 1.0:
            raise ValueError("Maximum mask-area ratio must be in (0, 1].")
        if self.max_mask_area_ratio <= self.min_mask_area_ratio:
            raise ValueError("Maximum mask-area ratio must exceed minimum.")
        if self.min_mask_score is not None and not (
            0.0 <= self.min_mask_score <= 1.0
        ):
            raise ValueError("Minimum mask score must be in [0, 1].")
        if self.module == "semantic":
            if not self.semantic_gate:
                raise ValueError("Semantic policies must enable the gate.")
            if (
                self.max_candidates_per_query is None
                or self.max_candidates_per_query <= 0
            ):
                raise ValueError(
                    "Semantic policies need a positive candidate limit."
                )


def _threshold_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_policy_grid(
    *,
    score_thresholds: Iterable[float],
    templates: Iterable[Mapping[str, Any]],
    min_mask_score: float | None,
    min_mask_area_ratio: float,
) -> list[VerifierDevPolicy]:
    """Expand pre-registered policy templates in stable order."""
    policies = [
        VerifierDevPolicy(
            policy_id="baseline",
            family="baseline",
            module="baseline",
            score_threshold=None,
            min_mask_score=None,
            min_mask_area_ratio=0.0,
            max_mask_area_ratio=1.0,
            max_candidates_per_query=None,
            semantic_gate=False,
        )
    ]
    thresholds = [float(value) for value in score_thresholds]
    if not thresholds or thresholds != sorted(set(thresholds)):
        raise ValueError(
            "Score thresholds must be non-empty, unique, and ascending."
        )
    for raw_template in templates:
        template = dict(raw_template)
        family = str(template["family"])
        module = str(template["module"])
        semantic_gate = bool(template.get("semantic_gate", False))
        max_area = float(template.get("max_mask_area_ratio", 1.0))
        candidate_limit = template.get("max_candidates_per_query")
        if candidate_limit is not None:
            candidate_limit = int(candidate_limit)
        for threshold in thresholds:
            policies.append(
                VerifierDevPolicy(
                    policy_id=(
                        f"{family}__score-{_threshold_slug(threshold)}"
                    ),
                    family=family,
                    module=module,
                    score_threshold=threshold,
                    min_mask_score=min_mask_score,
                    min_mask_area_ratio=min_mask_area_ratio,
                    max_mask_area_ratio=max_area,
                    max_candidates_per_query=candidate_limit,
                    semantic_gate=semantic_gate,
                )
            )
    policy_ids = [item.policy_id for item in policies]
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError("Ablation policy IDs must be unique.")
    return policies


def ordered_policy_ids_sha256(
    policies: Iterable[VerifierDevPolicy],
) -> str:
    payload = "\n".join(item.policy_id for item in policies) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def _validate_baseline_evaluation(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = evaluate_answer(
        str(record["prediction"]), str(record["gt_answer"])
    )
    if evaluation != record.get("evaluation"):
        raise RuntimeError(
            f"Baseline evaluation mismatch for {record.get('id')}."
        )
    if parse_yes_no(str(record["prediction"])) is None:
        raise ValueError(
            f"Baseline answer is not strict Yes/No: {record.get('id')}."
        )
    return evaluation


def _grounding_candidates(
    evidence: Mapping[str, Any],
    *,
    policy: VerifierDevPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grounding = evidence["grounding"]
    config = GroundingAnswerVerifierConfig(
        evidence_score_threshold=float(policy.score_threshold),
        promotion_score_threshold=float(policy.score_threshold),
        min_mask_score=policy.min_mask_score,
        min_mask_area_ratio=policy.min_mask_area_ratio,
    )
    accepted, rejected = normalize_verification_evidence(
        grounding.get("annotations", []),
        target=str(evidence["object"]),
        image_width=int(grounding["img_width"]),
        image_height=int(grounding["img_height"]),
        config=config,
    )
    kept = []
    for item in accepted:
        if float(item["mask_area_ratio"]) > policy.max_mask_area_ratio:
            rejected.append(
                {**item, "rejection_reasons": ["large_mask"]}
            )
        else:
            kept.append(item)
    kept.sort(
        key=lambda item: (
            -float(item["grounding_score"]),
            -float(item["estimated_area"]),
            int(item["annotation_index"]),
        )
    )
    return kept, rejected


def _semantic_candidates(
    evidence: Mapping[str, Any],
    *,
    policy: VerifierDevPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grounding = evidence["grounding"]
    config = SemanticAnswerVerifierConfig(
        evidence_score_threshold=float(policy.score_threshold),
        min_mask_score=policy.min_mask_score,
        min_mask_area_ratio=policy.min_mask_area_ratio,
        max_mask_area_ratio=policy.max_mask_area_ratio,
        max_candidates_per_query=int(policy.max_candidates_per_query),
    )
    return select_semantic_candidates(
        grounding.get("annotations", []),
        target=str(evidence["object"]),
        image_width=int(grounding["img_width"]),
        image_height=int(grounding["img_height"]),
        config=config,
    )


def _semantic_confirmation(
    evidence: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
    *,
    reviews_by_key: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, list[str]]:
    candidate_keys = [
        semantic_candidate_key(
            str(evidence["query_key"]), int(item["annotation_index"])
        )
        for item in candidates
    ]
    missing = [key for key in candidate_keys if key not in reviews_by_key]
    if missing:
        raise ValueError(
            f"Semantic reviews are missing for {evidence['baseline_id']}: "
            f"{missing}."
        )
    for key in candidate_keys:
        review = normalize_semantic_review(reviews_by_key[key])
        raw_exact = str(review.get("answer", "")).strip().lower() in {
            "yes",
            "no",
        }
        if review["parsed_answer"] == "yes" and raw_exact:
            return key, candidate_keys
    return None, candidate_keys


def evaluate_dev_policy(
    policy: VerifierDevPolicy,
    *,
    baseline_records: Iterable[Mapping[str, Any]],
    evidence_by_baseline_id: Mapping[str, Mapping[str, Any]],
    reviews_by_key: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize one policy over the same frozen Dev110 predictions."""
    predictions = []
    selected_review_keys: set[str] = set()
    for raw_record in baseline_records:
        baseline = dict(raw_record)
        baseline_id = str(baseline["id"])
        baseline_evaluation = _validate_baseline_evaluation(baseline)
        baseline_answer = parse_yes_no(str(baseline["prediction"]))
        final_answer = str(baseline_answer)
        status = "baseline_unchanged"
        selected_candidate_keys: list[str] = []
        confirmed_candidate_key = None

        if policy.module != "baseline" and baseline_answer == "no":
            if baseline_id not in evidence_by_baseline_id:
                raise ValueError(
                    f"Grounding evidence is missing for {baseline_id}."
                )
            evidence = evidence_by_baseline_id[baseline_id]
            if policy.module == "grounding":
                candidates, _ = _grounding_candidates(
                    evidence, policy=policy
                )
                selected_candidate_keys = [
                    semantic_candidate_key(
                        str(evidence["query_key"]),
                        int(item["annotation_index"]),
                    )
                    for item in candidates
                ]
                if candidates:
                    final_answer = "yes"
                    status = "promoted_by_grounding"
                else:
                    status = "negative_preserved_by_grounding"
            else:
                candidates, _ = _semantic_candidates(
                    evidence, policy=policy
                )
                (
                    confirmed_candidate_key,
                    selected_candidate_keys,
                ) = _semantic_confirmation(
                    evidence,
                    candidates,
                    reviews_by_key=reviews_by_key,
                )
                selected_review_keys.update(selected_candidate_keys)
                if confirmed_candidate_key is not None:
                    final_answer = "yes"
                    status = "promoted_by_semantic_confirmation"
                elif candidates:
                    status = "negative_preserved_by_semantic_rejection"
                else:
                    status = "negative_preserved_without_candidate"
        elif policy.module != "baseline":
            status = "positive_preserved_without_negative_recheck"

        evaluation = evaluate_answer(
            final_answer, str(baseline["gt_answer"])
        )
        baseline_correct = bool(baseline_evaluation["is_correct"])
        final_correct = bool(evaluation["is_correct"])
        changed = final_answer != baseline_answer
        correction = (
            "beneficial"
            if changed and not baseline_correct and final_correct
            else "harmful"
            if changed and baseline_correct and not final_correct
            else "neutral"
            if changed
            else "unchanged"
        )
        predictions.append(
            {
                "id": baseline_id,
                "pair_id": baseline["pair_id"],
                "pair_role": baseline["pair_role"],
                "image_id": baseline["image_id"],
                "image": baseline["image"],
                "question": baseline["question"],
                "object": baseline["object"],
                "gt_answer": baseline["gt_answer"],
                "policy_id": policy.policy_id,
                "baseline_prediction": baseline_answer,
                "baseline_evaluation": baseline_evaluation,
                "prediction": final_answer,
                "evaluation": evaluation,
                "changed": changed,
                "correction": correction,
                "status": status,
                "selected_candidate_keys": selected_candidate_keys,
                "confirmed_candidate_key": confirmed_candidate_key,
            }
        )

    summary = summarize_dev_policy(
        policy,
        predictions=predictions,
        selected_review_keys=selected_review_keys,
        evidence_by_baseline_id=evidence_by_baseline_id,
        reviews_by_key=reviews_by_key,
        baseline_records=baseline_records,
    )
    return predictions, summary


def summarize_dev_policy(
    policy: VerifierDevPolicy,
    *,
    predictions: Iterable[Mapping[str, Any]],
    selected_review_keys: Iterable[str],
    evidence_by_baseline_id: Mapping[str, Mapping[str, Any]],
    reviews_by_key: Mapping[str, Mapping[str, Any]],
    baseline_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    records = [dict(item) for item in predictions]
    baseline_records = [dict(item) for item in baseline_records]
    metrics = binary_metrics(records)
    baseline_metrics = binary_metrics(
        [
            {"evaluation": item["baseline_evaluation"]}
            for item in records
        ]
    )
    changed = [item for item in records if item["changed"]]
    beneficial = sum(
        item["correction"] == "beneficial" for item in changed
    )
    harmful = sum(item["correction"] == "harmful" for item in changed)
    selected_review_keys = sorted(set(selected_review_keys))

    baseline_latency = sum(
        float(item.get("latency_seconds", 0.0))
        for item in baseline_records
    )
    grounding_latency = 0.0
    grounding_peak = 0.0
    if policy.module != "baseline":
        for item in evidence_by_baseline_id.values():
            grounding = item["grounding"]
            grounding_latency += float(
                (grounding.get("latency_seconds") or {}).get("total", 0.0)
            )
            grounding_peak = max(
                grounding_peak,
                float(
                    item.get(
                        "cuda_peak_memory_allocated_gb",
                        grounding.get(
                            "cuda_peak_memory_allocated_gb", 0.0
                        ),
                    )
                    or 0.0
                ),
            )
    semantic_latency = sum(
        float(
            reviews_by_key[key].get(
                "end_to_end_latency_seconds",
                reviews_by_key[key].get("latency_seconds", 0.0),
            )
        )
        for key in selected_review_keys
    )
    semantic_peak = max(
        (
            float(reviews_by_key[key]["cuda_peak_memory_allocated_gb"])
            for key in selected_review_keys
            if reviews_by_key[key].get(
                "cuda_peak_memory_allocated_gb"
            )
            is not None
        ),
        default=0.0,
    )
    baseline_peak = max(
        (
            float(item["cuda_peak_memory_allocated_gb"])
            for item in baseline_records
            if item.get("cuda_peak_memory_allocated_gb") is not None
        ),
        default=0.0,
    )
    role_metrics = {}
    for role in ("positive", "hard_negative"):
        role_items = [
            item for item in records if item["pair_role"] == role
        ]
        role_metrics[role] = binary_metrics(role_items)

    return {
        "policy": asdict(policy),
        "metrics": metrics,
        "delta": {
            key: round(
                float(metrics[key]) - float(baseline_metrics[key]), 6
            )
            for key in ("accuracy", "precision", "recall", "f1", "yes_ratio")
        },
        "pair_roles": role_metrics,
        "corrections": {
            "changed_answers": len(changed),
            "beneficial": beneficial,
            "harmful": harmful,
            "net_correct": beneficial - harmful,
            "changed_ids": [str(item["id"]) for item in changed],
            "mcnemar_exact_two_sided_p_value": exact_mcnemar_p_value(
                harmful, beneficial
            ),
        },
        "runtime_projection": {
            "grounding_queries": (
                len(evidence_by_baseline_id)
                if policy.module != "baseline"
                else 0
            ),
            "semantic_candidate_reviews": len(selected_review_keys),
            "grounding_latency_seconds": round(grounding_latency, 6),
            "semantic_latency_seconds": round(semantic_latency, 6),
            "incremental_latency_seconds": round(
                grounding_latency + semantic_latency, 6
            ),
            "incremental_mean_per_question": (
                round(
                    (grounding_latency + semantic_latency) / len(records),
                    6,
                )
                if records
                else 0.0
            ),
            "projected_end_to_end_total": round(
                baseline_latency + grounding_latency + semantic_latency, 6
            ),
            "sequential_peak_memory_gb": round(
                max(baseline_peak, grounding_peak, semantic_peak), 6
            ),
            "review_key_sha256": hashlib.sha256(
                (
                    "\n".join(selected_review_keys)
                    + ("\n" if selected_review_keys else "")
                ).encode("utf-8")
            ).hexdigest(),
        },
    }


def select_dev_policy(
    summaries: Iterable[Mapping[str, Any]],
    *,
    require_strict_accuracy_improvement: bool,
    require_non_decreasing_f1: bool,
    require_positive_net_corrections: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply pre-registered gates and deterministically lock one policy."""
    rows = [dict(item) for item in summaries]
    baseline = next(
        item
        for item in rows
        if item["policy"]["policy_id"] == "baseline"
    )
    baseline_metrics = baseline["metrics"]
    eligible = []
    for item in rows:
        policy_id = str(item["policy"]["policy_id"])
        reasons = []
        if policy_id == "baseline":
            reasons.append("reference_policy")
        else:
            if (
                require_strict_accuracy_improvement
                and float(item["metrics"]["accuracy"])
                <= float(baseline_metrics["accuracy"])
            ):
                reasons.append("no_strict_accuracy_improvement")
            if (
                require_non_decreasing_f1
                and float(item["metrics"]["f1"])
                < float(baseline_metrics["f1"])
            ):
                reasons.append("f1_decreased")
            if (
                require_positive_net_corrections
                and int(item["corrections"]["net_correct"]) <= 0
            ):
                reasons.append("non_positive_net_corrections")
        item["selection"] = {
            "eligible": not reasons,
            "rejection_reasons": reasons,
        }
        if not reasons:
            eligible.append(item)

    if eligible:
        selected = sorted(
            eligible,
            key=lambda item: (
                -float(item["metrics"]["accuracy"]),
                -float(item["metrics"]["f1"]),
                -float(item["metrics"]["precision"]),
                int(item["corrections"]["harmful"]),
                float(
                    item["runtime_projection"][
                        "incremental_latency_seconds"
                    ]
                ),
                str(item["policy"]["policy_id"]),
            ),
        )[0]
        decision = "lock_dev_selected_verifier"
    else:
        selected = baseline
        decision = "retain_baseline_no_eligible_verifier"
    return rows, {
        "decision": decision,
        "selected_policy_id": selected["policy"]["policy_id"],
        "eligible_policy_ids": [
            item["policy"]["policy_id"] for item in eligible
        ],
        "selection_gates": {
            "require_strict_accuracy_improvement": (
                require_strict_accuracy_improvement
            ),
            "require_non_decreasing_f1": require_non_decreasing_f1,
            "require_positive_net_corrections": (
                require_positive_net_corrections
            ),
        },
        "tie_breakers": [
            "accuracy_desc",
            "f1_desc",
            "precision_desc",
            "harmful_corrections_asc",
            "incremental_latency_asc",
            "policy_id_asc",
        ],
        "selected_metrics": selected["metrics"],
        "selected_corrections": selected["corrections"],
    }


def flatten_policy_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    """Create a stable flat row for JSONL/CSV comparison tables."""
    policy = item["policy"]
    metrics = item["metrics"]
    confusion = metrics["confusion"]
    corrections = item["corrections"]
    runtime = item["runtime_projection"]
    selection = item.get("selection") or {}
    return {
        "policy_id": policy["policy_id"],
        "family": policy["family"],
        "module": policy["module"],
        "score_threshold": policy["score_threshold"],
        "max_mask_area_ratio": policy["max_mask_area_ratio"],
        "max_candidates_per_query": policy["max_candidates_per_query"],
        "semantic_gate": policy["semantic_gate"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "yes_ratio": metrics["yes_ratio"],
        "tp": confusion["tp"],
        "fp": confusion["fp"],
        "tn": confusion["tn"],
        "fn": confusion["fn"],
        "accuracy_delta": item["delta"]["accuracy"],
        "f1_delta": item["delta"]["f1"],
        "changed_answers": corrections["changed_answers"],
        "beneficial": corrections["beneficial"],
        "harmful": corrections["harmful"],
        "net_correct": corrections["net_correct"],
        "mcnemar_p": corrections["mcnemar_exact_two_sided_p_value"],
        "semantic_reviews": runtime["semantic_candidate_reviews"],
        "incremental_latency_seconds": runtime[
            "incremental_latency_seconds"
        ],
        "eligible": selection.get("eligible", False),
        "rejection_reasons": ";".join(
            selection.get("rejection_reasons", [])
        ),
    }
