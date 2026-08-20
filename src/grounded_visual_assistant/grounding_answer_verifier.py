"""Grounding-aware verification for binary visual answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .evaluation import normalize_text, parse_yes_no
from .grounding_evaluation import canonicalize_category


GROUNDING_ANSWER_VERIFIER_PROTOCOL = "grounding_positive_rescue_v1"


@dataclass(frozen=True)
class GroundingAnswerVerifierConfig:
    """Frozen evidence thresholds for the positive-rescue policy."""

    evidence_score_threshold: float = 0.30
    promotion_score_threshold: float = 0.45
    min_mask_score: float | None = None
    min_mask_area_ratio: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_score_threshold", self.evidence_score_threshold),
            ("promotion_score_threshold", self.promotion_score_threshold),
            ("min_mask_area_ratio", self.min_mask_area_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        if self.promotion_score_threshold < self.evidence_score_threshold:
            raise ValueError(
                "promotion_score_threshold cannot be lower than "
                "evidence_score_threshold."
            )
        if self.min_mask_score is not None and not (
            0.0 <= self.min_mask_score <= 1.0
        ):
            raise ValueError(
                "min_mask_score must be between 0 and 1 or None."
            )


def _normalize_target(value: str) -> str:
    target = normalize_text(value)
    if not target:
        raise ValueError("Verification target must not be empty.")
    return target


def _box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def normalize_verification_evidence(
    annotations: Iterable[Mapping[str, Any]],
    *,
    target: str,
    image_width: int,
    image_height: int,
    config: GroundingAnswerVerifierConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize single-target evidence and apply the verifier evidence gate."""
    target = _normalize_target(target)
    image_area = max(int(image_width) * int(image_height), 1)
    accepted = []
    rejected = []
    for index, raw_annotation in enumerate(annotations):
        annotation = dict(raw_annotation)
        box = annotation.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(
                f"Grounding annotation {index} has an invalid bbox: {box}"
            )
        numeric_box = [float(value) for value in box]
        raw_label = str(annotation.get("class_name", "unknown"))
        mapped_label = canonicalize_category(raw_label, (target,))
        mapping = "canonical_label"
        if normalize_text(mapped_label) != target:
            # Grounding DINO was queried with one phrase, so generic or
            # composite processor labels still refer to that phrase.
            mapped_label = target
            mapping = "single_query_fallback"

        grounding_score = float(annotation.get("score", 0.0))
        raw_mask_score = annotation.get("mask_score")
        mask_score = (
            float(raw_mask_score) if raw_mask_score is not None else None
        )
        mask_area = int(annotation.get("mask_area", 0))
        mask_area_ratio = mask_area / image_area
        rejection_reasons = []
        if grounding_score < config.evidence_score_threshold:
            rejection_reasons.append("low_grounding_score")
        if (
            config.min_mask_score is not None
            and (mask_score is None or mask_score < config.min_mask_score)
        ):
            rejection_reasons.append("low_mask_score")
        if mask_area_ratio < config.min_mask_area_ratio:
            rejection_reasons.append("small_mask")

        evidence = {
            "annotation_index": index,
            "target": target,
            "raw_label": raw_label,
            "mapped_label": mapped_label,
            "label_mapping": mapping,
            "bbox": numeric_box,
            "grounding_score": round(grounding_score, 6),
            "mask_score": (
                round(mask_score, 6) if mask_score is not None else None
            ),
            "mask_area": mask_area,
            "mask_area_ratio": round(mask_area_ratio, 8),
            "estimated_area": round(
                float(mask_area) if mask_area > 0 else _box_area(numeric_box),
                3,
            ),
        }
        if rejection_reasons:
            rejected.append(
                {**evidence, "rejection_reasons": rejection_reasons}
            )
        else:
            accepted.append(evidence)
    return accepted, rejected


def _strongest_evidence(
    evidence: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    evidence = list(evidence)
    if not evidence:
        return None
    return max(
        evidence,
        key=lambda item: (
            float(item["grounding_score"]),
            float(item["estimated_area"]),
            -int(item["annotation_index"]),
        ),
    )


def verify_binary_answer(
    baseline_answer: str,
    *,
    target: str,
    annotations: Iterable[Mapping[str, Any]],
    image_width: int,
    image_height: int,
    config: GroundingAnswerVerifierConfig,
) -> dict[str, Any]:
    """Apply asymmetric positive rescue to one baseline Yes/No answer."""
    parsed_baseline = parse_yes_no(str(baseline_answer))
    if parsed_baseline is None:
        raise ValueError(
            "Grounding verification requires an unambiguous Yes/No baseline "
            f"answer, found {baseline_answer!r}."
        )
    target = _normalize_target(target)
    accepted, rejected = normalize_verification_evidence(
        annotations,
        target=target,
        image_width=image_width,
        image_height=image_height,
        config=config,
    )
    strongest = _strongest_evidence(accepted)
    promotable = [
        item
        for item in accepted
        if float(item["grounding_score"])
        >= config.promotion_score_threshold
    ]
    strongest_promotion = _strongest_evidence(promotable)

    final_answer = parsed_baseline
    changed = False
    correction_direction = None
    if parsed_baseline == "no" and strongest_promotion is not None:
        final_answer = "yes"
        changed = True
        correction_direction = "no_to_yes"
        status = "promoted_by_localized_evidence"
        selected = strongest_promotion
    elif parsed_baseline == "no" and strongest is not None:
        status = "negative_preserved_below_promotion_threshold"
        selected = strongest
    elif parsed_baseline == "no":
        status = "negative_preserved_without_evidence"
        selected = None
    elif strongest is not None:
        status = "positive_supported_by_localized_evidence"
        selected = strongest
    else:
        # Detector silence is not proof of absence, so V1 never demotes a
        # positive VLM answer.
        status = "positive_preserved_without_evidence"
        selected = None

    strongest_score = (
        float(strongest["grounding_score"]) if strongest is not None else 0.0
    )
    evidence_level = (
        "promotion"
        if strongest_score >= config.promotion_score_threshold
        else "accepted"
        if strongest is not None
        else "none"
    )
    return {
        "protocol": GROUNDING_ANSWER_VERIFIER_PROTOCOL,
        "target": target,
        "baseline_answer": parsed_baseline,
        "final_answer": final_answer,
        "changed": changed,
        "correction_direction": correction_direction,
        "status": status,
        "evidence_level": evidence_level,
        "accepted_evidence_count": len(accepted),
        "rejected_evidence_count": len(rejected),
        "strongest_grounding_score": round(strongest_score, 6),
        "selected_evidence": selected,
        "accepted_evidence": accepted,
        "rejected_evidence": rejected,
        "policy": {
            "evidence_score_threshold": config.evidence_score_threshold,
            "promotion_score_threshold": config.promotion_score_threshold,
            "min_mask_score": config.min_mask_score,
            "min_mask_area_ratio": config.min_mask_area_ratio,
            "negative_evidence_rule": (
                "detector silence never demotes a positive baseline answer"
            ),
        },
    }


def compact_grounding_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove mask RLE payloads while retaining verifier diagnostics."""
    annotations = []
    for raw_item in result.get("annotations", []):
        item = dict(raw_item)
        item.pop("segmentation", None)
        annotations.append(item)
    return {
        key: value
        for key, value in result.items()
        if key != "annotations"
    } | {"annotations": annotations}
