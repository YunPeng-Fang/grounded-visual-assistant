"""Semantic crop verification for grounding-aware binary answers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from .evaluation import normalize_text, parse_yes_no
from .grounding_answer_verifier import (
    GroundingAnswerVerifierConfig,
    normalize_verification_evidence,
)


SEMANTIC_ANSWER_VERIFIER_PROTOCOL = "grounding_semantic_rescue_v2"
SEMANTIC_REVIEW_SYSTEM_PROMPT = (
    "You are a strict visual object verifier. The image is a detector crop. "
    "Answer exactly one word: Yes or No. Answer Yes only when the target "
    "object itself is clearly visible. Color, written words, background "
    "context, and a visually related but different object category are not "
    "sufficient evidence."
)


@dataclass(frozen=True)
class SemanticAnswerVerifierConfig:
    """Frozen candidate, geometry, and crop controls for V2."""

    evidence_score_threshold: float = 0.30
    min_mask_score: float | None = 0.50
    min_mask_area_ratio: float = 0.0
    max_mask_area_ratio: float = 0.90
    max_candidates_per_query: int = 2
    crop_padding_ratio: float = 0.25
    min_crop_size: int = 160
    require_exact_semantic_answer: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_score_threshold", self.evidence_score_threshold),
            ("min_mask_area_ratio", self.min_mask_area_ratio),
            ("max_mask_area_ratio", self.max_mask_area_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        if self.min_mask_score is not None and not (
            0.0 <= self.min_mask_score <= 1.0
        ):
            raise ValueError(
                "min_mask_score must be between 0 and 1 or None."
            )
        if self.max_mask_area_ratio <= self.min_mask_area_ratio:
            raise ValueError(
                "max_mask_area_ratio must exceed min_mask_area_ratio."
            )
        if self.max_candidates_per_query <= 0:
            raise ValueError("max_candidates_per_query must be positive.")
        if self.crop_padding_ratio < 0.0:
            raise ValueError("crop_padding_ratio must be non-negative.")
        if self.min_crop_size <= 0:
            raise ValueError("min_crop_size must be positive.")


def semantic_review_question(target: str) -> str:
    """Build the deterministic candidate-crop question."""
    normalized = normalize_text(target)
    if not normalized:
        raise ValueError("Semantic review target must not be empty.")
    return (
        f'Target object: "{normalized}". Is at least one instance of this '
        "exact object category clearly visible in the crop? Answer exactly "
        "Yes or No."
    )


def semantic_candidate_key(query_key: str, annotation_index: int) -> str:
    """Create a stable key for a query/evidence pair."""
    if not query_key:
        raise ValueError("query_key must not be empty.")
    if annotation_index < 0:
        raise ValueError("annotation_index must be non-negative.")
    return f"{query_key}__annotation-{annotation_index:03d}"


def select_semantic_candidates(
    annotations: Iterable[Mapping[str, Any]],
    *,
    target: str,
    image_width: int,
    image_height: int,
    config: SemanticAnswerVerifierConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply evidence, geometry, and candidate-count gates."""
    evidence_config = GroundingAnswerVerifierConfig(
        evidence_score_threshold=config.evidence_score_threshold,
        promotion_score_threshold=config.evidence_score_threshold,
        min_mask_score=config.min_mask_score,
        min_mask_area_ratio=config.min_mask_area_ratio,
    )
    accepted, rejected = normalize_verification_evidence(
        annotations,
        target=target,
        image_width=image_width,
        image_height=image_height,
        config=evidence_config,
    )
    geometry_accepted = []
    for item in accepted:
        if float(item["mask_area_ratio"]) > config.max_mask_area_ratio:
            rejected.append(
                {
                    **item,
                    "rejection_reasons": ["large_mask"],
                }
            )
        else:
            geometry_accepted.append(item)

    ordered = sorted(
        geometry_accepted,
        key=lambda item: (
            -float(item["grounding_score"]),
            -float(item["estimated_area"]),
            int(item["annotation_index"]),
        ),
    )
    selected = ordered[: config.max_candidates_per_query]
    for item in ordered[config.max_candidates_per_query :]:
        rejected.append(
            {
                **item,
                "rejection_reasons": ["candidate_limit"],
            }
        )
    return selected, rejected


def context_crop_box(
    box: Iterable[float],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
    min_crop_size: int,
) -> list[int]:
    """Expand a detector box into a clamped square context crop."""
    values = [float(value) for value in box]
    if len(values) != 4:
        raise ValueError(f"Expected xyxy box with four values, found {values}.")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Detector box has non-positive area: {values}.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    width = x2 - x1
    height = y2 - y1
    side = max(
        max(width, height) * (1.0 + 2.0 * padding_ratio),
        float(min_crop_size),
    )
    side = min(side, float(image_width), float(image_height))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = max(0.0, min(center_x - side / 2.0, image_width - side))
    top = max(0.0, min(center_y - side / 2.0, image_height - side))
    right = min(float(image_width), left + side)
    bottom = min(float(image_height), top + side)
    return [
        int(round(left)),
        int(round(top)),
        int(round(right)),
        int(round(bottom)),
    ]


def write_semantic_crop(
    image_path: str | Path,
    output_path: str | Path,
    *,
    box: Iterable[float],
    config: SemanticAnswerVerifierConfig,
) -> dict[str, Any]:
    """Write one deterministic RGB candidate crop."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        crop_box = context_crop_box(
            box,
            image_width=image.width,
            image_height=image.height,
            padding_ratio=config.crop_padding_ratio,
            min_crop_size=config.min_crop_size,
        )
        crop = image.crop(tuple(crop_box))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(
            output_path,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
        )
    return {
        "source_image": str(image_path),
        "crop_image": str(output_path),
        "source_box_xyxy": [round(float(value), 3) for value in box],
        "crop_box_xyxy": crop_box,
        "crop_width": crop_box[2] - crop_box[0],
        "crop_height": crop_box[3] - crop_box[1],
    }


def normalize_semantic_review(review: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one saved semantic-review answer."""
    answer = str(review.get("answer", ""))
    parsed = parse_yes_no(answer)
    exact = normalize_text(answer) in {"yes", "no"}
    return {
        **dict(review),
        "parsed_answer": parsed,
        "exact_answer": exact,
        "valid": parsed is not None,
    }


def verify_binary_answer_v2(
    baseline_answer: str,
    *,
    target: str,
    annotations: Iterable[Mapping[str, Any]],
    image_width: int,
    image_height: int,
    semantic_reviews: Iterable[Mapping[str, Any]],
    config: SemanticAnswerVerifierConfig,
) -> dict[str, Any]:
    """Promote a negative answer only after strict crop-level confirmation."""
    parsed_baseline = parse_yes_no(str(baseline_answer))
    if parsed_baseline is None:
        raise ValueError(
            "Semantic verification requires an unambiguous Yes/No baseline "
            f"answer, found {baseline_answer!r}."
        )
    candidates, rejected = select_semantic_candidates(
        annotations,
        target=target,
        image_width=image_width,
        image_height=image_height,
        config=config,
    )
    normalized_reviews = [
        normalize_semantic_review(item) for item in semantic_reviews
    ]
    reviews_by_index = {
        int(item["annotation_index"]): item for item in normalized_reviews
    }
    if len(reviews_by_index) != len(normalized_reviews):
        raise ValueError("Semantic reviews contain duplicate annotation indices.")

    final_answer = parsed_baseline
    changed = False
    correction_direction = None
    selected_evidence = candidates[0] if candidates else None
    selected_review = None

    if parsed_baseline == "yes":
        status = "positive_preserved_without_negative_recheck"
    elif not candidates:
        rejection_reasons = {
            reason
            for item in rejected
            for reason in item.get("rejection_reasons", [])
        }
        status = (
            "negative_preserved_by_geometry_gate"
            if "large_mask" in rejection_reasons
            else "negative_preserved_without_evidence"
        )
    else:
        missing = [
            int(item["annotation_index"])
            for item in candidates
            if int(item["annotation_index"]) not in reviews_by_index
        ]
        if missing:
            raise ValueError(
                f"Semantic reviews are missing candidate indices: {missing}."
            )
        candidate_reviews = [
            reviews_by_index[int(item["annotation_index"])]
            for item in candidates
        ]
        confirmed = [
            (candidate, review)
            for candidate, review in zip(candidates, candidate_reviews)
            if review["parsed_answer"] == "yes"
            and (
                review["exact_answer"]
                or not config.require_exact_semantic_answer
            )
        ]
        if confirmed:
            selected_evidence, selected_review = confirmed[0]
            final_answer = "yes"
            changed = True
            correction_direction = "no_to_yes"
            status = "promoted_by_semantic_confirmation"
        elif any(not item["valid"] for item in candidate_reviews):
            status = "negative_preserved_invalid_semantic_review"
        elif (
            config.require_exact_semantic_answer
            and any(not item["exact_answer"] for item in candidate_reviews)
        ):
            status = "negative_preserved_non_exact_semantic_review"
        else:
            status = "negative_preserved_by_semantic_rejection"

    return {
        "protocol": SEMANTIC_ANSWER_VERIFIER_PROTOCOL,
        "target": normalize_text(target),
        "baseline_answer": parsed_baseline,
        "final_answer": final_answer,
        "changed": changed,
        "correction_direction": correction_direction,
        "status": status,
        "selected_evidence": selected_evidence,
        "selected_semantic_review": selected_review,
        "semantic_candidates": candidates,
        "semantic_reviews": normalized_reviews,
        "rejected_evidence": rejected,
        "policy": {
            "evidence_score_threshold": config.evidence_score_threshold,
            "min_mask_score": config.min_mask_score,
            "min_mask_area_ratio": config.min_mask_area_ratio,
            "max_mask_area_ratio": config.max_mask_area_ratio,
            "max_candidates_per_query": config.max_candidates_per_query,
            "crop_padding_ratio": config.crop_padding_ratio,
            "min_crop_size": config.min_crop_size,
            "require_exact_semantic_answer": (
                config.require_exact_semantic_answer
            ),
            "negative_evidence_rule": (
                "detector silence never demotes a positive baseline answer"
            ),
        },
    }
