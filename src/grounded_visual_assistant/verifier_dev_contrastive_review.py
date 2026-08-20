"""Contrastive category review for the Verifier Dev V3 cascade."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw

from .evaluation import COCO_CATEGORIES, normalize_text, parse_yes_no
from .pope_evaluation import binary_metrics, evaluate_answer
from .pope_verifier_evaluation import exact_mcnemar_p_value


VERIFIER_DEV_CONTRASTIVE_REVIEW_PROTOCOL = (
    "verifier_dev_contrastive_review_v3"
)
COCO_ONTOLOGY_PROTOCOL = "coco_80_supercategory_ontology_v1"
CONTRASTIVE_REVIEW_SYSTEM_PROMPT = (
    "You are a strict object classifier. A red rectangle marks the detector "
    "candidate to classify. Choose exactly one label from the allowed list. "
    "Classify the object inside the rectangle, not nearby context. If the "
    "region is unclear or no listed category applies, answer none. Output "
    "only the chosen label with no explanation or punctuation."
)


def validate_coco_ontology(
    payload: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Validate a standalone COCO-80 supercategory ontology."""
    if payload.get("protocol") != COCO_ONTOLOGY_PROTOCOL:
        raise ValueError(
            f"Unsupported COCO ontology protocol: {payload.get('protocol')}"
        )
    raw_groups = payload.get("supercategories")
    if not isinstance(raw_groups, Mapping) or not raw_groups:
        raise ValueError("COCO ontology has no supercategory mapping.")
    groups = {}
    category_to_group = {}
    for raw_name, raw_categories in raw_groups.items():
        group = normalize_text(str(raw_name))
        if not group or not isinstance(raw_categories, list):
            raise ValueError(f"Invalid COCO supercategory: {raw_name!r}.")
        categories = tuple(
            normalize_text(str(item)) for item in raw_categories
        )
        if not categories or len(categories) != len(set(categories)):
            raise ValueError(
                f"COCO supercategory {group!r} is empty or duplicated."
            )
        groups[group] = categories
        for category in categories:
            if category in category_to_group:
                raise ValueError(
                    f"COCO category appears in multiple groups: {category}."
                )
            category_to_group[category] = group
    expected = set(COCO_CATEGORIES)
    observed = set(category_to_group)
    if observed != expected:
        raise ValueError(
            "COCO ontology categories mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}."
        )
    return groups, category_to_group


def contrastive_options(
    target: str,
    *,
    groups: Mapping[str, Iterable[str]],
    category_to_group: Mapping[str, str],
    none_label: str = "none",
) -> tuple[str, ...]:
    """Return stable same-supercategory options plus an abstention label."""
    target = normalize_text(target)
    none_label = normalize_text(none_label)
    if target not in category_to_group:
        raise ValueError(f"Target is absent from COCO ontology: {target}.")
    if not none_label or none_label in category_to_group:
        raise ValueError("Contrastive none label must be unique and non-empty.")
    group = category_to_group[target]
    options = tuple(sorted({*groups[group], none_label}))
    if target not in options:
        raise RuntimeError(f"Target is absent from its option set: {target}.")
    return options


def contrastive_question(options: Iterable[str]) -> str:
    """Build a target-neutral forced-choice category prompt."""
    options = tuple(str(item) for item in options)
    if not options:
        raise ValueError("Contrastive prompt options must not be empty.")
    return "Allowed labels: " + ", ".join(options) + "."


def _raw_exact_binary_answer(value: Any) -> str | None:
    answer = str(value).strip().lower()
    return answer if answer in {"yes", "no"} else None


def select_v2_top_candidates(
    semantic_reviews: Iterable[Mapping[str, Any]],
    *,
    evidence_score_threshold: float,
    min_mask_score: float | None,
    min_mask_area_ratio: float,
    max_mask_area_ratio: float,
    max_candidates_per_query: int,
) -> list[dict[str, Any]]:
    """Replay the frozen V2 candidate gates from Stage 37 rows."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_item in semantic_reviews:
        item = dict(raw_item)
        grouped.setdefault(str(item["query_key"]), []).append(item)
    selected = []
    for items in grouped.values():
        accepted = []
        for item in items:
            mask_score = item.get("mask_score")
            area_ratio = float(item["mask_area_ratio"])
            if float(item["grounding_score"]) < evidence_score_threshold:
                continue
            if min_mask_score is not None and (
                mask_score is None or float(mask_score) < min_mask_score
            ):
                continue
            if not min_mask_area_ratio <= area_ratio <= max_mask_area_ratio:
                continue
            accepted.append(item)
        accepted.sort(
            key=lambda item: (
                -float(item["grounding_score"]),
                -float(item["mask_area_ratio"]),
                int(item["annotation_index"]),
            )
        )
        selected.extend(accepted[:max_candidates_per_query])
    return selected


def build_contrastive_review_jobs(
    semantic_reviews: Iterable[Mapping[str, Any]],
    *,
    groups: Mapping[str, Iterable[str]],
    category_to_group: Mapping[str, str],
    evidence_score_threshold: float,
    min_mask_score: float | None,
    min_mask_area_ratio: float,
    max_mask_area_ratio: float,
    max_candidates_per_query: int,
    none_label: str,
    require_exact_v2_yes: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select V2-confirmed candidates for a GT-free contrastive review."""
    selected_v2 = select_v2_top_candidates(
        semantic_reviews,
        evidence_score_threshold=evidence_score_threshold,
        min_mask_score=min_mask_score,
        min_mask_area_ratio=min_mask_area_ratio,
        max_mask_area_ratio=max_mask_area_ratio,
        max_candidates_per_query=max_candidates_per_query,
    )
    jobs = []
    for item in selected_v2:
        v2_answer = (
            _raw_exact_binary_answer(item.get("answer"))
            if require_exact_v2_yes
            else parse_yes_no(str(item.get("answer", "")))
        )
        if v2_answer != "yes":
            continue
        target = normalize_text(str(item["object"]))
        options = contrastive_options(
            target,
            groups=groups,
            category_to_group=category_to_group,
            none_label=none_label,
        )
        candidate_key = str(item["candidate_key"])
        jobs.append(
            {
                "v3_key": f"{candidate_key}__contrastive-v3",
                "candidate_key": candidate_key,
                "query_key": str(item["query_key"]),
                "baseline_id": str(item["baseline_id"]),
                "image": str(item["image"]),
                "image_id": int(item["image_id"]),
                "question": str(item["question"]),
                "object": target,
                "annotation_index": int(item["annotation_index"]),
                "grounding_score": float(item["grounding_score"]),
                "mask_score": float(item["mask_score"]),
                "mask_area_ratio": float(item["mask_area_ratio"]),
                "source_crop_image": str(item["crop_image"]),
                "source_crop_sha256": str(item["crop_sha256"]),
                "source_box_xyxy": list(item["source_box_xyxy"]),
                "crop_box_xyxy": list(item["crop_box_xyxy"]),
                "ontology_supercategory": category_to_group[target],
                "allowed_labels": list(options),
                "contrastive_question": contrastive_question(options),
                "v2_answer": v2_answer,
            }
        )
    keys = [str(item["v3_key"]) for item in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("Contrastive review jobs contain duplicate keys.")
    forbidden = {"gt_answer", "pair_role", "expected_answer"}
    for item in jobs:
        if forbidden.intersection(item):
            raise RuntimeError("Contrastive job contains diagnostic labels.")
    return jobs, selected_v2


def ordered_v3_keys_sha256(
    jobs: Iterable[Mapping[str, Any]],
) -> str:
    payload = "\n".join(str(item["v3_key"]) for item in jobs) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def marker_box_in_crop(job: Mapping[str, Any]) -> list[int]:
    source = [float(value) for value in job["source_box_xyxy"]]
    crop = [float(value) for value in job["crop_box_xyxy"]]
    width = max(int(round(crop[2] - crop[0])), 1)
    height = max(int(round(crop[3] - crop[1])), 1)
    return [
        max(0, min(int(round(source[0] - crop[0])), width - 1)),
        max(0, min(int(round(source[1] - crop[1])), height - 1)),
        max(0, min(int(round(source[2] - crop[0])), width - 1)),
        max(0, min(int(round(source[3] - crop[1])), height - 1)),
    ]


def write_marked_candidate_crop(
    source_path: str | Path,
    output_path: str | Path,
    *,
    job: Mapping[str, Any],
    marker_color: Iterable[int],
    marker_width: int,
) -> dict[str, Any]:
    """Draw only a red candidate rectangle on the frozen Stage 37 crop."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    color = tuple(int(value) for value in marker_color)
    if len(color) != 3 or any(not 0 <= value <= 255 for value in color):
        raise ValueError("Marker color must contain three RGB bytes.")
    if marker_width <= 0:
        raise ValueError("Marker width must be positive.")
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    marker_box = marker_box_in_crop(job)
    draw = ImageDraw.Draw(image)
    draw.rectangle(marker_box, outline=color, width=marker_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        output_path,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    return {
        "source_crop_image": str(source_path),
        "marked_crop_image": str(output_path),
        "marker_box_xyxy": marker_box,
        "marker_color_rgb": list(color),
        "marker_width": marker_width,
        "crop_width": image.width,
        "crop_height": image.height,
    }


def parse_contrastive_answer(
    answer: Any, allowed_labels: Iterable[str]
) -> dict[str, Any]:
    """Parse only an exact allowed category label."""
    raw = str(answer).strip().lower()
    allowed = tuple(normalize_text(str(item)) for item in allowed_labels)
    normalized = normalize_text(raw)
    exact = raw in allowed
    return {
        "raw_answer": str(answer),
        "normalized_answer": normalized,
        "selected_label": normalized if exact else None,
        "exact_allowed_label": exact,
        "valid": exact,
    }


def evaluate_contrastive_cascade(
    baseline_records: Iterable[Mapping[str, Any]],
    *,
    jobs: Iterable[Mapping[str, Any]],
    contrastive_reviews: Iterable[Mapping[str, Any]],
    require_strict_accuracy_improvement: bool,
    require_non_decreasing_f1: bool,
    require_positive_net_corrections: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate the V3 cascade after all review inference is frozen."""
    baseline_records = [dict(item) for item in baseline_records]
    jobs = [dict(item) for item in jobs]
    reviews = [dict(item) for item in contrastive_reviews]
    jobs_by_id = {str(item["baseline_id"]): item for item in jobs}
    reviews_by_key = {str(item["v3_key"]): item for item in reviews}
    if len(jobs_by_id) != len(jobs):
        raise ValueError("V3 jobs contain duplicate baseline IDs.")
    if len(reviews_by_key) != len(reviews):
        raise ValueError("V3 reviews contain duplicate keys.")
    missing = [
        str(item["v3_key"])
        for item in jobs
        if str(item["v3_key"]) not in reviews_by_key
    ]
    if missing:
        raise ValueError(f"V3 reviews are incomplete: {missing}.")

    predictions = []
    label_counts = Counter()
    for baseline in baseline_records:
        baseline_id = str(baseline["id"])
        baseline_evaluation = evaluate_answer(
            str(baseline["prediction"]), str(baseline["gt_answer"])
        )
        if baseline_evaluation != baseline.get("evaluation"):
            raise RuntimeError(
                f"Baseline evaluation mismatch for {baseline_id}."
            )
        baseline_answer = parse_yes_no(str(baseline["prediction"]))
        if baseline_answer is None:
            raise ValueError(
                f"Baseline answer is not strict Yes/No: {baseline_id}."
            )
        final_answer = baseline_answer
        selected_label = None
        status = "baseline_unchanged"
        v3_key = None
        if baseline_answer == "no" and baseline_id in jobs_by_id:
            job = jobs_by_id[baseline_id]
            v3_key = str(job["v3_key"])
            review = reviews_by_key[v3_key]
            parsed = parse_contrastive_answer(
                review.get("answer"), job["allowed_labels"]
            )
            selected_label = parsed["selected_label"]
            label_counts[str(selected_label or "invalid")] += 1
            if selected_label == str(job["object"]):
                final_answer = "yes"
                status = "promoted_by_contrastive_target_match"
            elif selected_label is None:
                status = "negative_preserved_invalid_contrastive_answer"
            else:
                status = "negative_preserved_by_contrastive_rejection"
        elif baseline_answer == "yes":
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
                "image": baseline["image"],
                "image_id": baseline["image_id"],
                "question": baseline["question"],
                "object": baseline["object"],
                "gt_answer": baseline["gt_answer"],
                "baseline_prediction": baseline_answer,
                "baseline_evaluation": baseline_evaluation,
                "prediction": final_answer,
                "evaluation": evaluation,
                "changed": changed,
                "correction": correction,
                "status": status,
                "v3_key": v3_key,
                "contrastive_selected_label": selected_label,
            }
        )

    baseline_metrics = binary_metrics(
        [
            {"evaluation": item["baseline_evaluation"]}
            for item in predictions
        ]
    )
    v3_metrics = binary_metrics(predictions)
    changed = [item for item in predictions if item["changed"]]
    beneficial = sum(
        item["correction"] == "beneficial" for item in changed
    )
    harmful = sum(item["correction"] == "harmful" for item in changed)
    rejection_reasons = []
    if (
        require_strict_accuracy_improvement
        and v3_metrics["accuracy"] <= baseline_metrics["accuracy"]
    ):
        rejection_reasons.append("no_strict_accuracy_improvement")
    if (
        require_non_decreasing_f1
        and v3_metrics["f1"] < baseline_metrics["f1"]
    ):
        rejection_reasons.append("f1_decreased")
    if (
        require_positive_net_corrections
        and beneficial - harmful <= 0
    ):
        rejection_reasons.append("non_positive_net_corrections")
    role_metrics = {}
    for role in ("positive", "hard_negative"):
        role_metrics[role] = binary_metrics(
            [
                item
                for item in predictions
                if item["pair_role"] == role
            ]
        )
    metrics = {
        "baseline": baseline_metrics,
        "v3": v3_metrics,
        "delta": {
            name: round(
                float(v3_metrics[name]) - float(baseline_metrics[name]), 6
            )
            for name in ("accuracy", "precision", "recall", "f1", "yes_ratio")
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
        "contrastive_review": {
            "jobs": len(jobs),
            "parsed_label_counts": dict(sorted(label_counts.items())),
        },
        "selection": {
            "decision": (
                "lock_v3_for_held_out_evaluation"
                if not rejection_reasons
                else "reject_v3_on_dev"
            ),
            "eligible": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
            "selection_gates": {
                "require_strict_accuracy_improvement": (
                    require_strict_accuracy_improvement
                ),
                "require_non_decreasing_f1": require_non_decreasing_f1,
                "require_positive_net_corrections": (
                    require_positive_net_corrections
                ),
            },
        },
    }
    return predictions, metrics
