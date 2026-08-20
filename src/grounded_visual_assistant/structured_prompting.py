"""Ontology-constrained JSON category generation for grounding prompts."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from typing import Any, Iterable

from .evaluation import CATEGORY_ALIASES, COCO_CATEGORIES, normalize_text
from .vlm_grounding import evaluate_prompt_categories


STRUCTURED_PROMPT_VERSION = "coco80_json_v1"
STRUCTURED_PROMPT_PARSER = "structured_coco_json_v1"
STRUCTURED_SYSTEM_PROMPT = (
    "You are a visual object recognition module. Follow the requested output "
    "schema exactly. Base every selected category only on visible image evidence."
)


def build_structured_category_question() -> str:
    """Request a concise, ontology-constrained JSON category list."""
    allowed = json.dumps(list(COCO_CATEGORIES), ensure_ascii=True)
    return (
        "Identify every clearly visible object whose category appears in the "
        "allowed list below. Use only exact lowercase strings from the list. "
        "Return one valid JSON array of unique strings and nothing else. Do not "
        "use Markdown, explanations, counts, attributes, or broader categories. "
        "If no allowed object is visible, return [].\n\n"
        f"Allowed categories: {allowed}"
    )


def _alias_lookup() -> dict[str, str]:
    lookup = {normalize_text(category): category for category in COCO_CATEGORIES}
    for category, aliases in CATEGORY_ALIASES.items():
        for alias in aliases:
            lookup[normalize_text(alias)] = category
    return lookup


ALIAS_TO_CATEGORY = _alias_lookup()


def _recover_json_value(answer: str) -> tuple[Any, str]:
    stripped = answer.strip()
    try:
        return json.loads(stripped), "direct_json"
    except json.JSONDecodeError:
        pass

    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        try:
            return json.loads(fenced.group(1)), "code_fence"
        except json.JSONDecodeError:
            pass

    array_start = stripped.find("[")
    if array_start >= 0:
        try:
            payload, _ = json.JSONDecoder().raw_decode(stripped[array_start:])
            return payload, "embedded_json"
        except json.JSONDecodeError:
            pass
    return None, "unparseable"


def parse_structured_category_answer(answer: str) -> dict[str, Any]:
    """Parse strict JSON first and recover common formatting deviations."""
    payload, source = _recover_json_value(answer)
    array_found = isinstance(payload, list)
    invalid_items = []
    alias_normalizations = []
    parsed_categories = []
    non_string_items = 0
    if array_found:
        for item in payload:
            if not isinstance(item, str):
                non_string_items += 1
                invalid_items.append(item)
                continue
            normalized = normalize_text(item)
            category = ALIAS_TO_CATEGORY.get(normalized)
            if category is None:
                invalid_items.append(item)
                continue
            if normalized != normalize_text(category):
                alias_normalizations.append({"raw": item, "canonical": category})
            parsed_categories.append(category)

    unique_categories = sorted(set(parsed_categories))
    duplicate_categories = len(parsed_categories) - len(unique_categories)
    schema_valid = (
        array_found
        and non_string_items == 0
        and not invalid_items
        and duplicate_categories == 0
    )
    return {
        "parser": STRUCTURED_PROMPT_PARSER,
        "parse_source": source,
        "strict_json_array": source == "direct_json" and array_found,
        "array_found": array_found,
        "schema_valid": schema_valid,
        "parsed_categories": unique_categories,
        "invalid_items": invalid_items,
        "non_string_items": non_string_items,
        "duplicate_categories": duplicate_categories,
        "alias_normalizations": alias_normalizations,
        "empty_array": array_found and not payload,
    }


def evaluate_structured_category_answer(
    answer: str,
    target_categories: Iterable[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return structured parsing diagnostics and benchmark category scores."""
    structured = parse_structured_category_answer(answer)
    target = sorted({str(category) for category in target_categories})
    scores = evaluate_prompt_categories(
        structured["parsed_categories"],
        target,
    )
    evaluation = {
        "score": scores["f1"],
        "is_correct": scores["exact_match"],
        "predicted_categories": structured["parsed_categories"],
        "target_categories": target,
        "parse_valid": structured["schema_valid"],
        **scores,
    }
    return structured, evaluation


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def aggregate_structured_output(
    prediction_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate JSON adherence, recovery, and generation-length diagnostics."""
    records = list(prediction_records)
    outputs = [record["structured_output"] for record in records]
    source_counts = Counter(item["parse_source"] for item in outputs)
    generated_tokens = [
        int(record["generated_tokens"])
        for record in records
        if record.get("generated_tokens") is not None
    ]
    return {
        "count": len(records),
        "strict_json_array_rate": _mean(
            float(item["strict_json_array"]) for item in outputs
        ),
        "schema_valid_rate": _mean(
            float(item["schema_valid"]) for item in outputs
        ),
        "array_recovery_rate": _mean(
            float(item["array_found"]) for item in outputs
        ),
        "empty_array_rate": _mean(
            float(item["empty_array"]) for item in outputs
        ),
        "invalid_item_count": sum(len(item["invalid_items"]) for item in outputs),
        "duplicate_category_count": sum(
            int(item["duplicate_categories"]) for item in outputs
        ),
        "alias_normalization_count": sum(
            len(item["alias_normalizations"]) for item in outputs
        ),
        "parse_source_counts": dict(sorted(source_counts.items())),
        "hit_max_new_tokens_rate": _mean(
            float(record.get("hit_max_new_tokens", False)) for record in records
        ),
        "generated_tokens_mean": _mean(generated_tokens),
        "generated_tokens_max": max(generated_tokens) if generated_tokens else 0,
    }
