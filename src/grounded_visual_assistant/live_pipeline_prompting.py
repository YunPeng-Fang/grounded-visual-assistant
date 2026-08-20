"""Prompt policies for the live answer-to-grounding evaluation."""

from __future__ import annotations

from typing import Any, Mapping

from .demo import GROUNDED_DEMO_SYSTEM_PROMPT
from .evaluation import COCO_CATEGORIES
from .evidence_answering import parse_question_entities


GENERIC_PROMPT_POLICY = "generic-v1"
TASK_AWARE_COCO_POLICY = "task-aware-coco-v1"
TASK_AWARE_COCO_V2_POLICY = "task-aware-coco-v2"
PROMPT_POLICIES = (
    GENERIC_PROMPT_POLICY,
    TASK_AWARE_COCO_POLICY,
    TASK_AWARE_COCO_V2_POLICY,
)

_JSON_RULE = (
    'Return exactly one valid JSON object with keys "answer" and '
    '"evidence_targets". The answer must be a string and evidence_targets must '
    "be a JSON list of strings. Do not use Markdown or add any text outside "
    "the JSON object."
)


def build_live_pipeline_system_prompt(
    sample: Mapping[str, Any],
    prompt_policy: str,
) -> str:
    """Build a question-conditioned prompt without consulting ground truth."""
    if prompt_policy == GENERIC_PROMPT_POLICY:
        return GROUNDED_DEMO_SYSTEM_PROMPT
    if prompt_policy not in {
        TASK_AWARE_COCO_POLICY,
        TASK_AWARE_COCO_V2_POLICY,
    }:
        raise ValueError(f"Unsupported live prompt policy: {prompt_policy}")

    task_type = str(sample["task_type"])
    question = str(sample["question"])
    if task_type == "object_listing":
        vocabulary = ", ".join(COCO_CATEGORIES)
        if prompt_policy == TASK_AWARE_COCO_V2_POLICY:
            return (
                "Inspect the image first, then return a compact one-line JSON "
                "answer. Select only high-confidence, clearly visible objects "
                "from the COCO-80 vocabulary below. Return at most eight unique "
                "categories. Never copy, continue, or enumerate the vocabulary; "
                "do not add background materials, scene elements, body parts, "
                "or synonyms. Put the same exact lowercase category names in "
                "the comma-separated answer and as individual strings in "
                "evidence_targets. If uncertain about a category, omit it. "
                f"COCO-80 vocabulary: {vocabulary}. {_JSON_RULE}"
            )
        return (
            "You are evaluating visible object recognition on the COCO-80 "
            "vocabulary. Inspect the image and list every clearly visible "
            "category from the allowed vocabulary, using each category's "
            "exact lowercase name once. Put the same comma-separated category "
            "names in answer and as individual strings in evidence_targets. "
            "Use an empty answer and an empty list only if none are visible. "
            f"Allowed vocabulary: {vocabulary}. {_JSON_RULE}"
        )

    entities = parse_question_entities(question, task_type)
    if task_type == "object_existence":
        entity = entities[0]
        return (
            f"Determine whether a visible {entity} is present in the image. "
            'Set answer to exactly "yes" or "no". If the answer is "yes", set '
            f'evidence_targets to ["{entity}"]; if it is "no", set '
            f"evidence_targets to []. {_JSON_RULE}"
        )

    if task_type == "spatial_relation":
        first, second = entities
        return (
            f"Treat both named categories as present. Locate the largest "
            f"visible {first} and the largest visible {second}. Compare their "
            "bounding-box centers; if the relation is close, choose the "
            "single closest direction instead of refusing. Set answer to "
            'exactly one of "to the left of", "to the right of", "above", or '
            f'"below". Set evidence_targets to exactly ["{first}", '
            f'"{second}"] in that order. {_JSON_RULE}'
        )

    raise ValueError(f"Unsupported task type: {task_type}")


def evidence_target_limit(prompt_policy: str) -> int:
    """Return the parser limit associated with a frozen prompt policy."""
    if prompt_policy == GENERIC_PROMPT_POLICY:
        return 6
    if prompt_policy == TASK_AWARE_COCO_POLICY:
        return len(COCO_CATEGORIES)
    if prompt_policy == TASK_AWARE_COCO_V2_POLICY:
        return 8
    raise ValueError(f"Unsupported live prompt policy: {prompt_policy}")
