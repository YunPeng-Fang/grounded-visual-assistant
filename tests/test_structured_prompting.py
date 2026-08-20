from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.prompts import build_vlm_messages
from grounded_visual_assistant.structured_prompting import (
    STRUCTURED_SYSTEM_PROMPT,
    aggregate_structured_output,
    build_structured_category_question,
    evaluate_structured_category_answer,
    parse_structured_category_answer,
)


class StructuredPromptingTest(unittest.TestCase):
    def test_question_requires_json_and_contains_ontology(self) -> None:
        question = build_structured_category_question()
        self.assertIn("valid JSON array", question)
        self.assertIn('"person"', question)
        self.assertIn('"toothbrush"', question)

    def test_custom_system_prompt_is_forwarded(self) -> None:
        messages = build_vlm_messages(
            "image.jpg",
            "question",
            system_prompt=STRUCTURED_SYSTEM_PROMPT,
        )
        self.assertEqual(messages[0]["content"], STRUCTURED_SYSTEM_PROMPT)

    def test_strict_json_array(self) -> None:
        parsed = parse_structured_category_answer('["person", "dog"]')
        self.assertTrue(parsed["strict_json_array"])
        self.assertTrue(parsed["schema_valid"])
        self.assertEqual(parsed["parsed_categories"], ["dog", "person"])

    def test_code_fence_and_alias_are_recovered_but_not_strict(self) -> None:
        parsed = parse_structured_category_answer(
            '```json\n["people", "sofa"]\n```'
        )
        self.assertFalse(parsed["strict_json_array"])
        self.assertTrue(parsed["schema_valid"])
        self.assertEqual(parsed["parse_source"], "code_fence")
        self.assertEqual(parsed["parsed_categories"], ["couch", "person"])
        self.assertEqual(len(parsed["alias_normalizations"]), 2)

    def test_invalid_and_duplicate_items_fail_schema(self) -> None:
        parsed = parse_structured_category_answer(
            '["person", "person", "vehicle", 3]'
        )
        self.assertFalse(parsed["schema_valid"])
        self.assertEqual(parsed["parsed_categories"], ["person"])
        self.assertEqual(parsed["duplicate_categories"], 1)
        self.assertEqual(parsed["non_string_items"], 1)
        self.assertEqual(parsed["invalid_items"], ["vehicle", 3])

    def test_evaluation_uses_only_parsed_categories(self) -> None:
        structured, evaluation = evaluate_structured_category_answer(
            '["person", "cat"]',
            ["person", "dog"],
        )
        self.assertTrue(structured["schema_valid"])
        self.assertEqual(evaluation["precision"], 0.5)
        self.assertEqual(evaluation["recall"], 0.5)
        self.assertEqual(evaluation["f1"], 0.5)

    def test_aggregate_structured_diagnostics(self) -> None:
        strict = parse_structured_category_answer('["person"]')
        recovered = parse_structured_category_answer('```json\n["dog"]\n```')
        metrics = aggregate_structured_output(
            [
                {
                    "structured_output": strict,
                    "generated_tokens": 4,
                    "hit_max_new_tokens": False,
                },
                {
                    "structured_output": recovered,
                    "generated_tokens": 8,
                    "hit_max_new_tokens": True,
                },
            ]
        )
        self.assertEqual(metrics["strict_json_array_rate"], 0.5)
        self.assertEqual(metrics["schema_valid_rate"], 1.0)
        self.assertEqual(metrics["hit_max_new_tokens_rate"], 0.5)
        self.assertEqual(metrics["generated_tokens_mean"], 6.0)


if __name__ == "__main__":
    unittest.main()
