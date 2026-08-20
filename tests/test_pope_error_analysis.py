from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_error_analysis import (
    analyze_pope_predictions,
    render_case_sheet,
    render_pope_error_report,
    validate_pope_analysis_sources,
)
from grounded_visual_assistant.pope_evaluation import (
    POPE_PROTOCOL,
    evaluate_answer,
)


def prediction(
    sample_id: str,
    *,
    strategy: str,
    image_id: int,
    object_name: str,
    target: str,
    answer: str,
    image: str = "image.jpg",
) -> dict:
    return {
        "id": sample_id,
        "strategy": strategy,
        "image_id": image_id,
        "image": image,
        "question": f"Is there a {object_name} in the image?",
        "object": object_name,
        "gt_answer": target,
        "prediction": answer,
        "evaluation": evaluate_answer(answer, target),
        "latency_seconds": 0.2,
        "generated_tokens": 2,
    }


def sample_records() -> list[dict]:
    records = [
        prediction(
            f"{strategy}-positive",
            strategy=strategy,
            image_id=1,
            object_name="cat",
            target="yes",
            answer="No",
        )
        for strategy in ("random", "popular", "adversarial")
    ]
    records.extend(
        [
            prediction(
                "random-negative",
                strategy="random",
                image_id=1,
                object_name="dog",
                target="no",
                answer="Yes",
            ),
            prediction(
                "popular-negative",
                strategy="popular",
                image_id=2,
                object_name="chair",
                target="no",
                answer="No",
            ),
            prediction(
                "adversarial-negative",
                strategy="adversarial",
                image_id=2,
                object_name="table",
                target="no",
                answer="No",
            ),
        ]
    )
    return records


class PopeErrorAnalysisTest(unittest.TestCase):
    def test_deduplicates_repeated_positive_queries(self) -> None:
        analysis = analyze_pope_predictions(sample_records())

        attribution = analysis.summary["error_attribution"]
        self.assertEqual(attribution["false_negative_questions"], 3)
        self.assertEqual(attribution["unique_false_negative_queries"], 1)
        self.assertEqual(attribution["false_positive_questions"], 1)
        repetition = analysis.summary["positive_query_repetition"]
        self.assertEqual(repetition["unique_positive_queries"], 1)
        self.assertEqual(repetition["complete_three_strategy_groups"], 1)
        self.assertEqual(repetition["cross_strategy_disagreements"], 0)

        cat = next(
            item for item in analysis.per_object if item["object"] == "cat"
        )
        self.assertEqual(cat["false_negatives"], 3)
        self.assertEqual(cat["unique_false_negative_queries"], 1)
        self.assertEqual(cat["recall"], 0.0)

    def test_validates_saved_metrics_and_config(self) -> None:
        records = sample_records()
        analysis = analyze_pope_predictions(records)
        metrics = {
            "status": "completed",
            "protocol": POPE_PROTOCOL,
            "coverage": {"expected": 6, "completed": 6},
            "overall": {
                "confusion": analysis.summary["overall"]["confusion"]
            },
        }
        config = {
            "protocol": POPE_PROTOCOL,
            "strategy": "all",
            "require_complete": True,
        }

        validate_pope_analysis_sources(
            analysis,
            metrics=metrics,
            run_config=config,
        )
        metrics["coverage"]["completed"] = 5
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            validate_pope_analysis_sources(
                analysis,
                metrics=metrics,
                run_config=config,
            )

    def test_rejects_duplicate_ids_and_modified_evaluation(self) -> None:
        records = sample_records()
        records.append(dict(records[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_pope_predictions(records)

        modified = sample_records()
        modified[0]["evaluation"] = dict(modified[0]["evaluation"])
        modified[0]["evaluation"]["is_correct"] = True
        with self.assertRaisesRegex(RuntimeError, "does not reproduce"):
            analyze_pope_predictions(modified)

    def test_renders_report_and_visual_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (100, 80), "white").save(root / "image.jpg")
            records = sample_records()
            analysis = analyze_pope_predictions(
                records,
                representative_limit=2,
            )
            output = root / "cases.jpg"
            result = render_case_sheet(
                analysis.representative_cases,
                project_root=root,
                output_path=output,
                title="POPE Cases",
            )
            self.assertEqual(result["rendered"], 2)
            self.assertTrue(output.is_file())
            report = render_pope_error_report(
                analysis,
                predictions_path="predictions.jsonl",
                visual_paths={"false_negative": "cases.jpg"},
            )
            self.assertIn("Positive-Query Repetition Audit", report)
            self.assertIn("cases.jpg", report)


if __name__ == "__main__":
    unittest.main()
