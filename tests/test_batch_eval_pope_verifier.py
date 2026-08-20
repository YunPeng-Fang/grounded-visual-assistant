from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from grounded_visual_assistant.grounding_answer_verifier import (
    GroundingAnswerVerifierConfig,
)
from grounded_visual_assistant.pope_evaluation import evaluate_answer
from grounded_visual_assistant.pope_verifier_evaluation import (
    verification_query_key,
)
from scripts.batch_eval_pope_verifier import (
    materialize_outputs,
    preflight,
    validate_or_create_run_config,
)


def baseline(
    sample_id: str,
    *,
    strategy: str,
    image: Path,
    target: str,
    answer: str,
) -> dict:
    return {
        "id": sample_id,
        "strategy": strategy,
        "question_id": 1,
        "image": str(image),
        "image_id": 9,
        "question": "Is there a dog in the image?",
        "object": "dog",
        "gt_answer": target,
        "prediction": answer,
        "evaluation": evaluate_answer(answer, target),
        "latency_seconds": 0.2,
    }


class BatchEvalPopeVerifierTest(unittest.TestCase):
    def test_preflight_deduplicates_repeated_queries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"image")
            records = [
                baseline(
                    strategy,
                    strategy=strategy,
                    image=image,
                    target="yes",
                    answer="No",
                )
                for strategy in ("random", "popular", "adversarial")
            ]
            summary, groups = preflight(
                records,
                require_complete=False,
                requested_strategy="all",
            )

        self.assertEqual(summary["questions"], 3)
        self.assertEqual(summary["unique_queries"], 1)
        self.assertEqual(summary["grounding_queries_saved"], 2)
        self.assertEqual(len(groups), 1)

    def test_materializes_paired_predictions_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"image")
            record = baseline(
                "one",
                strategy="random",
                image=image,
                target="yes",
                answer="No",
            )
            _, groups = preflight(
                [record],
                require_complete=False,
                requested_strategy="random",
            )
            key = verification_query_key(record)
            evidence = {
                key: {
                    "query_key": key,
                    "grounding": {
                        "annotations": [
                            {
                                "class_name": "dog",
                                "bbox": [0, 0, 10, 10],
                                "score": 0.8,
                                "mask_score": 0.95,
                                "mask_area": 100,
                            }
                        ],
                        "img_width": 100,
                        "img_height": 100,
                        "latency_seconds": {"total": 0.4},
                    },
                }
            }
            metrics, predictions = materialize_outputs(
                records=[record],
                groups=groups,
                evidence_by_key=evidence,
                verifier_config=GroundingAnswerVerifierConfig(),
                predictions_path=root / "predictions.jsonl",
                metrics_path=root / "metrics.json",
                error_attempts=0,
                status="completed",
            )

            self.assertEqual(predictions[0]["prediction"], "yes")
            self.assertEqual(
                metrics["paired_outcomes"]["beneficial_corrections"], 1
            )
            self.assertTrue((root / "predictions.jsonl").is_file())

    def test_run_config_rejects_threshold_changes(self) -> None:
        config = {
            "protocol": "batch",
            "verifier_protocol": "verifier",
            "promotion_score_threshold": 0.45,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            validate_or_create_run_config(path, config)
            changed = dict(config, promotion_score_threshold=0.55)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_or_create_run_config(path, changed)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["promotion_score_threshold"], 0.45)


if __name__ == "__main__":
    unittest.main()
