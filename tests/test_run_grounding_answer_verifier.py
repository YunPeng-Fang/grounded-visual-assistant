from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_grounding_answer_verifier import (
    load_prediction,
    resolve_request,
)


class RunGroundingAnswerVerifierTest(unittest.TestCase):
    def test_resolves_saved_pope_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            record = {
                "id": "pope-one",
                "image": str(root / "image.jpg"),
                "object": "dog",
                "question": "Is there a dog in the image?",
                "prediction": "No",
                "gt_answer": "yes",
                "strategy": "random",
            }
            predictions.write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            loaded = load_prediction(predictions, "pope-one")
            self.assertEqual(loaded["object"], "dog")

            args = argparse.Namespace(
                sample_id="pope-one",
                baseline_predictions=str(predictions),
            )
            request = resolve_request(args)
            self.assertEqual(request["baseline_answer"], "No")
            self.assertEqual(request["gt_answer"], "yes")
            self.assertEqual(request["source"], "saved_pope_baseline")

    def test_missing_sample_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "id": "other",
                        "image": "image.jpg",
                        "object": "dog",
                        "question": "Question",
                        "prediction": "No",
                        "gt_answer": "yes",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(KeyError, "not found"):
                load_prediction(predictions, "missing")


if __name__ == "__main__":
    unittest.main()
