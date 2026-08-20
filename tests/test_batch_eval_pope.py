from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_eval_pope import (
    preflight,
    validate_or_create_run_config,
)


class BatchEvalPopeTest(unittest.TestCase):
    def test_preflight_counts_strategies_labels_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"image")
            records = [
                {
                    "id": "random-yes",
                    "strategy": "random",
                    "image": str(image),
                    "question": "Is there a dog in the image?",
                    "object": "dog",
                    "gt_answer": "yes",
                },
                {
                    "id": "random-no",
                    "strategy": "random",
                    "image": str(image),
                    "question": "Is there a cat in the image?",
                    "object": "cat",
                    "gt_answer": "no",
                },
            ]

            summary = preflight(
                records,
                require_complete=False,
                requested_strategy="random",
            )

        self.assertEqual(summary["questions"], 2)
        self.assertEqual(summary["images"], 1)
        self.assertEqual(summary["strategies"], {"random": 2})
        self.assertEqual(
            summary["labels"]["random"], {"no": 1, "yes": 1}
        )

    def test_run_config_rejects_changed_selection(self) -> None:
        config = {
            "protocol": "protocol",
            "dataset_sha256": "dataset",
            "dataset_manifest_sha256": "manifest",
            "selected_ids_sha256": "selection-a",
            "model_id": "model",
            "torch_dtype": "float16",
            "device_map": "cuda",
            "max_new_tokens": 4,
            "do_sample": False,
            "strategy": "all",
            "samples_per_strategy": 30,
            "system_prompt_sha256": "prompt",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            validate_or_create_run_config(path, config)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["selected_ids_sha256"], "selection-a")

            changed = dict(config, selected_ids_sha256="selection-b")
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_or_create_run_config(path, changed)


if __name__ == "__main__":
    unittest.main()

