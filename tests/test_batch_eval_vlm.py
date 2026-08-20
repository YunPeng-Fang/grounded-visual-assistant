from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_eval_vlm import (
    preflight_dataset,
    validate_or_create_run_config,
)


class BatchEvalVlmTest(unittest.TestCase):
    def test_run_config_migrates_new_immutable_fields(self) -> None:
        current = {
            "dataset_sha256": "dataset",
            "dataset_manifest_sha256": "manifest",
            "model_id": "model",
            "torch_dtype": "float16",
            "device_map": "cuda",
            "max_new_tokens": 64,
            "do_sample": False,
            "task_type": "all",
            "required_split": "dev",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            old_config = {
                key: value
                for key, value in current.items()
                if key not in {
                    "dataset_manifest_sha256",
                    "task_type",
                    "required_split",
                }
            }
            path.write_text(json.dumps(old_config), encoding="utf-8")
            validate_or_create_run_config(path, current)
            migrated = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(migrated["dataset_manifest_sha256"], "manifest")
            self.assertEqual(migrated["task_type"], "all")
            self.assertEqual(migrated["required_split"], "dev")

    def test_preflight_counts_images_and_rejects_wrong_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"image")
            dataset = root / "questions.jsonl"
            records = [
                {
                    "id": "one",
                    "image": str(image),
                    "task_type": "object_existence",
                    "source": "open_images",
                    "split": "dev",
                },
                {
                    "id": "two",
                    "image": str(image),
                    "task_type": "spatial_relation",
                    "source": "visual_genome",
                    "split": "dev",
                },
            ]
            summary = preflight_dataset(records, dataset, "dev")
            self.assertEqual(summary["questions"], 2)
            self.assertEqual(summary["images"], 1)
            self.assertEqual(summary["splits"], {"dev": 2})
            with self.assertRaisesRegex(RuntimeError, "required 'test'"):
                preflight_dataset(records, dataset, "test")


if __name__ == "__main__":
    unittest.main()
