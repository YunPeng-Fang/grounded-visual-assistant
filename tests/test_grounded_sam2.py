from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.grounded_sam2 import (
    GroundedSam2Config,
    class_aware_nms_indices,
    normalize_grounding_prompt,
)


class GroundedSam2Test(unittest.TestCase):
    def test_prompt_normalization(self) -> None:
        self.assertEqual(
            normalize_grounding_prompt("Car;  Tire. PERSON"),
            "car. tire. person.",
        )

    def test_empty_prompt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_grounding_prompt(" . ; ")

    def test_threshold_validation(self) -> None:
        with self.assertRaises(ValueError):
            GroundedSam2Config(
                grounding_model_id="model",
                sam2_checkpoint="checkpoint",
                sam2_model_config="config",
                box_threshold=1.1,
            )

    def test_nms_threshold_validation(self) -> None:
        with self.assertRaises(ValueError):
            GroundedSam2Config(
                grounding_model_id="model",
                sam2_checkpoint="checkpoint",
                sam2_model_config="config",
                nms_iou_threshold=-0.1,
            )

    def test_class_aware_nms_only_suppresses_same_label(self) -> None:
        boxes = np.asarray(
            [
                [0, 0, 10, 10],
                [1, 1, 11, 11],
                [1, 1, 11, 11],
                [30, 30, 40, 40],
            ],
            dtype=np.float32,
        )
        scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
        labels = ["bottle", "bottle", "cup", "bottle"]
        kept = class_aware_nms_indices(boxes, scores, labels, 0.5)
        self.assertEqual(kept.tolist(), [0, 2, 3])


if __name__ == "__main__":
    unittest.main()
