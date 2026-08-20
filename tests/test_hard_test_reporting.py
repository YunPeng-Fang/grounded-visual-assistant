from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_dataset import (
    OPEN_IMAGES_SOURCE,
    VISUAL_GENOME_SOURCE,
)
from grounded_visual_assistant.hard_test_reporting import (
    build_generalization_rows,
)


class HardTestReportingTest(unittest.TestCase):
    def test_generalization_rows_use_selected_source_metrics(self) -> None:
        relation = {
            "exact_accuracy": 0.6,
            "balanced_accuracy": 0.5,
            "parse_valid_rate": 1.0,
        }
        dev = {
            "object_existence": {"exact_accuracy": 0.8},
            "object_listing": {"macro_f1": 0.7},
            "relations": {
                OPEN_IMAGES_SOURCE: relation,
                VISUAL_GENOME_SOURCE: relation,
            },
        }
        test = {
            "tasks": {
                "object_existence": {"exact_accuracy": 0.85},
                "object_listing": {"macro_f1": 0.72},
            },
            "sources": {
                source: {"tasks": {"spatial_relation": relation}}
                for source in (OPEN_IMAGES_SOURCE, VISUAL_GENOME_SOURCE)
            },
        }
        rows = build_generalization_rows(dev, test)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["delta_test_minus_dev"], 0.05)


if __name__ == "__main__":
    unittest.main()
