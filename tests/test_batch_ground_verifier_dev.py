from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_ground_verifier_dev import (
    validate_or_create_run_config,
)


class BatchGroundVerifierDevTest(unittest.TestCase):
    def test_run_config_rejects_changed_query_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text(
                json.dumps(
                    {
                        "protocol": "dev-grounding",
                        "ordered_query_keys_sha256": "old",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_or_create_run_config(
                    path,
                    {
                        "protocol": "dev-grounding",
                        "ordered_query_keys_sha256": "new",
                    },
                )


if __name__ == "__main__":
    unittest.main()
