from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_eval_pope_verifier_v2 import (
    validate_or_create_run_config,
)


class BatchEvalPopeVerifierV2Test(unittest.TestCase):
    def test_run_config_rejects_changed_semantic_prompt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            existing = {
                "protocol": "v2",
                "semantic_review_system_prompt_sha256": "old",
            }
            path.write_text(json.dumps(existing), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_or_create_run_config(
                    path,
                    {
                        "protocol": "v2",
                        "semantic_review_system_prompt_sha256": "new",
                    },
                )


if __name__ == "__main__":
    unittest.main()
