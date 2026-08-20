from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_benchmark import freeze_hard_benchmark
from grounded_visual_assistant.hard_dataset import (
    OPEN_IMAGES_SOURCE,
    VISUAL_GENOME_SOURCE,
)
from grounded_visual_assistant.image_dedup import sha256sum


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )


class HardBenchmarkTest(unittest.TestCase):
    def test_freeze_creates_verifies_and_detects_pixel_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data" / "hard"
            audit = dataset / "image_audit"
            images = root / "data" / "hard" / "images"
            images.mkdir(parents=True)
            (images / "one.jpg").write_bytes(b"first-image")
            (images / "two.jpg").write_bytes(b"second-image")
            candidates = [
                {
                    "sample_id": "open_images:one",
                    "source": OPEN_IMAGES_SOURCE,
                    "source_image_id": "one",
                    "split": "dev",
                },
                {
                    "sample_id": "visual_genome:two",
                    "source": VISUAL_GENOME_SOURCE,
                    "source_image_id": "two",
                    "split": "test",
                },
            ]
            write_jsonl(dataset / "candidates.jsonl", candidates)
            write_json(dataset / "manifest.json", {"name": "candidate"})
            write_json(
                dataset / "splits" / "dev_sample_ids.json",
                {"sample_ids": ["open_images:one"]},
            )
            write_json(
                dataset / "splits" / "test_sample_ids.json",
                {"sample_ids": ["visual_genome:two"]},
            )
            downloads = []
            for sample_id, filename, split in (
                ("open_images:one", "one.jpg", "dev"),
                ("visual_genome:two", "two.jpg", "test"),
            ):
                path = images / filename
                downloads.append(
                    {
                        "sample_id": sample_id,
                        "path": str(path.relative_to(root)).replace("/", "\\"),
                        "sha256": sha256sum(path),
                        "dhash": "0" * 16,
                        "width": 10,
                        "height": 10,
                        "format": "JPEG",
                        "mode": "RGB",
                        "bytes": path.stat().st_size,
                        "split": split,
                    }
                )
            write_jsonl(audit / "downloads.jsonl", downloads)
            write_jsonl(
                audit / "sample_status.jsonl",
                [
                    {"sample_id": item["sample_id"], "status": "accepted"}
                    for item in candidates
                ],
            )
            ids = [item["sample_id"] for item in candidates]
            write_json(audit / "accepted_sample_ids.json", ids)
            write_json(audit / "review_sample_ids.json", [])
            write_json(audit / "excluded_sample_ids.json", [])
            write_json(
                audit / "summary.json",
                {"status": "complete", "accepted": 2},
            )

            output = dataset / "frozen"
            created = freeze_hard_benchmark(
                project_root=root,
                dataset_dir=dataset,
                output_dir=output,
                expected_count=2,
            )
            self.assertEqual(created["status"], "created")
            verified = freeze_hard_benchmark(
                project_root=root,
                dataset_dir=dataset,
                output_dir=output,
                expected_count=2,
            )
            self.assertEqual(verified["status"], "verified")

            (images / "one.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                freeze_hard_benchmark(
                    project_root=root,
                    dataset_dir=dataset,
                    output_dir=output,
                    expected_count=2,
                )


if __name__ == "__main__":
    unittest.main()
