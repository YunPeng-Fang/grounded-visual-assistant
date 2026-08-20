from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.image_dedup import (
    classify_samples,
    find_duplicate_pairs,
    fingerprint_image,
    hamming_distance,
)


class ImageDedupTest(unittest.TestCase):
    def test_fingerprint_detects_identical_pixel_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            image = Image.new("RGB", (32, 24), "white")
            for x in range(16):
                for y in range(24):
                    image.putpixel((x, y), (x * 10, y * 5, 20))
            image.save(first)
            image.save(second)
            first_fingerprint = fingerprint_image(first)
            second_fingerprint = fingerprint_image(second)
            self.assertEqual(first_fingerprint["sha256"], second_fingerprint["sha256"])
            self.assertEqual(first_fingerprint["dhash"], second_fingerprint["dhash"])
            self.assertEqual(first_fingerprint["width"], 32)
            self.assertEqual(first_fingerprint["height"], 24)

    def test_hamming_distance_uses_hash_bits(self) -> None:
        self.assertEqual(hamming_distance("0000000000000000", "0000000000000003"), 2)

    def test_duplicate_pairs_include_exact_and_review_only_near_matches(self) -> None:
        selected = [
            {
                "sample_id": "a",
                "split": "dev",
                "sha256": "same",
                "dhash": "0000000000000000",
                "width": 100,
                "height": 50,
            },
            {
                "sample_id": "b",
                "split": "test",
                "sha256": "same",
                "dhash": "0000000000000000",
                "width": 100,
                "height": 50,
            },
            {
                "sample_id": "c",
                "split": "test",
                "sha256": "different",
                "dhash": "0000000000000001",
                "width": 200,
                "height": 100,
            },
        ]
        pairs = find_duplicate_pairs(selected, [], near_threshold=1)
        exact = [item for item in pairs if item["kind"] == "exact"]
        near = [item for item in pairs if item["kind"] == "near"]
        self.assertEqual(len(exact), 1)
        self.assertEqual(len(near), 2)
        self.assertTrue(exact[0]["cross_split"])

    def test_classification_excludes_exact_and_reviews_near_duplicates(self) -> None:
        selected = [
            {
                "sample_id": "a",
                "source": "source",
                "split": "dev",
                "sha256": "same",
            },
            {
                "sample_id": "b",
                "source": "source",
                "split": "test",
                "sha256": "same",
            },
            {
                "sample_id": "c",
                "source": "source",
                "split": "test",
                "sha256": "other",
            },
        ]
        pairs = [
            {
                "kind": "exact",
                "left_id": "a",
                "left_kind": "selected",
                "right_id": "b",
                "right_kind": "selected",
                "hamming_distance": 0,
            },
            {
                "kind": "near",
                "left_id": "a",
                "left_kind": "selected",
                "right_id": "c",
                "right_kind": "selected",
                "hamming_distance": 2,
            },
        ]
        statuses = {
            item["sample_id"]: item["status"]
            for item in classify_samples(selected, pairs)
        }
        self.assertEqual(statuses["a"], "review_near_duplicate")
        self.assertEqual(statuses["b"], "exclude_exact_selected_duplicate")
        self.assertEqual(statuses["c"], "review_near_duplicate")


if __name__ == "__main__":
    unittest.main()
