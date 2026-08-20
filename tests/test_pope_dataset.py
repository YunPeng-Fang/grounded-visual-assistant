from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_dataset import (
    POPE_STRATEGIES,
    build_image_manifest,
    extract_object_phrase,
    normalize_questions,
    question_statistics,
    read_json_records,
)


def official_record(
    question_id: int,
    *,
    image_id: int = 123,
    object_name: str = "person",
    label: str = "yes",
) -> dict:
    return {
        "question_id": question_id,
        "image": f"COCO_val2014_{image_id:012d}.jpg",
        "text": f"Is there a {object_name} in the image?",
        "label": label,
    }


class PopeDatasetTest(unittest.TestCase):
    def test_reads_jsonl_and_json_arrays(self) -> None:
        records = [official_record(1), official_record(2, label="no")]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jsonl_path = root / "questions.jsonl"
            jsonl_path.write_text(
                "\n".join(json.dumps(item) for item in records) + "\n",
                encoding="utf-8",
            )
            array_path = root / "questions.json"
            array_path.write_text(json.dumps(records), encoding="utf-8")

            self.assertEqual(read_json_records(jsonl_path), records)
            self.assertEqual(read_json_records(array_path), records)

    def test_normalizes_official_record(self) -> None:
        normalized = normalize_questions(
            [official_record(7, object_name="baseball bat")],
            "random",
        )

        self.assertEqual(normalized[0]["id"], "pope_coco_random_7")
        self.assertEqual(normalized[0]["image_id"], 123)
        self.assertEqual(normalized[0]["object"], "baseball bat")
        self.assertEqual(normalized[0]["gt_answer"], "yes")
        self.assertEqual(
            normalized[0]["image"],
            "data/pope/images/COCO_val2014_000000000123.jpg",
        )

    def test_extracts_supported_articles(self) -> None:
        self.assertEqual(
            extract_object_phrase("Is there an umbrella in the image?"),
            "umbrella",
        )
        self.assertEqual(
            extract_object_phrase("Is there any traffic light in the image?"),
            "traffic light",
        )

    def test_rejects_unsafe_images_and_duplicate_ids(self) -> None:
        unsafe = official_record(1)
        unsafe["image"] = "../image.jpg"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            normalize_questions([unsafe], "random")

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            normalize_questions(
                [official_record(1), official_record(1)],
                "random",
            )

    def test_builds_cross_strategy_statistics_and_image_manifest(self) -> None:
        questions_by_strategy = {
            strategy: normalize_questions(
                [
                    official_record(1, image_id=1),
                    official_record(
                        2,
                        image_id=2,
                        object_name="dog",
                        label="no",
                    ),
                ],
                strategy,
            )
            for strategy in POPE_STRATEGIES
        }

        statistics = question_statistics(questions_by_strategy)
        images = build_image_manifest(questions_by_strategy)

        self.assertEqual(statistics["questions"], 6)
        self.assertEqual(statistics["images"], 2)
        self.assertEqual(statistics["per_strategy"]["popular"]["yes"], 1)
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0]["questions"], 3)
        self.assertEqual(images[0]["strategies"], list(POPE_STRATEGIES))
        self.assertTrue(
            images[0]["url"].endswith(
                "COCO_val2014_000000000001.jpg"
            )
        )
        http_images = build_image_manifest(
            questions_by_strategy,
            image_base_url="http://images.cocodataset.org/val2014",
        )
        self.assertTrue(http_images[0]["url"].startswith("http://"))


if __name__ == "__main__":
    unittest.main()
