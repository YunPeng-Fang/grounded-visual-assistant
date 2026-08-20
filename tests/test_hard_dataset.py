from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_dataset import (
    OPEN_IMAGES_SOURCE,
    VISUAL_GENOME_SOURCE,
    iter_json_array,
    load_open_images_candidates,
    load_visual_genome_candidates,
    select_hard_candidates,
    split_hard_candidates,
)


class HardDatasetTest(unittest.TestCase):
    def test_iter_json_array_streams_across_small_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.json"
            expected = [{"id": 1, "value": "alpha"}, {"id": 2, "value": "beta"}]
            path.write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(list(iter_json_array(path, chunk_size=7)), expected)

    def test_open_images_loader_builds_relation_and_hard_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classes_path = root / "classes.csv"
            boxes_path = root / "boxes.csv"
            with classes_path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(
                    [
                        ["/m/cat", "Cat"],
                        ["/m/dog", "Dog"],
                        ["/m/crowd", "Crowd"],
                    ]
                )
            fieldnames = [
                "ImageID",
                "LabelName",
                "XMin",
                "XMax",
                "YMin",
                "YMax",
                "IsOccluded",
                "IsTruncated",
                "IsGroupOf",
            ]
            with boxes_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "ImageID": "abc",
                            "LabelName": "/m/cat",
                            "XMin": 0.05,
                            "XMax": 0.10,
                            "YMin": 0.10,
                            "YMax": 0.15,
                            "IsOccluded": 1,
                            "IsTruncated": 0,
                            "IsGroupOf": 0,
                        },
                        {
                            "ImageID": "abc",
                            "LabelName": "/m/dog",
                            "XMin": 0.70,
                            "XMax": 0.95,
                            "YMin": 0.10,
                            "YMax": 0.40,
                            "IsOccluded": 0,
                            "IsTruncated": 0,
                            "IsGroupOf": 0,
                        },
                        {
                            "ImageID": "abc",
                            "LabelName": "/m/crowd",
                            "XMin": 0.0,
                            "XMax": 1.0,
                            "YMin": 0.0,
                            "YMax": 1.0,
                            "IsOccluded": 0,
                            "IsTruncated": 0,
                            "IsGroupOf": 1,
                        },
                    ]
                )

            candidates = load_open_images_candidates(boxes_path, classes_path)
            self.assertEqual(len(candidates), 1)
            candidate = candidates[0]
            self.assertEqual(candidate["sample_id"], "open_images:abc")
            self.assertEqual(candidate["relations"][0]["subject_category"], "dog")
            self.assertEqual(candidate["relations"][0]["predicate"], "to the right of")
            self.assertIn("tiny_object", candidate["difficulty"]["tags"])
            self.assertIn("occluded", candidate["difficulty"]["tags"])
            self.assertIn("group_of", candidate["difficulty"]["tags"])
            endpoint_ids = {
                candidate["relations"][0]["subject_annotation_id"],
                candidate["relations"][0]["object_annotation_id"],
            }
            endpoints = [
                item for item in candidate["objects"]
                if item["annotation_id"] in endpoint_ids
            ]
            self.assertFalse(any(item["is_group_of"] for item in endpoints))

    def test_visual_genome_loader_excludes_coco_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_data_path = root / "image_data.json"
            relationships_path = root / "relationships.json"
            image_data_path.write_text(
                json.dumps(
                    [
                        {
                            "image_id": 1,
                            "coco_id": 99,
                            "width": 100,
                            "height": 100,
                            "url": "https://example.com/1.jpg",
                        },
                        {
                            "image_id": 2,
                            "coco_id": 100,
                            "width": 200,
                            "height": 100,
                            "url": "https://example.com/2.jpg",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            relationship = {
                "relationship_id": 7,
                "predicate": "RIGHT OF",
                "subject": {
                    "object_id": 10,
                    "name": "Person",
                    "x": 120,
                    "y": 10,
                    "w": 40,
                    "h": 80,
                },
                "object": {
                    "object_id": 11,
                    "name": "Chair",
                    "x": 10,
                    "y": 20,
                    "w": 60,
                    "h": 60,
                },
            }
            relationships_path.write_text(
                json.dumps(
                    [
                        {"image_id": 1, "relationships": [relationship]},
                        {"image_id": 2, "relationships": [relationship]},
                    ]
                ),
                encoding="utf-8",
            )

            candidates = load_visual_genome_candidates(
                relationships_path,
                image_data_path,
                excluded_coco_ids={99},
            )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["sample_id"], "visual_genome:2")
            self.assertEqual(
                candidates[0]["relations"][0]["predicate"], "to the right of"
            )
            self.assertEqual(
                candidates[0]["annotation_scope"],
                "unambiguous_explicit_spatial_relationship_endpoints",
            )

    def test_visual_genome_loader_keeps_only_unambiguous_relations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_data_path = root / "image_data.json"
            relationships_path = root / "relationships.json"
            image_data_path.write_text(
                json.dumps(
                    [
                        {
                            "image_id": 2,
                            "width": 200,
                            "height": 100,
                            "url": "https://example.com/2.jpg",
                        },
                        {
                            "image_id": 3,
                            "width": 200,
                            "height": 100,
                            "url": "https://example.com/3.jpg",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            def entity(object_id: int, name: str, x: int, w: int) -> dict:
                return {
                    "object_id": object_id,
                    "name": name,
                    "x": x,
                    "y": 10,
                    "w": w,
                    "h": 50,
                }

            chair = entity(20, "chair", 80, 30)
            relationships_path.write_text(
                json.dumps(
                    [
                        {
                            "image_id": 2,
                            "relationships": [
                                {
                                    "relationship_id": 1,
                                    "predicate": "RIGHT OF",
                                    "subject": entity(10, "dog", 140, 15),
                                    "object": chair,
                                },
                                {
                                    "relationship_id": 2,
                                    "predicate": "LEFT OF",
                                    "subject": entity(11, "dog", 10, 50),
                                    "object": chair,
                                },
                            ],
                        },
                        {
                            "image_id": 3,
                            "relationships": [
                                {
                                    "relationship_id": 3,
                                    "predicate": "RIGHT OF",
                                    "subject": entity(30, "cat", 140, 15),
                                    "object": chair,
                                },
                                {
                                    "relationship_id": 4,
                                    "predicate": "ON",
                                    "subject": entity(31, "cat", 10, 50),
                                    "object": chair,
                                },
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            candidates = load_visual_genome_candidates(
                relationships_path, image_data_path
            )
            self.assertEqual([item["sample_id"] for item in candidates], ["visual_genome:2"])
            relations = candidates[0]["relations"]
            self.assertEqual(len(relations), 1)
            self.assertEqual(relations[0]["relationship_id"], 2)
            self.assertEqual(
                relations[0]["subject_instance_rule"], "largest_instance"
            )
            self.assertEqual(
                relations[0]["object_instance_rule"], "unique_category"
            )

    def test_selection_and_split_are_deterministic_and_source_balanced(self) -> None:
        candidates = []
        for source in (OPEN_IMAGES_SOURCE, VISUAL_GENOME_SOURCE):
            for index in range(6):
                candidates.append(
                    {
                        "sample_id": f"{source}:{index}",
                        "source": source,
                        "categories": [f"category-{index % 3}", "person"],
                        "difficulty": {
                            "score": float(index % 3 + 1),
                            "tags": ["tiny_object" if index % 2 else "occluded"],
                        },
                    }
                )
        quotas = {OPEN_IMAGES_SOURCE: 4, VISUAL_GENOME_SOURCE: 4}
        first = select_hard_candidates(candidates, quotas, seed=2026)
        second = select_hard_candidates(candidates, quotas, seed=2026)
        self.assertEqual(
            [item["sample_id"] for item in first],
            [item["sample_id"] for item in second],
        )
        self.assertEqual(len(first), 8)

        dev_ids, test_ids = split_hard_candidates(
            first, dev_fraction=0.5, seed=2026
        )
        self.assertEqual(len(dev_ids), 4)
        self.assertEqual(len(test_ids), 4)
        self.assertFalse(set(dev_ids) & set(test_ids))
        for source in (OPEN_IMAGES_SOURCE, VISUAL_GENOME_SOURCE):
            self.assertEqual(sum(source in value for value in dev_ids), 2)
            self.assertEqual(sum(source in value for value in test_ids), 2)


if __name__ == "__main__":
    unittest.main()
