from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_dataset import (
    OPEN_IMAGES_SOURCE,
    VISUAL_GENOME_SOURCE,
)
from grounded_visual_assistant.hard_questions import (
    apply_locked_source_aware_relation_prompt,
    apply_relation_prompt_v2,
    apply_visual_genome_relation_prompt_v3,
    build_hard_questions,
    load_verified_image_labels,
)


def open_images_candidate(image_id: str, split: str) -> dict:
    return {
        "sample_id": f"open_images:{image_id}",
        "source": OPEN_IMAGES_SOURCE,
        "source_image_id": image_id,
        "split": split,
        "categories": ["cat", "dog"],
        "annotation_scope": "exhaustive_for_verified_boxable_classes",
        "objects": [
            {
                "annotation_id": f"{image_id}:cat",
                "category": "cat",
                "source_label_id": "/m/cat",
                "bbox_xyxy_normalized": [0.1, 0.1, 0.3, 0.3],
                "area_ratio": 0.04,
                "is_occluded": True,
                "is_truncated": False,
            },
            {
                "annotation_id": f"{image_id}:dog",
                "category": "dog",
                "source_label_id": "/m/dog",
                "bbox_xyxy_normalized": [0.6, 0.2, 0.9, 0.8],
                "area_ratio": 0.18,
                "is_occluded": False,
                "is_truncated": False,
            },
        ],
        "relations": [
            {
                "subject_annotation_id": f"{image_id}:dog",
                "subject_category": "dog",
                "object_annotation_id": f"{image_id}:cat",
                "object_category": "cat",
                "predicate": "to the right of",
                "native_predicate": "derived_from_box_centers",
            }
        ],
    }


class HardQuestionsTest(unittest.TestCase):
    def test_relation_prompt_v2_changes_only_relation_wording(self) -> None:
        relation = {
            "id": "relation",
            "task_type": "spatial_relation",
            "question": "Where is the person?",
            "categories": ["person", "chair"],
            "metadata": {
                "instance_rules": {
                    "subject": "largest_instance",
                    "object": "unique_category",
                }
            },
        }
        updated = apply_relation_prompt_v2(relation)
        self.assertIn("Treat both named instances as present", updated["question"])
        self.assertIn("the largest visible person", updated["question"])
        self.assertIn("the visible chair", updated["question"])
        self.assertTrue(updated["metadata"]["forced_choice"])
        self.assertEqual(relation["question"], "Where is the person?")

        listing = {"id": "listing", "task_type": "object_listing", "question": "List."}
        self.assertEqual(apply_relation_prompt_v2(listing), listing)

    def test_visual_genome_relation_prompt_v3_preserves_semantic_relation(self) -> None:
        relation = {
            "id": "vg-relation",
            "source": VISUAL_GENOME_SOURCE,
            "task_type": "spatial_relation",
            "question": "Where is the person relative to the chair?",
            "categories": ["person", "chair"],
            "metadata": {
                "instance_rules": {
                    "subject": "unique_category",
                    "object": "largest_instance",
                },
                "relation_provenance": "visual_genome_explicit_relationship",
            },
        }
        updated = apply_visual_genome_relation_prompt_v3(relation)

        self.assertIn("both named object instances as present", updated["question"])
        self.assertIn("depicted spatial relationship", updated["question"])
        self.assertIn("the visible person", updated["question"])
        self.assertIn("the largest visible chair", updated["question"])
        self.assertNotIn("center", updated["question"].lower())
        self.assertTrue(updated["metadata"]["forced_choice"])
        self.assertEqual(
            updated["metadata"]["prompt_version"],
            "visual_genome_semantic_forced_choice_v3",
        )
        self.assertEqual(
            updated["metadata"]["relation_geometry"],
            "depicted_semantic_spatial_relationship",
        )
        self.assertEqual(
            relation["question"], "Where is the person relative to the chair?"
        )

        wrong_source = dict(relation, source=OPEN_IMAGES_SOURCE)
        with self.assertRaises(ValueError):
            apply_visual_genome_relation_prompt_v3(wrong_source)
        wrong_task = dict(relation, task_type="object_listing")
        with self.assertRaises(ValueError):
            apply_visual_genome_relation_prompt_v3(wrong_task)

    def test_locked_source_aware_relation_prompt_dispatches_by_source(self) -> None:
        policy = {
            "protocol": "hard_relation_source_aware_prompt_policy_v1",
            "immutable": True,
            "selected_on_split": "dev",
            "status": "locked",
            "sources": {
                OPEN_IMAGES_SOURCE: {
                    "selected_variant": "v2",
                    "selection_passed": True,
                },
                VISUAL_GENOME_SOURCE: {
                    "selected_variant": "v3",
                    "selection_passed": True,
                },
            },
        }
        base = {
            "id": "relation",
            "task_type": "spatial_relation",
            "categories": ["person", "chair"],
            "metadata": {
                "instance_rules": {
                    "subject": "largest_instance",
                    "object": "unique_category",
                }
            },
        }
        open_images = apply_locked_source_aware_relation_prompt(
            dict(base, source=OPEN_IMAGES_SOURCE), policy
        )
        visual_genome = apply_locked_source_aware_relation_prompt(
            dict(base, source=VISUAL_GENOME_SOURCE), policy
        )
        self.assertEqual(
            open_images["metadata"]["prompt_version"],
            "relation_center_forced_choice_v2",
        )
        self.assertEqual(
            visual_genome["metadata"]["prompt_version"],
            "visual_genome_semantic_forced_choice_v3",
        )

        listing = {
            "id": "listing",
            "source": OPEN_IMAGES_SOURCE,
            "task_type": "object_listing",
            "question": "List.",
        }
        self.assertEqual(
            apply_locked_source_aware_relation_prompt(listing, policy), listing
        )

    def test_verified_labels_parser_preserves_confidence_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["ImageID", "Source", "LabelName", "Confidence"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "ImageID": "one",
                            "Source": "verification",
                            "LabelName": "/m/cat",
                            "Confidence": 1,
                        },
                        {
                            "ImageID": "one",
                            "Source": "verification",
                            "LabelName": "/m/car",
                            "Confidence": 0.0,
                        },
                        {
                            "ImageID": "two",
                            "Source": "verification",
                            "LabelName": "/m/dog",
                            "Confidence": 0,
                        },
                    ]
                )
            labels = load_verified_image_labels(
                path, selected_image_ids={"one"}
            )
            self.assertEqual(labels["one"]["positive"], {"/m/cat"})
            self.assertEqual(labels["one"]["negative"], {"/m/car"})
            self.assertNotIn("two", labels)

    def test_generation_is_source_aware_balanced_and_deterministic(self) -> None:
        candidates = [
            open_images_candidate("one", "dev"),
            open_images_candidate("two", "test"),
            {
                "sample_id": "visual_genome:three",
                "source": VISUAL_GENOME_SOURCE,
                "source_image_id": "three",
                "split": "test",
                "annotation_scope": (
                    "unambiguous_explicit_spatial_relationship_endpoints"
                ),
                "objects": [
                    {
                        "annotation_id": "chair",
                        "category": "chair",
                        "bbox_xyxy_normalized": [0.1, 0.1, 0.3, 0.8],
                    },
                    {
                        "annotation_id": "person",
                        "category": "person",
                        "bbox_xyxy_normalized": [0.6, 0.1, 0.9, 0.9],
                    },
                ],
                "relations": [
                    {
                        "subject_annotation_id": "person",
                        "subject_category": "person",
                        "subject_instance_rule": "unique_category",
                        "object_annotation_id": "chair",
                        "object_category": "chair",
                        "object_instance_rule": "largest_instance",
                        "predicate": "to the right of",
                        "native_predicate": "right of",
                    }
                ],
            },
        ]
        images = [
            {
                "sample_id": item["sample_id"],
                "path": f"images/{item['source_image_id']}.jpg",
                "width": 100,
                "height": 80,
            }
            for item in candidates
        ]
        negative_mids = {f"/m/n{index}" for index in range(1, 6)}
        verified = {
            "one": {"positive": {"/m/cat"}, "negative": set()},
            "two": {"positive": {"/m/cat"}, "negative": negative_mids},
        }
        class_names = {
            **{f"/m/n{index}": f"negative {index}" for index in range(1, 6)},
            "/m/cat": "cat",
            "/m/dog": "dog",
        }

        first = build_hard_questions(
            candidates, images, verified, class_names, seed=17
        )
        second = build_hard_questions(
            candidates, images, verified, class_names, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        existence = [
            item for item in first if item["task_type"] == "object_existence"
        ]
        self.assertEqual(
            sorted(item["gt_answer"] for item in existence), ["no", "yes"]
        )
        negative = next(item for item in existence if item["gt_answer"] == "no")
        self.assertEqual(
            negative["metadata"]["verification"]["confidence"], 0
        )
        positive_only_listing = next(
            item
            for item in first
            if item["sample_id"] == "open_images:one"
            and item["task_type"] == "object_listing"
        )
        self.assertFalse(
            positive_only_listing["metadata"]["has_negative_distractor"]
        )
        vg_questions = [
            item for item in first if item["source"] == VISUAL_GENOME_SOURCE
        ]
        self.assertEqual(len(vg_questions), 1)
        self.assertEqual(vg_questions[0]["task_type"], "spatial_relation")
        self.assertIn("the largest chair", vg_questions[0]["question"])


if __name__ == "__main__":
    unittest.main()
