from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.artifact_paths import (
    portable_gallery,
    portable_project_path,
    resolve_project_path,
)


class ArtifactPathsTest(unittest.TestCase):
    def test_current_checkout_path_becomes_relative(self) -> None:
        artifact = PROJECT_ROOT / "outputs" / "run" / "mask.jpg"
        self.assertEqual(
            portable_project_path(artifact, PROJECT_ROOT),
            "outputs/run/mask.jpg",
        )

    def test_linux_checkout_path_maps_to_synced_checkout(self) -> None:
        artifact = (
            "/data/fyp/EngineeringProjects/grounded-visual-assistant/"
            "outputs/run/mask.jpg"
        )
        self.assertEqual(
            portable_project_path(artifact, PROJECT_ROOT),
            "outputs/run/mask.jpg",
        )

    def test_gallery_conversion_and_resolution(self) -> None:
        gallery = [
            (
                PROJECT_ROOT / "outputs" / "run" / "mask.jpg",
                "Segmentation masks",
            )
        ]
        converted = portable_gallery(gallery, PROJECT_ROOT)
        self.assertEqual(
            converted,
            [["outputs/run/mask.jpg", "Segmentation masks"]],
        )
        self.assertEqual(
            resolve_project_path(converted[0][0], PROJECT_ROOT),
            PROJECT_ROOT / "outputs" / "run" / "mask.jpg",
        )


if __name__ == "__main__":
    unittest.main()
