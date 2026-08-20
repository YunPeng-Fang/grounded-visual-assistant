"""Freeze validated cross-dataset candidates and audited image pixels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_benchmark import freeze_hard_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete image audit and create an immutable "
            "cross-dataset benchmark snapshot."
        )
    )
    parser.add_argument(
        "--dataset-dir", default="data/cross_dataset_hard_v1"
    )
    parser.add_argument(
        "--output-dir", default="data/cross_dataset_hard_v1/frozen"
    )
    parser.add_argument("--expected-count", type=int, default=400)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    result = freeze_hard_benchmark(
        project_root=PROJECT_ROOT,
        dataset_dir=project_path(args.dataset_dir),
        output_dir=project_path(args.output_dir),
        expected_count=args.expected_count,
    )
    print(f"Freeze status: {result['status']}")
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
