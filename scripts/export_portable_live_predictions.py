"""Export live-pipeline predictions with project-relative artifact paths."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.artifact_paths import (
    portable_gallery,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create portable copies without changing frozen predictions."
    )
    parser.add_argument(
        "--root", default="outputs/eval_live_pipeline_v0"
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write exports even when a referenced artifact was not copied.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    source_paths = sorted(root.rglob("predictions.jsonl"))
    if not source_paths:
        raise FileNotFoundError(f"No predictions.jsonl found under {root}")

    summary = []
    for source_path in source_paths:
        records = read_jsonl(source_path)
        converted = []
        artifact_count = 0
        missing = []
        for source_record in records:
            record = deepcopy(source_record)
            gallery = record.get("grounding", {}).get("artifacts", [])
            portable = portable_gallery(gallery, PROJECT_ROOT)
            record.setdefault("grounding", {})["artifacts"] = portable
            for artifact_path, _ in portable:
                artifact_count += 1
                resolved = resolve_project_path(artifact_path, PROJECT_ROOT)
                if not resolved.is_file():
                    missing.append(str(artifact_path))
            converted.append(record)
        if missing and not args.allow_missing:
            raise FileNotFoundError(
                f"{source_path} references {len(missing)} missing artifacts; "
                f"first: {missing[0]}"
            )
        destination = source_path.with_name("predictions_portable.jsonl")
        temporary = destination.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False) + "\n"
                for item in converted
            ),
            encoding="utf-8",
        )
        temporary.replace(destination)
        summary.append(
            {
                "source": str(source_path),
                "destination": str(destination),
                "records": len(converted),
                "artifacts": artifact_count,
                "missing_artifacts": len(missing),
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
