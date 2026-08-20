"""Build full-instance COCO ground truth for oracle-conditioned grounding."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.coco_grounding_evaluation import (
    build_oracle_coco_ground_truth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore all COCO instances for the categories prompted on each eval_v0 image."
        )
    )
    parser.add_argument(
        "--questions", default="data/eval_v0/questions.jsonl"
    )
    parser.add_argument(
        "--source-annotations",
        default="data/raw/coco/annotations/instances_val2017.json",
    )
    parser.add_argument(
        "--output", default="data/eval_v0/coco_grounding_gt.json"
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return records


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    questions_path = project_path(args.questions)
    source_path = project_path(args.source_annotations)
    output_path = project_path(args.output)
    if not questions_path.is_file():
        raise FileNotFoundError(f"Question dataset not found: {questions_path}")
    if not source_path.is_file():
        raise FileNotFoundError(
            "COCO source annotations were not found. Expected: "
            f"{source_path}. Re-run scripts/build_eval_v0.py or upload "
            "instances_val2017.json."
        )

    source_coco = json.loads(source_path.read_text(encoding="utf-8"))
    questions = load_jsonl(questions_path)
    ground_truth, report = build_oracle_coco_ground_truth(source_coco, questions)
    ground_truth["grounding_protocol"].update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "questions": str(questions_path),
            "questions_sha256": sha256sum(questions_path),
            "source_annotations": str(source_path),
            "source_annotations_sha256": sha256sum(source_path),
        }
    )
    write_json_atomic(output_path, ground_truth)

    print(f"Ground truth: {output_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
