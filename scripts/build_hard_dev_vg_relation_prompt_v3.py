"""Build a Dev-only Visual Genome semantic relation prompt variant."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_benchmark import read_jsonl
from grounded_visual_assistant.hard_dataset import VISUAL_GENOME_SOURCE
from grounded_visual_assistant.hard_questions import (
    apply_visual_genome_relation_prompt_v3,
)
from grounded_visual_assistant.image_dedup import sha256sum


INPUT_PATH = (
    PROJECT_ROOT
    / "data/cross_dataset_hard_v1/questions_v1/dev_questions.jsonl"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/cross_dataset_hard_v1/questions_v3_dev_vg"
)
EXPECTED_QUESTIONS = 100


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in records
    ).encode("utf-8")


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    records = read_jsonl(INPUT_PATH)
    splits = {str(item.get("split")) for item in records}
    if splits != {"dev"}:
        raise RuntimeError(f"Prompt-v3 construction is Dev-only, found: {splits}")

    selected = [
        item
        for item in records
        if item.get("source") == VISUAL_GENOME_SOURCE
        and item.get("task_type") == "spatial_relation"
    ]
    if len(selected) != EXPECTED_QUESTIONS:
        raise RuntimeError(
            "Expected exactly 100 Visual Genome Dev relation questions, found "
            f"{len(selected)}."
        )
    transformed = [
        apply_visual_genome_relation_prompt_v3(item) for item in selected
    ]
    if any(
        before["question"] == after["question"]
        for before, after in zip(selected, transformed)
    ):
        raise RuntimeError("Prompt v3 did not change every selected question.")
    if any("center" in item["question"].lower() for item in transformed):
        raise RuntimeError("Prompt v3 must not impose center-based geometry.")

    questions_payload = jsonl_bytes(transformed)
    manifest = {
        "name": "cross_dataset_hard_dev_vg_relation_prompt_v3",
        "schema_version": 1,
        "immutable": True,
        "split": "dev",
        "source": VISUAL_GENOME_SOURCE,
        "questions": len(transformed),
        "task_counts": dict(
            sorted(Counter(item["task_type"] for item in transformed).items())
        ),
        "prompt_version": "visual_genome_semantic_forced_choice_v3",
        "design_evidence": (
            "Visual Genome uses explicit human relationship annotations, while "
            "the center-based prompt v2 was designed for Open Images geometry. "
            "Only Hard-Dev v1 and v2 results informed this design; Hard-Test "
            "questions and predictions were not inspected."
        ),
        "acceptance_criteria": {
            "parse_valid_rate_min": 0.98,
            "hit_max_new_tokens_max": 0,
            "balanced_accuracy_min": 0.508782,
            "exact_accuracy_min": 0.58,
            "comparison_scope": "visual_genome_dev_100_paired_against_v1_and_v2",
        },
        "input_sha256": {
            "dev_questions_v1": sha256sum(INPUT_PATH),
            "questions_v1_manifest": sha256sum(INPUT_PATH.parent / "manifest.json"),
        },
        "artifact_sha256": {
            "dev_questions.jsonl": bytes_sha256(questions_payload)
        },
    }
    manifest_payload = json_bytes(manifest)
    artifacts = {
        "dev_questions.jsonl": questions_payload,
        "manifest.json": manifest_payload,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists():
        for relative_path, payload in artifacts.items():
            path = OUTPUT_DIR / relative_path
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"Prompt-v3 artifact differs: {path}")
        status = "verified"
    else:
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            raise RuntimeError(f"Refusing non-empty output directory: {OUTPUT_DIR}")
        for relative_path, payload in artifacts.items():
            write_atomic(OUTPUT_DIR / relative_path, payload)
        status = "created"
    print(f"Prompt-v3 status: {status}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
