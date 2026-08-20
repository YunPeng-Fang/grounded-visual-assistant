"""Build a Dev-only forced-choice relation prompt variant."""

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
from grounded_visual_assistant.hard_questions import apply_relation_prompt_v2
from grounded_visual_assistant.image_dedup import sha256sum


INPUT_PATH = (
    PROJECT_ROOT
    / "data/cross_dataset_hard_v1/questions_v1/dev_questions.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / "data/cross_dataset_hard_v1/questions_v2_dev"


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
        raise RuntimeError(f"Prompt-v2 construction is Dev-only, found: {splits}")
    transformed = [apply_relation_prompt_v2(item) for item in records]
    changed_ids = [
        after["id"]
        for before, after in zip(records, transformed)
        if before["question"] != after["question"]
    ]
    expected_changed = sum(
        item["task_type"] == "spatial_relation" for item in records
    )
    if len(changed_ids) != expected_changed:
        raise RuntimeError("Prompt v2 did not change exactly the relation questions.")

    questions_payload = jsonl_bytes(transformed)
    manifest = {
        "name": "cross_dataset_hard_dev_relation_prompt_v2",
        "schema_version": 1,
        "immutable": True,
        "split": "dev",
        "questions": len(transformed),
        "changed_relation_questions": len(changed_ids),
        "task_counts": dict(
            sorted(Counter(item["task_type"] for item in transformed).items())
        ),
        "prompt_version": "relation_center_forced_choice_v2",
        "design_evidence": (
            "Hard-Dev v1 relation refusals, parse failures, and broad-label "
            "absence claims; no Hard-Test predictions were inspected."
        ),
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
                raise RuntimeError(f"Prompt-v2 artifact differs: {path}")
        status = "verified"
    else:
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            raise RuntimeError(f"Refusing non-empty output directory: {OUTPUT_DIR}")
        for relative_path, payload in artifacts.items():
            write_atomic(OUTPUT_DIR / relative_path, payload)
        status = "created"
    print(f"Prompt-v2 status: {status}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
