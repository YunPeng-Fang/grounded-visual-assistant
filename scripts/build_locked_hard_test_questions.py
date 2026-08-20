"""Apply the locked source-aware relation policy to the complete Hard-Test."""

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
from grounded_visual_assistant.hard_dataset import (
    OPEN_IMAGES_SOURCE,
    VISUAL_GENOME_SOURCE,
)
from grounded_visual_assistant.hard_questions import (
    apply_locked_source_aware_relation_prompt,
)
from grounded_visual_assistant.image_dedup import sha256sum


INPUT_PATH = (
    PROJECT_ROOT
    / "data/cross_dataset_hard_v1/questions_v1/test_questions.jsonl"
)
INPUT_MANIFEST = INPUT_PATH.parent / "manifest.json"
POLICY_PATH = (
    PROJECT_ROOT
    / "outputs/cross_dataset_hard_v1/relation_prompt_policy_dev_v1/"
    "selected_policy.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/cross_dataset_hard_v1/questions_locked_test_v1"
)
EXPECTED_QUESTIONS = 400
EXPECTED_RELATIONS = 200


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


def validate_policy(policy: dict[str, Any]) -> None:
    if (
        policy.get("protocol") != "hard_relation_source_aware_prompt_policy_v1"
        or policy.get("status") != "locked"
        or policy.get("selected_on_split") != "dev"
        or not policy.get("immutable")
        or not policy.get("acceptance", {}).get("source_aware_policy")
        or policy.get("test_status") != "not_generated_not_evaluated"
    ):
        raise RuntimeError("The selected relation policy is not a valid Dev lock.")
    expected = {
        OPEN_IMAGES_SOURCE: "v2",
        VISUAL_GENOME_SOURCE: "v3",
    }
    observed = {
        source: item.get("selected_variant")
        for source, item in (policy.get("sources") or {}).items()
    }
    if observed != expected:
        raise RuntimeError(f"Unexpected source-aware policy: {observed}")


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    validate_policy(policy)
    records = read_jsonl(INPUT_PATH)
    if len(records) != EXPECTED_QUESTIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_QUESTIONS} Hard-Test questions, found "
            f"{len(records)}."
        )
    if {str(item.get("split")) for item in records} != {"test"}:
        raise RuntimeError("The locked Test builder received a non-Test record.")

    transformed = [
        apply_locked_source_aware_relation_prompt(item, policy)
        for item in records
    ]
    relation_count = sum(
        item.get("task_type") == "spatial_relation" for item in records
    )
    if relation_count != EXPECTED_RELATIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_RELATIONS} relation questions, found "
            f"{relation_count}."
        )
    changed_ids = [
        before["id"]
        for before, after in zip(records, transformed)
        if before["question"] != after["question"]
    ]
    if len(changed_ids) != EXPECTED_RELATIONS:
        raise RuntimeError("The locked policy did not change exactly the relations.")

    invariant_fields = (
        "id",
        "image",
        "image_id",
        "sample_id",
        "source_image_id",
        "source",
        "split",
        "task_type",
        "gt_answer",
        "categories",
        "evidence_boxes",
    )
    for before, after in zip(records, transformed):
        for field in invariant_fields:
            if before.get(field) != after.get(field):
                raise RuntimeError(
                    f"Locked Test field {field!r} changed for {before['id']}."
                )
        if (
            before["task_type"] != "spatial_relation"
            and before != after
        ):
            raise RuntimeError(
                f"Non-relation Test question changed: {before['id']}."
            )

    questions_payload = jsonl_bytes(transformed)
    manifest = {
        "name": "cross_dataset_hard_locked_test_questions_v1",
        "schema_version": 1,
        "immutable": True,
        "split": "test",
        "questions": len(transformed),
        "changed_relation_questions": len(changed_ids),
        "sources": dict(
            sorted(Counter(item["source"] for item in transformed).items())
        ),
        "task_counts": dict(
            sorted(Counter(item["task_type"] for item in transformed).items())
        ),
        "policy": {
            "protocol": policy["protocol"],
            "open_images_prompt": "relation_center_forced_choice_v2",
            "visual_genome_prompt": (
                "visual_genome_semantic_forced_choice_v3"
            ),
            "selected_on_split": "dev",
        },
        "evaluation_protocol": (
            "single_complete_held_out_run_no_partial_test_evaluation"
        ),
        "test_prediction_status": "not_run",
        "input_sha256": {
            "test_questions_v1": sha256sum(INPUT_PATH),
            "questions_v1_manifest": sha256sum(INPUT_MANIFEST),
            "selected_policy": sha256sum(POLICY_PATH),
        },
        "artifact_sha256": {
            "test_questions.jsonl": bytes_sha256(questions_payload)
        },
    }
    artifacts = {
        "test_questions.jsonl": questions_payload,
        "manifest.json": json_bytes(manifest),
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists():
        for relative_path, payload in artifacts.items():
            path = OUTPUT_DIR / relative_path
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"Locked Test artifact differs: {path}")
        status = "verified"
    else:
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            raise RuntimeError(f"Refusing non-empty output directory: {OUTPUT_DIR}")
        for relative_path, payload in artifacts.items():
            write_atomic(OUTPUT_DIR / relative_path, payload)
        status = "created"

    print(f"Locked Hard-Test questions: {status}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
