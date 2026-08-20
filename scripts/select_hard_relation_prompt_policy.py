"""Compare relation prompts and freeze the source-aware Hard-Dev policy."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.hard_benchmark import read_jsonl
from grounded_visual_assistant.relation_prompt_selection import (
    build_relation_prompt_selection,
    render_relation_prompt_selection,
)


V1_RUN = (
    PROJECT_ROOT
    / "outputs/cross_dataset_hard_v1/vlm/"
    "hard-dev400__qwen3-vl-8b-instruct"
)
V2_RUN = (
    PROJECT_ROOT
    / "outputs/cross_dataset_hard_v1/vlm/"
    "hard-dev200-relation__qwen3-vl-8b-instruct__prompt-v2"
)
V3_RUN = (
    PROJECT_ROOT
    / "outputs/cross_dataset_hard_v1/vlm/"
    "hard-dev100-vg-relation__qwen3-vl-8b-instruct__prompt-v3"
)
V3_MANIFEST = (
    PROJECT_ROOT
    / "data/cross_dataset_hard_v1/questions_v3_dev_vg/manifest.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/cross_dataset_hard_v1/relation_prompt_policy_dev_v1"
)


def sha256sum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in records
    ).encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def validate_run(run_dir: Path, expected_count: int) -> None:
    errors_path = run_dir / "errors.jsonl"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if errors_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"Run contains errors: {run_dir}")
    coverage = metrics["coverage"]
    if (
        metrics.get("status") != "completed"
        or int(coverage["expected"]) != expected_count
        or int(coverage["completed"]) != expected_count
        or int(coverage["remaining"]) != 0
    ):
        raise RuntimeError(f"Run is incomplete: {run_dir}")
    if config.get("required_split") != "dev":
        raise RuntimeError(f"Run was not restricted to Dev: {run_dir}")


def main() -> None:
    validate_run(V1_RUN, 400)
    validate_run(V2_RUN, 200)
    validate_run(V3_RUN, 100)
    v3_manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    summary, selected_policy, transitions = build_relation_prompt_selection(
        read_jsonl(V1_RUN / "predictions.jsonl"),
        read_jsonl(V2_RUN / "predictions.jsonl"),
        read_jsonl(V3_RUN / "predictions.jsonl"),
        v3_manifest,
    )

    input_paths = {
        "v1_predictions": V1_RUN / "predictions.jsonl",
        "v1_run_config": V1_RUN / "run_config.json",
        "v2_predictions": V2_RUN / "predictions.jsonl",
        "v2_run_config": V2_RUN / "run_config.json",
        "v3_predictions": V3_RUN / "predictions.jsonl",
        "v3_run_config": V3_RUN / "run_config.json",
        "v3_manifest": V3_MANIFEST,
    }
    input_sha256 = {
        name: sha256sum(path) for name, path in sorted(input_paths.items())
    }
    summary["input_sha256"] = input_sha256
    selected_policy["input_sha256"] = input_sha256
    artifacts = {
        "summary.json": json_bytes(summary),
        "selected_policy.json": json_bytes(selected_policy),
        "paired_transitions.jsonl": jsonl_bytes(transitions),
        "report.md": render_relation_prompt_selection(summary).encode("utf-8"),
    }

    if (OUTPUT_DIR / "selected_policy.json").exists():
        for relative_path, payload in artifacts.items():
            path = OUTPUT_DIR / relative_path
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"Locked policy artifact differs: {path}")
        status = "verified"
    else:
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            raise RuntimeError(f"Refusing non-empty output directory: {OUTPUT_DIR}")
        for relative_path, payload in artifacts.items():
            write_atomic(OUTPUT_DIR / relative_path, payload)
        status = "created"

    print(f"Source-aware relation policy: {status}")
    print(json.dumps(selected_policy, ensure_ascii=False, indent=2))
    print(f"Report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
