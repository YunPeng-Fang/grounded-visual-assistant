"""Build or audit the COCO verifier Dev110 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_dataset import sha256sum
from grounded_visual_assistant.verifier_dev_dataset import (
    VERIFIER_DEV_PROTOCOL,
    build_verifier_dev_records,
    records_sha256,
    validate_verifier_dev_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced COCO verifier development set with no POPE "
            "image overlap."
        )
    )
    parser.add_argument(
        "--coco-ground-truth",
        default="data/eval_v0/coco_grounding_gt.json",
    )
    parser.add_argument(
        "--dev-split", default="data/eval_v0/splits/dev_image_ids.json"
    )
    parser.add_argument(
        "--pope-questions", default="data/pope/questions.jsonl"
    )
    parser.add_argument(
        "--output-dir", default="data/verifier_dev_v1"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}."
                ) from error
    return records


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(
    path: Path, records: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(
                json.dumps(dict(item), ensure_ascii=False) + "\n"
            )
    temporary.replace(path)


def ordered_ids_sha256(values: Iterable[object]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_sources(
    args: argparse.Namespace,
) -> tuple[
    Path,
    Path,
    Path,
    dict[str, Any],
    list[int],
    set[int],
]:
    coco_path = project_path(args.coco_ground_truth)
    split_path = project_path(args.dev_split)
    pope_path = project_path(args.pope_questions)
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("name") != "dev":
        raise RuntimeError("Verifier development input is not the Dev split.")
    dev_ids = [int(value) for value in split["image_ids"]]
    pope_ids = {
        int(item["image_id"]) for item in read_jsonl(pope_path)
    }
    return coco_path, split_path, pope_path, coco, dev_ids, pope_ids


def audit(args: argparse.Namespace) -> dict[str, Any]:
    (
        coco_path,
        split_path,
        pope_path,
        coco,
        dev_ids,
        pope_ids,
    ) = load_sources(args)
    output_dir = project_path(args.output_dir)
    questions_path = output_dir / "questions.jsonl"
    manifest_path = output_dir / "manifest.json"
    if not questions_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Verifier Dev questions.jsonl and manifest.json are required."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != VERIFIER_DEV_PROTOCOL:
        raise RuntimeError("Verifier Dev manifest protocol mismatch.")
    source_checks = {
        "coco_ground_truth_sha256": sha256sum(coco_path),
        "dev_split_sha256": sha256sum(split_path),
        "pope_questions_sha256": sha256sum(pope_path),
    }
    for field, actual in source_checks.items():
        if manifest.get("inputs", {}).get(field) != actual:
            raise RuntimeError(f"Verifier Dev source hash mismatch: {field}.")
    actual_questions_hash = sha256sum(questions_path)
    if (
        manifest.get("artifact_sha256", {}).get("questions.jsonl")
        != actual_questions_hash
    ):
        raise RuntimeError("Verifier Dev questions hash mismatch.")
    records = read_jsonl(questions_path)
    selected_ids = [
        image_id for image_id in dev_ids if image_id not in pope_ids
    ]
    validation = validate_verifier_dev_records(
        records,
        coco_ground_truth=coco,
        allowed_image_ids=selected_ids,
        excluded_image_ids=pope_ids,
    )
    if records_sha256(records) != manifest.get("records_sha256"):
        raise RuntimeError("Verifier Dev canonical records hash mismatch.")
    for image_id in selected_ids:
        image_path = (
            PROJECT_ROOT
            / "data"
            / "eval_v0"
            / "images"
            / f"{image_id:012d}.jpg"
        )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Verifier Dev image is missing: {image_path}"
            )
    return {
        "status": "verified",
        **validation,
        "questions_sha256": actual_questions_hash,
        "manifest_sha256": sha256sum(manifest_path),
        "excluded_pope_overlap": sorted(set(dev_ids).intersection(pope_ids)),
    }


def main() -> None:
    args = parse_args()
    if args.audit_only:
        print(json.dumps(audit(args), ensure_ascii=False, indent=2))
        return
    (
        coco_path,
        split_path,
        pope_path,
        coco,
        dev_ids,
        pope_ids,
    ) = load_sources(args)
    records, summary = build_verifier_dev_records(
        coco,
        dev_image_ids=dev_ids,
        excluded_image_ids=pope_ids,
        seed=args.seed,
    )
    output_dir = project_path(args.output_dir)
    questions_path = output_dir / "questions.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl_atomic(questions_path, records)
    manifest = {
        "protocol": VERIFIER_DEV_PROTOCOL,
        "seed": args.seed,
        "selection": {
            "positive": (
                "every COCO-annotated image/category pair in eligible images"
            ),
            "negative": (
                "least-used absent category from the same official COCO "
                "supercategory, with deterministic balanced fallback"
            ),
            "pope_isolation": (
                "exclude every image ID referenced by official POPE Full500"
            ),
        },
        "inputs": {
            "coco_ground_truth": portable_path(coco_path),
            "coco_ground_truth_sha256": sha256sum(coco_path),
            "dev_split": portable_path(split_path),
            "dev_split_sha256": sha256sum(split_path),
            "pope_questions": portable_path(pope_path),
            "pope_questions_sha256": sha256sum(pope_path),
            "pope_image_count": len(pope_ids),
        },
        "summary": summary,
        "ordered_record_ids_sha256": ordered_ids_sha256(
            item["id"] for item in records
        ),
        "records_sha256": records_sha256(records),
    }
    write_json_atomic(manifest_path, manifest)
    manifest["artifact_sha256"] = {
        "questions.jsonl": sha256sum(questions_path)
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(audit(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
