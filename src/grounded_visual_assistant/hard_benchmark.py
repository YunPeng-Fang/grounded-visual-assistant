"""Validation and immutable freezing for the cross-dataset benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .hard_dataset import OPEN_IMAGES_SOURCE, VISUAL_GENOME_SOURCE
from .image_dedup import sha256sum


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON record at {source}:{line_number}."
                ) from error
    return records


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in records
    ).encode("utf-8")


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_unique_ids(records: list[dict[str, Any]], label: str) -> set[str]:
    ids = [str(item.get("sample_id", "")) for item in records]
    if not ids or any(not sample_id for sample_id in ids):
        raise ValueError(f"{label} contains a missing sample_id.")
    duplicates = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"{label} contains duplicate IDs: {duplicates[:5]}")
    return set(ids)


def _assert_same_ids(
    expected: set[str], observed: set[str], label: str
) -> None:
    if expected == observed:
        return
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    raise RuntimeError(
        f"{label} is stale or incomplete: missing={missing[:5]}, "
        f"extra={extra[:5]}, missing_count={len(missing)}, "
        f"extra_count={len(extra)}."
    )


def _resolve_project_path(project_root: Path, value: str) -> Path:
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    return path if path.is_absolute() else project_root / path


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def freeze_hard_benchmark(
    *,
    project_root: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    expected_count: int = 400,
) -> dict[str, Any]:
    """Validate all audited pixels and create or verify an immutable snapshot."""
    project_root = Path(project_root).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    audit_dir = dataset_dir / "image_audit"

    required = {
        "candidates": dataset_dir / "candidates.jsonl",
        "candidate_manifest": dataset_dir / "manifest.json",
        "downloads": audit_dir / "downloads.jsonl",
        "statuses": audit_dir / "sample_status.jsonl",
        "audit_summary": audit_dir / "summary.json",
        "accepted_ids": audit_dir / "accepted_sample_ids.json",
        "review_ids": audit_dir / "review_sample_ids.json",
        "excluded_ids": audit_dir / "excluded_sample_ids.json",
        "dev_ids": dataset_dir / "splits" / "dev_sample_ids.json",
        "test_ids": dataset_dir / "splits" / "test_sample_ids.json",
    }
    missing_files = [str(path) for path in required.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Missing freeze inputs:\n- " + "\n- ".join(missing_files))

    candidates = read_jsonl(required["candidates"])
    downloads = read_jsonl(required["downloads"])
    statuses = read_jsonl(required["statuses"])
    candidate_ids = _require_unique_ids(candidates, "Candidates")
    download_ids = _require_unique_ids(downloads, "Image downloads")
    status_ids = _require_unique_ids(statuses, "Image statuses")
    if len(candidate_ids) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} candidates, found {len(candidate_ids)}."
        )
    _assert_same_ids(candidate_ids, download_ids, "Image audit downloads")
    _assert_same_ids(candidate_ids, status_ids, "Image audit statuses")

    accepted_ids = {str(value) for value in _read_json(required["accepted_ids"])}
    review_ids = {str(value) for value in _read_json(required["review_ids"])}
    excluded_ids = {str(value) for value in _read_json(required["excluded_ids"])}
    _assert_same_ids(candidate_ids, accepted_ids, "Accepted image IDs")
    if review_ids or excluded_ids:
        raise RuntimeError(
            f"Image audit is unresolved: review={len(review_ids)}, "
            f"excluded={len(excluded_ids)}."
        )
    invalid_statuses = [
        item for item in statuses if item.get("status") != "accepted"
    ]
    if invalid_statuses:
        raise RuntimeError(
            f"Image audit contains {len(invalid_statuses)} non-accepted statuses."
        )

    summary = _read_json(required["audit_summary"])
    if summary.get("status") != "complete" or int(summary.get("accepted", -1)) != expected_count:
        raise RuntimeError("Image audit summary is not complete for all candidates.")

    dev_ids = {
        str(value) for value in _read_json(required["dev_ids"])["sample_ids"]
    }
    test_ids = {
        str(value) for value in _read_json(required["test_ids"])["sample_ids"]
    }
    if dev_ids & test_ids or dev_ids | test_ids != candidate_ids:
        raise RuntimeError("Dev/Test split files are not disjoint and complete.")
    for candidate in candidates:
        expected_split = "dev" if candidate["sample_id"] in dev_ids else "test"
        if candidate.get("split") != expected_split:
            raise RuntimeError(
                f"Candidate split mismatch for {candidate['sample_id']}."
            )

    source_counts = Counter(str(item["source"]) for item in candidates)
    expected_sources = {
        OPEN_IMAGES_SOURCE: expected_count // 2,
        VISUAL_GENOME_SOURCE: expected_count // 2,
    }
    if dict(source_counts) != expected_sources:
        raise RuntimeError(
            f"Expected balanced sources {expected_sources}, found {dict(source_counts)}."
        )

    download_by_id = {str(item["sample_id"]): item for item in downloads}
    frozen_images = []
    for candidate in sorted(candidates, key=lambda item: item["sample_id"]):
        audit = download_by_id[candidate["sample_id"]]
        image_path = _resolve_project_path(project_root, str(audit["path"]))
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Audited image is missing for {candidate['sample_id']}: {image_path}"
            )
        actual_sha256 = sha256sum(image_path)
        if actual_sha256 != audit.get("sha256"):
            raise RuntimeError(
                f"Image hash mismatch for {candidate['sample_id']}: "
                f"expected {audit.get('sha256')}, found {actual_sha256}."
            )
        relative_path = image_path.resolve().relative_to(project_root).as_posix()
        frozen_images.append(
            {
                "schema_version": 1,
                "sample_id": candidate["sample_id"],
                "source": candidate["source"],
                "source_image_id": candidate["source_image_id"],
                "split": candidate["split"],
                "path": relative_path,
                "sha256": actual_sha256,
                "dhash": audit["dhash"],
                "width": int(audit["width"]),
                "height": int(audit["height"]),
                "format": audit.get("format"),
                "mode": audit.get("mode"),
                "bytes": int(audit["bytes"]),
            }
        )

    candidate_payload = _jsonl_bytes(
        sorted(candidates, key=lambda item: item["sample_id"])
    )
    image_payload = _jsonl_bytes(frozen_images)
    dev_payload = _json_bytes(
        {"name": "hard_dev", "sample_ids": sorted(dev_ids)}
    )
    test_payload = _json_bytes(
        {"name": "hard_test", "sample_ids": sorted(test_ids)}
    )
    artifacts = {
        "candidates.jsonl": candidate_payload,
        "images.jsonl": image_payload,
        "splits/dev_sample_ids.json": dev_payload,
        "splits/test_sample_ids.json": test_payload,
    }
    manifest = {
        "name": "cross_dataset_hard_v1_frozen",
        "schema_version": 1,
        "immutable": True,
        "samples": expected_count,
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": {"dev": len(dev_ids), "test": len(test_ids)},
        "input_sha256": {
            name: sha256sum(path) for name, path in required.items()
        },
        "artifact_sha256": {
            name: _bytes_sha256(payload) for name, payload in artifacts.items()
        },
        "pixel_set_sha256": _bytes_sha256(
            _json_bytes(
                [
                    {"sample_id": item["sample_id"], "sha256": item["sha256"]}
                    for item in frozen_images
                ]
            )
        ),
    }
    manifest_payload = _json_bytes(manifest)
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_payload:
            raise RuntimeError(
                "Frozen benchmark already exists but current inputs differ. "
                "Create a new benchmark version instead of overwriting it."
            )
        for relative_path, payload in artifacts.items():
            path = output_dir / relative_path
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"Frozen artifact was modified: {path}")
        return {"status": "verified", "manifest": manifest}

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to populate non-empty freeze directory: {output_dir}"
        )
    for relative_path, payload in artifacts.items():
        _write_atomic(output_dir / relative_path, payload)
    _write_atomic(manifest_path, manifest_payload)
    return {"status": "created", "manifest": manifest}
