"""Image validation and conservative duplicate detection utilities."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageOps


def sha256sum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Return a 64-bit dHash after applying EXIF orientation."""
    normalized = ImageOps.exif_transpose(image).convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = list(normalized.getdata())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(first: str, second: str) -> int:
    if len(first) != len(second):
        raise ValueError("Perceptual hashes must use the same bit length.")
    return (int(first, 16) ^ int(second, 16)).bit_count()


def fingerprint_image(path: str | Path) -> dict[str, Any]:
    """Decode an image and return stable content and perceptual fingerprints."""
    image_path = Path(path)
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        width, height = image.size
        image_format = image.format
        mode = image.mode
        dhash = difference_hash(image)
    if width <= 0 or height <= 0:
        raise ValueError(f"Image has invalid dimensions: {image_path}")
    return {
        "sha256": sha256sum(image_path),
        "dhash": dhash,
        "width": width,
        "height": height,
        "format": image_format,
        "mode": mode,
        "bytes": image_path.stat().st_size,
    }


def _aspect_ratio(item: Mapping[str, Any]) -> float:
    return float(item["width"]) / float(item["height"])


def _near_duplicate(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    threshold: int,
    maximum_aspect_ratio_delta: float,
) -> tuple[bool, int]:
    distance = hamming_distance(str(first["dhash"]), str(second["dhash"]))
    first_ratio = _aspect_ratio(first)
    second_ratio = _aspect_ratio(second)
    ratio_delta = abs(first_ratio - second_ratio) / max(first_ratio, second_ratio)
    return distance <= threshold and ratio_delta <= maximum_aspect_ratio_delta, distance


def find_duplicate_pairs(
    selected: Iterable[Mapping[str, Any]],
    references: Iterable[Mapping[str, Any]],
    *,
    near_threshold: int = 4,
    maximum_aspect_ratio_delta: float = 0.03,
) -> list[dict[str, Any]]:
    """Find exact and review-only near duplicates across selected/reference sets."""
    selected = list(selected)
    references = list(references)
    pairs: list[dict[str, Any]] = []

    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            if left["sha256"] == right["sha256"]:
                kind = "exact"
                distance = 0
            else:
                is_near, distance = _near_duplicate(
                    left, right, near_threshold, maximum_aspect_ratio_delta
                )
                if not is_near:
                    continue
                kind = "near"
            pairs.append(
                {
                    "kind": kind,
                    "left_id": left["sample_id"],
                    "left_kind": "selected",
                    "right_id": right["sample_id"],
                    "right_kind": "selected",
                    "hamming_distance": distance,
                    "cross_split": left.get("split") != right.get("split"),
                }
            )

        for reference in references:
            if left["sha256"] == reference["sha256"]:
                kind = "exact"
                distance = 0
            else:
                is_near, distance = _near_duplicate(
                    left, reference, near_threshold, maximum_aspect_ratio_delta
                )
                if not is_near:
                    continue
                kind = "near"
            pairs.append(
                {
                    "kind": kind,
                    "left_id": left["sample_id"],
                    "left_kind": "selected",
                    "right_id": reference["reference_id"],
                    "right_kind": "reference",
                    "hamming_distance": distance,
                    "cross_split": False,
                }
            )
    return sorted(
        pairs,
        key=lambda item: (
            item["kind"] != "exact",
            item["left_id"],
            item["right_id"],
        ),
    )


def classify_samples(
    selected: Iterable[Mapping[str, Any]],
    pairs: Iterable[Mapping[str, Any]],
    *,
    invalid: Mapping[str, str] | None = None,
    dimension_mismatches: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Assign conservative keep, exclude, or manual-review statuses."""
    selected = list(selected)
    selected_by_id = {str(item["sample_id"]): item for item in selected}
    statuses = {sample_id: "accepted" for sample_id in selected_by_id}
    reasons: dict[str, list[str]] = defaultdict(list)

    for sample_id, error in (invalid or {}).items():
        if sample_id in statuses:
            statuses[sample_id] = "exclude_invalid_or_download_error"
            reasons[sample_id].append(error)

    exact_selected_groups: dict[str, list[str]] = defaultdict(list)
    for item in selected:
        exact_selected_groups[str(item["sha256"])].append(str(item["sample_id"]))
    for sample_ids in exact_selected_groups.values():
        if len(sample_ids) < 2:
            continue
        keeper = min(sample_ids)
        for sample_id in sorted(sample_ids):
            if sample_id == keeper or statuses[sample_id].startswith("exclude_"):
                continue
            statuses[sample_id] = "exclude_exact_selected_duplicate"
            reasons[sample_id].append(f"exact duplicate of selected sample {keeper}")

    for pair in pairs:
        left_id = str(pair["left_id"])
        if left_id not in statuses:
            continue
        if pair["kind"] == "exact" and pair["right_kind"] == "reference":
            statuses[left_id] = "exclude_exact_reference_duplicate"
            reasons[left_id].append(
                f"exact duplicate of reference image {pair['right_id']}"
            )
        elif pair["kind"] == "near":
            involved = [left_id]
            if pair["right_kind"] == "selected":
                involved.append(str(pair["right_id"]))
            for sample_id in involved:
                if sample_id in statuses and statuses[sample_id] == "accepted":
                    statuses[sample_id] = "review_near_duplicate"
                if sample_id in statuses:
                    reasons[sample_id].append(
                        "near duplicate candidate: "
                        f"{pair['left_id']} vs {pair['right_id']} "
                        f"(dHash distance {pair['hamming_distance']})"
                    )

    for sample_id in dimension_mismatches:
        if sample_id in statuses and statuses[sample_id] == "accepted":
            statuses[sample_id] = "review_dimension_mismatch"
        if sample_id in statuses:
            reasons[sample_id].append("downloaded dimensions differ from metadata")

    return [
        {
            "sample_id": sample_id,
            "source": selected_by_id[sample_id].get("source"),
            "split": selected_by_id[sample_id].get("split"),
            "status": statuses[sample_id],
            "reasons": reasons.get(sample_id, []),
        }
        for sample_id in sorted(statuses)
    ]


def status_statistics(statuses: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = list(statuses)
    return {
        "samples": len(statuses),
        "status_counts": dict(
            sorted(Counter(str(item["status"]) for item in statuses).items())
        ),
        "source_counts": dict(
            sorted(Counter(str(item["source"]) for item in statuses).items())
        ),
        "split_counts": dict(
            sorted(Counter(str(item["split"]) for item in statuses).items())
        ),
    }

