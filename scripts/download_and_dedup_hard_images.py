"""Download selected hard-case images and create a conservative dedup audit."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from grounded_visual_assistant.image_dedup import (
    classify_samples,
    find_duplicate_pairs,
    fingerprint_image,
    status_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the selected hard-case images, validate them, and audit "
            "exact/perceptual duplicates without auto-removing near matches."
        )
    )
    parser.add_argument(
        "--manifest", default="data/cross_dataset_hard_v1/download_manifest.jsonl"
    )
    parser.add_argument(
        "--candidates", default="data/cross_dataset_hard_v1/candidates.jsonl"
    )
    parser.add_argument("--reference-images", default="data/eval_v0/images")
    parser.add_argument("--output-dir", default="data/cross_dataset_hard_v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument(
        "--backend",
        choices=("auto", "curl", "urllib"),
        default="auto",
        help="auto prefers curl when available and falls back to urllib.",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def safe_target(output_dir: Path, relative_path: str) -> Path:
    root = output_dir.resolve()
    target = (output_dir / relative_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Image path escapes the dataset directory: {relative_path}")
    return target


def download_one(
    item: dict[str, Any],
    output_dir: Path,
    retries: int,
    timeout: int,
    backend: str,
    audit_only: bool,
) -> dict[str, Any]:
    target = safe_target(output_dir, str(item["relative_path"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        try:
            fingerprint_image(target)
            return {"sample_id": item["sample_id"], "state": "reused", "path": target}
        except Exception:
            if audit_only:
                return {
                    "sample_id": item["sample_id"],
                    "state": "error",
                    "path": target,
                    "error": "existing image is invalid and --audit-only was set",
                }
            target.unlink()
    if audit_only:
        return {
            "sample_id": item["sample_id"],
            "state": "error",
            "path": target,
            "error": "image is missing and --audit-only was set",
        }

    partial = target.with_suffix(target.suffix + ".part")
    last_error = "unknown download error"
    if backend == "curl":
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if curl is None:
            return {
                "sample_id": item["sample_id"],
                "state": "error",
                "path": target,
                "error": "curl backend requested but curl was not found",
            }
        command = [
            curl,
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            str(max(0, retries - 1)),
            "--connect-timeout",
            str(timeout),
            "--max-time",
            str(timeout * max(2, retries)),
            "--output",
            str(partial),
            str(item["url"]),
        ]
        if sys.platform == "win32":
            command.insert(1, "--ssl-no-revoke")
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout * max(2, retries) + 15,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    completed.stderr.strip()
                    or f"curl exited with code {completed.returncode}"
                )
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise ValueError("downloaded file is empty")
            partial.replace(target)
            fingerprint_image(target)
            return {
                "sample_id": item["sample_id"],
                "state": "downloaded",
                "path": target,
            }
        except Exception as error:
            if partial.exists():
                partial.unlink()
            if target.exists():
                target.unlink()
            return {
                "sample_id": item["sample_id"],
                "state": "error",
                "path": target,
                "error": f"{type(error).__name__}: {error}",
            }

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                str(item["url"]), headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if partial.stat().st_size <= 0:
                raise ValueError("downloaded file is empty")
            partial.replace(target)
            fingerprint_image(target)
            return {
                "sample_id": item["sample_id"],
                "state": "downloaded",
                "path": target,
            }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            if partial.exists():
                partial.unlink()
            if target.exists():
                target.unlink()
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return {
        "sample_id": item["sample_id"],
        "state": "error",
        "path": target,
        "error": last_error,
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retries < 1:
        raise ValueError("workers and retries must both be positive.")
    backend = args.backend
    if backend == "auto":
        backend = (
            "curl"
            if shutil.which("curl.exe") or shutil.which("curl")
            else "urllib"
        )
    manifest_path = project_path(args.manifest)
    candidates_path = project_path(args.candidates)
    reference_dir = project_path(args.reference_images)
    output_dir = project_path(args.output_dir)
    audit_dir = output_dir / "image_audit"
    manifest = read_jsonl(manifest_path)
    candidates = read_jsonl(candidates_path)
    candidate_by_id = {item["sample_id"]: item for item in candidates}
    if len(manifest) != len(candidate_by_id):
        raise ValueError("Download manifest and candidate count differ.")
    if {item["sample_id"] for item in manifest} != set(candidate_by_id):
        raise ValueError("Download manifest and candidate IDs differ.")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                item,
                output_dir,
                args.retries,
                args.timeout,
                backend,
                args.audit_only,
            ): item["sample_id"]
            for item in manifest
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading images"
        ):
            results.append(future.result())
    results.sort(key=lambda item: item["sample_id"])

    invalid = {
        item["sample_id"]: item["error"]
        for item in results
        if item["state"] == "error"
    }
    selected_audit = []
    dimension_mismatches = []
    for result in results:
        if result["sample_id"] in invalid:
            continue
        fingerprint = fingerprint_image(result["path"])
        candidate = candidate_by_id[result["sample_id"]]
        expected_width = candidate["image"].get("width")
        expected_height = candidate["image"].get("height")
        dimension_matches = True
        if expected_width and expected_height:
            actual = (fingerprint["width"], fingerprint["height"])
            expected = (int(expected_width), int(expected_height))
            dimension_matches = actual in {expected, expected[::-1]}
            if not dimension_matches:
                dimension_mismatches.append(result["sample_id"])
        selected_audit.append(
            {
                "sample_id": result["sample_id"],
                "source": candidate["source"],
                "split": candidate["split"],
                "path": str(Path(result["path"]).relative_to(PROJECT_ROOT)),
                "download_state": result["state"],
                "dimension_matches_metadata": dimension_matches,
                **fingerprint,
            }
        )

    reference_audit = []
    if reference_dir.is_dir():
        for path in sorted(reference_dir.iterdir()):
            if not path.is_file():
                continue
            try:
                reference_audit.append(
                    {
                        "reference_id": f"coco:{path.stem}",
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        **fingerprint_image(path),
                    }
                )
            except Exception as error:
                print(f"Skipping invalid reference image {path}: {error}")

    pairs = find_duplicate_pairs(
        selected_audit,
        reference_audit,
        near_threshold=args.near_threshold,
    )
    statuses = classify_samples(
        selected_audit,
        pairs,
        invalid=invalid,
        dimension_mismatches=dimension_mismatches,
    )
    status_by_id = {item["sample_id"]: item for item in statuses}
    for item in manifest:
        if item["sample_id"] not in status_by_id:
            statuses.append(
                {
                    "sample_id": item["sample_id"],
                    "source": item["source"],
                    "split": item["split"],
                    "status": "exclude_invalid_or_download_error",
                    "reasons": [invalid.get(item["sample_id"], "missing audit record")],
                }
            )
    statuses.sort(key=lambda item: item["sample_id"])

    accepted = [
        item["sample_id"] for item in statuses if item["status"] == "accepted"
    ]
    review = [
        item["sample_id"]
        for item in statuses
        if item["status"].startswith("review_")
    ]
    excluded = [
        item["sample_id"]
        for item in statuses
        if item["status"].startswith("exclude_")
    ]
    summary = {
        "status": "review_required" if review else "complete",
        "downloaded": sum(item["state"] == "downloaded" for item in results),
        "reused": sum(item["state"] == "reused" for item in results),
        "download_errors": len(invalid),
        "reference_images": len(reference_audit),
        "near_threshold": args.near_threshold,
        "download_backend": backend,
        "duplicate_pairs": {
            "exact": sum(item["kind"] == "exact" for item in pairs),
            "near": sum(item["kind"] == "near" for item in pairs),
            "cross_split": sum(item["cross_split"] for item in pairs),
        },
        "samples": status_statistics(statuses),
        "accepted": len(accepted),
        "review": len(review),
        "excluded": len(excluded),
        "next_required_step": (
            "Review flagged samples and replenish exclusions before freezing."
            if review or excluded
            else "Freeze the validated image manifest, then generate source-aware questions."
        ),
    }
    write_jsonl(audit_dir / "downloads.jsonl", selected_audit)
    write_jsonl(audit_dir / "reference_images.jsonl", reference_audit)
    write_jsonl(audit_dir / "duplicate_pairs.jsonl", pairs)
    write_jsonl(audit_dir / "sample_status.jsonl", statuses)
    write_json(audit_dir / "accepted_sample_ids.json", accepted)
    write_json(audit_dir / "review_sample_ids.json", review)
    write_json(audit_dir / "excluded_sample_ids.json", excluded)
    write_json(audit_dir / "summary.json", summary)
    print(f"Download backend: {backend}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Audit: {audit_dir}")


if __name__ == "__main__":
    main()
