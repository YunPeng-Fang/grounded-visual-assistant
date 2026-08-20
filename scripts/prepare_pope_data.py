"""Download and audit the official COCO POPE metadata and referenced images."""

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
from typing import Any, Callable

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.pope_dataset import (
    POPE_STRATEGIES,
    build_image_manifest,
    normalize_questions,
    question_statistics,
    read_json_records,
    sha256sum,
)


OFFICIAL_REPOSITORY = "https://github.com/RUCAIBox/POPE"
EXPECTED_QUESTIONS_PER_STRATEGY = 3000
EXPECTED_IMAGES_PER_STRATEGY = 500
EXPECTED_UNIQUE_IMAGES = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the official COCO POPE benchmark without downloading the "
            "complete COCO val2014 archive."
        )
    )
    parser.add_argument("--output-dir", default="data/pope")
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision used for official POPE metadata URLs.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--backend",
        choices=("auto", "curl", "urllib"),
        default="auto",
    )
    parser.add_argument(
        "--image-transport",
        choices=("https", "http"),
        default="https",
        help="Use http when a local proxy breaks COCO HTTPS certificates.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download and validate question files without downloading images.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Perform no network requests and validate existing local files.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Download only the first N images for a resumable smoke run.",
    )
    parser.add_argument(
        "--allow-nonstandard",
        action="store_true",
        help="Do not enforce the official 3000-question/500-image counts.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary.replace(path)


def resolve_backend(value: str) -> str:
    if value != "auto":
        return value
    return (
        "curl"
        if shutil.which("curl.exe") or shutil.which("curl")
        else "urllib"
    )


def safe_target(output_dir: Path, relative_path: str) -> Path:
    root = output_dir.resolve()
    target = (output_dir / relative_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"POPE path escapes output directory: {relative_path}")
    return target


def validate_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return {
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "format": image_format,
        "sha256": sha256sum(path),
    }


def download_file(
    url: str,
    target: Path,
    *,
    retries: int,
    timeout: int,
    backend: str,
    audit_only: bool,
    validator: Callable[[Path], Any] | None = None,
) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        try:
            if validator is not None:
                validator(target)
            return "reused"
        except Exception:
            if audit_only:
                raise
            target.unlink()
    if audit_only:
        raise FileNotFoundError(f"Required POPE file is missing: {target}")

    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if backend == "curl":
                curl = shutil.which("curl.exe") or shutil.which("curl")
                if curl is None:
                    raise RuntimeError("curl backend selected but curl is missing")
                command = [
                    curl,
                    "--location",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    str(timeout),
                    "--max-time",
                    str(timeout * 2),
                    "--output",
                    str(partial),
                    url,
                ]
                if sys.platform == "win32":
                    command.insert(1, "--ssl-no-revoke")
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout * 2 + 15,
                )
            else:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(
                    request, timeout=timeout
                ) as response, partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise ValueError("downloaded file is empty")
            partial.replace(target)
            if validator is not None:
                validator(target)
            return "downloaded"
        except Exception as error:
            last_error = error
            if partial.exists():
                partial.unlink()
            if target.exists():
                target.unlink()
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"Failed to download {url}: {type(last_error).__name__}: {last_error}"
    )


def metadata_url(revision: str, strategy: str) -> str:
    file_name = f"coco_pope_{strategy}.json"
    return (
        "https://raw.githubusercontent.com/RUCAIBox/POPE/"
        f"{revision}/output/coco/{file_name}"
    )


def require_official_counts(statistics: dict[str, Any]) -> None:
    if statistics["questions"] != (
        EXPECTED_QUESTIONS_PER_STRATEGY * len(POPE_STRATEGIES)
    ):
        raise RuntimeError(
            f"Expected 9000 POPE questions, found {statistics['questions']}."
        )
    if statistics["images"] != EXPECTED_UNIQUE_IMAGES:
        raise RuntimeError(
            f"Expected 500 unique POPE images, found {statistics['images']}."
        )
    for strategy, values in statistics["per_strategy"].items():
        if values["questions"] != EXPECTED_QUESTIONS_PER_STRATEGY:
            raise RuntimeError(
                f"Expected 3000 {strategy} questions, "
                f"found {values['questions']}."
            )
        if values["images"] != EXPECTED_IMAGES_PER_STRATEGY:
            raise RuntimeError(
                f"Expected 500 {strategy} images, found {values['images']}."
            )
        if values["yes"] != values["no"]:
            raise RuntimeError(
                f"POPE {strategy} labels are not balanced: "
                f"{values['yes']} yes / {values['no']} no."
            )


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retries < 1 or args.timeout < 1:
        raise ValueError("workers, retries, and timeout must be positive.")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be positive.")
    backend = resolve_backend(args.backend)
    output_dir = project_path(args.output_dir)
    annotations_dir = output_dir / "annotations"
    questions_by_strategy = {}
    metadata_states = {}

    for strategy in POPE_STRATEGIES:
        path = annotations_dir / f"coco_pope_{strategy}.json"
        state = download_file(
            metadata_url(args.revision, strategy),
            path,
            retries=args.retries,
            timeout=args.timeout,
            backend=backend,
            audit_only=args.audit_only,
            validator=read_json_records,
        )
        metadata_states[strategy] = state
        questions_by_strategy[strategy] = normalize_questions(
            read_json_records(path), strategy
        )

    statistics = question_statistics(questions_by_strategy)
    if not args.allow_nonstandard:
        require_official_counts(statistics)
    questions = [
        item
        for strategy in POPE_STRATEGIES
        for item in questions_by_strategy[strategy]
    ]
    image_manifest = build_image_manifest(
        questions_by_strategy,
        image_base_url=(
            f"{args.image_transport}://images.cocodataset.org/val2014"
        ),
    )
    write_jsonl(output_dir / "questions.jsonl", questions)
    write_jsonl(output_dir / "image_manifest.jsonl", image_manifest)

    selected_images = (
        image_manifest[: args.max_images]
        if args.max_images is not None
        else image_manifest
    )
    image_results = []
    if not args.metadata_only:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for item in selected_images:
                target = safe_target(output_dir, item["relative_path"])
                future = executor.submit(
                    download_file,
                    item["url"],
                    target,
                    retries=args.retries,
                    timeout=args.timeout,
                    backend=backend,
                    audit_only=args.audit_only,
                    validator=validate_image,
                )
                futures[future] = (item, target)
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="POPE images",
            ):
                item, target = futures[future]
                try:
                    state = future.result()
                    image_results.append(
                        {
                            "image_file": item["image_file"],
                            "image_id": item["image_id"],
                            "path": str(target.relative_to(PROJECT_ROOT)),
                            "state": state,
                            **validate_image(target),
                        }
                    )
                except Exception as error:
                    image_results.append(
                        {
                            "image_file": item["image_file"],
                            "image_id": item["image_id"],
                            "path": str(target.relative_to(PROJECT_ROOT)),
                            "state": "error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        image_results.sort(key=lambda item: item["image_file"])
    write_jsonl(output_dir / "image_audit.jsonl", image_results)

    image_errors = [
        item for item in image_results if item["state"] == "error"
    ]
    complete_images = len(image_results) == len(image_manifest) and not image_errors
    status = (
        "metadata_ready"
        if args.metadata_only
        else "complete"
        if complete_images
        else "partial"
    )
    artifact_paths = {
        "questions.jsonl": output_dir / "questions.jsonl",
        "image_manifest.jsonl": output_dir / "image_manifest.jsonl",
        "image_audit.jsonl": output_dir / "image_audit.jsonl",
    }
    summary = {
        "status": status,
        "source": OFFICIAL_REPOSITORY,
        "revision": args.revision,
        "download_backend": backend,
        "image_transport": args.image_transport,
        "metadata_states": metadata_states,
        **statistics,
        "images_selected_this_run": len(selected_images),
        "images_validated_this_run": sum(
            item["state"] != "error" for item in image_results
        ),
        "image_errors": len(image_errors),
        "metadata_only": args.metadata_only,
        "audit_only": args.audit_only,
    }
    write_json(output_dir / "summary.json", summary)
    manifest = {
        "protocol": "official_coco_pope_selective_download_v1",
        "source": OFFICIAL_REPOSITORY,
        "revision": args.revision,
        "image_transport": args.image_transport,
        "metadata_urls": {
            strategy: metadata_url(args.revision, strategy)
            for strategy in POPE_STRATEGIES
        },
        "metadata_sha256": {
            strategy: sha256sum(
                annotations_dir / f"coco_pope_{strategy}.json"
            )
            for strategy in POPE_STRATEGIES
        },
        "artifact_sha256": {
            name: sha256sum(path)
            for name, path in sorted(artifact_paths.items())
        },
        "statistics": statistics,
        "image_download": {
            "expected": len(image_manifest),
            "validated_this_run": summary["images_validated_this_run"],
            "errors": len(image_errors),
            "complete": complete_images,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Manifest: {output_dir / 'manifest.json'}")
    if image_errors:
        raise RuntimeError(
            f"{len(image_errors)} POPE images failed; inspect image_audit.jsonl."
        )
    if args.audit_only and not complete_images:
        raise RuntimeError("POPE audit is incomplete.")


if __name__ == "__main__":
    main()
