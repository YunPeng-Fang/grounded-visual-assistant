"""I/O helpers for experiments and result logging."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def utc_timestamp_slug() -> str:
    """Return a filesystem-safe timestamp."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def save_json(payload: dict[str, Any], output_dir: str | Path, prefix: str) -> Path:
    """Save a JSON payload with a timestamped filename."""
    out_dir = ensure_dir(output_dir)
    path = out_dir / f"{prefix}_{utc_timestamp_slug()}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

