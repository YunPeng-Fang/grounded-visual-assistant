"""Portable paths for evaluation artifacts copied between machines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def portable_project_path(value: str | Path, project_root: str | Path) -> str:
    """Return a POSIX project-relative path when the artifact is in the repo."""
    raw = str(value)
    root = Path(project_root).resolve()
    normalized = raw.replace("\\", "/")
    marker = f"/{root.name}/"
    if marker in normalized:
        return normalized.split(marker, 1)[1].lstrip("/")

    path = Path(raw)
    if not path.is_absolute():
        return path.as_posix()

    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        pass
    return normalized


def portable_gallery(
    gallery: Iterable[Any], project_root: str | Path
) -> list[list[Any]]:
    """Convert gallery path/caption pairs to JSON-friendly relative paths."""
    converted = []
    for item in gallery:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid gallery item: {item!r}")
        converted.append(
            [portable_project_path(item[0], project_root), item[1]]
        )
    return converted


def resolve_project_path(value: str | Path, project_root: str | Path) -> Path:
    """Resolve a stored portable path against a project checkout."""
    path = Path(value)
    return path if path.is_absolute() else Path(project_root).resolve() / path
