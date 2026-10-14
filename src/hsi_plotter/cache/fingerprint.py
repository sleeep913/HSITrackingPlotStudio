"""Deterministic lightweight fingerprints for source files and parameters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _file_entries(paths: Sequence[Path]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for raw_path in sorted((Path(value) for value in paths), key=lambda item: str(item).casefold()):
        path = raw_path.resolve(strict=False)
        if path.is_dir():
            files = sorted(
                (item for item in path.rglob("*") if item.is_file()),
                key=lambda item: item.relative_to(path).as_posix().casefold(),
            )
            for item in files:
                stat = item.stat()
                entries.append(
                    {
                        "root": str(path),
                        "relative": item.relative_to(path).as_posix(),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
        elif path.is_file():
            stat = path.stat()
            entries.append(
                {
                    "root": str(path.parent),
                    "relative": path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        else:
            entries.append({"root": str(path), "missing": True})
    return entries


def fingerprint_inputs(
    paths: Sequence[Path], parameters: Mapping[str, Any] | None = None
) -> str:
    payload = {
        "files": _file_entries(paths),
        "parameters": dict(parameters or {}),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["fingerprint_inputs"]
