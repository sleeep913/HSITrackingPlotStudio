"""Project cache records shared by every business module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CacheRecord:
    module: str
    cache_schema_version: int
    algorithm_version: str
    created_at: str
    input_fingerprint: str
    payload: dict[str, Any] | None = None
    payload_kind: str = "embedded_json"
    relative_path: str | None = None


@dataclass(slots=True)
class CacheManifest:
    records: dict[str, CacheRecord] = field(default_factory=dict)


__all__ = ["CacheManifest", "CacheRecord"]
