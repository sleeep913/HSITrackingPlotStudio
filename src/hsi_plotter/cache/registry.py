"""In-memory access and invalidation for project cache records."""

from __future__ import annotations

from hsi_plotter.models.cache import CacheManifest, CacheRecord


class CacheRegistry:
    def __init__(self, manifest: CacheManifest) -> None:
        self.manifest = manifest

    def put(self, record: CacheRecord) -> None:
        self.manifest.records[record.module] = record

    def get(self, module: str, input_fingerprint: str) -> CacheRecord | None:
        record = self.manifest.records.get(module)
        if record is None or record.input_fingerprint != input_fingerprint:
            return None
        return record

    def clear(self, module: str | None = None) -> None:
        if module is None:
            self.manifest.records.clear()
        else:
            self.manifest.records.pop(module, None)


__all__ = ["CacheRegistry"]
