from pathlib import Path

from hsi_plotter.cache.fingerprint import fingerprint_inputs
from hsi_plotter.cache.registry import CacheRegistry
from hsi_plotter.models.cache import CacheManifest, CacheRecord


def test_fingerprint_changes_with_file_metadata_but_not_parameter_order(tmp_path: Path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("first", encoding="utf-8")
    first = fingerprint_inputs((source,), {"algorithm": "v1", "threshold": 20})
    reordered = fingerprint_inputs((source,), {"threshold": 20, "algorithm": "v1"})
    source.write_text("second value", encoding="utf-8")
    changed = fingerprint_inputs((source,), {"algorithm": "v1", "threshold": 20})

    assert first == reordered
    assert changed != first


def test_registry_supports_hit_invalidation_and_clear() -> None:
    manifest = CacheManifest()
    registry = CacheRegistry(manifest)
    record = CacheRecord(
        module="attribute_evaluation",
        cache_schema_version=1,
        algorithm_version="attributes-v1",
        created_at="2026-10-02T00:00:00+08:00",
        input_fingerprint="abc",
        payload={"value": 1},
    )

    registry.put(record)

    assert registry.get("attribute_evaluation", "abc") == record
    assert registry.get("attribute_evaluation", "changed") is None
    registry.clear("attribute_evaluation")
    assert manifest.records == {}
