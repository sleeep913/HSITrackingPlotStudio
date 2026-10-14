from pathlib import Path

from hsi_plotter.io.cache_store import cache_directory, clear_sidecar_cache, write_json_cache


def test_sidecar_cache_uses_project_stem_and_round_trips_json(tmp_path: Path) -> None:
    project_path = tmp_path / "paper.hsiproj"
    relative = write_json_cache(project_path, "visualization", {"frames": [1, 2]})

    target = project_path.parent / relative
    assert cache_directory(project_path).name == "paper.hsicache"
    assert target.is_file()
    assert target.read_text(encoding="utf-8").startswith("{")

    clear_sidecar_cache(project_path)
    assert not cache_directory(project_path).exists()
