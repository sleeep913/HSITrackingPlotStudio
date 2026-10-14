import json
from pathlib import Path

import pytest

from hsi_plotter.io.project_store import ProjectFormatError, load_project, save_project
from hsi_plotter.models.cache import CacheManifest, CacheRecord
from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.tracker_import import ResultLayout


def test_project_round_trip_uses_paths_relative_to_project_file(tmp_path: Path) -> None:
    project_dir = tmp_path / "paper"
    project_dir.mkdir()
    ground_truth = project_dir / "data" / "gt"
    results = project_dir / "data" / "results"
    project = Project(
        name="HOT2023",
        sources=ProjectSources(
            ground_truth=ground_truth,
            results_root=results,
            export_directory=project_dir / "figures",
        ),
        trackers=[
            Tracker(
                tracker_id="mht",
                display_name="MHT",
                results_path=results / "MHT",
                style=TrackerStyle(color="#D95319", line_style="-", line_width=2.5),
            )
        ],
    )
    destination = project_dir / "experiment.hsiproj"

    save_project(project, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    loaded = load_project(destination)

    assert payload["schema_version"] == 4
    assert payload["plot_style"]["preset"] == "paper_double_column"
    assert payload["trackers"][0]["layout"] == "flat"
    assert payload["sources"]["ground_truth"] == "data/gt"
    assert payload["trackers"][0]["results_path"] == "data/results/MHT"
    assert loaded == project


def test_project_round_trip_preserves_cache_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "cached.hsiproj"
    record = CacheRecord(
        module="attribute_evaluation",
        cache_schema_version=1,
        algorithm_version="attributes-v1",
        created_at="2026-10-02T00:00:00+08:00",
        input_fingerprint="fingerprint",
        payload={"attributes": []},
    )
    project = Project(
        name="cached",
        cache_manifest=CacheManifest(records={record.module: record}),
    )

    save_project(project, destination)
    loaded = load_project(destination)

    assert loaded.cache_manifest == project.cache_manifest


def test_project_save_externalizes_and_save_as_copies_sidecar_cache(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.hsiproj"
    second = tmp_path / "second.hsiproj"
    record = CacheRecord(
        module="tracking_visualization_index",
        cache_schema_version=1,
        algorithm_version="visualization-v1",
        created_at="2026-10-02T00:00:00+08:00",
        input_fingerprint="fingerprint",
        payload={"sequences": [{"name": "ball"}]},
        payload_kind="sidecar_json",
    )
    project = Project(
        name="cached",
        cache_manifest=CacheManifest(records={record.module: record}),
    )

    save_project(project, first)
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    first_sidecar = tmp_path / first_payload["cache_manifest"]["records"][record.module][
        "relative_path"
    ]
    assert first_sidecar.is_file()
    assert first_payload["cache_manifest"]["records"][record.module]["payload"] is None

    save_project(project, second)
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    second_sidecar = tmp_path / second_payload["cache_manifest"]["records"][record.module][
        "relative_path"
    ]
    assert second_sidecar.is_file()
    assert json.loads(second_sidecar.read_text(encoding="utf-8")) == {
        "sequences": [{"name": "ball"}]
    }


def test_load_project_still_accepts_schema_version_one(tmp_path: Path) -> None:
    path = tmp_path / "legacy.hsiproj"
    results_root = tmp_path / "results"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "legacy",
                "sources": {"results_root": str(results_root)},
                "trackers": [
                    {
                        "tracker_id": "mht",
                        "display_name": "MHT",
                        "results_path": str(results_root / "MHT"),
                        "style": {"color": "#2563EB", "line_style": "-", "line_width": 2.0},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    project = load_project(path)

    assert project.trackers[0].layout is ResultLayout.FLAT


def test_project_round_trip_preserves_project_plot_style(tmp_path: Path) -> None:
    from hsi_plotter.models.style import PlotStyle

    destination = tmp_path / "styled.hsiproj"
    project = Project(
        name="styled",
        plot_style=PlotStyle(
            font_family="Arial",
            width_inches=7.5,
            dpi=240,
            legend_scale=1.25,
        ),
    )

    save_project(project, destination)
    loaded = load_project(destination)

    assert loaded.plot_style == project.plot_style


def test_project_round_trip_preserves_tracker_layout(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    destination = tmp_path / "study.hsiproj"
    project = Project(
        name="HOT2023",
        sources=ProjectSources(results_root=results_root),
        trackers=[
            Tracker(
                tracker_id="mht",
                display_name="MHT",
                results_path=results_root / "MHT",
                layout=ResultLayout.MODALITY_SUBDIR,
                style=TrackerStyle(color="#2563EB"),
            )
        ],
    )

    save_project(project, destination)
    loaded = load_project(destination)

    assert loaded.trackers[0].layout is ResultLayout.MODALITY_SUBDIR
    assert loaded == project


def test_load_project_rejects_unknown_tracker_layout(tmp_path: Path) -> None:
    path = tmp_path / "broken.hsiproj"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "name": "broken",
                "sources": {},
                "trackers": [
                    {
                        "tracker_id": "mht",
                        "display_name": "MHT",
                        "results_path": "results/MHT",
                        "layout": "unknown-layout",
                        "style": {"color": "#2563EB", "line_style": "-", "line_width": 2.0},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProjectFormatError, match="layout"):
        load_project(path)


def test_load_project_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "future.hsiproj"
    path.write_text('{"schema_version": 999, "name": "future"}', encoding="utf-8")

    with pytest.raises(ProjectFormatError, match="schema_version"):
        load_project(path)


def test_load_project_reports_invalid_json_without_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "broken.hsiproj"
    original = "{not-json"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ProjectFormatError, match="JSON"):
        load_project(path)

    assert path.read_text(encoding="utf-8") == original
