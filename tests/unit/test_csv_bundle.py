from pathlib import Path

import pytest

from hsi_plotter.io.csv_bundle import CsvBundleError, load_csv_bundle
from hsi_plotter.io.sequence_discovery import discover_sequence_files
from hsi_plotter.io.tracker_discovery import discover_trackers
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.tracker_import import ResultLayout


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_submission_csv_and_materializes_sequences(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "submission.csv",
        "ID,x,y,width,height\n"
        "vis-ball_1,1,2,30,40\n"
        "vis-ball_2,2,3,30,40\n"
        "nir-car_one_1,4,5,20,10\n",
    )

    bundle = load_csv_bundle(source, dataset="project", cache_root=tmp_path / "cache")

    assert bundle.sequence_count == 2
    assert bundle.frame_count == 3
    ball = SequenceKey("project", Modality.VIS, "ball")
    car = SequenceKey("project", Modality.NIR, "car_one")
    assert bundle.files[ball].read_text(encoding="utf-8").splitlines() == [
        "1,2,30,40",
        "2,3,30,40",
    ]
    assert bundle.files[car].is_file()


def test_loads_canonical_ground_truth_csv(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "groundtruth.csv",
        "dataset,modality,sequence,frame,x,y,width,height\n"
        "HOTC2024,rednir,coin,1,1,2,3,4\n",
    )

    result = discover_sequence_files(source, dataset="project")

    assert not result.report.has_errors
    assert SequenceKey("project", Modality.REDNIR, "coin") in result.files


def test_rejects_duplicate_or_non_contiguous_frames(tmp_path: Path) -> None:
    duplicate = _write(
        tmp_path / "duplicate.csv",
        "ID,x,y,width,height\nvis-ball_1,1,2,3,4\nvis-ball_1,1,2,3,4\n",
    )
    with pytest.raises(CsvBundleError, match="重复定义"):
        load_csv_bundle(duplicate, dataset="project", cache_root=tmp_path / "cache-a")

    gap = _write(
        tmp_path / "gap.csv",
        "ID,x,y,width,height\nvis-ball_1,1,2,3,4\nvis-ball_3,1,2,3,4\n",
    )
    with pytest.raises(CsvBundleError, match="帧号不连续"):
        load_csv_bundle(gap, dataset="project", cache_root=tmp_path / "cache-b")


def test_discovers_single_result_csv_as_one_tracker(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "method-a.csv",
        "ID,x,y,width,height\nvis-ball_1,1,2,3,4\n",
    )
    key = SequenceKey("project", Modality.VIS, "ball")
    ground_truth = {key: tmp_path / "ground-truth.txt"}

    discoveries, report = discover_trackers(source, ground_truth, dataset="project")

    assert not report.has_errors
    assert len(discoveries) == 1
    assert discoveries[0].tracker_name == "method-a"
    assert discoveries[0].preferred is not None
    assert discoveries[0].preferred.layout is ResultLayout.PREFIXED_FLAT
    assert set(discoveries[0].preferred.files) == {key}


def test_bundled_ground_truth_manifests_are_consistent() -> None:
    from hsi_plotter.builtin_datasets import built_in_datasets

    datasets = {item.dataset_id: item for item in built_in_datasets()}
    assert set(datasets) == {"HOTC2020", "HOTC2023", "HOTC2024"}
    assert (datasets["HOTC2020"].sequence_count, datasets["HOTC2020"].frame_count) == (
        35,
        16574,
    )
    assert (datasets["HOTC2023"].sequence_count, datasets["HOTC2023"].frame_count) == (
        87,
        28430,
    )
    assert (datasets["HOTC2024"].sequence_count, datasets["HOTC2024"].frame_count) == (
        117,
        51730,
    )


def test_frozen_ground_truth_uses_pyinstaller_bundle_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from hsi_plotter.builtin_datasets import groundtruth_root

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert groundtruth_root() == tmp_path / "Groundtruth"


def test_frozen_ground_truth_prefers_visible_external_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from hsi_plotter.builtin_datasets import groundtruth_root

    executable = tmp_path / "app" / "HSITrackingPlotStudio.exe"
    visible = executable.parent / "Groundtruth"
    visible.mkdir(parents=True)
    (visible / "datasets.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "_internal"), raising=False)

    assert groundtruth_root() == visible
