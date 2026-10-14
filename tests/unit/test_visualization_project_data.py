from pathlib import Path

from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.tracker_import import ResultLayout
from hsi_plotter.visualization.project_data import prepare_visualization_project


def _write(path: Path, text: str = "1,2,3,4\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_prepares_project_sequences_and_reuses_tracker_mapping(tmp_path: Path) -> None:
    ground_truth = tmp_path / "ground"
    _write(ground_truth / "ALL" / "vis-ball.txt")
    images = tmp_path / "images"
    image = images / "HSI-VIS-FalseColor" / "ball" / "0001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    results = tmp_path / "results"
    result_file = _write(results / "vis-ball.txt")
    tracker = Tracker(
        tracker_id="method",
        display_name="Method",
        results_path=results,
        style=TrackerStyle("#123456"),
        layout=ResultLayout.PREFIXED_FLAT,
    )
    project = Project(
        "paper",
        sources=ProjectSources(ground_truth=ground_truth, image_root=images),
        trackers=[tracker],
    )

    data = prepare_visualization_project(project)

    key = SequenceKey("project", Modality.VIS, "ball")
    assert not data.has_errors
    assert data.sequences[key].frames[1] == image.resolve()
    assert data.ground_truth_files[key] == (ground_truth / "ALL" / "vis-ball.txt").resolve()
    assert data.tracker_result_files["method"][key] == result_file.resolve()


def test_maps_legacy_unspecified_image_sequence_by_unique_ground_truth_name(
    tmp_path: Path,
) -> None:
    ground_truth = tmp_path / "ground"
    _write(ground_truth / "VIS" / "ball.txt")
    images = tmp_path / "images"
    image = images / "ball" / "img" / "0001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    project = Project(
        "paper",
        sources=ProjectSources(ground_truth=ground_truth, image_root=images),
    )

    data = prepare_visualization_project(project)

    key = SequenceKey("project", Modality.VIS, "ball")
    assert key in data.sequences
    assert data.sequences[key].key == key


def test_reports_ambiguous_legacy_image_sequence_name(tmp_path: Path) -> None:
    ground_truth = tmp_path / "ground"
    _write(ground_truth / "VIS" / "ball.txt")
    _write(ground_truth / "NIR" / "ball.txt")
    images = tmp_path / "images"
    image = images / "ball" / "0001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    project = Project(
        "paper",
        sources=ProjectSources(ground_truth=ground_truth, image_root=images),
    )

    data = prepare_visualization_project(project)

    assert data.has_errors
    assert any(issue.code == "AMBIGUOUS_IMAGE_SEQUENCE" for issue in data.issues)
