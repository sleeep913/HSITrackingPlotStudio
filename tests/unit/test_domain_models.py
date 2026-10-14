from pathlib import Path

from hsi_plotter.models.dataset import Modality, SequenceKey, parse_sequence_identity
from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker


def test_sequence_identity_preserves_modality_in_unique_key() -> None:
    modality, name = parse_sequence_identity("RedNIR-basketball")
    key = SequenceKey(dataset="HOT2023", modality=modality, name=name)

    assert modality is Modality.REDNIR
    assert name == "basketball"
    assert key.identifier == "HOT2023:rednir:basketball"
    assert SequenceKey("HOT2023", Modality.VIS, "basketball") != key


def test_sequence_identity_without_prefix_is_unspecified() -> None:
    modality, name = parse_sequence_identity("worker")

    assert modality is Modality.UNSPECIFIED
    assert name == "worker"


def test_project_dataclasses_hold_paths_and_stable_tracker_style(tmp_path: Path) -> None:
    style = TrackerStyle(color="#0072BD", line_style="--", line_width=2.0)
    tracker = Tracker(
        tracker_id="mht",
        display_name="MHT",
        results_path=tmp_path / "results" / "MHT",
        style=style,
    )
    project = Project(
        name="HOT paper",
        sources=ProjectSources(ground_truth=tmp_path / "gt"),
        trackers=[tracker],
    )

    assert project.trackers[0].style == style
    assert project.trackers[0].results_path == tmp_path / "results" / "MHT"
