from pathlib import Path

from hsi_plotter.cache.visualization_codec import (
    decode_visualization_data,
    encode_visualization_data,
)
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.project import Project
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.visualization import ImageSequenceIndex
from hsi_plotter.visualization.project_data import ProjectVisualizationData


def test_visualization_cache_codec_round_trip(tmp_path: Path) -> None:
    key = SequenceKey("HOT2023", Modality.VIS, "ball")
    tracker = Tracker(
        tracker_id="lrsst",
        display_name="LRSST",
        results_path=tmp_path / "results" / "LRSST",
        style=TrackerStyle(color="#FF0000"),
    )
    project = Project(name="HOT2023", trackers=[tracker])
    data = ProjectVisualizationData(
        sequences={
            key: ImageSequenceIndex(
                key,
                tmp_path / "images" / "VIS" / "ball",
                {1: tmp_path / "images" / "VIS" / "ball" / "0001.jpg"},
            )
        },
        ground_truth_files={key: tmp_path / "groundtruth" / "VIS_ball.txt"},
        tracker_result_files={
            "lrsst": {key: tmp_path / "results" / "LRSST" / "VIS_ball.txt"}
        },
        trackers=(tracker,),
        issues=(),
    )

    restored = decode_visualization_data(encode_visualization_data(data), project)

    assert restored.sequences[key].frame_numbers == (1,)
    assert restored.ground_truth_files == data.ground_truth_files
    assert restored.tracker_result_files == data.tracker_result_files
    assert restored.trackers == (tracker,)
