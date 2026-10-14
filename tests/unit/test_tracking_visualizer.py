from pathlib import Path

from matplotlib.colors import to_hex
from PIL import Image

from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.plotting.tracking_visualizer import (
    TrackingRenderOptions,
    build_tracking_figure,
)
from hsi_plotter.visualization.session import (
    BoxOverlay,
    FrameComposition,
    OverlayStatus,
)


def _composition(tmp_path: Path) -> FrameComposition:
    image_path = tmp_path / "0002.png"
    Image.new("RGB", (100, 50), (240, 240, 240)).save(image_path)
    return FrameComposition(
        key=SequenceKey("HOT", Modality.VIS, "ball"),
        frame_number=2,
        image_path=image_path,
        ground_truth=BoxOverlay(
            tracker_id="ground_truth",
            display_name="Groundtruth",
            style=TrackerStyle("#0000FF", "-", 2.0),
            status=OverlayStatus.VISIBLE,
            box=(10.0, 5.0, 20.0, 10.0),
        ),
        trackers=(
            BoxOverlay(
                tracker_id="tracker",
                display_name="Tracker",
                style=TrackerStyle("#123456", "--", 1.5),
                status=OverlayStatus.VISIBLE,
                box=(12.0, 7.0, 18.0, 9.0),
            ),
            BoxOverlay(
                tracker_id="missing",
                display_name="Missing",
                style=TrackerStyle("#FF0000"),
                status=OverlayStatus.MISSING,
            ),
        ),
    )


def test_builds_original_coordinate_rectangles_and_draws_ground_truth_last(
    tmp_path: Path,
) -> None:
    figure, axis = build_tracking_figure(_composition(tmp_path))

    assert len(axis.patches) == 2
    tracker, ground_truth = axis.patches
    assert (tracker.get_x(), tracker.get_y(), tracker.get_width(), tracker.get_height()) == (
        12.0,
        7.0,
        18.0,
        9.0,
    )
    assert to_hex(tracker.get_edgecolor()) == "#123456"
    assert tracker.get_linestyle() == "--"
    assert tracker.get_linewidth() == 1.5
    assert to_hex(ground_truth.get_edgecolor()) == "#0000ff"
    assert axis.get_xlim() == (0.0, 100.0)
    assert axis.get_ylim() == (50.0, 0.0)
    figure.clear()


def test_view_zoom_does_not_modify_rectangle_coordinates(tmp_path: Path) -> None:
    figure, axis = build_tracking_figure(_composition(tmp_path))
    rectangle = axis.patches[0]
    original = rectangle.get_bbox().bounds

    axis.set_xlim(20.0, 60.0)
    axis.set_ylim(40.0, 10.0)

    assert rectangle.get_bbox().bounds == original
    figure.clear()


def test_frame_number_names_and_legend_can_be_disabled(tmp_path: Path) -> None:
    figure, axis = build_tracking_figure(
        _composition(tmp_path),
        TrackingRenderOptions(
            show_legend=False,
            show_names=False,
            show_frame_number=False,
        ),
    )

    assert axis.get_legend() is None
    assert len(axis.texts) == 0
    figure.clear()

