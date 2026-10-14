from pathlib import Path

import pytest
from PIL import Image

from hsi_plotter.io.visualization_exporter import export_tracking_montage
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.visualization import ImageSequenceIndex, VisualSelectionList
from hsi_plotter.plotting.tracking_montage import (
    MontageRenderOptions,
    build_tracking_montage,
)
from hsi_plotter.visualization.session import TrackerSequenceSource, load_sequence_session


def _session(
    tmp_path: Path,
    key: SequenceKey,
    stem: str,
    *,
    tracker_count: int = 1,
    display_name_prefix: str = "Method",
):
    directory = tmp_path / stem
    directory.mkdir()
    frames = {}
    for frame in (1, 2, 3, 4, 5):
        path = directory / f"{frame:04d}.png"
        Image.new("RGB", (100, 50), (240, 240, 240)).save(path)
        frames[frame] = path
    gt = directory / "gt.txt"
    gt.write_text("10,5,20,10\n" * 5, encoding="utf-8")
    palette = (
        "#FF0000",
        "#0072BD",
        "#77AC30",
        "#EDB120",
        "#7E2F8E",
        "#D95319",
        "#4DBEEE",
        "#A2142F",
        "#111111",
    )
    sources = []
    for index in range(tracker_count):
        result = directory / f"result-{index}.txt"
        result.write_text("12,7,18,9\n" * 5, encoding="utf-8")
        tracker = Tracker(
            tracker_id=f"method-{index}",
            display_name=f"{display_name_prefix}-{index + 1}",
            results_path=directory,
            style=TrackerStyle(
                palette[index],
                "--" if index % 2 else "-",
                1.25,
            ),
        )
        sources.append(TrackerSequenceSource(tracker, result))
    return load_sequence_session(
        ImageSequenceIndex(key=key, directory=directory, frames=frames),
        gt,
        tuple(sources),
    )


def test_builds_fixed_four_column_visual_with_sorted_selected_frames(
    tmp_path: Path,
) -> None:
    vis = SequenceKey("HOT", Modality.VIS, "ball")
    nir = SequenceKey("HOT", Modality.NIR, "car")
    sessions = {
        vis: _session(tmp_path, vis, "vis"),
        nir: _session(tmp_path, nir, "nir"),
    }
    selections = (
        VisualSelectionList()
        .add(vis, 5)
        .add(vis, 2)
        .add(vis, 3)
        .add(nir, 2)
        .add(nir, 5)
        .add(nir, 4)
    )

    figure = build_tracking_montage(sessions, selections)

    assert len(figure.axes) == 9
    assert figure.get_size_inches()[0] * figure.dpi == 2415
    assert figure.get_size_inches()[1] * figure.dpi == 682
    assert [axis.texts[0].get_text() for axis in figure.axes[:4]] == [
        "#0001",
        "#0002",
        "#0003",
        "#0005",
    ]
    assert len(figure.axes[0].patches) == 1  # fixed first column: GT only
    assert len(figure.axes[1].patches) == 2
    assert len(figure.axes[4].patches) == 1
    assert len(figure.axes[5].patches) == 2
    assert all(
        patch.get_linewidth() == pytest.approx(0.6)
        for axis in figure.axes[:-1]
        for patch in axis.patches
    )
    assert figure.axes[-1].get_legend() is not None
    figure.clear()


def test_montage_can_hide_frame_one_special_case(tmp_path: Path) -> None:
    key = SequenceKey("HOT", Modality.VIS, "ball")
    session = _session(tmp_path, key, "vis")
    selections = VisualSelectionList().add(key, 2).add(key, 3).add(key, 4)

    figure = build_tracking_montage(
        {key: session},
        selections,
        MontageRenderOptions(first_frame_ground_truth_only=False),
    )

    assert len(figure.axes[0].patches) == 2
    figure.clear()


def test_montage_requires_exactly_three_selected_frames(tmp_path: Path) -> None:
    key = SequenceKey("HOT", Modality.VIS, "ball")
    session = _session(tmp_path, key, "vis")
    selections = VisualSelectionList().add(key, 2).add(key, 3)

    with pytest.raises(ValueError, match="必须选择 3 帧"):
        build_tracking_montage({key: session}, selections)


@pytest.mark.parametrize(
    ("tracker_count", "expected_columns"),
    ((6, 7), (9, 10)),
)
def test_single_row_legend_for_six_to_nine_trackers_stays_inside_canvas(
    tmp_path: Path,
    tracker_count: int,
    expected_columns: int,
) -> None:
    key = SequenceKey("HOT", Modality.VIS, "ball")
    session = _session(tmp_path, key, "vis", tracker_count=tracker_count)
    selections = VisualSelectionList().add(key, 2).add(key, 3).add(key, 4)

    figure = build_tracking_montage({key: session}, selections)
    figure.canvas.draw()
    legend = figure.axes[-1].get_legend()
    legend_bounds = legend.get_window_extent(figure.canvas.get_renderer())

    assert legend._ncols == expected_columns
    assert figure.bbox.contains(legend_bounds.x0, legend_bounds.y0)
    assert figure.bbox.contains(legend_bounds.x1, legend_bounds.y1)
    figure.clear()


def test_single_row_legend_shrinks_to_fit_long_method_names(tmp_path: Path) -> None:
    key = SequenceKey("HOT", Modality.VIS, "ball")
    session = _session(
        tmp_path,
        key,
        "vis",
        tracker_count=9,
        display_name_prefix="Tracker",
    )
    selections = VisualSelectionList().add(key, 2).add(key, 3).add(key, 4)

    figure = build_tracking_montage(
        {key: session},
        selections,
        MontageRenderOptions(legend_font_size=24.0),
    )
    figure.canvas.draw()
    legend = figure.axes[-1].get_legend()
    legend_bounds = legend.get_window_extent(figure.canvas.get_renderer())

    assert legend._ncols == 10
    assert legend.get_texts()[0].get_fontsize() < 24.0
    assert figure.bbox.contains(legend_bounds.x0, legend_bounds.y0)
    assert figure.bbox.contains(legend_bounds.x1, legend_bounds.y1)
    figure.clear()


def test_exports_montage_using_same_builder(tmp_path: Path) -> None:
    key = SequenceKey("HOT", Modality.VIS, "ball")
    session = _session(tmp_path, key, "vis")
    selections = VisualSelectionList().add(key, 2).add(key, 3).add(key, 4)

    output = export_tracking_montage(
        {key: session},
        selections,
        tmp_path / "visual.pdf",
    )

    assert output.is_file()
    assert output.stat().st_size > 0
