from pathlib import Path

from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.visualization import ImageSequenceIndex
from hsi_plotter.visualization.session import (
    OverlayStatus,
    TrackerSequenceSource,
    load_sequence_session,
)


def _write_boxes(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _tracker(tmp_path: Path, identifier: str, color: str) -> Tracker:
    return Tracker(
        tracker_id=identifier,
        display_name=identifier.upper(),
        results_path=tmp_path / identifier,
        style=TrackerStyle(color=color, line_style="--", line_width=1.5),
    )


def test_session_composes_current_frame_with_stable_tracker_styles(tmp_path: Path) -> None:
    key = SequenceKey("HOT", Modality.VIS, "ball")
    images = ImageSequenceIndex(
        key=key,
        directory=tmp_path,
        frames={1: tmp_path / "0001.jpg", 3: tmp_path / "0003.jpg"},
    )
    gt = _write_boxes(
        tmp_path / "gt.txt",
        ["10,10,20,20", "-1,-1,-1,-1", "30,30,10,10"],
    )
    first = _tracker(tmp_path, "first", "#112233")
    second = _tracker(tmp_path, "second", "#445566")
    first_result = _write_boxes(
        tmp_path / "first.txt",
        ["11,12,20,20", "-1,-1,-1,-1", "nan,nan,nan,nan"],
    )
    second_result = _write_boxes(
        tmp_path / "second.txt",
        ["12,13,20,20", "15,15,5,5"],
    )

    session = load_sequence_session(
        images,
        gt,
        (
            TrackerSequenceSource(first, first_result),
            TrackerSequenceSource(second, second_result),
        ),
    )
    frame = session.compose(1)

    assert frame.image_path == (tmp_path / "0001.jpg").resolve()
    assert frame.ground_truth.status is OverlayStatus.VISIBLE
    assert frame.ground_truth.box == (10.0, 10.0, 20.0, 20.0)
    assert frame.ground_truth.style.color == "#0000FF"
    assert [overlay.tracker_id for overlay in frame.trackers] == ["first", "second"]
    assert frame.trackers[0].box == (11.0, 12.0, 20.0, 20.0)
    assert frame.trackers[0].style == first.style


def test_session_marks_offscreen_missing_and_out_of_range_without_stale_boxes(
    tmp_path: Path,
) -> None:
    key = SequenceKey("HOT", Modality.VIS, "ball")
    images = ImageSequenceIndex(
        key=key,
        directory=tmp_path,
        frames={1: tmp_path / "0001.jpg", 3: tmp_path / "0003.jpg"},
    )
    gt = _write_boxes(
        tmp_path / "gt.txt",
        ["10,10,20,20", "-1,-1,-1,-1", "30,30,10,10"],
    )
    first = _tracker(tmp_path, "first", "#112233")
    second = _tracker(tmp_path, "second", "#445566")
    first_result = _write_boxes(
        tmp_path / "first.txt",
        ["11,12,20,20", "-1,-1,-1,-1", "nan,nan,nan,nan"],
    )
    second_result = _write_boxes(
        tmp_path / "second.txt",
        ["12,13,20,20", "15,15,5,5"],
    )
    session = load_sequence_session(
        images,
        gt,
        (
            TrackerSequenceSource(first, first_result),
            TrackerSequenceSource(second, second_result),
        ),
    )

    frame_two = session.compose(2)
    frame_three = session.compose(3)

    assert frame_two.image_path is None
    assert frame_two.ground_truth.status is OverlayStatus.OFFSCREEN
    assert frame_two.ground_truth.box is None
    assert frame_two.trackers[0].status is OverlayStatus.OFFSCREEN
    assert frame_two.trackers[0].box is None
    assert frame_three.trackers[0].status is OverlayStatus.MISSING
    assert frame_three.trackers[0].box is None
    assert frame_three.trackers[1].status is OverlayStatus.OUT_OF_RANGE
    assert frame_three.trackers[1].box is None


def test_session_can_limit_composition_to_visible_tracker_ids(tmp_path: Path) -> None:
    key = SequenceKey("HOT", Modality.NIR, "car")
    images = ImageSequenceIndex(key=key, directory=tmp_path, frames={})
    gt = _write_boxes(tmp_path / "gt.txt", ["1,2,3,4"])
    first = _tracker(tmp_path, "first", "#112233")
    second = _tracker(tmp_path, "second", "#445566")
    result = _write_boxes(tmp_path / "result.txt", ["1,2,3,4"])
    session = load_sequence_session(
        images,
        gt,
        (TrackerSequenceSource(first, result), TrackerSequenceSource(second, result)),
    )

    frame = session.compose(1, visible_tracker_ids=frozenset({"second"}))

    assert [overlay.tracker_id for overlay in frame.trackers] == ["second"]
