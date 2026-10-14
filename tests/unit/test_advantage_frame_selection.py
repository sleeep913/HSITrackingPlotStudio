from pathlib import Path

import pytest

from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.visualization import ImageSequenceIndex
from hsi_plotter.visualization.frame_selection import (
    FrameSelectionError,
    select_advantage_frames,
)
from hsi_plotter.visualization.session import TrackerSequenceSource, load_sequence_session


def _write(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _tracker(tmp_path: Path, identifier: str) -> Tracker:
    return Tracker(
        tracker_id=identifier,
        display_name=identifier,
        results_path=tmp_path / identifier,
        style=TrackerStyle(color="#123456"),
    )


def _session(
    tmp_path: Path,
    *,
    primary_rows: list[str],
    competitor_rows: list[str],
):
    frame_count = len(primary_rows)
    key = SequenceKey("HOT", Modality.VIS, "ball")
    images = ImageSequenceIndex(
        key=key,
        directory=tmp_path,
        frames={number: tmp_path / f"{number:04d}.jpg" for number in range(1, frame_count + 1)},
    )
    gt = _write(tmp_path / "gt.txt", ["0,0,10,10"] * frame_count)
    primary = _tracker(tmp_path, "primary")
    competitor = _tracker(tmp_path, "competitor")
    return load_sequence_session(
        images,
        gt,
        (
            TrackerSequenceSource(primary, _write(tmp_path / "primary.txt", primary_rows)),
            TrackerSequenceSource(
                competitor,
                _write(tmp_path / "competitor.txt", competitor_rows),
            ),
        ),
    )


def test_selects_five_deterministic_advantage_frames_and_excludes_first(
    tmp_path: Path,
) -> None:
    primary = ["0,0,10,10"] * 12
    competitor = ["40,40,10,10"] * 12
    competitor[2] = "1,0,10,10"
    competitor[3] = "2,0,10,10"
    session = _session(
        tmp_path,
        primary_rows=primary,
        competitor_rows=competitor,
    )

    result = select_advantage_frames(
        session,
        primary_tracker_id="primary",
        competitor_ids=("competitor",),
    )

    assert len(result.candidates) == 5
    assert result.candidates[0].frame_number == 2
    assert all(item.frame_number != 1 for item in result.candidates)
    assert all(item.primary_iou >= 0.5 for item in result.candidates)
    assert all(item.advantage > 0 for item in result.candidates)
    assert result.candidates == select_advantage_frames(
        session,
        primary_tracker_id="primary",
        competitor_ids=("competitor",),
    ).candidates


def test_treats_competitor_nan_and_offscreen_as_zero_iou(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        primary_rows=["0,0,10,10"] * 4,
        competitor_rows=[
            "0,0,10,10",
            "nan,nan,nan,nan",
            "-1,-1,-1,-1",
            "0,0,10,10",
        ],
    )

    result = select_advantage_frames(
        session,
        primary_tracker_id="primary",
        competitor_ids=("competitor",),
    )

    assert [item.frame_number for item in result.candidates] == [2, 3]
    assert all(item.strongest_competitor_iou == 0.0 for item in result.candidates)
    assert "2" in result.message


def test_rejects_low_primary_iou_and_frames_where_competitor_is_better(
    tmp_path: Path,
) -> None:
    session = _session(
        tmp_path,
        primary_rows=[
            "0,0,10,10",
            "5,0,10,10",
            "2,0,10,10",
            "0,0,10,10",
        ],
        competitor_rows=[
            "0,0,10,10",
            "40,40,10,10",
            "1,0,10,10",
            "0,0,10,10",
        ],
    )

    result = select_advantage_frames(
        session,
        primary_tracker_id="primary",
        competitor_ids=("competitor",),
    )

    assert result.candidates == ()


def test_rejects_mismatched_result_length_as_structure_error(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        primary_rows=["0,0,10,10"] * 4,
        competitor_rows=["0,0,10,10"] * 4,
    )
    shortened = _write(tmp_path / "short.txt", ["0,0,10,10"] * 3)
    session = load_sequence_session(
        session.images,
        session.ground_truth.source_path,
        (
            TrackerSequenceSource(_tracker(tmp_path, "primary"), tmp_path / "primary.txt"),
            TrackerSequenceSource(_tracker(tmp_path, "competitor"), shortened),
        ),
    )

    with pytest.raises(FrameSelectionError, match="帧数"):
        select_advantage_frames(
            session,
            primary_tracker_id="primary",
            competitor_ids=("competitor",),
        )


def test_requires_primary_and_at_least_one_distinct_competitor(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        primary_rows=["0,0,10,10"] * 2,
        competitor_rows=["0,0,10,10"] * 2,
    )

    with pytest.raises(FrameSelectionError, match="对比方法"):
        select_advantage_frames(
            session,
            primary_tracker_id="primary",
            competitor_ids=(),
        )
