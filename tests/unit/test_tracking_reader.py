from pathlib import Path

import numpy as np
import pytest

from hsi_plotter.io.tracking_reader import TrackingDataError, read_bounding_boxes


@pytest.mark.parametrize(
    "content",
    [
        "1,2,30,40\n5,6,31,41\n",
        "1 2 30 40\n5 6 31 41\n",
        "1\t2\t30\t40\n5\t6\t31\t41\n",
    ],
)
def test_read_bounding_boxes_autodetects_common_delimiters(tmp_path: Path, content: str) -> None:
    path = tmp_path / "basketball.txt"
    path.write_text(content, encoding="utf-8")

    boxes = read_bounding_boxes(path)

    np.testing.assert_allclose(boxes.values, [[1, 2, 30, 40], [5, 6, 31, 41]])
    assert boxes.frame_count == 2
    assert boxes.invalid_frame_indices == ()


def test_result_reader_accepts_complete_nan_row_as_missing_frame(tmp_path: Path) -> None:
    path = tmp_path / "result.txt"
    path.write_text("1,2,30,40\nnan,nan,nan,nan\n", encoding="utf-8")

    boxes = read_bounding_boxes(path, allow_missing_rows=True)

    assert boxes.invalid_frame_indices == (2,)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("1,2,30\n", "4"),
        ("1,2,nan,40\n", "部分缺失"),
        ("1,2,inf,40\n", "无穷"),
        ("1,2,nope,40\n", "数值"),
    ],
)
def test_reader_rejects_invalid_rows(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(TrackingDataError, match=message):
        read_bounding_boxes(path, allow_missing_rows=True)


@pytest.mark.parametrize("row", ["-1,-1,-1,-1\n", "1,2,0,40\n", "1,2,-3,40\n"])
def test_reader_accepts_matlab_legacy_finite_boxes(tmp_path: Path, row: str) -> None:
    path = tmp_path / "legacy.txt"
    path.write_text(row, encoding="utf-8")

    boxes = read_bounding_boxes(path, allow_missing_rows=True)

    assert boxes.frame_count == 1
    assert boxes.invalid_frame_indices == ()


def test_ground_truth_rejects_complete_nan_row(tmp_path: Path) -> None:
    path = tmp_path / "gt.txt"
    path.write_text("nan,nan,nan,nan\n", encoding="utf-8")

    with pytest.raises(TrackingDataError, match="Ground Truth"):
        read_bounding_boxes(path, allow_missing_rows=False)
