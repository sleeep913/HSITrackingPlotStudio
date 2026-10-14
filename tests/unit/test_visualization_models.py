from pathlib import Path

import pytest

from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.visualization import (
    ImageSequenceIndex,
    VisualSelectionError,
    VisualSelectionList,
)


def test_image_sequence_index_exposes_sorted_frame_numbers(tmp_path: Path) -> None:
    key = SequenceKey("HOT", Modality.VIS, "ball")
    index = ImageSequenceIndex(
        key=key,
        directory=tmp_path,
        frames={10: tmp_path / "0010.jpg", 1: tmp_path / "0001.jpg"},
    )

    assert index.frame_numbers == (1, 10)
    assert index.path_for_frame(10) == tmp_path / "0010.jpg"
    assert index.path_for_frame(2) is None


def test_visual_selection_accumulates_sequences_and_limits_each_to_three_frames() -> None:
    vis = SequenceKey("HOT", Modality.VIS, "ball")
    nir = SequenceKey("HOT", Modality.NIR, "car")

    selections = VisualSelectionList().add(vis, 12).add(vis, 4).add(vis, 9)
    assert selections.add(vis, 9) is selections
    selections = selections.add(nir, 2)

    assert [item.key for item in selections.sequences] == [vis, nir]
    assert selections.sequences[0].frame_numbers == (4, 9, 12)
    assert selections.sequences[1].frame_numbers == (2,)
    assert not selections.is_complete

    with pytest.raises(VisualSelectionError, match="最多选择 3 帧"):
        selections.add(vis, 20)

    completed = selections.add(nir, 6).add(nir, 8)
    assert completed.is_complete


def test_visual_selection_rejects_non_positive_frame_numbers() -> None:
    key = SequenceKey("HOT", Modality.VIS, "ball")

    with pytest.raises(VisualSelectionError, match="正整数"):
        VisualSelectionList().add(key, 0)


def test_visual_selection_can_remove_frames_and_sequences() -> None:
    vis = SequenceKey("HOT", Modality.VIS, "ball")
    nir = SequenceKey("HOT", Modality.NIR, "car")
    selections = VisualSelectionList().add(vis, 1).add(vis, 3).add(nir, 2)

    without_frame = selections.remove_frame(vis, 1)
    assert without_frame.sequences[0].frame_numbers == (3,)
    without_sequence = without_frame.remove_sequence(vis)
    assert [item.key for item in without_sequence.sequences] == [nir]
