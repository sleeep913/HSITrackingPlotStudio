from pathlib import Path

import pytest
from PIL import Image

from hsi_plotter.io.visualization_exporter import (
    VisualizationExportError,
    export_tracking_frame,
    export_tracking_range,
)
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.visualization.session import (
    BoxOverlay,
    FrameComposition,
    OverlayStatus,
)


def _composition(tmp_path: Path) -> FrameComposition:
    image_path = tmp_path / "0002.png"
    Image.new("RGB", (80, 40), (255, 255, 255)).save(image_path)
    return FrameComposition(
        key=SequenceKey("HOT", Modality.VIS, "ball"),
        frame_number=2,
        image_path=image_path,
        ground_truth=BoxOverlay(
            tracker_id="ground_truth",
            display_name="Groundtruth",
            style=TrackerStyle("#0000FF"),
            status=OverlayStatus.VISIBLE,
            box=(1.0, 2.0, 10.0, 8.0),
        ),
        trackers=(),
    )


@pytest.mark.parametrize("suffix", ["png", "pdf", "svg", "eps"])
def test_exports_supported_frame_formats_atomically(tmp_path: Path, suffix: str) -> None:
    target = tmp_path / f"frame.{suffix}"

    output = export_tracking_frame(_composition(tmp_path), target)

    assert output == target
    assert output.is_file()
    assert output.stat().st_size > 0
    assert not list(tmp_path.glob("*.tmp"))


def test_export_uses_next_available_name(tmp_path: Path) -> None:
    target = tmp_path / "frame.png"
    target.write_bytes(b"existing")

    output = export_tracking_frame(_composition(tmp_path), target)

    assert output == tmp_path / "frame-1.png"
    assert target.read_bytes() == b"existing"


def test_export_rejects_missing_image_and_unknown_format(tmp_path: Path) -> None:
    composition = _composition(tmp_path)
    missing = FrameComposition(
        key=composition.key,
        frame_number=composition.frame_number,
        image_path=None,
        ground_truth=composition.ground_truth,
        trackers=(),
    )

    with pytest.raises(VisualizationExportError, match="缺少图像"):
        export_tracking_frame(missing, tmp_path / "missing.png")
    with pytest.raises(VisualizationExportError, match="不支持"):
        export_tracking_frame(composition, tmp_path / "frame.xyz")


def test_range_export_keeps_completed_frames_and_reports_missing(tmp_path: Path) -> None:
    first = _composition(tmp_path)
    compositions = {
        1: FrameComposition(
            key=first.key,
            frame_number=1,
            image_path=first.image_path,
            ground_truth=first.ground_truth,
            trackers=(),
        ),
        2: FrameComposition(
            key=first.key,
            frame_number=2,
            image_path=None,
            ground_truth=first.ground_truth,
            trackers=(),
        ),
        3: FrameComposition(
            key=first.key,
            frame_number=3,
            image_path=first.image_path,
            ground_truth=first.ground_truth,
            trackers=(),
        ),
    }

    result = export_tracking_range(
        lambda frame: compositions[frame],
        tmp_path / "range",
        start_frame=1,
        end_frame=3,
        format_name="png",
    )

    assert len(result.outputs) == 2
    assert result.missing_frames == (2,)
    assert not result.cancelled


def test_range_export_can_cancel_after_completed_frame(tmp_path: Path) -> None:
    composition = _composition(tmp_path)
    state = {"completed": 0}

    result = export_tracking_range(
        lambda frame: FrameComposition(
            key=composition.key,
            frame_number=frame,
            image_path=composition.image_path,
            ground_truth=composition.ground_truth,
            trackers=(),
        ),
        tmp_path / "cancelled-range",
        start_frame=1,
        end_frame=3,
        format_name="png",
        on_progress=lambda completed, total, message: state.update(completed=completed),
        is_cancelled=lambda: state["completed"] >= 1,
    )

    assert len(result.outputs) == 1
    assert result.cancelled
