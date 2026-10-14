from pathlib import Path

import cv2
import pytest
from PIL import Image

from hsi_plotter.io.video_exporter import (
    VideoExportCancelled,
    export_tracking_video,
)
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.visualization import ImageSequenceIndex
from hsi_plotter.visualization.session import TrackerSequenceSource, load_sequence_session


def _session(tmp_path: Path):
    key = SequenceKey("HOT", Modality.VIS, "ball")
    frames = {}
    for frame in range(1, 5):
        path = tmp_path / f"{frame:04d}.png"
        Image.new("RGB", (80, 40), (220, 220, 220)).save(path)
        frames[frame] = path
    gt = tmp_path / "gt.txt"
    gt.write_text("5,5,20,10\n" * 4, encoding="utf-8")
    result = tmp_path / "result.txt"
    result.write_text("6,6,20,10\n" * 4, encoding="utf-8")
    tracker = Tracker(
        tracker_id="method",
        display_name="Method",
        results_path=tmp_path,
        style=TrackerStyle("#FF0000"),
    )
    return load_sequence_session(
        ImageSequenceIndex(key=key, directory=tmp_path, frames=frames),
        gt,
        (TrackerSequenceSource(tracker, result),),
    )


def test_exports_and_validates_mp4_range_fps_and_resolution(tmp_path: Path) -> None:
    output = export_tracking_video(
        _session(tmp_path),
        tmp_path / "tracking.mp4",
        start_frame=2,
        end_frame=4,
        fps=5.0,
        resolution=(320, 160),
    )

    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(5.0, abs=0.2)
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 160
    capture.release()


def test_cancelled_video_removes_temporary_and_final_files(tmp_path: Path) -> None:
    state = {"completed": 0}

    def progress(completed: int, total: int, message: str) -> None:
        state["completed"] = completed

    with pytest.raises(VideoExportCancelled):
        export_tracking_video(
            _session(tmp_path),
            tmp_path / "cancelled.mp4",
            start_frame=1,
            end_frame=4,
            fps=5.0,
            resolution=(160, 80),
            on_progress=progress,
            is_cancelled=lambda: state["completed"] >= 1,
        )

    assert not (tmp_path / "cancelled.mp4").exists()
    assert not list(tmp_path.glob("*.tmp.mp4"))


@pytest.mark.parametrize(
    ("start", "end", "fps", "resolution", "message"),
    [
        (0, 2, 5.0, (160, 80), "帧范围"),
        (3, 2, 5.0, (160, 80), "帧范围"),
        (1, 2, 0.0, (160, 80), "帧率"),
        (1, 2, 5.0, (0, 80), "分辨率"),
    ],
)
def test_video_export_validates_parameters(
    tmp_path: Path,
    start: int,
    end: int,
    fps: float,
    resolution: tuple[int, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        export_tracking_video(
            _session(tmp_path),
            tmp_path / "invalid.mp4",
            start_frame=start,
            end_frame=end,
            fps=fps,
            resolution=resolution,
        )
