from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from hsi_plotter.io.result_writer import next_available_path
from hsi_plotter.plotting.tracking_visualizer import (
    TrackingRenderOptions,
    build_tracking_figure,
)
from hsi_plotter.visualization.session import SequenceVisualizationSession

ProgressCallback = Callable[[int, int, str], None]
CancelledCallback = Callable[[], bool]


class VideoExportError(OSError):
    """Raised when an MP4 cannot be encoded or validated."""


class VideoExportCancelled(VideoExportError):
    """Raised after a requested video cancellation has been cleaned up."""


def _validate_parameters(
    session: SequenceVisualizationSession,
    start_frame: int,
    end_frame: int,
    fps: float,
    resolution: tuple[int, int],
) -> None:
    if start_frame < 1 or end_frame < start_frame or end_frame > session.frame_count:
        raise ValueError(
            f"帧范围必须位于 1 到 {session.frame_count} 且起始帧不大于结束帧"
        )
    if fps <= 0:
        raise ValueError("帧率必须大于 0")
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError("分辨率宽度和高度必须大于 0")


def _validate_video(
    path: Path,
    *,
    expected_frames: int,
    fps: float,
    resolution: tuple[int, int],
) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoExportError(f"编码完成后无法重新打开视频：{path}")
        actual_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        actual_resolution = (
            int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
    finally:
        capture.release()
    if actual_frames != expected_frames:
        raise VideoExportError(
            f"视频帧数校验失败：期望 {expected_frames}，实际 {actual_frames}"
        )
    if abs(actual_fps - fps) > 0.25:
        raise VideoExportError(f"视频帧率校验失败：期望 {fps}，实际 {actual_fps}")
    if actual_resolution != resolution:
        raise VideoExportError(
            f"视频分辨率校验失败：期望 {resolution}，实际 {actual_resolution}"
        )


def export_tracking_video(
    session: SequenceVisualizationSession,
    destination: Path,
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
    resolution: tuple[int, int],
    visible_tracker_ids: frozenset[str] | None = None,
    show_legend: bool = True,
    on_progress: ProgressCallback | None = None,
    is_cancelled: CancelledCallback | None = None,
) -> Path:
    _validate_parameters(session, start_frame, end_frame, fps, resolution)
    requested = Path(destination)
    if requested.suffix.casefold() != ".mp4":
        requested = requested.with_suffix(".mp4")
    target = next_available_path(requested)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f"{target.stem}-",
        suffix=".tmp.mp4",
    )
    os.close(handle)
    temporary = Path(temporary_name)
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        resolution,
    )
    if not writer.isOpened():
        temporary.unlink(missing_ok=True)
        raise VideoExportError("无法初始化 MP4 编码器（mp4v）")

    total = end_frame - start_frame + 1
    options = TrackingRenderOptions(
        show_legend=show_legend,
        width_inches=max(1.0, resolution[0] / 100.0),
        dpi=100,
    )
    try:
        for completed, frame_number in enumerate(
            range(start_frame, end_frame + 1),
            start=1,
        ):
            if is_cancelled is not None and is_cancelled():
                raise VideoExportCancelled("视频导出已取消")
            composition = session.compose(
                frame_number,
                visible_tracker_ids=visible_tracker_ids,
            )
            if composition.image_path is None:
                raise VideoExportError(
                    f"序列 {session.key.name} 第 {frame_number} 帧图像缺失"
                )
            figure, _ = build_tracking_figure(composition, options)
            try:
                figure.canvas.draw()
                rgba = np.asarray(figure.canvas.buffer_rgba())
                rgb = np.ascontiguousarray(rgba[:, :, :3])
                resized = cv2.resize(rgb, resolution, interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            finally:
                figure.clear()
            if on_progress is not None:
                on_progress(completed, total, f"正在编码第 {frame_number} 帧")
        writer.release()
        _validate_video(
            temporary,
            expected_frames=total,
            fps=fps,
            resolution=resolution,
        )
        os.replace(temporary, target)
    except BaseException:
        writer.release()
        temporary.unlink(missing_ok=True)
        raise
    return target
