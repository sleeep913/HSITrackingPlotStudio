from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hsi_plotter.io.result_writer import (
    _temporary_target,
    next_available_path,
    sanitize_file_stem,
)
from hsi_plotter.models.dataset import SequenceKey
from hsi_plotter.models.visualization import VisualSelectionList
from hsi_plotter.plotting.tracking_montage import (
    MontageRenderOptions,
    build_tracking_montage,
)
from hsi_plotter.plotting.tracking_visualizer import (
    TrackingRenderOptions,
    build_tracking_figure,
)
from hsi_plotter.visualization.session import FrameComposition, SequenceVisualizationSession

SUPPORTED_VISUALIZATION_FORMATS = frozenset({"png", "pdf", "svg", "eps"})
LOGGER = logging.getLogger(__name__)


class VisualizationExportError(OSError):
    """Raised when a tracking visualization cannot be written."""


@dataclass(frozen=True, slots=True)
class RangeExportResult:
    outputs: tuple[Path, ...]
    missing_frames: tuple[int, ...]
    cancelled: bool = False


def export_tracking_frame(
    composition: FrameComposition,
    destination: Path,
    *,
    options: TrackingRenderOptions | None = None,
) -> Path:
    target = Path(destination)
    suffix = target.suffix.removeprefix(".").casefold()
    if suffix not in SUPPORTED_VISUALIZATION_FORMATS:
        raise VisualizationExportError(
            f"不支持的图形格式 {target.suffix!r}，支持 PNG、PDF、SVG 和 EPS。"
        )
    if composition.image_path is None:
        raise VisualizationExportError(
            f"序列 {composition.key.name} 第 {composition.frame_number} 帧缺少图像，无法导出。"
        )
    actual_target = next_available_path(target)
    figure = None
    try:
        figure, _ = build_tracking_figure(composition, options)
        with _temporary_target(actual_target) as temporary:
            figure.savefig(
                temporary,
                format=suffix,
                dpi=(options.dpi if options is not None else 150),
                facecolor="#FFFFFF",
                bbox_inches=None,
            )
    except Exception as exc:
        LOGGER.exception("tracking frame export failed: target=%s", actual_target)
        raise VisualizationExportError(f"无法导出跟踪可视化 {actual_target}：{exc}") from exc
    finally:
        if figure is not None:
            figure.clear()
    return actual_target


def export_tracking_montage(
    sessions: Mapping[SequenceKey, SequenceVisualizationSession],
    selections: VisualSelectionList,
    destination: Path,
    *,
    options: MontageRenderOptions | None = None,
    visible_tracker_ids: frozenset[str] | None = None,
) -> Path:
    target = Path(destination)
    suffix = target.suffix.removeprefix(".").casefold()
    if suffix not in SUPPORTED_VISUALIZATION_FORMATS:
        raise VisualizationExportError(
            f"不支持的图形格式 {target.suffix!r}，支持 PNG、PDF、SVG 和 EPS。"
        )
    actual_target = next_available_path(target)
    figure = None
    try:
        figure = build_tracking_montage(
            sessions,
            selections,
            options,
            visible_tracker_ids=visible_tracker_ids,
        )
        with _temporary_target(actual_target) as temporary:
            figure.savefig(
                temporary,
                format=suffix,
                dpi=(options.dpi if options is not None else 300),
                facecolor="#FFFFFF",
                bbox_inches=None,
            )
    except Exception as exc:
        LOGGER.exception("tracking montage export failed: target=%s", actual_target)
        raise VisualizationExportError(f"无法导出 Visual {actual_target}：{exc}") from exc
    finally:
        if figure is not None:
            figure.clear()
    return actual_target


def export_tracking_range(
    composition_for_frame: Callable[[int], FrameComposition],
    directory: Path,
    *,
    start_frame: int,
    end_frame: int,
    format_name: str,
    options: TrackingRenderOptions | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> RangeExportResult:
    if start_frame < 1 or end_frame < start_frame:
        raise ValueError("导出帧范围无效")
    suffix = format_name.removeprefix(".").casefold()
    if suffix not in SUPPORTED_VISUALIZATION_FORMATS:
        raise VisualizationExportError(
            f"不支持的图形格式 {format_name!r}，支持 PNG、PDF、SVG 和 EPS。"
        )
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    total = end_frame - start_frame + 1
    outputs: list[Path] = []
    missing: list[int] = []
    for completed, frame_number in enumerate(range(start_frame, end_frame + 1), start=1):
        if is_cancelled is not None and is_cancelled():
            return RangeExportResult(
                outputs=tuple(outputs),
                missing_frames=tuple(missing),
                cancelled=True,
            )
        composition = composition_for_frame(frame_number)
        if composition.image_path is None:
            missing.append(frame_number)
        else:
            stem = sanitize_file_stem(
                f"{composition.key.modality.value}-{composition.key.name}-{frame_number:04d}"
            )
            outputs.append(
                export_tracking_frame(
                    composition,
                    root / f"{stem}.{suffix}",
                    options=options,
                )
            )
        if on_progress is not None:
            on_progress(completed, total, f"已处理第 {frame_number} 帧")
    return RangeExportResult(
        outputs=tuple(outputs),
        missing_frames=tuple(missing),
    )
