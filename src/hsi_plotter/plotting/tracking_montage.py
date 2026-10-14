from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from hsi_plotter.models.dataset import SequenceKey
from hsi_plotter.models.visualization import VisualSelectionList
from hsi_plotter.visualization.session import BoxOverlay, SequenceVisualizationSession


@dataclass(frozen=True, slots=True)
class MontageRenderOptions:
    cell_width: int = 600
    cell_height: int = 300
    horizontal_gap: int = 5
    vertical_gap: int = 5
    legend_height: int = 72
    dpi: int = 300
    frame_text_color: str = "#FFF200"
    frame_text_size: float = 7.5
    legend_font_size: float = 8.5
    legend_min_font_size: float = 6.0
    legend_line_width: float = 3.0
    box_line_width: float = 0.6
    first_frame_ground_truth_only: bool = True


def _add_fitted_legend(
    figure: Figure,
    axis: Any,
    handles: list[Line2D],
    options: MontageRenderOptions,
) -> None:
    font_size = max(options.legend_font_size, options.legend_min_font_size)
    available_width = figure.bbox.width * 0.97
    available_height = options.legend_height * 0.92
    while True:
        legend = axis.legend(
            handles=handles,
            loc="center",
            ncol=len(handles),
            frameon=False,
            handlelength=2.0,
            handletextpad=0.4,
            columnspacing=0.55,
            borderaxespad=0.0,
            prop={
                "family": "serif",
                "weight": "bold",
                "size": font_size,
            },
        )
        figure.canvas.draw()
        bounds = legend.get_window_extent(figure.canvas.get_renderer())
        if (
            bounds.width <= available_width
            and bounds.height <= available_height
        ) or font_size <= options.legend_min_font_size:
            return
        scale = min(
            available_width / max(bounds.width, 1.0),
            available_height / max(bounds.height, 1.0),
        )
        next_size = max(
            options.legend_min_font_size,
            min(font_size - 0.25, font_size * scale * 0.98),
        )
        legend.remove()
        font_size = next_size


def _draw_scaled_overlay(
    axis: Any,
    overlay: BoxOverlay,
    *,
    source_width: int,
    source_height: int,
    options: MontageRenderOptions,
) -> None:
    if not overlay.is_visible or overlay.box is None:
        return
    x, y, width, height = overlay.box
    scale_x = options.cell_width / source_width
    scale_y = options.cell_height / source_height
    axis.add_patch(
        Rectangle(
            (x * scale_x, y * scale_y),
            width * scale_x,
            height * scale_y,
            fill=False,
            edgecolor=overlay.style.color,
            linestyle=overlay.style.line_style,
            linewidth=options.box_line_width,
            joinstyle="miter",
            clip_on=True,
        )
    )


def build_tracking_montage(
    sessions: Mapping[SequenceKey, SequenceVisualizationSession],
    selections: VisualSelectionList,
    options: MontageRenderOptions | None = None,
    *,
    visible_tracker_ids: frozenset[str] | None = None,
) -> Figure:
    if not selections.sequences:
        raise ValueError("Visual 绘制清单为空")
    incomplete = [
        selection.key.name
        for selection in selections.sequences
        if len(selection.frame_numbers) != 3
    ]
    if incomplete:
        raise ValueError(f"每个序列必须选择 3 帧：{', '.join(incomplete)}")
    settings = options or MontageRenderOptions()
    rows = len(selections.sequences)
    columns = 4
    legend_height = settings.legend_height
    figure_width = (
        columns * settings.cell_width
        + max(0, columns - 1) * settings.horizontal_gap
    )
    figure_height = (
        rows * settings.cell_height
        + max(0, rows - 1) * settings.vertical_gap
        + legend_height
        + settings.vertical_gap
    )
    figure = Figure(
        figsize=(figure_width / settings.dpi, figure_height / settings.dpi),
        dpi=settings.dpi,
        facecolor="#FFFFFF",
    )
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        rows + 1,
        columns,
        height_ratios=[settings.cell_height] * rows + [legend_height],
        wspace=settings.horizontal_gap / settings.cell_width,
        hspace=settings.vertical_gap / settings.cell_height,
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    legend_entries: dict[str, BoxOverlay] = {}
    ground_truth_entry: BoxOverlay | None = None
    for row, selection in enumerate(selections.sequences):
        try:
            session = sessions[selection.key]
        except KeyError as exc:
            raise ValueError(f"缺少序列会话：{selection.key.identifier}") from exc
        frame_numbers = (1, *sorted(selection.frame_numbers))
        for column, frame_number in enumerate(frame_numbers):
            axis = figure.add_subplot(grid[row, column])
            axis.set_axis_off()
            composition = session.compose(
                frame_number,
                visible_tracker_ids=visible_tracker_ids,
            )
            if composition.image_path is None:
                raise ValueError(
                    f"序列 {selection.key.name} 第 {frame_number} 帧缺少图像"
                )
            with Image.open(composition.image_path) as source:
                image = np.asarray(source.convert("RGB"))
            source_height, source_width = image.shape[:2]
            axis.imshow(
                image,
                extent=(0, settings.cell_width, settings.cell_height, 0),
                interpolation="bilinear",
                aspect="auto",
            )
            axis.set_xlim(0, settings.cell_width)
            axis.set_ylim(settings.cell_height, 0)
            if not (settings.first_frame_ground_truth_only and column == 0):
                for overlay in composition.trackers:
                    _draw_scaled_overlay(
                        axis,
                        overlay,
                        source_width=source_width,
                        source_height=source_height,
                        options=settings,
                    )
                    legend_entries.setdefault(overlay.tracker_id, overlay)
            else:
                for overlay in composition.trackers:
                    legend_entries.setdefault(overlay.tracker_id, overlay)
            _draw_scaled_overlay(
                axis,
                composition.ground_truth,
                source_width=source_width,
                source_height=source_height,
                options=settings,
            )
            ground_truth_entry = composition.ground_truth
            axis.text(
                4,
                4,
                f"#{frame_number:04d}",
                color=settings.frame_text_color,
                fontsize=settings.frame_text_size,
                fontweight="bold",
                fontfamily="serif",
                ha="left",
                va="top",
            )

    legend_axis = figure.add_subplot(grid[rows, :])
    legend_axis.set_axis_off()
    entries = list(legend_entries.values())
    if ground_truth_entry is not None:
        entries.append(ground_truth_entry)
    handles = [
        Line2D(
            [0],
            [0],
            color=entry.style.color,
            linestyle=entry.style.line_style,
            linewidth=settings.legend_line_width,
            label=entry.display_name,
        )
        for entry in entries
    ]
    if handles:
        _add_fitted_legend(figure, legend_axis, handles, settings)
    return figure

