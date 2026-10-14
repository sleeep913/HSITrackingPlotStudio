from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from hsi_plotter.visualization.session import BoxOverlay, FrameComposition


@dataclass(frozen=True, slots=True)
class TrackingRenderOptions:
    show_ground_truth: bool = True
    show_legend: bool = True
    show_names: bool = False
    show_frame_number: bool = True
    width_inches: float = 7.2
    dpi: int = 150
    frame_text_color: str = "#FFF200"


def _draw_overlay(axis: Any, overlay: BoxOverlay, *, show_name: bool) -> Rectangle | None:
    if not overlay.is_visible or overlay.box is None:
        return None
    x, y, width, height = overlay.box
    rectangle = Rectangle(
        (x, y),
        width,
        height,
        fill=False,
        edgecolor=overlay.style.color,
        linestyle=overlay.style.line_style,
        linewidth=overlay.style.line_width,
        joinstyle="miter",
        clip_on=True,
    )
    axis.add_patch(rectangle)
    if show_name:
        axis.text(
            x,
            max(0.0, y - 2.0),
            overlay.display_name,
            color=overlay.style.color,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    return rectangle


def build_tracking_figure(
    composition: FrameComposition,
    options: TrackingRenderOptions | None = None,
) -> tuple[Figure, Any]:
    if composition.image_path is None:
        raise ValueError(
            f"序列 {composition.key.name} 第 {composition.frame_number} 帧缺少图像"
        )
    settings = options or TrackingRenderOptions()
    with Image.open(composition.image_path) as source:
        image = np.asarray(source.convert("RGB"))
    image_height, image_width = image.shape[:2]
    figure_height = settings.width_inches * image_height / image_width
    figure = Figure(
        figsize=(settings.width_inches, figure_height),
        dpi=settings.dpi,
        facecolor="#FFFFFF",
    )
    FigureCanvasAgg(figure)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(
        image,
        extent=(0.0, float(image_width), float(image_height), 0.0),
        interpolation="bilinear",
        aspect="equal",
    )
    axis.set_xlim(0.0, float(image_width))
    axis.set_ylim(float(image_height), 0.0)
    axis.set_axis_off()

    legend_overlays: list[BoxOverlay] = []
    for overlay in composition.trackers:
        if _draw_overlay(axis, overlay, show_name=settings.show_names) is not None:
            legend_overlays.append(overlay)
    if settings.show_ground_truth and _draw_overlay(
        axis,
        composition.ground_truth,
        show_name=settings.show_names,
    ) is not None:
        legend_overlays.append(composition.ground_truth)

    if settings.show_frame_number:
        axis.text(
            4.0,
            4.0,
            f"#{composition.frame_number:04d}",
            color=settings.frame_text_color,
            fontsize=10,
            fontweight="bold",
            fontfamily="serif",
            ha="left",
            va="top",
        )
    if settings.show_legend and legend_overlays:
        handles = [
            Line2D(
                [0],
                [0],
                color=overlay.style.color,
                linestyle=overlay.style.line_style,
                linewidth=max(2.0, overlay.style.line_width),
                label=overlay.display_name,
            )
            for overlay in legend_overlays
        ]
        axis.legend(
            handles=handles,
            loc="lower right",
            frameon=True,
            facecolor="#FFFFFF",
            framealpha=0.82,
            edgecolor="#D1D5DB",
            fontsize=8,
            ncol=1,
        )
    return figure, axis

