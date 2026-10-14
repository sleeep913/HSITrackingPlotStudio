"""Combined AUC/DP attribute radar figure based on the HOT23 paper script."""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.projections.polar import PolarAxes

from hsi_plotter.models.style import TrackerStyle


@dataclass(frozen=True, slots=True)
class AttributeRadarSeries:
    tracker_id: str
    display_name: str
    auc_values: tuple[float | None, ...]
    dp_values: tuple[float | None, ...]
    style: TrackerStyle


@dataclass(frozen=True, slots=True)
class RadialScale:
    minimum: float
    maximum: float
    ticks: tuple[float, ...]


def plan_radial_scale(values: list[float]) -> RadialScale:
    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        raise ValueError("雷达图没有可计算的指标值")
    lower = max(0.0, math.floor((min(finite) - 0.04) * 10.0) / 10.0)
    upper = min(1.0, math.ceil((max(finite) + 0.04) * 10.0) / 10.0)
    if upper - lower < 0.3:
        center = (upper + lower) / 2.0
        lower = max(0.0, math.floor((center - 0.15) * 10.0) / 10.0)
        upper = min(1.0, math.ceil((center + 0.15) * 10.0) / 10.0)
    if upper <= lower:
        upper = min(1.0, lower + 0.3)
        lower = max(0.0, upper - 0.3)
    ticks = tuple(
        float(round(value, 10))
        for value in np.arange(lower, upper + 0.0001, 0.1, dtype=np.float64)
        if value < upper - 0.0001
    )
    return RadialScale(minimum=lower, maximum=upper, ticks=ticks)


def _label_placement(angle: float) -> tuple[float, float, str, str]:
    degrees = math.degrees(angle) % 360.0
    if degrees < 15.0 or degrees >= 345.0:
        return -4.0, 0.0, "left", "center"
    if degrees < 75.0:
        return -2.0, 3.0, "left", "bottom"
    if degrees < 105.0:
        return 0.0, 3.0, "center", "bottom"
    if degrees < 165.0:
        return 2.0, 3.0, "right", "bottom"
    if degrees < 195.0:
        return 4.0, 0.0, "right", "center"
    if degrees < 255.0:
        return 2.0, -3.0, "right", "top"
    if degrees < 285.0:
        return 0.0, -3.0, "center", "top"
    return -2.0, -3.0, "left", "top"


def _draw_radar_axis(
    axis: PolarAxes,
    attributes: tuple[str, ...],
    series: tuple[AttributeRadarSeries, ...],
    *,
    value_field: str,
    scale: RadialScale,
    chart_side: str,
) -> None:
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_ylim(scale.minimum, scale.maximum)
    axis.grid(True, color="#bcbcbc", linewidth=1.0, alpha=0.65)
    axis.spines["polar"].set_color("#666666")
    axis.spines["polar"].set_linewidth(1.15)
    axis.set_yticks(scale.ticks)
    axis.set_yticklabels(
        [f"{value:.1f}" for value in scale.ticks],
        fontsize=12,
        color="#999999",
        fontfamily="Times New Roman",
    )
    axis.set_rlabel_position(90)
    angles = np.linspace(0.0, 2.0 * np.pi, len(attributes), endpoint=False)
    closed_angles = np.concatenate((angles, angles[:1]))
    axis.set_xticks(angles)
    axis.set_xticklabels([])
    for item in series:
        raw_values = getattr(item, value_field)
        values = np.asarray(
            [np.nan if value is None else float(value) for value in raw_values],
            dtype=np.float64,
        )
        axis.plot(
            closed_angles,
            np.concatenate((values, values[:1])),
            color=item.style.color,
            linestyle=item.style.line_style,
            linewidth=2.8,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )
    for index, (attribute, angle) in enumerate(zip(attributes, angles, strict=True)):
        values = [
            float(getattr(item, value_field)[index])
            for item in series
            if getattr(item, value_field)[index] is not None
            and math.isfinite(float(getattr(item, value_field)[index]))
        ]
        range_text = "—" if not values else f"{min(values):.3f},{max(values):.3f}"
        dx, dy, horizontal, vertical = _label_placement(float(angle))
        if chart_side == "left" and index == 0:
            dx -= 4.0
            dy += 16.0
        elif chart_side == "right" and index == len(attributes) // 2:
            dx += 4.0
            dy -= 16.0
        axis.annotate(
            f"{attribute}\n[{range_text}]",
            xy=(angle, scale.maximum),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=horizontal,
            va=vertical,
            fontsize=18,
            fontfamily="Times New Roman",
            fontweight="bold",
            annotation_clip=False,
            zorder=10,
        )


def _reference_scale(values: list[float], *, minimum: float, maximum: float) -> RadialScale:
    if not values:
        raise ValueError("雷达图没有可计算的指标值")
    lower = max(0.0, min(minimum, math.floor(min(values) * 10.0) / 10.0))
    upper = min(1.0, max(maximum, math.ceil(max(values) * 10.0) / 10.0))
    ticks = tuple(
        float(round(value, 10))
        for value in np.arange(lower, upper + 0.0001, 0.1)
        if value < upper - 0.0001
    )
    return RadialScale(lower, upper, ticks)


def build_attribute_radar_figure(
    attributes: tuple[str, ...],
    series: tuple[AttributeRadarSeries, ...],
) -> tuple[Figure, tuple[PolarAxes, PolarAxes]]:
    """Build side-by-side AUC and DP radars with one shared legend."""
    if len(attributes) < 3:
        raise ValueError("雷达图至少需要 3 个可计算属性")
    if not series:
        raise ValueError("雷达图至少需要一个跟踪器")
    if any(
        len(item.auc_values) != len(attributes) or len(item.dp_values) != len(attributes)
        for item in series
    ):
        raise ValueError("跟踪器指标数量必须与属性数量一致")
    auc_values = [
        float(value)
        for item in series
        for value in item.auc_values
        if value is not None and math.isfinite(float(value))
    ]
    dp_values = [
        float(value)
        for item in series
        for value in item.dp_values
        if value is not None and math.isfinite(float(value))
    ]
    figure = plt.figure(figsize=(20.0, 10.5), facecolor="white")
    left = figure.add_axes([0.055, 0.215, 0.420, 0.720], projection="polar")
    right = figure.add_axes([0.525, 0.215, 0.420, 0.720], projection="polar")
    _draw_radar_axis(
        left,
        attributes,
        series,
        value_field="auc_values",
        scale=_reference_scale(auc_values, minimum=0.3, maximum=0.75),
        chart_side="left",
    )
    _draw_radar_axis(
        right,
        attributes,
        series,
        value_field="dp_values",
        scale=_reference_scale(dp_values, minimum=0.3, maximum=1.0),
        chart_side="right",
    )
    legend_font = max(8.0, min(22.0, 22.0 - 1.1 * max(len(series) - 6, 0)))
    handles = [
        Line2D(
            [0],
            [0],
            color=item.style.color,
            linestyle=item.style.line_style,
            linewidth=6,
        )
        for item in series
    ]
    figure.legend(
        handles,
        [item.display_name for item in series],
        loc="lower left",
        bbox_to_anchor=(0.10, -0.005, 0.80, 0.06),
        ncol=len(series),
        mode="expand",
        borderaxespad=0,
        frameon=False,
        prop={"family": "Times New Roman", "size": legend_font},
        handlelength=2.3,
        handletextpad=0.45,
        columnspacing=1.25,
    )
    return figure, (left, right)


__all__ = [
    "AttributeRadarSeries",
    "RadialScale",
    "build_attribute_radar_figure",
    "plan_radial_scale",
]
