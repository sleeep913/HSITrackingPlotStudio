from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from hsi_plotter.metrics.tracking import DISTANCE_THRESHOLDS, SUCCESS_THRESHOLDS
from hsi_plotter.models.recipe import CurveRecipe
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.plotting.matlab_curve_plot import (
    build_curve_figure as _build_matlab_curve_figure,
)
from hsi_plotter.plotting.matlab_curve_plot import (
    build_metric_figure as _build_matlab_metric_figure,
)


@dataclass(frozen=True, slots=True)
class CurveSeries:
    tracker_id: str
    display_name: str
    auc: float
    dp: float
    precision_curve: NDArray[np.float64]
    success_curve: NDArray[np.float64]
    style: TrackerStyle

    def validate(self) -> None:
        if np.asarray(self.precision_curve).shape != (51,):
            raise ValueError(f"{self.display_name} 的 Precision 曲线必须包含 51 个点")
        if np.asarray(self.success_curve).shape != (51,):
            raise ValueError(f"{self.display_name} 的 Success 曲线必须包含 51 个点")


def series_from_recipe(recipe: CurveRecipe) -> list[CurveSeries]:
    """Restore all curve inputs embedded by a version-2 recipe."""
    restored: list[CurveSeries] = []
    for entry in sorted(recipe.trackers, key=lambda item: item.order):
        if not entry.has_embedded_data or entry.style is None:
            continue
        restored.append(
            CurveSeries(
                tracker_id=entry.tracker_id,
                display_name=entry.display_name,
                auc=float(entry.auc),
                dp=float(entry.dp),
                precision_curve=np.asarray(entry.precision_curve, dtype=np.float64),
                success_curve=np.asarray(entry.success_curve, dtype=np.float64),
                style=entry.style,
            )
        )
    return restored


@dataclass(frozen=True, slots=True)
class RenderDiagnostics:
    requested_font: str
    resolved_font: str
    font_substitution: str | None = None
    clipping_warning: str | None = None


def resolve_font_family(requested: str) -> tuple[str, str | None]:
    try:
        path = font_manager.findfont(
            font_manager.FontProperties(family=[requested]), fallback_to_default=False
        )
    except ValueError:
        fallback = "DejaVu Sans"
        return fallback, f"字体“{requested}”不可用，将使用 {fallback}。"
    resolved = font_manager.FontProperties(fname=path).get_name()
    return resolved, None


def _ordered_visible_series(
    series: list[CurveSeries], recipe: CurveRecipe, *, kind: str
) -> list[tuple[CurveSeries, str]]:
    by_id = {item.tracker_id: item for item in series}
    manual = [item.tracker_id for item in sorted(recipe.trackers, key=lambda item: item.order)]
    requested = recipe.precision_order if kind == "precision" else recipe.success_order
    order = tuple(requested) or tuple(manual)
    position = {identifier: index for index, identifier in enumerate(order)}
    configured = sorted(
        recipe.trackers,
        key=lambda item: (position.get(item.tracker_id, len(position) + item.order), item.order),
    )
    result: list[tuple[CurveSeries, str]] = []
    consumed: set[str] = set()
    for entry in configured:
        item = by_id.get(entry.tracker_id)
        if item is None or not entry.visible or entry.tracker_id in consumed:
            continue
        result.append((item, entry.display_name))
        consumed.add(entry.tracker_id)
    result.extend((item, item.display_name) for item in series if item.tracker_id not in consumed)
    return result


def _configure_axis(axis, recipe: CurveRecipe, *, kind: str, font_family: str) -> None:
    style = recipe.style
    if kind == "precision":
        axis.set_title(recipe.precision_title, fontsize=style.title_size, fontfamily=font_family)
        axis.set_xlabel(recipe.precision_x_label, fontfamily=font_family)
        axis.set_ylabel(recipe.precision_y_label, fontfamily=font_family)
        axis.set_xlim(recipe.precision_x_min, recipe.precision_x_max)
        axis.set_xticks(
            np.arange(
                recipe.precision_x_min,
                recipe.precision_x_max + recipe.precision_tick_step * 0.5,
                recipe.precision_tick_step,
            )
        )
    else:
        axis.set_title(recipe.success_title, fontsize=style.title_size, fontfamily=font_family)
        axis.set_xlabel(recipe.success_x_label, fontfamily=font_family)
        axis.set_ylabel(recipe.success_y_label, fontfamily=font_family)
        axis.set_xlim(recipe.success_x_min, recipe.success_x_max)
        axis.set_xticks(
            np.arange(
                recipe.success_x_min,
                recipe.success_x_max + recipe.success_tick_step * 0.5,
                recipe.success_tick_step,
            )
        )
    axis.set_ylim(recipe.y_min, recipe.y_max)
    axis.set_yticks(
        np.arange(recipe.y_min, recipe.y_max + recipe.y_tick_step * 0.5, recipe.y_tick_step)
    )
    axis.tick_params(
        labelsize=style.font_size,
        colors=style.text_color,
        width=style.axes_line_width,
    )
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        label.set_fontfamily(font_family)
    for spine in axis.spines.values():
        spine.set_color(style.axes_color)
        spine.set_linewidth(style.axes_line_width)
    axis.grid(
        style.grid_enabled,
        color=style.grid_color,
        alpha=style.grid_alpha,
        linewidth=0.6,
    )
    axis.set_facecolor(style.background_color)


def _build_configurable_curve_figure(
    series: list[CurveSeries], recipe: CurveRecipe
) -> tuple[Figure, RenderDiagnostics]:
    recipe.validate()
    for item in series:
        item.validate()
    if not series:
        raise ValueError("至少需要一个可绘制的跟踪器结果")
    font_family, substitution = resolve_font_family(recipe.style.font_family)
    kinds = ["precision", "success"] if recipe.plot_mode == "both" else [recipe.plot_mode]
    figure = Figure(
        figsize=(recipe.style.width_inches, recipe.style.height_inches),
        dpi=recipe.style.dpi,
        facecolor=recipe.style.background_color,
    )
    FigureCanvasAgg(figure)
    axes_value = figure.subplots(1, len(kinds), squeeze=False)[0]
    for axis, kind in zip(axes_value, kinds, strict=True):
        ordered = _ordered_visible_series(series, recipe, kind=kind)
        if not ordered:
            raise ValueError("当前配方隐藏了全部跟踪器")
        _configure_axis(axis, recipe, kind=kind, font_family=font_family)
        thresholds = DISTANCE_THRESHOLDS if kind == "precision" else SUCCESS_THRESHOLDS
        for item, display_name in ordered:
            curve = item.precision_curve if kind == "precision" else item.success_curve
            metric = item.dp if kind == "precision" else item.auc
            label = f"{display_name} [{metric:.{recipe.decimal_places}f}]"
            axis.plot(
                thresholds,
                curve,
                color=item.style.color,
                linestyle=item.style.line_style,
                linewidth=item.style.line_width,
                marker=recipe.style.marker or None,
                label=label,
            )
        legend = axis.legend(
            loc=recipe.style.legend_location,
            ncol=recipe.style.legend_columns,
            fontsize=recipe.style.legend_size,
            frameon=False,
        )
        for text in legend.get_texts():
            text.set_fontfamily(font_family)
    figure.subplots_adjust(
        left=recipe.style.margin_left,
        right=recipe.style.margin_right,
        bottom=recipe.style.margin_bottom,
        top=recipe.style.margin_top,
        wspace=0.28,
    )
    figure.canvas.draw()
    total_label_characters = sum(
        len(entry.display_name) + recipe.decimal_places + 4
        for entry in recipe.trackers
        if entry.visible
    )
    clipping = None
    if total_label_characters > recipe.style.width_inches * 28 * recipe.style.legend_columns:
        clipping = "图例文本较长，建议增加画布宽度、图例列数或缩短显示名称。"
    return figure, RenderDiagnostics(
        requested_font=recipe.style.font_family,
        resolved_font=font_family,
        font_substitution=substitution,
        clipping_warning=clipping,
    )


def build_curve_figure(
    series: list[CurveSeries],
    recipe: CurveRecipe,
    *,
    figure_size_inches: tuple[float, float] | None = None,
    render_dpi: int | None = None,
):
    return _build_matlab_curve_figure(
        series,
        recipe,
        figure_size_inches=figure_size_inches,
        render_dpi=render_dpi,
    )


def build_metric_figure(series: list[CurveSeries], recipe: CurveRecipe, kind: str):
    return _build_matlab_metric_figure(series, recipe, kind)
