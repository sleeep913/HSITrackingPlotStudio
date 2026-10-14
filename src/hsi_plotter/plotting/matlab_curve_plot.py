from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from hsi_plotter.metrics.tracking import DISTANCE_THRESHOLDS, SUCCESS_THRESHOLDS
from hsi_plotter.models.recipe import CurveRecipe
from hsi_plotter.models.style import matlab_hot2023_plot_style

CurveKind = Literal["precision", "success"]


@dataclass(frozen=True, slots=True)
class MatlabRenderDiagnostics:
    requested_font: str
    resolved_font: str
    font_substitution: str | None = None
    clipping_warning: str | None = None


@dataclass(frozen=True, slots=True)
class MatlabCurveLayout:
    figure_width: float
    figure_height: float
    title_size: float
    label_size: float
    tick_size: float
    legend_font_size: float
    legend_columns: int
    legend_label_spacing: float
    legend_handle_length: float
    legend_handle_text_pad: float
    legend_border_pad: float
    legend_border_axes_pad: float
    line_width: float
    subplot_spacing: float


def plan_matlab_layout(
    visible_count: int,
    *,
    combined: bool,
    figure_size_inches: tuple[float, float] | None = None,
    longest_label_characters: int = 0,
    legend_scale: float = 1.0,
) -> MatlabCurveLayout:
    """Return a deterministic one-column layout for the current density."""
    style = matlab_hot2023_plot_style(combined=combined)
    dense_items = max(0, visible_count - 8)
    if figure_size_inches is None:
        width = style.width_inches
        height = (
            style.height_inches
            + min(1.35, dense_items * 0.11)
            + max(0.0, legend_scale - 1.0) * min(0.8, visible_count * 0.04)
        )
    else:
        width = max(3.2, float(figure_size_inches[0]))
        height = max(3.0, float(figure_size_inches[1]))

    base_width = style.width_inches
    width_scale = min(1.0, max(0.82, width / base_width))
    density_scale = max(0.74, 1.0 - dense_items * 0.035)
    name_scale = min(1.0, max(0.82, 20.0 / max(longest_label_characters, 20)))
    scale = min(width_scale, density_scale, name_scale)
    typography_scale = max(0.84, scale)

    return MatlabCurveLayout(
        figure_width=width,
        figure_height=height,
        title_size=(10.0 if combined else 12.0) * typography_scale,
        label_size=(9.0 if combined else 11.0) * typography_scale,
        tick_size=(8.0 if combined else 10.0) * typography_scale,
        legend_font_size=max(4.5, (7.0 if combined else 8.0) * scale * legend_scale),
        legend_columns=1,
        legend_label_spacing=max(0.14, 0.46 * scale * legend_scale),
        legend_handle_length=max(1.1, 2.0 * scale * legend_scale),
        legend_handle_text_pad=max(0.35, 0.65 * scale * legend_scale),
        legend_border_pad=max(0.2, 0.42 * scale * legend_scale),
        legend_border_axes_pad=0.9 * legend_scale,
        line_width=max(1.55, 2.0 * scale),
        subplot_spacing=0.34 if combined else 0.0,
    )


def _ordered(series: list[Any], recipe: CurveRecipe, kind: CurveKind) -> list[tuple[Any, str]]:
    by_id = {item.tracker_id: item for item in series}
    configured = {item.tracker_id: item for item in recipe.trackers}
    manual = tuple(item.tracker_id for item in sorted(recipe.trackers, key=lambda item: item.order))
    requested = recipe.precision_order if kind == "precision" else recipe.success_order
    order = tuple(requested) or manual
    order += tuple(item.tracker_id for item in series if item.tracker_id not in order)
    result: list[tuple[Any, str]] = []
    for tracker_id in order:
        item = by_id.get(tracker_id)
        setting = configured.get(tracker_id)
        if item is None or (setting is not None and not setting.visible):
            continue
        result.append((item, setting.display_name if setting is not None else item.display_name))
    return result


def _configure_axis(
    axis: Any,
    recipe: CurveRecipe,
    kind: CurveKind,
    font: str,
    *,
    layout: MatlabCurveLayout,
) -> None:
    title = recipe.precision_title if kind == "precision" else recipe.success_title
    x_label = recipe.precision_x_label if kind == "precision" else recipe.success_x_label
    y_label = recipe.precision_y_label if kind == "precision" else recipe.success_y_label
    axis.set_title(title, fontsize=layout.title_size, fontweight="bold", fontfamily=font)
    axis.set_xlabel(x_label, fontsize=layout.label_size, fontfamily=font)
    axis.set_ylabel(y_label, fontsize=layout.label_size, fontfamily=font)
    if kind == "precision":
        axis.set_xlim(0.0, 50.0)
        axis.set_xticks(np.arange(0.0, 51.0, 10.0))
    else:
        axis.set_xlim(0.0, 1.0)
        success_ticks = np.arange(0.0, 1.01, 0.2)
        axis.set_xticks(success_ticks)
        axis.set_xticklabels(
            ["0" if value == 0.0 else f"{value:.1f}" for value in success_ticks]
        )
    axis.set_ylim(0.0, 1.0)
    y_ticks = np.arange(0.0, 1.01, 0.2)
    axis.set_yticks(y_ticks)
    axis.set_yticklabels(
        ["" if value == 0.0 else f"{value:.1f}" for value in y_ticks]
    )
    axis.tick_params(
        labelsize=layout.tick_size,
        colors="#000000",
        width=0.8,
        direction="in",
    )
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        label.set_fontfamily(font)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(0.8)
    axis.grid(True, linestyle="--", color="#B3B3B3", alpha=0.5, linewidth=0.6)
    axis.set_facecolor("#FFFFFF")


def _draw_axis(
    axis: Any,
    series: list[Any],
    recipe: CurveRecipe,
    kind: CurveKind,
    font: str,
    *,
    layout: MatlabCurveLayout,
) -> None:
    ordered = _ordered(series, recipe, kind)
    if not ordered:
        raise ValueError("当前配方隐藏了全部跟踪器")
    _configure_axis(axis, recipe, kind, font, layout=layout)
    handles: dict[str, Any] = {}
    for item, display_name in reversed(ordered):
        if kind == "precision":
            x_values, curve, metric = DISTANCE_THRESHOLDS[1:], item.precision_curve[1:], item.dp
        else:
            x_values, curve, metric = SUCCESS_THRESHOLDS[1:], item.success_curve[1:], item.auc
        label = f"{display_name}[{metric:.3f}]"
        (line,) = axis.plot(
            x_values,
            curve,
            color=item.style.color,
            linestyle=item.style.line_style,
            linewidth=layout.line_width,
            label=label,
        )
        handles[item.tracker_id] = line
    labels = [
        f"{name}[{(item.dp if kind == 'precision' else item.auc):.3f}]"
        for item, name in ordered
    ]
    legend = axis.legend(
        [handles[item.tracker_id] for item, _ in ordered],
        labels,
        loc="lower right" if kind == "precision" else "lower left",
        fontsize=layout.legend_font_size,
        ncol=layout.legend_columns,
        labelspacing=layout.legend_label_spacing,
        handlelength=layout.legend_handle_length,
        handletextpad=layout.legend_handle_text_pad,
        borderpad=layout.legend_border_pad,
        borderaxespad=layout.legend_border_axes_pad,
        frameon=True,
        fancybox=False,
        edgecolor="#000000",
        facecolor="#FFFFFF",
        framealpha=1.0,
    )
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_fontfamily(font)


def build_metric_figure(
    series: list[Any], recipe: CurveRecipe, kind: CurveKind
) -> tuple[Figure, MatlabRenderDiagnostics]:
    from hsi_plotter.plotting.curve_plot import resolve_font_family

    recipe.validate()
    for item in series:
        item.validate()
    if not series:
        raise ValueError("至少需要一个可绘制的跟踪器结果")
    style = matlab_hot2023_plot_style(combined=False)
    font, substitution = resolve_font_family(style.font_family)
    ordered_count = len(_ordered(series, recipe, kind))
    longest_label = max(
        (len(name) + 7 for _, name in _ordered(series, recipe, kind)),
        default=0,
    )
    layout = plan_matlab_layout(
        ordered_count,
        combined=False,
        longest_label_characters=longest_label,
        legend_scale=recipe.style.legend_scale,
    )
    figure = Figure(
        figsize=(layout.figure_width, layout.figure_height),
        dpi=style.dpi,
        facecolor="white",
    )
    FigureCanvasAgg(figure)
    axis = figure.subplots(1, 1)
    _draw_axis(axis, series, recipe, kind, font, layout=layout)
    figure.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.88)
    figure.canvas.draw()
    return figure, MatlabRenderDiagnostics("Times New Roman", font, substitution)


def build_curve_figure(
    series: list[Any],
    recipe: CurveRecipe,
    *,
    figure_size_inches: tuple[float, float] | None = None,
    render_dpi: int | None = None,
) -> tuple[Figure, MatlabRenderDiagnostics]:
    from hsi_plotter.plotting.curve_plot import resolve_font_family

    recipe.validate()
    for item in series:
        item.validate()
    if not series:
        raise ValueError("至少需要一个可绘制的跟踪器结果")
    kinds: list[CurveKind] = (
        ["precision", "success"] if recipe.plot_mode == "both" else [recipe.plot_mode]  # type: ignore[list-item]
    )
    style = matlab_hot2023_plot_style(combined=len(kinds) == 2)
    font, substitution = resolve_font_family(style.font_family)
    configured = [_ordered(series, recipe, kind) for kind in kinds]
    visible_count = max((len(items) for items in configured), default=0)
    longest_label = max(
        (len(name) + 7 for items in configured for _, name in items),
        default=0,
    )
    layout = plan_matlab_layout(
        visible_count,
        combined=len(kinds) == 2,
        figure_size_inches=figure_size_inches,
        longest_label_characters=longest_label,
        legend_scale=recipe.style.legend_scale,
    )
    figure = Figure(
        figsize=(layout.figure_width, layout.figure_height),
        dpi=render_dpi or style.dpi,
        facecolor="white",
    )
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(kinds), squeeze=False)[0]
    for axis, kind in zip(axes, kinds, strict=True):
        _draw_axis(axis, series, recipe, kind, font, layout=layout)
    figure.subplots_adjust(
        left=0.07 if len(kinds) == 2 else 0.16,
        right=0.985 if len(kinds) == 2 else 0.97,
        bottom=0.16,
        top=0.88,
        wspace=layout.subplot_spacing,
    )
    figure.canvas.draw()
    return figure, MatlabRenderDiagnostics("Times New Roman", font, substitution)
