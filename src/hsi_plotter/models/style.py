from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class TrackerStyle:
    color: str
    line_style: str = "-"
    line_width: float = 2.0


class PlotStylePreset(StrEnum):
    PAPER_SINGLE_COLUMN = "paper_single_column"
    PAPER_DOUBLE_COLUMN = "paper_double_column"
    PRESENTATION = "presentation"


@dataclass(frozen=True, slots=True)
class PlotStyle:
    preset: PlotStylePreset = PlotStylePreset.PAPER_DOUBLE_COLUMN
    font_family: str = "Times New Roman"
    font_size: float = 9.0
    title_size: float = 10.0
    legend_size: float = 8.0
    legend_scale: float = 1.0
    text_color: str = "#111827"
    background_color: str = "#FFFFFF"
    axes_color: str = "#334155"
    grid_enabled: bool = True
    grid_color: str = "#CBD5E1"
    grid_alpha: float = 0.45
    axes_line_width: float = 0.8
    default_line_width: float = 2.0
    marker: str = ""
    legend_location: str = "best"
    legend_columns: int = 1
    width_inches: float = 7.2
    height_inches: float = 3.2
    dpi: int = 300
    margin_left: float = 0.10
    margin_right: float = 0.97
    margin_bottom: float = 0.17
    margin_top: float = 0.88

    def validate(self) -> None:
        if not self.font_family.strip():
            raise ValueError("字体名称不能为空")
        if self.width_inches <= 0 or self.height_inches <= 0:
            raise ValueError("画布宽度和高度必须大于 0")
        if self.dpi <= 0:
            raise ValueError("DPI 必须大于 0")
        if self.legend_columns < 1:
            raise ValueError("图例列数必须至少为 1")
        if not 0.6 <= self.legend_scale <= 1.8:
            raise ValueError("图例缩放必须位于 0.6 到 1.8 之间")
        if not 0.0 <= self.grid_alpha <= 1.0:
            raise ValueError("网格透明度必须位于 0 到 1 之间")
        if not (
            0 <= self.margin_left < self.margin_right <= 1
            and 0 <= self.margin_bottom < self.margin_top <= 1
        ):
            raise ValueError("画布边距必须位于 0 到 1 之间且左右、上下顺序正确")


BUILTIN_PLOT_PRESETS: dict[PlotStylePreset, PlotStyle] = {
    PlotStylePreset.PAPER_SINGLE_COLUMN: PlotStyle(
        preset=PlotStylePreset.PAPER_SINGLE_COLUMN,
        width_inches=3.5,
        height_inches=2.65,
        font_size=8.0,
        title_size=9.0,
        legend_size=7.0,
        default_line_width=1.6,
    ),
    PlotStylePreset.PAPER_DOUBLE_COLUMN: PlotStyle(),
    PlotStylePreset.PRESENTATION: PlotStyle(
        preset=PlotStylePreset.PRESENTATION,
        font_family="Arial",
        width_inches=10.0,
        height_inches=5.6,
        font_size=15.0,
        title_size=18.0,
        legend_size=13.0,
        default_line_width=2.8,
        margin_left=0.09,
        margin_right=0.98,
        margin_bottom=0.14,
        margin_top=0.90,
    ),
}


def plot_style_preset(preset: PlotStylePreset | str) -> PlotStyle:
    try:
        key = PlotStylePreset(preset)
    except ValueError as exc:
        raise ValueError(f"未知绘图样式预设：{preset}") from exc
    return replace(BUILTIN_PLOT_PRESETS[key])


#: Deterministic low-saturation palette used for tracks, curves and bounding
#: boxes so the same tracker keeps the same appearance across modules.
DEFAULT_COLOR_PALETTE: tuple[str, ...] = (
    "#2563EB",
    "#C2410C",
    "#0F766E",
    "#7C3AED",
    "#B91C1C",
    "#0369A1",
    "#A16207",
    "#BE185D",
    "#4D7C0F",
    "#4338CA",
)

DEFAULT_LINE_STYLES: tuple[str, ...] = ("-", "--", "-.", ":")

DEFAULT_GROUND_TRUTH_STYLE = TrackerStyle(color="#111827", line_style="-", line_width=2.5)

MATLAB_LINES_PALETTE: tuple[str, ...] = (
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
)


def matlab_style_for_index(index: int) -> TrackerStyle:
    """Match the legacy app's ``lines(n)`` colour order and dash rule."""
    order = max(int(index), 0)
    return TrackerStyle(
        color=MATLAB_LINES_PALETTE[order % len(MATLAB_LINES_PALETTE)],
        line_style="--" if order % 4 == 1 else "-",
        line_width=2.0,
    )


def matlab_hot2023_plot_style(*, combined: bool = False) -> PlotStyle:
    """The single fixed publication style used by the curve module."""
    return PlotStyle(
        font_family="Times New Roman",
        font_size=10.0,
        title_size=12.0,
        legend_size=8.0,
        text_color="#000000",
        background_color="#FFFFFF",
        axes_color="#000000",
        grid_enabled=True,
        grid_color="#B3B3B3",
        grid_alpha=0.5,
        axes_line_width=0.8,
        default_line_width=2.0,
        marker="",
        legend_location="best",
        legend_columns=1,
        width_inches=9.32 if combined else 4.0,
        height_inches=4.0 if combined else 3.2,
        dpi=300,
        margin_left=0.16,
        margin_right=0.97,
        margin_bottom=0.17,
        margin_top=0.88,
    )


def default_style_for_index(index: int) -> TrackerStyle:
    """Return a stable style for the n-th tracker of a project."""
    order = max(int(index), 0)
    cycle = order // len(DEFAULT_COLOR_PALETTE)
    return TrackerStyle(
        color=DEFAULT_COLOR_PALETTE[order % len(DEFAULT_COLOR_PALETTE)],
        line_style=DEFAULT_LINE_STYLES[cycle % len(DEFAULT_LINE_STYLES)],
        line_width=2.0,
    )
