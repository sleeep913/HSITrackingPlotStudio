from __future__ import annotations

import logging
from pathlib import Path

from hsi_plotter.io.result_writer import _temporary_target
from hsi_plotter.models.recipe import CurveRecipe
from hsi_plotter.plotting.curve_plot import CurveSeries, build_curve_figure, build_metric_figure

SUPPORTED_FIGURE_FORMATS = frozenset({"png", "pdf", "svg", "eps"})
LOGGER = logging.getLogger(__name__)


class FigureExportError(OSError):
    """Raised when a publication figure cannot be exported."""


def export_curve_figure(series: list[CurveSeries], recipe: CurveRecipe, destination: Path) -> Path:
    target = Path(destination)
    format_name = target.suffix.removeprefix(".").casefold()
    if format_name not in SUPPORTED_FIGURE_FORMATS:
        raise FigureExportError(f"不支持的图形格式 {target.suffix!r}，支持 PNG、PDF、SVG 和 EPS。")
    figure, _ = build_curve_figure(series, recipe)
    try:
        with _temporary_target(target) as temporary:
            figure.savefig(
                temporary,
                format=format_name,
                dpi=300,
                facecolor="#FFFFFF",
                bbox_inches=None,
            )
    except OSError as exc:
        raise FigureExportError(f"无法导出图形 {target}：{exc}") from exc
    finally:
        figure.clear()
    return target


def export_curve_pair(
    series: list[CurveSeries],
    recipe: CurveRecipe,
    directory: Path,
    format_name: str,
) -> tuple[Path, Path]:
    """Export independent Precision and Success figures in one action."""
    suffix = format_name.removeprefix(".").casefold()
    if suffix not in SUPPORTED_FIGURE_FORMATS:
        raise FigureExportError(f"不支持的图形格式 {format_name!r}，支持 PNG、PDF、SVG 和 EPS。")
    root = Path(directory)
    try:
        root.mkdir(parents=True, exist_ok=True)
        sequence = 1
        while True:
            postfix = "" if sequence == 1 else f"-{sequence}"
            targets = (
                root / f"Precision_plots_OPE{postfix}.{suffix}",
                root / f"Success_plots_OPE{postfix}.{suffix}",
            )
            if not any(target.exists() for target in targets):
                break
            sequence += 1
        for target, kind in zip(targets, ("precision", "success"), strict=True):
            figure = None
            try:
                figure, _ = build_metric_figure(series, recipe, kind)
                with _temporary_target(target) as temporary:
                    figure.savefig(
                        temporary,
                        format=suffix,
                        dpi=300,
                        facecolor="#FFFFFF",
                        bbox_inches=None,
                    )
            finally:
                if figure is not None:
                    figure.clear()
    except Exception as exc:
        LOGGER.exception(
            "curve figure pair export failed: directory=%s format=%s",
            root,
            suffix,
        )
        raise FigureExportError(f"无法导出两张曲线图到 {root}：{exc}") from exc
    return targets
