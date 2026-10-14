"""Atomic PDF export for the combined AUC/DP attribute radar figure."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from hsi_plotter.io.result_writer import (
    _temporary_target,
    next_available_path,
    sanitize_file_stem,
)
from hsi_plotter.plotting.attribute_radar import (
    AttributeRadarSeries,
    build_attribute_radar_figure,
)

LOGGER = logging.getLogger(__name__)


class AttributePdfExportError(OSError):
    """Raised when an attribute radar PDF cannot be written."""


def export_attribute_radar_pdf(
    directory: Path,
    attributes: tuple[str, ...],
    series: tuple[AttributeRadarSeries, ...],
    *,
    project_name: str,
) -> Path:
    figure = None
    stem = sanitize_file_stem(project_name or "Attributes")
    target = Path(directory) / f"{stem}_attributes.pdf"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target = next_available_path(target)
        LOGGER.info(
            "exporting combined attribute radar: target=%s attributes=%d trackers=%d",
            target,
            len(attributes),
            len(series),
        )
        figure, _ = build_attribute_radar_figure(attributes, series)
        with _temporary_target(target) as temporary:
            figure.savefig(
                temporary,
                format="pdf",
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.35,
            )
    except Exception as exc:
        LOGGER.exception("combined attribute radar PDF export failed: target=%s", target)
        raise AttributePdfExportError(f"无法导出属性雷达图 {target}：{exc}") from exc
    finally:
        if figure is not None:
            plt.close(figure)
    return target


__all__ = ["AttributePdfExportError", "export_attribute_radar_pdf"]
