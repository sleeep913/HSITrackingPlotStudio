from pathlib import Path

import pytest

from hsi_plotter.io.attribute_pdf_exporter import export_attribute_radar_pdf
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.plotting.attribute_radar import (
    AttributeRadarSeries,
    build_attribute_radar_figure,
    plan_radial_scale,
)


def _series(
    name: str,
    auc_values: tuple[float | None, ...],
    dp_values: tuple[float | None, ...] | None = None,
    color: str = "#FF2222",
):
    return AttributeRadarSeries(
        tracker_id=name.casefold(),
        display_name=name,
        auc_values=auc_values,
        dp_values=dp_values or auc_values,
        style=TrackerStyle(color=color, line_style="--", line_width=2.0),
    )


def test_dual_metric_radar_closes_curves_and_uses_project_style() -> None:
    figure, axes = build_attribute_radar_figure(
        ("BC", "DEF", "FM"),
        (_series("LRSST", (0.6, 0.7, 0.8), (0.8, 0.9, 0.7)),),
    )

    assert len(figure.axes) == 2
    assert len(axes) == 2
    assert all(axis.name == "polar" for axis in axes)
    assert axes[0].lines[0].get_ydata()[1] == pytest.approx(0.7)
    assert axes[1].lines[0].get_ydata()[1] == pytest.approx(0.9)
    for axis in axes:
        assert axis.lines[0].get_ydata()[0] == axis.lines[0].get_ydata()[-1]
        assert axis.lines[0].get_color().upper() == "#FF2222"
        assert axis.lines[0].get_linestyle() == "--"
    assert len(figure.legends) == 1
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["LRSST"]


def test_radar_requires_three_computable_attributes() -> None:
    with pytest.raises(ValueError, match="至少.*3 个"):
        build_attribute_radar_figure(
            ("BC", "DEF"),
            (_series("LRSST", (0.6, 0.7)),),
        )


def test_radial_scale_is_deterministic_and_contains_all_values() -> None:
    first = plan_radial_scale([0.52, 0.63, 0.91])
    second = plan_radial_scale([0.52, 0.63, 0.91])

    assert first == second
    assert first.minimum <= 0.52
    assert first.maximum >= 0.91
    assert len(first.ticks) >= 3


def test_pdf_export_uses_project_filename_and_auto_numbering(tmp_path: Path) -> None:
    series = (_series("LRSST", (0.6, 0.7, 0.8)),)

    first = export_attribute_radar_pdf(
        tmp_path,
        ("BC", "DEF", "FM"),
        series,
        project_name="HOT2023",
    )
    second = export_attribute_radar_pdf(
        tmp_path,
        ("BC", "DEF", "FM"),
        series,
        project_name="HOT2023",
    )

    assert first.name == "HOT2023_attributes.pdf"
    assert second.name == "HOT2023_attributes-1.pdf"
    assert first.stat().st_size > 0
    assert second.stat().st_size > 0
    assert first.read_bytes().startswith(b"%PDF")
