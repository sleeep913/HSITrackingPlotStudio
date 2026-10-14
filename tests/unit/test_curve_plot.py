from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hsi_plotter.models.recipe import CurveRecipe, RecipeTracker
from hsi_plotter.models.style import PlotStyle, TrackerStyle
from hsi_plotter.plotting.curve_plot import CurveSeries, build_curve_figure, build_metric_figure
from hsi_plotter.plotting.exporters import (
    FigureExportError,
    export_curve_figure,
    export_curve_pair,
)
from hsi_plotter.plotting.matlab_curve_plot import plan_matlab_layout


def _series(identifier: str, offset: float = 0.0) -> CurveSeries:
    return CurveSeries(
        tracker_id=identifier,
        display_name=identifier.upper(),
        auc=0.55 + offset,
        dp=0.75 + offset,
        precision_curve=np.linspace(0.0, 1.0, 51),
        success_curve=np.linspace(1.0, 0.0, 51),
        style=TrackerStyle(color="#2563EB", line_style="--", line_width=2.25),
    )


def _recipe() -> CurveRecipe:
    return CurveRecipe(
        name="test",
        plot_mode="both",
        trackers=(RecipeTracker("a", "A", True, 0),),
        style=PlotStyle(width_inches=6.0, height_inches=3.0, dpi=120),
    )


def test_renderer_builds_two_axes_with_stable_legend_and_styles() -> None:
    figure, diagnostics = build_curve_figure([_series("a"), _series("b", 0.1)], _recipe())

    assert len(figure.axes) == 2
    assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == [
        "A[0.750]",
        "B[0.850]",
    ]
    assert figure.axes[0].lines[-1].get_linestyle() == "--"
    assert figure.axes[0].get_legend()._loc == 4
    assert figure.axes[1].get_legend()._loc == 3
    assert figure.axes[0].title.get_fontweight() == "bold"
    assert diagnostics.font_substitution is None


def test_curve_origins_show_one_zero_on_preview_and_independent_exports() -> None:
    combined, _ = build_curve_figure([_series("a")], _recipe())
    precision, _ = build_metric_figure([_series("a")], _recipe(), "precision")
    success, _ = build_metric_figure([_series("a")], _recipe(), "success")

    for figure in (combined, precision, success):
        for axis in figure.axes:
            axis.figure.canvas.draw()
            x_origin = axis.get_xticklabels()[0].get_text()
            y_origin = axis.get_yticklabels()[0].get_text()
            assert x_origin == "0"
            assert y_origin == ""
            assert axis.get_yticklabels()[1].get_text() == "0.2"


def test_renderer_honors_single_plot_mode() -> None:
    recipe = _recipe()
    recipe = recipe.with_updates(plot_mode="success")

    figure, _ = build_curve_figure([_series("a")], recipe)

    assert len(figure.axes) == 1
    assert figure.axes[0].get_xlabel() == recipe.success_x_label


def test_precision_and_success_can_use_independent_orders() -> None:
    recipe = _recipe().with_updates(
        trackers=(
            RecipeTracker("a", "A", True, 0),
            RecipeTracker("b", "B", True, 1),
        ),
        precision_order=("b", "a"),
        success_order=("a", "b"),
        follow_metric_sort=False,
    )

    figure, _ = build_curve_figure([_series("a"), _series("b", 0.1)], recipe)

    assert [text.get_text()[0] for text in figure.axes[0].get_legend().get_texts()] == ["B", "A"]
    assert [text.get_text()[0] for text in figure.axes[1].get_legend().get_texts()] == ["A", "B"]
    assert [line.get_label()[0] for line in figure.axes[0].lines] == ["A", "B"]


def test_exporter_preserves_png_pixel_size_and_writes_vector_formats(tmp_path: Path) -> None:
    recipe = _recipe()
    series = [_series("a")]

    outputs = [
        export_curve_figure(series, recipe, tmp_path / f"curves.{suffix}")
        for suffix in ("png", "pdf", "svg", "eps")
    ]

    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
    with Image.open(outputs[0]) as image:
        assert image.size == (2796, 1200)


def test_repeated_rendering_has_identical_structural_signature() -> None:
    recipe = _recipe()

    first, _ = build_curve_figure([_series("a")], recipe)
    second, _ = build_curve_figure([_series("a")], recipe)

    def signature(fig):
        return [
            (
                axis.get_title(),
                axis.get_xlabel(),
                tuple(tuple(line.get_ydata()) for line in axis.lines),
            )
            for axis in fig.axes
        ]

    assert signature(first) == signature(second)


def test_export_pair_writes_two_independent_figures(tmp_path: Path) -> None:
    outputs = export_curve_pair([_series("a")], _recipe(), tmp_path, "png")

    assert [path.name for path in outputs] == [
        "Precision_plots_OPE.png",
        "Success_plots_OPE.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    precision, _ = build_metric_figure([_series("a")], _recipe(), "precision")
    success, _ = build_metric_figure([_series("a")], _recipe(), "success")
    assert len(precision.axes) == len(success.axes) == 1


def test_export_pair_uses_the_same_automatic_number_for_both_files(tmp_path: Path) -> None:
    (tmp_path / "Success_plots_OPE.png").write_bytes(b"occupied")

    outputs = export_curve_pair([_series("a")], _recipe(), tmp_path, "png")

    assert [path.name for path in outputs] == [
        "Precision_plots_OPE-2.png",
        "Success_plots_OPE-2.png",
    ]


def test_export_pair_wraps_and_logs_directory_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied", encoding="utf-8")

    with pytest.raises(FigureExportError, match="无法导出"):
        export_curve_pair([_series("a")], _recipe(), blocked / "child", "png")

    assert "curve figure pair export failed" in caplog.text


def test_dynamic_layout_keeps_one_column_and_scales_dense_legends() -> None:
    sparse = plan_matlab_layout(
        6,
        combined=True,
        figure_size_inches=(9.8, 4.4),
        longest_label_characters=14,
    )
    dense = plan_matlab_layout(
        15,
        combined=True,
        figure_size_inches=(9.8, 5.1),
        longest_label_characters=18,
    )
    enlarged = plan_matlab_layout(
        15,
        combined=True,
        figure_size_inches=(9.8, 5.1),
        longest_label_characters=18,
        legend_scale=1.4,
    )

    assert sparse.legend_columns == dense.legend_columns == 1
    assert dense.figure_height > sparse.figure_height
    assert dense.legend_font_size < sparse.legend_font_size
    assert dense.legend_label_spacing < sparse.legend_label_spacing
    assert dense.line_width < sparse.line_width
    assert enlarged.legend_columns == 1
    assert enlarged.legend_font_size > dense.legend_font_size
    assert enlarged.figure_height >= dense.figure_height


def test_ten_tracker_legend_stays_inside_axes_with_single_column() -> None:
    series = [_series(f"tracker-{index:02d}", index * 0.001) for index in range(10)]
    recipe = _recipe().with_updates(
        trackers=tuple(
            RecipeTracker(item.tracker_id, item.display_name, True, index)
            for index, item in enumerate(series)
        )
    )

    figure, _ = build_curve_figure(
        series,
        recipe,
        figure_size_inches=(10.2, 4.8),
        render_dpi=140,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    for axis in figure.axes:
        legend = axis.get_legend()
        legend_box = legend.get_window_extent(renderer)
        axes_box = axis.get_window_extent(renderer)
        assert legend._ncols == 1
        assert axes_box.contains(legend_box.x0, legend_box.y0)
        assert axes_box.contains(legend_box.x1, legend_box.y1)
        assert legend_box.height / axes_box.height < 0.64
