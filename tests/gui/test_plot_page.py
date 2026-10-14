import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, QSettings, QSizeF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QFileDialog

from hsi_plotter.metrics.tracking import AggregateMetrics
from hsi_plotter.models.evaluation import EvaluationRun, TrackerEvaluation
from hsi_plotter.models.project import Project
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.tracker_import import ResultLayout
from hsi_plotter.plotting.curve_plot import CurveSeries
from hsi_plotter.ui.color_picker import (
    STANDARD_TRACKER_COLORS,
    ColorPickerButton,
    SimpleColorDialog,
)
from hsi_plotter.ui.main_window import MainWindow
from hsi_plotter.ui.plot_page import InteractiveLegendCanvas, PlotPage, legend_scale_from_drag


@pytest.fixture
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _aggregate() -> AggregateMetrics:
    precision = np.linspace(0.0, 1.0, 51)
    success = np.linspace(1.0, 0.0, 51)
    return AggregateMetrics(precision, success, 0.49, 0.4, 8.0, 2, 20, 20, 0)


def _run() -> EvaluationRun:
    return EvaluationRun(
        trackers=(
            TrackerEvaluation(
                tracker_id="mht",
                tracker_name="MHT",
                layout=ResultLayout.FLAT,
                overall=_aggregate(),
            ),
        )
    )


def _project(tmp_path: Path) -> Project:
    return Project(
        name="paper",
        trackers=[
            Tracker(
                tracker_id="mht",
                display_name="MHT",
                results_path=tmp_path / "results",
                style=TrackerStyle("#2563EB", "--", 2.5),
            )
        ],
    )


def test_plot_page_consumes_current_evaluation_without_reading_files(
    app: QApplication, tmp_path: Path
) -> None:
    page = PlotPage()
    page.set_project(_project(tmp_path))

    page.set_evaluation_run(_run())

    assert page.tracker_table.rowCount() == 1
    assert page.tracker_table.item(0, 2).text() == "MHT"
    assert page.preview_canvas.figure.axes
    assert page.preview_canvas.figure.dpi <= 192
    assert page.status_label.text().startswith("已载入")


def test_main_window_registers_real_curve_plot_page(app: QApplication, tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)

    assert isinstance(window.module_pages["曲线绘制"], PlotPage)
    assert window.module_tabs.tabText(1) == "曲线绘制"

    window.close()
    app.processEvents()


def test_plot_page_imports_legacy_mat_without_project_data_scan(
    app: QApplication, tmp_path: Path
) -> None:
    path = tmp_path / "LegacyTracker-0.500-0.700.mat"
    precision = np.linspace(0.0, 1.0, 51)
    success = np.linspace(1.0, 0.0, 51)
    savemat(path, {"distance_rec": precision[None, :], "PASCAL_rec": success[None, :]})
    page = PlotPage()
    page.set_project(_project(tmp_path))

    imported = page.import_mat_files([path])

    assert imported == 1
    assert page.tracker_table.rowCount() == 1
    assert page.tracker_table.item(0, 2).text() == "LegacyTracker"
    assert len(page.preview_canvas.figure.axes) == 2


def test_plot_page_exposes_only_the_calibrated_controls(
    app: QApplication, tmp_path: Path
) -> None:
    page = PlotPage()
    page.set_project(_project(tmp_path))
    page.set_evaluation_run(_run())

    assert page.tracker_table.columnCount() == 7
    assert page.tracker_table.horizontalHeaderItem(1).text() == "排名"
    assert page.import_mat_button.text() == "加入跟踪器…"
    assert page.load_recipe_button.text() == "导入配置…"
    assert page.save_recipe_button.text() == "导出配置…"
    assert page.export_button.text() == "保存两张图…"
    assert page.select_all_button.text() == "全选"
    assert page.refresh_button.text() == "刷新预览"
    assert page.advanced_panel.isHidden()
    assert page.reset_panel.isHidden()
    assert not page.auc_sort_combo.isHidden()
    assert not page.dp_sort_combo.isHidden()
    assert not page.table_rank_combo.isHidden()


def test_tracker_rows_use_color_preview_and_line_style_selector(
    app: QApplication, tmp_path: Path
) -> None:
    page = PlotPage()
    page.set_project(_project(tmp_path))
    page.set_evaluation_run(_run())

    color_cell = page.tracker_table.cellWidget(0, 5)
    color_button = color_cell.findChild(ColorPickerButton) if color_cell is not None else None
    line_combo = page.tracker_table.cellWidget(0, 6)
    assert color_button is not None
    assert color_button.property("trackerColor") == "#0072BD"
    assert color_button.text() == ""
    assert color_button.width() <= 42
    assert "background-color" in color_button.styleSheet()
    assert isinstance(line_combo, QComboBox)
    assert {line_combo.itemData(index) for index in range(line_combo.count())} >= {
        "-",
        "--",
        "-.",
        ":",
    }


def test_simple_color_dialog_offers_standard_and_persistent_custom_colors(
    app: QApplication, tmp_path: Path
) -> None:
    settings = QSettings(str(tmp_path / "colors.ini"), QSettings.Format.IniFormat)
    dialog = SimpleColorDialog("#0072BD", settings=settings)

    assert len(STANDARD_TRACKER_COLORS) == 28
    assert len(set(STANDARD_TRACKER_COLORS)) == 28
    assert len(dialog.swatch_buttons) == 28
    assert all(button.text() == "" for button in dialog.swatch_buttons)

    dialog.hex_edit.setText("#12abef")
    assert dialog.selected_color() == "#12ABEF"
    assert dialog.preview.property("previewColor") == "#12ABEF"
    assert dialog.button_box.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.add_custom_button.click()
    assert dialog.custom_colors == ("#12ABEF",)

    restored = SimpleColorDialog("#000000", settings=settings)
    assert restored.custom_colors == ("#12ABEF",)
    assert len(restored.custom_swatch_buttons) == 1
    restored.custom_swatch_buttons[0].click()
    restored.remove_custom_button.click()
    assert restored.custom_colors == ()

    dialog.hex_edit.setText("not-a-color")
    assert not dialog.button_box.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_plot_edits_wait_for_explicit_preview_refresh(
    app: QApplication, tmp_path: Path
) -> None:
    page = PlotPage()
    page.set_project(_project(tmp_path))
    page.set_evaluation_run(_run())
    original_canvas = page.preview_canvas
    color_cell = page.tracker_table.cellWidget(0, 5)
    color_button = color_cell.findChild(ColorPickerButton)

    color_button.set_color("#FF0000")
    color_button.color_changed.emit("#FF0000")

    assert page.preview_canvas is original_canvas
    assert page.refresh_button.isEnabled()
    assert "刷新预览" in page.status_label.text()

    page.refresh_button.click()
    app.processEvents()

    assert page.preview_canvas is not original_canvas
    assert not page.refresh_button.isEnabled()
    assert page.preview_canvas.figure.axes[0].lines[0].get_color() == "#FF0000"


def test_select_all_stages_visibility_without_immediate_redraw(
    app: QApplication, tmp_path: Path
) -> None:
    page = PlotPage()
    page.set_project(_project(tmp_path))
    page.set_evaluation_run(_run())
    original_canvas = page.preview_canvas
    visible_item = page.tracker_table.item(0, 0)
    visible_item.setCheckState(Qt.CheckState.Unchecked)

    page.select_all_button.click()

    assert visible_item.checkState() == Qt.CheckState.Checked
    assert page.preview_canvas is original_canvas
    assert page.refresh_button.isEnabled()

    page.select_all_button.click()

    assert all(
        page.tracker_table.item(row, 0).checkState() == Qt.CheckState.Unchecked
        for row in range(page.tracker_table.rowCount())
    )
    assert page.preview_canvas is original_canvas


def test_preview_dpi_is_stable_on_high_dpi_displays(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(PlotPage, "logicalDpiX", lambda _self: 240)
    page = PlotPage()
    page.set_project(_project(tmp_path))
    page.set_evaluation_run(_run())

    assert page.preview_canvas.figure.dpi == 140
    assert page.preview_canvas.figure._original_dpi == 140
    assert max(axis.title.get_fontsize() for axis in page.preview_canvas.figure.axes) <= 10


def test_preview_host_keeps_a_wide_coordinated_aspect_ratio(app: QApplication) -> None:
    page = PlotPage()

    assert page.preview_plot_host.hasHeightForWidth()
    page.preview_plot_host.set_content_count(10)
    ten_tracker_height = page.preview_plot_host.heightForWidth(1000)
    page.preview_plot_host.set_content_count(20)
    twenty_tracker_height = page.preview_plot_host.heightForWidth(1000)

    assert ten_tracker_height >= 480
    assert twenty_tracker_height > ten_tracker_height


def test_preview_host_receives_nonzero_height_when_page_is_shown(app: QApplication) -> None:
    page = PlotPage()
    page.resize(1500, 900)
    page.show()
    app.processEvents()
    app.processEvents()

    assert page.preview_plot_host.width() > 800
    assert page.preview_plot_host.height() > 350


def test_preview_figure_reflows_to_a_larger_window(
    app: QApplication, tmp_path: Path
) -> None:
    page = PlotPage()
    page.resize(1450, 850)
    page.show()
    page.set_project(_project(tmp_path))
    page.set_evaluation_run(_run())
    app.processEvents()
    first_width = page.preview_canvas.figure.get_figwidth()

    page.resize(1800, 1000)
    app.processEvents()
    page._refresh_preview_after_resize()

    assert page.preview_canvas.figure.get_figwidth() > first_width


def test_self_contained_recipe_restores_curves_without_mat_files(
    app: QApplication, tmp_path: Path
) -> None:
    first = PlotPage()
    first.set_project(_project(tmp_path))
    first.set_evaluation_run(_run())
    recipe = first._recipe
    assert recipe.is_self_contained

    restored = PlotPage()
    restored.set_project(_project(tmp_path))
    restored._apply_recipe(recipe)

    assert len(restored._series) == 1
    assert restored._series[0].display_name == "MHT"
    np.testing.assert_allclose(restored._series[0].precision_curve, np.linspace(0.0, 1.0, 51))
    assert restored.status_label.text().startswith("已载入")


def test_auc_and_dp_legend_orders_are_independent(app: QApplication, tmp_path: Path) -> None:
    page = PlotPage()
    page.set_project(_project(tmp_path))
    page.set_evaluation_run(_run())
    second = CurveSeries(
        "second",
        "Second",
        0.9,
        0.1,
        np.linspace(0.0, 0.8, 51),
        np.linspace(0.8, 0.0, 51),
        TrackerStyle("#D95319", "-", 2.0),
    )
    page._series.append(second)
    page._recipe = replace(
        page._recipe,
        trackers=page._recipe.trackers
        + (page._embedded_entry(second, visible=True, order=1),),
    )
    page.auc_sort_combo.setCurrentIndex(page.auc_sort_combo.findData("desc"))
    page.dp_sort_combo.setCurrentIndex(page.dp_sort_combo.findData("desc"))
    page._apply_sort()

    assert page._recipe.success_order == ("second", "mht")
    assert page._recipe.precision_order == ("mht", "second")


def test_table_ranking_switches_metric_without_changing_legend_orders(
    app: QApplication, tmp_path: Path
) -> None:
    page = PlotPage()
    page.set_project(_project(tmp_path))
    page.set_evaluation_run(_run())
    second = CurveSeries(
        "second",
        "Second",
        0.9,
        0.1,
        np.linspace(0.0, 0.8, 51),
        np.linspace(0.8, 0.0, 51),
        TrackerStyle("#D95319", "-", 2.0),
    )
    page._series.append(second)
    page._recipe = replace(
        page._recipe,
        trackers=page._recipe.trackers
        + (page._embedded_entry(second, visible=True, order=1),),
    )
    page._apply_sort()
    page._fill_table()
    legend_orders = (page._recipe.precision_order, page._recipe.success_order)

    page.table_rank_combo.setCurrentIndex(page.table_rank_combo.findData("auc"))
    assert page.tracker_table.item(0, 1).text() == "1"
    assert page.tracker_table.item(0, 2).text() == "Second"

    page.table_rank_combo.setCurrentIndex(page.table_rank_combo.findData("dp"))
    assert page.tracker_table.item(0, 1).text() == "1"
    assert page.tracker_table.item(0, 2).text() == "MHT"
    assert (page._recipe.precision_order, page._recipe.success_order) == legend_orders


def test_legend_scale_from_drag_is_uniform_and_clamped() -> None:
    assert legend_scale_from_drag(
        1.0, QSizeF(100.0, 80.0), QPointF(25.0, 20.0)
    ) == pytest.approx(1.25)
    assert legend_scale_from_drag(
        1.0, QSizeF(100.0, 80.0), QPointF(500.0, 500.0)
    ) == pytest.approx(1.8)
    assert legend_scale_from_drag(
        1.0, QSizeF(100.0, 80.0), QPointF(-100.0, -80.0)
    ) == pytest.approx(0.6)


def test_interactive_legend_scale_updates_preview_and_project(
    app: QApplication, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    page = PlotPage()
    page.set_project(project)
    page.set_evaluation_run(_run())
    before = page.preview_canvas.figure.axes[0].get_legend().get_texts()[0].get_fontsize()

    assert isinstance(page.preview_canvas, InteractiveLegendCanvas)
    page.preview_canvas.legend_scale_requested.emit(1.4)

    legends = [axis.get_legend() for axis in page.preview_canvas.figure.axes]
    assert page._recipe.style.legend_scale == pytest.approx(1.4)
    assert project.plot_style.legend_scale == pytest.approx(1.4)
    assert all(legend.get_texts()[0].get_fontsize() > before for legend in legends)


def test_legend_handle_can_be_dragged_directly_on_canvas(
    app: QApplication, tmp_path: Path
) -> None:
    page = PlotPage()
    page.resize(1600, 900)
    page.show()
    page.set_project(_project(tmp_path))
    page.set_evaluation_run(_run())
    app.processEvents()
    canvas = page.preview_canvas
    assert isinstance(canvas, InteractiveLegendCanvas)
    rectangle = canvas.legend_rects()[0]
    handle = rectangle.bottomRight().toPoint() - QPoint(2, 2)

    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=handle)
    QTest.mouseMove(canvas, handle + QPoint(24, 18))
    QTest.mouseRelease(
        canvas, Qt.MouseButton.LeftButton, pos=handle + QPoint(24, 18)
    )
    app.processEvents()

    assert page._recipe.style.legend_scale > 1.0


def test_save_two_figures_from_page_writes_both_outputs(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = PlotPage()
    page.set_project(_project(tmp_path))
    page.set_evaluation_run(_run())
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "Precision_plots_OPE.png"), "PNG (*.png)"),
    )

    page._export()

    assert (tmp_path / "Precision_plots_OPE.png").is_file()
    assert (tmp_path / "Success_plots_OPE.png").is_file()
    assert "两张图已导出" in page.status_label.text()
