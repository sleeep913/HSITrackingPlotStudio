import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from hsi_plotter.ui.main_window import MainWindow
from hsi_plotter.ui.source_config_page import SourceConfigPage
from hsi_plotter.ui.theme import apply_application_theme


@pytest.fixture
def app() -> QApplication:
    application = QApplication.instance() or QApplication([])
    apply_application_theme(application)
    return application


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def test_application_theme_uses_restrained_research_palette(app: QApplication) -> None:
    stylesheet = app.styleSheet()

    assert "#F4F6F8" in stylesheet
    assert "#2563EB" in stylesheet
    assert "QPushButton" in stylesheet
    assert "QTableView" in stylesheet


def test_source_editor_updates_project_and_dirty_state(
    app: QApplication, settings: QSettings, tmp_path: Path
) -> None:
    window = MainWindow(settings=settings)
    window.create_new_project("HOT2023")
    window.save_project_to(tmp_path / "paper.hsiproj")
    ground_truth = tmp_path / "ground-truth"
    ground_truth.mkdir()

    window.show_source_configuration()
    window.source_config_page.update_source("ground_truth", ground_truth)

    assert window.current_project is not None
    assert window.current_project.sources.ground_truth == ground_truth.resolve()
    assert window.is_project_modified
    assert window.workspace_stack.currentWidget() is window.source_config_page
    assert window.project_panel.data_status_label.text() == "1 / 4"
    assert window.project_panel.save_status_label.text() == "有未保存更改"

    window.set_project_modified(False)
    window.close()
    app.processEvents()


def test_source_preflight_renders_error_summary(app: QApplication, settings: QSettings) -> None:
    window = MainWindow(settings=settings)
    window.create_new_project("Empty project")
    window.show_source_configuration()

    preflight = window.source_config_page.run_preflight()
    report = preflight.report

    assert report.error_count == 2
    assert window.source_config_page.issue_table.rowCount() == len(report.issues)
    assert "2" in window.source_config_page.error_badge.text()

    window.set_project_modified(False)
    window.close()
    app.processEvents()


def test_source_config_page_exposes_all_supported_sources(app: QApplication) -> None:
    page = SourceConfigPage()

    assert set(page.source_rows) == {
        "ground_truth",
        "results_root",
        "image_root",
        "attribute_file",
        "export_directory",
    }


def test_source_config_page_selects_bundled_ground_truth(app: QApplication) -> None:
    page = SourceConfigPage()
    row = page.source_rows["ground_truth"]

    assert row.format_combo is not None
    assert row.format_combo.count() == 2
    assert row.builtin_combo is not None
    assert row.builtin_combo.count() == 4

    page._use_builtin_ground_truth("HOTC2023")

    assert page._sources.ground_truth is not None
    assert page._sources.ground_truth.name == "groundtruth.csv"
    assert page._sources.ground_truth.parent.name == "HOTC2023"


def test_preflight_report_can_be_filtered_by_tracker_and_keyword(
    app: QApplication, settings: QSettings, tmp_path: Path
) -> None:
    window = MainWindow(settings=settings)
    window.create_new_project("HOT2023")
    ground_truth = tmp_path / "gt"
    results_root = tmp_path / "results"
    (ground_truth / "VIS").mkdir(parents=True)
    (ground_truth / "VIS" / "coin.txt").write_text("1,2,30,40\n", encoding="utf-8")
    (results_root / "MHT" / "VIS").mkdir(parents=True)
    (results_root / "MHT" / "VIS" / "coin.txt").write_text(
        "1,2,30,40\n1,2,30,40\n", encoding="utf-8"
    )
    from dataclasses import replace

    from hsi_plotter.models.style import default_style_for_index
    from hsi_plotter.models.tracker import Tracker
    from hsi_plotter.models.tracker_import import ResultLayout

    window.current_project.sources = replace(
        window.current_project.sources,
        ground_truth=ground_truth,
        results_root=results_root,
    )
    window.current_project.trackers = [
        Tracker(
            tracker_id="mht",
            display_name="MHT",
            results_path=results_root / "MHT",
            layout=ResultLayout.MODALITY_SUBDIR,
            style=default_style_for_index(0),
        )
    ]
    window.source_config_page.set_project(window.current_project)

    preflight = window.source_config_page.run_preflight()

    assert preflight.mappings["mht"].matched_count == 1
    assert preflight.has_errors
    page = window.source_config_page
    assert [issue.code for issue in preflight.report.issues].count("FRAME_COUNT_MISMATCH") == 1

    page.tracker_combo.setCurrentIndex(page.tracker_combo.findData("MHT"))
    visible_tracker_rows = [
        row for row in range(page.issue_table.rowCount()) if not page.issue_table.isRowHidden(row)
    ]
    assert len(visible_tracker_rows) == 1
    assert page.issue_table.item(visible_tracker_rows[0], 2).text() == "MHT"

    page.search_edit.setText("不存在的序列")
    assert all(page.issue_table.isRowHidden(row) for row in range(page.issue_table.rowCount()))

    window.set_project_modified(False)
    window.close()
    app.processEvents()
