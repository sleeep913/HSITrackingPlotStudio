import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from hsi_plotter import APP_DISPLAY_NAME
from hsi_plotter.app import create_application
from hsi_plotter.models.cache import CacheManifest, CacheRecord
from hsi_plotter.models.project import Project
from hsi_plotter.ui.main_window import MainWindow
from hsi_plotter.ui.tracking_visualization_page import TrackingVisualizationPage


@pytest.mark.gui
def test_application_metadata_is_initialized() -> None:
    app = create_application(["hsi-tracking-plotter"])

    assert app.applicationDisplayName() == APP_DISPLAY_NAME
    assert app.applicationName() == "HSI Tracking Plot Studio"


@pytest.mark.gui
def test_main_window_can_be_constructed_and_closed() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.windowTitle() == APP_DISPLAY_NAME
    assert window.objectName() == "mainWindow"
    assert window.minimumWidth() >= 960
    assert window.minimumHeight() >= 640

    window.close()
    app.processEvents()


@pytest.mark.gui
def test_tracking_visualization_tab_is_backed_by_real_page() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert isinstance(window.module_pages["跟踪可视化"], TrackingVisualizationPage)
    assert window.module_tabs.widget(2) is window.tracking_visualization_page

    window.close()
    app.processEvents()


@pytest.mark.gui
def test_clear_project_cache_keeps_project_but_removes_cached_results() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    record = CacheRecord(
        module="evaluation",
        cache_schema_version=1,
        algorithm_version="evaluation-v1",
        created_at="2026-10-02T00:00:00+08:00",
        input_fingerprint="fingerprint",
        payload={"trackers": []},
    )
    project = Project(
        name="cached",
        cache_manifest=CacheManifest(records={record.module: record}),
    )
    window.current_project = project

    window.clear_project_cache()

    assert window.current_project is project
    assert project.cache_manifest.records == {}
    assert window.is_project_modified
    window.set_project_modified(False)
    window.close()
    app.processEvents()
