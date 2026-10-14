import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QTabWidget

from hsi_plotter.ui.attribute_page import AttributePage
from hsi_plotter.ui.main_window import MainWindow


@pytest.fixture
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def test_main_window_exposes_four_fixed_module_tabs(
    app: QApplication, settings: QSettings
) -> None:
    window = MainWindow(settings=settings)

    tabs = window.findChild(QTabWidget, "moduleTabs")

    assert tabs is not None
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "指标评估",
        "曲线绘制",
        "跟踪可视化",
        "属性评估",
    ]
    assert isinstance(tabs.widget(3), AttributePage)
    window.close()
    app.processEvents()


def test_new_project_updates_window_and_project_summary(
    app: QApplication, settings: QSettings
) -> None:
    window = MainWindow(settings=settings)

    window.create_new_project("HOT 论文")

    assert window.current_project is not None
    assert window.current_project.name == "HOT 论文"
    assert window.is_project_modified
    assert "HOT 论文" in window.windowTitle()
    assert "*" in window.windowTitle()
    assert window.project_panel.project_name_label.text() == "HOT 论文"
    assert window.project_panel.save_status_label.text() == "尚未保存"
    assert window.save_action.isEnabled()

    window.set_project_modified(False)
    window.close()
    app.processEvents()


def test_project_save_and_open_round_trip(
    app: QApplication, settings: QSettings, tmp_path: Path
) -> None:
    path = tmp_path / "paper.hsiproj"
    window = MainWindow(settings=settings)
    window.create_new_project("HOT2023")

    window.save_project_to(path)

    assert path.exists()
    assert not window.is_project_modified
    assert window.current_project_path == path.resolve()
    assert path.resolve() in window.recent_project_paths

    reopened = MainWindow(settings=settings)
    reopened.open_project_from(path)

    assert reopened.current_project is not None
    assert reopened.current_project.name == "HOT2023"
    assert reopened.current_project_path == path.resolve()
    assert reopened.project_panel.project_path_label.toolTip() == str(path.resolve())
    assert reopened.project_panel.save_status_label.text() == "已保存"
    assert not reopened.is_project_modified

    window.close()
    reopened.close()
    app.processEvents()


def test_recent_projects_are_deduplicated_and_limited(
    app: QApplication, settings: QSettings, tmp_path: Path
) -> None:
    window = MainWindow(settings=settings)

    for index in range(10):
        path = tmp_path / f"project-{index}.hsiproj"
        window.create_new_project(f"Project {index}")
        window.save_project_to(path)
    window.open_project_from(tmp_path / "project-7.hsiproj")

    assert len(window.recent_project_paths) == 8
    assert window.recent_project_paths[0].name == "project-7.hsiproj"
    assert len(set(window.recent_project_paths)) == 8

    window.close()
    app.processEvents()
