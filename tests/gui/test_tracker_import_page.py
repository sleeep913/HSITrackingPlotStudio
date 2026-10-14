import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hsi_plotter.models.project import Project, ProjectSources  # noqa: E402
from hsi_plotter.models.tracker_import import ResultLayout  # noqa: E402
from hsi_plotter.tasks.worker import TaskRunner  # noqa: E402
from hsi_plotter.ui.tracker_import_page import (  # noqa: E402
    TrackerImportPage,
    scan_tracker_imports,
)

pytestmark = pytest.mark.gui


@pytest.fixture
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _write(path: Path, rows: str = "1,2,30,40\n2,3,30,40\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows, encoding="utf-8")


def _build_project(root: Path, *, ambiguous: bool = False) -> Project:
    ground_truth = root / "gt"
    _write(ground_truth / "VIS" / "basketball.txt")
    _write(ground_truth / "NIR" / "worker.txt")
    results_root = root / "results"
    _write(results_root / "MHT" / "vis-basketball.txt")
    _write(results_root / "MHT" / "nir-worker.txt")
    _write(results_root / "SiamHSI" / "VIS" / "basketball.txt")
    _write(results_root / "SiamHSI" / "NIR" / "worker.txt")
    if ambiguous:
        _write(results_root / "Ambiguous" / "vis-basketball.txt")
        _write(results_root / "Ambiguous" / "VIS" / "basketball.txt")
    return Project(
        name="HOT2023",
        sources=ProjectSources(ground_truth=ground_truth, results_root=results_root),
    )


def _column(page: TrackerImportPage, header: str) -> int:
    table = page.tracker_table
    for index in range(table.columnCount()):
        if table.horizontalHeaderItem(index).text() == header:
            return index
    raise AssertionError(f"缺少列：{header}")


def _text(page: TrackerImportPage, row: int, header: str) -> str:
    column = _column(page, header)
    item = page.tracker_table.item(row, column)
    if item is not None:
        return item.text()
    widget = page.tracker_table.cellWidget(row, column)
    return widget.currentText() if widget is not None else ""


def _row_of(page: TrackerImportPage, name: str) -> int:
    column = _column(page, "跟踪器")
    for row in range(page.tracker_table.rowCount()):
        if page.tracker_table.item(row, column).text() == name:
            return row
    raise AssertionError(f"缺少跟踪器行：{name}")


def test_scan_collects_every_supported_layout(app: QApplication, tmp_path: Path) -> None:
    project = _build_project(tmp_path)

    outcome = scan_tracker_imports(project)

    assert len(outcome.discoveries) == 2
    assert len(outcome.ground_truth_files) == 2
    assert outcome.discoveries[0].tracker_name == "MHT"
    assert outcome.discoveries[0].preferred is not None
    assert outcome.discoveries[0].preferred.layout is ResultLayout.PREFIXED_FLAT
    assert outcome.discoveries[1].preferred is not None
    assert outcome.discoveries[1].preferred.layout is ResultLayout.MODALITY_SUBDIR


def test_import_page_shows_coverage_per_tracker(app: QApplication, tmp_path: Path) -> None:
    page = TrackerImportPage()
    page.set_project(_build_project(tmp_path))

    outcome = page.rescan()

    assert outcome is not None
    assert page.tracker_table.rowCount() == 2
    assert _text(page, _row_of(page, "MHT"), "匹配") == "2"
    assert _text(page, _row_of(page, "MHT"), "缺失") == "0"
    assert _text(page, _row_of(page, "MHT"), "结果布局") == "带模态前缀"
    assert _text(page, _row_of(page, "SiamHSI"), "模态覆盖") == "NIR, VIS"
    assert _text(page, _row_of(page, "SiamHSI"), "状态") == "就绪"


def test_import_page_reports_missing_sequences(app: QApplication, tmp_path: Path) -> None:
    project = _build_project(tmp_path)
    (tmp_path / "results" / "MHT" / "nir-worker.txt").unlink()
    page = TrackerImportPage()
    page.set_project(project)

    page.rescan()

    row = _row_of(page, "MHT")
    assert _text(page, row, "匹配") == "1"
    assert _text(page, row, "缺失") == "1"
    assert "缺失" in _text(page, row, "状态")


def test_ambiguous_layout_must_be_confirmed_before_import(
    app: QApplication, tmp_path: Path
) -> None:
    page = TrackerImportPage()
    page.set_project(_build_project(tmp_path, ambiguous=True))

    page.rescan()
    row = _row_of(page, "Ambiguous")

    assert "布局歧义" in _text(page, row, "状态")
    page._rows[row].checkbox.setChecked(True)
    assert page.selected_trackers() == ()

    combo = page._rows[row].combo
    combo.setCurrentIndex(combo.findText("模态子目录"))
    selected = page.selected_trackers()

    assert len(selected) == 1
    assert selected[0].layout is ResultLayout.MODALITY_SUBDIR


def test_commit_emits_selected_trackers_with_display_names(
    app: QApplication, tmp_path: Path
) -> None:
    page = TrackerImportPage()
    project = _build_project(tmp_path)
    page.set_project(project)
    page.rescan()
    received: list[tuple] = []
    page.trackers_committed.connect(lambda trackers: received.append(trackers))
    row = _row_of(page, "MHT")
    page._rows[row].checkbox.setChecked(True)
    page.tracker_table.item(row, _column(page, "显示名称")).setText("MHT*")

    committed = page.commit_selection()

    assert len(received) == 1
    assert len(committed) == 1
    assert committed[0].display_name == "MHT*"
    assert committed[0].tracker_id == "mht"
    assert project.trackers == []


def test_rescan_runs_in_background_with_progress(app: QApplication, tmp_path: Path) -> None:
    runner = TaskRunner()
    page = TrackerImportPage()
    page.set_task_runner(runner)
    page.set_project(_build_project(tmp_path))
    messages: list[str] = []
    page.status_message.connect(messages.append)

    page.rescan()
    deadline = time.monotonic() + 10
    while runner.is_busy and time.monotonic() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)

    assert page.tracker_table.rowCount() == 2
    assert runner.active_task_ids == ()
    assert messages
    assert "2" in messages[-1]


def test_import_page_handles_missing_results_root(app: QApplication, tmp_path: Path) -> None:
    page = TrackerImportPage()
    page.set_project(Project(name="empty", sources=ProjectSources()))

    outcome = page.rescan()

    assert outcome is not None
    assert outcome.discoveries == ()
    assert page.summary_label.text().startswith("未发现可导入的跟踪器")
    assert "RESULTS_ROOT_NOT_CONFIGURED" in {issue.code for issue in page._report.issues}
