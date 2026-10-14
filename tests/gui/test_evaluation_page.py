import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hsi_plotter.io.project_store import load_project, save_project  # noqa: E402
from hsi_plotter.io.result_writer import export_results_csv  # noqa: E402
from hsi_plotter.models.dataset import Modality  # noqa: E402
from hsi_plotter.models.project import Project, ProjectSources  # noqa: E402
from hsi_plotter.models.style import default_style_for_index  # noqa: E402
from hsi_plotter.models.tracker import Tracker  # noqa: E402
from hsi_plotter.models.tracker_import import ResultLayout  # noqa: E402
from hsi_plotter.tasks.worker import TaskRunner  # noqa: E402
from hsi_plotter.ui.evaluation_page import EvaluationPage, format_metric  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def project(tmp_path: Path) -> Project:
    ground_truth = tmp_path / "gt"
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    export_directory = tmp_path / "export"
    export_directory.mkdir(parents=True, exist_ok=True)
    return Project(
        name="HOT2023",
        sources=ProjectSources(
            ground_truth=ground_truth,
            results_root=results_root,
            export_directory=export_directory,
        ),
    )


def _write(path: Path, rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows, encoding="utf-8")


def add_tracker(project: Project, name: str = "MHT") -> Tracker:
    tracker = Tracker(
        tracker_id=name.casefold(),
        display_name=name,
        results_path=Path(project.sources.results_root) / name,
        style=default_style_for_index(len(project.trackers)),
        layout=ResultLayout.PREFIXED_FLAT,
    )
    project.trackers.append(tracker)
    return tracker


def add_sequence(
    project: Project,
    tracker: Tracker,
    *,
    modality: Modality,
    name: str,
    predicted_row: str = "0,0,20,20",
    frames: int = 4,
    ground_truth_row: str = "0,0,20,20",
) -> None:
    ground_truth = Path(project.sources.ground_truth)
    _write(
        ground_truth / f"{modality.value}-{name}.txt",
        "".join(f"{ground_truth_row}\n" for _ in range(frames)),
    )
    _write(
        Path(tracker.results_path) / f"{modality.value}-{name}.txt",
        "".join(f"{predicted_row}\n" for _ in range(frames)),
    )


def seed(project: Project) -> Tracker:
    tracker = add_tracker(project)
    add_sequence(project, tracker, modality=Modality.VIS, name="coin")
    add_sequence(
        project,
        tracker,
        modality=Modality.NIR,
        name="worker",
        predicted_row="20,0,20,20",
    )
    return tracker


def _header(table, header: str) -> int:
    for index in range(table.columnCount()):
        if table.horizontalHeaderItem(index).text() == header:
            return index
    raise AssertionError(f"缺少列：{header}")


def _cell(page: EvaluationPage, table_name: str, row: int, header: str) -> str:
    table = getattr(page, table_name)
    item = table.item(row, _header(table, header))
    return item.text() if item is not None else ""


def test_format_metric_uses_four_decimals() -> None:
    assert format_metric(0.123456789) == "0.1235"
    assert format_metric(None) == "—"


def test_page_lists_project_trackers(app: QApplication, project: Project) -> None:
    seed(project)
    page = EvaluationPage()

    page.set_project(project)

    assert page.tracker_list.count() == 1
    assert page.tracker_list.item(0).text() == "MHT"
    assert "请勾选至少一个" in page.summary_label.text()


def test_tracker_selection_uses_left_panel_and_actions_are_on_the_right(
    app: QApplication, project: Project
) -> None:
    seed(project)
    page = EvaluationPage()
    page.set_project(project)

    assert page.tracker_panel.objectName() == "evaluationTrackerPanel"
    assert page.action_panel.objectName() == "evaluationActionPanel"
    assert page.run_button.parentWidget() is page.action_panel
    assert page.cancel_button.parentWidget() is page.action_panel
    assert page.select_all_button.parentWidget() is page.action_panel
    assert page.clear_all_button.parentWidget() is page.action_panel
    assert page.settings_card.minimumHeight() >= 240


def test_select_all_enables_selection(app: QApplication, project: Project) -> None:
    seed(project)
    page = EvaluationPage()
    page.set_project(project)

    page._set_all_checked(True)

    assert page.selected_tracker_ids() == ("mht",)
    assert "已选 1 个跟踪器" in page.summary_label.text()


def test_run_evaluation_fills_every_table(app: QApplication, project: Project) -> None:
    seed(project)
    page = EvaluationPage()
    page.set_project(project)
    page._set_all_checked(True)

    run = page.run_evaluation()

    assert run is not None
    assert page.overview_table.rowCount() == 1
    assert _cell(page, "overview_table", 0, "Method") == "MHT"
    assert _cell(page, "overview_table", 0, "Average AUC") == "0.5000"
    assert _cell(page, "overview_table", 0, "Average DP@20") == "0.5000"
    assert page.overview_table.columnCount() == 9
    assert page.result_tabs.count() == 3
    assert page.sequence_table.rowCount() == 2
    assert "已完成 1 个跟踪器" in page.result_label.text()
    assert page.export_mat_button.isEnabled()
    assert page.export_csv_button.isEnabled()
    assert page.export_json_button.isEnabled()


def test_overview_shows_per_modality_values(app: QApplication, project: Project) -> None:
    seed(project)
    page = EvaluationPage()
    page.set_project(project)
    page._set_all_checked(True)

    page.run_evaluation()

    assert _cell(page, "overview_table", 0, "VIS AUC") == "1.0000"
    assert _cell(page, "overview_table", 0, "VIS DP@20") == "1.0000"
    assert _cell(page, "overview_table", 0, "NIR AUC") == "0.0000"
    assert _cell(page, "overview_table", 0, "NIR DP@20") == "0.0000"
    assert _cell(page, "overview_table", 0, "RedNIR AUC") == "—"


def test_failure_tab_reports_broken_sequence(app: QApplication, project: Project) -> None:
    seed(project)
    tracker = project.trackers[0]
    _write(Path(project.sources.ground_truth) / "vis-short.txt", "0,0,20,20\n0,0,20,20\n")
    _write(Path(tracker.results_path) / "vis-short.txt", "0,0,20,20\n")
    page = EvaluationPage()
    page.set_project(project)
    page._set_all_checked(True)

    page.run_evaluation()

    assert page._run is not None
    assert page._run.trackers[0].succeeded_count == 2
    assert page._run.trackers[0].failed_count == 1
    assert page.failure_table.rowCount() == 1
    assert _cell(page, "failure_table", 0, "序列") == "short"
    assert "帧数不一致" in _cell(page, "failure_table", 0, "说明")


def test_failure_tab_reports_missing_frames(app: QApplication, project: Project) -> None:
    tracker = add_tracker(project)
    add_sequence(project, tracker, modality=Modality.VIS, name="gaps")
    result_file = Path(tracker.results_path) / "vis-gaps.txt"
    result_file.write_text("0,0,20,20\nNaN,NaN,NaN,NaN\n0,0,20,20\n0,0,20,20\n", encoding="utf-8")
    page = EvaluationPage()
    page.set_project(project)
    page._set_all_checked(True)

    page.run_evaluation()

    assert page.failure_table.rowCount() == 1
    assert _cell(page, "failure_table", 0, "无效帧数") == "1"
    assert _cell(page, "failure_table", 0, "无效帧号") == "2"


def test_page_has_no_modality_filter_and_always_computes_all(
    app: QApplication, project: Project
) -> None:
    seed(project)
    page = EvaluationPage()
    page.set_project(project)
    page._set_all_checked(True)
    page.run_evaluation()

    assert not hasattr(page, "modality_list")
    assert page.sequence_table.rowCount() == 2
    assert { _cell(page, "sequence_table", row, "模态") for row in range(2) } == {"VIS", "NIR"}


def test_run_without_selection_is_blocked(app: QApplication, project: Project) -> None:
    seed(project)
    page = EvaluationPage()
    page.set_project(project)
    messages: list[str] = []
    page.status_message.connect(messages.append)

    assert page.run_evaluation() is None
    assert page.blocked_message
    assert any("无法开始计算" in message for message in messages)


def test_export_csv_writes_selected_scope(
    app: QApplication, tmp_path: Path, project: Project
) -> None:
    seed(project)
    page = EvaluationPage()
    page.set_project(project)
    page._set_all_checked(True)
    page.run_evaluation()
    target = tmp_path / "out.csv"

    page._write(lambda: [export_results_csv(page._run, target)])

    assert target.is_file()
    assert "序列" in target.read_text(encoding="utf-8-sig")


def test_csv_scope_selector_offers_main_table_only(app: QApplication) -> None:
    page = EvaluationPage()

    assert page.csv_scope_combo.count() == 2
    assert page.csv_scope_combo.findData("main") >= 0
    assert page.csv_scope_combo.findData("all") >= 0


def test_start_evaluation_runs_in_background(app: QApplication, project: Project) -> None:
    seed(project)
    runner = TaskRunner()
    page = EvaluationPage()
    page.set_task_runner(runner)
    page.set_project(project)
    page._set_all_checked(True)
    messages: list[str] = []
    page.status_message.connect(messages.append)

    page.start_evaluation()
    deadline = time.monotonic() + 10
    while runner.is_busy and time.monotonic() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)

    assert page.overview_table.rowCount() == 1
    assert runner.active_task_ids == ()
    assert any("指标计算完成" in message for message in messages)


def test_cancelling_keeps_completed_results(app: QApplication, project: Project) -> None:
    tracker = seed(project)
    for index in range(6):
        add_sequence(project, tracker, modality=Modality.VIS, name=f"seq{index}")
    runner = TaskRunner()
    page = EvaluationPage()
    page.set_task_runner(runner)
    page.set_project(project)
    page._set_all_checked(True)
    requested = {"sent": False}

    def request_cancel(task_id: str, value: int, message: str) -> None:
        if value is not None and value >= 25 and not requested["sent"]:
            requested["sent"] = True
            page.cancel_evaluation()

    runner.taskProgress.connect(request_cancel)

    page.start_evaluation()
    deadline = time.monotonic() + 10
    while runner.is_busy and time.monotonic() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)

    assert page._run is not None
    assert requested["sent"]
    assert page._run.cancelled
    assert 1 <= page._run.succeeded_sequence_count < 8
    assert "任务已取消" in page.result_label.text()


def test_evaluation_cache_restores_after_project_reopen(
    app: QApplication, project: Project, tmp_path: Path
) -> None:
    seed(project)
    page = EvaluationPage()
    page.set_project(project)
    page._set_all_checked(True)
    page.run_evaluation()
    project_path = tmp_path / "cached.hsiproj"
    save_project(project, project_path)

    restored_page = EvaluationPage()
    restored_page.set_project(load_project(project_path))

    assert restored_page.current_run is not None
    assert restored_page.overview_table.rowCount() == 1
    assert _cell(restored_page, "overview_table", 0, "Average AUC") == "0.5000"
