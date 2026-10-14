"""Indicator evaluation page (module one).

The page only collects input and renders results: planning, computation and
export live in ``metrics.evaluation`` / ``io.result_writer``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hsi_plotter.cache.evaluation_codec import (
    decode_evaluation_run,
    encode_evaluation_run,
)
from hsi_plotter.cache.project_fingerprint import (
    EVALUATION_ALGORITHM_VERSION,
    evaluation_fingerprint,
)
from hsi_plotter.cache.registry import CacheRegistry
from hsi_plotter.io.result_writer import (
    ResultExportError,
    default_mat_filename,
    export_main_table_csv,
    export_results_csv,
    export_results_json,
    export_tracker_mat,
    export_trackers_mat,
    next_available_path,
)
from hsi_plotter.metrics.evaluation import (
    prepare_evaluation_inputs,
    run_evaluation,
)
from hsi_plotter.models.cache import CacheRecord
from hsi_plotter.models.evaluation import (
    EVALUATION_MODALITIES,
    EvaluationPlan,
    EvaluationRun,
    EvaluationStatus,
    TrackerEvaluation,
)
from hsi_plotter.models.project import Project
from hsi_plotter.tasks.worker import CancelToken, ProgressReporter, TaskRunner
from hsi_plotter.validation.issues import Severity

_OVERVIEW_COLUMNS = (
    "Method",
    "VIS AUC",
    "VIS DP@20",
    "RedNIR AUC",
    "RedNIR DP@20",
    "NIR AUC",
    "NIR DP@20",
    "Average AUC",
    "Average DP@20",
)

_SEQUENCE_COLUMNS = (
    "跟踪器",
    "序列",
    "模态",
    "AUC",
    "DP",
    "CLE",
    "总帧数",
    "有效帧数",
    "无效帧数",
    "状态",
    "说明",
)

_FAILURE_COLUMNS = (
    "跟踪器",
    "序列",
    "模态",
    "状态",
    "无效帧数",
    "无效帧号",
    "说明",
)

_MAX_INVALID_FRAMES_SHOWN = 10
_DECIMALS = 4


def format_metric(value: float | None) -> str:
    """Default interface precision is four decimals (ALG-EVAL-011)."""
    if value is None:
        return "—"
    return f"{value:.{_DECIMALS}f}"


@dataclass(slots=True)
class _TrackerRow:
    tracker_id: str
    display_name: str
    layout_label: str
    checked: bool = False


class EvaluationPage(QWidget):
    status_message = Signal(str)
    evaluation_finished = Signal(object)
    project_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("evaluationPage")
        self._project: Project | None = None
        self._task_runner: TaskRunner | None = None
        self._runner_connected = False
        self._active_task_id: str | None = None
        self.blocked_message: str = ""
        self._plan: EvaluationPlan | None = None
        self._run: EvaluationRun | None = None
        self._checked_trackers: set[str] = set()
        self._current_tracker_id: str | None = None
        self._rows: list[_TrackerRow] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 24)
        root.setSpacing(14)

        header = QHBoxLayout()
        title_layout = QVBoxLayout()
        title = QLabel("指标评估")
        title.setProperty("role", "heading")
        title_layout.addWidget(title)
        subtitle = QLabel("按序列、模态和总体等权计算 AUC、DP 与 CLE，并检查失败序列。")
        subtitle.setProperty("role", "caption")
        title_layout.addWidget(subtitle)
        header.addLayout(title_layout, 1)
        self.cancel_button = QPushButton("取消计算")
        self.cancel_button.setObjectName("cancelEvaluationButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_evaluation)
        self.run_button = QPushButton("开始计算")
        self.run_button.setObjectName("runEvaluationButton")
        self.run_button.setProperty("primary", True)
        self.run_button.clicked.connect(self.start_evaluation)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setObjectName("evaluationSplitter")
        self.settings_card = QFrame()
        self.settings_card.setObjectName("evaluationSettingsCard")
        self.settings_card.setMinimumHeight(250)
        settings_layout = QHBoxLayout(self.settings_card)
        settings_layout.setContentsMargins(16, 14, 16, 16)
        settings_layout.setSpacing(24)

        self.tracker_panel = QWidget()
        self.tracker_panel.setObjectName("evaluationTrackerPanel")
        tracker_box = QVBoxLayout(self.tracker_panel)
        tracker_box.setContentsMargins(0, 0, 0, 0)
        tracker_label = QLabel("跟踪器")
        tracker_label.setStyleSheet("font-weight: 600; color: #111827;")
        tracker_box.addWidget(tracker_label)
        self.tracker_list = QListWidget()
        self.tracker_list.setObjectName("evaluationTrackerList")
        self.tracker_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tracker_list.itemChanged.connect(self._on_tracker_item_changed)
        self.tracker_list.currentRowChanged.connect(self._on_current_tracker_changed)
        tracker_box.addWidget(self.tracker_list, 1)
        settings_layout.addWidget(self.tracker_panel, 11)

        self.summary_label = QLabel("尚未选择跟踪器。")
        self.summary_label.setObjectName("evaluationSummaryLabel")
        self.summary_label.setProperty("role", "caption")
        self.summary_label.setWordWrap(True)
        self.action_panel = QFrame()
        self.action_panel.setObjectName("evaluationActionPanel")
        action_layout = QVBoxLayout(self.action_panel)
        action_layout.setContentsMargins(18, 18, 18, 18)
        action_layout.setSpacing(12)
        action_title = QLabel("计算设置")
        action_title.setStyleSheet("font-weight: 600; color: #111827;")
        action_layout.addWidget(action_title)
        action_layout.addWidget(self.summary_label)
        action_layout.addStretch(1)
        selection_buttons = QHBoxLayout()
        self.select_all_button = QPushButton("全选")
        self.select_all_button.setObjectName("selectAllTrackersButton")
        self.select_all_button.clicked.connect(lambda: self._set_all_checked(True))
        selection_buttons.addWidget(self.select_all_button)
        self.clear_all_button = QPushButton("清空")
        self.clear_all_button.setObjectName("clearTrackersButton")
        self.clear_all_button.clicked.connect(lambda: self._set_all_checked(False))
        selection_buttons.addWidget(self.clear_all_button)
        action_layout.addLayout(selection_buttons)
        run_buttons = QHBoxLayout()
        run_buttons.addWidget(self.cancel_button)
        run_buttons.addWidget(self.run_button)
        action_layout.addLayout(run_buttons)
        settings_layout.addWidget(self.action_panel, 9)
        splitter.addWidget(self.settings_card)

        result_card = QFrame()
        result_card.setObjectName("evaluationResultCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(16, 14, 16, 16)
        result_layout.setSpacing(10)
        result_header = QHBoxLayout()
        self.result_label = QLabel("尚未计算")
        self.result_label.setObjectName("evaluationResultLabel")
        self.result_label.setStyleSheet("font-weight: 600; color: #111827;")
        result_header.addWidget(self.result_label, 1)
        self.export_mat_button = QPushButton("导出 MAT")
        self.export_mat_button.setObjectName("exportMatButton")
        self.export_mat_button.clicked.connect(self._export_mat)
        result_header.addWidget(self.export_mat_button)
        self.export_csv_button = QPushButton("导出 CSV")
        self.export_csv_button.setObjectName("exportCsvButton")
        self.export_csv_button.clicked.connect(self._export_csv)
        self.csv_scope_combo = QComboBox()
        self.csv_scope_combo.setObjectName("csvScopeCombo")
        self.csv_scope_combo.addItem("完整结果", "all")
        self.csv_scope_combo.addItem("仅指标主表", "main")
        result_header.addWidget(self.csv_scope_combo)
        result_header.addWidget(self.export_csv_button)
        self.export_json_button = QPushButton("导出 JSON")
        self.export_json_button.setObjectName("exportJsonButton")
        self.export_json_button.clicked.connect(self._export_json)
        result_header.addWidget(self.export_json_button)
        result_layout.addLayout(result_header)

        self.result_tabs = QTabWidget()
        self.result_tabs.setObjectName("evaluationResultTabs")
        self.overview_table = self._make_table(_OVERVIEW_COLUMNS, "evaluationOverviewTable")
        self.sequence_table = self._make_table(_SEQUENCE_COLUMNS, "evaluationSequenceTable")
        self.failure_table = self._make_table(_FAILURE_COLUMNS, "evaluationFailureTable")
        self.result_tabs.addTab(self.overview_table, "指标主表")
        self.result_tabs.addTab(self.sequence_table, "逐序列")
        self.result_tabs.addTab(self.failure_table, "失败与异常帧")
        result_layout.addWidget(self.result_tabs, 1)
        splitter.addWidget(result_card)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([270, 590])
        root.addWidget(splitter, 1)
        self._set_result_enabled(False)

    # ------------------------------------------------------------------ build

    def _make_table(self, columns: Sequence[str], name: str) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setObjectName(name)
        table.setHorizontalHeaderLabels(list(columns))
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        table.setWordWrap(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.itemSelectionChanged.connect(self._on_overview_selection)
        return table

    # ------------------------------------------------------------------ state

    def set_project(self, project: Project | None) -> None:
        changed = project is not self._project
        self._project = project
        self.setEnabled(project is not None)
        self._rebuild_tracker_list()
        self._refresh_parameter_summary()
        if project is None:
            self.clear_results()
        elif changed:
            self._restore_cached_run()

    def set_task_runner(self, runner: TaskRunner | None) -> None:
        self._task_runner = runner
        if runner is None or self._runner_connected:
            return
        runner.taskProgress.connect(self._on_task_progress)
        runner.taskSucceeded.connect(self._on_task_succeeded)
        runner.taskFailed.connect(self._on_task_failed)
        runner.taskCancelled.connect(self._on_task_cancelled)
        runner.taskFinished.connect(self._on_task_finished)
        self._runner_connected = True

    def _rebuild_tracker_list(self) -> None:
        project = self._project
        self.tracker_list.blockSignals(True)
        self.tracker_list.clear()
        self._rows = []
        if project is not None:
            for tracker in project.trackers:
                checked = tracker.tracker_id in self._checked_trackers
                row = _TrackerRow(
                    tracker_id=tracker.tracker_id,
                    display_name=tracker.display_name,
                    layout_label=tracker.layout.value,
                    checked=checked,
                )
                self._rows.append(row)
                item = QListWidgetItem(tracker.display_name)
                item.setData(Qt.ItemDataRole.UserRole, tracker.tracker_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                self.tracker_list.addItem(item)
        self.tracker_list.blockSignals(False)

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for index in range(self.tracker_list.count()):
            self.tracker_list.item(index).setCheckState(state)

    def _on_tracker_item_changed(self, item: QListWidgetItem) -> None:
        identifier = item.data(Qt.ItemDataRole.UserRole)
        checked = item.checkState() is Qt.CheckState.Checked
        for row in self._rows:
            if row.tracker_id == identifier:
                row.checked = checked
                break
        if checked:
            self._checked_trackers.add(identifier)
        else:
            self._checked_trackers.discard(identifier)
        self._refresh_parameter_summary()

    def _on_current_tracker_changed(self, row: int) -> None:
        self._current_tracker_id = None
        if 0 <= row < len(self._rows):
            self._current_tracker_id = self._rows[row].tracker_id
        self._fill_detail_tables()

    def selected_tracker_ids(self) -> tuple[str, ...]:
        return tuple(row.tracker_id for row in self._rows if row.checked)

    @property
    def current_run(self) -> EvaluationRun | None:
        """Latest successful run, consumed by the curve module in phase five."""
        return self._run

    @property
    def current_plan(self) -> EvaluationPlan | None:
        return self._plan

    def _refresh_parameter_summary(self) -> None:
        selected = self.selected_tracker_ids()
        if self._project is None:
            self.summary_label.setText("尚未选择跟踪器。")
            return
        if not selected:
            self.summary_label.setText(
                f"项目共有 {len(self._project.trackers)} 个跟踪器，请勾选至少一个。"
            )
            return
        self.summary_label.setText(
            f"已选 {len(selected)} 个跟踪器；固定计算 VIS、RedNIR、NIR 全部可用模态。"
            "点击“开始计算”执行后台计算。"
        )

    # -------------------------------------------------------------- planning

    def build_plan(self) -> EvaluationPlan:
        project = self._project
        if project is None:
            return EvaluationPlan()
        return prepare_evaluation_inputs(
            project,
            tracker_ids=self.selected_tracker_ids(),
        )

    def run_evaluation(self) -> EvaluationRun | None:
        """Prepare and evaluate synchronously, returning ``None`` when blocked."""
        project = self._project
        if project is None:
            return None
        plan = self.build_plan()
        if not self._plan_is_runnable(plan):
            return None
        run = run_evaluation(plan, is_cancelled=lambda: False)
        self._plan = plan
        self._accept_computed_run(run)
        return run

    def start_evaluation(self) -> None:
        if self._project is None:
            return
        if self._task_runner is None:
            self.run_evaluation()
            return
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.result_label.setText("正在后台计算…")
        self.status_message.emit("正在准备计算数据…")
        self._active_task_id = self._task_runner.submit("指标计算", self._evaluation_handler)

    def cancel_evaluation(self) -> None:
        if self._task_runner is not None and self._active_task_id:
            self._task_runner.cancel(self._active_task_id)
        elif self._task_runner is not None:
            self._task_runner.cancel_all()

    def _evaluation_handler(self, reporter: ProgressReporter, token: CancelToken) -> EvaluationRun:
        plan = self.build_plan()
        if not self._plan_is_runnable(plan):
            raise RuntimeError(self.blocked_message)
        self._plan = plan
        reporter.report(
            0,
            plan.total_units,
            f"{len(plan.tracker_ids)} 个跟踪器 · {plan.total_units} 个序列",
        )
        return run_evaluation(
            plan,
            on_progress=lambda completed, total, message: reporter.notify(
                int(round(completed * 100 / max(total, 1))),
                f"{message}（剩余 {max(total - completed, 0)}）",
            ),
            is_cancelled=lambda: token.is_cancelled,
        )

    def _plan_is_runnable(self, plan: EvaluationPlan) -> bool:
        """Report blocking reasons without blocking the caller."""
        if not plan.has_blocking_errors:
            self.blocked_message = ""
            return True
        self.blocked_message = self._blocked_message(plan)
        self.result_label.setText("未能开始计算")
        self.status_message.emit(f"无法开始计算：{self.blocked_message}")
        return False

    def _blocked_message(self, plan: EvaluationPlan) -> str:
        errors = [issue.message for issue in plan.issues if issue.severity is Severity.ERROR]
        if not errors:
            errors = ["所选跟踪器没有可计算的序列。"]
        return "\n".join(f"· {message}" for message in errors[:8])

    # --------------------------------------------------------------- results

    def display_run(self, run: EvaluationRun) -> None:
        self._run = run
        success = sum(1 for tracker in run.trackers if tracker.overall is not None)
        failed = len(run.trackers) - success
        self.result_label.setText(
            f"已完成 {success} 个跟踪器"
            + (f"，{failed} 个未生成结果" if failed else "")
            + f"，成功序列 {run.succeeded_sequence_count} 个"
            + f"，失败序列 {run.failed_sequence_count} 个"
            + ("（任务已取消）" if run.cancelled else "")
        )
        self._set_result_enabled(bool(run.trackers))
        self._fill_overview_table()
        self.result_tabs.setCurrentIndex(0)
        self._current_tracker_id = run.trackers[0].tracker_id if run.trackers else None
        self._fill_detail_tables()
        message = "指标计算完成。" + ("（任务已取消，仅保留已完成结果）" if run.cancelled else "")
        self.status_message.emit(message)
        self.evaluation_finished.emit(run)

    def _accept_computed_run(self, run: EvaluationRun) -> None:
        self.display_run(run)
        if self._project is None:
            return
        tracker_ids = tuple(tracker.tracker_id for tracker in run.trackers)
        CacheRegistry(self._project.cache_manifest).put(
            CacheRecord(
                module="evaluation",
                cache_schema_version=1,
                algorithm_version=EVALUATION_ALGORITHM_VERSION,
                created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                input_fingerprint=evaluation_fingerprint(self._project, tracker_ids),
                payload=encode_evaluation_run(run),
            )
        )
        self.project_changed.emit()

    def _restore_cached_run(self) -> bool:
        if self._project is None:
            return False
        registry = CacheRegistry(self._project.cache_manifest)
        record = registry.manifest.records.get("evaluation")
        if (
            record is None
            or record.payload is None
            or record.algorithm_version != EVALUATION_ALGORITHM_VERSION
        ):
            return False
        try:
            run = decode_evaluation_run(record.payload)
            tracker_ids = tuple(tracker.tracker_id for tracker in run.trackers)
            if record.input_fingerprint != evaluation_fingerprint(
                self._project, tracker_ids
            ):
                return False
        except (KeyError, TypeError, ValueError):
            registry.clear("evaluation")
            self.status_message.emit("指标缓存已损坏，需要重新计算。")
            return False
        self._checked_trackers = set(tracker_ids)
        self._rebuild_tracker_list()
        self.display_run(run)
        self.status_message.emit("已恢复指标评估缓存。")
        return True

    def clear_results(self) -> None:
        self._run = None
        self.result_label.setText("尚未计算")
        for table in (
            self.overview_table,
            self.sequence_table,
            self.failure_table,
        ):
            table.setRowCount(0)
        self._set_result_enabled(False)

    def _set_result_enabled(self, enabled: bool) -> None:
        for button in (
            self.export_mat_button,
            self.export_csv_button,
            self.export_json_button,
        ):
            button.setEnabled(enabled)

    def _fill_overview_table(self) -> None:
        run = self._run
        if run is None:
            self.overview_table.setRowCount(0)
            return
        self.overview_table.setRowCount(len(run.trackers))
        for row, tracker in enumerate(run.trackers):
            values = [tracker.tracker_name]
            for modality in EVALUATION_MODALITIES:
                aggregate = tracker.modality_metrics(modality)
                values.append(format_metric(aggregate.auc if aggregate else None))
                values.append(format_metric(aggregate.dp if aggregate else None))
            values.append(format_metric(tracker.auc))
            values.append(format_metric(tracker.dp))
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, tracker.tracker_id)
                self.overview_table.setItem(row, column, item)

    def _selected_trackers(self) -> tuple[TrackerEvaluation, ...]:
        run = self._run
        if run is None:
            return ()
        if self._current_tracker_id is None:
            return run.trackers
        return tuple(
            tracker for tracker in run.trackers if tracker.tracker_id == self._current_tracker_id
        )

    def _fill_detail_tables(self) -> None:
        trackers = self._selected_trackers()
        self._fill_sequence_table(trackers)
        self._fill_failure_table(trackers)

    def _fill_sequence_table(self, trackers: Sequence[TrackerEvaluation]) -> None:
        rows: list[list[str]] = []
        for tracker in trackers:
            for sequence in tracker.sequences:
                rows.append(
                    [
                        tracker.tracker_name,
                        sequence.sequence_name,
                        sequence.modality_label,
                        format_metric(sequence.auc),
                        format_metric(sequence.dp),
                        format_metric(sequence.cle),
                        str(sequence.total_frames),
                        str(sequence.valid_frames),
                        str(sequence.invalid_frames),
                        sequence.status_label,
                        sequence.message,
                    ]
                )
        _populate_table(self.sequence_table, rows)

    def _fill_failure_table(self, trackers: Sequence[TrackerEvaluation]) -> None:
        rows: list[list[str]] = []
        for tracker in trackers:
            for sequence in tracker.sequences:
                if (
                    sequence.status is EvaluationStatus.SUCCEEDED
                    and not sequence.invalid_frame_indices
                ):
                    continue
                frames = sequence.invalid_frame_indices
                shown = ", ".join(str(value) for value in frames[:_MAX_INVALID_FRAMES_SHOWN])
                if len(frames) > _MAX_INVALID_FRAMES_SHOWN:
                    shown += ", …"
                rows.append(
                    [
                        tracker.tracker_name,
                        sequence.sequence_name,
                        sequence.modality_label,
                        sequence.status_label,
                        str(sequence.invalid_frames),
                        shown or "—",
                        sequence.message or "该序列包含被排除的完整 NaN 帧。",
                    ]
                )
        _populate_table(self.failure_table, rows)

    def _on_overview_selection(self) -> None:
        selected = self.overview_table.selectedItems()
        if not selected or self._run is None:
            return
        tracker_id = selected[0].data(Qt.ItemDataRole.UserRole)
        if tracker_id == self._current_tracker_id:
            return
        self._current_tracker_id = tracker_id
        self.overview_table.blockSignals(True)
        self.sequence_table.blockSignals(True)
        self.failure_table.blockSignals(True)
        self._fill_detail_tables()
        for table in (
            self.overview_table,
            self.sequence_table,
            self.failure_table,
        ):
            table.blockSignals(False)

    # --------------------------------------------------------------- export

    def _export_mat(self) -> None:
        run = self._run
        if run is None or not run.trackers:
            return
        candidates = [tracker for tracker in run.trackers if tracker.overall is not None]
        if not candidates:
            QMessageBox.information(self, "导出 MAT", "当前没有成功计算的跟踪器结果。")
            return
        start = str(self._default_export_directory())
        if len(candidates) == 1:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "导出 MAT 结果",
                str(self._default_export_directory() / default_mat_filename(candidates[0])),
                "MAT 文件 (*.mat)",
            )
            if not filename:
                return
            destination = self._resolve_conflict(Path(filename))
            if destination is None:
                return
            self._write(lambda: [export_tracker_mat(candidates[0], destination)])
            return
        directory = QFileDialog.getExistingDirectory(self, "选择 MAT 导出目录", start)
        if not directory:
            return
        self._write(lambda: list(export_trackers_mat(candidates, Path(directory))))

    def _export_csv(self) -> None:
        main_only = self.csv_scope_combo.currentData() == "main"
        default_name = "metrics-main-table.csv" if main_only else "results.csv"
        target = self._single_file_target(default_name, "CSV 文件 (*.csv)")
        if target is None:
            return
        run = self._run
        if run is None:  # pragma: no cover - guarded by button state
            return
        writer = export_main_table_csv if main_only else export_results_csv
        self._write(lambda: [writer(run, target)])

    def _export_json(self) -> None:
        target = self._single_file_target("results.json", "JSON 文件 (*.json)")
        if target is None:
            return
        run = self._run
        if run is None:  # pragma: no cover - guarded by button state
            return
        self._write(lambda: [export_results_json(run, target)])

    def _single_file_target(self, default_name: str, pattern: str) -> Path | None:
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出结果", str(self._default_export_directory() / default_name), pattern
        )
        if not filename:
            return None
        return self._resolve_conflict(Path(filename))

    def _default_export_directory(self) -> Path:
        if self._project is not None and self._project.sources.export_directory:
            return Path(self._project.sources.export_directory)
        return Path.home()

    def _resolve_conflict(self, destination: Path) -> Path | None:
        """Ask before overwriting; offer automatic numbering (FR-EVAL-008)."""
        if not destination.exists():
            return destination
        choice = QMessageBox.question(
            self,
            "目标文件已存在",
            f"{destination}\n\n该文件已存在。是否覆盖？选择“否”将自动编号保存。",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No,
        )
        if choice is QMessageBox.StandardButton.Yes:
            return destination
        if choice is QMessageBox.StandardButton.No:
            return next_available_path(destination)
        return None

    def _write(self, action) -> None:
        try:
            written = action()
        except ResultExportError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        if not written:
            return
        paths = ", ".join(str(path) for path in written[:3])
        suffix = "…" if len(written) > 3 else ""
        self.status_message.emit(f"结果已导出 {len(written)} 个文件：{paths}{suffix}")

    # ------------------------------------------------------------ task hooks

    def _on_task_progress(self, task_id: str, value: int, message: str) -> None:
        if task_id != self._active_task_id:
            return
        self.result_label.setText(f"计算中 {value}% · {message}")

    def _on_task_succeeded(self, task_id: str, result: object) -> None:
        if task_id != self._active_task_id or not isinstance(result, EvaluationRun):
            return
        self._accept_computed_run(result)

    def _on_task_failed(self, task_id: str, message: str) -> None:
        if task_id != self._active_task_id:
            return
        self.result_label.setText("未能开始计算，请查看提示。")
        self.status_message.emit(f"指标计算失败：{message}")
        QMessageBox.warning(self, "无法开始计算", message)

    def _on_task_cancelled(self, task_id: str) -> None:
        if task_id != self._active_task_id:
            return
        self.result_label.setText("计算已取消")
        self.status_message.emit("指标计算已取消")

    def _on_task_finished(self, task_id: str) -> None:
        if task_id != self._active_task_id:
            return
        self._active_task_id = None
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)


def _populate_table(table: QTableWidget, rows: Sequence[Sequence[str]]) -> None:
    table.setRowCount(len(rows))
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            table.setItem(row, column, QTableWidgetItem(value))
