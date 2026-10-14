from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hsi_plotter.builtin_datasets import BuiltinDataset, built_in_datasets
from hsi_plotter.io.preflight_exporter import export_report_csv, export_report_json
from hsi_plotter.models.dataset import MODALITY_LABELS
from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.tasks.worker import CancelToken, ProgressReporter, TaskRunner
from hsi_plotter.validation.issues import PreflightReport, Severity, ValidationIssue
from hsi_plotter.validation.preflight import ProjectPreflight, collect_project_preflight

SOURCE_DEFINITIONS = (
    ("ground_truth", "Ground Truth", "逐序列 TXT 目录、合并 CSV 或内置数据集", "tracking"),
    ("results_root", "跟踪结果", "跟踪器 TXT 结果根目录或单个合并 CSV", "tracking"),
    ("image_root", "图像序列", "用于跟踪结果可视化的图像根目录", "directory"),
    ("attribute_file", "属性字典", "用于属性评估的 TXT、CSV 或 JSON 文件", "file"),
    ("export_directory", "默认导出目录", "论文图、指标和报告的默认输出位置", "directory"),
)

_COLUMNS = ("级别", "问题", "跟踪器", "模态", "位置", "建议")
(
    _COL_SEVERITY,
    _COL_MESSAGE,
    _COL_TRACKER,
    _COL_MODALITY,
    _COL_PATH,
    _COL_SUGGESTION,
) = range(len(_COLUMNS))

_SEVERITY_NAMES = {
    Severity.ERROR: "错误",
    Severity.WARNING: "警告",
    Severity.INFO: "信息",
}

_MODALITY_TEXT = {modality.value: label for modality, label in MODALITY_LABELS.items()}


class SourceRow(QFrame):
    choose_requested = Signal()
    clear_requested = Signal()
    builtin_requested = Signal(str)

    def __init__(
        self,
        title: str,
        description: str,
        *,
        tracking_source: bool = False,
        builtin_datasets: tuple[BuiltinDataset, ...] = (),
    ) -> None:
        super().__init__()
        self.setObjectName("sourceCard")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 12, 12)
        layout.setSpacing(12)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 600; color: #111827;")
        text_layout.addWidget(title_label)
        description_label = QLabel(description)
        description_label.setProperty("role", "caption")
        text_layout.addWidget(description_label)
        self.path_label = QLabel("尚未配置")
        self.path_label.setObjectName("sourcePathLabel")
        self.path_label.setProperty("role", "caption")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        text_layout.addWidget(self.path_label)
        layout.addLayout(text_layout, 1)
        self.format_combo: QComboBox | None = None
        if tracking_source:
            self.format_combo = QComboBox()
            self.format_combo.setObjectName("sourceFormatCombo")
            self.format_combo.addItem("TXT 目录", "directory")
            self.format_combo.addItem("CSV 文件", "csv")
            layout.addWidget(self.format_combo)
        self.builtin_combo: QComboBox | None = None
        if builtin_datasets:
            self.builtin_combo = QComboBox()
            self.builtin_combo.setObjectName("builtinGroundTruthCombo")
            self.builtin_combo.addItem("内置数据集…", None)
            for dataset in builtin_datasets:
                label = (
                    f"{dataset.label}（{dataset.sequence_count} 序列 / "
                    f"{dataset.frame_count:,} 帧）"
                )
                self.builtin_combo.addItem(label, dataset.dataset_id)
            layout.addWidget(self.builtin_combo)
            builtin_button = QPushButton("使用内置")
            builtin_button.setObjectName("useBuiltinGroundTruthButton")
            builtin_button.clicked.connect(self._request_builtin)
            layout.addWidget(builtin_button)
        self.clear_button = QPushButton("清除")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(lambda: self.clear_requested.emit())
        layout.addWidget(self.clear_button)
        self.choose_button = QPushButton("导入外部…" if tracking_source else "选择…")
        self.choose_button.clicked.connect(lambda: self.choose_requested.emit())
        layout.addWidget(self.choose_button)

    def _request_builtin(self) -> None:
        if self.builtin_combo is None:
            return
        dataset_id = self.builtin_combo.currentData()
        if dataset_id:
            self.builtin_requested.emit(str(dataset_id))

    def selected_kind(self) -> str | None:
        if self.format_combo is None:
            return None
        return str(self.format_combo.currentData())

    def set_path(self, path: Path | None) -> None:
        self.path_label.setText(str(path) if path else "尚未配置")
        self.path_label.setStyleSheet("color: #334155;" if path else "color: #94A3B8;")
        self.clear_button.setEnabled(path is not None)
        if path is not None and self.format_combo is not None:
            target = "csv" if path.suffix.casefold() == ".csv" else "directory"
            index = self.format_combo.findData(target)
            if index >= 0:
                self.format_combo.setCurrentIndex(index)


class SourceConfigPage(QWidget):
    sources_changed = Signal(object)
    back_requested = Signal()
    tracker_import_requested = Signal()
    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("sourceConfigPage")
        self._sources = ProjectSources()
        self._source_kinds = {key: kind for key, _, _, kind in SOURCE_DEFINITIONS}
        self._builtin_datasets = {item.dataset_id: item for item in built_in_datasets()}
        self.source_rows: dict[str, SourceRow] = {}
        self._project: Project | None = None
        self._task_runner: TaskRunner | None = None
        self._active_task_id: str | None = None
        self._runner_connected = False
        self._issues: list[ValidationIssue] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 24)
        root.setSpacing(14)
        header = QHBoxLayout()
        back_button = QPushButton("← 返回模块")
        back_button.clicked.connect(lambda: self.back_requested.emit())
        header.addWidget(back_button)
        title_layout = QVBoxLayout()
        title = QLabel("数据源配置")
        title.setProperty("role", "heading")
        title_layout.addWidget(title)
        subtitle = QLabel("集中管理项目输入路径，并在计算前发现数据问题。")
        subtitle.setProperty("role", "caption")
        title_layout.addWidget(subtitle)
        header.addLayout(title_layout, 1)
        self.tracker_button = QPushButton("跟踪器与序列映射…")
        self.tracker_button.setObjectName("openTrackerImportButton")
        self.tracker_button.clicked.connect(lambda: self.tracker_import_requested.emit())
        header.addWidget(self.tracker_button)
        self.preflight_button = QPushButton("运行预检")
        self.preflight_button.setProperty("primary", True)
        self.preflight_button.clicked.connect(lambda: self.start_preflight())
        header.addWidget(self.preflight_button)
        root.addLayout(header)

        for key, title_text, description, kind in SOURCE_DEFINITIONS:
            row = SourceRow(
                title_text,
                description,
                tracking_source=kind == "tracking",
                builtin_datasets=(
                    tuple(self._builtin_datasets.values()) if key == "ground_truth" else ()
                ),
            )
            row.choose_requested.connect(lambda selected=key: self._choose_source(selected))
            row.clear_requested.connect(lambda selected=key: self.update_source(selected, None))
            if key == "ground_truth":
                row.builtin_requested.connect(self._use_builtin_ground_truth)
            self.source_rows[key] = row
            root.addWidget(row)

        preflight_card = QFrame()
        preflight_card.setObjectName("preflightCard")
        card_layout = QVBoxLayout(preflight_card)
        card_layout.setContentsMargins(16, 14, 16, 16)
        card_layout.setSpacing(10)
        summary = QHBoxLayout()
        summary_title = QLabel("预检结果")
        summary_title.setStyleSheet("font-weight: 600; color: #111827;")
        summary.addWidget(summary_title)
        self.error_badge = self._make_badge("错误 0", "#FEE2E2", "#B91C1C")
        self.warning_badge = self._make_badge("警告 0", "#FEF3C7", "#92400E")
        self.info_badge = self._make_badge("信息 0", "#DBEAFE", "#1D4ED8")
        summary.addWidget(self.error_badge)
        summary.addWidget(self.warning_badge)
        summary.addWidget(self.info_badge)
        summary.addStretch()
        self.export_csv_button = QPushButton("导出 CSV")
        self.export_csv_button.setObjectName("exportPreflightCsvButton")
        self.export_csv_button.clicked.connect(lambda: self._export_report("csv"))
        summary.addWidget(self.export_csv_button)
        self.export_json_button = QPushButton("导出 JSON")
        self.export_json_button.setObjectName("exportPreflightJsonButton")
        self.export_json_button.clicked.connect(lambda: self._export_report("json"))
        summary.addWidget(self.export_json_button)
        card_layout.addLayout(summary)

        filters = QHBoxLayout()
        filters.setSpacing(8)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("全部级别", None)
        self.filter_combo.addItem("仅错误", Severity.ERROR)
        self.filter_combo.addItem("仅警告", Severity.WARNING)
        self.filter_combo.addItem("仅信息", Severity.INFO)
        self.filter_combo.currentIndexChanged.connect(self._apply_issue_filter)
        filters.addWidget(self.filter_combo)
        self.tracker_combo = QComboBox()
        self.tracker_combo.setObjectName("preflightTrackerFilter")
        self.tracker_combo.currentIndexChanged.connect(self._apply_issue_filter)
        filters.addWidget(self.tracker_combo)
        self.modality_combo = QComboBox()
        self.modality_combo.setObjectName("preflightModalityFilter")
        self.modality_combo.currentIndexChanged.connect(self._apply_issue_filter)
        filters.addWidget(self.modality_combo)
        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("preflightSearchEdit")
        self.search_edit.setPlaceholderText("按序列、路径或关键字筛选")
        self.search_edit.textChanged.connect(self._apply_issue_filter)
        filters.addWidget(self.search_edit, 1)
        card_layout.addLayout(filters)

        self.issue_table = QTableWidget(0, len(_COLUMNS))
        self.issue_table.setObjectName("preflightIssueTable")
        self.issue_table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.issue_table.setAlternatingRowColors(True)
        self.issue_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.issue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.issue_table.verticalHeader().setVisible(False)
        header_view = self.issue_table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(_COL_MESSAGE, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(_COL_PATH, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(_COL_SUGGESTION, QHeaderView.ResizeMode.Stretch)
        card_layout.addWidget(self.issue_table, 1)
        root.addWidget(preflight_card, 1)
        self._reset_filters()

    @staticmethod
    def _make_badge(text: str, background: str, foreground: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            f"background: {background}; color: {foreground}; border-radius: 9px; "
            "padding: 3px 9px; font-weight: 600;"
        )
        return label

    # ------------------------------------------------------------------ state

    def set_project(self, project: Project | None) -> None:
        self._project = project
        self._sources = project.sources if project is not None else ProjectSources()
        for key, row in self.source_rows.items():
            row.set_path(getattr(self._sources, key))
        self.setEnabled(project is not None)
        self._clear_report()

    def set_task_runner(self, runner: TaskRunner | None) -> None:
        self._task_runner = runner
        if runner is None or self._runner_connected:
            return
        runner.taskSucceeded.connect(self._on_task_succeeded)
        runner.taskFailed.connect(self._on_task_failed)
        runner.taskCancelled.connect(self._on_task_cancelled)
        runner.taskFinished.connect(self._on_task_finished)
        self._runner_connected = True

    def update_source(self, key: str, path: Path | None) -> None:
        valid_fields = {item.name for item in fields(ProjectSources)}
        if key not in valid_fields or key == "hsi_files":
            raise ValueError(f"不支持的数据源字段：{key}")
        normalized = Path(path).resolve(strict=False) if path is not None else None
        self._sources = replace(self._sources, **{key: normalized})
        self.source_rows[key].set_path(normalized)
        self.sources_changed.emit(self._sources)
        self._clear_report()

    def _choose_source(self, key: str) -> None:
        current = getattr(self._sources, key)
        start = str(current) if current else ""
        selected_kind = self.source_rows[key].selected_kind()
        if selected_kind == "csv":
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "选择合并 CSV",
                start,
                "CSV 文件 (*.csv);;所有文件 (*)",
            )
        elif self._source_kinds[key] == "file":
            selected, _ = QFileDialog.getOpenFileName(
                self, "选择属性字典", start, "数据文件 (*.txt *.csv *.json);;所有文件 (*)"
            )
        else:
            selected = QFileDialog.getExistingDirectory(self, "选择目录", start)
        if selected:
            self.update_source(key, Path(selected))

    def _use_builtin_ground_truth(self, dataset_id: str) -> None:
        dataset = self._builtin_datasets.get(dataset_id)
        if dataset is None:
            QMessageBox.warning(self, "内置数据集不可用", "未找到所选内置数据集。")
            return
        self.update_source("ground_truth", dataset.csv_path)
        self.status_message.emit(
            f"已使用内置 {dataset.label}：{dataset.sequence_count} 个序列，"
            f"{dataset.frame_count:,} 帧。"
        )

    # -------------------------------------------------------------- preflight

    def run_preflight(self) -> ProjectPreflight:
        """Run and display the complete project preflight synchronously."""
        project = self._project
        if project is None:
            return ProjectPreflight()
        preflight = collect_project_preflight(project)
        self.display_preflight(preflight)
        return preflight

    def start_preflight(self) -> None:
        """Run the preflight in the background when a task runner is available."""
        if self._project is None:
            return
        if self._task_runner is None:
            self.run_preflight()
            return
        self.preflight_button.setEnabled(False)
        self.status_message.emit("正在执行数据预检…")
        self._active_task_id = self._task_runner.submit("数据预检", self._preflight_handler)

    def _preflight_handler(
        self,
        reporter: ProgressReporter,
        token: CancelToken,
    ) -> ProjectPreflight:
        project = self._project
        if project is None:  # pragma: no cover - guarded by start_preflight
            raise RuntimeError("当前没有打开的项目")
        reporter.report(0, 1, "开始预检")
        return collect_project_preflight(
            project,
            on_progress=lambda completed, total, message: reporter.report(
                completed, total, message
            ),
            is_cancelled=token.is_cancelled,
        )

    def display_preflight(self, preflight: ProjectPreflight) -> None:
        report = preflight.report
        self._issues = list(report.issues)
        self.error_badge.setText(f"错误 {report.error_count}")
        self.warning_badge.setText(f"警告 {report.warning_count}")
        self.info_badge.setText(f"信息 {report.info_count}")
        self._reset_filters()
        self._fill_issue_table()
        summary = (
            f"预检完成：错误 {report.error_count}，警告 {report.warning_count}，"
            f"信息 {report.info_count}。"
        )
        self.status_message.emit(summary)

    def _on_task_succeeded(self, task_id: str, result: object) -> None:
        if task_id != self._active_task_id or not isinstance(result, ProjectPreflight):
            return
        self.display_preflight(result)

    def _on_task_failed(self, task_id: str, message: str) -> None:
        if task_id != self._active_task_id:
            return
        self.status_message.emit(f"预检失败：{message}")

    def _on_task_cancelled(self, task_id: str) -> None:
        if task_id != self._active_task_id:
            return
        self.status_message.emit("预检已取消")

    def _on_task_finished(self, task_id: str) -> None:
        if task_id != self._active_task_id:
            return
        self._active_task_id = None
        self.preflight_button.setEnabled(True)

    # ---------------------------------------------------------------- reports

    def _reset_filters(self) -> None:
        trackers = sorted({issue.tracker for issue in self._issues if issue.tracker})
        self.tracker_combo.blockSignals(True)
        self.tracker_combo.clear()
        self.tracker_combo.addItem("全部跟踪器", None)
        for tracker in trackers:
            self.tracker_combo.addItem(tracker, tracker)
        self.tracker_combo.blockSignals(False)
        self.modality_combo.blockSignals(True)
        self.modality_combo.clear()
        self.modality_combo.addItem("全部模态", None)
        for modality, label in MODALITY_LABELS.items():
            self.modality_combo.addItem(label, modality.value)
        self.modality_combo.blockSignals(False)

    def _fill_issue_table(self) -> None:
        self.issue_table.setRowCount(len(self._issues))
        for row, issue in enumerate(self._issues):
            values = (
                _SEVERITY_NAMES[issue.severity],
                issue.message,
                issue.tracker or "—",
                _MODALITY_TEXT.get(issue.modality or "", "—"),
                str(issue.path) if issue.path else "—",
                issue.suggestion or "—",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, issue)
                self.issue_table.setItem(row, column, item)
        self._apply_issue_filter()

    def _apply_issue_filter(self) -> None:
        severity = self.filter_combo.currentData()
        tracker = self.tracker_combo.currentData()
        modality = self.modality_combo.currentData()
        keyword = self.search_edit.text().strip().casefold()
        for row in range(self.issue_table.rowCount()):
            item = self.issue_table.item(row, 0)
            issue: ValidationIssue = item.data(Qt.ItemDataRole.UserRole)
            if severity is not None and issue.severity is not severity:
                self.issue_table.setRowHidden(row, True)
                continue
            if tracker is not None and issue.tracker != tracker:
                self.issue_table.setRowHidden(row, True)
                continue
            if modality is not None and issue.modality != modality:
                self.issue_table.setRowHidden(row, True)
                continue
            if keyword and keyword not in (
                f"{issue.message} {issue.sequence or ''} {issue.path or ''} {issue.code}".casefold()
            ):
                self.issue_table.setRowHidden(row, True)
                continue
            self.issue_table.setRowHidden(row, False)

    def _export_report(self, kind: str) -> None:
        if not self._issues:
            QMessageBox.information(self, "导出预检报告", "当前没有可导出的预检结果。")
            return
        default_name = "preflight_report.csv" if kind == "csv" else "preflight_report.json"
        pattern = "CSV 文件 (*.csv)" if kind == "csv" else "JSON 文件 (*.json)"
        start = (
            str(self._sources.export_directory / default_name)
            if (self._sources.export_directory)
            else default_name
        )
        filename, _ = QFileDialog.getSaveFileName(self, "导出预检报告", start, pattern)
        if not filename:
            return
        report = PreflightReport(issues=list(self._issues))
        destination = Path(filename)
        try:
            target = (
                export_report_csv(report, destination)
                if kind == "csv"
                else export_report_json(report, destination)
            )
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.status_message.emit(f"预检报告已导出：{target}")

    def _clear_report(self) -> None:
        self._issues = []
        self.issue_table.setRowCount(0)
        self.error_badge.setText("错误 0")
        self.warning_badge.setText("警告 0")
        self.info_badge.setText("信息 0")
        self._reset_filters()
