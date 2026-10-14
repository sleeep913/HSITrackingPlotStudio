from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hsi_plotter.io.tracker_discovery import discover_trackers
from hsi_plotter.models.dataset import SequenceKey
from hsi_plotter.models.project import Project
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.tracker_import import (
    RESULT_LAYOUT_DESCRIPTIONS,
    LayoutCandidate,
    SequenceMapping,
    TrackerDiscovery,
    layout_label,
)
from hsi_plotter.tasks.worker import CancelToken, ProgressReporter, TaskRunner
from hsi_plotter.validation.issues import PreflightReport, Severity, ValidationIssue
from hsi_plotter.validation.preflight import DEFAULT_DATASET_NAME, discover_ground_truth_files
from hsi_plotter.validation.sequence_mapping import build_sequence_mapping

PLACEHOLDER_STYLE = TrackerStyle(color="#2563EB")

_COLUMNS = (
    "导入",
    "跟踪器",
    "显示名称",
    "结果布局",
    "结果文件",
    "匹配",
    "缺失",
    "额外",
    "模态覆盖",
    "状态",
)
(
    _COL_CHECK,
    _COL_NAME,
    _COL_DISPLAY,
    _COL_LAYOUT,
    _COL_FILES,
    _COL_MATCHED,
    _COL_MISSING,
    _COL_EXTRA,
    _COL_MODALITY,
    _COL_STATUS,
) = range(len(_COLUMNS))

_SEVERITY_NAMES = {
    Severity.ERROR: "错误",
    Severity.WARNING: "警告",
    Severity.INFO: "信息",
}


@dataclass(slots=True)
class ScanOutcome:
    """Pure result of scanning tracker directories, safe for unit tests."""

    ground_truth_files: dict[SequenceKey, Path] = field(default_factory=dict)
    discoveries: tuple[TrackerDiscovery, ...] = ()
    report: PreflightReport = field(default_factory=PreflightReport)


def _missing_results_issue(results_root: Path | None) -> ValidationIssue:
    return ValidationIssue(
        severity=Severity.ERROR,
        code="RESULTS_ROOT_NOT_CONFIGURED",
        message="跟踪结果 TXT 目录或 CSV 文件尚未配置或不存在。",
        path=results_root,
        suggestion="先在数据源配置中选择 TXT 结果目录或 CSV 文件。",
    )


def scan_tracker_imports(project: Project) -> ScanOutcome:
    """Collect everything the import table needs from disk."""
    ground_truth_files, report = discover_ground_truth_files(project)
    results_root = project.sources.results_root
    discoveries: tuple[TrackerDiscovery, ...] = ()
    if results_root is not None and (
        results_root.is_dir()
        or (results_root.is_file() and results_root.suffix.casefold() == ".csv")
    ):
        discoveries, discovery_report = discover_trackers(
            results_root, ground_truth_files, dataset=DEFAULT_DATASET_NAME
        )
        report.extend(discovery_report.issues)
    else:
        report.add(_missing_results_issue(results_root))
    return ScanOutcome(
        ground_truth_files=ground_truth_files,
        discoveries=discoveries,
        report=report,
    )


@dataclass(slots=True)
class _RowWidgets:
    checkbox: QCheckBox
    combo: QComboBox
    status_item: QTableWidgetItem
    files_item: QTableWidgetItem
    matched_item: QTableWidgetItem
    missing_item: QTableWidgetItem
    extra_item: QTableWidgetItem
    modality_item: QTableWidgetItem


class TrackerImportPage(QWidget):
    """Discovers tracker result directories and lets the user confirm mappings."""

    trackers_committed = Signal(object)
    back_requested = Signal()
    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("trackerImportPage")
        self._project: Project | None = None
        self._discoveries: tuple[TrackerDiscovery, ...] = ()
        self._ground_truth_files: dict[SequenceKey, Path] = {}
        self._mappings: dict[str, SequenceMapping] = {}
        self._rows: list[_RowWidgets] = []
        self._report = PreflightReport()
        self._task_runner: TaskRunner | None = None
        self._active_task_id: str | None = None
        self._runner_connected = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 24)
        root.setSpacing(14)

        header = QHBoxLayout()
        back_button = QPushButton("← 返回模块")
        back_button.setObjectName("backToModulesButton")
        back_button.clicked.connect(lambda: self.back_requested.emit())
        header.addWidget(back_button)
        title_layout = QVBoxLayout()
        title_layout.setSpacing(3)
        title = QLabel("跟踪器导入与序列映射")
        title.setProperty("role", "heading")
        title_layout.addWidget(title)
        subtitle = QLabel(
            "扫描 TXT 结果目录或合并 CSV，确认 Ground Truth 与结果的序列对应关系后再导入项目。"
        )
        subtitle.setProperty("role", "caption")
        title_layout.addWidget(subtitle)
        header.addLayout(title_layout, 1)
        self.scan_button = QPushButton("重新扫描")
        self.scan_button.setObjectName("rescanTrackersButton")
        self.scan_button.clicked.connect(self.rescan)
        header.addWidget(self.scan_button)
        self.import_button = QPushButton("导入到项目")
        self.import_button.setObjectName("importTrackersButton")
        self.import_button.setProperty("primary", True)
        self.import_button.clicked.connect(self.commit_selection)
        header.addWidget(self.import_button)
        root.addLayout(header)

        card = QFrame()
        card.setObjectName("importCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 16)
        card_layout.setSpacing(10)

        summary_row = QHBoxLayout()
        self.summary_label = QLabel("尚未扫描")
        self.summary_label.setObjectName("importSummaryLabel")
        summary_row.addWidget(self.summary_label)
        summary_row.addStretch()
        self.error_badge = self._make_badge("错误 0", "#FEE2E2", "#B91C1C")
        self.warning_badge = self._make_badge("警告 0", "#FEF3C7", "#92400E")
        summary_row.addWidget(self.error_badge)
        summary_row.addWidget(self.warning_badge)
        card_layout.addLayout(summary_row)

        self.tracker_table = QTableWidget(0, len(_COLUMNS))
        self.tracker_table.setObjectName("trackerImportTable")
        self.tracker_table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.tracker_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tracker_table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.EditKeyPressed
        )
        self.tracker_table.verticalHeader().setVisible(False)
        self.tracker_table.setAlternatingRowColors(True)
        header_view = self.tracker_table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(_COL_NAME, QHeaderView.ResizeMode.Interactive)
        header_view.setSectionResizeMode(_COL_STATUS, QHeaderView.ResizeMode.Stretch)
        card_layout.addWidget(self.tracker_table, 1)

        issues_header = QHBoxLayout()
        issues_title = QLabel("扫描问题")
        issues_title.setStyleSheet("font-weight: 600; color: #111827;")
        issues_header.addWidget(issues_title)
        issues_header.addStretch()
        self.filter_combo = QComboBox()
        self.filter_combo.setObjectName("importIssueFilterCombo")
        self.filter_combo.addItem("全部问题", None)
        self.filter_combo.addItem("仅错误", Severity.ERROR)
        self.filter_combo.addItem("仅警告", Severity.WARNING)
        self.filter_combo.addItem("仅信息", Severity.INFO)
        self.filter_combo.currentIndexChanged.connect(self._apply_issue_filter)
        issues_header.addWidget(self.filter_combo)
        card_layout.addLayout(issues_header)

        self.issue_table = QTableWidget(0, 3)
        self.issue_table.setObjectName("importIssueTable")
        self.issue_table.setHorizontalHeaderLabels(["级别", "问题", "位置"])
        self.issue_table.setAlternatingRowColors(True)
        self.issue_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.issue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.issue_table.verticalHeader().setVisible(False)
        issue_header = self.issue_table.horizontalHeader()
        issue_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        issue_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        issue_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        card_layout.addWidget(self.issue_table, 1)
        root.addWidget(card, 1)

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

    def set_task_runner(self, runner: TaskRunner | None) -> None:
        self._task_runner = runner
        if runner is None or self._runner_connected:
            return
        runner.taskSucceeded.connect(self._on_task_succeeded)
        runner.taskFailed.connect(self._on_task_failed)
        runner.taskCancelled.connect(self._on_task_cancelled)
        runner.taskFinished.connect(self._on_task_finished)
        self._runner_connected = True

    def _on_task_succeeded(self, task_id: str, result: object) -> None:
        if task_id != self._active_task_id or not isinstance(result, ScanOutcome):
            return
        self._apply_outcome(result)

    def _on_task_failed(self, task_id: str, message: str) -> None:
        if task_id != self._active_task_id:
            return
        self.scan_button.setEnabled(True)
        self.import_button.setEnabled(True)
        self.summary_label.setText("扫描失败")
        self.status_message.emit(f"扫描跟踪器失败：{message}")

    def _on_task_cancelled(self, task_id: str) -> None:
        if task_id != self._active_task_id:
            return
        self.scan_button.setEnabled(True)
        self.import_button.setEnabled(True)
        self.summary_label.setText("扫描已取消")
        self.status_message.emit("已取消跟踪器扫描")

    def _on_task_finished(self, task_id: str) -> None:
        if task_id != self._active_task_id:
            return
        self._active_task_id = None

    def set_project(self, project: Project | None) -> None:
        self._project = project
        self.setEnabled(project is not None)
        if project is None:
            self._discoveries = ()
            self._ground_truth_files = {}
            self._mappings = {}
            self.tracker_table.setRowCount(0)
            self.issue_table.setRowCount(0)
            self.summary_label.setText("尚未扫描")
            self.error_badge.setText("错误 0")
            self.warning_badge.setText("警告 0")
            return

    def rescan(self) -> ScanOutcome | None:
        """Scan with the shared task runner when available, else synchronously."""
        if self._project is None:
            return None
        if self._task_runner is None:
            return self._apply_outcome(scan_tracker_imports(self._project))

        self.scan_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.summary_label.setText("正在扫描跟踪器结果…")
        self._active_task_id = self._task_runner.submit("扫描跟踪器", self._scan_handler)
        return None

    def _scan_handler(self, reporter: ProgressReporter, token: CancelToken) -> ScanOutcome:
        reporter.report(0, 1, "扫描 Ground Truth 与跟踪器结果")
        project = self._project
        if project is None:  # pragma: no cover - guarded by rescan()
            raise RuntimeError("当前没有打开的项目")
        outcome = scan_tracker_imports(project)
        token.raise_if_cancelled()
        return outcome

    def apply_outcome(self, outcome: ScanOutcome) -> ScanOutcome:
        """Render a scanned outcome. Public so unit tests can drive the page."""
        return self._apply_outcome(outcome)

    def _apply_outcome(self, outcome: ScanOutcome) -> ScanOutcome:
        self.scan_button.setEnabled(True)
        self.import_button.setEnabled(True)
        self._ground_truth_files = outcome.ground_truth_files
        self._discoveries = outcome.discoveries
        self._report = outcome.report
        self._mappings = {}
        self._rows = []
        self.tracker_table.setRowCount(0)
        existing = {
            tracker.tracker_id: tracker
            for tracker in (self._project.trackers if self._project else [])
        }
        for discovery in outcome.discoveries:
            self._append_row(discovery, existing.get(discovery.tracker_id))
        self._display_scan_report(outcome)
        return outcome

    def _display_scan_report(self, outcome: ScanOutcome) -> None:
        report = outcome.report
        for discovery in outcome.discoveries:
            report.extend(discovery.report.issues)
        self.error_badge.setText(f"错误 {report.error_count}")
        self.warning_badge.setText(f"警告 {report.warning_count}")
        self.issue_table.setRowCount(len(report.issues))
        for row, issue in enumerate(report.issues):
            values = (
                _SEVERITY_NAMES[issue.severity],
                issue.message,
                str(issue.path) if issue.path else "—",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, issue.severity)
                self.issue_table.setItem(row, column, item)
        self._apply_issue_filter()
        matched_total = sum(mapping.matched_count for mapping in self._mappings.values())
        summary = (
            f"Ground Truth {len(self._ground_truth_files)} 个序列，"
            f"发现 {len(outcome.discoveries)} 个跟踪器，"
            f"共可匹配 {matched_total} 个序列对。"
        )
        if not outcome.discoveries:
            summary = "未发现可导入的跟踪器，请检查 TXT 结果目录、CSV 文件或布局。"
            self.summary_label.setProperty("role", "status-warning")
        elif report.has_errors:
            self.summary_label.setProperty("role", "status-error")
        else:
            self.summary_label.setProperty("role", "status-ok")
        self.summary_label.setText(summary)
        self.summary_label.style().polish(self.summary_label)
        self.status_message.emit(summary)

    def _apply_issue_filter(self) -> None:
        selected = self.filter_combo.currentData()
        for row in range(self.issue_table.rowCount()):
            item = self.issue_table.item(row, 0)
            self.issue_table.setRowHidden(
                row, selected is not None and item.data(Qt.ItemDataRole.UserRole) != selected
            )

    # ------------------------------------------------------------------- rows

    def _append_row(self, discovery: TrackerDiscovery, existing: Tracker | None) -> None:
        row = self.tracker_table.rowCount()
        self.tracker_table.insertRow(row)

        checkbox = QCheckBox()
        checkbox.setChecked(existing is not None)
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(checkbox)
        self.tracker_table.setCellWidget(row, _COL_CHECK, container)

        name_item = QTableWidgetItem(discovery.tracker_name)
        name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.tracker_table.setItem(row, _COL_NAME, name_item)

        display_item = QTableWidgetItem(
            existing.display_name if existing is not None else discovery.tracker_name
        )
        self.tracker_table.setItem(row, _COL_DISPLAY, display_item)

        combo = QComboBox()
        combo.setObjectName(f"layoutCombo_{row}")
        for index, candidate in enumerate(discovery.candidates):
            combo.addItem(layout_label(candidate.layout), candidate)
            tooltip = RESULT_LAYOUT_DESCRIPTIONS[candidate.layout]
            combo.setItemData(index, tooltip, Qt.ItemDataRole.ToolTipRole)
        if discovery.preferred is not None:
            combo.setCurrentIndex(discovery.candidates.index(discovery.preferred))
        if discovery.is_ambiguous:
            combo.insertItem(0, "请选择…", None)
            combo.setCurrentIndex(0)
        combo.currentIndexChanged.connect(lambda _index, r=row: self._refresh_row(r))
        self.tracker_table.setCellWidget(row, _COL_LAYOUT, combo)

        files_item = self._read_only_item("")
        self.tracker_table.setItem(row, _COL_FILES, files_item)
        matched_item = self._read_only_item("")
        self.tracker_table.setItem(row, _COL_MATCHED, matched_item)
        missing_item = self._read_only_item("")
        self.tracker_table.setItem(row, _COL_MISSING, missing_item)
        extra_item = self._read_only_item("")
        self.tracker_table.setItem(row, _COL_EXTRA, extra_item)
        modality_item = self._read_only_item("")
        self.tracker_table.setItem(row, _COL_MODALITY, modality_item)
        status_item = self._read_only_item("")
        self.tracker_table.setItem(row, _COL_STATUS, status_item)

        self._rows.append(
            _RowWidgets(
                checkbox=checkbox,
                combo=combo,
                status_item=status_item,
                files_item=files_item,
                matched_item=matched_item,
                missing_item=missing_item,
                extra_item=extra_item,
                modality_item=modality_item,
            )
        )
        self._refresh_row(row)

    @staticmethod
    def _read_only_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        return item

    def _candidate_at(self, row: int) -> LayoutCandidate | None:
        combo = self._rows[row].combo
        data = combo.currentData()
        return data if isinstance(data, LayoutCandidate) else None

    def _refresh_row(self, row: int) -> None:
        discovery = self._discoveries[row]
        widgets = self._rows[row]
        candidate = self._candidate_at(row)
        tracker_name = self.tracker_table.item(row, _COL_DISPLAY).text().strip() or (
            discovery.tracker_name
        )
        if candidate is None:
            preferred = discovery.preferred
            widgets.files_item.setText(str(preferred.file_count) if preferred else "0")
            for item in (widgets.matched_item, widgets.missing_item, widgets.extra_item):
                item.setText("—")
            widgets.modality_item.setText("—")
            widgets.status_item.setText("布局歧义，请手动选择")
            return
        mapping = build_sequence_mapping(self._ground_truth_files, candidate, tracker=tracker_name)
        self._mappings[discovery.tracker_id] = mapping
        widgets.files_item.setText(str(candidate.file_count))
        widgets.matched_item.setText(str(mapping.matched_count))
        widgets.missing_item.setText(str(mapping.missing_count))
        widgets.extra_item.setText(str(mapping.extra_count))
        widgets.modality_item.setText(mapping.modality_label)
        widgets.status_item.setText(self._status_text(discovery, mapping))

    @staticmethod
    def _status_text(discovery: TrackerDiscovery, mapping: SequenceMapping) -> str:
        if discovery.report.has_errors:
            return "存在错误，请检查扫描问题"
        if mapping.ambiguous_count:
            return f"{mapping.ambiguous_count} 个序列存在同名歧义"
        if mapping.matched_count == 0:
            return "没有任何匹配序列"
        if mapping.missing_count or mapping.extra_count:
            return "可导入，存在缺失或额外序列"
        return "就绪"

    # ----------------------------------------------------------------- commit

    def selected_trackers(self) -> tuple[Tracker, ...]:
        if self._project is None:
            return ()
        trackers: list[Tracker] = []
        for row, discovery in enumerate(self._discoveries):
            if not self._rows[row].checkbox.isChecked():
                continue
            candidate = self._candidate_at(row)
            if candidate is None:
                continue
            display_name = self.tracker_table.item(row, _COL_DISPLAY).text().strip()
            trackers.append(
                Tracker(
                    tracker_id=discovery.tracker_id,
                    display_name=display_name or discovery.tracker_name,
                    results_path=discovery.root_path,
                    layout=candidate.layout,
                    style=PLACEHOLDER_STYLE,
                )
            )
        return tuple(trackers)

    def commit_selection(self) -> tuple[Tracker, ...]:
        selected = self.selected_trackers()
        if not selected:
            self.status_message.emit("请先勾选至少一个跟踪器并为歧义布局选择结果布局")
            return ()
        self.trackers_committed.emit(selected)
        self.status_message.emit(f"已导入 {len(selected)} 个跟踪器")
        return selected
