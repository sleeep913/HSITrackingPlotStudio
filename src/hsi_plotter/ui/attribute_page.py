"""Streamlined attribute evaluation page with table results and PDF export."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hsi_plotter.cache.attribute_codec import decode_attribute_run, encode_attribute_run
from hsi_plotter.cache.project_fingerprint import (
    ATTRIBUTE_ALGORITHM_VERSION,
    attribute_fingerprint,
)
from hsi_plotter.cache.registry import CacheRegistry
from hsi_plotter.io.attribute_pdf_exporter import export_attribute_radar_pdf
from hsi_plotter.io.attribute_reader import AttributeDataError, read_attribute_definitions
from hsi_plotter.metrics.attributes import run_attribute_evaluation
from hsi_plotter.models.attribute_evaluation import (
    AttributeDefinition,
    AttributeEvaluationRun,
)
from hsi_plotter.models.cache import CacheRecord
from hsi_plotter.models.project import Project
from hsi_plotter.models.style import default_style_for_index
from hsi_plotter.models.tracker import Tracker, assign_stable_styles
from hsi_plotter.plotting.attribute_radar import AttributeRadarSeries
from hsi_plotter.tasks.worker import CancelToken, ProgressReporter, TaskRunner
from hsi_plotter.ui.evaluation_page import format_metric
from hsi_plotter.ui.tracker_import_page import ScanOutcome, scan_tracker_imports

Metric = Literal["auc", "dp"]
RADAR_SELECTION_MODULE = "attribute_radar_selection"
RADAR_SELECTION_VERSION = "attribute-radar-selection-v1"


class AttributePage(QWidget):
    status_message = Signal(str)
    project_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("attributePage")
        self._project: Project | None = None
        self._attributes: tuple[AttributeDefinition, ...] = ()
        self.current_run: AttributeEvaluationRun | None = None
        self._metric: Metric = "auc"
        self._radar_selected_tracker_ids: set[str] = set()
        self._radar_known_tracker_ids: set[str] = set()
        self._radar_selection_initialized = False
        self._task_runner: TaskRunner | None = None
        self._active_task_id: str | None = None
        self._runner_connected = False

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 24)
        root.setSpacing(14)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("属性评估")
        title.setProperty("role", "heading")
        titles.addWidget(title)
        subtitle = QLabel("导入属性字典、Ground Truth 和跟踪结果，计算各属性 AUC 或 DP@20。")
        subtitle.setProperty("role", "caption")
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        self.run_button = QPushButton("计算 AUC / DP")
        self.run_button.setObjectName("attributeRunButton")
        self.run_button.setProperty("primary", True)
        self.run_button.clicked.connect(self.start_evaluation)
        header.addWidget(self.run_button)
        root.addLayout(header)

        import_card = QFrame()
        import_card.setObjectName("attributeImportCard")
        import_layout = QHBoxLayout(import_card)
        import_layout.setContentsMargins(16, 14, 16, 14)
        self.import_attribute_button = QPushButton("导入属性字典…")
        self.import_attribute_button.setObjectName("attributeImportButton")
        self.import_attribute_button.clicked.connect(self._choose_attribute_file)
        self.import_ground_truth_button = QPushButton("导入 Ground Truth…")
        self.import_ground_truth_button.setObjectName("attributeGroundTruthButton")
        ground_truth_menu = QMenu(self.import_ground_truth_button)
        ground_truth_directory_action = QAction("导入 TXT 目录…", ground_truth_menu)
        ground_truth_directory_action.triggered.connect(self._choose_ground_truth)
        ground_truth_csv_action = QAction("导入 CSV 文件…", ground_truth_menu)
        ground_truth_csv_action.triggered.connect(self._choose_ground_truth_csv)
        ground_truth_menu.addActions(
            [ground_truth_directory_action, ground_truth_csv_action]
        )
        self.import_ground_truth_button.setMenu(ground_truth_menu)
        self.import_trackers_button = QPushButton("导入跟踪器…")
        self.import_trackers_button.setObjectName("attributeTrackerButton")
        tracker_menu = QMenu(self.import_trackers_button)
        tracker_directory_action = QAction("导入 TXT 结果目录…", tracker_menu)
        tracker_directory_action.triggered.connect(self._choose_trackers)
        tracker_csv_action = QAction("导入 CSV 结果…", tracker_menu)
        tracker_csv_action.triggered.connect(self._choose_tracker_csv)
        tracker_menu.addActions([tracker_directory_action, tracker_csv_action])
        self.import_trackers_button.setMenu(tracker_menu)
        for button in (
            self.import_attribute_button,
            self.import_ground_truth_button,
            self.import_trackers_button,
        ):
            import_layout.addWidget(button)
        self.source_summary = QLabel("尚未完成数据导入")
        self.source_summary.setObjectName("attributeSourceSummary")
        self.source_summary.setProperty("role", "caption")
        import_layout.addWidget(self.source_summary, 1)
        root.addWidget(import_card)

        self.content_tabs = QTabWidget()
        self.content_tabs.setObjectName("attributeContentTabs")
        attribute_tab = QWidget()
        attribute_tab_layout = QVBoxLayout(attribute_tab)
        attribute_tab_layout.setContentsMargins(0, 8, 0, 0)

        self.attribute_table = QTableWidget(0, 3)
        self.attribute_table.setObjectName("attributeSequenceTable")
        self.attribute_table.setHorizontalHeaderLabels(("属性", "序列数", "序列"))
        attribute_header = self.attribute_table.horizontalHeader()
        attribute_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        attribute_header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        attribute_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.attribute_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.attribute_table.setAlternatingRowColors(True)
        self.attribute_table.setWordWrap(True)
        attribute_tab_layout.addWidget(self.attribute_table)
        self.content_tabs.addTab(attribute_tab, "属性与序列")

        result_tab = QWidget()
        result_tab_layout = QVBoxLayout(result_tab)
        result_tab_layout.setContentsMargins(0, 8, 0, 0)

        results_header = QHBoxLayout()
        results_header.addWidget(QLabel("属性指标结果"))
        results_header.addStretch()
        self.auc_button = QPushButton("AUC")
        self.auc_button.setObjectName("attributeAucButton")
        self.auc_button.setCheckable(True)
        self.auc_button.clicked.connect(self.show_auc)
        self.dp_button = QPushButton("DP@20")
        self.dp_button.setObjectName("attributeDpButton")
        self.dp_button.setCheckable(True)
        self.dp_button.clicked.connect(self.show_dp)
        self.radar_tracker_button = QPushButton("绘图跟踪器 0/0")
        self.radar_tracker_button.setObjectName("attributeRadarTrackerButton")
        self.radar_tracker_menu = QMenu(self.radar_tracker_button)
        self.radar_tracker_button.setMenu(self.radar_tracker_menu)
        self.export_button = QPushButton("导出属性雷达图 PDF…")
        self.export_button.setObjectName("attributeExportPdfButton")
        self.export_button.clicked.connect(self._choose_and_export_radar)
        results_header.addWidget(self.auc_button)
        results_header.addWidget(self.dp_button)
        results_header.addWidget(self.radar_tracker_button)
        results_header.addWidget(self.export_button)
        result_tab_layout.addLayout(results_header)

        self.result_table = QTableWidget(0, 1)
        self.result_table.setObjectName("attributeResultTable")
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.setAlternatingRowColors(True)
        result_tab_layout.addWidget(self.result_table)
        self.content_tabs.addTab(result_tab, "计算结果")
        root.addWidget(self.content_tabs, 1)
        self.status_label = QLabel("请依次导入数据并开始计算。")
        self.status_label.setObjectName("attributeStatusLabel")
        self.status_label.setProperty("role", "caption")
        root.addWidget(self.status_label)
        self._update_metric_buttons()
        self._update_enabled_state()

    def set_project(self, project: Project | None) -> None:
        changed = project is not self._project
        self._project = project
        if changed:
            self.current_run = None
            self._attributes = ()
            self._reset_radar_selection()
            if project is not None and project.sources.attribute_file is not None:
                try:
                    self._attributes = read_attribute_definitions(project.sources.attribute_file)
                except AttributeDataError as exc:
                    self._set_status(str(exc))
            self._render_attributes()
            self._render_results()
            self._restore_cached_run()
        self._update_source_summary()
        self._update_enabled_state()

    def set_task_runner(self, runner: TaskRunner) -> None:
        self._task_runner = runner
        if self._runner_connected:
            return
        runner.taskSucceeded.connect(self._on_task_succeeded)
        runner.taskFailed.connect(self._on_task_failed)
        runner.taskFinished.connect(self._on_task_finished)
        self._runner_connected = True

    def load_attribute_file(self, source: Path) -> None:
        definitions = read_attribute_definitions(source)
        self._attributes = definitions
        if self._project is not None:
            self._project.sources = replace(self._project.sources, attribute_file=Path(source))
            self.project_changed.emit()
        self.current_run = None
        self._reset_radar_selection()
        self._render_attributes()
        self._render_results()
        self._update_source_summary()
        self._update_enabled_state()
        self._set_status(f"已导入 {len(definitions)} 个属性。")

    def set_ground_truth_directory(self, directory: Path) -> None:
        if self._project is None:
            return
        self._project.sources = replace(self._project.sources, ground_truth=Path(directory))
        self.project_changed.emit()
        self._update_source_summary()
        self._update_enabled_state()
        self._set_status(f"Ground Truth：{directory}")

    def import_trackers_from_root(self, directory: Path) -> tuple[Tracker, ...]:
        if self._project is None:
            return ()
        self._project.sources = replace(self._project.sources, results_root=Path(directory))
        outcome = scan_tracker_imports(self._project)
        trackers = self.apply_tracker_scan(outcome)
        self._update_source_summary()
        self._update_enabled_state()
        return trackers

    def apply_tracker_scan(self, outcome: ScanOutcome) -> tuple[Tracker, ...]:
        if self._project is None:
            return ()
        selected: list[Tracker] = []
        for discovery in outcome.discoveries:
            if len(discovery.best_candidates) != 1:
                continue
            candidate = discovery.best_candidates[0]
            selected.append(
                Tracker(
                    tracker_id=discovery.tracker_id,
                    display_name=discovery.tracker_name,
                    results_path=discovery.root_path,
                    style=default_style_for_index(0),
                    layout=candidate.layout,
                )
            )
        styled = assign_stable_styles(self._project.trackers, selected)
        self._project.trackers = list(styled)
        self.current_run = None
        self._reset_radar_selection()
        self.project_changed.emit()
        self._set_status(f"已导入 {len(styled)} 个跟踪器。")
        return styled

    def start_evaluation(self) -> AttributeEvaluationRun | None:
        message = self._validation_message()
        if message:
            self._set_status(message)
            return None
        assert self._project is not None
        if self._task_runner is None:
            run = run_attribute_evaluation(self._project, self._attributes)
            self._accept_computed_run(run)
            return run
        if self._active_task_id is not None:
            return None
        project = self._project
        attributes = self._attributes

        def handler(reporter: ProgressReporter, token: CancelToken) -> AttributeEvaluationRun:
            return run_attribute_evaluation(
                project,
                attributes,
                on_progress=lambda done, total, text: reporter.notify(
                    int(round(done * 100 / max(total, 1))), text
                ),
                is_cancelled=lambda: token.is_cancelled,
            )

        self._active_task_id = self._task_runner.submit("计算属性指标", handler)
        self.run_button.setEnabled(False)
        self._set_status("正在计算属性指标…")
        return None

    def display_run(self, run: AttributeEvaluationRun) -> None:
        self.current_run = run
        self._attributes = run.attributes
        self._sync_radar_selection(run)
        self._render_attributes()
        self._render_results()
        self._update_enabled_state()
        self._set_status(f"计算完成：{len(run.attributes)} 个属性，{len(run.trackers)} 个跟踪器。")

    def clear_results(self) -> None:
        self.current_run = None
        self._reset_radar_selection()
        self._render_results()
        self._update_enabled_state()
        self._set_status("属性缓存已清除，需要重新计算。")

    def _accept_computed_run(self, run: AttributeEvaluationRun) -> None:
        self.display_run(run)
        if self._project is None:
            return
        record = CacheRecord(
            module="attribute_evaluation",
            cache_schema_version=1,
            algorithm_version=ATTRIBUTE_ALGORITHM_VERSION,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            input_fingerprint=attribute_fingerprint(self._project),
            payload=encode_attribute_run(run),
        )
        CacheRegistry(self._project.cache_manifest).put(record)
        self._write_radar_selection_cache()
        self.project_changed.emit()

    def _restore_cached_run(self) -> bool:
        if self._project is None or not self._attributes:
            return False
        record = CacheRegistry(self._project.cache_manifest).get(
            "attribute_evaluation", attribute_fingerprint(self._project)
        )
        if record is None or record.payload is None:
            return False
        try:
            run = decode_attribute_run(record.payload)
        except (KeyError, TypeError, ValueError):
            CacheRegistry(self._project.cache_manifest).clear("attribute_evaluation")
            self._set_status("属性缓存已损坏，需要重新计算。")
            return False
        self.display_run(run)
        self._restore_radar_selection()
        self._set_status("已恢复属性评估缓存。")
        return True

    def _reset_radar_selection(self) -> None:
        self._radar_selected_tracker_ids.clear()
        self._radar_known_tracker_ids.clear()
        self._radar_selection_initialized = False
        if hasattr(self, "radar_tracker_menu"):
            self._rebuild_radar_tracker_menu()

    def _sync_radar_selection(self, run: AttributeEvaluationRun) -> None:
        available = {tracker.tracker_id for tracker in run.trackers}
        if not self._radar_selection_initialized:
            selected = set(available)
        else:
            selected = self._radar_selected_tracker_ids & available
            selected.update(available - self._radar_known_tracker_ids)
        self._radar_selected_tracker_ids = selected
        self._radar_known_tracker_ids = available
        self._radar_selection_initialized = True
        self._rebuild_radar_tracker_menu()

    def _radar_selection_fingerprint(self) -> str:
        if self._project is None:
            return ""
        evaluation_record = self._project.cache_manifest.records.get(
            "attribute_evaluation"
        )
        if evaluation_record is not None:
            return evaluation_record.input_fingerprint
        return attribute_fingerprint(self._project)

    def _write_radar_selection_cache(self) -> None:
        if self._project is None or self.current_run is None:
            return
        ordered_ids = [
            tracker.tracker_id
            for tracker in self.current_run.trackers
            if tracker.tracker_id in self._radar_selected_tracker_ids
        ]
        CacheRegistry(self._project.cache_manifest).put(
            CacheRecord(
                module=RADAR_SELECTION_MODULE,
                cache_schema_version=1,
                algorithm_version=RADAR_SELECTION_VERSION,
                created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                input_fingerprint=self._radar_selection_fingerprint(),
                payload={"tracker_ids": ordered_ids},
            )
        )

    def _restore_radar_selection(self) -> bool:
        if self._project is None or self.current_run is None:
            return False
        registry = CacheRegistry(self._project.cache_manifest)
        record = registry.get(
            RADAR_SELECTION_MODULE, self._radar_selection_fingerprint()
        )
        if (
            record is None
            or record.algorithm_version != RADAR_SELECTION_VERSION
            or record.payload is None
        ):
            return False
        raw_ids = record.payload.get("tracker_ids")
        if not isinstance(raw_ids, list) or not all(
            isinstance(tracker_id, str) for tracker_id in raw_ids
        ):
            registry.clear(RADAR_SELECTION_MODULE)
            return False
        self._radar_selected_tracker_ids = set(raw_ids) & self._radar_known_tracker_ids
        self._rebuild_radar_tracker_menu()
        return True

    def set_radar_tracker_selected(self, tracker_id: str, selected: bool) -> None:
        if tracker_id not in self._radar_known_tracker_ids:
            return
        if selected:
            self._radar_selected_tracker_ids.add(tracker_id)
        else:
            self._radar_selected_tracker_ids.discard(tracker_id)
        self._write_radar_selection_cache()
        self._update_radar_tracker_button()
        self._update_enabled_state()
        self.project_changed.emit()

    def set_all_radar_trackers_selected(self, selected: bool) -> None:
        self._radar_selected_tracker_ids = (
            set(self._radar_known_tracker_ids) if selected else set()
        )
        self._write_radar_selection_cache()
        self._rebuild_radar_tracker_menu()
        self._update_enabled_state()
        self.project_changed.emit()

    def _rebuild_radar_tracker_menu(self) -> None:
        self.radar_tracker_menu.clear()
        select_all = self.radar_tracker_menu.addAction("全选")
        select_all.triggered.connect(lambda: self.set_all_radar_trackers_selected(True))
        clear_all = self.radar_tracker_menu.addAction("清空")
        clear_all.triggered.connect(lambda: self.set_all_radar_trackers_selected(False))
        self.radar_tracker_menu.addSeparator()
        if self.current_run is not None:
            for tracker in self.current_run.trackers:
                action = QAction(tracker.tracker_name, self.radar_tracker_menu)
                action.setCheckable(True)
                action.setChecked(
                    tracker.tracker_id in self._radar_selected_tracker_ids
                )
                action.toggled.connect(
                    lambda checked, tracker_id=tracker.tracker_id: (
                        self.set_radar_tracker_selected(tracker_id, checked)
                    )
                )
                self.radar_tracker_menu.addAction(action)
        self._update_radar_tracker_button()

    def _update_radar_tracker_button(self) -> None:
        total = len(self._radar_known_tracker_ids)
        selected = len(self._radar_selected_tracker_ids & self._radar_known_tracker_ids)
        self.radar_tracker_button.setText(f"绘图跟踪器 {selected}/{total}")
        self.radar_tracker_button.setEnabled(self.current_run is not None)

    def show_auc(self) -> None:
        self._metric = "auc"
        self._update_metric_buttons()
        self._render_results()

    def show_dp(self) -> None:
        self._metric = "dp"
        self._update_metric_buttons()
        self._render_results()

    def radar_data(self) -> tuple[tuple[str, ...], tuple[AttributeRadarSeries, ...]]:
        if self.current_run is None or self._project is None:
            return (), ()
        attributes = tuple(item.name for item in self.current_run.attributes)
        styles = {tracker.tracker_id: tracker.style for tracker in self._project.trackers}
        output: list[AttributeRadarSeries] = []
        for index, tracker in enumerate(self.current_run.trackers):
            if tracker.tracker_id not in self._radar_selected_tracker_ids:
                continue
            values = tuple(
                tracker.scores[attribute].auc
                for attribute in attributes
            )
            output.append(
                AttributeRadarSeries(
                    tracker_id=tracker.tracker_id,
                    display_name=tracker.tracker_name,
                    auc_values=values,
                    dp_values=tuple(
                        tracker.scores[attribute].dp for attribute in attributes
                    ),
                    style=styles.get(tracker.tracker_id, default_style_for_index(index)),
                )
            )
        return attributes, tuple(output)

    def export_current_pdf(self, directory: Path | None = None) -> Path | None:
        attributes, series = self.radar_data()
        if not attributes:
            self._set_status("请先完成属性指标计算。")
            return None
        if not series:
            self._set_status("请至少选择一个跟踪器用于绘制雷达图。")
            return None
        if directory is None:
            initial = ""
            if self._project is not None and self._project.sources.export_directory:
                initial = str(self._project.sources.export_directory)
            selected = QFileDialog.getExistingDirectory(self, "选择 PDF 导出目录", initial)
            if not selected:
                return None
            directory = Path(selected)
        try:
            target = export_attribute_radar_pdf(
                Path(directory),
                attributes,
                series,
                project_name=self._project.name if self._project is not None else "Attributes",
            )
        except OSError as exc:
            self._set_status(str(exc))
            return None
        self._set_status(f"已导出：{target}")
        return target

    def _render_attributes(self) -> None:
        self.attribute_table.setRowCount(len(self._attributes))
        for row, attribute in enumerate(self._attributes):
            for column, text in enumerate(
                (attribute.name, str(len(attribute.sequences)), ", ".join(attribute.sequences))
            ):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.attribute_table.setItem(row, column, item)
            self.attribute_table.resizeRowToContents(row)

    def _render_results(self) -> None:
        run = self.current_run
        trackers = () if run is None else run.trackers
        self.result_table.setColumnCount(1 + len(self._attributes))
        self.result_table.setHorizontalHeaderLabels(
            ["跟踪器", *[attribute.name for attribute in self._attributes]]
        )
        self.result_table.setRowCount(len(trackers))
        for row, tracker in enumerate(trackers):
            self.result_table.setItem(row, 0, QTableWidgetItem(tracker.tracker_name))
            for column, attribute in enumerate(self._attributes, start=1):
                score = tracker.scores.get(attribute.name)
                value = None if score is None else getattr(score, self._metric)
                item = QTableWidgetItem(format_metric(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.result_table.setItem(row, column, item)
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.result_table.setColumnWidth(0, 150)
        for column in range(1, self.result_table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self.result_table.setColumnWidth(column, 92)

    def _update_metric_buttons(self) -> None:
        self.auc_button.setChecked(self._metric == "auc")
        self.dp_button.setChecked(self._metric == "dp")
        self.export_button.setText("导出属性雷达图 PDF…")

    def _update_source_summary(self) -> None:
        if self._project is None:
            self.source_summary.setText("尚未创建项目")
            return
        sources = self._project.sources
        parts = [
            f"属性 {len(self._attributes)}",
            f"GT {'已导入' if sources.ground_truth else '未导入'}",
            f"跟踪器 {len(self._project.trackers)}",
        ]
        self.source_summary.setText("  ·  ".join(parts))

    def _validation_message(self) -> str:
        if self._project is None:
            return "请先新建或打开项目。"
        if not self._attributes:
            return "请先导入属性字典。"
        if self._project.sources.ground_truth is None:
            return "请先导入 Ground Truth。"
        if not self._project.trackers:
            return "请先导入至少一个跟踪器。"
        return ""

    def _update_enabled_state(self) -> None:
        ready = not self._validation_message()
        self.run_button.setEnabled(ready and self._active_task_id is None)
        has_radar_trackers = bool(
            self._radar_selected_tracker_ids & self._radar_known_tracker_ids
        )
        self.export_button.setEnabled(
            self.current_run is not None and has_radar_trackers
        )
        self.radar_tracker_button.setEnabled(self.current_run is not None)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_message.emit(message)

    def _choose_attribute_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "导入属性字典", "", "属性字典 (*.txt);;所有文件 (*)"
        )
        if filename:
            try:
                self.load_attribute_file(Path(filename))
            except AttributeDataError as exc:
                self._set_status(str(exc))

    def _choose_ground_truth(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "导入 Ground Truth")
        if directory:
            self.set_ground_truth_directory(Path(directory))

    def _choose_ground_truth_csv(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "导入 Ground Truth CSV", "", "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if filename:
            self.set_ground_truth_directory(Path(filename))

    def _choose_trackers(self) -> None:
        if self._project is None or self._project.sources.ground_truth is None:
            self._set_status("请先导入 Ground Truth。")
            return
        directory = QFileDialog.getExistingDirectory(self, "导入跟踪器结果根目录")
        if directory:
            self.import_trackers_from_root(Path(directory))

    def _choose_tracker_csv(self) -> None:
        if self._project is None or self._project.sources.ground_truth is None:
            self._set_status("请先导入 Ground Truth。")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "导入跟踪结果 CSV", "", "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if filename:
            self.import_trackers_from_root(Path(filename))

    def _choose_and_export_radar(self) -> None:
        self.export_current_pdf()

    def _on_task_succeeded(self, task_id: str, result: object) -> None:
        if task_id == self._active_task_id and isinstance(result, AttributeEvaluationRun):
            self._accept_computed_run(result)

    def _on_task_failed(self, task_id: str, message: str) -> None:
        if task_id == self._active_task_id:
            self._set_status(f"属性指标计算失败：{message}")

    def _on_task_finished(self, task_id: str) -> None:
        if task_id == self._active_task_id:
            self._active_task_id = None
            self._update_enabled_state()


__all__ = ["AttributePage"]
