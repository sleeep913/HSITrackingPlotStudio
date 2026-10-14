from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolBar,
    QWidget,
)

from hsi_plotter import APP_DISPLAY_NAME
from hsi_plotter.cache.registry import CacheRegistry
from hsi_plotter.io.cache_store import clear_sidecar_cache
from hsi_plotter.io.project_store import ProjectFormatError, load_project, save_project
from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.models.tracker import Tracker, assign_stable_styles
from hsi_plotter.tasks.worker import TaskRunner
from hsi_plotter.ui.attribute_page import AttributePage
from hsi_plotter.ui.evaluation_page import EvaluationPage
from hsi_plotter.ui.plot_page import PlotPage
from hsi_plotter.ui.project_panel import ProjectPanel
from hsi_plotter.ui.source_config_page import SourceConfigPage
from hsi_plotter.ui.tracker_import_page import TrackerImportPage
from hsi_plotter.ui.tracking_visualization_page import TrackingVisualizationPage

MODULE_TABS: Sequence[str] = (
    "指标评估",
    "曲线绘制",
    "跟踪可视化",
    "属性评估",
)

MAX_RECENT_PROJECTS = 8


class MainWindow(QMainWindow):
    def __init__(self, *, settings: QSettings | None = None) -> None:
        super().__init__()
        self.setObjectName("mainWindow")
        self.setMinimumSize(1100, 720)
        self.settings = settings if settings is not None else QSettings()
        self.current_project: Project | None = None
        self.current_project_path: Path | None = None
        self.is_project_modified = False
        self.task_runner = TaskRunner(self, max_concurrent=1)
        self._create_actions()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_workspace()
        self._create_status_widgets()
        self._connect_task_runner()
        self._refresh_recent_projects_menu()
        self._refresh_project_state()
        self.statusBar().showMessage("就绪")

    @property
    def recent_project_paths(self) -> list[Path]:
        stored = self.settings.value("recentProjects", [])
        if isinstance(stored, str):
            stored = [stored]
        return [Path(value) for value in stored]

    def _create_actions(self) -> None:
        self.new_action = QAction("新建项目", self)
        self.new_action.setObjectName("newProjectAction")
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.triggered.connect(self._new_project_interactive)
        self.open_action = QAction("打开项目…", self)
        self.open_action.setObjectName("openProjectAction")
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self._open_project_interactive)
        self.save_action = QAction("保存项目", self)
        self.save_action.setObjectName("saveProjectAction")
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(lambda: self._save_project_interactive())
        self.save_as_action = QAction("项目另存为…", self)
        self.save_as_action.setObjectName("saveProjectAsAction")
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(lambda: self._save_project_interactive(save_as=True))
        self.exit_action = QAction("退出", self)
        self.exit_action.setObjectName("exitAction")
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)
        self.clear_cache_action = QAction("清除项目缓存", self)
        self.clear_cache_action.setObjectName("clearProjectCacheAction")
        self.clear_cache_action.triggered.connect(self.clear_project_cache)

    def _create_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        file_menu.setObjectName("fileMenu")
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        self.recent_menu = QMenu("最近项目", self)
        self.recent_menu.setObjectName("recentProjectsMenu")
        file_menu.addMenu(self.recent_menu)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addAction(self.clear_cache_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        help_menu = self.menuBar().addMenu("帮助")
        help_menu.setObjectName("helpMenu")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("项目", self)
        toolbar.setObjectName("projectToolBar")
        toolbar.setMovable(False)
        toolbar.addAction(self.new_action)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)
        self.addToolBar(toolbar)

    def _create_workspace(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setObjectName("workspaceSplitter")
        self.project_panel = ProjectPanel()
        self.project_panel.configure_sources_requested.connect(self.show_source_configuration)
        self.project_panel.manage_trackers_requested.connect(self.show_tracker_import)
        splitter.addWidget(self.project_panel)
        self.module_pages: dict[str, QWidget] = {}
        self.evaluation_page = EvaluationPage()
        self.evaluation_page.status_message.connect(self._show_status_message)
        self.evaluation_page.project_changed.connect(
            lambda: self.set_project_modified(True)
        )
        self.evaluation_page.set_task_runner(self.task_runner)
        self.plot_page = PlotPage()
        self.plot_page.status_message.connect(self._show_status_message)
        self.plot_page.project_changed.connect(lambda: self.set_project_modified(True))
        self.evaluation_page.evaluation_finished.connect(self.plot_page.set_evaluation_run)
        self.tracking_visualization_page = TrackingVisualizationPage()
        self.tracking_visualization_page.status_message.connect(self._show_status_message)
        self.tracking_visualization_page.project_changed.connect(
            lambda: self.set_project_modified(True)
        )
        self.tracking_visualization_page.set_task_runner(self.task_runner)
        self.attribute_page = AttributePage()
        self.attribute_page.status_message.connect(self._show_status_message)
        self.attribute_page.project_changed.connect(
            lambda: self.set_project_modified(True)
        )
        self.attribute_page.set_task_runner(self.task_runner)
        self.module_pages["指标评估"] = self.evaluation_page
        self.module_pages["曲线绘制"] = self.plot_page
        self.module_pages["跟踪可视化"] = self.tracking_visualization_page
        self.module_pages["属性评估"] = self.attribute_page
        self.module_tabs = QTabWidget()
        self.module_tabs.setObjectName("moduleTabs")
        self.module_tabs.setDocumentMode(True)
        for title in MODULE_TABS:
            self.module_tabs.addTab(self.module_pages[title], title)
        self.source_config_page = SourceConfigPage()
        self.source_config_page.sources_changed.connect(self._on_sources_changed)
        self.source_config_page.back_requested.connect(self.show_module_workspace)
        self.source_config_page.tracker_import_requested.connect(self.show_tracker_import)
        self.source_config_page.status_message.connect(self._show_status_message)
        self.source_config_page.set_task_runner(self.task_runner)
        self.tracker_import_page = TrackerImportPage()
        self.tracker_import_page.back_requested.connect(self.show_module_workspace)
        self.tracker_import_page.trackers_committed.connect(self.apply_trackers)
        self.tracker_import_page.status_message.connect(self._show_status_message)
        self.tracker_import_page.set_task_runner(self.task_runner)
        self.workspace_stack = QStackedWidget()
        self.workspace_stack.setObjectName("workspaceStack")
        self.workspace_stack.addWidget(self.module_tabs)
        self.workspace_stack.addWidget(self.source_config_page)
        self.workspace_stack.addWidget(self.tracker_import_page)
        splitter.addWidget(self.workspace_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([290, 810])
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

    def _create_status_widgets(self) -> None:
        self.task_progress_bar = QProgressBar()
        self.task_progress_bar.setObjectName("taskProgressBar")
        self.task_progress_bar.setRange(0, 100)
        self.task_progress_bar.setFixedWidth(180)
        self.task_progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.task_progress_bar)
        self.cancel_task_button = QPushButton("取消任务")
        self.cancel_task_button.setObjectName("cancelTaskButton")
        self.cancel_task_button.setVisible(False)
        self.cancel_task_button.clicked.connect(self.task_runner.cancel_all)
        self.statusBar().addPermanentWidget(self.cancel_task_button)

    def _connect_task_runner(self) -> None:
        self.task_runner.taskStarted.connect(self._on_task_started)
        self.task_runner.taskProgress.connect(self._on_task_progress)
        self.task_runner.taskFailed.connect(self._on_task_failed)
        self.task_runner.taskCancelled.connect(self._on_task_cancelled)
        self.task_runner.busyChanged.connect(self._on_task_busy_changed)

    # ------------------------------------------------------------- project IO

    def create_new_project(self, name: str) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("项目名称不能为空")
        self.current_project = Project(name=normalized_name)
        self.current_project_path = None
        self.set_project_modified(True)
        self.statusBar().showMessage(f"已创建项目：{normalized_name}", 5000)

    def open_project_from(self, source: Path) -> None:
        path = Path(source).resolve(strict=False)
        self.current_project = load_project(path)
        self.current_project_path = path
        self.set_project_modified(False)
        self._remember_recent_project(path)
        self.statusBar().showMessage(f"已打开项目：{path}", 5000)

    def save_project_to(self, destination: Path) -> None:
        if self.current_project is None:
            raise RuntimeError("当前没有可保存的项目")
        path = Path(destination)
        if path.suffix.casefold() != ".hsiproj":
            path = path.with_suffix(".hsiproj")
        path = path.resolve(strict=False)
        save_project(self.current_project, path)
        self.current_project_path = path
        self.set_project_modified(False)
        self._remember_recent_project(path)
        self.statusBar().showMessage(f"项目已保存：{path}", 5000)

    def set_project_modified(self, modified: bool) -> None:
        self.is_project_modified = modified
        self._refresh_project_state()

    def apply_trackers(self, trackers: Sequence[Tracker]) -> tuple[Tracker, ...]:
        """Replace the project tracker list while keeping stable styles."""
        if self.current_project is None:
            return ()
        styled = assign_stable_styles(self.current_project.trackers, list(trackers))
        self.current_project.trackers = list(styled)
        self.set_project_modified(True)
        self.statusBar().showMessage(f"已导入 {len(styled)} 个跟踪器", 5000)
        return styled

    def clear_project_cache(self) -> None:
        project = self.current_project
        if project is None:
            return
        CacheRegistry(project.cache_manifest).clear()
        if self.current_project_path is not None:
            try:
                clear_sidecar_cache(self.current_project_path)
            except (OSError, ValueError) as exc:
                self.statusBar().showMessage(f"旁路缓存清除失败：{exc}", 10000)
                return
        self.evaluation_page.clear_results()
        self.plot_page.clear_results()
        self.attribute_page.clear_results()
        self.set_project_modified(True)
        self.statusBar().showMessage("项目缓存已清除，数据源与跟踪器配置保持不变。", 8000)

    # ------------------------------------------------------------- navigation

    def show_source_configuration(self) -> None:
        if self.current_project is None:
            return
        self.source_config_page.set_project(self.current_project)
        self.workspace_stack.setCurrentWidget(self.source_config_page)

    def show_tracker_import(self) -> None:
        if self.current_project is None:
            return
        self.tracker_import_page.set_project(self.current_project)
        self.workspace_stack.setCurrentWidget(self.tracker_import_page)
        self.tracker_import_page.rescan()

    def show_module_workspace(self) -> None:
        self.workspace_stack.setCurrentWidget(self.module_tabs)

    def _on_sources_changed(self, sources: ProjectSources) -> None:
        if self.current_project is None:
            return
        self.current_project.sources = sources
        self.set_project_modified(True)

    def _show_status_message(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)

    # ----------------------------------------------------------- task runner

    def _on_task_started(self, task_id: str, name: str) -> None:
        self.task_progress_bar.setValue(0)
        self.statusBar().showMessage(f"{name}…", 0)

    def _on_task_progress(self, task_id: str, value: int, message: str) -> None:
        self.task_progress_bar.setValue(value)
        if message:
            self.statusBar().showMessage(message, 0)

    def _on_task_failed(self, task_id: str, message: str) -> None:
        self.statusBar().showMessage(f"任务失败：{message}", 10000)

    def _on_task_cancelled(self, task_id: str) -> None:
        self.statusBar().showMessage("任务已取消", 6000)

    def _on_task_busy_changed(self, busy: bool) -> None:
        self.task_progress_bar.setVisible(busy)
        self.cancel_task_button.setVisible(busy)
        if not busy:
            QTimer.singleShot(0, lambda: self.task_progress_bar.setValue(0))

    # ---------------------------------------------------------------- chrome

    def _refresh_project_state(self) -> None:
        has_project = self.current_project is not None
        self.save_action.setEnabled(has_project)
        self.save_as_action.setEnabled(has_project)
        self.clear_cache_action.setEnabled(has_project)
        self.module_tabs.setEnabled(has_project)
        self.project_panel.set_project(
            self.current_project, self.current_project_path, self.is_project_modified
        )
        self.source_config_page.set_project(self.current_project)
        self.tracker_import_page.set_project(self.current_project)
        self.evaluation_page.set_project(self.current_project)
        self.plot_page.set_project(self.current_project)
        self.tracking_visualization_page.set_project(self.current_project)
        self.attribute_page.set_project(self.current_project)
        title = APP_DISPLAY_NAME
        if self.current_project is not None:
            title = f"{title} - {self.current_project.name}"
            if self.is_project_modified:
                title = f"{title} *"
        self.setWindowTitle(title)

    def _remember_recent_project(self, path: Path) -> None:
        resolved = path.resolve(strict=False)
        recent = [item for item in self.recent_project_paths if item != resolved]
        recent.insert(0, resolved)
        self.settings.setValue(
            "recentProjects", [str(item) for item in recent[:MAX_RECENT_PROJECTS]]
        )
        self.settings.sync()
        self._refresh_recent_projects_menu()

    def _refresh_recent_projects_menu(self) -> None:
        self.recent_menu.clear()
        paths = self.recent_project_paths
        if not paths:
            empty_action = self.recent_menu.addAction("暂无最近项目")
            empty_action.setEnabled(False)
            return
        for path in paths:
            action = self.recent_menu.addAction(str(path))
            action.setEnabled(path.is_file())
            action.triggered.connect(
                lambda checked=False, selected=path: self._open_recent_project(selected)
            )

    def _new_project_interactive(self) -> None:
        if not self._confirm_discard_or_save():
            return
        name, accepted = QInputDialog.getText(self, "新建项目", "项目名称：")
        if accepted and name.strip():
            self.create_new_project(name)

    def _open_project_interactive(self) -> None:
        if not self._confirm_discard_or_save():
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", "HSI 项目 (*.hsiproj);;JSON 文件 (*.json);;所有文件 (*)"
        )
        if filename:
            self._open_with_error_dialog(Path(filename))

    def _open_recent_project(self, path: Path) -> None:
        if self._confirm_discard_or_save():
            self._open_with_error_dialog(path)

    def _open_with_error_dialog(self, path: Path) -> None:
        try:
            self.open_project_from(path)
        except (OSError, ProjectFormatError) as exc:
            QMessageBox.critical(self, "无法打开项目", str(exc))

    def _save_project_interactive(self, *, save_as: bool = False) -> bool:
        if self.current_project is None:
            return False
        destination = None if save_as else self.current_project_path
        if destination is None:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "保存项目",
                f"{self.current_project.name}.hsiproj",
                "HSI 项目 (*.hsiproj)",
            )
            if not filename:
                return False
            destination = Path(filename)
        try:
            self.save_project_to(destination)
        except OSError as exc:
            QMessageBox.critical(self, "无法保存项目", str(exc))
            return False
        return True

    def _confirm_discard_or_save(self) -> bool:
        if self.current_project is None or not self.is_project_modified:
            return True
        choice = QMessageBox.question(
            self,
            "项目尚未保存",
            "当前项目包含未保存的更改。是否先保存？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Save:
            return self._save_project_interactive()
        return choice == QMessageBox.StandardButton.Discard

    def _confirm_cancel_running_tasks(self) -> bool:
        if not self.task_runner.is_busy:
            return True
        choice = QMessageBox.question(
            self,
            "后台任务仍在运行",
            "有后台任务正在执行。是否取消任务并退出？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return False
        self.task_runner.cancel_all()
        self.task_runner.wait_for_idle(3000)
        return True

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "关于", "高光谱目标跟踪科研绘图平台\n本地离线 Python/PySide6 桌面应用"
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        if not self._confirm_cancel_running_tasks():
            event.ignore()
            return
        if self._confirm_discard_or_save():
            event.accept()
        else:
            event.ignore()
