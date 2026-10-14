from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hsi_plotter.cache.project_fingerprint import (
    VISUALIZATION_ALGORITHM_VERSION,
    visualization_fingerprint,
)
from hsi_plotter.cache.registry import CacheRegistry
from hsi_plotter.cache.visualization_codec import (
    decode_visualization_data,
    encode_visualization_data,
)
from hsi_plotter.io.cache_store import read_json_cache
from hsi_plotter.io.video_exporter import VideoExportError, export_tracking_video
from hsi_plotter.io.visualization_exporter import (
    RangeExportResult,
    VisualizationExportError,
    export_tracking_frame,
    export_tracking_montage,
    export_tracking_range,
)
from hsi_plotter.io.visualization_recipe_store import (
    VisualizationRecipeFormatError,
    decode_visualization_recipe,
    encode_visualization_recipe,
    load_visualization_recipe,
    save_visualization_recipe,
    validate_visualization_recipe_references,
)
from hsi_plotter.models.cache import CacheRecord
from hsi_plotter.models.dataset import MODALITY_LABELS, Modality, SequenceKey
from hsi_plotter.models.project import Project
from hsi_plotter.models.visualization import VisualSelectionError, VisualSelectionList
from hsi_plotter.models.visualization_recipe import VisualizationRecipe
from hsi_plotter.plotting.tracking_visualizer import (
    TrackingRenderOptions,
    build_tracking_figure,
)
from hsi_plotter.tasks.worker import CancelToken, ProgressReporter, TaskRunner
from hsi_plotter.visualization.frame_selection import (
    AdvantageFrameSelection,
    FrameSelectionError,
    select_advantage_frames,
)
from hsi_plotter.visualization.project_data import (
    ProjectVisualizationData,
    prepare_visualization_project,
)
from hsi_plotter.visualization.session import SequenceVisualizationSession


class TrackingVisualizationPage(QWidget):
    status_message = Signal(str)
    project_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("trackingVisualizationPage")
        self._project: Project | None = None
        self._data: ProjectVisualizationData | None = None
        self._session: SequenceVisualizationSession | None = None
        self._task_runner: TaskRunner | None = None
        self._runner_connected = False
        self._active_scan_task: str | None = None
        self._active_filter_task: str | None = None
        self._active_export_task: str | None = None
        self._active_export_kind: str | None = None
        self._visual_selections = VisualSelectionList()
        self._pan_anchor: tuple[
            float,
            float,
            tuple[float, float],
            tuple[float, float],
        ] | None = None
        self.preview_canvas: FigureCanvasQTAgg | None = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_playback)
        self._build_ui()
        self._connect_controls()
        self._set_empty_state("请先创建或打开项目。")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 22)
        root.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("跟踪可视化")
        title.setProperty("role", "heading")
        subtitle = QLabel("逐帧查看跟踪结果，筛选优势帧并生成跨序列论文 Visual。")
        subtitle.setProperty("role", "caption")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        self.export_frame_button = QPushButton("导出当前帧…")
        self.export_range_button = QPushButton("导出区间…")
        self.export_video_button = QPushButton("导出 MP4…")
        self.load_recipe_button = QPushButton("导入配置…")
        self.save_recipe_button = QPushButton("导出配置…")
        header.addWidget(self.load_recipe_button)
        header.addWidget(self.save_recipe_button)
        header.addWidget(self.export_frame_button)
        header.addWidget(self.export_range_button)
        header.addWidget(self.export_video_button)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(12)
        controls = self._build_controls()
        controls.setFixedWidth(286)
        body.addWidget(controls)
        body.addWidget(self._build_preview(), 1)
        visual = self._build_visual_list()
        visual.setFixedWidth(315)
        body.addWidget(visual)
        root.addLayout(body, 1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("trackingVisualizationStatus")
        self.status_label.setProperty("role", "caption")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

    def _build_controls(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName("trackingVisualizationControlsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        frame = QFrame()
        frame.setObjectName("trackingVisualizationControls")
        frame.setMinimumWidth(250)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        heading = QLabel("查看设置")
        heading.setStyleSheet("font-weight: 600; color: #111827;")
        layout.addWidget(heading)
        layout.addWidget(QLabel("模态"))
        self.modality_combo = QComboBox()
        self.modality_combo.setObjectName("visualizationModalityCombo")
        layout.addWidget(self.modality_combo)
        layout.addWidget(QLabel("序列"))
        self.sequence_combo = QComboBox()
        self.sequence_combo.setObjectName("visualizationSequenceCombo")
        layout.addWidget(self.sequence_combo)
        layout.addWidget(QLabel("主方法"))
        self.primary_combo = QComboBox()
        self.primary_combo.setObjectName("visualizationPrimaryTrackerCombo")
        layout.addWidget(self.primary_combo)
        layout.addWidget(QLabel("显示及对比方法"))
        self.tracker_list = QListWidget()
        self.tracker_list.setObjectName("visualizationTrackerList")
        self.tracker_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tracker_list.setMinimumHeight(180)
        layout.addWidget(self.tracker_list, 1)
        self.ground_truth_checkbox = QCheckBox("Ground Truth")
        self.ground_truth_checkbox.setChecked(True)
        self.legend_checkbox = QCheckBox("图例")
        self.legend_checkbox.setChecked(True)
        self.names_checkbox = QCheckBox("名称标签")
        self.frame_number_checkbox = QCheckBox("帧号")
        self.frame_number_checkbox.setChecked(True)
        display_options = QGridLayout()
        display_options.setHorizontalSpacing(10)
        display_options.setVerticalSpacing(6)
        display_options.addWidget(self.ground_truth_checkbox, 0, 0)
        display_options.addWidget(self.legend_checkbox, 0, 1)
        display_options.addWidget(self.names_checkbox, 1, 0)
        display_options.addWidget(self.frame_number_checkbox, 1, 1)
        layout.addLayout(display_options)
        layout.addWidget(QLabel("播放与导出范围"))
        range_row = QHBoxLayout()
        self.range_start_spin = QSpinBox()
        self.range_end_spin = QSpinBox()
        self.range_start_spin.setPrefix("起 ")
        self.range_end_spin.setPrefix("止 ")
        range_row.addWidget(self.range_start_spin)
        range_row.addWidget(self.range_end_spin)
        layout.addLayout(range_row)
        self.range_format_combo = QComboBox()
        for suffix in ("png", "pdf", "svg", "eps"):
            self.range_format_combo.addItem(suffix.upper(), suffix)
        layout.addWidget(self.range_format_combo)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("1200 × 600", (1200, 600))
        self.resolution_combo.addItem("1280 × 720", (1280, 720))
        self.resolution_combo.addItem("1920 × 1080", (1920, 1080))
        layout.addWidget(self.resolution_combo)
        view_actions = QHBoxLayout()
        self.fit_button = QPushButton("适应窗口")
        self.original_button = QPushButton("原始尺寸")
        view_actions.addWidget(self.fit_button)
        view_actions.addWidget(self.original_button)
        layout.addLayout(view_actions)
        scroll.setWidget(frame)
        return scroll

    def _build_preview(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("trackingVisualizationCenter")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.preview_frame = QFrame()
        self.preview_frame.setObjectName("trackingPreviewFrame")
        self.preview_frame.setMaximumHeight(540)
        self.preview_frame.setMinimumHeight(320)
        self.preview_layout = QVBoxLayout(self.preview_frame)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_placeholder = QLabel("尚未加载图像")
        self.preview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_placeholder.setProperty("role", "caption")
        self.preview_layout.addWidget(self.preview_placeholder)
        layout.addWidget(self.preview_frame, 1, Qt.AlignmentFlag.AlignCenter)

        playback = QHBoxLayout()
        self.previous_button = QPushButton("上一帧")
        self.play_button = QPushButton("播放")
        self.stop_button = QPushButton("停止")
        self.next_button = QPushButton("下一帧")
        self.frame_spin = QSpinBox()
        self.frame_spin.setPrefix("帧 ")
        self.frame_spin.setMinimum(1)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(1)
        self.speed_combo = QComboBox()
        for fps in (1, 2, 5, 10, 15, 25):
            self.speed_combo.addItem(f"{fps} FPS", fps)
        self.speed_combo.setCurrentIndex(3)
        self.loop_checkbox = QCheckBox("循环")
        playback.addWidget(self.previous_button)
        playback.addWidget(self.play_button)
        playback.addWidget(self.stop_button)
        playback.addWidget(self.next_button)
        playback.addWidget(self.frame_spin)
        playback.addWidget(self.frame_slider, 1)
        playback.addWidget(self.speed_combo)
        playback.addWidget(self.loop_checkbox)
        layout.addLayout(playback)

        candidates = QHBoxLayout()
        self.filter_frames_button = QPushButton("帧筛选")
        self.filter_frames_button.setProperty("primary", True)
        candidates.addWidget(self.filter_frames_button)
        self.candidate_buttons: list[QPushButton] = []
        for _ in range(5):
            button = QPushButton("—")
            button.setHidden(True)
            button.clicked.connect(self._candidate_clicked)
            self.candidate_buttons.append(button)
            candidates.addWidget(button)
        candidates.addStretch(1)
        layout.addLayout(candidates)
        return panel

    def _build_visual_list(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("trackingVisualList")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        heading = QLabel("Visual 绘制清单")
        heading.setStyleSheet("font-weight: 600; color: #111827;")
        layout.addWidget(heading)
        hint = QLabel("可跨序列累计；每个序列选满三帧后生成固定四列 Visual。")
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.visual_table = QTableWidget(0, 3)
        self.visual_table.setObjectName("trackingVisualTable")
        self.visual_table.setHorizontalHeaderLabels(["模态", "序列", "帧号"])
        self.visual_table.verticalHeader().setVisible(False)
        self.visual_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.visual_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.visual_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.visual_table, 1)
        self.add_visual_button = QPushButton("加入 Visual")
        self.add_visual_button.setProperty("primary", True)
        self.remove_visual_button = QPushButton("删除选中")
        self.clear_visual_button = QPushButton("清空")
        row = QHBoxLayout()
        row.addWidget(self.add_visual_button)
        row.addWidget(self.remove_visual_button)
        row.addWidget(self.clear_visual_button)
        layout.addLayout(row)
        self.draw_visual_button = QPushButton("绘制 Visual…")
        self.draw_visual_button.setEnabled(False)
        layout.addWidget(self.draw_visual_button)
        return frame

    def _connect_controls(self) -> None:
        self.modality_combo.currentIndexChanged.connect(self._populate_sequences)
        self.sequence_combo.currentIndexChanged.connect(self._load_current_sequence)
        self.tracker_list.itemChanged.connect(lambda _item: self._render_current_frame())
        self.frame_spin.valueChanged.connect(self._frame_spin_changed)
        self.frame_slider.valueChanged.connect(self._frame_slider_changed)
        self.previous_button.clicked.connect(lambda: self._step_frame(-1))
        self.next_button.clicked.connect(lambda: self._step_frame(1))
        self.play_button.clicked.connect(self._toggle_playback)
        self.stop_button.clicked.connect(self._stop_playback)
        self.speed_combo.currentIndexChanged.connect(self._update_playback_interval)
        for checkbox in (
            self.ground_truth_checkbox,
            self.legend_checkbox,
            self.names_checkbox,
            self.frame_number_checkbox,
        ):
            checkbox.toggled.connect(lambda _checked: self._render_current_frame())
        self.fit_button.clicked.connect(self._fit_view)
        self.original_button.clicked.connect(self._original_view)
        self.filter_frames_button.clicked.connect(self.start_frame_filter)
        self.add_visual_button.clicked.connect(self.add_current_to_visual)
        self.remove_visual_button.clicked.connect(self._remove_selected_visual)
        self.clear_visual_button.clicked.connect(self._clear_visuals)
        self.export_frame_button.clicked.connect(self._export_current_frame_interactive)
        self.export_range_button.clicked.connect(self._export_range_interactive)
        self.export_video_button.clicked.connect(self._export_video_interactive)
        self.draw_visual_button.clicked.connect(self._export_visual_interactive)
        self.save_recipe_button.clicked.connect(self._save_recipe_interactive)
        self.load_recipe_button.clicked.connect(self._load_recipe_interactive)

    def set_task_runner(self, runner: TaskRunner) -> None:
        self._task_runner = runner
        if self._runner_connected:
            return
        runner.taskSucceeded.connect(self._task_succeeded)
        runner.taskFailed.connect(self._task_failed)
        runner.taskFinished.connect(self._task_finished)
        self._runner_connected = True

    def set_project(self, project: Project | None) -> None:
        if project is self._project:
            return
        if self._active_scan_task is not None and self._task_runner is not None:
            self._task_runner.cancel(self._active_scan_task)
            self._active_scan_task = None
        self._project = project
        self._data = None
        self._session = None
        self._play_timer.stop()
        self._visual_selections = VisualSelectionList()
        self._refresh_visual_table()
        if project is None:
            self._set_empty_state("请先创建或打开项目。")
            return
        if project.sources.ground_truth is None:
            self._set_empty_state("请先配置 Ground Truth 目录。")
            return
        if project.sources.image_root is None:
            self._set_empty_state("请先配置图像序列根目录。")
            return
        if self._task_runner is not None:
            self._set_empty_state("正在扫描图像序列与跟踪结果…")
            self._active_scan_task = self._task_runner.submit(
                "扫描跟踪可视化数据",
                lambda _reporter, _token: self._load_or_prepare_project_data(project),
            )
            return
        data, from_cache = self._load_or_prepare_project_data(project)
        self._apply_project_data(data)
        if not from_cache:
            self._cache_project_data(data)

    def _load_or_prepare_project_data(
        self, project: Project
    ) -> tuple[ProjectVisualizationData, bool]:
        registry = CacheRegistry(project.cache_manifest)
        record = registry.manifest.records.get("tracking_visualization_index")
        fingerprint = visualization_fingerprint(project)
        if (
            record is not None
            and record.input_fingerprint == fingerprint
            and record.algorithm_version == VISUALIZATION_ALGORITHM_VERSION
        ):
            try:
                payload = record.payload
                if payload is None and record.relative_path and project.storage_path:
                    payload = read_json_cache(project.storage_path, record.relative_path)
                if payload is not None:
                    return decode_visualization_data(payload, project), True
            except (OSError, KeyError, TypeError, ValueError):
                registry.clear("tracking_visualization_index")
        return prepare_visualization_project(project), False

    def _cache_project_data(self, data: ProjectVisualizationData) -> None:
        if self._project is None:
            return
        CacheRegistry(self._project.cache_manifest).put(
            CacheRecord(
                module="tracking_visualization_index",
                cache_schema_version=1,
                algorithm_version=VISUALIZATION_ALGORITHM_VERSION,
                created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                input_fingerprint=visualization_fingerprint(self._project),
                payload=encode_visualization_data(data),
                payload_kind="sidecar_json",
            )
        )
        self.project_changed.emit()

    def _apply_project_data(self, data: ProjectVisualizationData) -> None:
        self._data = data
        if not self._data.sequences:
            message = next(
                (issue.message for issue in self._data.issues if issue.severity.value == "error"),
                "没有可显示的图像序列。",
            )
            self._set_empty_state(message)
            return
        self._populate_trackers()
        with QSignalBlocker(self.modality_combo):
            self.modality_combo.clear()
            modalities = sorted(
                {key.modality for key in self._data.sequences},
                key=lambda item: {
                    Modality.VIS: 0,
                    Modality.REDNIR: 1,
                    Modality.NIR: 2,
                    Modality.UNSPECIFIED: 3,
                }[item],
            )
            for modality in modalities:
                self.modality_combo.addItem(MODALITY_LABELS[modality], modality.value)
        self._populate_sequences()
        self._restore_cached_state()

    def _cache_current_state(self) -> None:
        if self._project is None or self._data is None:
            return
        try:
            payload = encode_visualization_recipe(self.current_recipe())
        except ValueError:
            return
        CacheRegistry(self._project.cache_manifest).put(
            CacheRecord(
                module="tracking_visualization_state",
                cache_schema_version=1,
                algorithm_version=VISUALIZATION_ALGORITHM_VERSION,
                created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                input_fingerprint=visualization_fingerprint(self._project),
                payload=payload,
            )
        )
        self.project_changed.emit()

    def _restore_cached_state(self) -> None:
        if self._project is None or self._data is None:
            return
        registry = CacheRegistry(self._project.cache_manifest)
        record = registry.get(
            "tracking_visualization_state", visualization_fingerprint(self._project)
        )
        if (
            record is None
            or record.algorithm_version != VISUALIZATION_ALGORITHM_VERSION
            or record.payload is None
        ):
            return
        try:
            self._apply_recipe(decode_visualization_recipe(record.payload), announce=False)
        except (KeyError, TypeError, ValueError, VisualizationRecipeFormatError):
            registry.clear("tracking_visualization_state")

    def _populate_trackers(self) -> None:
        self.tracker_list.blockSignals(True)
        self.tracker_list.clear()
        self.primary_combo.clear()
        if self._data is not None:
            for tracker in self._data.trackers:
                self.primary_combo.addItem(tracker.display_name, tracker.tracker_id)
                item = QListWidgetItem(tracker.display_name)
                item.setData(Qt.ItemDataRole.UserRole, tracker.tracker_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.tracker_list.addItem(item)
        self.tracker_list.blockSignals(False)

    def _populate_sequences(self) -> None:
        with QSignalBlocker(self.sequence_combo):
            self.sequence_combo.clear()
            if self._data is None or self.modality_combo.currentIndex() < 0:
                return
            modality = Modality(self.modality_combo.currentData())
            keys = sorted(
                (key for key in self._data.sequences if key.modality is modality),
                key=lambda key: key.name.casefold(),
            )
            for key in keys:
                self.sequence_combo.addItem(key.name, key)
        self._load_current_sequence()

    def _current_key(self) -> SequenceKey | None:
        value = self.sequence_combo.currentData()
        return value if isinstance(value, SequenceKey) else None

    def _load_current_sequence(self) -> None:
        key = self._current_key()
        if self._data is None or key is None:
            self._session = None
            self._set_empty_state("请选择序列。")
            return
        try:
            self._session = self._data.load_session(key)
        except (OSError, ValueError) as exc:
            self._session = None
            self._set_empty_state(str(exc))
            return
        maximum = max(1, self._session.frame_count)
        with QSignalBlocker(self.frame_spin), QSignalBlocker(self.frame_slider):
            self.frame_spin.setRange(1, maximum)
            self.frame_slider.setRange(1, maximum)
            self.frame_spin.setValue(1)
            self.frame_slider.setValue(1)
        with QSignalBlocker(self.range_start_spin), QSignalBlocker(self.range_end_spin):
            self.range_start_spin.setRange(1, maximum)
            self.range_end_spin.setRange(1, maximum)
            self.range_start_spin.setValue(1)
            self.range_end_spin.setValue(maximum)
        self._hide_candidates()
        self._render_current_frame()

    def _checked_tracker_ids(self) -> frozenset[str]:
        return frozenset(self._checked_tracker_id_order())

    def _checked_tracker_id_order(self) -> tuple[str, ...]:
        return tuple(
            str(item.data(Qt.ItemDataRole.UserRole))
            for row in range(self.tracker_list.count())
            if (item := self.tracker_list.item(row)).checkState() == Qt.CheckState.Checked
        )

    def _render_options(self) -> TrackingRenderOptions:
        return TrackingRenderOptions(
            show_ground_truth=self.ground_truth_checkbox.isChecked(),
            show_legend=self.legend_checkbox.isChecked(),
            show_names=self.names_checkbox.isChecked(),
            show_frame_number=self.frame_number_checkbox.isChecked(),
        )

    def _render_current_frame(self) -> None:
        if self._session is None:
            return
        composition = self._session.compose(
            self.frame_spin.value(),
            visible_tracker_ids=self._checked_tracker_ids(),
        )
        if composition.image_path is None:
            self._clear_canvas(f"第 {composition.frame_number} 帧图像缺失")
            self._set_status(f"第 {composition.frame_number} 帧图像缺失。")
            return
        try:
            figure, _ = build_tracking_figure(composition, self._render_options())
        except (OSError, ValueError) as exc:
            self._clear_canvas(str(exc))
            self._set_status(str(exc))
            return
        self._clear_preview_widgets()
        canvas = FigureCanvasQTAgg(figure)
        canvas.setMaximumHeight(520)
        canvas.setMinimumSize(560, 280)
        self.preview_canvas = canvas
        canvas.mpl_connect("scroll_event", self._canvas_scrolled)
        canvas.mpl_connect("button_press_event", self._canvas_pressed)
        canvas.mpl_connect("motion_notify_event", self._canvas_moved)
        canvas.mpl_connect("button_release_event", self._canvas_released)
        self.preview_layout.addWidget(canvas, 1, Qt.AlignmentFlag.AlignCenter)
        missing = [item.display_name for item in composition.trackers if not item.is_visible]
        if missing:
            self._set_status(
                f"第 {composition.frame_number} 帧：{', '.join(missing)} 无可绘制结果。"
            )
        else:
            self._set_status(f"第 {composition.frame_number}/{self._session.frame_count} 帧")

    def _clear_preview_widgets(self) -> None:
        while self.preview_layout.count():
            item = self.preview_layout.takeAt(0)
            if widget := item.widget():
                widget.setParent(None)
                widget.deleteLater()
        self.preview_canvas = None

    def _clear_canvas(self, message: str) -> None:
        self._clear_preview_widgets()
        self.preview_placeholder = QLabel(message)
        self.preview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_placeholder.setProperty("role", "caption")
        self.preview_layout.addWidget(self.preview_placeholder)

    def _set_empty_state(self, message: str) -> None:
        self.modality_combo.clear()
        self.sequence_combo.clear()
        self.tracker_list.clear()
        self.primary_combo.clear()
        self._clear_canvas(message)
        self._set_status(message)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_message.emit(message)

    def _frame_spin_changed(self, value: int) -> None:
        if self.frame_slider.value() != value:
            self.frame_slider.setValue(value)
        self._render_current_frame()

    def _frame_slider_changed(self, value: int) -> None:
        if self.frame_spin.value() != value:
            self.frame_spin.setValue(value)

    def _step_frame(self, step: int) -> None:
        target = self.frame_spin.value() + step
        lower = self.range_start_spin.value()
        upper = self.range_end_spin.value()
        if target > upper:
            target = lower if self.loop_checkbox.isChecked() else target - step
        if target < lower:
            target = upper if self.loop_checkbox.isChecked() else target - step
        self.frame_spin.setValue(target)

    def _toggle_playback(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.setText("播放")
        else:
            self._update_playback_interval()
            self._play_timer.start()
            self.play_button.setText("暂停")

    def _stop_playback(self) -> None:
        self._play_timer.stop()
        self.play_button.setText("播放")
        self.frame_spin.setValue(self.range_start_spin.value())

    def _update_playback_interval(self) -> None:
        fps = int(self.speed_combo.currentData() or 10)
        self._play_timer.setInterval(max(1, round(1000 / fps)))

    def _advance_playback(self) -> None:
        at_end = self.frame_spin.value() >= self.range_end_spin.value()
        if at_end and not self.loop_checkbox.isChecked():
            self._toggle_playback()
            return
        self._step_frame(1)

    def _fit_view(self) -> None:
        if self.preview_canvas is None or not self.preview_canvas.figure.axes:
            return
        axis = self.preview_canvas.figure.axes[0]
        if axis.images:
            extent = axis.images[0].get_extent()
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
            self.preview_canvas.draw_idle()

    def _original_view(self) -> None:
        if self.preview_canvas is None or not self.preview_canvas.figure.axes:
            return
        axis = self.preview_canvas.figure.axes[0]
        if not axis.images:
            return
        extent = axis.images[0].get_extent()
        image_width = abs(float(extent[1]) - float(extent[0]))
        image_height = abs(float(extent[3]) - float(extent[2]))
        device_ratio = max(1.0, self.preview_canvas.devicePixelRatioF())
        viewport_width = min(image_width, self.preview_canvas.width() / device_ratio)
        viewport_height = min(image_height, self.preview_canvas.height() / device_ratio)
        center_x = image_width / 2.0
        center_y = image_height / 2.0
        axis.set_xlim(center_x - viewport_width / 2.0, center_x + viewport_width / 2.0)
        axis.set_ylim(center_y + viewport_height / 2.0, center_y - viewport_height / 2.0)
        self.preview_canvas.draw_idle()

    def _canvas_scrolled(self, event: Any) -> None:
        if self.preview_canvas is None or event.inaxes is None:
            return
        axis = event.inaxes
        if event.xdata is None or event.ydata is None:
            return
        scale = 1 / 1.2 if event.button == "up" else 1.2
        x_limits = axis.get_xlim()
        y_limits = axis.get_ylim()
        x_width = (x_limits[1] - x_limits[0]) * scale
        y_height = (y_limits[1] - y_limits[0]) * scale
        x_ratio = (event.xdata - x_limits[0]) / (x_limits[1] - x_limits[0])
        y_ratio = (event.ydata - y_limits[0]) / (y_limits[1] - y_limits[0])
        axis.set_xlim(event.xdata - x_width * x_ratio, event.xdata + x_width * (1 - x_ratio))
        axis.set_ylim(event.ydata - y_height * y_ratio, event.ydata + y_height * (1 - y_ratio))
        self.preview_canvas.draw_idle()

    def _canvas_pressed(self, event: Any) -> None:
        if event.button != 1 or event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        self._pan_anchor = (
            event.xdata,
            event.ydata,
            event.inaxes.get_xlim(),
            event.inaxes.get_ylim(),
        )

    def _canvas_moved(self, event: Any) -> None:
        if (
            self.preview_canvas is None
            or self._pan_anchor is None
            or event.inaxes is None
            or event.xdata is None
            or event.ydata is None
        ):
            return
        start_x, start_y, x_limits, y_limits = self._pan_anchor
        delta_x = event.xdata - start_x
        delta_y = event.ydata - start_y
        event.inaxes.set_xlim(x_limits[0] - delta_x, x_limits[1] - delta_x)
        event.inaxes.set_ylim(y_limits[0] - delta_y, y_limits[1] - delta_y)
        self.preview_canvas.draw_idle()

    def _canvas_released(self, _event: Any) -> None:
        self._pan_anchor = None

    def _filter_inputs(self) -> tuple[str, tuple[str, ...]]:
        primary = str(self.primary_combo.currentData() or "")
        competitors = tuple(
            tracker_id for tracker_id in self._checked_tracker_ids() if tracker_id != primary
        )
        if not primary or not competitors:
            raise FrameSelectionError("请选择一个主方法和至少一个不同的对比方法。")
        return primary, competitors

    def start_frame_filter(self) -> None:
        if self._session is None:
            self._set_status("请先选择可用序列。")
            return
        try:
            primary, competitors = self._filter_inputs()
        except FrameSelectionError as exc:
            self._set_status(str(exc))
            return

        def calculate(
            reporter: ProgressReporter | None = None,
            token: CancelToken | None = None,
        ) -> AdvantageFrameSelection:
            if reporter is not None:
                reporter.notify(10, "正在计算优势帧…")
            if token is not None and token.is_cancelled:
                return AdvantageFrameSelection((), "帧筛选已取消。", 1)
            result = select_advantage_frames(
                self._session,
                primary_tracker_id=primary,
                competitor_ids=competitors,
            )
            if reporter is not None:
                reporter.notify(100, result.message)
            return result

        if self._task_runner is None:
            try:
                self._show_candidates(calculate())
            except FrameSelectionError as exc:
                self._set_status(str(exc))
            return
        self.filter_frames_button.setEnabled(False)
        self._active_filter_task = self._task_runner.submit(
            "筛选优势帧",
            lambda reporter, token: calculate(reporter, token),
        )

    def _task_succeeded(self, task_id: str, result: object) -> None:
        if task_id == self._active_scan_task and isinstance(result, tuple):
            data, from_cache = result
            if isinstance(data, ProjectVisualizationData):
                self._apply_project_data(data)
                if from_cache:
                    self._set_status("已恢复跟踪可视化索引缓存。")
                else:
                    self._cache_project_data(data)
        elif task_id == self._active_filter_task and isinstance(
            result, AdvantageFrameSelection
        ):
            self._show_candidates(result)
        elif task_id == self._active_export_task:
            if isinstance(result, RangeExportResult):
                suffix = "（已取消）" if result.cancelled else ""
                message = (
                    f"区间导出完成 {len(result.outputs)} 帧，"
                    f"缺失 {len(result.missing_frames)} 帧{suffix}。"
                )
                self._set_status(message)
            elif isinstance(result, Path):
                self._set_status(f"导出完成：{result}")

    def _task_failed(self, task_id: str, message: str) -> None:
        if task_id == self._active_scan_task:
            self._set_empty_state(f"跟踪可视化数据扫描失败：{message}")
        elif task_id == self._active_filter_task:
            self._set_status(f"帧筛选失败：{message}")
        elif task_id == self._active_export_task:
            self._set_status(f"导出失败：{message}")

    def _task_finished(self, task_id: str) -> None:
        if task_id == self._active_scan_task:
            self._active_scan_task = None
        elif task_id == self._active_filter_task:
            self._active_filter_task = None
            self.filter_frames_button.setEnabled(True)
        elif task_id == self._active_export_task:
            self._active_export_task = None
            self._active_export_kind = None
            self.export_range_button.setEnabled(True)
            self.export_video_button.setEnabled(True)

    def _hide_candidates(self) -> None:
        for button in self.candidate_buttons:
            button.setHidden(True)
            button.setProperty("frameNumber", None)

    def _show_candidates(self, result: AdvantageFrameSelection) -> None:
        self._hide_candidates()
        for button, candidate in zip(
            self.candidate_buttons,
            result.candidates,
            strict=False,
        ):
            button.setText(f"#{candidate.frame_number:04d}")
            button.setProperty("frameNumber", candidate.frame_number)
            button.setToolTip(
                f"主方法 IoU {candidate.primary_iou:.4f}\n"
                f"最强对手 IoU {candidate.strongest_competitor_iou:.4f}\n"
                f"平均对手 IoU {candidate.mean_competitor_iou:.4f}\n"
                f"领先值 {candidate.advantage:.4f}"
            )
            button.setHidden(False)
        self._cache_current_state()
        self._set_status(result.message)

    def _candidate_clicked(self) -> None:
        button = self.sender()
        if isinstance(button, QPushButton):
            frame_number = button.property("frameNumber")
            if isinstance(frame_number, int):
                self.frame_spin.setValue(frame_number)

    def add_current_to_visual(self) -> None:
        key = self._current_key()
        if key is None or self._session is None:
            self._set_status("请先选择可用序列和帧。")
            return
        try:
            updated = self._visual_selections.add(key, self.frame_spin.value())
        except VisualSelectionError as exc:
            self._set_status(str(exc))
            return
        if updated is self._visual_selections:
            self._set_status("该序列和帧号已在 Visual 清单中。")
            return
        self._visual_selections = updated
        self._refresh_visual_table()
        self._cache_current_state()
        self._set_status(f"已加入 {key.name} 第 {self.frame_spin.value()} 帧。")

    def _refresh_visual_table(self) -> None:
        self.visual_table.setRowCount(len(self._visual_selections.sequences))
        for row, selection in enumerate(self._visual_selections.sequences):
            values = (
                MODALITY_LABELS[selection.key.modality],
                selection.key.name,
                ", ".join(str(frame) for frame in selection.frame_numbers),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, selection.key)
                self.visual_table.setItem(row, column, item)
        self.draw_visual_button.setEnabled(self._visual_selections.is_complete)

    def _remove_selected_visual(self) -> None:
        row = self.visual_table.currentRow()
        if row < 0 or row >= len(self._visual_selections.sequences):
            self._set_status("请先选择要删除的序列。")
            return
        key = self._visual_selections.sequences[row].key
        self._visual_selections = self._visual_selections.remove_sequence(key)
        self._refresh_visual_table()
        self._cache_current_state()

    def _clear_visuals(self) -> None:
        self._visual_selections = self._visual_selections.clear()
        self._refresh_visual_table()
        self._cache_current_state()

    def export_current_frame_to(self, destination: Path) -> Path:
        if self._session is None:
            raise VisualizationExportError("当前没有可导出的序列帧")
        composition = self._session.compose(
            self.frame_spin.value(),
            visible_tracker_ids=self._checked_tracker_ids(),
        )
        return export_tracking_frame(
            composition,
            destination,
            options=self._render_options(),
        )

    def export_visual_to(self, destination: Path) -> Path:
        if self._data is None or not self._visual_selections.sequences:
            raise VisualizationExportError("Visual 绘制清单为空")
        if not self._visual_selections.is_complete:
            incomplete = [
                f"{selection.key.name}（{len(selection.frame_numbers)}/3）"
                for selection in self._visual_selections.sequences
                if len(selection.frame_numbers) != 3
            ]
            raise VisualizationExportError(
                f"每个序列必须选择 3 帧：{', '.join(incomplete)}"
            )
        sessions = {
            selection.key: self._data.load_session(selection.key)
            for selection in self._visual_selections.sequences
        }
        return export_tracking_montage(
            sessions,
            self._visual_selections,
            destination,
            visible_tracker_ids=self._checked_tracker_ids(),
        )

    def _default_export_directory(self) -> Path:
        if self._project is not None and self._project.sources.export_directory is not None:
            return self._project.sources.export_directory
        return Path.cwd()

    def _export_current_frame_interactive(self) -> None:
        key = self._current_key()
        if key is None:
            self._set_status("请先选择可用序列和帧。")
            return
        suggested = self._default_export_directory() / (
            f"{key.modality.value}-{key.name}-{self.frame_spin.value():04d}.png"
        )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前帧",
            str(suggested),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps)",
        )
        if not filename:
            return
        try:
            output = self.export_current_frame_to(Path(filename))
        except (VisualizationExportError, OSError, ValueError) as exc:
            self._set_status(str(exc))
            return
        self._set_status(f"当前帧已导出：{output}")

    def _export_visual_interactive(self) -> None:
        suggested = self._default_export_directory() / "tracking_visual.png"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "绘制 Visual",
            str(suggested),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps)",
        )
        if not filename:
            return
        try:
            output = self.export_visual_to(Path(filename))
        except (VisualizationExportError, OSError, ValueError) as exc:
            self._set_status(str(exc))
            return
        self._set_status(f"Visual 已导出：{output}")

    def _range_values(self) -> tuple[int, int]:
        start = self.range_start_spin.value()
        end = self.range_end_spin.value()
        if start > end:
            raise ValueError("起始帧不能大于结束帧")
        return start, end

    def export_range_to(
        self,
        directory: Path,
        *,
        format_name: str | None = None,
        reporter: ProgressReporter | None = None,
        token: CancelToken | None = None,
    ) -> RangeExportResult:
        if self._session is None:
            raise VisualizationExportError("当前没有可导出的序列")
        start, end = self._range_values()
        visible = self._checked_tracker_ids()
        return export_tracking_range(
            lambda frame: self._session.compose(frame, visible_tracker_ids=visible),
            directory,
            start_frame=start,
            end_frame=end,
            format_name=format_name or str(self.range_format_combo.currentData()),
            options=self._render_options(),
            on_progress=(
                None
                if reporter is None
                else lambda completed, total, message: reporter.notify(
                    round(completed * 100 / max(total, 1)), message
                )
            ),
            is_cancelled=(None if token is None else lambda: token.is_cancelled),
        )

    def export_video_to(
        self,
        destination: Path,
        *,
        reporter: ProgressReporter | None = None,
        token: CancelToken | None = None,
    ) -> Path:
        if self._session is None:
            raise VideoExportError("当前没有可导出的序列")
        start, end = self._range_values()
        resolution = self.resolution_combo.currentData()
        if not isinstance(resolution, tuple):
            resolution = (1200, 600)
        fps = float(self.speed_combo.currentData() or 10)
        return export_tracking_video(
            self._session,
            destination,
            start_frame=start,
            end_frame=end,
            fps=fps,
            resolution=resolution,
            visible_tracker_ids=self._checked_tracker_ids(),
            show_legend=self.legend_checkbox.isChecked(),
            on_progress=(
                None
                if reporter is None
                else lambda completed, total, message: reporter.notify(
                    round(completed * 100 / max(total, 1)), message
                )
            ),
            is_cancelled=(None if token is None else lambda: token.is_cancelled),
        )

    def _submit_export(self, name: str, handler: Any, *, kind: str) -> None:
        if self._task_runner is None:
            try:
                result = handler(None, None)
            except (OSError, ValueError) as exc:
                self._set_status(str(exc))
            else:
                if isinstance(result, RangeExportResult):
                    self._set_status(f"区间导出完成 {len(result.outputs)} 帧。")
                else:
                    self._set_status(f"导出完成：{result}")
            return
        self.export_range_button.setEnabled(False)
        self.export_video_button.setEnabled(False)
        self._active_export_kind = kind
        self._active_export_task = self._task_runner.submit(name, handler)

    def _export_range_interactive(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择区间导出目录",
            str(self._default_export_directory()),
        )
        if not directory:
            return
        self._submit_export(
            "导出跟踪帧区间",
            lambda reporter, token: self.export_range_to(
                Path(directory), reporter=reporter, token=token
            ),
            kind="range",
        )

    def _export_video_interactive(self) -> None:
        key = self._current_key()
        if key is None:
            self._set_status("请先选择可用序列。")
            return
        suggested = self._default_export_directory() / (
            f"{key.modality.value}-{key.name}.mp4"
        )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出 MP4",
            str(suggested),
            "MP4 视频 (*.mp4)",
        )
        if not filename:
            return
        self._submit_export(
            "导出跟踪视频",
            lambda reporter, token: self.export_video_to(
                Path(filename), reporter=reporter, token=token
            ),
            kind="video",
        )

    def current_recipe(self) -> VisualizationRecipe:
        key = self._current_key()
        candidate_frames = tuple(
            int(frame)
            for button in self.candidate_buttons
            if isinstance((frame := button.property("frameNumber")), int)
        )
        return VisualizationRecipe(
            modality=key.modality if key is not None else None,
            sequence_name=key.name if key is not None else None,
            primary_tracker_id=(
                str(self.primary_combo.currentData())
                if self.primary_combo.currentData() is not None
                else None
            ),
            tracker_ids=self._checked_tracker_id_order(),
            show_ground_truth=self.ground_truth_checkbox.isChecked(),
            show_legend=self.legend_checkbox.isChecked(),
            show_names=self.names_checkbox.isChecked(),
            show_frame_number=self.frame_number_checkbox.isChecked(),
            play_start=self.range_start_spin.value(),
            play_end=self.range_end_spin.value(),
            play_fps=float(self.speed_combo.currentData() or 10),
            loop=self.loop_checkbox.isChecked(),
            video_resolution=self.resolution_combo.currentData() or (1200, 600),
            candidate_frames=candidate_frames,
            visual_selections=self._visual_selections,
        )

    def save_recipe_to(self, destination: Path) -> Path:
        return save_visualization_recipe(self.current_recipe(), destination)

    def load_recipe_from(self, source: Path) -> None:
        if self._data is None:
            raise VisualizationRecipeFormatError("当前项目没有可映射的数据")
        recipe = load_visualization_recipe(source)
        self._apply_recipe(recipe)
        self._cache_current_state()

    def _apply_recipe(
        self, recipe: VisualizationRecipe, *, announce: bool = True
    ) -> None:
        if self._data is None:
            raise VisualizationRecipeFormatError("当前项目没有可映射的数据")
        references = validate_visualization_recipe_references(
            recipe,
            available_tracker_ids=frozenset(item.tracker_id for item in self._data.trackers),
            available_sequences=frozenset(self._data.sequences),
        )
        if recipe.modality is not None:
            index = self.modality_combo.findData(recipe.modality.value)
            if index >= 0:
                self.modality_combo.setCurrentIndex(index)
        if recipe.sequence_name:
            index = self.sequence_combo.findText(recipe.sequence_name)
            if index >= 0:
                self.sequence_combo.setCurrentIndex(index)
        primary_index = self.primary_combo.findData(recipe.primary_tracker_id)
        if primary_index >= 0:
            self.primary_combo.setCurrentIndex(primary_index)
        wanted = set(recipe.tracker_ids)
        for row in range(self.tracker_list.count()):
            item = self.tracker_list.item(row)
            item.setCheckState(
                Qt.CheckState.Checked
                if item.data(Qt.ItemDataRole.UserRole) in wanted
                else Qt.CheckState.Unchecked
            )
        self.ground_truth_checkbox.setChecked(recipe.show_ground_truth)
        self.legend_checkbox.setChecked(recipe.show_legend)
        self.names_checkbox.setChecked(recipe.show_names)
        self.frame_number_checkbox.setChecked(recipe.show_frame_number)
        self.range_start_spin.setValue(min(recipe.play_start, self.range_start_spin.maximum()))
        self.range_end_spin.setValue(min(recipe.play_end, self.range_end_spin.maximum()))
        speed_index = self.speed_combo.findData(round(recipe.play_fps))
        if speed_index >= 0:
            self.speed_combo.setCurrentIndex(speed_index)
        self.loop_checkbox.setChecked(recipe.loop)
        resolution_index = self.resolution_combo.findData(recipe.video_resolution)
        if resolution_index >= 0:
            self.resolution_combo.setCurrentIndex(resolution_index)
        self._visual_selections = VisualSelectionList(
            sequences=tuple(
                item
                for item in recipe.visual_selections.sequences
                if item.key in self._data.sequences
            )
        )
        self._refresh_visual_table()
        self._hide_candidates()
        for button, frame in zip(
            self.candidate_buttons,
            recipe.candidate_frames,
            strict=False,
        ):
            button.setText(f"#{frame:04d}")
            button.setProperty("frameNumber", frame)
            button.setHidden(False)
        details: list[str] = []
        if references.missing_tracker_ids:
            details.append("缺失跟踪器：" + ", ".join(references.missing_tracker_ids))
        if references.missing_sequences:
            details.append(
                "缺失序列：" + ", ".join(key.identifier for key in references.missing_sequences)
            )
        if announce:
            self._set_status("配置已载入。" + (" " + "；".join(details) if details else ""))

    def _save_recipe_interactive(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出可视化配置",
            str(self._default_export_directory() / "tracking_visual.hsirecipe.json"),
            "HSI 配方 (*.hsirecipe.json);;JSON (*.json)",
        )
        if not filename:
            return
        try:
            output = self.save_recipe_to(Path(filename))
        except VisualizationRecipeFormatError as exc:
            self._set_status(str(exc))
            return
        self._set_status(f"配置已导出：{output}")

    def _load_recipe_interactive(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "导入可视化配置",
            str(self._default_export_directory()),
            "HSI 配方 (*.hsirecipe.json);;JSON (*.json)",
        )
        if not filename:
            return
        try:
            self.load_recipe_from(Path(filename))
        except (VisualizationRecipeFormatError, OSError, ValueError) as exc:
            self._set_status(str(exc))

