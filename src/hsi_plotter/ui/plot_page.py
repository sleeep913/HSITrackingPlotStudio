from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hsi_plotter.io.curve_data_reader import CurveDataError, read_curve_mat
from hsi_plotter.io.recipe_store import load_curve_recipe, save_curve_recipe
from hsi_plotter.models.evaluation import EvaluationRun
from hsi_plotter.models.project import Project
from hsi_plotter.models.recipe import (
    CurveRecipe,
    RecipeCompatibilityError,
    RecipeModalityMetrics,
    RecipeTracker,
    resolve_recipe_trackers,
)
from hsi_plotter.models.style import (
    PlotStylePreset,
    TrackerStyle,
    matlab_hot2023_plot_style,
    matlab_style_for_index,
    plot_style_preset,
)
from hsi_plotter.plotting.curve_plot import CurveSeries, build_curve_figure, series_from_recipe
from hsi_plotter.plotting.exporters import FigureExportError, export_curve_pair
from hsi_plotter.ui.color_picker import ColorPickerButton, ColorSwatchCell
from hsi_plotter.ui.interactive_legend_canvas import (
    InteractiveLegendCanvas,
    legend_scale_from_drag,
)

__all__ = ["InteractiveLegendCanvas", "PlotPage", "legend_scale_from_drag"]

PREVIEW_DPI = 140

DISPLAY_COLUMN = 0
RANK_COLUMN = 1
NAME_COLUMN = 2
AUC_COLUMN = 3
DP_COLUMN = 4
COLOR_COLUMN = 5
LINE_COLUMN = 6


class AspectRatioWidget(QWidget):
    """A non-stretching host that keeps the combined preview comfortably wide."""

    def __init__(self, ratio: float = 2.15) -> None:
        super().__init__()
        self._ratio = ratio
        self._content_count = 0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_content_count(self, count: int, legend_scale: float = 1.0) -> None:
        self._content_count = max(0, int(count))
        dense_items = max(0, self._content_count - 8)
        scale_height = max(0.0, float(legend_scale) - 1.0) * 0.24
        self._ratio = max(1.72, 2.15 - dense_items * 0.035 - scale_height)
        self.updateGeometry()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(340, round(width / self._ratio))

    def sizeHint(self) -> QSize:
        return QSize(980, self.heightForWidth(980))


class LineStyleComboBox(QComboBox):
    line_style_changed = Signal(str)

    _STYLES = (
        ("实线", "-", Qt.PenStyle.SolidLine),
        ("虚线", "--", Qt.PenStyle.DashLine),
        ("点划线", "-.", Qt.PenStyle.DashDotLine),
        ("点线", ":", Qt.PenStyle.DotLine),
    )

    def __init__(self, line_style: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        for label, value, pen_style in self._STYLES:
            self.addItem(self._line_icon(pen_style), label, value)
        index = self.findData(line_style)
        self.setCurrentIndex(index if index >= 0 else 0)
        self.currentIndexChanged.connect(
            lambda: self.line_style_changed.emit(str(self.currentData()))
        )

    @staticmethod
    def _line_icon(style: Qt.PenStyle) -> QIcon:
        pixmap = QPixmap(64, 18)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#2563EB"), 2.2, style))
        painter.drawLine(4, 9, 60, 9)
        painter.end()
        return QIcon(pixmap)


class PlotPage(QWidget):
    project_changed = Signal()
    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("plotPage")
        self._project: Project | None = None
        self._series: list[CurveSeries] = []
        self._recipe = CurveRecipe(style=matlab_hot2023_plot_style(combined=True))
        self._pending_recipe: CurveRecipe | None = None
        self._pending_mapping: dict[str, str] = {}
        self._pending_skipped: set[str] = set()
        self._pending_missing: list[str] = []
        self._updating = False
        self._preview_dirty = False
        self._last_preview_size = QSize()
        self._preview_resize_timer = QTimer(self)
        self._preview_resize_timer.setSingleShot(True)
        self._preview_resize_timer.setInterval(120)
        self._preview_resize_timer.timeout.connect(self._refresh_preview_after_resize)
        self._build_ui()
        self._set_enabled(False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 20)
        root.setSpacing(12)
        header = QHBoxLayout()
        headings = QVBoxLayout()
        heading = QLabel("曲线绘制")
        heading.setProperty("role", "heading")
        headings.addWidget(heading)
        caption = QLabel("直接使用指标评估结果生成可复现的论文曲线。")
        caption.setProperty("role", "caption")
        headings.addWidget(caption)
        header.addLayout(headings, 1)
        self.import_mat_button = QPushButton("加入跟踪器…")
        self.remove_tracker_button = QPushButton("移除选中")
        self.load_recipe_button = QPushButton("导入配置…")
        self.save_recipe_button = QPushButton("导出配置…")
        self.export_button = QPushButton("保存两张图…")
        self.export_button.setProperty("primary", True)
        header.addWidget(self.import_mat_button)
        header.addWidget(self.remove_tracker_button)
        header.addWidget(self.load_recipe_button)
        header.addWidget(self.save_recipe_button)
        header.addWidget(self.export_button)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        editor = QFrame()
        editor.setObjectName("sourceCard")
        editor.setMinimumWidth(500)
        editor.setMaximumWidth(680)
        editor_layout = QVBoxLayout(editor)
        self.tracker_table = QTableWidget(0, 7)
        self.tracker_table.setObjectName("plotTrackerTable")
        self.tracker_table.setHorizontalHeaderLabels(
            ["显示", "排名", "名称", "AUC", "DP", "颜色", "线型"]
        )
        self.tracker_table.verticalHeader().setVisible(False)
        self.tracker_table.setAlternatingRowColors(True)
        self.tracker_table.setDragDropMode(QTableWidget.DragDropMode.InternalMove)
        self.tracker_table.setDefaultDropAction(Qt.DropAction.MoveAction)
        editor_layout.addWidget(self.tracker_table, 1)
        table_action_row = QHBoxLayout()
        self.select_all_button = QPushButton("全选")
        self.refresh_button = QPushButton("刷新预览")
        self.refresh_button.setProperty("primary", True)
        table_action_row.addWidget(self.select_all_button)
        table_action_row.addStretch(1)
        table_action_row.addWidget(self.refresh_button)
        editor_layout.addLayout(table_action_row)
        rank_row = QHBoxLayout()
        rank_row.addWidget(QLabel("跟踪器排名"))
        self.table_rank_combo = QComboBox()
        self.table_rank_combo.addItem("按 AUC", "auc")
        self.table_rank_combo.addItem("按 DP", "dp")
        self.table_rank_combo.addItem("当前顺序", "manual")
        rank_row.addWidget(self.table_rank_combo, 1)
        editor_layout.addLayout(rank_row)
        sort_row = QHBoxLayout()
        self.auc_sort_combo = QComboBox()
        self.dp_sort_combo = QComboBox()
        for combo in (self.auc_sort_combo, self.dp_sort_combo):
            combo.addItem("不排序", "none")
            combo.addItem("从高到低", "desc")
            combo.addItem("从低到高", "asc")
            combo.setCurrentIndex(1)
        sort_row.addWidget(QLabel("AUC 图例"))
        sort_row.addWidget(self.auc_sort_combo)
        sort_row.addWidget(QLabel("DP 图例"))
        sort_row.addWidget(self.dp_sort_combo)
        editor_layout.addLayout(sort_row)
        self.sort_combo = QComboBox()
        self.sort_combo.setVisible(False)
        self.follow_sort_checkbox = QCheckBox("另一张图跟随指标排序")
        self.follow_sort_checkbox.setChecked(True)
        self.follow_sort_checkbox.setVisible(False)

        self.advanced_panel = QWidget()
        form = QFormLayout(self.advanced_panel)
        self.preset_combo = QComboBox()
        for label, value in (
            ("论文单栏", PlotStylePreset.PAPER_SINGLE_COLUMN),
            ("论文双栏", PlotStylePreset.PAPER_DOUBLE_COLUMN),
            ("演示文稿", PlotStylePreset.PRESENTATION),
        ):
            self.preset_combo.addItem(label, value)
        self.preset_combo.setCurrentIndex(1)
        form.addRow("尺寸预设", self.preset_combo)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Precision + Success", "both")
        self.mode_combo.addItem("仅 Precision", "precision")
        self.mode_combo.addItem("仅 Success", "success")
        form.addRow("图形内容", self.mode_combo)
        self.font_edit = QLineEdit("Times New Roman")
        self.precision_title_edit = QLineEdit("Precision plots of OPE")
        self.success_title_edit = QLineEdit("Success plots of OPE")
        form.addRow("字体", self.font_edit)
        form.addRow("Precision 标题", self.precision_title_edit)
        form.addRow("Success 标题", self.success_title_edit)
        self.precision_x_label_edit = QLineEdit("Location error threshold")
        self.precision_y_label_edit = QLineEdit("Precision")
        self.success_x_label_edit = QLineEdit("Overlap threshold")
        self.success_y_label_edit = QLineEdit("Success rate")
        form.addRow("Precision 横轴", self.precision_x_label_edit)
        form.addRow("Precision 纵轴", self.precision_y_label_edit)
        form.addRow("Success 横轴", self.success_x_label_edit)
        form.addRow("Success 纵轴", self.success_y_label_edit)
        self.width_spin = self._double_spin(2, 20, 7.2)
        self.height_spin = self._double_spin(2, 15, 3.2)
        size_widget = QWidget()
        size_layout = QHBoxLayout(size_widget)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(self.height_spin)
        form.addRow("画布（英寸）", size_widget)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        self.font_size_spin = self._double_spin(5, 40, 9)
        self.title_size_spin = self._double_spin(5, 48, 10)
        self.legend_size_spin = self._double_spin(5, 40, 8)
        self.decimal_spin = QSpinBox()
        self.decimal_spin.setRange(0, 8)
        self.decimal_spin.setValue(4)
        self.legend_columns_spin = QSpinBox()
        self.legend_columns_spin.setRange(1, 10)
        self.legend_location_combo = QComboBox()
        for label, value in (
            ("自动", "best"),
            ("左下", "lower left"),
            ("右下", "lower right"),
            ("左上", "upper left"),
            ("右上", "upper right"),
        ):
            self.legend_location_combo.addItem(label, value)
        self.grid_checkbox = QCheckBox("显示网格")
        self.grid_checkbox.setChecked(True)
        self.marker_combo = QComboBox()
        for label, value in (("无", ""), ("圆点", "o"), ("方块", "s"), ("三角", "^")):
            self.marker_combo.addItem(label, value)
        self.text_color_edit = QLineEdit("#111827")
        self.background_color_edit = QLineEdit("#FFFFFF")
        self.axes_color_edit = QLineEdit("#334155")
        self.grid_color_edit = QLineEdit("#CBD5E1")
        self.axes_width_spin = self._double_spin(0.1, 5, 0.8)
        self.grid_alpha_spin = self._double_spin(0, 1, 0.45)
        form.addRow("DPI", self.dpi_spin)
        form.addRow("正文字号", self.font_size_spin)
        form.addRow("标题字号", self.title_size_spin)
        form.addRow("图例字号", self.legend_size_spin)
        form.addRow("图例小数位", self.decimal_spin)
        form.addRow("图例列数", self.legend_columns_spin)
        form.addRow("图例位置", self.legend_location_combo)
        form.addRow("网格", self.grid_checkbox)
        form.addRow("曲线标记", self.marker_combo)
        form.addRow("文字颜色", self.text_color_edit)
        form.addRow("背景颜色", self.background_color_edit)
        form.addRow("坐标轴颜色", self.axes_color_edit)
        form.addRow("网格颜色", self.grid_color_edit)
        form.addRow("坐标轴线宽", self.axes_width_spin)
        form.addRow("网格透明度", self.grid_alpha_spin)
        self.precision_max_spin = self._double_spin(1, 200, 50)
        self.success_max_spin = self._double_spin(0.1, 1, 1)
        self.y_max_spin = self._double_spin(0.1, 1.5, 1)
        self.precision_tick_spin = self._double_spin(0.1, 50, 10)
        self.success_tick_spin = self._double_spin(0.01, 1, 0.2)
        self.y_tick_spin = self._double_spin(0.01, 1, 0.2)
        form.addRow("Precision 横轴上限", self.precision_max_spin)
        form.addRow("Success 横轴上限", self.success_max_spin)
        form.addRow("纵轴上限", self.y_max_spin)
        form.addRow("Precision 刻度", self.precision_tick_spin)
        form.addRow("Success 刻度", self.success_tick_spin)
        form.addRow("纵轴刻度", self.y_tick_spin)
        self.margin_left_spin = self._double_spin(0, 0.9, 0.10)
        self.margin_right_spin = self._double_spin(0.1, 1, 0.97)
        self.margin_bottom_spin = self._double_spin(0, 0.9, 0.17)
        self.margin_top_spin = self._double_spin(0.1, 1, 0.88)
        form.addRow("左边距", self.margin_left_spin)
        form.addRow("右边距", self.margin_right_spin)
        form.addRow("下边距", self.margin_bottom_spin)
        form.addRow("上边距", self.margin_top_spin)
        editor_layout.addWidget(self.advanced_panel)
        self.advanced_panel.setVisible(False)
        self.reset_panel = QWidget()
        reset_row = QHBoxLayout(self.reset_panel)
        self.project_default_button = QPushButton("恢复项目默认")
        self.software_default_button = QPushButton("恢复软件默认")
        reset_row.addWidget(self.project_default_button)
        reset_row.addWidget(self.software_default_button)
        editor_layout.addWidget(self.reset_panel)
        self.reset_panel.setVisible(False)
        editor_scroll = QScrollArea()
        editor_scroll.setObjectName("plotEditorScroll")
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setFrameShape(QFrame.Shape.NoFrame)
        editor_scroll.setWidget(editor)
        splitter.addWidget(editor_scroll)

        preview = QFrame()
        preview.setObjectName("preflightCard")
        self.preview_layout = QVBoxLayout(preview)
        self.preview_plot_host = AspectRatioWidget()
        self.preview_plot_host.setObjectName("curvePreviewHost")
        self.preview_plot_layout = QVBoxLayout(self.preview_plot_host)
        self.preview_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_canvas = FigureCanvasQTAgg(Figure(figsize=(6, 3)))
        self.preview_canvas.setObjectName("curvePreviewCanvas")
        self.preview_plot_layout.addWidget(self.preview_canvas)
        self.preview_layout.addWidget(
            self.preview_plot_host,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        self.status_label = QLabel("请先在“指标评估”页面完成计算。")
        self.status_label.setProperty("role", "caption")
        self.status_label.setWordWrap(True)
        self.preview_layout.addWidget(self.status_label)
        self.mapping_panel = QWidget()
        mapping_layout = QHBoxLayout(self.mapping_panel)
        mapping_layout.setContentsMargins(0, 0, 0, 0)
        mapping_layout.addWidget(QLabel("缺失跟踪器处理"))
        self.mapping_combo = QComboBox()
        mapping_layout.addWidget(self.mapping_combo, 1)
        self.map_button = QPushButton("映射")
        self.skip_button = QPushButton("跳过")
        self.cancel_mapping_button = QPushButton("取消应用")
        mapping_layout.addWidget(self.map_button)
        mapping_layout.addWidget(self.skip_button)
        mapping_layout.addWidget(self.cancel_mapping_button)
        self.mapping_panel.setVisible(False)
        self.preview_layout.addWidget(self.mapping_panel)
        self.preview_layout.addStretch(1)
        splitter.addWidget(preview)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([590, 900])
        root.addWidget(splitter, 1)
        self._connect_signals()

    @staticmethod
    def _double_spin(minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(0.1)
        spin.setDecimals(2)
        spin.setValue(value)
        return spin

    def _connect_signals(self) -> None:
        self.tracker_table.itemChanged.connect(self._on_tracker_changed)
        self.tracker_table.model().rowsMoved.connect(lambda: self._sync_order())
        self.select_all_button.clicked.connect(self._select_all_trackers)
        self.refresh_button.clicked.connect(self._refresh_from_controls)
        self.table_rank_combo.currentIndexChanged.connect(self._on_table_rank_changed)
        self.auc_sort_combo.currentIndexChanged.connect(lambda: self._apply_sort())
        self.dp_sort_combo.currentIndexChanged.connect(lambda: self._apply_sort())
        self.follow_sort_checkbox.toggled.connect(lambda: self._apply_sort())
        self.preset_combo.currentIndexChanged.connect(lambda: self._apply_preset())
        for combo in (self.mode_combo, self.legend_location_combo, self.marker_combo):
            combo.currentIndexChanged.connect(lambda: self._controls_changed())
        for spin in (
            self.width_spin,
            self.height_spin,
            self.dpi_spin,
            self.font_size_spin,
            self.title_size_spin,
            self.legend_size_spin,
            self.decimal_spin,
            self.legend_columns_spin,
            self.axes_width_spin,
            self.grid_alpha_spin,
            self.precision_max_spin,
            self.success_max_spin,
            self.y_max_spin,
            self.precision_tick_spin,
            self.success_tick_spin,
            self.y_tick_spin,
            self.margin_left_spin,
            self.margin_right_spin,
            self.margin_bottom_spin,
            self.margin_top_spin,
        ):
            spin.valueChanged.connect(lambda: self._controls_changed())
        for edit in (
            self.font_edit,
            self.precision_title_edit,
            self.success_title_edit,
            self.precision_x_label_edit,
            self.precision_y_label_edit,
            self.success_x_label_edit,
            self.success_y_label_edit,
            self.text_color_edit,
            self.background_color_edit,
            self.axes_color_edit,
            self.grid_color_edit,
        ):
            edit.editingFinished.connect(self._controls_changed)
        self.grid_checkbox.toggled.connect(lambda: self._controls_changed())
        self.project_default_button.clicked.connect(lambda: self._restore_project_default())
        self.software_default_button.clicked.connect(lambda: self._restore_software_default())
        self.save_recipe_button.clicked.connect(lambda: self._save_recipe())
        self.load_recipe_button.clicked.connect(lambda: self._load_recipe())
        self.import_mat_button.clicked.connect(lambda: self._import_mat())
        self.remove_tracker_button.clicked.connect(lambda: self._remove_selected())
        self.export_button.clicked.connect(lambda: self._export())
        self.map_button.clicked.connect(lambda: self._resolve_next_missing(skip=False))
        self.skip_button.clicked.connect(lambda: self._resolve_next_missing(skip=True))
        self.cancel_mapping_button.clicked.connect(self._cancel_pending_recipe)

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (
            self.tracker_table,
            self.remove_tracker_button,
            self.select_all_button,
            self.sort_combo,
            self.table_rank_combo,
            self.auc_sort_combo,
            self.dp_sort_combo,
            self.follow_sort_checkbox,
            self.preset_combo,
            self.mode_combo,
            self.font_edit,
            self.precision_title_edit,
            self.success_title_edit,
            self.width_spin,
            self.height_spin,
            self.dpi_spin,
            self.font_size_spin,
            self.title_size_spin,
            self.legend_size_spin,
            self.decimal_spin,
            self.legend_columns_spin,
            self.legend_location_combo,
            self.grid_checkbox,
            self.marker_combo,
            self.text_color_edit,
            self.background_color_edit,
            self.axes_color_edit,
            self.grid_color_edit,
            self.axes_width_spin,
            self.grid_alpha_spin,
            self.precision_max_spin,
            self.success_max_spin,
            self.y_max_spin,
            self.precision_tick_spin,
            self.success_tick_spin,
            self.y_tick_spin,
            self.margin_left_spin,
            self.margin_right_spin,
            self.margin_bottom_spin,
            self.margin_top_spin,
            self.project_default_button,
            self.software_default_button,
            self.save_recipe_button,
            self.export_button,
        ):
            widget.setEnabled(enabled)
        self.refresh_button.setEnabled(enabled and self._preview_dirty)

    def set_project(self, project: Project | None) -> None:
        self._project = project
        self.import_mat_button.setEnabled(project is not None)
        if project is None:
            self._series.clear()
            self.tracker_table.setRowCount(0)
            self._set_enabled(False)
            self.status_label.setText("请先创建或打开项目。")
            return
        self._recipe = replace(
            self._recipe,
            style=replace(
                matlab_hot2023_plot_style(combined=True),
                legend_scale=project.plot_style.legend_scale,
            ),
        )
        self._load_controls()

    def set_evaluation_run(self, run: object) -> None:
        if not isinstance(run, EvaluationRun):
            return
        self._series = []
        recipe_entries: list[RecipeTracker] = []
        for index, tracker in enumerate(run.trackers):
            if tracker.overall is None:
                continue
            series = CurveSeries(
                tracker.tracker_id,
                tracker.tracker_name,
                tracker.overall.auc,
                tracker.overall.dp,
                tracker.overall.precision_curve,
                tracker.overall.success_curve,
                matlab_style_for_index(index),
            )
            self._series.append(series)
            recipe_entries.append(
                self._embedded_entry(
                    series,
                    visible=True,
                    order=index,
                    modality_metrics=tuple(
                        RecipeModalityMetrics(modality.value, metrics.auc, metrics.dp)
                        for modality, metrics in tracker.modalities.items()
                    ),
                )
            )
        self._recipe = replace(
            self._recipe,
            trackers=tuple(recipe_entries),
            precision_order=(),
            success_order=(),
        )
        self._fill_table()
        self._set_enabled(bool(self._series))
        if self._series:
            self._apply_sort(refresh=True)
        else:
            self.status_label.setText("本次评估没有可绘制的成功结果。")

    def clear_results(self) -> None:
        self._series = []
        self._recipe = replace(
            self._recipe,
            trackers=(),
            precision_order=(),
            success_order=(),
        )
        self._fill_table()
        self._set_enabled(False)
        self.status_label.setText("尚未载入可绘制结果。")

    def _fill_table(self) -> None:
        self._updating = True
        entries = {item.tracker_id: item for item in self._recipe.trackers}
        manual_order = sorted(
            self._series,
            key=lambda item: (
                entries.get(
                    item.tracker_id,
                    RecipeTracker(item.tracker_id, item.display_name, True, 9999),
                ).order
            ),
        )
        rank_metric = self._recipe.table_rank_metric
        if rank_metric in {"auc", "dp"}:
            positions = {item.tracker_id: index for index, item in enumerate(manual_order)}
            ordered = sorted(
                manual_order,
                key=lambda item: (-getattr(item, rank_metric), positions[item.tracker_id]),
            )
        else:
            ordered = manual_order
        self.tracker_table.setRowCount(len(ordered))
        for row, series in enumerate(ordered):
            entry = entries.get(
                series.tracker_id,
                RecipeTracker(series.tracker_id, series.display_name, True, row),
            )
            visible = QTableWidgetItem()
            visible.setFlags(visible.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            visible.setCheckState(
                Qt.CheckState.Checked if entry.visible else Qt.CheckState.Unchecked
            )
            visible.setData(Qt.ItemDataRole.UserRole, series.tracker_id)
            items = [
                visible,
                QTableWidgetItem(str(row + 1)),
                QTableWidgetItem(entry.display_name),
                QTableWidgetItem(f"{series.auc:.4f}"),
                QTableWidgetItem(f"{series.dp:.4f}"),
            ]
            for column, item in enumerate(items):
                if column in (RANK_COLUMN, AUC_COLUMN, DP_COLUMN):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tracker_table.setItem(row, column, item)
            color_cell = ColorSwatchCell(series.style.color)
            color_button = color_cell.color_button
            color_button.color_changed.connect(lambda _value: self._on_style_widget_changed())
            self.tracker_table.setCellWidget(row, COLOR_COLUMN, color_cell)
            line_combo = LineStyleComboBox(series.style.line_style)
            line_combo.line_style_changed.connect(
                lambda _value: self._on_style_widget_changed()
            )
            self.tracker_table.setCellWidget(row, LINE_COLUMN, line_combo)
        self.tracker_table.resizeColumnsToContents()
        self.tracker_table.setColumnWidth(DISPLAY_COLUMN, 52)
        self.tracker_table.setColumnWidth(RANK_COLUMN, 52)
        self.tracker_table.setColumnWidth(NAME_COLUMN, 130)
        self.tracker_table.setColumnWidth(COLOR_COLUMN, 64)
        self.tracker_table.setColumnWidth(LINE_COLUMN, 112)
        self._updating = False

    def _on_tracker_changed(self, _item: QTableWidgetItem) -> None:
        if self._updating:
            return
        entries: list[RecipeTracker] = []
        styles: dict[str, TrackerStyle] = {}
        previous = {item.tracker_id: item for item in self._recipe.trackers}
        series_map = {item.tracker_id: item for item in self._series}
        for row in range(self.tracker_table.rowCount()):
            identifier = self.tracker_table.item(row, DISPLAY_COLUMN).data(
                Qt.ItemDataRole.UserRole
            )
            color_cell = self.tracker_table.cellWidget(row, COLOR_COLUMN)
            color_button = (
                color_cell.findChild(ColorPickerButton) if color_cell is not None else None
            )
            line_combo = self.tracker_table.cellWidget(row, LINE_COLUMN)
            if not isinstance(color_button, ColorPickerButton) or not isinstance(
                line_combo, LineStyleComboBox
            ):
                continue
            style = TrackerStyle(
                color_button.color(),
                str(line_combo.currentData()),
                2.0,
            )
            styles[identifier] = style
            source = replace(series_map[identifier], style=style)
            entries.append(
                self._embedded_entry(
                    source,
                    display_name=self.tracker_table.item(row, NAME_COLUMN).text().strip()
                    or identifier,
                    visible=self.tracker_table.item(row, DISPLAY_COLUMN).checkState()
                    == Qt.CheckState.Checked,
                    order=(
                        row
                        if self._recipe.table_rank_metric == "manual"
                        else previous.get(identifier, RecipeTracker(identifier, identifier)).order
                    ),
                    modality_metrics=previous.get(
                        identifier,
                        RecipeTracker(identifier, identifier),
                    ).modality_metrics,
                )
            )
        self._recipe = replace(self._recipe, trackers=tuple(entries))
        self._series = [
            replace(item, style=styles.get(item.tracker_id, item.style)) for item in self._series
        ]
        self._persist_trackers(entries, styles)
        self._mark_preview_dirty()

    def _on_style_widget_changed(self) -> None:
        if self._updating or not self.tracker_table.rowCount():
            return
        self._on_tracker_changed(self.tracker_table.item(0, 0))

    def _select_all_trackers(self) -> None:
        if self._updating or not self.tracker_table.rowCount():
            return
        all_checked = all(
            self.tracker_table.item(row, DISPLAY_COLUMN).checkState()
            == Qt.CheckState.Checked
            for row in range(self.tracker_table.rowCount())
        )
        target_state = (
            Qt.CheckState.Unchecked if all_checked else Qt.CheckState.Checked
        )
        changed = False
        self._updating = True
        try:
            for row in range(self.tracker_table.rowCount()):
                item = self.tracker_table.item(row, DISPLAY_COLUMN)
                if item.checkState() != target_state:
                    item.setCheckState(target_state)
                    changed = True
        finally:
            self._updating = False
        if changed:
            self._on_tracker_changed(self.tracker_table.item(0, DISPLAY_COLUMN))

    def _mark_preview_dirty(self) -> None:
        if not self._series:
            return
        self._preview_dirty = True
        self.refresh_button.setEnabled(True)
        self.status_label.setText("设置已修改，点击“刷新预览”应用到两张图。")

    def _refresh_from_controls(self) -> None:
        if not self._series:
            return
        self._apply_sort(refresh=True)

    def _persist_trackers(
        self, entries: list[RecipeTracker], styles: dict[str, TrackerStyle]
    ) -> None:
        if self._project is None:
            return
        entry_map = {item.tracker_id: item for item in entries}
        order = {item.tracker_id: item.order for item in entries}
        self._project.trackers = sorted(
            [
                replace(
                    tracker,
                    display_name=(
                        entry_map[tracker.tracker_id].display_name
                        if tracker.tracker_id in entry_map
                        else tracker.display_name
                    ),
                    style=styles.get(tracker.tracker_id, tracker.style),
                )
                for tracker in self._project.trackers
            ],
            key=lambda tracker: order.get(tracker.tracker_id, 9999),
        )
        self.project_changed.emit()

    def _sync_order(self) -> None:
        if not self._updating and self.tracker_table.rowCount():
            self._recipe = replace(self._recipe, table_rank_metric="manual")
            self._updating = True
            self.table_rank_combo.setCurrentIndex(self.table_rank_combo.findData("manual"))
            self._updating = False
            self._on_tracker_changed(self.tracker_table.item(0, 0))

    def _on_table_rank_changed(self) -> None:
        if self._updating:
            return
        metric = str(self.table_rank_combo.currentData())
        self._recipe = replace(self._recipe, table_rank_metric=metric)
        self._fill_table()

    @staticmethod
    def _embedded_entry(
        series: CurveSeries,
        *,
        display_name: str | None = None,
        visible: bool,
        order: int,
        modality_metrics: tuple[RecipeModalityMetrics, ...] = (),
    ) -> RecipeTracker:
        return RecipeTracker(
            tracker_id=series.tracker_id,
            display_name=display_name or series.display_name,
            visible=visible,
            order=order,
            style=series.style,
            auc=series.auc,
            dp=series.dp,
            precision_curve=tuple(float(value) for value in series.precision_curve),
            success_curve=tuple(float(value) for value in series.success_curve),
            modality_metrics=modality_metrics,
        )

    def _apply_sort(self, *, refresh: bool = False) -> None:
        if self._updating:
            return
        manual_ids = tuple(
            entry.tracker_id for entry in sorted(self._recipe.trackers, key=lambda item: item.order)
        )
        auc_mode = str(self.auc_sort_combo.currentData())
        dp_mode = str(self.dp_sort_combo.currentData())

        def metric_order(metric: str, mode: str) -> tuple[str, ...]:
            if mode == "none":
                return ()
            return tuple(
                item.tracker_id
                for item in sorted(
                    self._series,
                    key=lambda item: getattr(item, metric),
                    reverse=mode == "desc",
                )
            )

        success_order = metric_order("auc", auc_mode)
        precision_order = metric_order("dp", dp_mode)
        if not success_order and precision_order:
            success_order = precision_order
        elif not precision_order and success_order:
            precision_order = success_order
        elif not success_order and not precision_order:
            success_order = precision_order = manual_ids
        self._recipe = replace(
            self._recipe,
            precision_order=precision_order,
            success_order=success_order,
            follow_metric_sort=auc_mode == "none" or dp_mode == "none",
        )
        if refresh:
            self.refresh_preview()
        else:
            self._mark_preview_dirty()

    def _apply_preset(self) -> None:
        if self._updating:
            return
        self._recipe = replace(
            self._recipe, style=plot_style_preset(self.preset_combo.currentData())
        )
        self._load_controls()
        self._persist_style()
        self.refresh_preview()

    def _controls_changed(self) -> None:
        if self._updating:
            return
        style = replace(
            self._recipe.style,
            font_family=self.font_edit.text().strip() or "Times New Roman",
            width_inches=self.width_spin.value(),
            height_inches=self.height_spin.value(),
            dpi=self.dpi_spin.value(),
            font_size=self.font_size_spin.value(),
            title_size=self.title_size_spin.value(),
            legend_size=self.legend_size_spin.value(),
            legend_columns=self.legend_columns_spin.value(),
            legend_location=self.legend_location_combo.currentData(),
            grid_enabled=self.grid_checkbox.isChecked(),
            marker=self.marker_combo.currentData(),
            text_color=self.text_color_edit.text().strip(),
            background_color=self.background_color_edit.text().strip(),
            axes_color=self.axes_color_edit.text().strip(),
            grid_color=self.grid_color_edit.text().strip(),
            axes_line_width=self.axes_width_spin.value(),
            grid_alpha=self.grid_alpha_spin.value(),
            margin_left=self.margin_left_spin.value(),
            margin_right=self.margin_right_spin.value(),
            margin_bottom=self.margin_bottom_spin.value(),
            margin_top=self.margin_top_spin.value(),
        )
        self._recipe = replace(
            self._recipe,
            plot_mode=self.mode_combo.currentData(),
            style=style,
            precision_title=self.precision_title_edit.text(),
            success_title=self.success_title_edit.text(),
            precision_x_label=self.precision_x_label_edit.text(),
            precision_y_label=self.precision_y_label_edit.text(),
            success_x_label=self.success_x_label_edit.text(),
            success_y_label=self.success_y_label_edit.text(),
            precision_x_max=self.precision_max_spin.value(),
            success_x_max=self.success_max_spin.value(),
            y_max=self.y_max_spin.value(),
            precision_tick_step=self.precision_tick_spin.value(),
            success_tick_step=self.success_tick_spin.value(),
            y_tick_step=self.y_tick_spin.value(),
            decimal_places=self.decimal_spin.value(),
        )
        self._persist_style()
        self.refresh_preview()

    def _load_controls(self) -> None:
        self._updating = True
        style = self._recipe.style
        self.preset_combo.setCurrentIndex(self.preset_combo.findData(style.preset))
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(self._recipe.plot_mode))
        self.font_edit.setText(style.font_family)
        self.precision_title_edit.setText(self._recipe.precision_title)
        self.success_title_edit.setText(self._recipe.success_title)
        self.precision_x_label_edit.setText(self._recipe.precision_x_label)
        self.precision_y_label_edit.setText(self._recipe.precision_y_label)
        self.success_x_label_edit.setText(self._recipe.success_x_label)
        self.success_y_label_edit.setText(self._recipe.success_y_label)
        self.width_spin.setValue(style.width_inches)
        self.height_spin.setValue(style.height_inches)
        self.dpi_spin.setValue(style.dpi)
        self.font_size_spin.setValue(style.font_size)
        self.title_size_spin.setValue(style.title_size)
        self.legend_size_spin.setValue(style.legend_size)
        self.decimal_spin.setValue(self._recipe.decimal_places)
        self.legend_columns_spin.setValue(style.legend_columns)
        self.legend_location_combo.setCurrentIndex(
            self.legend_location_combo.findData(style.legend_location)
        )
        self.grid_checkbox.setChecked(style.grid_enabled)
        self.follow_sort_checkbox.setChecked(self._recipe.follow_metric_sort)
        self.table_rank_combo.setCurrentIndex(
            self.table_rank_combo.findData(self._recipe.table_rank_metric)
        )
        self.marker_combo.setCurrentIndex(self.marker_combo.findData(style.marker))
        self.text_color_edit.setText(style.text_color)
        self.background_color_edit.setText(style.background_color)
        self.axes_color_edit.setText(style.axes_color)
        self.grid_color_edit.setText(style.grid_color)
        self.axes_width_spin.setValue(style.axes_line_width)
        self.grid_alpha_spin.setValue(style.grid_alpha)
        self.precision_max_spin.setValue(self._recipe.precision_x_max)
        self.success_max_spin.setValue(self._recipe.success_x_max)
        self.y_max_spin.setValue(self._recipe.y_max)
        self.precision_tick_spin.setValue(self._recipe.precision_tick_step)
        self.success_tick_spin.setValue(self._recipe.success_tick_step)
        self.y_tick_spin.setValue(self._recipe.y_tick_step)
        self.margin_left_spin.setValue(style.margin_left)
        self.margin_right_spin.setValue(style.margin_right)
        self.margin_bottom_spin.setValue(style.margin_bottom)
        self.margin_top_spin.setValue(style.margin_top)
        self._updating = False

    def _persist_style(self) -> None:
        if self._project is not None and self._project.plot_style != self._recipe.style:
            self._project.plot_style = self._recipe.style
            self.project_changed.emit()

    def _restore_project_default(self) -> None:
        if self._project is not None:
            self._recipe = replace(self._recipe, style=self._project.plot_style)
            self._load_controls()
            self.refresh_preview()

    def _restore_software_default(self) -> None:
        self._recipe = replace(
            self._recipe, style=plot_style_preset(PlotStylePreset.PAPER_DOUBLE_COLUMN)
        )
        self._load_controls()
        self._persist_style()
        self.refresh_preview()

    def refresh_preview(self) -> None:
        if not self._series:
            return
        visible_count = sum(item.visible for item in self._recipe.trackers)
        self.preview_plot_host.set_content_count(
            visible_count, self._recipe.style.legend_scale
        )
        self._update_preview_height()
        viewport_width = max(self.preview_plot_host.width(), 980)
        viewport_height = max(self.preview_plot_host.height(), 420)
        try:
            figure, diagnostics = build_curve_figure(
                self._series,
                self._recipe,
                figure_size_inches=(
                    viewport_width / PREVIEW_DPI,
                    viewport_height / PREVIEW_DPI,
                ),
                render_dpi=PREVIEW_DPI,
            )
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        old = self.preview_canvas
        self.preview_plot_layout.removeWidget(old)
        old.setParent(None)
        old.deleteLater()
        self.preview_canvas = InteractiveLegendCanvas(
            figure, legend_scale=self._recipe.style.legend_scale
        )
        self.preview_canvas.legend_scale_requested.connect(self._set_legend_scale)
        self.preview_canvas.setObjectName("curvePreviewCanvas")
        self.preview_plot_layout.addWidget(self.preview_canvas)
        self._last_preview_size = QSize(viewport_width, viewport_height)
        QTimer.singleShot(0, self._update_preview_height)
        messages = [
            f"已载入 {len(self._series)} 个跟踪器，预览已更新。",
            "点击图例后拖动右下角可同步缩放；双击图例恢复默认大小。",
        ]
        if diagnostics.font_substitution:
            messages.append(diagnostics.font_substitution)
        if diagnostics.clipping_warning:
            messages.append(diagnostics.clipping_warning)
        self.status_label.setText(" ".join(messages))
        self._preview_dirty = False
        self.refresh_button.setEnabled(False)

    def _set_legend_scale(self, scale: float) -> None:
        normalized = min(1.8, max(0.6, float(scale)))
        if abs(normalized - self._recipe.style.legend_scale) < 1e-6:
            return
        self._recipe = replace(
            self._recipe,
            style=replace(self._recipe.style, legend_scale=normalized),
        )
        self._persist_style()
        self.refresh_preview()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_preview_height)
        if self._series:
            self._preview_resize_timer.start()

    def _refresh_preview_after_resize(self) -> None:
        if not self._series or self._preview_dirty:
            return
        current = QSize(
            max(self.preview_plot_host.width(), 980),
            max(self.preview_plot_host.height(), 420),
        )
        if (
            abs(current.width() - self._last_preview_size.width()) < 32
            and abs(current.height() - self._last_preview_size.height()) < 24
        ):
            return
        self.refresh_preview()

    def _update_preview_height(self) -> None:
        width = self.preview_plot_host.width()
        if width <= 0:
            return
        available_height = max(360, self.height() - 210)
        target = min(self.preview_plot_host.heightForWidth(width), available_height)
        if self.preview_plot_host.height() != target:
            self.preview_plot_host.setFixedHeight(target)

    def _default_directory(self) -> Path:
        if self._project and self._project.sources.export_directory:
            return self._project.sources.export_directory
        return Path.home()

    def import_mat_files(self, paths: list[Path]) -> int:
        imported = 0
        messages: list[str] = []
        existing = {item.tracker_id for item in self._series}
        for path in paths:
            try:
                data = read_curve_mat(path)
            except CurveDataError as exc:
                messages.append(str(exc))
                continue
            identifier = data.tracker_id
            suffix = 2
            while identifier in existing:
                identifier = f"{data.tracker_id}-{suffix}"
                suffix += 1
            existing.add(identifier)
            self._series.append(
                CurveSeries(
                    identifier,
                    data.display_name,
                    data.auc,
                    data.dp,
                    data.precision_curve,
                    data.success_curve,
                    matlab_style_for_index(len(self._series)),
                )
            )
            imported += 1
        self._recipe = replace(
            self._recipe,
            trackers=tuple(
                self._embedded_entry(item, visible=True, order=index)
                for index, item in enumerate(self._series)
            ),
            precision_order=(),
            success_order=(),
        )
        self._fill_table()
        self._set_enabled(bool(self._series))
        self.import_mat_button.setEnabled(self._project is not None)
        if self._series:
            self._apply_sort(refresh=True)
        if messages:
            self.status_label.setText("；".join(messages[:3]))
        elif imported:
            self.status_label.setText(f"已从 MAT 文件导入 {imported} 个跟踪器。")
        return imported

    def _import_mat(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "导入已有 MAT 曲线结果",
            str(self._default_directory()),
            "MAT 文件 (*.mat)",
        )
        if filenames:
            self.import_mat_files([Path(value) for value in filenames])

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.tracker_table.selectedIndexes()})
        if not rows:
            self.status_label.setText("请先在跟踪器表格中选择要移除的行。")
            return
        identifiers = {
            self.tracker_table.item(row, DISPLAY_COLUMN).data(Qt.ItemDataRole.UserRole)
            for row in rows
        }
        self._series = [item for item in self._series if item.tracker_id not in identifiers]
        self._recipe = replace(
            self._recipe,
            trackers=tuple(
                self._embedded_entry(item, visible=True, order=index)
                for index, item in enumerate(self._series)
            ),
            precision_order=(),
            success_order=(),
        )
        self._fill_table()
        self._set_enabled(bool(self._series))
        self.import_mat_button.setEnabled(self._project is not None)
        if self._series:
            self.refresh_preview()
        else:
            self.status_label.setText("当前没有可绘制的跟踪器；可从指标页计算或导入 MAT。")

    def _save_recipe(self) -> None:
        if self.tracker_table.rowCount():
            self._on_tracker_changed(self.tracker_table.item(0, 0))
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "保存绘图配方",
            str(self._default_directory() / "tracking-curves.hsirecipe.json"),
            "HSI 绘图配方 (*.hsirecipe.json)",
        )
        if not filename:
            return
        destination = save_curve_recipe(self._recipe, Path(filename))
        if self._project is not None and destination not in self._project.recipe_files:
            self._project.recipe_files.append(destination)
            self.project_changed.emit()
        self.status_label.setText(f"配方已保存：{destination}")

    def _load_recipe(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "加载绘图配方",
            str(self._default_directory()),
            "HSI 绘图配方 (*.hsirecipe.json)",
        )
        if not filename:
            return
        try:
            recipe = load_curve_recipe(Path(filename))
        except RecipeCompatibilityError as exc:
            self.status_label.setText(str(exc))
            return
        if recipe.is_self_contained:
            self._series = series_from_recipe(recipe)
            self._apply_recipe(recipe)
            self.status_label.setText(
                f"已从自包含配置恢复 {len(self._series)} 个跟踪器，无需原 MAT 文件。"
            )
            return
        resolution = resolve_recipe_trackers(recipe, {item.tracker_id for item in self._series})
        if resolution.missing_ids:
            self._pending_recipe = recipe
            self._pending_mapping.clear()
            self._pending_skipped.clear()
            self._pending_missing = list(resolution.missing_ids)
            self.mapping_combo.clear()
            for item in self._series:
                self.mapping_combo.addItem(item.display_name, item.tracker_id)
            self.mapping_panel.setVisible(True)
            self._show_pending_mapping()
            return
        self._apply_recipe(recipe)

    def _show_pending_mapping(self) -> None:
        if not self._pending_missing:
            self._finish_pending_recipe()
            return
        missing = self._pending_missing[0]
        self.status_label.setText(
            f"配方引用的跟踪器“{missing}”不存在。请选择映射目标、跳过该项或取消应用。"
        )

    def _resolve_next_missing(self, *, skip: bool) -> None:
        if not self._pending_missing:
            return
        missing = self._pending_missing.pop(0)
        if skip:
            self._pending_skipped.add(missing)
        else:
            target = self.mapping_combo.currentData()
            if target is None:
                self._pending_missing.insert(0, missing)
                self.status_label.setText("当前没有可用于映射的跟踪器。")
                return
            self._pending_mapping[missing] = str(target)
        self._show_pending_mapping()

    def _finish_pending_recipe(self) -> None:
        recipe = self._pending_recipe
        if recipe is None:
            return
        transformed: list[RecipeTracker] = []
        for entry in recipe.trackers:
            if entry.tracker_id in self._pending_skipped:
                continue
            identifier = self._pending_mapping.get(entry.tracker_id, entry.tracker_id)
            transformed.append(replace(entry, tracker_id=identifier))
        self.mapping_panel.setVisible(False)
        self._pending_recipe = None
        self._apply_recipe(replace(recipe, trackers=tuple(transformed)))

    def _cancel_pending_recipe(self) -> None:
        self._pending_recipe = None
        self._pending_missing.clear()
        self.mapping_panel.setVisible(False)
        self.status_label.setText("已取消应用绘图配方，当前设置保持不变。")

    def _apply_recipe(self, recipe: CurveRecipe) -> None:
        self._recipe = recipe
        if recipe.is_self_contained:
            self._series = series_from_recipe(recipe)
        self._load_controls()
        self._fill_table()
        self._set_enabled(bool(self._series))
        self._persist_style()
        self.refresh_preview()

    def _export(self) -> None:
        if not self._series:
            return
        filename, selected = QFileDialog.getSaveFileName(
            self,
            "保存 Precision 和 Success 两张图",
            str(self._default_directory() / "Precision_plots_OPE.png"),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps)",
        )
        if not filename:
            return
        selected_path = Path(filename)
        if not selected_path.suffix:
            suffixes = {
                "PNG (*.png)": ".png",
                "PDF (*.pdf)": ".pdf",
                "SVG (*.svg)": ".svg",
                "EPS (*.eps)": ".eps",
            }
            selected_path = selected_path.with_suffix(suffixes.get(selected, ".png"))
        try:
            destinations = export_curve_pair(
                self._series,
                self._recipe,
                selected_path.parent,
                selected_path.suffix,
            )
        except FigureExportError as exc:
            self.status_label.setText(str(exc))
            return
        message = f"两张图已导出：{destinations[0]}；{destinations[1]}"
        self.status_label.setText(message)
        self.status_message.emit(message)
