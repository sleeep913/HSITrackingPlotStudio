from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QPointF, QRectF, QSizeF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen

MIN_LEGEND_SCALE = 0.6
MAX_LEGEND_SCALE = 1.8
_HANDLE_SIZE = 9.0


def legend_scale_from_drag(
    start_scale: float, start_size: QSizeF, delta: QPointF
) -> float:
    """Convert a bottom-right handle drag into a clamped uniform scale."""
    width = max(1.0, start_size.width())
    height = max(1.0, start_size.height())
    factor = max((width + delta.x()) / width, (height + delta.y()) / height)
    return min(MAX_LEGEND_SCALE, max(MIN_LEGEND_SCALE, start_scale * factor))


class InteractiveLegendCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas with direct, synchronized legend resizing."""

    legend_scale_requested = Signal(float)

    def __init__(self, figure: Figure, *, legend_scale: float = 1.0) -> None:
        super().__init__(figure)
        self._legend_scale = float(legend_scale)
        self._selected_legend: int | None = None
        self._dragging = False
        self._drag_origin = QPointF()
        self._drag_start_size = QSizeF()
        self._drag_start_scale = self._legend_scale
        self._drag_preview_scale = self._legend_scale
        self.setMouseTracking(True)
        self.draw()

    def legend_rects(self) -> list[QRectF]:
        renderer = self.figure.canvas.get_renderer()
        figure_width = max(1.0, float(self.figure.bbox.width))
        figure_height = max(1.0, float(self.figure.bbox.height))
        scale_x = self.width() / figure_width
        scale_y = self.height() / figure_height
        rectangles: list[QRectF] = []
        for axis in self.figure.axes:
            legend = axis.get_legend()
            if legend is None:
                continue
            bbox = legend.get_window_extent(renderer=renderer)
            rectangles.append(
                QRectF(
                    bbox.x0 * scale_x,
                    self.height() - bbox.y1 * scale_y,
                    bbox.width * scale_x,
                    bbox.height * scale_y,
                )
            )
        return rectangles

    @staticmethod
    def _handle_rect(rectangle: QRectF) -> QRectF:
        half = _HANDLE_SIZE / 2.0
        return QRectF(
            rectangle.right() - half,
            rectangle.bottom() - half,
            _HANDLE_SIZE,
            _HANDLE_SIZE,
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        position = event.position()
        rectangles = self.legend_rects()
        hit = next(
            (index for index, rectangle in enumerate(rectangles) if rectangle.contains(position)),
            None,
        )
        if hit is None:
            self._selected_legend = None
            self._dragging = False
            self.update()
            return
        self._selected_legend = hit
        rectangle = rectangles[hit]
        if self._handle_rect(rectangle).adjusted(-5, -5, 5, 5).contains(position):
            self._dragging = True
            self._drag_origin = QPointF(position)
            self._drag_start_size = QSizeF(rectangle.size())
            self._drag_start_scale = self._legend_scale
            self._drag_preview_scale = self._legend_scale
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging:
            super().mouseMoveEvent(event)
            return
        self._drag_preview_scale = legend_scale_from_drag(
            self._drag_start_scale,
            self._drag_start_size,
            event.position() - self._drag_origin,
        )
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._legend_scale = self._drag_preview_scale
            self.legend_scale_requested.emit(self._legend_scale)
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and any(
            rectangle.contains(event.position()) for rectangle in self.legend_rects()
        ):
            self._legend_scale = 1.0
            self._drag_preview_scale = 1.0
            self.legend_scale_requested.emit(1.0)
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._selected_legend is None:
            return
        rectangles = self.legend_rects()
        if not rectangles:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor("#2563EB"), 1.4, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for rectangle in rectangles:
            painter.drawRoundedRect(rectangle.adjusted(-3, -3, 3, 3), 3, 3)
        selected = rectangles[min(self._selected_legend, len(rectangles) - 1)]
        painter.setPen(QPen(QColor("#FFFFFF"), 1.0))
        painter.setBrush(QColor("#2563EB"))
        painter.drawRect(self._handle_rect(selected))
        painter.end()
