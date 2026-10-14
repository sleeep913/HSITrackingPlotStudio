from __future__ import annotations

from PySide6.QtCore import QRegularExpression, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

STANDARD_TRACKER_COLORS: tuple[str, ...] = (
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
    "#F5D90A",
    "#1D9BF0",
    "#FF5A36",
    "#52D20A",
    "#B044E5",
    "#19D3D3",
    "#F30A8B",
    "#FFFFFF",
    "#F2F2F2",
    "#D9D9D9",
    "#BFBFBF",
    "#9E9E9E",
    "#737373",
    "#262626",
    "#EF1111",
    "#E600D7",
    "#FFD400",
    "#00C853",
    "#00BCD4",
    "#1D39C4",
    "#000000",
)

# Compatibility alias for callers written before the standard palette expansion.
COMMON_TRACKER_COLORS = STANDARD_TRACKER_COLORS

_CUSTOM_COLORS_KEY = "plotting/customTrackerColors"
_MAX_CUSTOM_COLORS = 8

_HEX_PATTERN = QRegularExpression(r"^#[0-9A-Fa-f]{6}$")


class SimpleColorDialog(QDialog):
    """Compact standard/custom tracker-color chooser with hex input."""

    def __init__(
        self,
        color: str,
        parent: QWidget | None = None,
        *,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择跟踪器颜色")
        self.setModal(True)
        self.setFixedWidth(368)
        self._settings = settings or QSettings()
        self._selected_color = self._normalize(color)
        self._selected_custom: str | None = None
        self._custom_colors = self._load_custom_colors()

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(12)

        standard_label = QLabel("标准颜色")
        standard_label.setProperty("role", "section-heading")
        root.addWidget(standard_label)
        palette_layout = QGridLayout()
        palette_layout.setHorizontalSpacing(7)
        palette_layout.setVerticalSpacing(8)
        self.swatch_buttons: list[QPushButton] = []
        for index, swatch_color in enumerate(STANDARD_TRACKER_COLORS):
            button = self._make_swatch(swatch_color)
            palette_layout.addWidget(button, index // 7, index % 7)
            self.swatch_buttons.append(button)
        root.addLayout(palette_layout)

        custom_header = QHBoxLayout()
        custom_label = QLabel("我的常用颜色")
        custom_label.setProperty("role", "section-heading")
        custom_header.addWidget(custom_label)
        custom_header.addStretch(1)
        self.add_custom_button = QPushButton("加入常用")
        self.remove_custom_button = QPushButton("删除选中")
        self.remove_custom_button.setEnabled(False)
        custom_header.addWidget(self.add_custom_button)
        custom_header.addWidget(self.remove_custom_button)
        root.addLayout(custom_header)
        self.custom_palette_layout = QGridLayout()
        self.custom_palette_layout.setHorizontalSpacing(7)
        self.custom_palette_layout.setVerticalSpacing(8)
        self.custom_swatch_buttons: list[QPushButton] = []
        root.addLayout(self.custom_palette_layout)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("十六进制"))
        self.hex_edit = QLineEdit(self._selected_color)
        self.hex_edit.setPlaceholderText("#RRGGBB")
        self.hex_edit.setMaxLength(7)
        self.hex_edit.setValidator(QRegularExpressionValidator(_HEX_PATTERN, self))
        input_row.addWidget(self.hex_edit, 1)
        self.preview = QWidget()
        self.preview.setObjectName("selectedColorPreview")
        self.preview.setFixedSize(44, 24)
        input_row.addWidget(self.preview)
        root.addLayout(input_row)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

        self.hex_edit.textChanged.connect(self._sync_hex_input)
        self.add_custom_button.clicked.connect(self._add_current_to_custom)
        self.remove_custom_button.clicked.connect(self._remove_selected_custom)
        self._rebuild_custom_palette()
        self._apply_preview(self._selected_color)

    @staticmethod
    def _normalize(color: str) -> str:
        value = QColor(color)
        return value.name().upper() if value.isValid() else "#000000"

    @staticmethod
    def _swatch_style(color: str, *, selected: bool = False) -> str:
        width = 3 if selected else 1
        border = "#2563EB" if selected else "#94A3B8"
        return (
            f"QPushButton {{ background-color: {color}; border: {width}px solid {border}; "
            "border-radius: 4px; }}"
            "QPushButton:hover { border: 2px solid #2563EB; }"
            "QPushButton:focus { border: 2px solid #2563EB; }"
        )

    def _make_swatch(self, color: str, *, custom: bool = False) -> QPushButton:
        button = QPushButton()
        button.setObjectName("customColorSwatch" if custom else "colorPaletteSwatch")
        button.setFixedSize(40, 28)
        button.setToolTip(color)
        button.setProperty("swatchColor", color)
        button.setStyleSheet(
            self._swatch_style(color, selected=custom and color == self._selected_custom)
        )
        if custom:
            button.clicked.connect(
                lambda _checked=False, value=color: self._select_custom_swatch(value)
            )
        else:
            button.clicked.connect(
                lambda _checked=False, value=color: self._select_swatch(value)
            )
        return button

    @property
    def custom_colors(self) -> tuple[str, ...]:
        return tuple(self._custom_colors)

    def _load_custom_colors(self) -> list[str]:
        raw = self._settings.value(_CUSTOM_COLORS_KEY, [])
        values = [raw] if isinstance(raw, str) else list(raw or [])
        colors: list[str] = []
        for value in values:
            color = QColor(str(value))
            if not color.isValid():
                continue
            normalized = color.name().upper()
            if normalized not in colors:
                colors.append(normalized)
        return colors[:_MAX_CUSTOM_COLORS]

    def _save_custom_colors(self) -> None:
        self._settings.setValue(_CUSTOM_COLORS_KEY, self._custom_colors)
        self._settings.sync()

    def _rebuild_custom_palette(self) -> None:
        while self.custom_palette_layout.count():
            item = self.custom_palette_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.custom_swatch_buttons = []
        if not self._custom_colors:
            empty = QLabel("尚未添加，可输入十六进制颜色后点击“加入常用”。")
            empty.setProperty("role", "caption")
            self.custom_palette_layout.addWidget(empty, 0, 0, 1, 8)
            return
        for index, color in enumerate(self._custom_colors):
            button = self._make_swatch(color, custom=True)
            self.custom_palette_layout.addWidget(button, index // 8, index % 8)
            self.custom_swatch_buttons.append(button)

    def selected_color(self) -> str:
        return self._selected_color

    def _select_swatch(self, color: str) -> None:
        had_custom_selection = self._selected_custom is not None
        self._selected_custom = None
        self.remove_custom_button.setEnabled(False)
        self.hex_edit.setText(color)
        if had_custom_selection:
            self._rebuild_custom_palette()

    def _select_custom_swatch(self, color: str) -> None:
        self._selected_custom = color
        self.remove_custom_button.setEnabled(True)
        self.hex_edit.setText(color)
        self._rebuild_custom_palette()

    def _add_current_to_custom(self) -> None:
        if not _HEX_PATTERN.match(self.hex_edit.text()).hasMatch():
            return
        color = self._selected_color
        self._custom_colors = [
            color, *(value for value in self._custom_colors if value != color)
        ][:_MAX_CUSTOM_COLORS]
        self._selected_custom = color
        self.remove_custom_button.setEnabled(True)
        self._save_custom_colors()
        self._rebuild_custom_palette()

    def _remove_selected_custom(self) -> None:
        if self._selected_custom is None:
            return
        self._custom_colors = [
            value for value in self._custom_colors if value != self._selected_custom
        ]
        self._selected_custom = None
        self.remove_custom_button.setEnabled(False)
        self._save_custom_colors()
        self._rebuild_custom_palette()

    def _sync_hex_input(self, text: str) -> None:
        valid = bool(_HEX_PATTERN.match(text).hasMatch())
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)
        self.add_custom_button.setEnabled(valid)
        if not valid:
            return
        self._selected_color = text.upper()
        if (
            self._selected_custom is not None
            and self._selected_custom != self._selected_color
        ):
            self._selected_custom = None
            self.remove_custom_button.setEnabled(False)
            self._rebuild_custom_palette()
        self._apply_preview(self._selected_color)

    def _apply_preview(self, color: str) -> None:
        self.preview.setProperty("previewColor", color)
        self.preview.setStyleSheet(
            f"background-color: {color}; border: 1px solid #94A3B8; border-radius: 3px;"
        )


class ColorPickerButton(QPushButton):
    color_changed = Signal(str)

    def __init__(self, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trackerColorButton")
        self.setFixedSize(38, 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._choose_color)
        self.set_color(color)

    def color(self) -> str:
        return str(self.property("trackerColor"))

    def set_color(self, color: str) -> None:
        value = QColor(color)
        normalized = value.name().upper() if value.isValid() else "#000000"
        self.setProperty("trackerColor", normalized)
        self.setText("")
        self.setToolTip(normalized)
        self.setAccessibleName(f"跟踪器颜色 {normalized}")
        self.setStyleSheet(
            f"QPushButton {{ background-color: {normalized}; border: 1px solid #94A3B8; "
            "border-radius: 3px; padding: 0; }}"
            "QPushButton:hover { border: 2px solid #2563EB; }"
        )

    def _choose_color(self) -> None:
        dialog = SimpleColorDialog(self.color(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.set_color(dialog.selected_color())
        self.color_changed.emit(self.color())


class ColorSwatchCell(QWidget):
    """Centers the compact color rectangle inside a table cell."""

    def __init__(self, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.addStretch(1)
        self.color_button = ColorPickerButton(color, self)
        layout.addWidget(self.color_button)
        layout.addStretch(1)
