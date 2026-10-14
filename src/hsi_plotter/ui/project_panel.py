from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from hsi_plotter.models.project import Project


class ElidedPathLabel(QLabel):
    """Keep a long project path on one line while retaining the full tooltip."""

    def __init__(self, text: str = "—") -> None:
        super().__init__()
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self.setToolTip("" if text == "—" else text)
        self._refresh_elision()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_elision()

    def _refresh_elision(self) -> None:
        available = max(20, self.contentsRect().width())
        shown = QFontMetrics(self.font()).elidedText(
            self._full_text, Qt.TextElideMode.ElideMiddle, available
        )
        super().setText(shown)


class ProjectPanel(QFrame):
    configure_sources_requested = Signal()
    manage_trackers_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("projectPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(250)
        self.setMaximumWidth(380)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        header = QHBoxLayout()
        heading = QLabel("项目概览")
        heading.setProperty("role", "heading")
        header.addWidget(heading, 1)
        self.save_status_label = QLabel("未打开")
        self.save_status_label.setObjectName("projectSaveStatus")
        self.save_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.save_status_label)
        layout.addLayout(header)

        identity_card = QFrame()
        identity_card.setObjectName("projectIdentityCard")
        identity_layout = QVBoxLayout(identity_card)
        identity_layout.setContentsMargins(12, 11, 12, 11)
        identity_layout.setSpacing(4)
        name_caption = QLabel("项目名称")
        name_caption.setProperty("role", "caption")
        identity_layout.addWidget(name_caption)
        self.project_name_label = QLabel("尚未打开项目")
        self.project_name_label.setObjectName("projectNameLabel")
        self.project_name_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        identity_layout.addWidget(self.project_name_label)
        path_caption = QLabel("项目文件")
        path_caption.setProperty("role", "caption")
        identity_layout.addWidget(path_caption)
        self.project_path_label = ElidedPathLabel("—")
        self.project_path_label.setObjectName("projectPathLabel")
        identity_layout.addWidget(self.project_path_label)
        layout.addWidget(identity_card)

        stats = QHBoxLayout()
        stats.setSpacing(8)
        data_card, self.data_status_label = self._stat_card("数据源", "0 / 4")
        self.data_status_label.setObjectName("dataStatusLabel")
        tracker_card, self.tracker_count_label = self._stat_card("跟踪器", "0")
        self.tracker_count_label.setObjectName("trackerCountLabel")
        stats.addWidget(data_card, 1)
        stats.addWidget(tracker_card, 1)
        layout.addLayout(stats)

        hint = QLabel("数据配置和映射会随项目一并保存。")
        hint.setProperty("role", "caption")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()
        self.configure_sources_button = QPushButton("配置数据源")
        self.configure_sources_button.setObjectName("configureSourcesButton")
        self.configure_sources_button.setProperty("primary", True)
        self.configure_sources_button.clicked.connect(self.configure_sources_requested)
        self.configure_sources_button.setEnabled(False)
        layout.addWidget(self.configure_sources_button)
        self.manage_trackers_button = QPushButton("跟踪器与序列映射")
        self.manage_trackers_button.setObjectName("manageTrackersButton")
        self.manage_trackers_button.clicked.connect(self.manage_trackers_requested)
        self.manage_trackers_button.setEnabled(False)
        layout.addWidget(self.manage_trackers_button)

    @staticmethod
    def _stat_card(caption: str, value: str) -> tuple[QFrame, QLabel]:
        card = QFrame()
        card.setObjectName("projectStatCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(11, 9, 11, 9)
        card_layout.setSpacing(2)
        value_label = QLabel(value)
        value_label.setProperty("role", "stat-value")
        caption_label = QLabel(caption)
        caption_label.setProperty("role", "caption")
        card_layout.addWidget(value_label)
        card_layout.addWidget(caption_label)
        return card, value_label

    def _set_save_status(self, text: str, tone: str) -> None:
        self.save_status_label.setText(text)
        self.save_status_label.setProperty("statusTone", tone)
        self.save_status_label.style().unpolish(self.save_status_label)
        self.save_status_label.style().polish(self.save_status_label)

    def set_project(
        self,
        project: Project | None,
        project_path: Path | None,
        modified: bool = False,
    ) -> None:
        if project is None:
            self.project_name_label.setText("尚未打开项目")
            self.project_path_label.set_full_text("—")
            self.data_status_label.setText("0 / 4")
            self.tracker_count_label.setText("0")
            self._set_save_status("未打开", "muted")
            self.configure_sources_button.setEnabled(False)
            self.manage_trackers_button.setEnabled(False)
            return
        self.project_name_label.setText(project.name)
        self.project_path_label.set_full_text(
            str(project_path) if project_path else "尚未保存"
        )
        configured_sources = sum(
            value is not None
            for value in (
                project.sources.ground_truth,
                project.sources.results_root,
                project.sources.image_root,
                project.sources.attribute_file,
            )
        )
        self.data_status_label.setText(f"{configured_sources} / 4")
        self.tracker_count_label.setText(str(len(project.trackers)))
        if project_path is None:
            self._set_save_status("尚未保存", "warning")
        elif modified:
            self._set_save_status("有未保存更改", "warning")
        else:
            self._set_save_status("已保存", "ok")
        self.configure_sources_button.setEnabled(True)
        self.manage_trackers_button.setEnabled(True)
