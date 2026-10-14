from __future__ import annotations

from PySide6.QtWidgets import QApplication

APPLICATION_STYLESHEET = """
QWidget {
    color: #1F2937;
    font-family: "Microsoft YaHei UI", "Segoe UI";
    font-size: 13px;
}
QMainWindow, QDialog { background: #F4F6F8; }
QMenuBar { background: #FFFFFF; border-bottom: 1px solid #E5E7EB; padding: 3px; }
QMenuBar::item { padding: 6px 10px; border-radius: 5px; }
QMenuBar::item:selected, QMenu::item:selected { background: #EEF4FF; color: #1D4ED8; }
QMenu { background: #FFFFFF; border: 1px solid #DDE2E8; padding: 6px; }
QMenu::item { padding: 7px 28px 7px 12px; border-radius: 5px; }
QToolBar {
    background: #FFFFFF; border: 0; border-bottom: 1px solid #E5E7EB;
    spacing: 5px; padding: 6px 10px;
}
QToolButton { padding: 6px 10px; border-radius: 6px; }
QToolButton:hover { background: #EEF4FF; color: #1D4ED8; }
QStatusBar { background: #FFFFFF; color: #64748B; border-top: 1px solid #E5E7EB; }
QFrame#projectPanel, QFrame#sourceCard, QFrame#preflightCard, QFrame#importCard {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
}
QFrame#projectIdentityCard, QFrame#projectStatCard {
    background: #F8FAFC;
    border: 1px solid #E7ECF2;
    border-radius: 7px;
}
QLabel#projectNameLabel { color: #0F172A; font-weight: 600; font-size: 14px; }
QLabel#projectPathLabel { color: #64748B; }
QLabel[role="stat-value"] { color: #1E3A8A; font-size: 17px; font-weight: 600; }
QLabel#projectSaveStatus {
    border-radius: 9px; padding: 3px 8px; font-size: 11px; font-weight: 600;
}
QLabel#projectSaveStatus[statusTone="muted"] { color: #64748B; background: #EEF2F6; }
QLabel#projectSaveStatus[statusTone="warning"] { color: #92400E; background: #FEF3C7; }
QLabel#projectSaveStatus[statusTone="ok"] { color: #047857; background: #D1FAE5; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px; border: 1px solid #CBD5E1;
    border-radius: 3px; background: #FFFFFF;
}
QCheckBox::indicator:checked { background: #2563EB; border-color: #2563EB; }
QProgressBar {
    background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px;
    min-height: 14px; max-height: 14px; text-align: center; color: #64748B;
}
QProgressBar::chunk { background: #BFDBFE; border-radius: 5px; }
QLabel[role="status-error"] { color: #B91C1C; font-weight: 600; }
QLabel[role="status-warning"] { color: #92400E; font-weight: 600; }
QLabel[role="status-ok"] { color: #047857; font-weight: 600; }
QLabel[role="heading"] { font-size: 18px; font-weight: 600; color: #111827; }
QLabel[role="caption"] { color: #64748B; }
QLabel[role="section-heading"] { color: #1F2937; font-weight: 600; }
QPushButton {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    min-height: 30px;
    padding: 0 12px;
}
QPushButton:hover { border-color: #2563EB; color: #1D4ED8; background: #F8FAFF; }
QPushButton:pressed { background: #E8F0FE; }
QPushButton:disabled { color: #94A3B8; background: #F8FAFC; border-color: #E2E8F0; }
QPushButton[primary="true"] {
    background: #2563EB; color: #FFFFFF; border-color: #2563EB; font-weight: 600;
}
QPushButton[primary="true"]:hover { background: #1D4ED8; border-color: #1D4ED8; }
QLineEdit, QComboBox {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    min-height: 30px;
    padding: 0 9px;
    selection-background-color: #BFDBFE;
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #2563EB; }
QTabWidget::pane { border: 0; background: #F4F6F8; }
QTabBar::tab {
    color: #64748B; background: transparent; padding: 10px 15px;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:hover { color: #1D4ED8; }
QTabBar::tab:selected { color: #1D4ED8; font-weight: 600; border-bottom: 2px solid #2563EB; }
QTableView, QTableWidget {
    background: #FFFFFF;
    alternate-background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    gridline-color: #EEF2F7;
    selection-background-color: #DBEAFE;
    selection-color: #1E3A8A;
}
QHeaderView::section {
    background: #F8FAFC; color: #475569; border: 0;
    border-bottom: 1px solid #E2E8F0; padding: 8px; font-weight: 600;
}
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #CBD5E1; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSplitter::handle { background: #E5E7EB; width: 1px; }
"""


def apply_application_theme(application: QApplication) -> None:
    application.setStyle("Fusion")
    application.setStyleSheet(APPLICATION_STYLESHEET)
