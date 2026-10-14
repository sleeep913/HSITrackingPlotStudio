from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from hsi_plotter import APP_DISPLAY_NAME, APP_NAME, ORGANIZATION_NAME, __version__
from hsi_plotter.logging_config import configure_logging
from hsi_plotter.smoke import (
    verify_attribute_radar_export,
    verify_figure_export_backends,
    verify_tracking_visualization_exports,
)
from hsi_plotter.ui.main_window import MainWindow
from hsi_plotter.ui.theme import apply_application_theme


def is_smoke_test(argv: Sequence[str]) -> bool:
    return "--smoke-test" in argv


def _install_exception_hook() -> None:
    def handle_exception(
        exc_type: type[BaseException], exc: BaseException, traceback: object
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, traceback)
            return

        logging.getLogger(__name__).critical(
            "Unhandled exception",
            exc_info=(exc_type, exc, traceback),
        )
        QMessageBox.critical(
            None,
            "未处理的错误",
            "程序遇到未处理的错误。详细信息已写入本地日志。",
        )

    sys.excepthook = handle_exception


def _apply_metadata(app: QApplication) -> None:
    """Always publish application metadata, even when a QApplication exists.

    Tests may create a QApplication before :func:`create_application` runs; the
    displayed name must then still be deterministic.
    """
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(ORGANIZATION_NAME)


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        _apply_metadata(existing)
        apply_application_theme(existing)
        return existing

    app = QApplication(list(argv) if argv is not None else sys.argv)
    _apply_metadata(app)
    apply_application_theme(app)
    return app


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else list(sys.argv)
    smoke_test = is_smoke_test(arguments)
    qt_arguments = [argument for argument in arguments if argument != "--smoke-test"]
    app = create_application(qt_arguments)
    log_file = configure_logging()
    _install_exception_hook()
    logging.getLogger(__name__).info(
        "Starting %s version %s; log=%s",
        APP_NAME,
        __version__,
        log_file,
    )

    if smoke_test:
        try:
            with TemporaryDirectory(prefix="hsi-plotter-smoke-") as directory:
                verify_figure_export_backends(Path(directory))
                verify_attribute_radar_export(Path(directory))
                verify_tracking_visualization_exports(Path(directory))
        except Exception:
            logging.getLogger(__name__).exception("Figure export smoke test failed")
            return 2

    window = MainWindow()
    window.show()
    if smoke_test:
        QTimer.singleShot(100, app.quit)
    return app.exec()
