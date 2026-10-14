from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from hsi_plotter import APP_NAME

_HANDLER_MARKER = "hsi_plotter_managed"


def default_log_directory() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return base / APP_NAME / "logs"


def configure_logging(log_directory: Path | None = None) -> Path:
    directory = Path(log_directory) if log_directory is not None else default_log_directory()
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / "hsi_plotter.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    setattr(file_handler, _HANDLER_MARKER, True)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_file
