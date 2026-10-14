import logging
from pathlib import Path

from hsi_plotter.logging_config import configure_logging


def test_configure_logging_creates_rotating_log(tmp_path: Path) -> None:
    log_file = configure_logging(tmp_path)

    logging.getLogger("hsi_plotter.test").warning("stage-zero-smoke")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_file == tmp_path / "hsi_plotter.log"
    assert log_file.exists()
    assert "stage-zero-smoke" in log_file.read_text(encoding="utf-8")
