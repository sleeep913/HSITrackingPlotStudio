from pathlib import Path

from hsi_plotter import APP_DISPLAY_NAME, APP_NAME, __version__
from hsi_plotter.app import is_smoke_test
from hsi_plotter.smoke import (
    verify_attribute_radar_export,
    verify_figure_export_backends,
    verify_tracking_visualization_exports,
)


def test_package_metadata_is_stable() -> None:
    assert APP_NAME == "HSI Tracking Plot Studio"
    assert APP_DISPLAY_NAME == "高光谱目标跟踪科研绘图平台"
    assert __version__ == "1.4.0"


def test_smoke_test_flag_is_explicit() -> None:
    assert is_smoke_test(["hsi-tracking-plotter", "--smoke-test"])
    assert not is_smoke_test(["hsi-tracking-plotter"])
    assert not is_smoke_test(["hsi-tracking-plotter", "--smoke-testing"])


def test_smoke_check_exports_every_supported_figure_format(tmp_path: Path) -> None:
    outputs = verify_figure_export_backends(tmp_path)

    assert len(outputs) == 8
    assert {path.suffix for path in outputs} == {".png", ".pdf", ".svg", ".eps"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_smoke_check_exports_combined_attribute_radar_pdf(tmp_path: Path) -> None:
    output = verify_attribute_radar_export(tmp_path)

    assert output.suffix == ".pdf"
    assert output.is_file() and output.stat().st_size > 0


def test_pyinstaller_collects_dynamic_matplotlib_export_backends() -> None:
    spec = (
        Path(__file__).resolve().parents[2] / "packaging" / "hsi_tracking_plotter.spec"
    ).read_text(encoding="utf-8")

    for backend in (
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_pdf",
        "matplotlib.backends.backend_ps",
        "matplotlib.backends.backend_svg",
    ):
        assert backend in spec


def test_smoke_check_exports_tracking_frames_and_mp4(tmp_path: Path) -> None:
    outputs = verify_tracking_visualization_exports(tmp_path)

    assert len(outputs) == 9
    assert {path.suffix for path in outputs} == {".png", ".pdf", ".svg", ".eps", ".mp4"}
    assert sum(path.stem.startswith("tracking-visual") for path in outputs) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
