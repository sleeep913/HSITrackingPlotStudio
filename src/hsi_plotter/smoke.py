from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from hsi_plotter.io.attribute_pdf_exporter import export_attribute_radar_pdf
from hsi_plotter.io.video_exporter import export_tracking_video
from hsi_plotter.io.visualization_exporter import (
    export_tracking_frame,
    export_tracking_montage,
)
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.recipe import CurveRecipe, RecipeTracker
from hsi_plotter.models.style import TrackerStyle, matlab_hot2023_plot_style
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.visualization import ImageSequenceIndex, VisualSelectionList
from hsi_plotter.plotting.attribute_radar import AttributeRadarSeries
from hsi_plotter.plotting.curve_plot import CurveSeries
from hsi_plotter.plotting.exporters import SUPPORTED_FIGURE_FORMATS, export_curve_pair
from hsi_plotter.visualization.session import TrackerSequenceSource, load_sequence_session


def verify_figure_export_backends(directory: Path) -> tuple[Path, ...]:
    """Exercise every dynamically loaded Matplotlib backend used by the app."""
    root = Path(directory)
    style = TrackerStyle("#0072BD", "-", 2.0)
    series = CurveSeries(
        tracker_id="smoke",
        display_name="Smoke",
        auc=0.5,
        dp=0.75,
        precision_curve=np.linspace(0.0, 1.0, 51),
        success_curve=np.linspace(1.0, 0.0, 51),
        style=style,
    )
    recipe = CurveRecipe(
        trackers=(RecipeTracker("smoke", "Smoke", style=style),),
        style=matlab_hot2023_plot_style(combined=True),
    )
    outputs: list[Path] = []
    for format_name in sorted(SUPPORTED_FIGURE_FORMATS):
        outputs.extend(export_curve_pair([series], recipe, root, format_name))
    return tuple(outputs)


def run_figure_export_smoke() -> None:
    with TemporaryDirectory(prefix="hsi-plotter-smoke-") as directory:
        outputs = verify_figure_export_backends(Path(directory))
        if len(outputs) != 8 or not all(path.stat().st_size > 0 for path in outputs):
            raise RuntimeError("图形导出冒烟检查未生成完整输出")


def verify_attribute_radar_export(directory: Path) -> Path:
    """Exercise the packaged combined AUC/DP radar PDF path."""
    style = TrackerStyle("#0072BD", "-", 2.0)
    output = export_attribute_radar_pdf(
        Path(directory),
        ("BC", "DEF", "OCC"),
        (
            AttributeRadarSeries(
                tracker_id="smoke",
                display_name="Smoke",
                auc_values=(0.51, 0.58, 0.63),
                dp_values=(0.72, 0.79, 0.84),
                style=style,
            ),
        ),
        project_name="smoke",
    )
    if output.stat().st_size <= 0:
        raise RuntimeError("属性雷达图冒烟检查未生成有效 PDF")
    return output


def verify_tracking_visualization_exports(directory: Path) -> tuple[Path, ...]:
    """Exercise packaged frame, four-column Visual, and MP4 exporters."""
    root = Path(directory)
    frames: dict[int, Path] = {}
    for frame_number in (1, 2, 3, 4):
        path = root / f"{frame_number:04d}.png"
        Image.new("RGB", (80, 40), (235, 235, 235)).save(path)
        frames[frame_number] = path
    ground_truth = root / "groundtruth.txt"
    ground_truth.write_text("5,5,20,10\n" * 4, encoding="utf-8")
    result = root / "result.txt"
    result.write_text("6,6,20,10\n" * 4, encoding="utf-8")
    tracker = Tracker(
        tracker_id="smoke",
        display_name="Smoke",
        results_path=root,
        style=TrackerStyle("#FF0000"),
    )
    session = load_sequence_session(
        ImageSequenceIndex(
            key=SequenceKey("smoke", Modality.VIS, "sequence"),
            directory=root,
            frames=frames,
        ),
        ground_truth,
        (TrackerSequenceSource(tracker, result),),
    )
    composition = session.compose(1)
    outputs = [
        export_tracking_frame(composition, root / f"tracking-frame.{suffix}")
        for suffix in sorted(SUPPORTED_FIGURE_FORMATS)
    ]
    selections = (
        VisualSelectionList()
        .add(session.key, 4)
        .add(session.key, 2)
        .add(session.key, 3)
    )
    outputs.extend(
        export_tracking_montage(
            {session.key: session},
            selections,
            root / f"tracking-visual.{suffix}",
        )
        for suffix in sorted(SUPPORTED_FIGURE_FORMATS)
    )
    outputs.append(
        export_tracking_video(
            session,
            root / "tracking-video.mp4",
            start_frame=1,
            end_frame=2,
            fps=2.0,
            resolution=(160, 80),
        )
    )
    return tuple(outputs)
