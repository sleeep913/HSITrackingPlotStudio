import os
from pathlib import Path

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QScrollArea

from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.tracker_import import ResultLayout
from hsi_plotter.tasks.worker import TaskRunner
from hsi_plotter.ui.theme import apply_application_theme
from hsi_plotter.ui.tracking_visualization_page import TrackingVisualizationPage


@pytest.fixture
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _write(path: Path, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _project(tmp_path: Path) -> Project:
    ground = tmp_path / "ground"
    images = tmp_path / "images"
    frame_count = 8
    for modality, sequence in (("VIS", "ball"), ("NIR", "car")):
        _write(ground / modality / f"{sequence}.txt", ["0,0,20,20"] * frame_count)
        image_dir = images / f"HSI-{modality}-FalseColor" / sequence
        image_dir.mkdir(parents=True)
        for frame in range(1, frame_count + 1):
            Image.new("RGB", (160, 80), (230, 230, 230)).save(
                image_dir / f"{frame:04d}.jpg"
            )
    trackers: list[Tracker] = []
    for identifier, color, rows in (
        ("primary", "#FF0000", ["0,0,20,20"] * frame_count),
        ("competitor", "#0072BD", ["80,40,20,20"] * frame_count),
    ):
        result_root = tmp_path / identifier
        _write(result_root / "VIS" / "ball.txt", rows)
        _write(result_root / "NIR" / "car.txt", rows)
        trackers.append(
            Tracker(
                tracker_id=identifier,
                display_name=identifier.title(),
                results_path=result_root,
                style=TrackerStyle(color, "--" if identifier == "competitor" else "-", 2.0),
                layout=ResultLayout.MODALITY_SUBDIR,
            )
        )
    return Project(
        "paper",
        sources=ProjectSources(ground_truth=ground, image_root=images),
        trackers=trackers,
    )


def test_single_page_loads_sequences_trackers_and_fixed_preview(
    app: QApplication, tmp_path: Path
) -> None:
    page = TrackingVisualizationPage()
    page.set_project(_project(tmp_path))
    app.processEvents()

    assert page.modality_combo.count() == 2
    assert page.sequence_combo.count() == 1
    assert page.tracker_list.count() == 2
    assert page.preview_frame.maximumHeight() <= 560
    assert page.frame_spin.minimum() == 1
    assert page.frame_spin.maximum() == 8
    assert page.preview_canvas is not None
    assert len(page.preview_canvas.figure.axes[0].patches) == 3


def test_project_discovery_runs_through_task_runner(
    app: QApplication, tmp_path: Path
) -> None:
    runner = TaskRunner(max_concurrent=1)
    page = TrackingVisualizationPage()
    page.set_task_runner(runner)

    page.set_project(_project(tmp_path))

    assert page._active_scan_task is not None
    assert "正在扫描" in page.status_label.text()
    assert runner.wait_for_idle(10_000)
    app.processEvents()
    app.processEvents()

    assert page._active_scan_task is None
    assert page.modality_combo.count() == 2
    assert page.tracker_list.count() == 2


def test_frame_filter_buttons_preview_candidates_without_adding_them(
    app: QApplication, tmp_path: Path
) -> None:
    page = TrackingVisualizationPage()
    page.set_project(_project(tmp_path))
    page.primary_combo.setCurrentIndex(page.primary_combo.findData("primary"))

    page.filter_frames_button.click()
    app.processEvents()

    visible = [button for button in page.candidate_buttons if not button.isHidden()]
    assert len(visible) == 5
    assert page.visual_table.rowCount() == 0
    expected_frame = int(visible[0].property("frameNumber"))
    visible[0].click()
    assert page.frame_spin.value() == expected_frame
    assert page.visual_table.rowCount() == 0


def test_visual_list_accumulates_sequences_and_limits_each_to_three_frames(
    app: QApplication, tmp_path: Path
) -> None:
    page = TrackingVisualizationPage()
    page.set_project(_project(tmp_path))

    for frame in (4, 2, 3):
        page.frame_spin.setValue(frame)
        page.add_visual_button.click()
    assert page.visual_table.rowCount() == 1
    assert page.visual_table.item(0, 2).text() == "2, 3, 4"
    assert page.draw_visual_button.isEnabled()

    page.frame_spin.setValue(5)
    page.add_visual_button.click()
    assert "最多选择 3 帧" in page.status_label.text()

    nir_index = page.modality_combo.findData("nir")
    page.modality_combo.setCurrentIndex(nir_index)
    page.frame_spin.setValue(2)
    page.add_visual_button.click()

    assert page.visual_table.rowCount() == 2
    assert page.visual_table.item(1, 0).text() == "NIR"
    assert not page.draw_visual_button.isEnabled()


def test_add_visual_does_not_trigger_project_reload(
    app: QApplication, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    page = TrackingVisualizationPage()
    page.set_project(project)
    page.project_changed.connect(lambda: page.set_project(project))

    page.frame_spin.setValue(3)
    page.add_visual_button.click()
    app.processEvents()

    assert page.visual_table.rowCount() == 1
    assert page.visual_table.item(0, 2).text() == "3"
    assert "正在扫描" not in page.status_label.text()


def test_tracker_list_and_display_options_never_overlap(
    app: QApplication, tmp_path: Path
) -> None:
    apply_application_theme(app)
    page = TrackingVisualizationPage()
    page.resize(1670, 730)
    page.set_project(_project(tmp_path))
    page.show()
    app.processEvents()

    controls_scroll = page.findChild(
        QScrollArea, "trackingVisualizationControlsScroll"
    )

    tracker_bottom = page.tracker_list.mapToGlobal(
        page.tracker_list.rect().bottomLeft()
    ).y()
    ground_truth_top = page.ground_truth_checkbox.mapToGlobal(
        page.ground_truth_checkbox.rect().topLeft()
    ).y()

    assert controls_scroll is not None
    assert controls_scroll.widgetResizable()
    assert tracker_bottom < ground_truth_top
    page.close()


def test_tracker_visibility_uses_project_color_and_updates_only_current_frame(
    app: QApplication, tmp_path: Path
) -> None:
    page = TrackingVisualizationPage()
    page.set_project(_project(tmp_path))
    competitor = page.tracker_list.item(1)
    competitor.setCheckState(Qt.CheckState.Unchecked)
    app.processEvents()

    patches = page.preview_canvas.figure.axes[0].patches
    assert len(patches) == 2
    assert patches[0].get_edgecolor()[:3] == pytest.approx((1.0, 0.0, 0.0))


def test_page_exports_current_frame_and_cross_sequence_visual(
    app: QApplication, tmp_path: Path
) -> None:
    page = TrackingVisualizationPage()
    page.set_project(_project(tmp_path))
    for frame in (2, 3, 4):
        page.frame_spin.setValue(frame)
        page.add_visual_button.click()
    page.modality_combo.setCurrentIndex(page.modality_combo.findData("nir"))
    for frame in (3, 4, 5):
        page.frame_spin.setValue(frame)
        page.add_visual_button.click()

    frame_output = page.export_current_frame_to(tmp_path / "frame.png")
    visual_output = page.export_visual_to(tmp_path / "visual.pdf")

    assert frame_output.is_file()
    assert visual_output.is_file()
    assert visual_output.stat().st_size > 0


def test_page_recipe_restores_candidates_and_visual_list(
    app: QApplication, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    page = TrackingVisualizationPage()
    page.set_project(project)
    page.primary_combo.setCurrentIndex(page.primary_combo.findData("primary"))
    page.filter_frames_button.click()
    page.frame_spin.setValue(2)
    page.add_visual_button.click()
    recipe_path = page.save_recipe_to(tmp_path / "visual.hsirecipe.json")

    restored = TrackingVisualizationPage()
    restored.set_project(project)
    restored.load_recipe_from(recipe_path)

    assert restored.primary_combo.currentData() == "primary"
    assert restored.visual_table.rowCount() == 1
    assert len([button for button in restored.candidate_buttons if not button.isHidden()]) == 5


def test_project_cache_restores_candidates_and_visual_list_without_manual_import(
    app: QApplication, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    page = TrackingVisualizationPage()
    page.set_project(project)
    page.primary_combo.setCurrentIndex(page.primary_combo.findData("primary"))
    page.filter_frames_button.click()
    page.frame_spin.setValue(2)
    page.add_visual_button.click()

    restored = TrackingVisualizationPage()
    restored.set_project(project)
    app.processEvents()

    assert restored.primary_combo.currentData() == "primary"
    assert restored.visual_table.rowCount() == 1
    assert restored.visual_table.item(0, 2).text() == "2"
    assert len([button for button in restored.candidate_buttons if not button.isHidden()]) == 5
