import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog, QWidget  # noqa: E402

from hsi_plotter.io.project_store import load_project, save_project  # noqa: E402
from hsi_plotter.models.attribute_evaluation import (  # noqa: E402
    AttributeDefinition,
    AttributeEvaluationRun,
    AttributeScore,
    TrackerAttributeResult,
)
from hsi_plotter.models.project import Project, ProjectSources  # noqa: E402
from hsi_plotter.models.style import default_style_for_index  # noqa: E402
from hsi_plotter.models.tracker import Tracker  # noqa: E402
from hsi_plotter.models.tracker_import import ResultLayout  # noqa: E402
from hsi_plotter.ui.attribute_page import AttributePage  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _project(tmp_path: Path) -> Project:
    tracker = Tracker(
        tracker_id="mht",
        display_name="MHT",
        results_path=tmp_path / "results" / "MHT",
        style=default_style_for_index(0),
        layout=ResultLayout.PREFIXED_FLAT,
    )
    return Project(
        name="HOT2023",
        sources=ProjectSources(export_directory=tmp_path / "export"),
        trackers=[tracker],
    )


def _run() -> AttributeEvaluationRun:
    attributes = (
        AttributeDefinition("Scale Variation", ("vis-ball",)),
        AttributeDefinition("Occlusion", ("vis-ball", "nir-worker")),
        AttributeDefinition("Fast Motion", ("rednir-car",)),
    )
    return AttributeEvaluationRun(
        attributes=attributes,
        trackers=(
            TrackerAttributeResult(
                tracker_id="mht",
                tracker_name="MHT",
                scores={
                    "Scale Variation": AttributeScore(0.61, 0.82, 1, 1, 0),
                    "Occlusion": AttributeScore(0.52, 0.75, 2, 2, 0),
                    "Fast Motion": AttributeScore(None, None, 1, 0, 1),
                },
            ),
        ),
    )


def _project_with_two_trackers(tmp_path: Path) -> Project:
    project = _project(tmp_path)
    project.trackers.append(
        Tracker(
            tracker_id="lrsst",
            display_name="LRSST",
            results_path=tmp_path / "results" / "LRSST",
            style=default_style_for_index(1),
            layout=ResultLayout.PREFIXED_FLAT,
        )
    )
    return project


def _run_with_two_trackers() -> AttributeEvaluationRun:
    base = _run()
    return AttributeEvaluationRun(
        attributes=base.attributes,
        trackers=(
            *base.trackers,
            TrackerAttributeResult(
                tracker_id="lrsst",
                tracker_name="LRSST",
                scores={
                    "Scale Variation": AttributeScore(0.71, 0.88, 1, 1, 0),
                    "Occlusion": AttributeScore(0.62, 0.81, 2, 2, 0),
                    "Fast Motion": AttributeScore(0.59, 0.77, 1, 1, 0),
                },
            ),
        ),
    )


def test_page_has_streamlined_controls_and_no_plot_canvas(app: QApplication) -> None:
    page = AttributePage()

    assert page.import_attribute_button.text() == "导入属性字典…"
    assert page.import_ground_truth_button.text() == "导入 Ground Truth…"
    assert page.import_trackers_button.text() == "导入跟踪器…"
    assert page.run_button.text() == "计算 AUC / DP"
    assert page.findChild(QWidget, "attributeRadarCanvas") is None
    assert [page.content_tabs.tabText(index) for index in range(page.content_tabs.count())] == [
        "属性与序列",
        "计算结果",
    ]


def test_load_attribute_file_populates_sequence_table(
    app: QApplication, tmp_path: Path
) -> None:
    source = tmp_path / "attributes.txt"
    source.write_text("SV:vis-ball,nir-worker\nOC:rednir-car\n", encoding="utf-8")
    project = _project(tmp_path)
    page = AttributePage()
    page.set_project(project)

    page.load_attribute_file(source)

    assert page.attribute_table.rowCount() == 2
    assert page.attribute_table.item(0, 0).text() == "SV"
    assert page.attribute_table.item(0, 1).text() == "2"
    assert page.attribute_table.item(0, 2).text() == "vis-ball, nir-worker"
    assert project.sources.attribute_file == source


def test_one_result_table_switches_between_auc_and_dp(
    app: QApplication, tmp_path: Path
) -> None:
    page = AttributePage()
    page.set_project(_project(tmp_path))
    page.display_run(_run())

    assert page.result_table.horizontalHeaderItem(0).text() == "跟踪器"
    assert page.result_table.horizontalHeaderItem(1).text() == "Scale Variation"
    assert page.result_table.item(0, 0).text() == "MHT"
    assert page.result_table.item(0, 1).text() == "0.6100"
    assert page.export_button.text() == "导出属性雷达图 PDF…"

    page.show_dp()

    assert page.result_table.item(0, 1).text() == "0.8200"
    assert page.result_table.item(0, 3).text() == "—"
    assert page.export_button.text() == "导出属性雷达图 PDF…"


def test_radar_series_reuses_project_tracker_style(
    app: QApplication, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    page = AttributePage()
    page.set_project(project)
    page.display_run(_run())

    attributes, series = page.radar_data()

    assert attributes == ("Scale Variation", "Occlusion", "Fast Motion")
    assert series[0].auc_values == (0.61, 0.52, None)
    assert series[0].dp_values == (0.82, 0.75, None)
    assert series[0].style == project.trackers[0].style


def test_radar_tracker_menu_filters_both_auc_and_dp_series(
    app: QApplication, tmp_path: Path
) -> None:
    page = AttributePage()
    page.set_project(_project_with_two_trackers(tmp_path))
    page.display_run(_run_with_two_trackers())

    assert page.radar_tracker_button.text() == "绘图跟踪器 2/2"
    page.set_radar_tracker_selected("mht", False)
    attributes, series = page.radar_data()

    assert attributes == ("Scale Variation", "Occlusion", "Fast Motion")
    assert [item.tracker_id for item in series] == ["lrsst"]
    assert series[0].auc_values == (0.71, 0.62, 0.59)
    assert series[0].dp_values == (0.88, 0.81, 0.77)
    assert page.radar_tracker_button.text() == "绘图跟踪器 1/2"


def test_radar_tracker_select_all_toggles_and_empty_selection_blocks_export(
    app: QApplication, tmp_path: Path
) -> None:
    page = AttributePage()
    page.set_project(_project_with_two_trackers(tmp_path))
    page.display_run(_run_with_two_trackers())

    page.set_all_radar_trackers_selected(False)

    assert page.radar_data()[1] == ()
    assert not page.export_button.isEnabled()
    assert page.export_current_pdf(tmp_path) is None
    assert "至少选择一个跟踪器" in page.status_label.text()

    page.set_all_radar_trackers_selected(True)
    assert len(page.radar_data()[1]) == 2
    assert page.export_button.isEnabled()


def test_real_export_button_click_writes_one_dual_radar_pdf(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = AttributePage()
    page.set_project(_project(tmp_path))
    page.display_run(_run())
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path),
    )

    page.export_button.click()
    app.processEvents()

    outputs = list(tmp_path.glob("HOT2023_attributes*.pdf"))
    assert len(outputs) == 1
    assert outputs[0].read_bytes().startswith(b"%PDF")
    assert str(outputs[0]) in page.status_label.text()


def test_export_passes_only_selected_radar_trackers(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_ids: list[str] = []

    def fake_export(directory, attributes, series, *, project_name):
        captured_ids.extend(item.tracker_id for item in series)
        output = Path(directory) / f"{project_name}_attributes.pdf"
        output.write_bytes(b"%PDF-filtered")
        return output

    monkeypatch.setattr(
        "hsi_plotter.ui.attribute_page.export_attribute_radar_pdf", fake_export
    )
    page = AttributePage()
    page.set_project(_project_with_two_trackers(tmp_path))
    page.display_run(_run_with_two_trackers())
    page.set_radar_tracker_selected("mht", False)

    output = page.export_current_pdf(tmp_path)

    assert output == tmp_path / "HOT2023_attributes.pdf"
    assert captured_ids == ["lrsst"]


def test_attribute_result_cache_restores_after_project_reopen(
    app: QApplication, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    attribute_file = tmp_path / "attributes.txt"
    attribute_file.write_text(
        "Scale Variation:vis-ball\nOcclusion:vis-ball,nir-worker\n"
        "Fast Motion:rednir-car\n",
        encoding="utf-8",
    )
    project.sources = ProjectSources(
        ground_truth=tmp_path / "gt",
        results_root=tmp_path / "results",
        attribute_file=attribute_file,
        export_directory=tmp_path / "export",
    )
    page = AttributePage()
    page.set_project(project)
    page._accept_computed_run(_run())
    project_path = tmp_path / "cached.hsiproj"
    save_project(project, project_path)

    restored_page = AttributePage()
    restored_page.set_project(load_project(project_path))

    assert restored_page.current_run == _run()
    assert "已恢复属性评估缓存" in restored_page.status_label.text()


def test_radar_tracker_selection_restores_with_project_cache(
    app: QApplication, tmp_path: Path
) -> None:
    project = _project_with_two_trackers(tmp_path)
    attribute_file = tmp_path / "attributes.txt"
    attribute_file.write_text(
        "Scale Variation:vis-ball\nOcclusion:vis-ball,nir-worker\n"
        "Fast Motion:rednir-car\n",
        encoding="utf-8",
    )
    project.sources = ProjectSources(
        ground_truth=tmp_path / "gt",
        results_root=tmp_path / "results",
        attribute_file=attribute_file,
        export_directory=tmp_path / "export",
    )
    page = AttributePage()
    page.set_project(project)
    page._accept_computed_run(_run_with_two_trackers())
    page.set_radar_tracker_selected("mht", False)
    project_path = tmp_path / "cached-selection.hsiproj"
    save_project(project, project_path)

    restored_page = AttributePage()
    restored_page.set_project(load_project(project_path))

    assert [item.tracker_id for item in restored_page.radar_data()[1]] == ["lrsst"]
    assert restored_page.radar_tracker_button.text() == "绘图跟踪器 1/2"
