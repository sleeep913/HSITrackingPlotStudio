from pathlib import Path

from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.models.style import default_style_for_index
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.tracker_import import ResultLayout
from hsi_plotter.validation.preflight import collect_project_preflight

DATASET = "project"

_KEY_VIS = SequenceKey(DATASET, Modality.VIS, "basketball")
_KEY_NIR = SequenceKey(DATASET, Modality.NIR, "worker")


def _write(path: Path, rows: str = "1,2,30,40\n2,3,30,40\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows, encoding="utf-8")


def _tracker_root(root: Path, layout: ResultLayout) -> Path:
    tracker_root = root / "results" / "MHT"
    if layout is ResultLayout.PREFIXED_FLAT:
        _write(tracker_root / "vis-basketball.txt")
        _write(tracker_root / "nir-worker.txt")
    elif layout is ResultLayout.MODALITY_SUBDIR:
        _write(tracker_root / "VIS" / "basketball.txt")
        _write(tracker_root / "NIR" / "worker.txt")
    else:
        _write(tracker_root / "basketball.txt")
        _write(tracker_root / "worker.txt")
    return tracker_root


def _project(root: Path, layout: ResultLayout) -> Project:
    ground_truth = root / "gt"
    _write(ground_truth / "VIS" / "basketball.txt")
    _write(ground_truth / "NIR" / "worker.txt")
    tracker_root = _tracker_root(root, layout)
    return Project(
        name="HOT2023",
        sources=ProjectSources(ground_truth=ground_truth, results_root=root / "results"),
        trackers=[
            Tracker(
                tracker_id="mht",
                display_name="MHT",
                results_path=tracker_root,
                layout=layout,
                style=default_style_for_index(0),
            )
        ],
    )


def test_project_preflight_accepts_a_complete_project(tmp_path: Path) -> None:
    project = _project(tmp_path, ResultLayout.PREFIXED_FLAT)

    preflight = collect_project_preflight(project)

    assert preflight.ground_truth_files.keys() >= {_KEY_VIS, _KEY_NIR}
    assert "mht" in preflight.mappings
    assert preflight.mappings["mht"].matched_count == 2
    assert not preflight.has_errors


def test_project_preflight_stops_at_invalid_sources() -> None:
    project = Project(name="empty")

    preflight = collect_project_preflight(project)

    assert preflight.has_errors
    assert preflight.ground_truth_files == {}
    assert preflight.mappings == {}
    assert preflight.report.issues[0].code == "GROUND_TRUTH_NOT_CONFIGURED"


def test_project_preflight_reports_frame_count_mismatch(tmp_path: Path) -> None:
    project = _project(tmp_path, ResultLayout.MODALITY_SUBDIR)
    (tmp_path / "results" / "MHT" / "VIS" / "basketball.txt").write_text(
        "1,2,30,40\n", encoding="utf-8"
    )

    preflight = collect_project_preflight(project)

    codes = [issue.code for issue in preflight.report.issues]
    assert "FRAME_COUNT_MISMATCH" in codes
    assert preflight.has_errors


def test_project_preflight_reports_missing_sequences(tmp_path: Path) -> None:
    project = _project(tmp_path, ResultLayout.PREFIXED_FLAT)
    (tmp_path / "results" / "MHT" / "nir-worker.txt").unlink()

    preflight = collect_project_preflight(project)

    codes = [issue.code for issue in preflight.report.issues]
    assert "MISSING_SEQUENCE" in codes
    assert preflight.mappings["mht"].matched_count == 1


def test_project_preflight_emits_progress_and_honours_cancel(tmp_path: Path) -> None:
    project = _project(tmp_path, ResultLayout.PREFIXED_FLAT)

    seen: list[tuple[int, int, str]] = []
    preflight = collect_project_preflight(
        project,
        on_progress=lambda completed, total, message: seen.append((completed, total, message)),
        is_cancelled=lambda: len(seen) >= 1,
    )

    assert seen == [(1, 2, "MHT · project:nir:worker")]
    assert [issue.code for issue in preflight.report.issues][-1] == "PREFLIGHT_CANCELLED"


def test_project_preflight_reports_duplicate_ground_truth_files(tmp_path: Path) -> None:
    ground_truth = tmp_path / "gt"
    _write(ground_truth / "VIS" / "basketball.txt")
    _write(ground_truth / "vis-basketball.txt")
    project = Project(name="HOT2023", sources=ProjectSources(ground_truth=ground_truth))

    preflight = collect_project_preflight(project)

    codes = [issue.code for issue in preflight.report.issues]
    assert "DUPLICATE_SEQUENCE_KEY" in codes
    assert preflight.has_errors


def test_project_preflight_reports_duplicate_tracker_result_files(tmp_path: Path) -> None:
    project = _project(tmp_path, ResultLayout.MODALITY_SUBDIR)
    _write(tmp_path / "results" / "MHT" / "VIS" / "nested" / "basketball.txt")

    preflight = collect_project_preflight(project)

    codes = [issue.code for issue in preflight.report.issues]
    assert "DUPLICATE_SEQUENCE_FILE" in codes
    assert preflight.has_errors


def test_project_preflight_reports_layout_that_no_longer_matches(tmp_path: Path) -> None:
    project = _project(tmp_path, ResultLayout.MODALITY_SUBDIR)
    project.trackers = [
        Tracker(
            tracker_id="mht",
            display_name="MHT",
            results_path=tmp_path / "results" / "MHT",
            layout=ResultLayout.FLAT,
            style=default_style_for_index(0),
        )
    ]

    preflight = collect_project_preflight(project)

    codes = [issue.code for issue in preflight.report.issues]
    assert "TRACKER_NO_RESULT_FILES" in codes
    assert "TRACKER_NO_MATCHED_SEQUENCES" in codes
    assert preflight.has_errors


def test_project_preflight_reports_modality_conflict_in_tracker_results(tmp_path: Path) -> None:
    project = _project(tmp_path, ResultLayout.MODALITY_SUBDIR)
    _write(tmp_path / "results" / "MHT" / "VIS" / "nir-extra.txt")

    preflight = collect_project_preflight(project)

    codes = [issue.code for issue in preflight.report.issues]
    assert "MODALITY_CONFLICT" in codes
