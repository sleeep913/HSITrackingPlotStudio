from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from hsi_plotter.io.result_writer import (
    MAT_LEGACY_FIELDS,
    build_filename,
    default_mat_filename,
    export_main_table_csv,
    export_results_csv,
    export_results_json,
    export_tracker_mat,
    export_trackers_mat,
    next_available_path,
    sanitize_file_stem,
)
from hsi_plotter.metrics.evaluation import evaluate_project
from hsi_plotter.models.dataset import Modality
from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.models.style import default_style_for_index
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.tracker_import import ResultLayout


@pytest.fixture
def project(tmp_path: Path) -> Project:
    ground_truth = tmp_path / "gt"
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    return Project(
        name="HOT2023",
        sources=ProjectSources(ground_truth=ground_truth, results_root=results_root),
    )


def _write(path: Path, rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows, encoding="utf-8")


def build_tracker(project: Project, name: str = "MHT") -> Tracker:
    tracker = Tracker(
        tracker_id=name.casefold(),
        display_name=name,
        results_path=Path(project.sources.results_root) / name,
        style=default_style_for_index(0),
    )
    project.trackers.append(tracker)
    return tracker


def add_sequence(
    project: Project, tracker: Tracker, *, modality: Modality, name: str, row: str, frames: int = 4
) -> None:
    ground_truth = Path(project.sources.ground_truth)
    boxes = "0,0,20,20\n" * frames
    _write(ground_truth / f"{modality.value}-{name}.txt", boxes)
    _write(
        Path(tracker.results_path) / f"{modality.value}-{name}.txt",
        "".join(f"{row}\n" for _ in range(frames)),
    )


def seed_run(project: Project):
    tracker = build_tracker(project)
    add_sequence(project, tracker, modality=Modality.VIS, name="coin", row="0,0,20,20")
    add_sequence(project, tracker, modality=Modality.NIR, name="worker", row="5,0,20,20")
    _, run = evaluate_project(project)
    return run


def _scalar(payload: dict, key: str) -> float:
    return float(np.asarray(payload[key]).ravel()[0])


def _scalar(payload: dict, key: str) -> float:
    return float(np.asarray(payload[key]).ravel()[0])


def test_sanitize_file_stem_removes_windows_characters() -> None:
    assert sanitize_file_stem('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"
    assert sanitize_file_stem("  MHT  ") == "MHT"
    assert sanitize_file_stem("MHT.") == "MHT"
    assert sanitize_file_stem("   ") == "result"
    assert sanitize_file_stem("x" * 400) == "x" * 120


def test_build_filename_appends_extension() -> None:
    assert build_filename("MHT*", "mat") == "MHT_.mat"
    assert build_filename("MHT", ".csv") == "MHT.csv"


def test_next_available_path_avoids_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "MHT.mat"
    target.write_text("", encoding="utf-8")

    assert next_available_path(target) == tmp_path / "MHT-1.mat"
    (tmp_path / "MHT-1.mat").write_text("", encoding="utf-8")
    assert next_available_path(target) == tmp_path / "MHT-2.mat"


def test_default_mat_filename_uses_legacy_pattern(tmp_path: Path, project: Project) -> None:
    run = seed_run(project)
    tracker_result = run.trackers[0]

    name = default_mat_filename(tracker_result)

    assert name.startswith("MHT-")
    assert name.endswith(".mat")
    assert name.count("-") == 2


def test_csv_export_is_utf8_sig_with_all_levels(tmp_path: Path, project: Project) -> None:
    run = seed_run(project)

    target = export_results_csv(run, tmp_path / "results.csv")

    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    with target.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    levels = {row["层级"] for row in rows}
    assert levels == {"总体", "模态", "序列"}
    sequence_rows = [row for row in rows if row["层级"] == "序列"]
    assert len(sequence_rows) == 2
    assert {row["模态"] for row in sequence_rows} == {"VIS", "NIR"}
    # 未四舍五入：1/3 之类的值必须保留完整精度
    assert float([row for row in rows if row["层级"] == "总体"][0]["CLE"]) >= 0


def test_main_table_csv_matches_nine_column_ui_and_keeps_full_precision(
    tmp_path: Path, project: Project
) -> None:
    run = seed_run(project)

    target = export_main_table_csv(run, tmp_path / "main-table.csv")

    with target.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == [
        "Method",
        "VIS AUC",
        "VIS DP@20",
        "RedNIR AUC",
        "RedNIR DP@20",
        "NIR AUC",
        "NIR DP@20",
        "Average AUC",
        "Average DP@20",
    ]
    assert rows[0]["Method"] == "MHT"
    assert rows[0]["RedNIR AUC"] == ""
    assert float(rows[0]["Average AUC"]) == run.trackers[0].auc


def test_csv_export_reports_failed_sequence(tmp_path: Path, project: Project) -> None:
    tracker = build_tracker(project)
    add_sequence(project, tracker, modality=Modality.VIS, name="coin", row="0,0,20,20")
    ground_truth = Path(project.sources.ground_truth)
    _write(ground_truth / "vis-short.txt", "0,0,20,20\n0,0,20,20\n")
    _write(Path(tracker.results_path) / "vis-short.txt", "0,0,20,20\n")
    _, run = evaluate_project(project)

    target = export_results_csv(run, tmp_path / "results.csv")

    with target.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    failed = [row for row in rows if row["层级"] == "序列" and row["范围"] == "short"]
    assert len(failed) == 1
    assert failed[0]["状态"] == "失败"
    assert failed[0]["AUC"] == ""


def test_json_export_keeps_full_precision(tmp_path: Path, project: Project) -> None:
    run = seed_run(project)
    tracker_result = run.trackers[0]

    target = export_results_json(run, tmp_path / "results.json")

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "hsi-evaluation/1"
    tracker_payload = payload["trackers"][0]
    assert tracker_payload["overall"]["auc"] == tracker_result.auc
    assert len(tracker_payload["overall"]["success_curve"]) == 51
    assert len(tracker_payload["overall"]["precision_curve"]) == 51
    assert {"vis", "nir"} <= set(tracker_payload["modalities"])
    assert tracker_payload["sequences"][0]["auc"] == pytest.approx(tracker_result.sequences[0].auc)


def test_mat_export_keeps_legacy_fields(tmp_path: Path, project: Project) -> None:
    run = seed_run(project)
    tracker_result = run.trackers[0]

    target = export_tracker_mat(tracker_result, tmp_path / "MHT.mat")

    payload = loadmat(str(target))
    for field in MAT_LEGACY_FIELDS:
        assert field in payload
    assert payload["PASCAL_rec"].shape == (2, 51)
    assert payload["distance_rec"].shape == (2, 51)
    assert payload["average_cle_rec"].shape == (2, 51)
    assert _scalar(payload, "AUC") == pytest.approx(tracker_result.auc)
    assert _scalar(payload, "DP") == pytest.approx(tracker_result.dp)
    names = [item.item() for item in payload["sequence_names"].ravel()]
    assert sorted(names) == ["coin", "worker"]
    # 旧绘图脚本的直接用法：mean(mean(PASCAL_rec(:, 2:end)))
    legacy_auc = float(np.mean(np.mean(payload["PASCAL_rec"][:, 1:])))
    assert legacy_auc == pytest.approx(tracker_result.auc)


def test_mat_export_handles_empty_tracker(tmp_path: Path) -> None:
    from hsi_plotter.models.evaluation import TrackerEvaluation

    empty = TrackerEvaluation(tracker_id="mht", tracker_name="MHT", layout=ResultLayout.FLAT)

    target = export_tracker_mat(empty, tmp_path / "empty.mat")

    payload = loadmat(str(target))
    assert payload["PASCAL_rec"].shape == (0, 51)
    assert np.isnan(_scalar(payload, "AUC"))


def test_export_trackers_mat_writes_one_file_each(tmp_path: Path, project: Project) -> None:
    for name in ("Alpha", "Beta"):
        tracker = build_tracker(project, name=name)
        add_sequence(project, tracker, modality=Modality.VIS, name="coin", row="0,0,20,20")
    _, run = evaluate_project(project)

    written = export_trackers_mat(run.trackers, tmp_path / "out")

    assert len(written) == 2
    assert {path.name.split("-")[0] for path in written} == {"Alpha", "Beta"}
    assert default_mat_filename(run.trackers[0]).startswith("Alpha-")
    assert all(path.is_file() for path in written)


def test_export_overwrites_atomically(tmp_path: Path, project: Project) -> None:
    run = seed_run(project)
    destination = tmp_path / "results.json"
    destination.write_text("stale", encoding="utf-8")

    export_results_json(run, destination)

    assert destination.read_text(encoding="utf-8").strip().startswith("{")
    assert not any(child.suffix == ".tmp" for child in tmp_path.iterdir())
