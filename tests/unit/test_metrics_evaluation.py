from __future__ import annotations

from pathlib import Path

import pytest

from hsi_plotter.metrics.evaluation import (
    evaluate_project,
    evaluate_sequence,
    prepare_evaluation_inputs,
    run_evaluation,
)
from hsi_plotter.models.dataset import Modality
from hsi_plotter.models.evaluation import (
    EvaluationStatus,
    EvaluationTask,
)
from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.models.style import default_style_for_index
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.tracker_import import ResultLayout

_DATASET = "project"


def _write(path: Path, rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows, encoding="utf-8")


def _repeat(row: str, count: int) -> str:
    return "".join(f"{row}\n" for _ in range(count))


def _key(modality: Modality, name: str):
    from hsi_plotter.models.dataset import SequenceKey

    return SequenceKey(dataset=_DATASET, modality=modality, name=name)


def _tracker(project_dir: Path, name: str = "MHT", layout: ResultLayout = ResultLayout.FLAT):
    root = project_dir / "results" / name
    root.mkdir(parents=True, exist_ok=True)
    return Tracker(
        tracker_id=name.casefold(),
        display_name=name,
        results_path=root,
        style=default_style_for_index(0),
        layout=layout,
    )


@pytest.fixture
def project(tmp_path: Path) -> Project:
    ground_truth = tmp_path / "gt"
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    return Project(
        name="HOT2023",
        sources=ProjectSources(ground_truth=ground_truth, results_root=results_root),
        trackers=[],
    )


def _sequence_pair(
    project: Project,
    tracker: Tracker,
    *,
    modality: Modality,
    name: str,
    gt_row: str,
    result_row: str,
    frames: int,
    result_frames: int | None = None,
    prefix: bool = False,
) -> None:
    ground_truth = project.sources.ground_truth
    assert ground_truth is not None
    _write(ground_truth / f"{modality.value}-{name}.txt", _repeat(gt_row, frames))
    filename = f"{modality.value}-{name}.txt" if prefix else f"{name}.txt"
    _write(
        Path(tracker.results_path) / filename,
        _repeat(result_row, frames if result_frames is None else result_frames),
    )


def test_evaluate_sequence_reports_failure_instead_of_raising(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    result = tmp_path / "res.txt"
    _write(gt, _repeat("0,0,20,20", 3))
    _write(result, _repeat("0,0,20,20", 2))
    task = EvaluationTask(
        tracker_id="mht",
        tracker_name="MHT",
        layout=ResultLayout.FLAT,
        key=_key(Modality.VIS, "coin"),
        ground_truth_path=gt,
        result_path=result,
    )

    evaluation = evaluate_sequence(task)

    assert evaluation.status is EvaluationStatus.FAILED
    assert evaluation.metrics is None
    assert "帧数不一致" in evaluation.message


def test_evaluate_sequence_returns_invalid_frame_indices(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    result = tmp_path / "res.txt"
    _write(gt, _repeat("0,0,20,20", 3))
    _write(result, "0,0,20,20\nNaN,NaN,NaN,NaN\n0,0,20,20\n")
    task = EvaluationTask(
        tracker_id="mht",
        tracker_name="MHT",
        layout=ResultLayout.FLAT,
        key=_key(Modality.VIS, "coin"),
        ground_truth_path=gt,
        result_path=result,
    )

    evaluation = evaluate_sequence(task)

    assert evaluation.status is EvaluationStatus.SUCCEEDED
    assert evaluation.invalid_frame_indices == (2,)
    assert evaluation.valid_frames == 2
    assert evaluation.invalid_frames == 1


def test_overall_is_equal_weight_over_sequences(project: Project) -> None:
    tracker = _tracker(Path(project.sources.results_root))
    project.trackers = [tracker]
    _sequence_pair(
        project,
        tracker,
        modality=Modality.VIS,
        name="perfect",
        gt_row="0,0,20,20",
        result_row="0,0,20,20",
        frames=100,
        prefix=True,
    )
    _sequence_pair(
        project,
        tracker,
        modality=Modality.NIR,
        name="offset",
        gt_row="0,0,20,20",
        result_row="10,0,20,20",
        frames=3,
        prefix=True,
    )

    _, run = evaluate_project(project)

    evaluation = run.trackers[0]
    perfect = next(item for item in evaluation.sequences if item.sequence_name == "perfect")
    offset = next(item for item in evaluation.sequences if item.sequence_name == "offset")
    assert evaluation.overall is not None
    assert evaluation.overall.auc == pytest.approx((perfect.auc + offset.auc) / 2)
    assert evaluation.overall.dp == pytest.approx((perfect.dp + offset.dp) / 2)
    assert perfect.valid_frames == 100
    assert offset.valid_frames == 3
    assert evaluation.overall.sequence_count == 2


def test_overall_matches_modality_count_weighted_mean(project: Project) -> None:
    tracker = _tracker(Path(project.sources.results_root))
    project.trackers = [tracker]
    _sequence_pair(
        project,
        tracker,
        modality=Modality.VIS,
        name="a",
        gt_row="0,0,20,20",
        result_row="0,0,20,20",
        frames=5,
        prefix=True,
    )
    _sequence_pair(
        project,
        tracker,
        modality=Modality.VIS,
        name="b",
        gt_row="0,0,20,20",
        result_row="0,0,20,20",
        frames=5,
        prefix=True,
    )
    _sequence_pair(
        project,
        tracker,
        modality=Modality.NIR,
        name="c",
        gt_row="0,0,20,20",
        result_row="10,0,20,20",
        frames=5,
        prefix=True,
    )

    _, run = evaluate_project(project)

    evaluation = run.trackers[0]
    vis = evaluation.modality_metrics(Modality.VIS)
    nir = evaluation.modality_metrics(Modality.NIR)
    assert vis is not None and nir is not None
    assert vis.sequence_count == 2
    assert nir.sequence_count == 1
    assert evaluation.overall is not None
    assert evaluation.overall.auc == pytest.approx((2 * vis.auc + nir.auc) / 3)
    assert evaluation.overall.dp == pytest.approx((2 * vis.dp + nir.dp) / 3)


def test_failed_sequence_is_isolated(project: Project) -> None:
    tracker = _tracker(Path(project.sources.results_root))
    project.trackers = [tracker]
    _sequence_pair(
        project,
        tracker,
        modality=Modality.VIS,
        name="coin",
        gt_row="0,0,20,20",
        result_row="0,0,20,20",
        frames=4,
        prefix=True,
    )
    _sequence_pair(
        project,
        tracker,
        modality=Modality.NIR,
        name="worker",
        gt_row="0,0,20,20",
        result_row="0,0,20,20",
        frames=4,
        result_frames=3,
        prefix=True,
    )

    _, run = evaluate_project(project)

    evaluation = run.trackers[0]
    assert evaluation.status is EvaluationStatus.SUCCEEDED
    assert evaluation.succeeded_count == 1
    assert evaluation.failed_count == 1
    assert evaluation.overall is not None
    assert evaluation.overall.sequence_count == 1
    failed = evaluation.failed_sequences[0]
    assert failed.sequence_name == "worker"
    assert evaluation.message


def test_sequence_without_any_valid_frame_fails(project: Project) -> None:
    tracker = _tracker(Path(project.sources.results_root))
    project.trackers = [tracker]
    ground_truth = project.sources.ground_truth
    assert ground_truth is not None
    _write(ground_truth / "vis-empty.txt", _repeat("0,0,20,20", 3))
    _write(Path(tracker.results_path) / "vis-empty.txt", _repeat("NaN,NaN,NaN,NaN", 3))

    _, run = evaluate_project(project)

    evaluation = run.trackers[0]
    assert evaluation.failed_count == 1
    assert evaluation.overall is None
    assert evaluation.status is EvaluationStatus.FAILED
    assert not run.has_results


def test_legacy_modality_filter_is_ignored_and_all_modalities_are_evaluated(
    project: Project,
) -> None:
    tracker = _tracker(Path(project.sources.results_root))
    project.trackers = [tracker]
    for modality, name in ((Modality.VIS, "a"), (Modality.NIR, "b")):
        _sequence_pair(
            project,
            tracker,
            modality=modality,
            name=name,
            gt_row="0,0,20,20",
            result_row="0,0,20,20",
            frames=3,
            prefix=True,
        )

    plan, run = evaluate_project(project, modality_filter=(Modality.VIS,))

    assert plan.total_units == 2
    assert plan.modality_filter == ()
    assert run.trackers[0].modality_labels == "NIR, VIS"
    assert Modality.NIR in run.trackers[0].modalities


def test_plan_reports_blocking_errors_without_trackers(project: Project) -> None:
    plan = prepare_evaluation_inputs(project)

    assert plan.has_blocking_errors
    assert any(issue.code == "NO_TRACKERS_SELECTED" for issue in plan.issues)


def test_plan_reports_missing_sequences(project: Project) -> None:
    tracker = _tracker(Path(project.sources.results_root))
    project.trackers = [tracker]
    ground_truth = project.sources.ground_truth
    assert ground_truth is not None
    _write(ground_truth / "vis-only.txt", _repeat("0,0,20,20", 3))

    plan = prepare_evaluation_inputs(project)

    assert plan.total_units == 0
    assert any(issue.code == "MISSING_SEQUENCE" for issue in plan.issues)


def test_cancellation_keeps_finished_results(project: Project) -> None:
    tracker = _tracker(Path(project.sources.results_root))
    project.trackers = [tracker]
    for index in range(4):
        _sequence_pair(
            project,
            tracker,
            modality=Modality.VIS,
            name=f"seq{index}",
            gt_row="0,0,20,20",
            result_row="0,0,20,20",
            frames=3,
            prefix=True,
        )
    plan = prepare_evaluation_inputs(project)
    seen: list[int] = []

    def on_progress(completed: int, total: int, message: str) -> None:
        seen.append(completed)

    def is_cancelled() -> bool:
        return len(seen) >= 2

    run = run_evaluation(plan, on_progress=on_progress, is_cancelled=is_cancelled)

    assert run.cancelled
    evaluation = run.trackers[0]
    assert evaluation.succeeded_count == 2
    assert evaluation.skipped_sequences
    assert evaluation.status is EvaluationStatus.SKIPPED
    assert evaluation.overall is not None
    assert evaluation.overall.sequence_count == 2


def test_run_continues_when_one_tracker_has_no_results(project: Project) -> None:
    broken = _tracker(Path(project.sources.results_root), name="Broken")
    healthy = _tracker(Path(project.sources.results_root), name="Healthy")
    project.trackers = [broken, healthy]
    _sequence_pair(
        project,
        healthy,
        modality=Modality.VIS,
        name="coin",
        gt_row="0,0,20,20",
        result_row="0,0,20,20",
        frames=3,
        prefix=True,
    )

    _, run = evaluate_project(project)

    assert [tracker.tracker_name for tracker in run.trackers] == ["Broken", "Healthy"]
    assert run.trackers[0].overall is None
    assert run.trackers[1].overall is not None
    assert run.has_results
    assert run.failed_sequence_count == 0
