from pathlib import Path

import pytest

from hsi_plotter.metrics.attributes import run_attribute_evaluation
from hsi_plotter.metrics.tracking import compute_sequence_metrics
from hsi_plotter.models.attribute_evaluation import AttributeDefinition
from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.models.style import default_style_for_index
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.tracker_import import ResultLayout


def _write(path: Path, rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows, encoding="utf-8")


def test_attribute_run_aggregates_sequences_equally_and_leaves_unknown_empty(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt"
    tracker_root = tmp_path / "results" / "LRSST"
    perfect = "0,0,20,20\n" * 20
    offset = "10,0,20,20\n" * 2
    _write(gt / "vis-ball.txt", perfect)
    _write(gt / "nir-car59.txt", "0,0,20,20\n" * 2)
    _write(tracker_root / "vis-ball.txt", perfect)
    _write(tracker_root / "nir-car59.txt", offset)
    tracker = Tracker(
        tracker_id="lrsst",
        display_name="LRSST",
        results_path=tracker_root,
        style=default_style_for_index(0),
        layout=ResultLayout.PREFIXED_FLAT,
    )
    project = Project(
        name="HOT2023",
        sources=ProjectSources(ground_truth=gt, results_root=tmp_path / "results"),
        trackers=[tracker],
    )
    attributes = (
        AttributeDefinition("MIX", ("vis-ball", "nir-car59")),
        AttributeDefinition("UNKNOWN", ("vis-not-there",)),
    )

    run = run_attribute_evaluation(project, attributes)

    mixed = run.score("lrsst", "MIX")
    assert mixed is not None
    assert mixed.succeeded_count == 2
    assert mixed.resolved_count == 2
    perfect_metrics = compute_sequence_metrics(
        [[0, 0, 20, 20]] * 20,
        [[0, 0, 20, 20]] * 20,
    )
    offset_metrics = compute_sequence_metrics(
        [[10, 0, 20, 20]] * 2,
        [[0, 0, 20, 20]] * 2,
    )
    assert mixed.auc == pytest.approx((perfect_metrics.auc + offset_metrics.auc) / 2)
    assert mixed.dp == pytest.approx((perfect_metrics.dp + offset_metrics.dp) / 2)
    unknown = run.score("lrsst", "UNKNOWN")
    assert unknown is not None
    assert unknown.auc is None
    assert unknown.dp is None
    assert run.unknown_by_attribute["UNKNOWN"] == ("vis-not-there",)
