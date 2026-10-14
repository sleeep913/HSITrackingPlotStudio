from hsi_plotter.models.attribute_evaluation import (
    AttributeDefinition,
    AttributeEvaluationRun,
    AttributeScore,
    TrackerAttributeResult,
)


def test_attribute_result_exposes_metric_values_and_lookup() -> None:
    definition = AttributeDefinition(name="OCC", sequences=("vis-ball", "nir-car59"))
    score = AttributeScore(
        auc=0.625,
        dp=0.875,
        resolved_count=2,
        succeeded_count=2,
        failed_count=0,
    )
    tracker = TrackerAttributeResult(
        tracker_id="lrsst",
        tracker_name="LRSST",
        scores={"OCC": score},
    )
    run = AttributeEvaluationRun(attributes=(definition,), trackers=(tracker,))

    assert run.score("lrsst", "OCC") is score
    assert run.score("missing", "OCC") is None
    assert score.is_computable


def test_empty_attribute_score_is_not_zero() -> None:
    score = AttributeScore(
        auc=None,
        dp=None,
        resolved_count=0,
        succeeded_count=0,
        failed_count=0,
    )

    assert score.auc is None
    assert score.dp is None
    assert not score.is_computable

