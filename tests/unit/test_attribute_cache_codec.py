from hsi_plotter.cache.attribute_codec import decode_attribute_run, encode_attribute_run
from hsi_plotter.models.attribute_evaluation import (
    AttributeDefinition,
    AttributeEvaluationRun,
    AttributeScore,
    TrackerAttributeResult,
)


def test_attribute_run_cache_round_trip_preserves_none() -> None:
    run = AttributeEvaluationRun(
        attributes=(AttributeDefinition("SV", ("vis-ball",)),),
        trackers=(
            TrackerAttributeResult(
                "mht",
                "MHT",
                {"SV": AttributeScore(None, None, 1, 0, 1)},
            ),
        ),
        unknown_by_attribute={"SV": ("vis-missing",)},
        issues=("one issue",),
    )

    restored = decode_attribute_run(encode_attribute_run(run))

    assert restored == run
    assert restored.trackers[0].scores["SV"].auc is None
