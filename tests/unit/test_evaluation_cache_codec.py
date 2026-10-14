import numpy as np

from hsi_plotter.cache.evaluation_codec import decode_evaluation_run, encode_evaluation_run
from hsi_plotter.metrics.tracking import AggregateMetrics
from hsi_plotter.models.evaluation import EvaluationRun, TrackerEvaluation
from hsi_plotter.models.tracker_import import ResultLayout


def test_evaluation_cache_round_trip_preserves_curves() -> None:
    metrics = AggregateMetrics(
        precision_curve=np.linspace(0, 1, 51),
        success_curve=np.linspace(1, 0, 51),
        auc=0.61,
        dp=0.82,
        cle=4.5,
        sequence_count=2,
        total_frames=20,
        valid_frames=19,
        invalid_frames=1,
    )
    run = EvaluationRun(
        trackers=(
            TrackerEvaluation(
                "mht",
                "MHT",
                ResultLayout.FLAT,
                overall=metrics,
            ),
        )
    )

    restored = decode_evaluation_run(encode_evaluation_run(run))

    assert restored.trackers[0].auc == 0.61
    assert np.array_equal(
        restored.trackers[0].overall.precision_curve, metrics.precision_curve
    )
