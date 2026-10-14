import numpy as np
import pytest

from hsi_plotter.metrics.tracking import (
    MetricsError,
    aggregate_sequence_metrics,
    compute_sequence_metrics,
)


def test_identical_boxes_match_matlab_threshold_conventions() -> None:
    ground_truth = np.array([[0.0, 0.0, 10.0, 10.0]])
    predictions = ground_truth.copy()

    metrics = compute_sequence_metrics(predictions, ground_truth)

    assert metrics.precision_curve.shape == (51,)
    assert metrics.success_curve.shape == (51,)
    assert metrics.precision_curve[0] == 0.0
    np.testing.assert_array_equal(metrics.precision_curve[1:], 1.0)
    np.testing.assert_array_equal(metrics.success_curve, 1.0)
    assert metrics.dp == 1.0
    assert metrics.auc == 1.0
    assert metrics.cle == 0.0


def test_distance_precision_uses_strict_less_than_at_20_pixels() -> None:
    ground_truth = np.array([[0.0, 0.0, 3.0, 3.0]])
    predictions = np.array([[20.0, 0.0, 3.0, 3.0]])

    metrics = compute_sequence_metrics(predictions, ground_truth)

    assert metrics.precision_curve[20] == 0.0
    assert metrics.precision_curve[21] == 1.0
    assert metrics.dp == 0.0


def test_success_precision_uses_greater_than_or_equal_threshold() -> None:
    ground_truth = np.array([[0.0, 0.0, 3.0, 1.0]])
    predictions = np.array([[1.0, 0.0, 3.0, 1.0]])

    metrics = compute_sequence_metrics(predictions, ground_truth)

    assert metrics.overlaps[0] == pytest.approx(0.5)
    assert metrics.success_curve[25] == 1.0
    assert metrics.success_curve[26] == 0.0


def test_auc_excludes_zero_overlap_threshold() -> None:
    ground_truth = np.array([[0.0, 0.0, 2.0, 2.0]])
    predictions = np.array([[10.0, 10.0, 2.0, 2.0]])

    metrics = compute_sequence_metrics(predictions, ground_truth)

    assert metrics.success_curve[0] == 1.0
    np.testing.assert_array_equal(metrics.success_curve[1:], 0.0)
    assert metrics.auc == 0.0


def test_complete_nan_prediction_is_excluded_and_reported() -> None:
    ground_truth = np.array(
        [
            [0.0, 0.0, 3.0, 3.0],
            [10.0, 10.0, 3.0, 3.0],
        ]
    )
    predictions = np.array(
        [
            [0.0, 0.0, 3.0, 3.0],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )

    metrics = compute_sequence_metrics(predictions, ground_truth)

    assert metrics.total_frames == 2
    assert metrics.valid_frames == 1
    assert metrics.invalid_frames == 1
    assert metrics.auc == 1.0
    assert metrics.dp == 1.0


def test_minus_one_target_outside_frame_matches_matlab_legacy_math() -> None:
    ground_truth = np.array([[0.0, 0.0, 3.0, 3.0]])
    predictions = np.array([[-1.0, -1.0, -1.0, -1.0]])

    metrics = compute_sequence_metrics(predictions, ground_truth)

    # MATLAB converts both rows with (w - 1) / 2 before computing distance.
    assert metrics.distances[0] == pytest.approx(np.sqrt(18.0))
    assert metrics.overlaps[0] == pytest.approx(0.0)
    assert metrics.valid_frames == 1
    assert metrics.invalid_frames == 0
    assert metrics.dp == 1.0
    assert metrics.auc == 0.0


def test_finite_nonpositive_sizes_are_not_rejected() -> None:
    ground_truth = np.array([[0.0, 0.0, 0.0, 3.0]])
    predictions = np.array([[0.0, 0.0, 0.0, 3.0]])

    metrics = compute_sequence_metrics(predictions, ground_truth)

    assert metrics.dp == 1.0
    assert np.isnan(metrics.overlaps[0])


def test_zero_valid_predictions_raise_clear_error() -> None:
    ground_truth = np.array([[0.0, 0.0, 3.0, 3.0]])
    predictions = np.full((1, 4), np.nan)

    with pytest.raises(MetricsError, match="有效预测帧"):
        compute_sequence_metrics(predictions, ground_truth)


def test_frame_count_mismatch_is_rejected() -> None:
    ground_truth = np.ones((2, 4))
    predictions = np.ones((1, 4))

    with pytest.raises(MetricsError, match="帧数"):
        compute_sequence_metrics(predictions, ground_truth)


def test_aggregate_metrics_weights_sequences_equally() -> None:
    perfect = compute_sequence_metrics(
        np.array([[0.0, 0.0, 2.0, 2.0]]),
        np.array([[0.0, 0.0, 2.0, 2.0]]),
    )
    failed_long_sequence = compute_sequence_metrics(
        np.repeat([[10.0, 10.0, 2.0, 2.0]], repeats=9, axis=0),
        np.repeat([[0.0, 0.0, 2.0, 2.0]], repeats=9, axis=0),
    )

    aggregate = aggregate_sequence_metrics([perfect, failed_long_sequence])

    assert aggregate.auc == pytest.approx(0.5)
    assert aggregate.dp == pytest.approx(1.0)
    assert aggregate.sequence_count == 2
    assert aggregate.total_frames == 10
