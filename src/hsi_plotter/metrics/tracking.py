from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

DISTANCE_THRESHOLDS = np.arange(0.0, 51.0, 1.0, dtype=np.float64)
SUCCESS_THRESHOLDS = np.linspace(0.0, 1.0, 51, dtype=np.float64)


class MetricsError(ValueError):
    """Raised when tracking metrics cannot be computed unambiguously."""


@dataclass(frozen=True, slots=True)
class SequenceMetrics:
    precision_curve: NDArray[np.float64]
    success_curve: NDArray[np.float64]
    distances: NDArray[np.float64]
    overlaps: NDArray[np.float64]
    auc: float
    dp: float
    cle: float
    total_frames: int
    valid_frames: int
    invalid_frames: int


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    precision_curve: NDArray[np.float64]
    success_curve: NDArray[np.float64]
    auc: float
    dp: float
    cle: float
    sequence_count: int
    total_frames: int
    valid_frames: int
    invalid_frames: int


def _as_box_matrix(values: ArrayLike, label: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise MetricsError(f"{label} 必须是 N x 4 矩阵")
    return matrix


def _validate_ground_truth(matrix: NDArray[np.float64]) -> None:
    if not np.isfinite(matrix).all():
        raise MetricsError("Ground Truth 不允许包含 NaN 或无穷值")


def _prediction_valid_mask(matrix: NDArray[np.float64]) -> NDArray[np.bool_]:
    nan_mask = np.isnan(matrix)
    complete_nan_rows = nan_mask.all(axis=1)
    partial_nan_rows = nan_mask.any(axis=1) & ~complete_nan_rows
    if partial_nan_rows.any():
        raise MetricsError("预测结果包含部分 NaN 行")
    valid = ~complete_nan_rows
    if np.isinf(matrix[valid]).any():
        raise MetricsError("预测结果包含无穷值")
    return valid


def _centers(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    return boxes[:, :2] + (boxes[:, 2:] - 1.0) / 2.0


def _intersection_over_union(
    predictions: NDArray[np.float64], ground_truth: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Preserve the legacy MATLAB implementation exactly.  It first converts
    # [x, y, w, h] to centres using (size - 1) / 2, then intersects the
    # centre/size representation.  Finite non-positive sizes (notably the
    # HOT convention [-1, -1, -1, -1]) are deliberately retained.
    prediction_centres = _centers(predictions)
    ground_truth_centres = _centers(ground_truth)
    intersection_size = np.minimum(
        prediction_centres + predictions[:, 2:] / 2.0,
        ground_truth_centres + ground_truth[:, 2:] / 2.0,
    ) - np.maximum(
        prediction_centres - predictions[:, 2:] / 2.0,
        ground_truth_centres - ground_truth[:, 2:] / 2.0,
    )
    intersection_size[intersection_size < 0.0] = 0.0
    intersection_area = intersection_size[:, 0] * intersection_size[:, 1]
    prediction_area = predictions[:, 2] * predictions[:, 3]
    ground_truth_area = ground_truth[:, 2] * ground_truth[:, 3]
    union_area = prediction_area + ground_truth_area - intersection_area
    with np.errstate(divide="ignore", invalid="ignore"):
        return intersection_area / union_area


def compute_box_overlap(prediction: ArrayLike, ground_truth: ArrayLike) -> float:
    """Return the legacy MATLAB overlap for one ``[x, y, w, h]`` pair."""
    prediction_row = np.asarray(prediction, dtype=np.float64)
    ground_truth_row = np.asarray(ground_truth, dtype=np.float64)
    if prediction_row.shape != (4,) or ground_truth_row.shape != (4,):
        raise MetricsError("单帧边界框必须包含 4 个数值")
    if not np.isfinite(prediction_row).all() or not np.isfinite(ground_truth_row).all():
        raise MetricsError("单帧边界框不允许包含 NaN 或无穷值")
    return float(
        _intersection_over_union(
            prediction_row[np.newaxis, :],
            ground_truth_row[np.newaxis, :],
        )[0]
    )


def compute_sequence_metrics(predictions: ArrayLike, ground_truth: ArrayLike) -> SequenceMetrics:
    prediction_matrix = _as_box_matrix(predictions, "预测结果")
    ground_truth_matrix = _as_box_matrix(ground_truth, "Ground Truth")
    if prediction_matrix.shape[0] != ground_truth_matrix.shape[0]:
        raise MetricsError(
            "预测结果与 Ground Truth 帧数不一致："
            f"{prediction_matrix.shape[0]} != {ground_truth_matrix.shape[0]}"
        )
    _validate_ground_truth(ground_truth_matrix)
    valid_mask = _prediction_valid_mask(prediction_matrix)
    valid_frames = int(valid_mask.sum())
    if valid_frames == 0:
        raise MetricsError("序列不存在任何有效预测帧")

    valid_predictions = prediction_matrix[valid_mask]
    valid_ground_truth = ground_truth_matrix[valid_mask]
    distances = np.linalg.norm(_centers(valid_predictions) - _centers(valid_ground_truth), axis=1)
    overlaps = _intersection_over_union(valid_predictions, valid_ground_truth)

    precision_curve = np.mean(distances[:, np.newaxis] < DISTANCE_THRESHOLDS[np.newaxis, :], axis=0)
    success_curve = np.mean(overlaps[:, np.newaxis] >= SUCCESS_THRESHOLDS[np.newaxis, :], axis=0)
    total_frames = int(prediction_matrix.shape[0])
    return SequenceMetrics(
        precision_curve=precision_curve,
        success_curve=success_curve,
        distances=distances,
        overlaps=overlaps,
        auc=float(np.mean(success_curve[1:])),
        dp=float(precision_curve[20]),
        cle=float(np.mean(distances)),
        total_frames=total_frames,
        valid_frames=valid_frames,
        invalid_frames=total_frames - valid_frames,
    )


def aggregate_sequence_metrics(metrics: Sequence[SequenceMetrics]) -> AggregateMetrics:
    if not metrics:
        raise MetricsError("至少需要一个成功计算的序列")
    precision_curve = np.mean(np.stack([item.precision_curve for item in metrics], axis=0), axis=0)
    success_curve = np.mean(np.stack([item.success_curve for item in metrics], axis=0), axis=0)
    return AggregateMetrics(
        precision_curve=precision_curve,
        success_curve=success_curve,
        auc=float(np.mean(success_curve[1:])),
        dp=float(precision_curve[20]),
        cle=float(np.mean([item.cle for item in metrics])),
        sequence_count=len(metrics),
        total_frames=sum(item.total_frames for item in metrics),
        valid_frames=sum(item.valid_frames for item in metrics),
        invalid_frames=sum(item.invalid_frames for item in metrics),
    )
