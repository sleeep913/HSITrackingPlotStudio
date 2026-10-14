from __future__ import annotations

from pathlib import Path

from hsi_plotter.io.tracking_reader import (
    BoundingBoxData,
    TrackingDataError,
    read_bounding_boxes,
)
from hsi_plotter.models.dataset import SequenceKey
from hsi_plotter.validation.issues import (
    PreflightReport,
    Severity,
    ValidationIssue,
)


def _read_for_preflight(
    path: Path,
    *,
    allow_missing_rows: bool,
    error_code: str,
    tracker: str,
    sequence: SequenceKey,
    report: PreflightReport,
) -> BoundingBoxData | None:
    try:
        return read_bounding_boxes(path, allow_missing_rows=allow_missing_rows)
    except TrackingDataError as exc:
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code=error_code,
                message=str(exc),
                path=Path(path).resolve(strict=False),
                tracker=tracker,
                sequence=sequence.identifier,
                modality=sequence.modality.value,
                suggestion="修正文件内容后重新执行数据预检。",
            )
        )
        return None


def validate_tracking_pair(
    ground_truth_path: Path,
    result_path: Path,
    *,
    tracker: str,
    sequence: SequenceKey,
) -> PreflightReport:
    """Validate one Ground Truth / tracker-result pair down to file content."""
    report = PreflightReport()
    ground_truth = _read_for_preflight(
        ground_truth_path,
        allow_missing_rows=False,
        error_code="INVALID_GROUND_TRUTH",
        tracker=tracker,
        sequence=sequence,
        report=report,
    )
    result = _read_for_preflight(
        result_path,
        allow_missing_rows=True,
        error_code="INVALID_TRACKER_RESULT",
        tracker=tracker,
        sequence=sequence,
        report=report,
    )
    if ground_truth is None or result is None:
        return report

    if ground_truth.frame_count != result.frame_count:
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code="FRAME_COUNT_MISMATCH",
                message=(
                    f"Ground Truth 为 {ground_truth.frame_count} 帧，"
                    f"跟踪结果为 {result.frame_count} 帧。"
                ),
                path=result.source_path,
                tracker=tracker,
                sequence=sequence.identifier,
                modality=sequence.modality.value,
                suggestion="确保两个文件逐帧对应且行数相同。",
            )
        )
    if result.invalid_frame_indices:
        report.add(
            ValidationIssue(
                severity=Severity.WARNING,
                code="MISSING_RESULT_FRAMES",
                message=(
                    f"跟踪结果包含 {len(result.invalid_frame_indices)} 个完整 NaN 缺失帧："
                    + ", ".join(map(str, result.invalid_frame_indices[:10]))
                ),
                path=result.source_path,
                tracker=tracker,
                sequence=sequence.identifier,
                modality=sequence.modality.value,
                suggestion="确认这些帧确实表示跟踪失败；指标计算将排除这些帧。",
            )
        )
    return report
