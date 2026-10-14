"""Orchestrates metric computation for a prepared evaluation plan.

Failures are isolated at sequence level: one broken file never removes results
already computed for other sequences or other trackers (AC-02).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from hsi_plotter.io.tracking_reader import TrackingDataError, read_bounding_boxes
from hsi_plotter.metrics.tracking import (
    AggregateMetrics,
    MetricsError,
    SequenceMetrics,
    aggregate_sequence_metrics,
    compute_sequence_metrics,
)
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.evaluation import (
    EvaluationPlan,
    EvaluationRun,
    EvaluationStatus,
    EvaluationTask,
    SequenceEvaluation,
    TrackerEvaluation,
)
from hsi_plotter.models.project import Project
from hsi_plotter.models.tracker_import import ResultLayout, SequenceMapping
from hsi_plotter.validation.issues import Severity, ValidationIssue
from hsi_plotter.validation.preflight import (
    DEFAULT_DATASET_NAME,
    build_tracker_mapping,
    discover_ground_truth_files,
)
from hsi_plotter.validation.project_sources import validate_project_sources

_SEQUENCE_ERRORS = (TrackingDataError, MetricsError, ValueError, OSError, UnicodeError)

ProgressCallback = Callable[[int, int, str], None]
CancelledCallback = Callable[[], bool]


def _sorted_keys(keys: set[SequenceKey]) -> list[SequenceKey]:
    return sorted(keys, key=lambda item: item.identifier)


def prepare_evaluation_inputs(
    project: Project,
    *,
    tracker_ids: tuple[str, ...] | None = None,
    modality_filter: tuple[Modality, ...] = (),
    dataset: str = DEFAULT_DATASET_NAME,
) -> EvaluationPlan:
    """Re-index every selected tracker and produce a deterministic work list."""
    source_report = validate_project_sources(project.sources)
    issues: list[ValidationIssue] = list(source_report.issues)

    ground_truth_files: Mapping[SequenceKey, Path] = {}
    if not source_report.has_errors:
        ground_truth_files, discovery_report = discover_ground_truth_files(project)
        issues.extend(discovery_report.issues)
        if not ground_truth_files:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="GROUND_TRUTH_NO_SEQUENCES",
                    message="Ground Truth 目录中未发现任何可用于计算的序列文件。",
                    path=project.sources.ground_truth,
                    suggestion="确认 Ground Truth 目录包含 VIS/NIR/RedNIR 的序列 TXT 文件。",
                )
            )

    wanted = None if tracker_ids is None else set(tracker_ids)
    selected_trackers = [
        tracker for tracker in project.trackers if wanted is None or tracker.tracker_id in wanted
    ]
    if not selected_trackers:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="NO_TRACKERS_SELECTED",
                message="尚未选择任何跟踪器，无法计算指标。",
                suggestion="先在跟踪器导入页导入并勾选至少一个跟踪器。",
            )
        )

    plan = EvaluationPlan(
        ground_truth_files=ground_truth_files,
        tracker_ids=tuple(tracker.tracker_id for tracker in selected_trackers),
        tracker_names={t.tracker_id: t.display_name for t in selected_trackers},
        tracker_layouts={t.tracker_id: t.layout for t in selected_trackers},
        missing_counts={},
        extra_counts={},
        ambiguous_counts={},
        # V1.1 always evaluates every available modality.  Keep the argument
        # temporarily for source compatibility with callers from V1.0, but it
        # intentionally has no effect and is never persisted in the plan.
        modality_filter=(),
    )
    tasks: list[EvaluationTask] = []
    missing_counts: dict[str, int] = {}
    extra_counts: dict[str, int] = {}
    ambiguous_counts: dict[str, int] = {}

    for tracker in selected_trackers:
        if not ground_truth_files:
            missing_counts[tracker.tracker_id] = 0
            extra_counts[tracker.tracker_id] = 0
            ambiguous_counts[tracker.tracker_id] = 0
            continue
        mapping = build_tracker_mapping(ground_truth_files, tracker, dataset=dataset)
        issues.extend(mapping.report.issues)
        missing_counts[tracker.tracker_id] = mapping.missing_count
        extra_counts[tracker.tracker_id] = mapping.extra_count
        ambiguous_counts[tracker.tracker_id] = mapping.ambiguous_count
        for key in _sorted_keys(set(mapping.pairs)):
            ground_truth_path, result_path = mapping.pairs[key]
            tasks.append(
                EvaluationTask(
                    tracker_id=tracker.tracker_id,
                    tracker_name=tracker.display_name,
                    layout=mapping.layout,
                    key=key,
                    ground_truth_path=ground_truth_path,
                    result_path=result_path,
                )
            )

    plan.tasks = tuple(tasks)
    plan.missing_counts = missing_counts
    plan.extra_counts = extra_counts
    plan.ambiguous_counts = ambiguous_counts
    plan.issues = tuple(issues)
    return plan


def evaluate_sequence(task: EvaluationTask) -> SequenceEvaluation:
    """Compute one sequence, converting any data problem into a failed state."""
    try:
        ground_truth = read_bounding_boxes(task.ground_truth_path, allow_missing_rows=False)
        predictions = read_bounding_boxes(task.result_path)
        metrics = compute_sequence_metrics(predictions.values, ground_truth.values)
    except _SEQUENCE_ERRORS as exc:
        return SequenceEvaluation(
            key=task.key,
            ground_truth_path=task.ground_truth_path,
            result_path=task.result_path,
            status=EvaluationStatus.FAILED,
            message=str(exc),
        )
    return SequenceEvaluation(
        key=task.key,
        ground_truth_path=task.ground_truth_path,
        result_path=task.result_path,
        status=EvaluationStatus.SUCCEEDED,
        metrics=metrics,
        invalid_frame_indices=predictions.invalid_frame_indices,
    )


def _aggregate_grouped(
    succeeded: tuple[SequenceEvaluation, ...],
) -> dict[Modality, AggregateMetrics]:
    grouped: dict[Modality, list[SequenceMetrics]] = {}
    for item in succeeded:
        if item.metrics is None:  # pragma: no cover - guarded by selection
            continue
        grouped.setdefault(item.key.modality, []).append(item.metrics)
    aggregates: dict[Modality, AggregateMetrics] = {}
    for modality in sorted(grouped, key=lambda item: item.value):
        aggregates[modality] = aggregate_sequence_metrics(grouped[modality])
    return aggregates


def evaluate_tracker_group(
    tracker_id: str,
    tasks: tuple[EvaluationTask, ...],
    *,
    tracker_name: str,
    layout: ResultLayout = ResultLayout.FLAT,
    missing_count: int = 0,
    extra_count: int = 0,
    ambiguous_count: int = 0,
    on_progress: ProgressCallback | None = None,
    is_cancelled: CancelledCallback | None = None,
) -> TrackerEvaluation:
    """Evaluate every sequence of one tracker, isolating failures."""
    sequences: list[SequenceEvaluation] = []
    cancelled = False
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        if is_cancelled is not None and is_cancelled():
            cancelled = True
            for remaining in tasks[index - 1 :]:
                sequences.append(
                    SequenceEvaluation(
                        key=remaining.key,
                        ground_truth_path=remaining.ground_truth_path,
                        result_path=remaining.result_path,
                        status=EvaluationStatus.SKIPPED,
                        message="任务已取消，该序列未计算。",
                    )
                )
            break
        evaluation = evaluate_sequence(task)
        sequences.append(evaluation)
        if on_progress is not None:
            on_progress(index, total, task.key.identifier)

    succeeded = tuple(item for item in sequences if item.is_succeeded)
    metrics = tuple(item.metrics for item in succeeded if item.metrics is not None)
    overall = aggregate_sequence_metrics(metrics) if metrics else None
    messages: list[str] = []
    if tasks and overall is None:
        messages.append("所选范围内没有成功计算的序列，未生成聚合指标。")
    elif failed := [item for item in sequences if item.status is EvaluationStatus.FAILED]:
        messages.append(f"{len(failed)} 个序列计算失败，已在失败序列详情中列出。")
    if cancelled:
        messages.append("任务已取消，未完成序列保持跳过状态。")

    return TrackerEvaluation(
        tracker_id=tracker_id,
        tracker_name=tracker_name,
        layout=layout,
        sequences=tuple(sequences),
        overall=overall,
        modalities=_aggregate_grouped(succeeded),
        missing_count=missing_count,
        extra_count=extra_count,
        ambiguous_count=ambiguous_count,
        cancelled=cancelled,
        message=" ".join(messages),
    )


def run_evaluation(
    plan: EvaluationPlan,
    *,
    on_progress: ProgressCallback | None = None,
    is_cancelled: CancelledCallback | None = None,
) -> EvaluationRun:
    """Evaluate every tracker of the plan, preserving completed results on cancel."""
    total_units = max(plan.total_units, 1)
    completed = 0
    evaluations: list[TrackerEvaluation] = []
    cancelled = False
    remaining_trackers: list[str] = []

    for tracker_id, tasks in plan.iterate_groups():
        name = plan.tracker_name(tracker_id)
        layout = plan.layout_of(tracker_id)
        offset = completed

        def forward(
            index: int,
            count: int,
            message: str,
            *,
            offset: int = offset,
            name: str = name,
        ) -> None:
            if on_progress is not None:
                on_progress(offset + index, total_units, f"{name} · {message}")

        if cancelled:
            remaining_trackers.append(tracker_id)
            continue
        evaluation = evaluate_tracker_group(
            tracker_id,
            tasks,
            tracker_name=name,
            layout=layout,
            missing_count=plan.missing_count_of(tracker_id),
            extra_count=plan.extra_count_of(tracker_id),
            ambiguous_count=plan.ambiguous_count_of(tracker_id),
            on_progress=forward,
            is_cancelled=is_cancelled,
        )
        evaluations.append(evaluation)
        completed += len(tasks)
        if evaluation.cancelled:
            cancelled = True

    for tracker_id in remaining_trackers:
        evaluations.append(
            TrackerEvaluation(
                tracker_id=tracker_id,
                tracker_name=plan.tracker_name(tracker_id),
                layout=plan.layout_of(tracker_id),
                message="任务已取消，该跟踪器未计算。",
                cancelled=True,
            )
        )

    return EvaluationRun(
        trackers=tuple(evaluations),
        issues=plan.issues,
        cancelled=cancelled,
    )


def evaluate_project(
    project: Project,
    *,
    tracker_ids: tuple[str, ...] | None = None,
    modality_filter: tuple[Modality, ...] = (),
    on_progress: ProgressCallback | None = None,
    is_cancelled: CancelledCallback | None = None,
) -> tuple[EvaluationPlan, EvaluationRun]:
    """Convenience entry point used by tests and batch tasks."""
    plan = prepare_evaluation_inputs(
        project, tracker_ids=tracker_ids, modality_filter=modality_filter
    )
    run = run_evaluation(plan, on_progress=on_progress, is_cancelled=is_cancelled)
    return plan, run


def planned_sequences(mapping: SequenceMapping) -> tuple[SequenceKey, ...]:
    """Public helper kept for callers that only need the key order."""
    return tuple(_sorted_keys(set(mapping.pairs)))


__all__ = [
    "evaluate_project",
    "evaluate_sequence",
    "evaluate_tracker_group",
    "planned_sequences",
    "prepare_evaluation_inputs",
    "run_evaluation",
]
