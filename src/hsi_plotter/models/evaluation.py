"""Result models for the evaluation module.

The models are pure data containers: no file reading, no Qt import and no
metric formula lives here (实施计划 原则 4).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from hsi_plotter.metrics.tracking import AggregateMetrics, SequenceMetrics
from hsi_plotter.models.dataset import MODALITY_LABELS, Modality, SequenceKey
from hsi_plotter.models.tracker_import import ResultLayout
from hsi_plotter.validation.issues import Severity, ValidationIssue

EVALUATION_MODALITIES: tuple[Modality, ...] = (
    Modality.VIS,
    Modality.REDNIR,
    Modality.NIR,
)


class EvaluationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


STATUS_LABELS = {
    EvaluationStatus.SUCCEEDED: "成功",
    EvaluationStatus.FAILED: "失败",
    EvaluationStatus.SKIPPED: "已跳过",
}


@dataclass(frozen=True, slots=True)
class SequenceEvaluation:
    """Metrics for one Ground Truth / tracker-result pair."""

    key: SequenceKey
    ground_truth_path: Path
    result_path: Path
    status: EvaluationStatus
    metrics: SequenceMetrics | None = None
    invalid_frame_indices: tuple[int, ...] = ()
    message: str = ""

    @property
    def identifier(self) -> str:
        return self.key.identifier

    @property
    def sequence_name(self) -> str:
        return self.key.name

    @property
    def modality(self) -> Modality:
        return self.key.modality

    @property
    def modality_label(self) -> str:
        return MODALITY_LABELS[self.key.modality]

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def is_succeeded(self) -> bool:
        return self.status is EvaluationStatus.SUCCEEDED and self.metrics is not None

    @property
    def auc(self) -> float | None:
        return self.metrics.auc if self.metrics is not None else None

    @property
    def dp(self) -> float | None:
        return self.metrics.dp if self.metrics is not None else None

    @property
    def cle(self) -> float | None:
        return self.metrics.cle if self.metrics is not None else None

    @property
    def total_frames(self) -> int:
        return self.metrics.total_frames if self.metrics is not None else 0

    @property
    def valid_frames(self) -> int:
        return self.metrics.valid_frames if self.metrics is not None else 0

    @property
    def invalid_frames(self) -> int:
        return self.metrics.invalid_frames if self.metrics is not None else 0


@dataclass(slots=True)
class TrackerEvaluation:
    """All results of one tracker: per sequence, per modality and overall."""

    tracker_id: str
    tracker_name: str
    layout: ResultLayout
    sequences: tuple[SequenceEvaluation, ...] = ()
    overall: AggregateMetrics | None = None
    modalities: dict[Modality, AggregateMetrics] = field(default_factory=dict)
    missing_count: int = 0
    extra_count: int = 0
    ambiguous_count: int = 0
    cancelled: bool = False
    message: str = ""

    @property
    def succeeded_sequences(self) -> tuple[SequenceEvaluation, ...]:
        return tuple(item for item in self.sequences if item.is_succeeded)

    @property
    def failed_sequences(self) -> tuple[SequenceEvaluation, ...]:
        return tuple(item for item in self.sequences if item.status is EvaluationStatus.FAILED)

    @property
    def skipped_sequences(self) -> tuple[SequenceEvaluation, ...]:
        return tuple(item for item in self.sequences if item.status is EvaluationStatus.SKIPPED)

    @property
    def succeeded_count(self) -> int:
        return len(self.succeeded_sequences)

    @property
    def failed_count(self) -> int:
        return len(self.failed_sequences)

    @property
    def status(self) -> EvaluationStatus:
        if self.overall is None:
            return EvaluationStatus.FAILED
        if self.cancelled or self.skipped_sequences:
            return EvaluationStatus.SKIPPED
        return EvaluationStatus.SUCCEEDED

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def total_frames(self) -> int:
        return sum(item.total_frames for item in self.succeeded_sequences)

    @property
    def valid_frames(self) -> int:
        return sum(item.valid_frames for item in self.succeeded_sequences)

    @property
    def invalid_frames(self) -> int:
        return sum(item.invalid_frames for item in self.succeeded_sequences)

    def modality_metrics(self, modality: Modality) -> AggregateMetrics | None:
        return self.modalities.get(modality)

    @property
    def modality_labels(self) -> str:
        covered = sorted(self.modalities, key=lambda item: item.value)
        if not covered:
            return "—"
        return ", ".join(MODALITY_LABELS[modality] for modality in covered)

    @property
    def auc(self) -> float | None:
        return self.overall.auc if self.overall is not None else None

    @property
    def dp(self) -> float | None:
        return self.overall.dp if self.overall is not None else None

    @property
    def cle(self) -> float | None:
        return self.overall.cle if self.overall is not None else None


@dataclass(slots=True)
class EvaluationRun:
    """Complete outcome of one user-triggered evaluation."""

    trackers: tuple[TrackerEvaluation, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()
    cancelled: bool = False

    @property
    def has_results(self) -> bool:
        return any(tracker.overall is not None for tracker in self.trackers)

    @property
    def tracker_map(self) -> dict[str, TrackerEvaluation]:
        return {tracker.tracker_id: tracker for tracker in self.trackers}

    @property
    def sequence_count(self) -> int:
        return sum(len(tracker.sequences) for tracker in self.trackers)

    @property
    def succeeded_sequence_count(self) -> int:
        return sum(tracker.succeeded_count for tracker in self.trackers)

    @property
    def failed_sequence_count(self) -> int:
        return sum(tracker.failed_count for tracker in self.trackers)


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    """One unit of work: a single sequence pair of a single tracker."""

    tracker_id: str
    tracker_name: str
    layout: ResultLayout
    key: SequenceKey
    ground_truth_path: Path
    result_path: Path


@dataclass(slots=True)
class EvaluationPlan:
    """Prepared and validated work list for one evaluation run."""

    ground_truth_files: Mapping[SequenceKey, Path] = field(default_factory=dict)
    tracker_ids: tuple[str, ...] = ()
    tasks: tuple[EvaluationTask, ...] = ()
    tracker_names: Mapping[str, str] = field(default_factory=dict)
    tracker_layouts: Mapping[str, ResultLayout] = field(default_factory=dict)
    missing_counts: Mapping[str, int] = field(default_factory=dict)
    extra_counts: Mapping[str, int] = field(default_factory=dict)
    ambiguous_counts: Mapping[str, int] = field(default_factory=dict)
    modality_filter: tuple[Modality, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def has_blocking_errors(self) -> bool:
        if not self.tasks:
            return True
        return any(issue.severity is Severity.ERROR for issue in self.issues)

    @property
    def total_units(self) -> int:
        return len(self.tasks)

    def tracker_name(self, tracker_id: str) -> str:
        return self.tracker_names.get(tracker_id, tracker_id)

    def layout_of(self, tracker_id: str) -> ResultLayout:
        return self.tracker_layouts.get(tracker_id, ResultLayout.FLAT)

    def missing_count_of(self, tracker_id: str) -> int:
        return int(self.missing_counts.get(tracker_id, 0))

    def extra_count_of(self, tracker_id: str) -> int:
        return int(self.extra_counts.get(tracker_id, 0))

    def ambiguous_count_of(self, tracker_id: str) -> int:
        return int(self.ambiguous_counts.get(tracker_id, 0))

    def tasks_for(self, tracker_id: str) -> tuple[EvaluationTask, ...]:
        return tuple(task for task in self.tasks if task.tracker_id == tracker_id)

    def iterate_groups(self) -> Iterator[tuple[str, tuple[EvaluationTask, ...]]]:
        for tracker_id in self.tracker_ids:
            yield tracker_id, self.tasks_for(tracker_id)

    def modality_label(self) -> str:
        if not self.modality_filter:
            return "全部模态"
        return ", ".join(
            MODALITY_LABELS[modality] for modality in sorted_modalities(self.modality_filter)
        )


def sorted_modalities(modalities: Sequence[Modality]) -> tuple[Modality, ...]:
    return tuple(sorted(set(modalities), key=lambda item: item.value))
