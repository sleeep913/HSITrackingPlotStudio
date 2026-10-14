from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from hsi_plotter.io.sequence_discovery import discover_sequence_files
from hsi_plotter.io.tracker_discovery import build_layout_candidate
from hsi_plotter.models.dataset import SequenceKey
from hsi_plotter.models.project import Project
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.tracker_import import LayoutCandidate
from hsi_plotter.validation.issues import (
    PreflightReport,
    Severity,
    ValidationIssue,
)
from hsi_plotter.validation.project_sources import validate_project_sources
from hsi_plotter.validation.sequence_mapping import (
    SequenceMapping,
    build_sequence_mapping,
    validate_sequence_mapping,
)
from hsi_plotter.validation.tracking_pair import validate_tracking_pair

__all__ = [
    "DEFAULT_DATASET_NAME",
    "ProjectPreflight",
    "build_tracker_mapping",
    "collect_project_preflight",
    "discover_ground_truth_files",
    "validate_tracking_pair",
]

#: Sequence keys combine dataset, modality and name; V1 uses a single built-in
#: dataset per project.
DEFAULT_DATASET_NAME = "project"


@dataclass(slots=True)
class ProjectPreflight:
    """Result of a complete project validation pass."""

    report: PreflightReport = field(default_factory=PreflightReport)
    ground_truth_files: dict[SequenceKey, Path] = field(default_factory=dict)
    mappings: dict[str, SequenceMapping] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return self.report.has_errors


def discover_ground_truth_files(
    project: Project,
) -> tuple[dict[SequenceKey, Path], PreflightReport]:
    report = PreflightReport()
    ground_truth = project.sources.ground_truth
    if ground_truth is None or not (
        ground_truth.is_dir()
        or (ground_truth.is_file() and ground_truth.suffix.casefold() == ".csv")
    ):
        return {}, report
    discovery = discover_sequence_files(ground_truth, dataset=DEFAULT_DATASET_NAME)
    report.extend(discovery.report.issues)
    return discovery.files, report


def build_tracker_mapping(
    ground_truth_files: Mapping[SequenceKey, Path],
    tracker: Tracker,
    *,
    dataset: str = DEFAULT_DATASET_NAME,
) -> SequenceMapping:
    """Re-index a configured tracker and pair it with the Ground Truth files."""
    resolved = build_layout_candidate(
        tracker.results_path,
        tracker.layout,
        dataset=dataset,
        tracker=tracker.tracker_id,
    )
    if resolved is None:
        candidate = LayoutCandidate(
            layout=tracker.layout,
            root_path=Path(tracker.results_path).resolve(strict=False),
        )
        layout_issues: list[ValidationIssue] = [
            ValidationIssue(
                severity=Severity.WARNING,
                code="TRACKER_NO_RESULT_FILES",
                message=(
                    f"跟踪器 {tracker.display_name} 在所选布局 "
                    f"{tracker.layout.value} 下没有发现结果文件。"
                ),
                path=candidate.root_path,
                tracker=tracker.display_name,
                suggestion="重新扫描跟踪器并在导入页面确认结果布局。",
            )
        ]
    else:
        candidate, layout_issues = resolved

    mapping = build_sequence_mapping(ground_truth_files, candidate, tracker=tracker.display_name)
    if not layout_issues:
        return mapping
    return replace(
        mapping,
        report=PreflightReport(issues=[*layout_issues, *mapping.report.issues]),
    )


def collect_project_preflight(
    project: Project,
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ProjectPreflight:
    """Validate sources, Ground Truth files and every configured tracker."""
    report = validate_project_sources(project.sources)
    preflight = ProjectPreflight(report=report)
    if report.has_errors or project.sources.ground_truth is None:
        return preflight

    ground_truth_files, discovery_report = discover_ground_truth_files(project)
    report.extend(discovery_report.issues)
    preflight.ground_truth_files = ground_truth_files
    if not ground_truth_files:
        return preflight

    mappings = {
        tracker.tracker_id: build_tracker_mapping(ground_truth_files, tracker)
        for tracker in project.trackers
    }
    preflight.mappings = mappings
    for mapping in mappings.values():
        report.extend(mapping.report.issues)
    if not mappings:
        return preflight

    total_units = max(sum(len(mapping.pairs) for mapping in mappings.values()), 1)
    completed = 0
    for tracker in project.trackers:
        mapping = mappings.get(tracker.tracker_id)
        if mapping is None:
            continue
        offset = completed

        def forward(
            index: int,
            message: str,
            *,
            offset: int = offset,
            name: str = tracker.display_name,
        ) -> None:
            if on_progress is not None:
                on_progress(offset + index, total_units, f"{name} · {message}")

        if is_cancelled is not None and is_cancelled():
            report.add(
                ValidationIssue(
                    severity=Severity.INFO,
                    code="PREFLIGHT_CANCELLED",
                    message="预检已被取消。",
                    suggestion="重新运行预检以完成全部检查。",
                )
            )
            break
        report.extend(
            validate_sequence_mapping(
                mapping,
                on_progress=forward,
                is_cancelled=is_cancelled,
            ).issues
        )
        completed += len(mapping.pairs)
    return preflight
