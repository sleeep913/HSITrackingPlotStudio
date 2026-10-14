from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from hsi_plotter.io.image_sequence_discovery import discover_image_sequences
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.project import Project
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.visualization import ImageSequenceIndex
from hsi_plotter.validation.issues import Severity, ValidationIssue
from hsi_plotter.validation.preflight import (
    DEFAULT_DATASET_NAME,
    build_tracker_mapping,
    discover_ground_truth_files,
)
from hsi_plotter.visualization.session import (
    SequenceVisualizationSession,
    TrackerSequenceSource,
    load_sequence_session,
)


@dataclass(frozen=True, slots=True)
class ProjectVisualizationData:
    sequences: dict[SequenceKey, ImageSequenceIndex]
    ground_truth_files: dict[SequenceKey, Path]
    tracker_result_files: dict[str, dict[SequenceKey, Path]]
    trackers: tuple[Tracker, ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity is Severity.ERROR for issue in self.issues)

    def tracker(self, tracker_id: str) -> Tracker:
        for tracker in self.trackers:
            if tracker.tracker_id == tracker_id:
                return tracker
        raise KeyError(tracker_id)

    def load_session(
        self,
        key: SequenceKey,
        *,
        tracker_ids: tuple[str, ...] | None = None,
    ) -> SequenceVisualizationSession:
        wanted = None if tracker_ids is None else set(tracker_ids)
        sources = tuple(
            TrackerSequenceSource(tracker, result_path)
            for tracker in self.trackers
            if (wanted is None or tracker.tracker_id in wanted)
            and (result_path := self.tracker_result_files.get(tracker.tracker_id, {}).get(key))
            is not None
        )
        return load_sequence_session(
            self.sequences[key],
            self.ground_truth_files[key],
            sources,
        )


def _matching_ground_truth_keys(
    image_key: SequenceKey,
    ground_truth_keys: tuple[SequenceKey, ...],
) -> tuple[SequenceKey, ...]:
    same_name = tuple(
        key for key in ground_truth_keys if key.name.casefold() == image_key.name.casefold()
    )
    if image_key.modality is Modality.UNSPECIFIED:
        return same_name
    return tuple(key for key in same_name if key.modality is image_key.modality)


def prepare_visualization_project(
    project: Project,
    *,
    dataset: str = DEFAULT_DATASET_NAME,
) -> ProjectVisualizationData:
    issues: list[ValidationIssue] = []
    ground_truth_files, ground_truth_report = discover_ground_truth_files(project)
    issues.extend(ground_truth_report.issues)
    if not ground_truth_files:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="GROUND_TRUTH_NO_SEQUENCES",
                message="Ground Truth 目录中未发现可视化序列。",
                path=project.sources.ground_truth,
                suggestion="先配置包含序列 TXT 的 Ground Truth 目录。",
            )
        )

    discovered_images: dict[SequenceKey, ImageSequenceIndex] = {}
    image_root = project.sources.image_root
    if image_root is None:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="IMAGE_ROOT_NOT_CONFIGURED",
                message="尚未配置图像序列根目录。",
                suggestion="在数据源配置中选择图像序列根目录。",
            )
        )
    else:
        image_discovery = discover_image_sequences(image_root, dataset=dataset)
        issues.extend(image_discovery.report.issues)
        discovered_images = image_discovery.sequences

    mapped_images: dict[SequenceKey, ImageSequenceIndex] = {}
    ground_truth_keys = tuple(ground_truth_files)
    for image_key, sequence in discovered_images.items():
        candidates = _matching_ground_truth_keys(image_key, ground_truth_keys)
        if len(candidates) == 1:
            mapped_key = candidates[0]
        elif len(candidates) > 1:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="AMBIGUOUS_IMAGE_SEQUENCE",
                    message=(
                        f"图像序列 {image_key.name} 未指定模态，而 Ground Truth 中存在"
                        f" {len(candidates)} 个同名序列。"
                    ),
                    path=sequence.directory,
                    sequence=image_key.identifier,
                    suggestion="使用包含 VIS/NIR/RedNIR 的 FalseColor 模态目录。",
                )
            )
            continue
        else:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="UNMATCHED_IMAGE_SEQUENCE",
                    message=f"图像序列无法匹配 Ground Truth：{image_key.identifier}",
                    path=sequence.directory,
                    sequence=image_key.identifier,
                    modality=image_key.modality.value,
                    suggestion="检查模态目录和序列名称。",
                )
            )
            continue
        if mapped_key in mapped_images:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="DUPLICATE_MAPPED_IMAGE_SEQUENCE",
                    message=f"多个图像目录映射到同一序列：{mapped_key.identifier}",
                    path=sequence.directory,
                    sequence=mapped_key.identifier,
                    modality=mapped_key.modality.value,
                    suggestion="每个 Ground Truth 序列只保留一个图像目录。",
                )
            )
            continue
        mapped_images[mapped_key] = replace(sequence, key=mapped_key)

    tracker_result_files: dict[str, dict[SequenceKey, Path]] = {}
    for tracker in project.trackers:
        mapping = build_tracker_mapping(ground_truth_files, tracker, dataset=dataset)
        issues.extend(mapping.report.issues)
        tracker_result_files[tracker.tracker_id] = {
            key: result_path
            for key, (_, result_path) in mapping.pairs.items()
        }

    return ProjectVisualizationData(
        sequences=mapped_images,
        ground_truth_files=ground_truth_files,
        tracker_result_files=tracker_result_files,
        trackers=tuple(project.trackers),
        issues=tuple(issues),
    )
