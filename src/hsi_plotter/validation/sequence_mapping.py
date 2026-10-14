from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.tracker_import import LayoutCandidate, SequenceMapping
from hsi_plotter.validation.issues import PreflightReport, Severity, ValidationIssue
from hsi_plotter.validation.tracking_pair import validate_tracking_pair


def _sorted_keys(keys: set[SequenceKey] | frozenset[SequenceKey]) -> list[SequenceKey]:
    return sorted(keys, key=lambda item: item.identifier)


def _group_by_name(keys: set[SequenceKey]) -> dict[str, list[SequenceKey]]:
    grouped: dict[str, list[SequenceKey]] = {}
    for key in keys:
        grouped.setdefault(key.name.casefold(), []).append(key)
    for values in grouped.values():
        values.sort(key=lambda item: item.identifier)
    return grouped


def _issue(
    severity: Severity,
    code: str,
    message: str,
    *,
    tracker: str,
    sequence: SequenceKey,
    path: Path | None = None,
    suggestion: str,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        path=path,
        tracker=tracker,
        sequence=sequence.identifier,
        modality=sequence.modality.value,
        suggestion=suggestion,
    )


def build_sequence_mapping(
    ground_truth_files: Mapping[SequenceKey, Path],
    candidate: LayoutCandidate,
    *,
    tracker: str,
) -> SequenceMapping:
    """Pair Ground Truth files with tracker result files.

    Sequences are first matched on their full key ``(dataset, modality, name)``.
    Result files without modality information are matched by name only, and are
    reported as ambiguous when several Ground Truth modalities share that name
    (FR-DATA-009 / FR-DATA-013).
    """
    report = PreflightReport()
    pairs: dict[SequenceKey, tuple[Path, Path]] = {}
    remaining_ground_truth = set(ground_truth_files)
    remaining_results = set(candidate.files)

    for key in _sorted_keys(remaining_ground_truth & remaining_results):
        pairs[key] = (ground_truth_files[key], candidate.files[key])
    remaining_ground_truth -= set(pairs)
    remaining_results -= set(pairs)

    ambiguous: list[SequenceKey] = []
    extra: list[SequenceKey] = []
    ground_truth_by_name = _group_by_name(set(remaining_ground_truth))

    for key in _sorted_keys(remaining_results):
        result_path = candidate.files[key]
        candidates = ground_truth_by_name.get(key.name.casefold(), [])
        if key.modality is Modality.UNSPECIFIED and len(candidates) == 1:
            ground_truth_key = candidates[0]
            pairs[ground_truth_key] = (ground_truth_files[ground_truth_key], result_path)
            remaining_ground_truth.discard(ground_truth_key)
            continue
        if key.modality is Modality.UNSPECIFIED and len(candidates) > 1:
            ambiguous.append(key)
            report.add(
                _issue(
                    Severity.ERROR,
                    code="AMBIGUOUS_SEQUENCE_MATCH",
                    message=(
                        f"结果文件 {result_path.name} 不含模态信息，"
                        f"而 Ground Truth 中存在多个同名序列（{len(candidates)} 个），"
                        "无法自动判定所属模态。"
                    ),
                    tracker=tracker,
                    sequence=key,
                    path=result_path,
                    suggestion="使用带模态前缀的结果布局，或在导入映射中手动指定该序列。",
                )
            )
            continue
        extra.append(key)
        if candidates:
            report.add(
                _issue(
                    Severity.WARNING,
                    code="SEQUENCE_MODALITY_MISMATCH",
                    message=(
                        f"结果序列 {key.name} 的模态 {key.modality.value} "
                        "与 Ground Truth 中同名序列不一致，未能建立映射。"
                    ),
                    tracker=tracker,
                    sequence=key,
                    path=result_path,
                    suggestion="确认结果文件命名与所属模态目录是否匹配。",
                )
            )
        else:
            report.add(
                _issue(
                    Severity.INFO,
                    code="EXTRA_SEQUENCE",
                    message=f"Ground Truth 中不存在结果序列 {key.identifier}，该序列不参与计算。",
                    tracker=tracker,
                    sequence=key,
                    path=result_path,
                    suggestion="如不需要该序列，可从结果目录中移除以保持预检整洁。",
                )
            )

    missing: list[SequenceKey] = []
    for key in _sorted_keys(set(remaining_ground_truth)):
        missing.append(key)
        report.add(
            _issue(
                Severity.WARNING,
                code="MISSING_SEQUENCE",
                message=f"跟踪器缺少 Ground Truth 序列 {key.identifier} 的结果文件。",
                tracker=tracker,
                sequence=key,
                path=ground_truth_files.get(key),
                suggestion="补充该序列结果，或确认该跟踪器确实未覆盖此序列。"
                "该序列将被排除在此跟踪器的聚合结果之外。",
            )
        )

    if not pairs:
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code="TRACKER_NO_MATCHED_SEQUENCES",
                message=f"跟踪器 {tracker} 没有任何序列能与 Ground Truth 建立映射。",
                path=candidate.root_path,
                tracker=tracker,
                suggestion="检查结果目录布局、文件命名与模态配置。",
            )
        )

    return SequenceMapping(
        tracker_id=tracker.casefold(),
        tracker_name=tracker,
        layout=candidate.layout,
        root_path=candidate.root_path,
        pairs=pairs,
        missing=tuple(missing),
        extra=tuple(extra),
        ambiguous=tuple(ambiguous),
        report=report,
    )


def validate_sequence_mapping(
    mapping: SequenceMapping,
    *,
    on_progress: Callable[[int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> PreflightReport:
    """Read every mapped pair and report frame or bounding box problems.

    Only file-level findings are returned; structural findings of the mapping
    itself live in ``mapping.report`` and must be merged by the caller.
    """
    report = PreflightReport()
    keys = _sorted_keys(set(mapping.pairs))
    total = len(keys)
    for index, key in enumerate(keys, start=1):
        if is_cancelled is not None and is_cancelled():
            report.add(
                ValidationIssue(
                    severity=Severity.INFO,
                    code="PREFLIGHT_CANCELLED",
                    message=f"预检已被取消，已检查 {index - 1}/{total} 个序列。",
                    tracker=mapping.tracker_name,
                    suggestion="重新运行预检以完成全部检查。",
                )
            )
            break
        if on_progress is not None:
            on_progress(index, key.identifier)
        ground_truth_path, result_path = mapping.pairs[key]
        report.extend(
            validate_tracking_pair(
                ground_truth_path,
                result_path,
                tracker=mapping.tracker_name,
                sequence=key,
            ).issues
        )
    return report
