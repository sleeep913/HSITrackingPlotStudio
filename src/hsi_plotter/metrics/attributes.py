"""Attribute-subset evaluation built on the shared tracking metric pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from hsi_plotter.metrics.evaluation import prepare_evaluation_inputs, run_evaluation
from hsi_plotter.metrics.tracking import aggregate_sequence_metrics
from hsi_plotter.models.attribute_evaluation import (
    AttributeDefinition,
    AttributeEvaluationRun,
    AttributeResolution,
    AttributeScore,
    TrackerAttributeResult,
)
from hsi_plotter.models.dataset import Modality, SequenceKey, parse_sequence_identity
from hsi_plotter.models.project import Project

ProgressCallback = Callable[[int, int, str], None]
CancelledCallback = Callable[[], bool]


def resolve_attribute_sequences(
    attributes: Sequence[AttributeDefinition],
    ground_truth_files: Mapping[SequenceKey, Path],
) -> AttributeResolution:
    """Resolve prefixed and unique plain references against project sequence keys."""
    exact: dict[tuple[Modality, str], list[SequenceKey]] = {}
    plain: dict[str, list[SequenceKey]] = {}
    for key in ground_truth_files:
        folded = key.name.casefold()
        exact.setdefault((key.modality, folded), []).append(key)
        plain.setdefault(folded, []).append(key)

    resolved: dict[str, tuple[SequenceKey, ...]] = {}
    unknown: dict[str, tuple[str, ...]] = {}
    for attribute in attributes:
        keys: list[SequenceKey] = []
        misses: list[str] = []
        seen: set[SequenceKey] = set()
        for reference in attribute.sequences:
            modality, name = parse_sequence_identity(reference)
            candidates = (
                exact.get((modality, name.casefold()), [])
                if modality is not Modality.UNSPECIFIED
                else plain.get(name.casefold(), [])
            )
            if len(candidates) != 1:
                misses.append(reference)
                continue
            key = candidates[0]
            if key not in seen:
                seen.add(key)
                keys.append(key)
        resolved[attribute.name] = tuple(keys)
        if misses:
            unknown[attribute.name] = tuple(misses)
    return AttributeResolution(keys_by_attribute=resolved, unknown_by_attribute=unknown)


def run_attribute_evaluation(
    project: Project,
    attributes: Sequence[AttributeDefinition],
    *,
    on_progress: ProgressCallback | None = None,
    is_cancelled: CancelledCallback | None = None,
) -> AttributeEvaluationRun:
    """Compute every referenced sequence once, then aggregate per attribute."""
    definitions = tuple(attributes)
    plan = prepare_evaluation_inputs(project)
    resolution = resolve_attribute_sequences(definitions, plan.ground_truth_files)
    selected_keys = {
        key for keys in resolution.keys_by_attribute.values() for key in keys
    }
    filtered_plan = replace(
        plan,
        tasks=tuple(task for task in plan.tasks if task.key in selected_keys),
    )
    evaluation = run_evaluation(
        filtered_plan,
        on_progress=on_progress,
        is_cancelled=is_cancelled,
    )

    tracker_results: list[TrackerAttributeResult] = []
    for tracker in evaluation.trackers:
        succeeded = {
            item.key: item.metrics
            for item in tracker.succeeded_sequences
            if item.metrics is not None
        }
        scores: dict[str, AttributeScore] = {}
        for attribute in definitions:
            keys = resolution.keys_by_attribute.get(attribute.name, ())
            sequence_metrics = [succeeded[key] for key in keys if key in succeeded]
            aggregate = (
                aggregate_sequence_metrics(sequence_metrics) if sequence_metrics else None
            )
            scores[attribute.name] = AttributeScore(
                auc=None if aggregate is None else aggregate.auc,
                dp=None if aggregate is None else aggregate.dp,
                resolved_count=len(keys),
                succeeded_count=len(sequence_metrics),
                failed_count=max(len(keys) - len(sequence_metrics), 0),
            )
        tracker_results.append(
            TrackerAttributeResult(
                tracker_id=tracker.tracker_id,
                tracker_name=tracker.tracker_name,
                scores=scores,
            )
        )

    issues = tuple(issue.message for issue in evaluation.issues)
    return AttributeEvaluationRun(
        attributes=definitions,
        trackers=tuple(tracker_results),
        unknown_by_attribute=resolution.unknown_by_attribute,
        issues=issues,
    )


__all__ = ["resolve_attribute_sequences", "run_attribute_evaluation"]
