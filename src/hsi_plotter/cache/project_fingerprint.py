"""Module-specific fingerprints built from project sources."""

from __future__ import annotations

from pathlib import Path

from hsi_plotter.cache.fingerprint import fingerprint_inputs
from hsi_plotter.models.project import Project

ATTRIBUTE_ALGORITHM_VERSION = "attribute-evaluation-v1"
EVALUATION_ALGORITHM_VERSION = "tracking-evaluation-v1"
VISUALIZATION_ALGORITHM_VERSION = "tracking-visualization-index-v1"


def _tracking_paths(project: Project) -> tuple[Path, ...]:
    paths: list[Path] = []
    if project.sources.ground_truth is not None:
        paths.append(project.sources.ground_truth)
    paths.extend(tracker.results_path for tracker in project.trackers)
    return tuple(paths)


def evaluation_fingerprint(
    project: Project, tracker_ids: tuple[str, ...] | None = None
) -> str:
    selected = set(tracker_ids or (tracker.tracker_id for tracker in project.trackers))
    trackers = [tracker for tracker in project.trackers if tracker.tracker_id in selected]
    return fingerprint_inputs(
        tuple(
            path
            for path in _tracking_paths(project)
            if path == project.sources.ground_truth
            or any(path == tracker.results_path for tracker in trackers)
        ),
        {
            "algorithm": EVALUATION_ALGORITHM_VERSION,
            "trackers": [
                {"id": tracker.tracker_id, "layout": tracker.layout.value}
                for tracker in trackers
            ],
        },
    )


def attribute_fingerprint(project: Project) -> str:
    paths = list(_tracking_paths(project))
    if project.sources.attribute_file is not None:
        paths.append(project.sources.attribute_file)
    return fingerprint_inputs(
        tuple(paths),
        {
            "algorithm": ATTRIBUTE_ALGORITHM_VERSION,
            "trackers": [
                {"id": tracker.tracker_id, "layout": tracker.layout.value}
                for tracker in project.trackers
            ],
        },
    )


def visualization_fingerprint(project: Project) -> str:
    paths = list(_tracking_paths(project))
    if project.sources.image_root is not None:
        paths.append(project.sources.image_root)
    return fingerprint_inputs(
        tuple(paths),
        {
            "algorithm": VISUALIZATION_ALGORITHM_VERSION,
            "trackers": [
                {"id": tracker.tracker_id, "layout": tracker.layout.value}
                for tracker in project.trackers
            ],
        },
    )


__all__ = [
    "ATTRIBUTE_ALGORITHM_VERSION",
    "EVALUATION_ALGORITHM_VERSION",
    "VISUALIZATION_ALGORITHM_VERSION",
    "attribute_fingerprint",
    "evaluation_fingerprint",
    "visualization_fingerprint",
]
