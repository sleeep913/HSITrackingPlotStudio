"""Sidecar JSON codec for tracking-visualization project indexes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.project import Project
from hsi_plotter.models.visualization import ImageSequenceIndex
from hsi_plotter.validation.issues import Severity, ValidationIssue
from hsi_plotter.visualization.project_data import ProjectVisualizationData


def _key_payload(key: SequenceKey) -> dict[str, str]:
    return {"dataset": key.dataset, "modality": key.modality.value, "name": key.name}


def _decode_key(value: dict[str, Any]) -> SequenceKey:
    return SequenceKey(
        str(value["dataset"]), Modality(str(value["modality"])), str(value["name"])
    )


def encode_visualization_data(data: ProjectVisualizationData) -> dict[str, Any]:
    return {
        "codec_version": 1,
        "sequences": [
            {
                "key": _key_payload(key),
                "directory": str(sequence.directory),
                "frames": {str(frame): str(path) for frame, path in sequence.frames.items()},
            }
            for key, sequence in data.sequences.items()
        ],
        "ground_truth_files": [
            {"key": _key_payload(key), "path": str(path)}
            for key, path in data.ground_truth_files.items()
        ],
        "tracker_result_files": {
            tracker_id: [
                {"key": _key_payload(key), "path": str(path)}
                for key, path in files.items()
            ]
            for tracker_id, files in data.tracker_result_files.items()
        },
        "tracker_ids": [tracker.tracker_id for tracker in data.trackers],
        "issues": [
            {
                "severity": issue.severity.value,
                "code": issue.code,
                "message": issue.message,
                "path": None if issue.path is None else str(issue.path),
                "tracker": issue.tracker,
                "sequence": issue.sequence,
                "modality": issue.modality,
                "suggestion": issue.suggestion,
            }
            for issue in data.issues
        ],
    }


def decode_visualization_data(
    payload: dict[str, Any], project: Project
) -> ProjectVisualizationData:
    if payload.get("codec_version") != 1:
        raise ValueError("不支持的可视化缓存版本")
    sequences: dict[SequenceKey, ImageSequenceIndex] = {}
    for item in payload["sequences"]:
        key = _decode_key(item["key"])
        sequences[key] = ImageSequenceIndex(
            key,
            Path(item["directory"]),
            {int(frame): Path(path) for frame, path in item["frames"].items()},
        )
    ground_truth = {
        _decode_key(item["key"]): Path(item["path"])
        for item in payload["ground_truth_files"]
    }
    result_files = {
        str(tracker_id): {
            _decode_key(item["key"]): Path(item["path"]) for item in items
        }
        for tracker_id, items in payload["tracker_result_files"].items()
    }
    wanted = set(map(str, payload["tracker_ids"]))
    trackers = tuple(tracker for tracker in project.trackers if tracker.tracker_id in wanted)
    issues = tuple(
        ValidationIssue(
            Severity(str(item["severity"])),
            str(item["code"]),
            str(item["message"]),
            None if item["path"] is None else Path(item["path"]),
            item.get("tracker"),
            item.get("sequence"),
            item.get("modality"),
            item.get("suggestion"),
        )
        for item in payload.get("issues", [])
    )
    return ProjectVisualizationData(sequences, ground_truth, result_files, trackers, issues)


__all__ = ["decode_visualization_data", "encode_visualization_data"]
