"""JSON-safe codec for complete tracking evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hsi_plotter.metrics.tracking import AggregateMetrics, SequenceMetrics
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.evaluation import (
    EvaluationRun,
    EvaluationStatus,
    SequenceEvaluation,
    TrackerEvaluation,
)
from hsi_plotter.models.tracker_import import ResultLayout
from hsi_plotter.validation.issues import Severity, ValidationIssue


def _sequence_metrics_payload(metrics: SequenceMetrics) -> dict[str, Any]:
    return {
        "precision_curve": metrics.precision_curve.tolist(),
        "success_curve": metrics.success_curve.tolist(),
        "distances": metrics.distances.tolist(),
        "overlaps": metrics.overlaps.tolist(),
        "auc": metrics.auc,
        "dp": metrics.dp,
        "cle": metrics.cle,
        "total_frames": metrics.total_frames,
        "valid_frames": metrics.valid_frames,
        "invalid_frames": metrics.invalid_frames,
    }


def _decode_sequence_metrics(value: dict[str, Any]) -> SequenceMetrics:
    return SequenceMetrics(
        precision_curve=np.asarray(value["precision_curve"], dtype=np.float64),
        success_curve=np.asarray(value["success_curve"], dtype=np.float64),
        distances=np.asarray(value["distances"], dtype=np.float64),
        overlaps=np.asarray(value["overlaps"], dtype=np.float64),
        auc=float(value["auc"]),
        dp=float(value["dp"]),
        cle=float(value["cle"]),
        total_frames=int(value["total_frames"]),
        valid_frames=int(value["valid_frames"]),
        invalid_frames=int(value["invalid_frames"]),
    )


def _aggregate_payload(metrics: AggregateMetrics) -> dict[str, Any]:
    return {
        "precision_curve": metrics.precision_curve.tolist(),
        "success_curve": metrics.success_curve.tolist(),
        "auc": metrics.auc,
        "dp": metrics.dp,
        "cle": metrics.cle,
        "sequence_count": metrics.sequence_count,
        "total_frames": metrics.total_frames,
        "valid_frames": metrics.valid_frames,
        "invalid_frames": metrics.invalid_frames,
    }


def _decode_aggregate(value: dict[str, Any]) -> AggregateMetrics:
    return AggregateMetrics(
        precision_curve=np.asarray(value["precision_curve"], dtype=np.float64),
        success_curve=np.asarray(value["success_curve"], dtype=np.float64),
        auc=float(value["auc"]),
        dp=float(value["dp"]),
        cle=float(value["cle"]),
        sequence_count=int(value["sequence_count"]),
        total_frames=int(value["total_frames"]),
        valid_frames=int(value["valid_frames"]),
        invalid_frames=int(value["invalid_frames"]),
    )


def encode_evaluation_run(run: EvaluationRun) -> dict[str, Any]:
    return {
        "codec_version": 1,
        "cancelled": run.cancelled,
        "issues": [
            {
                "severity": item.severity.value,
                "code": item.code,
                "message": item.message,
                "path": None if item.path is None else str(item.path),
                "tracker": item.tracker,
                "sequence": item.sequence,
                "modality": item.modality,
                "suggestion": item.suggestion,
            }
            for item in run.issues
        ],
        "trackers": [
            {
                "tracker_id": tracker.tracker_id,
                "tracker_name": tracker.tracker_name,
                "layout": tracker.layout.value,
                "missing_count": tracker.missing_count,
                "extra_count": tracker.extra_count,
                "ambiguous_count": tracker.ambiguous_count,
                "cancelled": tracker.cancelled,
                "message": tracker.message,
                "overall": None
                if tracker.overall is None
                else _aggregate_payload(tracker.overall),
                "modalities": {
                    modality.value: _aggregate_payload(metrics)
                    for modality, metrics in tracker.modalities.items()
                },
                "sequences": [
                    {
                        "key": {
                            "dataset": sequence.key.dataset,
                            "modality": sequence.key.modality.value,
                            "name": sequence.key.name,
                        },
                        "ground_truth_path": str(sequence.ground_truth_path),
                        "result_path": str(sequence.result_path),
                        "status": sequence.status.value,
                        "metrics": None
                        if sequence.metrics is None
                        else _sequence_metrics_payload(sequence.metrics),
                        "invalid_frame_indices": list(sequence.invalid_frame_indices),
                        "message": sequence.message,
                    }
                    for sequence in tracker.sequences
                ],
            }
            for tracker in run.trackers
        ],
    }


def decode_evaluation_run(payload: dict[str, Any]) -> EvaluationRun:
    if payload.get("codec_version") != 1:
        raise ValueError("不支持的指标缓存版本")
    trackers: list[TrackerEvaluation] = []
    for item in payload["trackers"]:
        sequences = []
        for sequence in item["sequences"]:
            key = sequence["key"]
            sequences.append(
                SequenceEvaluation(
                    key=SequenceKey(
                        dataset=str(key["dataset"]),
                        modality=Modality(str(key["modality"])),
                        name=str(key["name"]),
                    ),
                    ground_truth_path=Path(sequence["ground_truth_path"]),
                    result_path=Path(sequence["result_path"]),
                    status=EvaluationStatus(str(sequence["status"])),
                    metrics=None
                    if sequence["metrics"] is None
                    else _decode_sequence_metrics(sequence["metrics"]),
                    invalid_frame_indices=tuple(map(int, sequence["invalid_frame_indices"])),
                    message=str(sequence["message"]),
                )
            )
        trackers.append(
            TrackerEvaluation(
                tracker_id=str(item["tracker_id"]),
                tracker_name=str(item["tracker_name"]),
                layout=ResultLayout(str(item["layout"])),
                sequences=tuple(sequences),
                overall=None
                if item["overall"] is None
                else _decode_aggregate(item["overall"]),
                modalities={
                    Modality(str(modality)): _decode_aggregate(metrics)
                    for modality, metrics in item["modalities"].items()
                },
                missing_count=int(item["missing_count"]),
                extra_count=int(item["extra_count"]),
                ambiguous_count=int(item["ambiguous_count"]),
                cancelled=bool(item["cancelled"]),
                message=str(item["message"]),
            )
        )
    issues = tuple(
        ValidationIssue(
            severity=Severity(str(item["severity"])),
            code=str(item["code"]),
            message=str(item["message"]),
            path=None if item["path"] is None else Path(item["path"]),
            tracker=item.get("tracker"),
            sequence=item.get("sequence"),
            modality=item.get("modality"),
            suggestion=item.get("suggestion"),
        )
        for item in payload.get("issues", [])
    )
    return EvaluationRun(
        trackers=tuple(trackers), issues=issues, cancelled=bool(payload["cancelled"])
    )


__all__ = ["decode_evaluation_run", "encode_evaluation_run"]
