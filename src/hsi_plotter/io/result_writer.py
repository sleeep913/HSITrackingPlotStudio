"""Export evaluation results to MAT, CSV and JSON.

All writers keep full double precision (ALG-EVAL-011) and replace existing
files atomically (NFR-ERR-006). MAT output keeps the legacy field names used
by the MATLAB plotting scripts (FR-EVAL-007).
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from hsi_plotter.models.dataset import MODALITY_LABELS, Modality
from hsi_plotter.models.evaluation import (
    EVALUATION_MODALITIES,
    EvaluationRun,
    EvaluationStatus,
    TrackerEvaluation,
)

CSV_ENCODING = "utf-8-sig"
JSON_ENCODING = "utf-8"

#: Column order of the exported CSV table.
CSV_COLUMNS = (
    "tracker",
    "level",
    "scope",
    "modality",
    "status",
    "auc",
    "dp",
    "cle",
    "sequence_count",
    "total_frames",
    "valid_frames",
    "invalid_frames",
    "message",
)

CSV_HEADER_LABELS = {
    "tracker": "跟踪器",
    "level": "层级",
    "scope": "范围",
    "modality": "模态",
    "status": "状态",
    "auc": "AUC",
    "dp": "DP",
    "cle": "CLE",
    "sequence_count": "序列数",
    "total_frames": "总帧数",
    "valid_frames": "有效帧数",
    "invalid_frames": "无效帧数",
    "message": "说明",
}

MAIN_TABLE_CSV_COLUMNS = (
    "Method",
    "VIS AUC",
    "VIS DP@20",
    "RedNIR AUC",
    "RedNIR DP@20",
    "NIR AUC",
    "NIR DP@20",
    "Average AUC",
    "Average DP@20",
)

STATUS_TEXT = {
    EvaluationStatus.SUCCEEDED: "成功",
    EvaluationStatus.FAILED: "失败",
    EvaluationStatus.SKIPPED: "已跳过",
}

LEVEL_TEXT = {"overall": "总体", "modality": "模态", "sequence": "序列"}

_INVALID_CHARACTERS = '<>:"/\\|?*'
_MAX_STEM_LENGTH = 120


class ResultExportError(OSError):
    """Raised when result files cannot be written."""


def sanitize_file_stem(name: str) -> str:
    """Return a stem that is safe on Windows and never empty."""
    cleaned = "".join(
        "_" if character in _INVALID_CHARACTERS or ord(character) < 32 else character
        for character in name.strip()
    )
    cleaned = cleaned.rstrip(". ")
    cleaned = cleaned or "result"
    return cleaned[:_MAX_STEM_LENGTH]


def build_filename(stem: str, suffix: str) -> str:
    """Join a sanitized stem with a normalized extension."""
    normalized = sanitize_file_stem(stem)
    extension = suffix if suffix.startswith(".") else f".{suffix}"
    return f"{normalized}{extension}"


def next_available_path(path: Path) -> Path:
    """Return ``path`` or the first non-existing ``name-N.ext`` variant."""
    candidate = Path(path)
    if not candidate.exists():
        return candidate
    for index in itertools.count(1):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate


def default_mat_filename(tracker: TrackerEvaluation, *, decimals: int = 3) -> str:
    """``TrackerName-AUC-DP.mat`` following the existing naming habit."""
    auc = tracker.auc
    dp = tracker.dp
    suffix = "" if auc is None or dp is None else f"-{auc:.{decimals}f}-{dp:.{decimals}f}"
    return f"{sanitize_file_stem(tracker.tracker_name)}{suffix}.mat"


@contextmanager
def _temporary_target(destination: Path) -> Iterator[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f"{destination.stem}-", suffix=".tmp"
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        yield temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, destination)


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return repr(float(value))


def _status_text(status: EvaluationStatus) -> str:
    return STATUS_TEXT.get(status, status.value)


def _modality_text(modality: Modality | None) -> str:
    if modality is None:
        return "全部"
    return MODALITY_LABELS.get(modality, modality.value)


def _csv_rows(run: EvaluationRun) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tracker in run.trackers:
        rows.append(_aggregate_row(tracker, "overall", "全部序列", None))
        for modality in sorted(tracker.modalities, key=lambda item: item.value):
            aggregate = tracker.modalities[modality]
            rows.append(
                _aggregate_row(tracker, "modality", _modality_text(modality), modality, aggregate)
            )
        for sequence in tracker.sequences:
            rows.append(
                {
                    "tracker": tracker.tracker_name,
                    "level": LEVEL_TEXT["sequence"],
                    "scope": sequence.sequence_name,
                    "modality": _modality_text(sequence.modality),
                    "status": _status_text(sequence.status),
                    "auc": _format_number(sequence.auc),
                    "dp": _format_number(sequence.dp),
                    "cle": _format_number(sequence.cle),
                    "sequence_count": "1",
                    "total_frames": str(sequence.total_frames),
                    "valid_frames": str(sequence.valid_frames),
                    "invalid_frames": str(sequence.invalid_frames),
                    "message": sequence.message,
                }
            )
    return rows


def _aggregate_row(
    tracker: TrackerEvaluation,
    level: str,
    scope: str,
    modality: Modality | None,
    aggregate=None,
) -> dict[str, str]:
    if aggregate is None:
        aggregate = tracker.overall
    if aggregate is None:
        values = {
            "auc": "",
            "dp": "",
            "cle": "",
            "sequence_count": "0",
            "total_frames": "0",
            "valid_frames": "0",
            "invalid_frames": "0",
        }
        status = EvaluationStatus.FAILED
    else:
        values = {
            "auc": _format_number(aggregate.auc),
            "dp": _format_number(aggregate.dp),
            "cle": _format_number(aggregate.cle),
            "sequence_count": str(aggregate.sequence_count),
            "total_frames": str(aggregate.total_frames),
            "valid_frames": str(aggregate.valid_frames),
            "invalid_frames": str(aggregate.invalid_frames),
        }
        status = tracker.status
    return {
        "tracker": tracker.tracker_name,
        "level": LEVEL_TEXT[level],
        "scope": scope,
        "modality": _modality_text(modality),
        "status": _status_text(status),
        "message": tracker.message,
        **values,
    }


def export_results_csv(run: EvaluationRun, destination: Path) -> Path:
    """Write a long-format CSV containing every level of every tracker."""
    target = Path(destination)
    try:
        with (
            _temporary_target(target) as temporary,
            temporary.open("w", encoding=CSV_ENCODING, newline="") as stream,
        ):
            writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS))
            writer.writerow(CSV_HEADER_LABELS)
            for row in _csv_rows(run):
                writer.writerow(row)
    except OSError as exc:
        raise ResultExportError(f"无法写入 CSV 结果 {target}：{exc}") from exc
    return target


def export_main_table_csv(run: EvaluationRun, destination: Path) -> Path:
    """Write the same nine-column wide table displayed by the evaluation page."""
    target = Path(destination)
    try:
        with (
            _temporary_target(target) as temporary,
            temporary.open("w", encoding=CSV_ENCODING, newline="") as stream,
        ):
            writer = csv.DictWriter(stream, fieldnames=list(MAIN_TABLE_CSV_COLUMNS))
            writer.writeheader()
            for tracker in run.trackers:
                row: dict[str, str] = {"Method": tracker.tracker_name}
                for modality in EVALUATION_MODALITIES:
                    label = MODALITY_LABELS[modality]
                    aggregate = tracker.modality_metrics(modality)
                    row[f"{label} AUC"] = _format_number(
                        aggregate.auc if aggregate is not None else None
                    )
                    row[f"{label} DP@20"] = _format_number(
                        aggregate.dp if aggregate is not None else None
                    )
                row["Average AUC"] = _format_number(tracker.auc)
                row["Average DP@20"] = _format_number(tracker.dp)
                writer.writerow(row)
    except OSError as exc:
        raise ResultExportError(f"无法写入指标主表 CSV {target}：{exc}") from exc
    return target


def _json_number(value: float | None) -> float | None:
    return None if value is None else float(value)


def _aggregate_payload(aggregate) -> dict[str, object] | None:
    if aggregate is None:
        return None
    return {
        "auc": float(aggregate.auc),
        "dp": float(aggregate.dp),
        "cle": float(aggregate.cle),
        "sequence_count": int(aggregate.sequence_count),
        "total_frames": int(aggregate.total_frames),
        "valid_frames": int(aggregate.valid_frames),
        "invalid_frames": int(aggregate.invalid_frames),
        "precision_curve": [float(value) for value in aggregate.precision_curve],
        "success_curve": [float(value) for value in aggregate.success_curve],
    }


def _sequence_payload(sequence) -> dict[str, object]:
    return {
        "sequence": sequence.sequence_name,
        "identifier": sequence.identifier,
        "modality": sequence.modality.value,
        "status": sequence.status.value,
        "auc": _json_number(sequence.auc),
        "dp": _json_number(sequence.dp),
        "cle": _json_number(sequence.cle),
        "total_frames": sequence.total_frames,
        "valid_frames": sequence.valid_frames,
        "invalid_frames": sequence.invalid_frames,
        "invalid_frame_indices": list(sequence.invalid_frame_indices),
        "ground_truth_path": str(sequence.ground_truth_path),
        "result_path": str(sequence.result_path),
        "message": sequence.message,
        "precision_curve": (
            [float(value) for value in sequence.metrics.precision_curve]
            if sequence.metrics is not None
            else None
        ),
        "success_curve": (
            [float(value) for value in sequence.metrics.success_curve]
            if sequence.metrics is not None
            else None
        ),
    }


def _tracker_payload(tracker: TrackerEvaluation) -> dict[str, object]:
    return {
        "tracker_id": tracker.tracker_id,
        "display_name": tracker.tracker_name,
        "layout": tracker.layout.value,
        "status": tracker.status.value,
        "cancelled": tracker.cancelled,
        "message": tracker.message,
        "missing_sequence_count": tracker.missing_count,
        "extra_sequence_count": tracker.extra_count,
        "ambiguous_sequence_count": tracker.ambiguous_count,
        "overall": _aggregate_payload(tracker.overall),
        "modalities": {
            modality.value: _aggregate_payload(aggregate)
            for modality, aggregate in sorted(
                tracker.modalities.items(), key=lambda item: item[0].value
            )
        },
        "sequences": [_sequence_payload(item) for item in tracker.sequences],
    }


def export_results_json(run: EvaluationRun, destination: Path) -> Path:
    """Write the full result tree with unrounded double precision values."""
    payload = {
        "schema": "hsi-evaluation/1",
        "cancelled": run.cancelled,
        "distance_thresholds": list(range(0, 51)),
        "success_thresholds": [round(index * 0.02, 4) for index in range(51)],
        "trackers": [_tracker_payload(tracker) for tracker in run.trackers],
    }
    target = Path(destination)
    try:
        with _temporary_target(target) as temporary:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding=JSON_ENCODING,
            )
    except OSError as exc:
        raise ResultExportError(f"无法写入 JSON 结果 {target}：{exc}") from exc
    return target


def export_tracker_mat(tracker: TrackerEvaluation, destination: Path) -> Path:
    """Write one tracker result as a legacy-compatible MAT file."""
    from scipy.io import savemat  # imported lazily to keep application start fast

    succeeded = tracker.succeeded_sequences
    names = [item.sequence_name for item in succeeded]
    modalities = [item.modality.value for item in succeeded]
    success_rows = (
        np.asarray(
            [item.metrics.success_curve for item in succeeded if item.metrics is not None],
            dtype=np.float64,
        ).reshape(len(succeeded), -1)
        if succeeded
        else np.zeros((0, 51), dtype=np.float64)
    )
    precision_rows = (
        np.asarray(
            [item.metrics.precision_curve for item in succeeded if item.metrics is not None],
            dtype=np.float64,
        ).reshape(len(succeeded), -1)
        if succeeded
        else np.zeros((0, 51), dtype=np.float64)
    )
    cle_values = np.asarray(
        [item.cle for item in succeeded if item.cle is not None], dtype=np.float64
    ).reshape(-1, 1)
    cle_matrix = (
        np.repeat(cle_values, success_rows.shape[1], axis=1)
        if len(succeeded)
        else np.zeros((0, 51), dtype=np.float64)
    )
    payload: dict[str, object] = {
        "PASCAL_rec": success_rows,
        "distance_rec": precision_rows,
        "average_cle_rec": cle_matrix,
        "average_cle": cle_values.reshape(-1),
        "sequence_names": np.asarray(names, dtype=object),
        "sequence_modality": np.asarray(modalities, dtype=object),
        "AUC": float(tracker.auc) if tracker.auc is not None else float("nan"),
        "DP": float(tracker.dp) if tracker.dp is not None else float("nan"),
        "CLE": float(tracker.cle) if tracker.cle is not None else float("nan"),
        "distance_thresholds": np.arange(0.0, 51.0, 1.0, dtype=np.float64),
        "PASCAL_thresholds": np.linspace(0.0, 1.0, 51, dtype=np.float64),
    }
    if tracker.overall is not None:
        payload["PASCAL_rec_mean"] = np.asarray(tracker.overall.success_curve, dtype=np.float64)
        payload["distance_rec_mean"] = np.asarray(tracker.overall.precision_curve, dtype=np.float64)

    target = Path(destination)
    try:
        with _temporary_target(target) as temporary:
            savemat(str(temporary), payload, format="5", do_compression=True)
    except (OSError, ValueError, ImportError) as exc:
        raise ResultExportError(f"无法写入 MAT 结果 {target}：{exc}") from exc
    return target


def export_trackers_mat(
    trackers, directory: Path, *, namer: Callable[[TrackerEvaluation], str] | None = None
) -> tuple[Path, ...]:
    """Write every tracker result into ``directory`` using its own file name."""
    namer = namer or default_mat_filename
    written: list[Path] = []
    for tracker in trackers:
        target = Path(directory) / namer(tracker)
        written.append(export_tracker_mat(tracker, target))
    return tuple(written)


MAT_LEGACY_FIELDS = ("PASCAL_rec", "distance_rec", "average_cle_rec")
