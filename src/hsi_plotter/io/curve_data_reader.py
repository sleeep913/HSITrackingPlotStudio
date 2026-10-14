from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class CurveDataError(ValueError):
    """Raised when a MAT file does not contain compatible tracking curves."""


@dataclass(frozen=True, slots=True)
class ImportedCurveData:
    tracker_id: str
    display_name: str
    precision_curve: NDArray[np.float64]
    success_curve: NDArray[np.float64]
    auc: float
    dp: float
    source_path: Path


def _curve(payload: dict[str, object], mean_key: str, rows_key: str) -> NDArray[np.float64]:
    if mean_key in payload:
        values = np.asarray(payload[mean_key], dtype=np.float64).reshape(-1)
    elif rows_key in payload:
        rows = np.asarray(payload[rows_key], dtype=np.float64)
        if rows.ndim == 1:
            values = rows
        elif rows.ndim == 2 and rows.shape[0] > 0:
            values = np.mean(rows, axis=0)
        else:
            values = np.asarray([], dtype=np.float64)
    else:
        values = np.asarray([], dtype=np.float64)
    if values.shape != (51,):
        raise CurveDataError(f"字段 {rows_key} 必须能够生成 51 点曲线")
    if not np.isfinite(values).all():
        raise CurveDataError(f"字段 {rows_key} 包含 NaN 或无穷值")
    return values


def _optional_scalar(payload: dict[str, object], key: str, fallback: float) -> float:
    if key not in payload:
        return fallback
    values = np.asarray(payload[key], dtype=np.float64).reshape(-1)
    return float(values[0]) if values.size and np.isfinite(values[0]) else fallback


def read_curve_mat(source: Path) -> ImportedCurveData:
    from scipy.io import loadmat

    path = Path(source)
    try:
        payload = loadmat(str(path), squeeze_me=True)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise CurveDataError(f"无法读取 MAT 曲线文件 {path}：{exc}") from exc
    precision = _curve(payload, "distance_rec_mean", "distance_rec")
    success = _curve(payload, "PASCAL_rec_mean", "PASCAL_rec")
    display_name = re.sub(r"-\d+(?:\.\d+)?-\d+(?:\.\d+)?$", "", path.stem) or path.stem
    tracker_id = re.sub(r"[^a-z0-9_-]+", "-", display_name.casefold()).strip("-")
    tracker_id = tracker_id or "imported-tracker"
    return ImportedCurveData(
        tracker_id=tracker_id,
        display_name=display_name,
        precision_curve=precision,
        success_curve=success,
        auc=_optional_scalar(payload, "AUC", float(np.mean(success[1:]))),
        dp=_optional_scalar(payload, "DP", float(precision[20])),
        source_path=path.resolve(strict=False),
    )
