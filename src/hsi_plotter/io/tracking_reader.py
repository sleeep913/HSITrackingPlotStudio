from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class TrackingDataError(ValueError):
    """Raised when a Ground Truth or tracker-result file is invalid."""


@dataclass(frozen=True, slots=True)
class BoundingBoxData:
    values: NDArray[np.float64]
    source_path: Path
    invalid_frame_indices: tuple[int, ...]

    @property
    def frame_count(self) -> int:
        return int(self.values.shape[0])


def _parse_row(path: Path, line_number: int, line: str) -> list[float]:
    fields = [field for field in re.split(r"[,\s]+", line.strip()) if field]
    if len(fields) != 4:
        raise TrackingDataError(
            f"{path} 第 {line_number} 行必须包含 4 个数值，实际为 {len(fields)} 个"
        )
    try:
        return [float(field) for field in fields]
    except ValueError as exc:
        raise TrackingDataError(f"{path} 第 {line_number} 行包含非数值内容") from exc


def read_bounding_boxes(source: Path, *, allow_missing_rows: bool = True) -> BoundingBoxData:
    path = Path(source)
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise TrackingDataError(f"无法读取边界框文件 {path}：{exc}") from exc

    rows = [
        _parse_row(path, line_number, line)
        for line_number, line in enumerate(lines, start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise TrackingDataError(f"边界框文件为空：{path}")

    values = np.asarray(rows, dtype=np.float64)
    missing_rows: list[int] = []
    for zero_based_index, row in enumerate(values):
        line_number = zero_based_index + 1
        nan_mask = np.isnan(row)
        if nan_mask.all():
            if not allow_missing_rows:
                raise TrackingDataError(f"Ground Truth 第 {line_number} 帧不允许使用完整 NaN 行")
            missing_rows.append(line_number)
            continue
        if nan_mask.any():
            raise TrackingDataError(f"{path} 第 {line_number} 行存在部分缺失值")
        if np.isinf(row).any():
            raise TrackingDataError(f"{path} 第 {line_number} 行包含无穷值")

    return BoundingBoxData(
        values=values,
        source_path=path.resolve(strict=False),
        invalid_frame_indices=tuple(missing_rows),
    )
