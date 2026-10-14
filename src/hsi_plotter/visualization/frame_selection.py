from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hsi_plotter.metrics.tracking import compute_box_overlap
from hsi_plotter.visualization.session import SequenceVisualizationSession


class FrameSelectionError(ValueError):
    """Raised when advantage-frame inputs cannot be compared fairly."""


@dataclass(frozen=True, slots=True)
class AdvantageFrameCandidate:
    frame_number: int
    primary_iou: float
    strongest_competitor_iou: float
    mean_competitor_iou: float
    advantage: float


@dataclass(frozen=True, slots=True)
class AdvantageFrameSelection:
    candidates: tuple[AdvantageFrameCandidate, ...]
    message: str
    temporal_gap: int


def _drawable(row: NDArray[np.float64]) -> bool:
    return bool(np.isfinite(row).all() and row[2] > 0.0 and row[3] > 0.0)


def _competitor_overlap(
    row: NDArray[np.float64],
    ground_truth: NDArray[np.float64],
) -> float:
    if np.isnan(row).all() or row[2] <= 0.0 or row[3] <= 0.0:
        return 0.0
    return compute_box_overlap(row, ground_truth)


def _select_with_gap(
    ranked: list[AdvantageFrameCandidate],
    *,
    limit: int,
    gap: int,
) -> list[AdvantageFrameCandidate]:
    selected: list[AdvantageFrameCandidate] = []
    for candidate in ranked:
        if all(
            abs(candidate.frame_number - previous.frame_number) >= gap
            for previous in selected
        ):
            selected.append(candidate)
            if len(selected) == limit:
                break
    return selected


def select_advantage_frames(
    session: SequenceVisualizationSession,
    *,
    primary_tracker_id: str,
    competitor_ids: tuple[str, ...],
    limit: int = 5,
    minimum_primary_iou: float = 0.5,
) -> AdvantageFrameSelection:
    if limit < 1:
        raise FrameSelectionError("候选帧数量必须大于 0")
    unique_competitors = tuple(dict.fromkeys(competitor_ids))
    if not unique_competitors or primary_tracker_id in unique_competitors:
        raise FrameSelectionError("必须选择至少一个不同于主方法的对比方法")
    try:
        primary = session.tracker_box_data(primary_tracker_id)
        competitors = tuple(session.tracker_box_data(item) for item in unique_competitors)
    except KeyError as exc:
        raise FrameSelectionError(f"跟踪器不存在：{exc.args[0]}") from exc

    expected_frames = session.ground_truth.frame_count
    sources = ((primary_tracker_id, primary), *zip(unique_competitors, competitors, strict=True))
    for tracker_id, boxes in sources:
        if boxes.frame_count != expected_frames:
            raise FrameSelectionError(
                f"跟踪器 {tracker_id} 的结果帧数 {boxes.frame_count} 与 Ground Truth "
                f"帧数 {expected_frames} 不一致"
            )

    ranked: list[AdvantageFrameCandidate] = []
    for frame_number in range(2, expected_frames + 1):
        if session.images.path_for_frame(frame_number) is None:
            continue
        ground_truth = session.ground_truth.values[frame_number - 1]
        primary_row = primary.values[frame_number - 1]
        if not _drawable(ground_truth) or not _drawable(primary_row):
            continue
        primary_iou = compute_box_overlap(primary_row, ground_truth)
        if not np.isfinite(primary_iou) or primary_iou < minimum_primary_iou:
            continue
        competitor_ious = tuple(
            _competitor_overlap(boxes.values[frame_number - 1], ground_truth)
            for boxes in competitors
        )
        strongest = max(competitor_ious)
        if primary_iou <= strongest:
            continue
        mean_iou = float(np.mean(competitor_ious))
        ranked.append(
            AdvantageFrameCandidate(
                frame_number=frame_number,
                primary_iou=primary_iou,
                strongest_competitor_iou=strongest,
                mean_competitor_iou=mean_iou,
                advantage=primary_iou - strongest,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.advantage,
            -item.primary_iou,
            item.mean_competitor_iou,
            item.frame_number,
        )
    )
    initial_gap = max(5, round(expected_frames * 0.01))
    selected: list[AdvantageFrameCandidate] = []
    used_gap = initial_gap
    for gap in range(initial_gap, 0, -1):
        attempt = _select_with_gap(ranked, limit=limit, gap=gap)
        if len(attempt) > len(selected):
            selected = attempt
            used_gap = gap
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        message = f"仅找到 {len(selected)} 个满足条件的优势帧。"
    else:
        message = f"已推荐 {len(selected)} 个优势帧。"
    return AdvantageFrameSelection(
        candidates=tuple(selected),
        message=message,
        temporal_gap=used_gap,
    )
