from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hsi_plotter.io.tracking_reader import BoundingBoxData, read_bounding_boxes
from hsi_plotter.models.dataset import SequenceKey
from hsi_plotter.models.style import TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.visualization import ImageSequenceIndex

GROUND_TRUTH_VISUAL_STYLE = TrackerStyle(
    color="#0000FF",
    line_style="-",
    line_width=2.0,
)


class OverlayStatus(StrEnum):
    VISIBLE = "visible"
    OFFSCREEN = "offscreen"
    MISSING = "missing"
    OUT_OF_RANGE = "out_of_range"


@dataclass(frozen=True, slots=True)
class TrackerSequenceSource:
    tracker: Tracker
    result_path: Path


@dataclass(frozen=True, slots=True)
class BoxOverlay:
    tracker_id: str
    display_name: str
    style: TrackerStyle
    status: OverlayStatus
    box: tuple[float, float, float, float] | None = None

    @property
    def is_visible(self) -> bool:
        return self.status is OverlayStatus.VISIBLE and self.box is not None


@dataclass(frozen=True, slots=True)
class FrameComposition:
    key: SequenceKey
    frame_number: int
    image_path: Path | None
    ground_truth: BoxOverlay
    trackers: tuple[BoxOverlay, ...]

    @property
    def image_missing(self) -> bool:
        return self.image_path is None


@dataclass(frozen=True, slots=True)
class _LoadedTracker:
    tracker: Tracker
    boxes: BoundingBoxData


@dataclass(frozen=True, slots=True)
class SequenceVisualizationSession:
    images: ImageSequenceIndex
    ground_truth: BoundingBoxData
    trackers: tuple[_LoadedTracker, ...]

    @property
    def key(self) -> SequenceKey:
        return self.images.key

    @property
    def frame_count(self) -> int:
        return self.ground_truth.frame_count

    def tracker_box_data(self, tracker_id: str) -> BoundingBoxData:
        for item in self.trackers:
            if item.tracker.tracker_id == tracker_id:
                return item.boxes
        raise KeyError(tracker_id)

    def compose(
        self,
        frame_number: int,
        *,
        visible_tracker_ids: frozenset[str] | None = None,
    ) -> FrameComposition:
        frame = int(frame_number)
        ground_truth = _overlay_for_frame(
            self.ground_truth.values,
            frame,
            tracker_id="ground_truth",
            display_name="Groundtruth",
            style=GROUND_TRUTH_VISUAL_STYLE,
        )
        overlays = tuple(
            _overlay_for_frame(
                item.boxes.values,
                frame,
                tracker_id=item.tracker.tracker_id,
                display_name=item.tracker.display_name,
                style=item.tracker.style,
            )
            for item in self.trackers
            if visible_tracker_ids is None
            or item.tracker.tracker_id in visible_tracker_ids
        )
        return FrameComposition(
            key=self.key,
            frame_number=frame,
            image_path=self.images.path_for_frame(frame),
            ground_truth=ground_truth,
            trackers=overlays,
        )


def _overlay_for_frame(
    values: NDArray[np.float64],
    frame_number: int,
    *,
    tracker_id: str,
    display_name: str,
    style: TrackerStyle,
) -> BoxOverlay:
    index = frame_number - 1
    if index < 0 or index >= len(values):
        return BoxOverlay(
            tracker_id=tracker_id,
            display_name=display_name,
            style=style,
            status=OverlayStatus.OUT_OF_RANGE,
        )
    row = values[index]
    if np.isnan(row).all():
        status = OverlayStatus.MISSING
    elif row[2] <= 0.0 or row[3] <= 0.0:
        status = OverlayStatus.OFFSCREEN
    else:
        return BoxOverlay(
            tracker_id=tracker_id,
            display_name=display_name,
            style=style,
            status=OverlayStatus.VISIBLE,
            box=tuple(float(value) for value in row),
        )
    return BoxOverlay(
        tracker_id=tracker_id,
        display_name=display_name,
        style=style,
        status=status,
    )


def load_sequence_session(
    images: ImageSequenceIndex,
    ground_truth_path: Path,
    tracker_sources: tuple[TrackerSequenceSource, ...],
) -> SequenceVisualizationSession:
    ground_truth = read_bounding_boxes(ground_truth_path, allow_missing_rows=False)
    trackers = tuple(
        _LoadedTracker(
            tracker=source.tracker,
            boxes=read_bounding_boxes(source.result_path),
        )
        for source in tracker_sources
    )
    return SequenceVisualizationSession(
        images=images,
        ground_truth=ground_truth,
        trackers=trackers,
    )
