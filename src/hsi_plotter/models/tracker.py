from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hsi_plotter.models.style import (
    DEFAULT_COLOR_PALETTE,
    DEFAULT_LINE_STYLES,
    TrackerStyle,
    default_style_for_index,
)
from hsi_plotter.models.tracker_import import ResultLayout

_MAX_STYLE_COMBINATIONS = len(DEFAULT_COLOR_PALETTE) * len(DEFAULT_LINE_STYLES)


@dataclass(frozen=True, slots=True)
class Tracker:
    tracker_id: str
    display_name: str
    results_path: Path
    style: TrackerStyle
    layout: ResultLayout = ResultLayout.FLAT


def assign_stable_styles(
    previous: Sequence[Tracker],
    selected: Sequence[Tracker],
) -> tuple[Tracker, ...]:
    """Keep known tracker styles and give new trackers the next unused one.

    A tracker keeps its colour, line style and width even when trackers are
    added, removed or reordered (FR-STYLE-003 / FR-PROJ-009).
    """
    known_styles = {tracker.tracker_id: tracker.style for tracker in previous}
    taken = {
        known_styles[tracker.tracker_id]
        for tracker in selected
        if tracker.tracker_id in known_styles
    }

    result: list[Tracker] = []
    next_index = 0
    for tracker in selected:
        style = known_styles.get(tracker.tracker_id)
        if style is None:
            attempts = 0
            while True:
                candidate = default_style_for_index(next_index)
                next_index += 1
                attempts += 1
                if candidate not in taken or attempts > _MAX_STYLE_COMBINATIONS:
                    break
            taken.add(candidate)
            style = candidate
        result.append(replace(tracker, style=style))
    return tuple(result)
