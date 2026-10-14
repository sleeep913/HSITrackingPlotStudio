from pathlib import Path

from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.models.style import TrackerStyle, default_style_for_index
from hsi_plotter.models.tracker import Tracker, assign_stable_styles
from hsi_plotter.models.tracker_import import ResultLayout


def _tracker(identifier: str, layout: ResultLayout = ResultLayout.FLAT) -> Tracker:
    return Tracker(
        tracker_id=identifier,
        display_name=identifier.upper(),
        results_path=Path("results") / identifier,
        layout=layout,
        style=default_style_for_index(0),
    )


def test_new_trackers_receive_the_palette_in_order() -> None:
    selected = [_tracker("a"), _tracker("b"), _tracker("c")]

    styled = assign_stable_styles([], selected)

    assert [tracker.style.color for tracker in styled] == [
        default_style_for_index(0).color,
        default_style_for_index(1).color,
        default_style_for_index(2).color,
    ]


def test_known_trackers_keep_their_style_when_reordered() -> None:
    previous = [
        _tracker("a"),
        Tracker(
            tracker_id="b",
            display_name="B",
            results_path=Path("results") / "b",
            layout=ResultLayout.FLAT,
            style=TrackerStyle(color="#010203", line_style="--", line_width=3.0),
        ),
    ]
    selected = [_tracker("b"), _tracker("a")]

    styled = assign_stable_styles(previous, selected)

    assert styled[0].tracker_id == "b"
    assert styled[0].style.color == "#010203"
    assert styled[1].tracker_id == "a"


def test_removed_then_reimported_tracker_recovers_its_style() -> None:
    known = Tracker(
        tracker_id="mht",
        display_name="MHT",
        results_path=Path("results") / "MHT",
        layout=ResultLayout.FLAT,
        style=TrackerStyle(color="#0F766E", line_style="-."),
    )

    restored = assign_stable_styles([known], [_tracker("mht")])

    assert restored[0].style == known.style


def test_added_tracker_does_not_reuse_a_style_still_in_use() -> None:
    previous = [_tracker("a")]
    selected = [_tracker("a"), _tracker("new")]

    styled = assign_stable_styles(previous, selected)

    assert styled[0].style == default_style_for_index(0)
    assert styled[1].style != styled[0].style


def test_project_model_accepts_layout_changes() -> None:
    project = Project(name="HOT2023", sources=ProjectSources())

    styled = assign_stable_styles([], [_tracker("mht", ResultLayout.MODALITY_SUBDIR)])

    project.trackers = list(styled)

    assert project.trackers[0].layout is ResultLayout.MODALITY_SUBDIR
