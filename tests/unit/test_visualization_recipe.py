from pathlib import Path

import pytest

from hsi_plotter.io.visualization_recipe_store import (
    VisualizationRecipeFormatError,
    load_visualization_recipe,
    save_visualization_recipe,
    validate_visualization_recipe_references,
)
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.visualization import VisualSelectionList
from hsi_plotter.models.visualization_recipe import VisualizationRecipe


def _recipe() -> VisualizationRecipe:
    vis = SequenceKey("project", Modality.VIS, "ball")
    nir = SequenceKey("project", Modality.NIR, "car")
    selections = VisualSelectionList().add(vis, 2).add(vis, 5).add(nir, 3)
    return VisualizationRecipe(
        modality=Modality.VIS,
        sequence_name="ball",
        primary_tracker_id="primary-id",
        tracker_ids=("primary-id", "competitor-id"),
        show_ground_truth=True,
        show_legend=False,
        show_names=True,
        show_frame_number=True,
        play_start=2,
        play_end=10,
        play_fps=5.0,
        loop=True,
        video_resolution=(1280, 640),
        candidate_frames=(4, 8, 12),
        visual_selections=selections,
    )


def test_visualization_recipe_round_trip_without_embedding_source_data(
    tmp_path: Path,
) -> None:
    target = tmp_path / "visual.hsirecipe.json"

    save_visualization_recipe(_recipe(), target)
    loaded = load_visualization_recipe(target)

    assert loaded == _recipe()
    payload = target.read_text(encoding="utf-8")
    assert "primary-id" in payload
    assert ".jpg" not in payload
    assert ".txt" not in payload
    assert "image_root" not in payload


def test_visualization_recipe_rejects_unsupported_version(tmp_path: Path) -> None:
    target = tmp_path / "future.hsirecipe.json"
    target.write_text(
        '{"recipe_type":"tracking_visualization","schema_version":99}',
        encoding="utf-8",
    )

    with pytest.raises(VisualizationRecipeFormatError, match="schema_version"):
        load_visualization_recipe(target)


def test_reference_validation_reports_ids_and_sequences_without_name_fallback() -> None:
    recipe = _recipe()
    issues = validate_visualization_recipe_references(
        recipe,
        available_tracker_ids=frozenset({"another-primary", "competitor-id"}),
        available_sequences=frozenset(
            {SequenceKey("project", Modality.VIS, "ball")}
        ),
    )

    assert issues.missing_tracker_ids == ("primary-id",)
    assert issues.missing_sequences == (
        SequenceKey("project", Modality.NIR, "car"),
    )
