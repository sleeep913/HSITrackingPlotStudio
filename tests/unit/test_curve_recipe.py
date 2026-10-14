from pathlib import Path

import pytest

from hsi_plotter.io.recipe_store import load_curve_recipe, save_curve_recipe
from hsi_plotter.models.recipe import (
    CurveRecipe,
    RecipeCompatibilityError,
    RecipeModalityMetrics,
    RecipeTracker,
    resolve_recipe_trackers,
)
from hsi_plotter.models.style import PlotStyle, TrackerStyle


def _recipe() -> CurveRecipe:
    return CurveRecipe(
        name="HOT2023 paper curves",
        plot_mode="both",
        scope="overall",
        trackers=(
            RecipeTracker(
                "mht",
                "MHT",
                True,
                0,
                TrackerStyle("#0072BD", "-", 2.0),
                0.55,
                0.75,
                tuple(range(51)),
                tuple(reversed(range(51))),
                (RecipeModalityMetrics("VIS", 0.6, 0.8),),
            ),
            RecipeTracker(
                "spirit",
                "SPIRIT",
                False,
                1,
                TrackerStyle("#D95319", "--", 2.0),
                0.5,
                0.7,
                tuple(range(51)),
                tuple(reversed(range(51))),
            ),
        ),
        style=PlotStyle(
            width_inches=7.2,
            height_inches=3.1,
            legend_columns=2,
            legend_scale=1.35,
        ),
        table_rank_metric="dp",
        precision_title="Precision plots of OPE",
        success_title="Success plots of OPE",
        export_formats=("png", "pdf", "svg", "eps"),
    )


def test_curve_recipe_round_trip_is_readable_json(tmp_path: Path) -> None:
    destination = tmp_path / "paper.hsirecipe.json"

    save_curve_recipe(_recipe(), destination)
    loaded = load_curve_recipe(destination)

    assert loaded == _recipe()
    assert loaded.is_self_contained
    assert '"recipe_version": 2' in destination.read_text(encoding="utf-8")


def test_legacy_v1_recipe_is_migrated_without_embedded_data(tmp_path: Path) -> None:
    path = tmp_path / "legacy.hsirecipe.json"
    path.write_text(
        '{"recipe_version":1,"figure_type":"tracking_curves","name":"old",'
        '"style":{"preset":"paper_double_column"},'
        '"trackers":[{"tracker_id":"mht","display_name":"MHT"}]}',
        encoding="utf-8",
    )

    loaded = load_curve_recipe(path)

    assert loaded.recipe_version == 2
    assert not loaded.is_self_contained
    assert loaded.trackers[0].display_name == "MHT"


def test_recipe_rejects_unknown_version(tmp_path: Path) -> None:
    path = tmp_path / "future.hsirecipe.json"
    path.write_text('{"recipe_version": 999, "figure_type": "tracking_curves"}')

    with pytest.raises(RecipeCompatibilityError, match="版本"):
        load_curve_recipe(path)


def test_version_two_recipe_cannot_be_saved_as_an_external_reference(tmp_path: Path) -> None:
    recipe = CurveRecipe(trackers=(RecipeTracker("mht", "MHT"),))

    with pytest.raises(RecipeCompatibilityError, match="必须包含"):
        save_curve_recipe(recipe, tmp_path / "incomplete.hsirecipe.json")


def test_missing_recipe_trackers_require_explicit_mapping_or_skip() -> None:
    recipe = _recipe()

    resolution = resolve_recipe_trackers(recipe, {"mht"})

    assert resolution.resolved_ids == ("mht",)
    assert resolution.missing_ids == ("spirit",)
    mapped = resolve_recipe_trackers(recipe, {"mht", "new"}, mapping={"spirit": "new"})
    assert mapped.resolved_ids == ("mht", "new")


def test_recipe_rejects_out_of_range_legend_scale() -> None:
    recipe = CurveRecipe(style=PlotStyle(legend_scale=1.9))

    with pytest.raises(ValueError, match="图例缩放"):
        recipe.validate()
