from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from hsi_plotter.models.recipe import (
    FIGURE_TYPE,
    LEGACY_RECIPE_VERSION,
    RECIPE_VERSION,
    CurveRecipe,
    RecipeCompatibilityError,
    RecipeModalityMetrics,
    RecipeTracker,
)
from hsi_plotter.models.style import PlotStyle, PlotStylePreset, TrackerStyle


def save_curve_recipe(recipe: CurveRecipe, destination: Path) -> Path:
    recipe.validate()
    if not recipe.is_self_contained:
        raise RecipeCompatibilityError("版本 2 配方必须包含全部跟踪器曲线、指标和样式")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(recipe)
    payload["style"]["preset"] = recipe.style.preset.value
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RecipeCompatibilityError(f"配方字段 {name} 必须是对象")
    return value


def load_curve_recipe(source: Path) -> CurveRecipe:
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecipeCompatibilityError(f"无法读取绘图配方 {path}：{exc}") from exc
    root = _require_mapping(payload, "root")
    version = root.get("recipe_version")
    if version not in {LEGACY_RECIPE_VERSION, RECIPE_VERSION}:
        raise RecipeCompatibilityError(
            f"不支持的配方版本 {version!r}，当前支持 {LEGACY_RECIPE_VERSION} 和 {RECIPE_VERSION}"
        )
    if root.get("figure_type") != FIGURE_TYPE:
        raise RecipeCompatibilityError(f"不支持的配方图形类型：{root.get('figure_type')!r}")
    style_payload = _require_mapping(root.get("style"), "style")
    try:
        style = PlotStyle(
            **{
                **style_payload,
                "preset": PlotStylePreset(
                    style_payload.get("preset", PlotStylePreset.PAPER_DOUBLE_COLUMN)
                ),
            }
        )
        tracker_values = root.get("trackers", [])
        if not isinstance(tracker_values, list):
            raise RecipeCompatibilityError("配方字段 trackers 必须是数组")
        trackers: list[RecipeTracker] = []
        for item in tracker_values:
            tracker_payload = _require_mapping(item, "tracker")
            tracker_style_payload = tracker_payload.get("style")
            tracker_style = None
            if tracker_style_payload is not None:
                tracker_style = TrackerStyle(
                    **_require_mapping(tracker_style_payload, "tracker.style")
                )
            modality_values = tracker_payload.get("modality_metrics", [])
            if not isinstance(modality_values, list):
                raise RecipeCompatibilityError("配方字段 modality_metrics 必须是数组")
            trackers.append(
                RecipeTracker(
                    tracker_id=str(tracker_payload["tracker_id"]),
                    display_name=str(tracker_payload["display_name"]),
                    visible=bool(tracker_payload.get("visible", True)),
                    order=int(tracker_payload.get("order", 0)),
                    style=tracker_style,
                    auc=(
                        None
                        if tracker_payload.get("auc") is None
                        else float(tracker_payload["auc"])
                    ),
                    dp=(
                        None
                        if tracker_payload.get("dp") is None
                        else float(tracker_payload["dp"])
                    ),
                    precision_curve=tuple(
                        float(value) for value in tracker_payload.get("precision_curve", [])
                    ),
                    success_curve=tuple(
                        float(value) for value in tracker_payload.get("success_curve", [])
                    ),
                    modality_metrics=tuple(
                        RecipeModalityMetrics(**_require_mapping(value, "modality_metric"))
                        for value in modality_values
                    ),
                )
            )
        recipe = CurveRecipe(
            **{
                **{
                    key: value
                    for key, value in root.items()
                    if key not in {"recipe_version", "style", "trackers"}
                },
                "recipe_version": RECIPE_VERSION,
                "style": style,
                "trackers": tuple(trackers),
                "precision_order": tuple(root.get("precision_order", ())),
                "success_order": tuple(root.get("success_order", ())),
                "export_formats": tuple(root.get("export_formats", ("png", "pdf", "svg", "eps"))),
            }
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, RecipeCompatibilityError):
            raise
        raise RecipeCompatibilityError(f"绘图配方字段无效：{exc}") from exc
    recipe.validate()
    return recipe
