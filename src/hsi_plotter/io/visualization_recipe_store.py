from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hsi_plotter.io.result_writer import _temporary_target
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.visualization import VisualSelectionError, VisualSelectionList
from hsi_plotter.models.visualization_recipe import VisualizationRecipe

RECIPE_TYPE = "tracking_visualization"
SCHEMA_VERSION = 1


class VisualizationRecipeFormatError(ValueError):
    """Raised when a tracking-visualization recipe is invalid."""


@dataclass(frozen=True, slots=True)
class VisualizationRecipeReferenceIssues:
    missing_tracker_ids: tuple[str, ...] = ()
    missing_sequences: tuple[SequenceKey, ...] = ()

    @property
    def has_issues(self) -> bool:
        return bool(self.missing_tracker_ids or self.missing_sequences)


def encode_visualization_recipe(recipe: VisualizationRecipe) -> dict[str, object]:
    recipe.validate()
    return {
        "recipe_type": RECIPE_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "modality": recipe.modality.value if recipe.modality is not None else None,
            "sequence_name": recipe.sequence_name,
        },
        "trackers": {
            "primary_tracker_id": recipe.primary_tracker_id,
            "tracker_ids": list(recipe.tracker_ids),
        },
        "display": {
            "show_ground_truth": recipe.show_ground_truth,
            "show_legend": recipe.show_legend,
            "show_names": recipe.show_names,
            "show_frame_number": recipe.show_frame_number,
        },
        "playback": {
            "start": recipe.play_start,
            "end": recipe.play_end,
            "fps": recipe.play_fps,
            "loop": recipe.loop,
        },
        "video": {
            "resolution": list(recipe.video_resolution),
        },
        "candidate_frames": list(recipe.candidate_frames),
        "visual_selections": [
            {
                "dataset": item.key.dataset,
                "modality": item.key.modality.value,
                "name": item.key.name,
                "frames": list(item.frame_numbers),
            }
            for item in recipe.visual_selections.sequences
        ],
    }


def save_visualization_recipe(recipe: VisualizationRecipe, destination: Path) -> Path:
    target = Path(destination)
    try:
        text = json.dumps(encode_visualization_recipe(recipe), ensure_ascii=False, indent=2)
        with _temporary_target(target) as temporary:
            temporary.write_text(text + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise VisualizationRecipeFormatError(f"无法保存可视化配方 {target}：{exc}") from exc
    return target


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VisualizationRecipeFormatError(f"字段 {field} 必须是对象")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VisualizationRecipeFormatError(f"字段 {field} 必须是整数")
    return value


def _selections(values: Any) -> VisualSelectionList:
    if not isinstance(values, list):
        raise VisualizationRecipeFormatError("字段 visual_selections 必须是数组")
    selections = VisualSelectionList()
    try:
        for index, raw in enumerate(values):
            item = _mapping(raw, f"visual_selections[{index}]")
            key = SequenceKey(
                dataset=str(item["dataset"]),
                modality=Modality(str(item["modality"])),
                name=str(item["name"]),
            )
            frames = item.get("frames")
            if not isinstance(frames, list):
                raise VisualizationRecipeFormatError(
                    f"字段 visual_selections[{index}].frames 必须是数组"
                )
            for frame in frames:
                selections = selections.add(key, _integer(frame, "frames[]"))
    except (KeyError, ValueError, VisualSelectionError) as exc:
        if isinstance(exc, VisualizationRecipeFormatError):
            raise
        raise VisualizationRecipeFormatError(f"Visual 清单无效：{exc}") from exc
    return selections


def decode_visualization_recipe(root: dict[str, Any]) -> VisualizationRecipe:
    if root.get("recipe_type") != RECIPE_TYPE:
        raise VisualizationRecipeFormatError("recipe_type 不是 tracking_visualization")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise VisualizationRecipeFormatError(
            f"不支持的 schema_version：{root.get('schema_version')!r}"
        )
    scope = _mapping(root.get("scope", {}), "scope")
    trackers = _mapping(root.get("trackers", {}), "trackers")
    display = _mapping(root.get("display", {}), "display")
    playback = _mapping(root.get("playback", {}), "playback")
    video = _mapping(root.get("video", {}), "video")
    tracker_ids = trackers.get("tracker_ids", [])
    candidate_frames = root.get("candidate_frames", [])
    resolution = video.get("resolution", [1280, 720])
    if not isinstance(tracker_ids, list) or not isinstance(candidate_frames, list):
        raise VisualizationRecipeFormatError("跟踪器或候选帧字段必须是数组")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise VisualizationRecipeFormatError("视频分辨率必须包含宽度和高度")
    modality_value = scope.get("modality")
    recipe = VisualizationRecipe(
        modality=Modality(str(modality_value)) if modality_value is not None else None,
        sequence_name=(
            str(scope["sequence_name"])
            if scope.get("sequence_name") is not None
            else None
        ),
        primary_tracker_id=(
            str(trackers["primary_tracker_id"])
            if trackers.get("primary_tracker_id") is not None
            else None
        ),
        tracker_ids=tuple(str(item) for item in tracker_ids),
        show_ground_truth=bool(display.get("show_ground_truth", True)),
        show_legend=bool(display.get("show_legend", True)),
        show_names=bool(display.get("show_names", False)),
        show_frame_number=bool(display.get("show_frame_number", True)),
        play_start=_integer(playback.get("start", 1), "playback.start"),
        play_end=_integer(playback.get("end", 1), "playback.end"),
        play_fps=float(playback.get("fps", 10.0)),
        loop=bool(playback.get("loop", False)),
        video_resolution=(
            _integer(resolution[0], "video.resolution[0]"),
            _integer(resolution[1], "video.resolution[1]"),
        ),
        candidate_frames=tuple(
            _integer(item, "candidate_frames[]") for item in candidate_frames
        ),
        visual_selections=_selections(root.get("visual_selections", [])),
    )
    try:
        recipe.validate()
    except ValueError as exc:
        raise VisualizationRecipeFormatError(str(exc)) from exc
    return recipe


def load_visualization_recipe(source: Path) -> VisualizationRecipe:
    path = Path(source)
    try:
        root = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VisualizationRecipeFormatError(f"无法读取可视化配方 {path}：{exc}") from exc
    return decode_visualization_recipe(_mapping(root, "root"))


def validate_visualization_recipe_references(
    recipe: VisualizationRecipe,
    *,
    available_tracker_ids: frozenset[str],
    available_sequences: frozenset[SequenceKey],
) -> VisualizationRecipeReferenceIssues:
    requested_trackers = tuple(
        dict.fromkeys(
            item
            for item in (recipe.primary_tracker_id, *recipe.tracker_ids)
            if item is not None
        )
    )
    missing_trackers = tuple(
        item for item in requested_trackers if item not in available_tracker_ids
    )
    requested_sequences = [item.key for item in recipe.visual_selections.sequences]
    if recipe.modality is not None and recipe.sequence_name:
        requested_sequences.insert(
            0,
            SequenceKey("project", recipe.modality, recipe.sequence_name),
        )
    missing_sequences = tuple(
        dict.fromkeys(
            key for key in requested_sequences if key not in available_sequences
        )
    )
    return VisualizationRecipeReferenceIssues(
        missing_tracker_ids=missing_trackers,
        missing_sequences=missing_sequences,
    )
