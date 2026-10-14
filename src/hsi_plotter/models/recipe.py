from __future__ import annotations

from dataclasses import dataclass, field, replace

from hsi_plotter.models.style import PlotStyle, TrackerStyle

RECIPE_VERSION = 2
LEGACY_RECIPE_VERSION = 1
FIGURE_TYPE = "tracking_curves"
PLOT_MODES = frozenset({"both", "precision", "success"})


class RecipeCompatibilityError(ValueError):
    """Raised when a recipe is invalid or belongs to an unsupported format."""


@dataclass(frozen=True, slots=True)
class RecipeModalityMetrics:
    modality: str
    auc: float
    dp: float


@dataclass(frozen=True, slots=True)
class RecipeTracker:
    tracker_id: str
    display_name: str
    visible: bool = True
    order: int = 0
    style: TrackerStyle | None = None
    auc: float | None = None
    dp: float | None = None
    precision_curve: tuple[float, ...] = ()
    success_curve: tuple[float, ...] = ()
    modality_metrics: tuple[RecipeModalityMetrics, ...] = ()

    @property
    def has_embedded_data(self) -> bool:
        return (
            self.auc is not None
            and self.dp is not None
            and len(self.precision_curve) == 51
            and len(self.success_curve) == 51
        )


@dataclass(frozen=True, slots=True)
class CurveRecipe:
    name: str = "曲线配方"
    recipe_version: int = RECIPE_VERSION
    figure_type: str = FIGURE_TYPE
    plot_mode: str = "both"
    scope: str = "overall"
    modality: str | None = None
    trackers: tuple[RecipeTracker, ...] = ()
    precision_order: tuple[str, ...] = ()
    success_order: tuple[str, ...] = ()
    follow_metric_sort: bool = True
    table_rank_metric: str = "auc"
    style: PlotStyle = field(default_factory=PlotStyle)
    precision_title: str = "Precision plots of OPE"
    success_title: str = "Success plots of OPE"
    precision_x_label: str = "Location error threshold"
    precision_y_label: str = "Precision"
    success_x_label: str = "Overlap threshold"
    success_y_label: str = "Success rate"
    precision_x_min: float = 0.0
    precision_x_max: float = 50.0
    success_x_min: float = 0.0
    success_x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    precision_tick_step: float = 10.0
    success_tick_step: float = 0.2
    y_tick_step: float = 0.2
    decimal_places: int = 3
    export_formats: tuple[str, ...] = ("png", "pdf", "svg", "eps")
    filename_template: str = "tracking-curves"

    def validate(self) -> None:
        if self.recipe_version != RECIPE_VERSION:
            raise RecipeCompatibilityError(
                f"不支持的配方版本 {self.recipe_version}，当前支持 {RECIPE_VERSION}"
            )
        if self.figure_type != FIGURE_TYPE:
            raise RecipeCompatibilityError(f"配方图形类型不是 {FIGURE_TYPE}")
        if self.plot_mode not in PLOT_MODES:
            raise RecipeCompatibilityError(f"不支持的曲线模式：{self.plot_mode}")
        if self.table_rank_metric not in {"auc", "dp", "manual"}:
            raise RecipeCompatibilityError(f"不支持的表格排名方式：{self.table_rank_metric}")
        if not 0 <= self.decimal_places <= 8:
            raise RecipeCompatibilityError("图例小数位数必须位于 0 到 8 之间")
        if self.precision_x_min >= self.precision_x_max:
            raise RecipeCompatibilityError("Precision 横轴范围无效")
        if self.success_x_min >= self.success_x_max or self.y_min >= self.y_max:
            raise RecipeCompatibilityError("坐标轴范围无效")
        if min(self.precision_tick_step, self.success_tick_step, self.y_tick_step) <= 0:
            raise RecipeCompatibilityError("刻度间隔必须大于 0")
        self.style.validate()
        for tracker in self.trackers:
            if tracker.precision_curve and len(tracker.precision_curve) != 51:
                raise RecipeCompatibilityError("Precision 曲线必须包含 51 个点")
            if tracker.success_curve and len(tracker.success_curve) != 51:
                raise RecipeCompatibilityError("Success 曲线必须包含 51 个点")

    @property
    def is_self_contained(self) -> bool:
        return bool(self.trackers) and all(tracker.has_embedded_data for tracker in self.trackers)

    def with_updates(self, **changes: object) -> CurveRecipe:
        updated = replace(self, **changes)
        updated.validate()
        return updated


@dataclass(frozen=True, slots=True)
class RecipeTrackerResolution:
    resolved_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]


def resolve_recipe_trackers(
    recipe: CurveRecipe,
    available_ids: set[str],
    *,
    mapping: dict[str, str] | None = None,
    skipped: set[str] | None = None,
) -> RecipeTrackerResolution:
    mapping = mapping or {}
    skipped = skipped or set()
    resolved: list[str] = []
    missing: list[str] = []
    for tracker in sorted(recipe.trackers, key=lambda item: item.order):
        if tracker.tracker_id in skipped:
            continue
        candidate = mapping.get(tracker.tracker_id, tracker.tracker_id)
        if candidate not in available_ids:
            missing.append(tracker.tracker_id)
            continue
        if candidate not in resolved:
            resolved.append(candidate)
    return RecipeTrackerResolution(tuple(resolved), tuple(missing))
