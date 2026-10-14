"""Pure result models for attribute-based tracker evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from hsi_plotter.models.dataset import SequenceKey


@dataclass(frozen=True, slots=True)
class AttributeDefinition:
    name: str
    sequences: tuple[str, ...]

    def __post_init__(self) -> None:
        name = self.name.strip()
        sequences = tuple(item.strip() for item in self.sequences if item.strip())
        if not name:
            raise ValueError("属性名不能为空")
        if not sequences:
            raise ValueError(f"属性 {name} 没有任何序列")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sequences", sequences)


@dataclass(frozen=True, slots=True)
class AttributeResolution:
    keys_by_attribute: dict[str, tuple[SequenceKey, ...]] = field(default_factory=dict)
    unknown_by_attribute: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttributeScore:
    auc: float | None
    dp: float | None
    resolved_count: int
    succeeded_count: int
    failed_count: int

    @property
    def is_computable(self) -> bool:
        return self.auc is not None and self.dp is not None


@dataclass(frozen=True, slots=True)
class TrackerAttributeResult:
    tracker_id: str
    tracker_name: str
    scores: dict[str, AttributeScore] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttributeEvaluationRun:
    attributes: tuple[AttributeDefinition, ...] = ()
    trackers: tuple[TrackerAttributeResult, ...] = ()
    unknown_by_attribute: dict[str, tuple[str, ...]] = field(default_factory=dict)
    issues: tuple[str, ...] = ()

    def score(self, tracker_id: str, attribute_name: str) -> AttributeScore | None:
        for tracker in self.trackers:
            if tracker.tracker_id == tracker_id:
                return tracker.scores.get(attribute_name)
        return None

