from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from hsi_plotter.models.dataset import MODALITY_LABELS, Modality, SequenceKey
from hsi_plotter.validation.issues import PreflightReport


class ResultLayout(StrEnum):
    """Supported tracker-result directory layouts.

    See FR-DATA-012: `<root>/<tracker>/<sequence>.txt`,
    `<root>/<tracker>/<modality>/<sequence>.txt` and
    `<root>/<tracker>/<modality>-<sequence>.txt`.
    """

    FLAT = "flat"
    PREFIXED_FLAT = "prefixed_flat"
    MODALITY_SUBDIR = "modality_subdir"


RESULT_LAYOUT_LABELS = {
    ResultLayout.FLAT: "序列目录",
    ResultLayout.PREFIXED_FLAT: "带模态前缀",
    ResultLayout.MODALITY_SUBDIR: "模态子目录",
}

RESULT_LAYOUT_DESCRIPTIONS = {
    ResultLayout.FLAT: "结果根目录 / 跟踪器 / 序列.txt",
    ResultLayout.PREFIXED_FLAT: "结果根目录 / 跟踪器 / [all/] / vis-序列.txt",
    ResultLayout.MODALITY_SUBDIR: "结果根目录 / 跟踪器 / VIS / 序列.txt",
}


@dataclass(frozen=True, slots=True)
class LayoutCandidate:
    """One deterministic interpretation of a tracker result directory."""

    layout: ResultLayout
    root_path: Path
    files: dict[SequenceKey, Path] = field(default_factory=dict)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def label(self) -> str:
        return RESULT_LAYOUT_LABELS[self.layout]


@dataclass(frozen=True, slots=True)
class TrackerDiscovery:
    tracker_id: str
    tracker_name: str
    root_path: Path
    candidates: tuple[LayoutCandidate, ...] = ()
    best_candidates: tuple[LayoutCandidate, ...] = ()
    report: PreflightReport = field(default_factory=PreflightReport)

    @property
    def preferred(self) -> LayoutCandidate | None:
        return self.best_candidates[0] if self.best_candidates else None

    @property
    def is_ambiguous(self) -> bool:
        """True when several layouts explain the data equally well (FR-DATA-013)."""
        return len(self.best_candidates) > 1

    @property
    def has_error(self) -> bool:
        return self.report.has_errors


@dataclass(frozen=True, slots=True)
class SequenceMapping:
    """Resolved Ground Truth / tracker-result pairing for a single tracker."""

    tracker_id: str
    tracker_name: str
    layout: ResultLayout
    root_path: Path
    pairs: dict[SequenceKey, tuple[Path, Path]] = field(default_factory=dict)
    missing: tuple[SequenceKey, ...] = ()
    extra: tuple[SequenceKey, ...] = ()
    ambiguous: tuple[SequenceKey, ...] = ()
    report: PreflightReport = field(default_factory=PreflightReport)

    @property
    def matched_count(self) -> int:
        return len(self.pairs)

    @property
    def missing_count(self) -> int:
        return len(self.missing)

    @property
    def extra_count(self) -> int:
        return len(self.extra)

    @property
    def ambiguous_count(self) -> int:
        return len(self.ambiguous)

    @property
    def modalities(self) -> tuple[Modality, ...]:
        return tuple(sorted({key.modality for key in self.pairs}, key=lambda item: item.value))

    @property
    def modality_label(self) -> str:
        covered = self.modalities
        if not covered:
            return "—"
        return ", ".join(MODALITY_LABELS[modality] for modality in covered)


def layout_label(layout: ResultLayout) -> str:
    return RESULT_LAYOUT_LABELS[layout]


def layout_description(layout: ResultLayout) -> str:
    return RESULT_LAYOUT_DESCRIPTIONS[layout]
