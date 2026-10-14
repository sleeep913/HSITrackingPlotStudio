from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hsi_plotter.models.cache import CacheManifest
from hsi_plotter.models.style import PlotStyle
from hsi_plotter.models.tracker import Tracker


@dataclass(frozen=True, slots=True)
class ProjectSources:
    ground_truth: Path | None = None
    results_root: Path | None = None
    image_root: Path | None = None
    attribute_file: Path | None = None
    export_directory: Path | None = None
    hsi_files: tuple[Path, ...] = ()


@dataclass(slots=True)
class Project:
    name: str
    sources: ProjectSources = field(default_factory=ProjectSources)
    trackers: list[Tracker] = field(default_factory=list)
    recipe_files: list[Path] = field(default_factory=list)
    plot_style: PlotStyle = field(default_factory=PlotStyle)
    cache_manifest: CacheManifest = field(default_factory=CacheManifest)
    storage_path: Path | None = field(default=None, repr=False, compare=False)
