from __future__ import annotations

from dataclasses import dataclass, field

from hsi_plotter.models.dataset import Modality
from hsi_plotter.models.visualization import VisualSelectionList


@dataclass(frozen=True, slots=True)
class VisualizationRecipe:
    modality: Modality | None = None
    sequence_name: str | None = None
    primary_tracker_id: str | None = None
    tracker_ids: tuple[str, ...] = ()
    show_ground_truth: bool = True
    show_legend: bool = True
    show_names: bool = False
    show_frame_number: bool = True
    play_start: int = 1
    play_end: int = 1
    play_fps: float = 10.0
    loop: bool = False
    video_resolution: tuple[int, int] = (1280, 720)
    candidate_frames: tuple[int, ...] = ()
    visual_selections: VisualSelectionList = field(default_factory=VisualSelectionList)

    def validate(self) -> None:
        if self.play_start < 1 or self.play_end < self.play_start:
            raise ValueError("播放帧范围无效")
        if self.play_fps <= 0:
            raise ValueError("播放帧率必须大于 0")
        if self.video_resolution[0] <= 0 or self.video_resolution[1] <= 0:
            raise ValueError("视频分辨率必须大于 0")
        if any(frame < 1 for frame in self.candidate_frames):
            raise ValueError("候选帧号必须为正整数")

