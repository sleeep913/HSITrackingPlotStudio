from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from hsi_plotter.models.dataset import SequenceKey


class VisualSelectionError(ValueError):
    """Raised when a requested Visual selection is not valid."""


@dataclass(frozen=True, slots=True)
class ImageSequenceIndex:
    key: SequenceKey
    directory: Path
    frames: Mapping[int, Path]

    def __post_init__(self) -> None:
        normalized = {
            int(frame): Path(path).resolve(strict=False)
            for frame, path in self.frames.items()
        }
        if any(frame < 1 for frame in normalized):
            raise ValueError("图像帧号必须为正整数")
        object.__setattr__(self, "directory", Path(self.directory).resolve(strict=False))
        object.__setattr__(self, "frames", MappingProxyType(normalized))

    @property
    def frame_numbers(self) -> tuple[int, ...]:
        return tuple(sorted(self.frames))

    def path_for_frame(self, frame_number: int) -> Path | None:
        return self.frames.get(int(frame_number))


@dataclass(frozen=True, slots=True)
class VisualSequenceSelection:
    key: SequenceKey
    frame_numbers: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class VisualSelectionList:
    sequences: tuple[VisualSequenceSelection, ...] = ()

    @property
    def is_complete(self) -> bool:
        return bool(self.sequences) and all(
            len(selection.frame_numbers) == 3 for selection in self.sequences
        )

    def add(self, key: SequenceKey, frame_number: int) -> VisualSelectionList:
        frame = int(frame_number)
        if frame < 1:
            raise VisualSelectionError("Visual 帧号必须为正整数")
        for index, selection in enumerate(self.sequences):
            if selection.key != key:
                continue
            if frame in selection.frame_numbers:
                return self
            if len(selection.frame_numbers) >= 3:
                raise VisualSelectionError(f"序列 {key.name} 最多选择 3 帧")
            updated = replace(
                selection,
                frame_numbers=tuple(sorted((*selection.frame_numbers, frame))),
            )
            values = list(self.sequences)
            values[index] = updated
            return replace(self, sequences=tuple(values))
        return replace(
            self,
            sequences=(
                *self.sequences,
                VisualSequenceSelection(key=key, frame_numbers=(frame,)),
            ),
        )

    def remove_frame(self, key: SequenceKey, frame_number: int) -> VisualSelectionList:
        values: list[VisualSequenceSelection] = []
        for selection in self.sequences:
            if selection.key != key:
                values.append(selection)
                continue
            frames = tuple(frame for frame in selection.frame_numbers if frame != frame_number)
            if frames:
                values.append(replace(selection, frame_numbers=frames))
        return replace(self, sequences=tuple(values))

    def remove_sequence(self, key: SequenceKey) -> VisualSelectionList:
        return replace(
            self,
            sequences=tuple(item for item in self.sequences if item.key != key),
        )

    def clear(self) -> VisualSelectionList:
        return replace(self, sequences=())

