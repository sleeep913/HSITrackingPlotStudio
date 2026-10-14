from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.visualization import ImageSequenceIndex
from hsi_plotter.validation.issues import (
    PreflightReport,
    Severity,
    ValidationIssue,
)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


@dataclass(slots=True)
class ImageSequenceDiscoveryResult:
    sequences: dict[SequenceKey, ImageSequenceIndex]
    report: PreflightReport


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


_FALSE_COLOR_MODALITIES = {
    "hsivisfalsecolor": Modality.VIS,
    "visfalsecolor": Modality.VIS,
    "hsinirfalsecolor": Modality.NIR,
    "nirfalsecolor": Modality.NIR,
    "hsirednirfalsecolor": Modality.REDNIR,
    "rednirfalsecolor": Modality.REDNIR,
}


def _directory_modality(path: Path) -> Modality | None:
    return _FALSE_COLOR_MODALITIES.get(_normalized_name(path.name))


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def _sequence_directories(root: Path) -> list[Path]:
    children = sorted(
        (child for child in root.iterdir() if child.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    with_images = [child for child in children if _image_files(child)]
    if with_images:
        return with_images
    if any(
        path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        for path in root.iterdir()
    ):
        return [root]
    return []


def _frame_index(
    directory: Path,
    *,
    key: SequenceKey,
    report: PreflightReport,
) -> dict[int, Path]:
    grouped: dict[int, list[Path]] = {}
    for path in _image_files(directory):
        if not path.stem.isdigit():
            continue
        frame_number = int(path.stem)
        if frame_number < 1:
            continue
        grouped.setdefault(frame_number, []).append(path.resolve(strict=False))

    frames: dict[int, Path] = {}
    for frame_number, paths in sorted(grouped.items()):
        if len(paths) > 1:
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="DUPLICATE_IMAGE_FRAME",
                    message=(
                        f"序列 {key.identifier} 的帧 {frame_number} 对应多个图像，"
                        "无法自动选择。"
                    ),
                    path=directory.resolve(strict=False),
                    sequence=key.identifier,
                    modality=key.modality.value,
                    suggestion="移除重复帧图像，确保每个数字帧号只对应一个文件。",
                )
            )
            continue
        frames[frame_number] = paths[0]
    return frames


def _modality_roots(directory: Path) -> list[tuple[Path, Modality]]:
    if modality := _directory_modality(directory):
        return [(directory, modality)]
    discovered = [
        (child, modality)
        for child in directory.iterdir()
        if child.is_dir() and (modality := _directory_modality(child)) is not None
    ]
    if discovered:
        return sorted(discovered, key=lambda item: item[0].name.casefold())
    return [(directory, Modality.UNSPECIFIED)]


def discover_image_sequences(root: Path, *, dataset: str) -> ImageSequenceDiscoveryResult:
    directory = Path(root)
    report = PreflightReport()
    sequences: dict[SequenceKey, ImageSequenceIndex] = {}
    if not directory.is_dir():
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code="DIRECTORY_NOT_FOUND",
                message=f"图像目录不存在或不可访问：{directory}",
                path=directory.resolve(strict=False),
                suggestion="检查图像序列根目录，或重新定位项目数据源。",
            )
        )
        return ImageSequenceDiscoveryResult(sequences=sequences, report=report)

    for modality_root, modality in _modality_roots(directory):
        for sequence_directory in _sequence_directories(modality_root):
            key = SequenceKey(dataset=dataset, modality=modality, name=sequence_directory.name)
            if key in sequences:
                report.add(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="DUPLICATE_IMAGE_SEQUENCE",
                        message=f"图像序列键重复：{key.identifier}",
                        path=sequence_directory.resolve(strict=False),
                        sequence=key.identifier,
                        modality=modality.value,
                        suggestion="只保留一个同模态同名序列目录。",
                    )
                )
                continue
            frames = _frame_index(sequence_directory, key=key, report=report)
            sequences[key] = ImageSequenceIndex(
                key=key,
                directory=sequence_directory,
                frames=frames,
            )

    if not sequences:
        report.add(
            ValidationIssue(
                severity=Severity.WARNING,
                code="NO_IMAGE_SEQUENCES",
                message=f"目录中未发现数字帧图像序列：{directory}",
                path=directory.resolve(strict=False),
                suggestion="确认图像目录层级和文件扩展名。",
            )
        )
    return ImageSequenceDiscoveryResult(sequences=sequences, report=report)
