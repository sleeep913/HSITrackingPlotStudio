from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hsi_plotter.io.csv_bundle import CsvBundleError, load_csv_bundle
from hsi_plotter.models.dataset import (
    MODALITY_DIRECTORY_NAMES,
    Modality,
    SequenceKey,
    parse_sequence_identity,
)
from hsi_plotter.validation.issues import (
    PreflightReport,
    Severity,
    ValidationIssue,
)


@dataclass(slots=True)
class SequenceDiscoveryResult:
    files: dict[SequenceKey, Path]
    report: PreflightReport


def _aggregate_directory(directory: Path) -> Path | None:
    """Return a HOT-style ``all`` directory when it is self-describing.

    HOT2023 distributes the same sequences twice: prefixed files in ``ALL``
    and unprefixed files below the modality directories.  Reading both trees
    recursively creates false duplicate keys, so the aggregate representation
    is canonical when it is present and contains modality-prefixed TXT files.
    """
    aggregate_directories = sorted(
        (
            child
            for child in directory.iterdir()
            if child.is_dir() and child.name.casefold() == "all"
        ),
        key=lambda item: item.name.casefold(),
    )
    for child in aggregate_directories:
        if any(
            parse_sequence_identity(path.stem)[0] is not Modality.UNSPECIFIED
            for path in child.rglob("*.txt")
            if path.is_file()
        ):
            return child
    return None


def _directory_modality(root: Path, path: Path) -> Modality:
    relative_parts = path.relative_to(root).parts[:-1]
    for part in reversed(relative_parts):
        if modality := MODALITY_DIRECTORY_NAMES.get(part.casefold()):
            return modality
    return Modality.UNSPECIFIED


def discover_sequence_files(root: Path, *, dataset: str) -> SequenceDiscoveryResult:
    directory = Path(root)
    report = PreflightReport()
    files: dict[SequenceKey, Path] = {}
    if directory.is_file() and directory.suffix.casefold() == ".csv":
        try:
            bundle = load_csv_bundle(directory, dataset=dataset)
        except CsvBundleError as exc:
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="INVALID_SEQUENCE_CSV",
                    message=str(exc),
                    path=directory.resolve(strict=False),
                    suggestion="修正 CSV 表头、序列标识、帧号或边界框数据后重新导入。",
                )
            )
            return SequenceDiscoveryResult(files=files, report=report)
        return SequenceDiscoveryResult(files=bundle.files, report=report)
    if not directory.is_dir():
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code="DIRECTORY_NOT_FOUND",
                message=f"TXT 目录或 CSV 文件不存在或不可访问：{directory}",
                path=directory.resolve(strict=False),
                suggestion="检查路径，或重新选择 TXT 目录/CSV 文件。",
            )
        )
        return SequenceDiscoveryResult(files=files, report=report)

    scan_root = _aggregate_directory(directory) or directory
    for path in sorted(scan_root.rglob("*.txt"), key=lambda item: item.as_posix().casefold()):
        file_modality, sequence_name = parse_sequence_identity(path.stem)
        directory_modality = _directory_modality(scan_root, path)
        if file_modality is not Modality.UNSPECIFIED:
            modality = file_modality
            if (
                directory_modality is not Modality.UNSPECIFIED
                and directory_modality is not file_modality
            ):
                report.add(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="MODALITY_CONFLICT",
                        message=(
                            f"文件名前缀模态 {file_modality.value} 与目录模态 "
                            f"{directory_modality.value} 冲突：{path}"
                        ),
                        path=path.resolve(strict=False),
                        sequence=sequence_name,
                        suggestion="统一目录与文件名前缀，或在导入映射中手动修正。",
                    )
                )
        else:
            modality = directory_modality

        key = SequenceKey(dataset=dataset, modality=modality, name=sequence_name)
        resolved = path.resolve(strict=False)
        if previous := files.get(key):
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="DUPLICATE_SEQUENCE_KEY",
                    message=(
                        f"序列键 {key.identifier} 同时对应 {previous} 和 {resolved}，无法自动选择。"
                    ),
                    path=resolved,
                    sequence=key.identifier,
                    modality=key.modality.value,
                    suggestion="移除重复文件或在导入映射中指定唯一文件。",
                )
            )
            continue
        files[key] = resolved

    if not files:
        report.add(
            ValidationIssue(
                severity=Severity.WARNING,
                code="NO_SEQUENCE_FILES",
                message=f"目录中未发现 TXT 序列文件：{directory}",
                path=directory.resolve(strict=False),
                suggestion="确认目录层级和文件扩展名。",
            )
        )
    return SequenceDiscoveryResult(files=files, report=report)
