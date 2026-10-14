from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from hsi_plotter.io.csv_bundle import CsvBundleError, load_csv_bundle
from hsi_plotter.models.dataset import (
    MODALITY_DIRECTORY_NAMES,
    Modality,
    SequenceKey,
    parse_sequence_identity,
)
from hsi_plotter.models.tracker_import import (
    LayoutCandidate,
    ResultLayout,
    TrackerDiscovery,
)
from hsi_plotter.validation.issues import PreflightReport, Severity, ValidationIssue

#: Layouts are evaluated in this order so that equally scored candidates are
#: presented to the user in a stable sequence.
LAYOUT_EVALUATION_ORDER = (
    ResultLayout.PREFIXED_FLAT,
    ResultLayout.MODALITY_SUBDIR,
    ResultLayout.FLAT,
)


def _sorted_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=lambda item: item.as_posix().casefold())


def _index_result_files(
    files: Iterable[Path],
    *,
    dataset: str,
    directory_modality: Modality,
    tracker: str,
    layout: ResultLayout,
) -> tuple[dict[SequenceKey, Path], list[ValidationIssue]]:
    indexed: dict[SequenceKey, Path] = {}
    issues: list[ValidationIssue] = []
    for path in _sorted_paths(files):
        file_modality, sequence_name = parse_sequence_identity(path.stem)
        modality = directory_modality
        if file_modality is not Modality.UNSPECIFIED:
            if (
                directory_modality is not Modality.UNSPECIFIED
                and file_modality is not directory_modality
            ):
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="MODALITY_CONFLICT",
                        message=(
                            f"文件名前缀模态 {file_modality.value} 与目录模态 "
                            f"{directory_modality.value} 冲突：{path}"
                        ),
                        path=path.resolve(strict=False),
                        tracker=tracker,
                        sequence=sequence_name,
                        modality=file_modality.value,
                        suggestion="统一目录与文件名前缀，或在导入映射中手动修正。",
                    )
                )
            modality = file_modality
        key = SequenceKey(dataset=dataset, modality=modality, name=sequence_name)
        resolved = path.resolve(strict=False)
        if previous := indexed.get(key):
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="DUPLICATE_SEQUENCE_FILE",
                    message=(
                        f"序列 {key.identifier} 同时对应 {previous} 和 {resolved}，"
                        "无法确定唯一结果文件。"
                    ),
                    path=resolved,
                    tracker=tracker,
                    sequence=key.identifier,
                    modality=modality.value,
                    suggestion="删除重复文件，或在导入映射中手动指定唯一文件。",
                )
            )
            continue
        indexed[key] = resolved
    return indexed, issues


def _direct_text_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return _sorted_paths(path for path in directory.glob("*.txt") if path.is_file())


def _aggregate_prefixed_files(directory: Path) -> tuple[Path, list[Path]] | None:
    """Find HOT-style prefixed results stored below ``all``/``ALL``."""
    aggregate_directories = _sorted_paths(
        child
        for child in directory.iterdir()
        if child.is_dir() and child.name.casefold() == "all"
    )
    for child in aggregate_directories:
        files = [
            path
            for path in _sorted_paths(child.rglob("*.txt"))
            if path.is_file()
            and parse_sequence_identity(path.stem)[0] is not Modality.UNSPECIFIED
        ]
        if files:
            return child, files
    return None


def _modality_subdirectories(directory: Path) -> list[tuple[Path, Modality]]:
    if not directory.is_dir():
        return []
    found: list[tuple[Path, Modality]] = []
    for child in directory.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        if modality := MODALITY_DIRECTORY_NAMES.get(child.name.casefold()):
            found.append((child, modality))
    return sorted(found, key=lambda item: item[0].name.casefold())


def _build_modality_subdir_candidate(
    directory: Path,
    *,
    dataset: str,
    tracker: str,
) -> tuple[LayoutCandidate, list[ValidationIssue]] | None:
    modality_directories = _modality_subdirectories(directory)
    if not modality_directories:
        return None
    indexed: dict[SequenceKey, Path] = {}
    issues: list[ValidationIssue] = []
    for child, modality in modality_directories:
        child_files = _sorted_paths(child.rglob("*.txt"))
        child_indexed, child_issues = _index_result_files(
            child_files,
            dataset=dataset,
            directory_modality=modality,
            tracker=tracker,
            layout=ResultLayout.MODALITY_SUBDIR,
        )
        issues.extend(child_issues)
        for key, path in child_indexed.items():
            if previous := indexed.get(key):
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="DUPLICATE_SEQUENCE_FILE",
                        message=(
                            f"序列 {key.identifier} 同时对应 {previous} 和 {path}，"
                            "无法确定唯一结果文件。"
                        ),
                        path=path,
                        tracker=tracker,
                        sequence=key.identifier,
                        modality=key.modality.value,
                        suggestion="删除重复文件，或在导入映射中手动指定唯一文件。",
                    )
                )
                continue
            indexed[key] = path
    if not indexed:
        return None
    return (
        LayoutCandidate(
            layout=ResultLayout.MODALITY_SUBDIR,
            root_path=directory.resolve(strict=False),
            files=indexed,
        ),
        issues,
    )


def build_layout_candidate(
    tracker_root: Path,
    layout: ResultLayout,
    *,
    dataset: str,
    tracker: str,
) -> tuple[LayoutCandidate, list[ValidationIssue]] | None:
    """Build one candidate interpretation of a tracker directory.

    Returns ``None`` when the directory does not contain usable files for the
    requested layout.
    """
    directory = Path(tracker_root)
    if directory.is_file() and directory.suffix.casefold() == ".csv":
        if layout is not ResultLayout.PREFIXED_FLAT:
            return None
        try:
            bundle = load_csv_bundle(directory, dataset=dataset)
        except CsvBundleError:
            return None
        return (
            LayoutCandidate(
                layout=ResultLayout.PREFIXED_FLAT,
                root_path=directory.resolve(strict=False),
                files=bundle.files,
            ),
            [],
        )
    if layout is ResultLayout.MODALITY_SUBDIR:
        return _build_modality_subdir_candidate(directory, dataset=dataset, tracker=tracker)

    if layout is ResultLayout.PREFIXED_FLAT:
        aggregate = _aggregate_prefixed_files(directory)
        if aggregate is not None:
            source_directory, selected = aggregate
        else:
            source_directory = directory
            selected = [
                path
                for path in _direct_text_files(directory)
                if parse_sequence_identity(path.stem)[0] is not Modality.UNSPECIFIED
            ]
        if not selected:
            return None
    else:
        source_directory = directory
        selected = _direct_text_files(directory)
        if not selected:
            return None
    indexed, issues = _index_result_files(
        selected,
        dataset=dataset,
        directory_modality=Modality.UNSPECIFIED,
        tracker=tracker,
        layout=layout,
    )
    if not indexed:
        return None
    return (
        LayoutCandidate(
            layout=layout,
            root_path=source_directory.resolve(strict=False),
            files=indexed,
        ),
        issues,
    )


def index_tracker_results(
    tracker_root: Path,
    layout: ResultLayout,
    *,
    dataset: str,
    tracker: str,
) -> LayoutCandidate:
    """Re-index a tracker directory with an explicitly selected layout."""
    candidate = build_layout_candidate(tracker_root, layout, dataset=dataset, tracker=tracker)
    if candidate is None:
        return LayoutCandidate(
            layout=layout,
            root_path=Path(tracker_root).resolve(strict=False),
            files={},
        )
    return candidate[0]


def _identity(key: SequenceKey) -> tuple[Modality, str]:
    return (key.modality, key.name)


def _candidate_score(
    candidate: LayoutCandidate,
    ground_truth_keys: frozenset[tuple[Modality, str]],
) -> tuple[int, int, int]:
    """Higher is better: matched sequences first, then fewer stray files."""
    identities = {_identity(key) for key in candidate.files}
    matched = len(identities & ground_truth_keys)
    extra = len(identities - ground_truth_keys)
    return (matched, -extra, -candidate.file_count)


def _drop_identical_candidates(
    candidates: list[tuple[LayoutCandidate, list[ValidationIssue]]],
) -> list[tuple[LayoutCandidate, list[ValidationIssue]]]:
    unique: list[tuple[LayoutCandidate, list[ValidationIssue]]] = []
    seen: list[frozenset[tuple[SequenceKey, Path]]] = []
    for candidate, issues in candidates:
        signature = frozenset(candidate.files.items())
        if signature in seen:
            continue
        seen.append(signature)
        unique.append((candidate, issues))
    return unique


def _drop_aggregate_mirror_candidates(
    candidates: list[tuple[LayoutCandidate, list[ValidationIssue]]],
) -> list[tuple[LayoutCandidate, list[ValidationIssue]]]:
    """Collapse HOT's ``ALL`` and modality-directory mirror representations.

    This is deliberately narrower than general tie handling: layouts remain
    ambiguous when the prefixed files live directly in the tracker directory.
    """
    aggregate_key_sets = {
        frozenset(candidate.files)
        for candidate, _ in candidates
        if candidate.layout is ResultLayout.PREFIXED_FLAT
        and candidate.root_path.name.casefold() == "all"
    }
    if not aggregate_key_sets:
        return candidates
    return [
        item
        for item in candidates
        if item[0].layout is ResultLayout.PREFIXED_FLAT
        or frozenset(item[0].files) not in aggregate_key_sets
    ]


def discover_trackers(
    results_root: Path,
    ground_truth_files: Mapping[SequenceKey, Path],
    *,
    dataset: str,
) -> tuple[tuple[TrackerDiscovery, ...], PreflightReport]:
    """Scan ``results_root`` for tracker directories and rank usable layouts."""
    root = Path(results_root)
    report = PreflightReport()
    if root.is_file() and root.suffix.casefold() == ".csv":
        try:
            bundle = load_csv_bundle(root, dataset=dataset)
        except CsvBundleError as exc:
            report.add(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="INVALID_RESULT_CSV",
                    message=str(exc),
                    path=root.resolve(strict=False),
                    suggestion="修正 CSV 表头、ID、帧号或边界框数据后重新导入。",
                )
            )
            return (), report
        candidate = LayoutCandidate(
            layout=ResultLayout.PREFIXED_FLAT,
            root_path=root.resolve(strict=False),
            files=bundle.files,
        )
        generic_names = {"submission", "result", "results", "prediction", "predictions"}
        tracker_name = root.parent.name if root.stem.casefold() in generic_names else root.stem
        discovery = TrackerDiscovery(
            tracker_id=tracker_name.casefold(),
            tracker_name=tracker_name,
            root_path=root.resolve(strict=False),
            candidates=(candidate,),
            best_candidates=(candidate,),
        )
        return (discovery,), report
    if not root.is_dir():
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code="RESULTS_DIRECTORY_INVALID",
                message=f"跟踪结果 TXT 目录或 CSV 文件不存在：{root}",
                path=root.resolve(strict=False),
                suggestion="重新选择有效的 TXT 结果目录或 CSV 文件。",
            )
        )
        return (), report

    ground_truth_keys = frozenset(_identity(key) for key in ground_truth_files)
    tracker_directories = _sorted_paths(
        child for child in root.iterdir() if child.is_dir() and not child.name.startswith(".")
    )
    if not tracker_directories:
        report.add(
            ValidationIssue(
                severity=Severity.WARNING,
                code="NO_TRACKER_DIRECTORIES",
                message=f"跟踪结果目录下未发现跟踪器子目录：{root}",
                path=root.resolve(strict=False),
                suggestion="确认结果目录层级为 结果根目录 / 跟踪器 / 序列文件。",
            )
        )
        return (), report

    discoveries: list[TrackerDiscovery] = []
    for tracker_directory in tracker_directories:
        tracker_name = tracker_directory.name
        discovery_report = PreflightReport()
        built: list[tuple[LayoutCandidate, list[ValidationIssue]]] = []
        for layout in LAYOUT_EVALUATION_ORDER:
            candidate = build_layout_candidate(
                tracker_directory,
                layout,
                dataset=dataset,
                tracker=tracker_name,
            )
            if candidate is not None:
                built.append(candidate)
        built = _drop_identical_candidates(built)
        built = _drop_aggregate_mirror_candidates(built)
        for _, issues in built:
            discovery_report.extend(issues)

        candidates = tuple(candidate for candidate, _ in built)
        if candidates:
            scored = [(_candidate_score(item, ground_truth_keys), item) for item in candidates]
            best_score = max(score for score, _ in scored)
            best_candidates = tuple(item for score, item in scored if score == best_score)
        else:
            best_candidates = ()
            discovery_report.add(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="TRACKER_NO_RESULT_FILES",
                    message=f"跟踪器目录中未发现可用于三种布局的 TXT 结果文件：{tracker_directory}",
                    path=tracker_directory.resolve(strict=False),
                    tracker=tracker_name,
                    suggestion="确认该跟踪器的结果文件位置与扩展名。",
                )
            )

        discoveries.append(
            TrackerDiscovery(
                tracker_id=tracker_name.casefold(),
                tracker_name=tracker_name,
                root_path=tracker_directory.resolve(strict=False),
                candidates=candidates,
                best_candidates=best_candidates,
                report=discovery_report,
            )
        )

    ordered = tuple(sorted(discoveries, key=lambda item: item.tracker_name.casefold()))
    return ordered, report
