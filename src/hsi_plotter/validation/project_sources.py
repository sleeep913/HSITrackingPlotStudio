from __future__ import annotations

from pathlib import Path

from hsi_plotter.io.csv_bundle import CsvBundleError, load_csv_bundle
from hsi_plotter.io.sequence_discovery import discover_sequence_files
from hsi_plotter.models.project import ProjectSources
from hsi_plotter.validation.issues import PreflightReport, Severity, ValidationIssue


def _check_directory(
    report: PreflightReport,
    path: Path | None,
    *,
    label: str,
    required: bool,
    missing_code: str,
    invalid_code: str,
) -> bool:
    if path is None:
        report.add(
            ValidationIssue(
                severity=Severity.ERROR if required else Severity.INFO,
                code=missing_code,
                message=f"{label}尚未配置。",
                suggestion=f"选择{label}后重新运行预检。",
            )
        )
        return False
    if not path.is_dir():
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code=invalid_code,
                message=f"{label}不存在或不是目录：{path}",
                path=path.resolve(strict=False),
                suggestion=f"重新选择有效的{label}。",
            )
        )
        return False
    return True


def _check_tracking_source(
    report: PreflightReport,
    path: Path | None,
    *,
    label: str,
    missing_code: str,
    invalid_code: str,
) -> bool:
    if path is None:
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code=missing_code,
                message=f"{label}尚未配置。",
                suggestion=f"选择{label}后重新运行预检。",
            )
        )
        return False
    if path.is_dir() or (path.is_file() and path.suffix.casefold() == ".csv"):
        return True
    report.add(
        ValidationIssue(
            severity=Severity.ERROR,
            code=invalid_code,
            message=f"{label}必须是 TXT 目录或 CSV 文件：{path}",
            path=path.resolve(strict=False),
            suggestion=f"重新选择有效的{label}。",
        )
    )
    return False


def validate_project_sources(sources: ProjectSources) -> PreflightReport:
    """Validate project-level source locations without mutating the project."""
    report = PreflightReport()
    ground_truth_valid = _check_tracking_source(
        report,
        sources.ground_truth,
        label="Ground Truth",
        missing_code="GROUND_TRUTH_NOT_CONFIGURED",
        invalid_code="GROUND_TRUTH_DIRECTORY_INVALID",
    )
    results_valid = _check_tracking_source(
        report,
        sources.results_root,
        label="跟踪结果",
        missing_code="RESULTS_ROOT_NOT_CONFIGURED",
        invalid_code="RESULTS_DIRECTORY_INVALID",
    )
    _check_directory(
        report,
        sources.image_root,
        label="图像序列目录",
        required=False,
        missing_code="IMAGE_ROOT_NOT_CONFIGURED",
        invalid_code="IMAGE_DIRECTORY_INVALID",
    )
    _check_directory(
        report,
        sources.export_directory,
        label="默认导出目录",
        required=False,
        missing_code="EXPORT_DIRECTORY_NOT_CONFIGURED",
        invalid_code="EXPORT_DIRECTORY_INVALID",
    )

    if sources.attribute_file is None:
        report.add(
            ValidationIssue(
                severity=Severity.INFO,
                code="ATTRIBUTE_FILE_NOT_CONFIGURED",
                message="属性字典尚未配置。",
                suggestion="仅在需要属性评估时选择属性文件。",
            )
        )
    elif not sources.attribute_file.is_file():
        report.add(
            ValidationIssue(
                severity=Severity.ERROR,
                code="ATTRIBUTE_FILE_INVALID",
                message=f"属性字典不存在或不是文件：{sources.attribute_file}",
                path=sources.attribute_file.resolve(strict=False),
                suggestion="重新选择有效的属性字典文件。",
            )
        )

    ground_truth_count = 0
    if ground_truth_valid and sources.ground_truth is not None:
        discovery = discover_sequence_files(sources.ground_truth, dataset="project")
        ground_truth_count = len(discovery.files)
        report.extend(discovery.report.issues)

    result_count = 0
    if results_valid and sources.results_root is not None:
        if sources.results_root.is_file():
            try:
                result_count = load_csv_bundle(
                    sources.results_root, dataset="project"
                ).sequence_count
            except CsvBundleError as exc:
                report.add(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="INVALID_RESULT_CSV",
                        message=str(exc),
                        path=sources.results_root.resolve(strict=False),
                        suggestion="修正 CSV 后重新运行预检。",
                    )
                )
        else:
            result_count = sum(1 for _ in sources.results_root.rglob("*.txt"))
        if result_count == 0:
            report.add(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="NO_RESULT_FILES",
                    message=f"跟踪结果中未发现可用序列：{sources.results_root}",
                    path=sources.results_root.resolve(strict=False),
                    suggestion="确认 TXT 目录层级或 CSV 文件内容。",
                )
            )

    if ground_truth_valid and results_valid:
        report.add(
            ValidationIssue(
                severity=Severity.INFO,
                code="SOURCE_SUMMARY",
                message=(
                    f"已发现 {ground_truth_count} 个 Ground Truth 序列，"
                    f"以及 {result_count} 个跟踪结果序列。"
                ),
            )
        )
    return report
