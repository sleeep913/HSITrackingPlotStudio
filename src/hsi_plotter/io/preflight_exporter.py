from __future__ import annotations

import csv
import json
from pathlib import Path

from hsi_plotter.validation.issues import PreflightReport, Severity, ValidationIssue

CSV_COLUMNS: tuple[str, ...] = (
    "severity",
    "code",
    "message",
    "tracker",
    "sequence",
    "modality",
    "path",
    "suggestion",
)

_CSV_HEADERS = {
    "severity": "级别",
    "code": "问题代码",
    "message": "问题描述",
    "tracker": "跟踪器",
    "sequence": "序列",
    "modality": "模态",
    "path": "文件路径",
    "suggestion": "修复建议",
}

_SEVERITY_NAMES = {
    Severity.ERROR: "错误",
    Severity.WARNING: "警告",
    Severity.INFO: "信息",
}


def _row_values(issue: ValidationIssue) -> dict[str, str]:
    return {
        "severity": _SEVERITY_NAMES.get(issue.severity, str(issue.severity)),
        "code": issue.code,
        "message": issue.message,
        "tracker": issue.tracker or "",
        "sequence": issue.sequence or "",
        "modality": issue.modality or "",
        "path": str(issue.path) if issue.path else "",
        "suggestion": issue.suggestion or "",
    }


def export_report_csv(report: PreflightReport, destination: Path) -> Path:
    """Write the preflight report as a UTF-8 CSV with Chinese column headers."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig keeps Excel on Windows from mis-decoding the Chinese headers.
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(CSV_COLUMNS),
            extrasaction="ignore",
        )
        writer.writerow(_CSV_HEADERS)
        for issue in report.issues:
            writer.writerow(_row_values(issue))
    return path


def export_report_json(report: PreflightReport, destination: Path) -> Path:
    """Write the preflight report as UTF-8 JSON with counts and machine readable codes."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "error_count": report.error_count,
            "warning_count": report.warning_count,
            "info_count": report.info_count,
        },
        "issues": [
            {
                "severity": issue.severity.value,
                "code": issue.code,
                "message": issue.message,
                "tracker": issue.tracker,
                "sequence": issue.sequence,
                "modality": issue.modality,
                "path": str(issue.path) if issue.path else None,
                "suggestion": issue.suggestion,
            }
            for issue in report.issues
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
