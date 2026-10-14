import csv
import json
from pathlib import Path

from hsi_plotter.io.preflight_exporter import export_report_csv, export_report_json
from hsi_plotter.validation.issues import PreflightReport, Severity, ValidationIssue


def _report() -> PreflightReport:
    report = PreflightReport()
    report.add(
        ValidationIssue(
            severity=Severity.ERROR,
            code="FRAME_COUNT_MISMATCH",
            message="Ground Truth 为 2 帧，跟踪结果为 1 帧。",
            path=Path("D:/data/MHT/vis-basketball.txt"),
            tracker="MHT",
            sequence="HOT2023:vis:basketball",
            modality="vis",
            suggestion="确保两个文件逐帧对应且行数相同。",
        )
    )
    report.add(
        ValidationIssue(
            severity=Severity.INFO,
            code="SOURCE_SUMMARY",
            message="已发现 2 个 Ground Truth 序列文件。",
        )
    )
    return report


def test_csv_export_is_utf8_and_readable(tmp_path: Path) -> None:
    destination = tmp_path / "reports" / "preflight.csv"

    export_report_csv(_report(), destination)

    raw = destination.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    with destination.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["级别"] == "错误"
    assert rows[0]["问题代码"] == "FRAME_COUNT_MISMATCH"
    assert rows[0]["序列"] == "HOT2023:vis:basketball"
    assert rows[1]["级别"] == "信息"
    assert rows[1]["文件路径"] == ""


def test_json_export_keeps_machine_readable_codes_and_counts(tmp_path: Path) -> None:
    destination = tmp_path / "preflight.json"

    export_report_json(_report(), destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["summary"] == {"error_count": 1, "warning_count": 0, "info_count": 1}
    assert payload["issues"][0]["severity"] == "error"
    assert payload["issues"][0]["code"] == "FRAME_COUNT_MISMATCH"
    assert payload["issues"][0]["path"] == str(Path("D:/data/MHT/vis-basketball.txt"))
    assert payload["issues"][1]["path"] is None


def test_exports_round_trip_chinese_text_without_escaping(tmp_path: Path) -> None:
    destination = tmp_path / "preflight.json"

    export_report_json(_report(), destination)
    text = destination.read_text(encoding="utf-8")

    assert "Ground Truth" in text
    assert "\\u" not in text
