from pathlib import Path

from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.validation.preflight import validate_tracking_pair


def test_tracking_pair_preflight_accepts_matching_valid_files(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    result = tmp_path / "result.txt"
    gt.write_text("1,2,30,40\n2,3,30,40\n", encoding="utf-8")
    result.write_text("1,2,30,40\nnan,nan,nan,nan\n", encoding="utf-8")

    report = validate_tracking_pair(
        gt,
        result,
        tracker="MHT",
        sequence=SequenceKey("HOT2023", Modality.VIS, "basketball"),
    )

    assert not report.has_errors
    assert report.warning_count == 1
    assert report.issues[0].code == "MISSING_RESULT_FRAMES"


def test_tracking_pair_preflight_blocks_frame_count_mismatch(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    result = tmp_path / "result.txt"
    gt.write_text("1,2,30,40\n2,3,30,40\n", encoding="utf-8")
    result.write_text("1,2,30,40\n", encoding="utf-8")

    report = validate_tracking_pair(
        gt,
        result,
        tracker="MHT",
        sequence=SequenceKey("HOT2023", Modality.NIR, "worker"),
    )

    assert report.has_errors
    issue = report.issues[0]
    assert issue.code == "FRAME_COUNT_MISMATCH"
    assert issue.tracker == "MHT"
    assert issue.sequence == "HOT2023:nir:worker"
    assert "2" in issue.message and "1" in issue.message


def test_tracking_pair_preflight_reports_invalid_source_file(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    result = tmp_path / "result.txt"
    gt.write_text("1,2,nan,40\n", encoding="utf-8")
    result.write_text("1,2,30,40\n", encoding="utf-8")

    report = validate_tracking_pair(
        gt,
        result,
        tracker="MHT",
        sequence=SequenceKey("HOT2023", Modality.UNSPECIFIED, "ball"),
    )

    assert report.has_errors
    assert report.issues[0].code == "INVALID_GROUND_TRUTH"
    assert report.issues[0].path == gt.resolve()


def test_tracking_pair_preflight_accepts_minus_one_offscreen_marker(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    result = tmp_path / "result.txt"
    gt.write_text("1,2,30,40\n", encoding="utf-8")
    result.write_text("-1,-1,-1,-1\n", encoding="utf-8")

    report = validate_tracking_pair(
        gt,
        result,
        tracker="MHT",
        sequence=SequenceKey("HOT2023", Modality.VIS, "ball"),
    )

    assert not report.has_errors
    assert not report.issues
