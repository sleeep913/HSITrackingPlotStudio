from pathlib import Path

from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.tracker_import import LayoutCandidate, ResultLayout
from hsi_plotter.validation.issues import Severity
from hsi_plotter.validation.sequence_mapping import (
    build_sequence_mapping,
    validate_sequence_mapping,
)

DATASET = "HOT2023"

_KEY_VIS = SequenceKey(DATASET, Modality.VIS, "basketball")
_KEY_NIR = SequenceKey(DATASET, Modality.NIR, "worker")


def _write(path: Path, rows: str = "1,2,30,40\n2,3,30,40\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows, encoding="utf-8")


def _candidate(
    files: dict[SequenceKey, Path],
    layout: ResultLayout = ResultLayout.PREFIXED_FLAT,
    root: Path | None = None,
) -> LayoutCandidate:
    if root is None:
        root = next(iter(files.values())).parent if files else Path()
    return LayoutCandidate(layout=layout, root_path=root, files=files)


def test_mapping_pairs_sequences_on_full_key(tmp_path: Path) -> None:
    ground_truth = {_KEY_VIS: tmp_path / "gt" / "basketball.txt"}
    results = {_KEY_VIS: tmp_path / "MHT" / "vis-basketball.txt"}
    for path in (*ground_truth.values(), *results.values()):
        _write(path)

    mapping = build_sequence_mapping(ground_truth, _candidate(results), tracker="MHT")

    assert mapping.matched_count == 1
    assert mapping.pairs[_KEY_VIS] == (ground_truth[_KEY_VIS], results[_KEY_VIS])
    assert mapping.missing == ()
    assert mapping.extra == ()
    assert not mapping.report.has_errors


def test_mapping_reports_missing_and_extra_sequences(tmp_path: Path) -> None:
    ground_truth = {_KEY_VIS: tmp_path / "gt" / "basketball.txt"}
    results = {_KEY_NIR: tmp_path / "MHT" / "nir-worker.txt"}
    for path in (*ground_truth.values(), *results.values()):
        _write(path)

    mapping = build_sequence_mapping(ground_truth, _candidate(results), tracker="MHT")

    assert mapping.matched_count == 0
    assert mapping.missing == (_KEY_VIS,)
    assert mapping.extra == (_KEY_NIR,)
    codes = [issue.code for issue in mapping.report.issues]
    assert "MISSING_SEQUENCE" in codes
    assert "EXTRA_SEQUENCE" in codes
    assert "TRACKER_NO_MATCHED_SEQUENCES" in codes


def test_mapping_reports_missing_sequence_as_warning(tmp_path: Path) -> None:
    ground_truth = {_KEY_VIS: tmp_path / "gt" / "basketball.txt"}
    _write(ground_truth[_KEY_VIS])

    mapping = build_sequence_mapping(
        ground_truth, _candidate({}, root=tmp_path / "MHT"), tracker="MHT"
    )

    missing = [issue for issue in mapping.report.issues if issue.code == "MISSING_SEQUENCE"]
    assert len(missing) == 1
    assert missing[0].severity is Severity.WARNING
    assert missing[0].tracker == "MHT"


def test_mapping_uses_name_only_match_when_result_has_no_modality(tmp_path: Path) -> None:
    ground_truth = {_KEY_VIS: tmp_path / "gt" / "basketball.txt"}
    plain = SequenceKey(DATASET, Modality.UNSPECIFIED, "basketball")
    results = {plain: tmp_path / "MHT" / "basketball.txt"}
    for path in (*ground_truth.values(), *results.values()):
        _write(path)

    mapping = build_sequence_mapping(
        ground_truth, _candidate(results, layout=ResultLayout.FLAT), tracker="MHT"
    )

    assert mapping.matched_count == 1
    assert mapping.pairs[_KEY_VIS][1] == results[plain]
    assert mapping.missing == ()
    assert not mapping.report.has_errors


def test_mapping_refuses_silent_choice_for_cross_modality_name_clash(tmp_path: Path) -> None:
    ground_truth = {
        SequenceKey(DATASET, Modality.VIS, "basketball"): tmp_path / "gt" / "vis.txt",
        SequenceKey(DATASET, Modality.NIR, "basketball"): tmp_path / "gt" / "nir.txt",
    }
    plain = SequenceKey(DATASET, Modality.UNSPECIFIED, "basketball")
    results = {plain: tmp_path / "MHT" / "basketball.txt"}
    for path in (*ground_truth.values(), *results.values()):
        _write(path)

    mapping = build_sequence_mapping(
        ground_truth, _candidate(results, layout=ResultLayout.FLAT), tracker="MHT"
    )

    assert mapping.matched_count == 0
    assert mapping.ambiguous_count == 1
    assert mapping.report.has_errors
    assert mapping.report.issues[0].code == "AMBIGUOUS_SEQUENCE_MATCH"


def test_mapping_reports_modality_mismatch_with_dedicated_code(tmp_path: Path) -> None:
    ground_truth = {_KEY_VIS: tmp_path / "gt" / "basketball.txt"}
    results = {SequenceKey(DATASET, Modality.NIR, "basketball"): tmp_path / "MHT" / "nir.txt"}
    for path in (*ground_truth.values(), *results.values()):
        _write(path)

    mapping = build_sequence_mapping(ground_truth, _candidate(results), tracker="MHT")

    codes = [issue.code for issue in mapping.report.issues]
    assert "SEQUENCE_MODALITY_MISMATCH" in codes
    assert mapping.extra_count == 1
    assert mapping.missing_count == 1


def test_mapping_summarises_modality_coverage(tmp_path: Path) -> None:
    ground_truth = {_KEY_VIS: tmp_path / "gt" / "a.txt", _KEY_NIR: tmp_path / "gt" / "b.txt"}
    for path in ground_truth.values():
        _write(path)

    mapping = build_sequence_mapping(
        ground_truth, _candidate(ground_truth, layout=ResultLayout.FLAT), tracker="MHT"
    )

    assert mapping.modality_label == "NIR, VIS"
    assert set(mapping.modalities) == {Modality.VIS, Modality.NIR}


def test_file_level_validation_detects_frame_count_mismatch(tmp_path: Path) -> None:
    ground_truth_path = tmp_path / "gt" / "basketball.txt"
    result_path = tmp_path / "MHT" / "vis-basketball.txt"
    _write(ground_truth_path, "1,2,30,40\n2,3,30,40\n")
    _write(result_path, "1,2,30,40\n")
    mapping = build_sequence_mapping(
        {_KEY_VIS: ground_truth_path}, _candidate({_KEY_VIS: result_path}), tracker="MHT"
    )

    report = validate_sequence_mapping(mapping)

    assert report.has_errors
    assert report.issues[0].code == "FRAME_COUNT_MISMATCH"
    assert report.issues[0].tracker == "MHT"


def test_file_level_validation_accepts_finite_nonpositive_sizes_and_rejects_partial_nan(
    tmp_path: Path,
) -> None:
    ground_truth_path = tmp_path / "gt" / "frame.txt"
    _write(ground_truth_path, "1,2,30,40\n")
    zero_width = tmp_path / "MHT" / "vis-a.txt"
    _write(zero_width, "1,2,0,40\n")
    partial_nan = tmp_path / "MHT" / "vis-b.txt"
    _write(partial_nan, "1,nan,30,40\n")
    zero_width_key = SequenceKey(DATASET, Modality.VIS, "a")
    partial_key = SequenceKey(DATASET, Modality.VIS, "b")
    mapping = build_sequence_mapping(
        {zero_width_key: ground_truth_path, partial_key: ground_truth_path},
        _candidate({zero_width_key: zero_width, partial_key: partial_nan}),
        tracker="MHT",
    )

    report = validate_sequence_mapping(mapping)

    codes = [issue.code for issue in report.issues]
    assert codes.count("INVALID_TRACKER_RESULT") == 1
    assert any("部分缺失值" in issue.message for issue in report.issues)


def test_file_level_validation_reports_full_nan_frames_as_warning(tmp_path: Path) -> None:
    ground_truth_path = tmp_path / "gt" / "basketball.txt"
    result_path = tmp_path / "MHT" / "vis-basketball.txt"
    _write(ground_truth_path, "1,2,30,40\n2,3,30,40\n")
    _write(result_path, "1,2,30,40\nnan,nan,nan,nan\n")
    mapping = build_sequence_mapping(
        {_KEY_VIS: ground_truth_path}, _candidate({_KEY_VIS: result_path}), tracker="MHT"
    )

    report = validate_sequence_mapping(mapping)

    assert not report.has_errors
    assert [issue.code for issue in report.issues] == ["MISSING_RESULT_FRAMES"]


def test_file_level_validation_reports_progress_and_can_be_cancelled(tmp_path: Path) -> None:
    keys = [SequenceKey(DATASET, Modality.VIS, name) for name in ("a", "b", "c")]
    ground_truth_path = tmp_path / "gt" / "frame.txt"
    _write(ground_truth_path)
    results: dict[SequenceKey, Path] = {}
    for key in keys:
        path = tmp_path / "MHT" / f"vis-{key.name}.txt"
        _write(path)
        results[key] = path
    mapping = build_sequence_mapping(
        {key: ground_truth_path for key in keys}, _candidate(results), tracker="MHT"
    )

    seen: list[str] = []
    report = validate_sequence_mapping(
        mapping,
        on_progress=lambda index, message: seen.append(message),
        is_cancelled=lambda: len(seen) >= 2,
    )

    assert seen == ["HOT2023:vis:a", "HOT2023:vis:b"]
    assert report.info_count == 1
    assert report.issues[-1].code == "PREFLIGHT_CANCELLED"
