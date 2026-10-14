from pathlib import Path

import pytest

from hsi_plotter.io.tracker_discovery import (
    discover_trackers,
    index_tracker_results,
)
from hsi_plotter.models.dataset import Modality, SequenceKey
from hsi_plotter.models.tracker_import import ResultLayout

DATASET = "HOT2023"


def _write(path: Path, rows: str = "1,2,30,40\n2,3,30,40\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows, encoding="utf-8")


def _build_ground_truth(root: Path) -> dict[SequenceKey, Path]:
    files = {
        SequenceKey(DATASET, Modality.VIS, "basketball"): root / "VIS" / "basketball.txt",
        SequenceKey(DATASET, Modality.NIR, "worker"): root / "NIR" / "worker.txt",
    }
    for path in files.values():
        _write(path)
    return files


def test_discovery_detects_prefixed_flat_layout(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write(results_root / "MHT" / "vis-basketball.txt")
    _write(results_root / "MHT" / "nir-worker.txt")
    ground_truth = _build_ground_truth(tmp_path / "gt")

    discoveries, report = discover_trackers(results_root, ground_truth, dataset=DATASET)

    assert not report.has_errors
    assert len(discoveries) == 1
    discovery = discoveries[0]
    assert discovery.tracker_name == "MHT"
    assert discovery.tracker_id == "mht"
    assert not discovery.is_ambiguous
    assert discovery.preferred is not None
    assert discovery.preferred.layout is ResultLayout.PREFIXED_FLAT
    assert discovery.preferred.file_count == 2


def test_discovery_detects_modality_subdirectory_layout(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write(results_root / "SiamHSI" / "VIS" / "basketball.txt")
    _write(results_root / "SiamHSI" / "NIR" / "worker.txt")
    ground_truth = _build_ground_truth(tmp_path / "gt")

    discoveries, _ = discover_trackers(results_root, ground_truth, dataset=DATASET)

    discovery = discoveries[0]
    assert len(discovery.candidates) == 1
    assert discovery.preferred is not None
    assert discovery.preferred.layout is ResultLayout.MODALITY_SUBDIR
    assert set(discovery.preferred.files) == set(ground_truth)


def test_discovery_detects_plain_flat_layout_without_modality(tmp_path: Path) -> None:
    root = tmp_path
    ground_truth = {
        SequenceKey(DATASET, Modality.UNSPECIFIED, "coin"): root / "gt" / "coin.txt",
    }
    _write(root / "gt" / "coin.txt")
    results_root = root / "results"
    _write(results_root / "KCF" / "coin.txt")

    discoveries, _ = discover_trackers(results_root, ground_truth, dataset=DATASET)

    discovery = discoveries[0]
    assert discovery.preferred is not None
    assert discovery.preferred.layout is ResultLayout.FLAT
    assert not discovery.is_ambiguous


def test_discovery_reports_ambiguity_without_silent_choice(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    tracker = results_root / "MHT"
    _write(tracker / "VIS" / "basketball.txt")
    _write(tracker / "vis-basketball.txt")
    ground_truth = _build_ground_truth(tmp_path / "gt")

    discoveries, _ = discover_trackers(results_root, ground_truth, dataset=DATASET)

    discovery = discoveries[0]
    assert discovery.is_ambiguous
    assert {item.layout for item in discovery.best_candidates} == {
        ResultLayout.MODALITY_SUBDIR,
        ResultLayout.PREFIXED_FLAT,
    }


def test_discovery_prefers_candidate_with_better_ground_truth_coverage(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    tracker = results_root / "MHT"
    _write(tracker / "VIS" / "basketball.txt")
    _write(tracker / "VIS" / "worker.txt")
    _write(tracker / "vis-basketball.txt")
    _write(tracker / "nir-worker.txt")
    ground_truth = _build_ground_truth(tmp_path / "gt")

    discoveries, _ = discover_trackers(results_root, ground_truth, dataset=DATASET)

    discovery = discoveries[0]
    assert not discovery.is_ambiguous
    assert discovery.preferred is not None
    assert discovery.preferred.layout is ResultLayout.PREFIXED_FLAT


def test_discovery_flags_duplicate_sequence_files(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    tracker = results_root / "MHT"
    _write(tracker / "VIS" / "basketball.txt")
    _write(tracker / "VIS" / "nested" / "basketball.txt")
    ground_truth = _build_ground_truth(tmp_path / "gt")

    discoveries, _ = discover_trackers(results_root, ground_truth, dataset=DATASET)

    codes = [issue.code for issue in discoveries[0].report.issues]
    assert "DUPLICATE_SEQUENCE_FILE" in codes


def test_discovery_flags_modality_conflict_between_directory_and_prefix(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    tracker = results_root / "MHT"
    _write(tracker / "NIR" / "vis-basketball.txt")
    ground_truth = _build_ground_truth(tmp_path / "gt")

    discoveries, _ = discover_trackers(results_root, ground_truth, dataset=DATASET)

    codes = [issue.code for issue in discoveries[0].report.issues]
    assert "MODALITY_CONFLICT" in codes
    assert discoveries[0].has_error


def test_discovery_reports_missing_results_root(tmp_path: Path) -> None:
    discoveries, report = discover_trackers(tmp_path / "absent", {}, dataset=DATASET)

    assert discoveries == ()
    assert report.has_errors
    assert report.issues[0].code == "RESULTS_DIRECTORY_INVALID"


def test_discovery_reports_empty_results_root(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()

    discoveries, report = discover_trackers(results_root, {}, dataset=DATASET)

    assert discoveries == ()
    assert report.warning_count == 1
    assert report.issues[0].code == "NO_TRACKER_DIRECTORIES"


def test_discovery_reports_tracker_without_usable_files(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    (results_root / "MHT" / "logs").mkdir(parents=True)
    _write(results_root / "MHT" / "logs" / "notes.txt")

    discoveries, _ = discover_trackers(results_root, {}, dataset=DATASET)

    codes = [issue.code for issue in discoveries[0].report.issues]
    assert "TRACKER_NO_RESULT_FILES" in codes
    assert discoveries[0].candidates == ()


def test_index_tracker_results_is_deterministic_for_known_layout(tmp_path: Path) -> None:
    tracker = tmp_path / "MHT"
    _write(tracker / "VIS" / "basketball.txt")
    _write(tracker / "NIR" / "worker.txt")

    first = index_tracker_results(
        tracker, ResultLayout.MODALITY_SUBDIR, dataset=DATASET, tracker="MHT"
    )
    second = index_tracker_results(
        tracker, ResultLayout.MODALITY_SUBDIR, dataset=DATASET, tracker="MHT"
    )

    assert first.files == second.files
    assert first.file_count == 2


def test_discovery_orders_trackers_by_name(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write(results_root / "zz" / "vis-basketball.txt")
    _write(results_root / "Alpha" / "vis-basketball.txt")
    ground_truth = _build_ground_truth(tmp_path / "gt")

    discoveries, _ = discover_trackers(results_root, ground_truth, dataset=DATASET)

    assert [item.tracker_name for item in discoveries] == ["Alpha", "zz"]


def test_discovery_supports_nested_all_prefixed_flat_layout(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write(results_root / "DaSSP-Net" / "all" / "vis-basketball.txt")
    _write(results_root / "DaSSP-Net" / "all" / "nir-worker.txt")
    ground_truth = _build_ground_truth(tmp_path / "gt")

    discoveries, _ = discover_trackers(results_root, ground_truth, dataset=DATASET)

    discovery = discoveries[0]
    assert not discovery.is_ambiguous
    assert discovery.preferred is not None
    assert discovery.preferred.layout is ResultLayout.PREFIXED_FLAT
    assert discovery.preferred.root_path.name == "all"
    assert set(discovery.preferred.files) == set(ground_truth)


def test_discovery_treats_all_and_modality_directories_as_mirrors(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    tracker = results_root / "BAE-Net"
    _write(tracker / "ALL" / "vis-basketball.txt")
    _write(tracker / "ALL" / "nir-worker.txt")
    _write(tracker / "VIS" / "basketball.txt")
    _write(tracker / "NIR" / "worker.txt")
    ground_truth = _build_ground_truth(tmp_path / "gt")

    discoveries, _ = discover_trackers(results_root, ground_truth, dataset=DATASET)

    discovery = discoveries[0]
    assert not discovery.is_ambiguous
    assert discovery.preferred is not None
    assert discovery.preferred.layout is ResultLayout.PREFIXED_FLAT
    assert discovery.preferred.root_path.name == "ALL"


def test_discovery_keeps_direct_prefixed_flat_layout_compatible(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    tracker = results_root / "LRSST"
    _write(tracker / "vis-basketball.txt")
    _write(tracker / "nir-worker.txt")
    ground_truth = _build_ground_truth(tmp_path / "gt")

    discoveries, _ = discover_trackers(results_root, ground_truth, dataset=DATASET)

    preferred = discoveries[0].preferred
    assert preferred is not None
    assert preferred.root_path == tracker.resolve()


@pytest.mark.parametrize("layout", list(ResultLayout))
def test_index_tracker_results_accepts_every_supported_layout(
    layout: ResultLayout, tmp_path: Path
) -> None:
    tracker = tmp_path / "MHT"
    _write(tracker / "vis-basketball.txt")

    candidate = index_tracker_results(tracker, layout, dataset=DATASET, tracker="MHT")

    assert candidate.layout is layout
    assert isinstance(candidate.file_count, int)
