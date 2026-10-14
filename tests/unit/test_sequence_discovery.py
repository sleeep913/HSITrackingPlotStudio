from pathlib import Path

from hsi_plotter.io.sequence_discovery import discover_sequence_files
from hsi_plotter.models.dataset import Modality, SequenceKey


def test_discovery_supports_prefixed_and_modality_directory_layouts(
    tmp_path: Path,
) -> None:
    (tmp_path / "VIS").mkdir()
    (tmp_path / "NIR").mkdir()
    (tmp_path / "VIS" / "basketball.txt").write_text("1,2,3,4\n", encoding="utf-8")
    (tmp_path / "NIR" / "worker.txt").write_text("1,2,3,4\n", encoding="utf-8")
    (tmp_path / "rednir-coin.txt").write_text("1,2,3,4\n", encoding="utf-8")

    result = discover_sequence_files(tmp_path, dataset="HOT2023")

    assert not result.report.has_errors
    assert result.files[SequenceKey("HOT2023", Modality.VIS, "basketball")].name == (
        "basketball.txt"
    )
    assert result.files[SequenceKey("HOT2023", Modality.NIR, "worker")].name == "worker.txt"
    assert result.files[SequenceKey("HOT2023", Modality.REDNIR, "coin")].name == "rednir-coin.txt"


def test_discovery_reports_duplicate_sequence_key_as_ambiguity(tmp_path: Path) -> None:
    (tmp_path / "VIS").mkdir()
    (tmp_path / "VIS" / "basketball.txt").write_text("1,2,3,4\n", encoding="utf-8")
    (tmp_path / "vis-basketball.txt").write_text("1,2,3,4\n", encoding="utf-8")

    result = discover_sequence_files(tmp_path, dataset="HOT2023")

    assert result.report.has_errors
    assert result.report.issues[0].code == "DUPLICATE_SEQUENCE_KEY"
    assert "basketball" in result.report.issues[0].message


def test_discovery_reports_missing_directory(tmp_path: Path) -> None:
    result = discover_sequence_files(tmp_path / "missing", dataset="HOT2023")

    assert result.files == {}
    assert result.report.has_errors
    assert result.report.issues[0].code == "DIRECTORY_NOT_FOUND"


def test_discovery_prefers_all_directory_over_duplicate_modality_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "ALL").mkdir()
    (tmp_path / "VIS").mkdir()
    (tmp_path / "NIR").mkdir()
    (tmp_path / "ALL" / "vis-ball.txt").write_text("1,2,3,4\n", encoding="utf-8")
    (tmp_path / "ALL" / "nir-car.txt").write_text("1,2,3,4\n", encoding="utf-8")
    (tmp_path / "VIS" / "ball.txt").write_text("1,2,3,4\n", encoding="utf-8")
    (tmp_path / "NIR" / "car.txt").write_text("1,2,3,4\n", encoding="utf-8")

    result = discover_sequence_files(tmp_path, dataset="HOT2023")

    assert not result.report.has_errors
    assert set(result.files) == {
        SequenceKey("HOT2023", Modality.VIS, "ball"),
        SequenceKey("HOT2023", Modality.NIR, "car"),
    }
    assert {path.parent.name for path in result.files.values()} == {"ALL"}


def test_discovery_accepts_all_directory_as_selected_root(tmp_path: Path) -> None:
    aggregate = tmp_path / "ALL"
    aggregate.mkdir()
    (aggregate / "rednir-coin.txt").write_text("1,2,3,4\n", encoding="utf-8")

    result = discover_sequence_files(aggregate, dataset="HOT2023")

    assert not result.report.has_errors
    key = SequenceKey("HOT2023", Modality.REDNIR, "coin")
    assert result.files[key] == (aggregate / "rednir-coin.txt").resolve()
