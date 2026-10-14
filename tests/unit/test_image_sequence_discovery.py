from pathlib import Path

from hsi_plotter.io.image_sequence_discovery import discover_image_sequences
from hsi_plotter.models.dataset import Modality, SequenceKey


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def test_discovers_three_false_color_modalities_and_numeric_frames(tmp_path: Path) -> None:
    _touch(tmp_path / "HSI-VIS-FalseColor" / "ball" / "0001.jpg")
    _touch(tmp_path / "HSI-VIS-FalseColor" / "ball" / "img" / "0003.PNG")
    _touch(tmp_path / "HSI-NIR-Falsecolor" / "car" / "0001.jpeg")
    _touch(tmp_path / "HSI-RedNIR-FalseColor" / "coin" / "0002.tiff")
    _touch(tmp_path / "HSI-VIS-FalseColor" / "ball" / "preview.jpg")

    result = discover_image_sequences(tmp_path, dataset="HOT")

    assert not result.report.has_errors
    vis = result.sequences[SequenceKey("HOT", Modality.VIS, "ball")]
    assert vis.frame_numbers == (1, 3)
    assert vis.frames[1].name == "0001.jpg"
    assert vis.frames[3].name == "0003.PNG"
    assert SequenceKey("HOT", Modality.NIR, "car") in result.sequences
    assert SequenceKey("HOT", Modality.REDNIR, "coin") in result.sequences


def test_discovers_legacy_single_modality_layout_recursively(tmp_path: Path) -> None:
    _touch(tmp_path / "basketball" / "img" / "0001.jpg")
    _touch(tmp_path / "basketball" / "img" / "0010.bmp")

    result = discover_image_sequences(tmp_path, dataset="HOT")

    assert not result.report.has_errors
    sequence = result.sequences[
        SequenceKey("HOT", Modality.UNSPECIFIED, "basketball")
    ]
    assert sequence.frame_numbers == (1, 10)


def test_accepts_a_false_color_modality_directory_as_root(tmp_path: Path) -> None:
    root = tmp_path / "HSI-NIR-FalseColor"
    _touch(root / "worker" / "0001.jpg")

    result = discover_image_sequences(root, dataset="HOT")

    assert set(result.sequences) == {SequenceKey("HOT", Modality.NIR, "worker")}


def test_duplicate_numeric_frame_is_reported_and_not_selected(tmp_path: Path) -> None:
    _touch(tmp_path / "HSI-VIS-FalseColor" / "ball" / "0001.jpg")
    _touch(tmp_path / "HSI-VIS-FalseColor" / "ball" / "nested" / "1.png")

    result = discover_image_sequences(tmp_path, dataset="HOT")

    assert result.report.has_errors
    assert any(issue.code == "DUPLICATE_IMAGE_FRAME" for issue in result.report.issues)
    sequence = result.sequences[SequenceKey("HOT", Modality.VIS, "ball")]
    assert sequence.frame_numbers == ()


def test_missing_image_directory_is_reported(tmp_path: Path) -> None:
    result = discover_image_sequences(tmp_path / "missing", dataset="HOT")

    assert result.sequences == {}
    assert result.report.has_errors
    assert result.report.issues[0].code == "DIRECTORY_NOT_FOUND"

