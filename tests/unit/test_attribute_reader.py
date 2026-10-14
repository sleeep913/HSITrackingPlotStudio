from pathlib import Path

import pytest

from hsi_plotter.io.attribute_reader import AttributeDataError, read_attribute_dictionary


def test_attribute_dictionary_trims_names_and_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "attributes.txt"
    path.write_text(
        "BC: ball, basketball\nDEF:book, excavator\n",
        encoding="utf-8",
    )

    attributes = read_attribute_dictionary(path)

    assert list(attributes) == ["BC", "DEF"]
    assert attributes["BC"] == ("ball", "basketball")
    assert attributes["DEF"] == ("book", "excavator")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("BC:ball\nBC:worker\n", "重复属性"),
        ("BC:ball,ball\n", "重复序列"),
        (":ball\n", "属性名"),
        ("BC:\n", "序列"),
        ("BC ball\n", "冒号"),
    ],
)
def test_attribute_dictionary_rejects_ambiguous_content(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(AttributeDataError, match=message):
        read_attribute_dictionary(path)
