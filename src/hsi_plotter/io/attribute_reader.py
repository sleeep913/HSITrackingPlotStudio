from __future__ import annotations

from pathlib import Path

from hsi_plotter.models.attribute_evaluation import AttributeDefinition


class AttributeDataError(ValueError):
    """Raised when an attribute dictionary cannot be parsed unambiguously."""


def read_attribute_dictionary(source: Path) -> dict[str, tuple[str, ...]]:
    path = Path(source)
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AttributeDataError(f"无法读取属性字典 {path}：{exc}") from exc

    attributes: dict[str, tuple[str, ...]] = {}
    for line_number, original_line in enumerate(lines, start=1):
        line = original_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise AttributeDataError(f"第 {line_number} 行缺少冒号分隔符")
        raw_name, raw_sequences = line.split(":", maxsplit=1)
        name = raw_name.strip()
        if not name:
            raise AttributeDataError(f"第 {line_number} 行的属性名为空")
        if name in attributes:
            raise AttributeDataError(f"第 {line_number} 行包含重复属性：{name}")
        sequences = tuple(item.strip() for item in raw_sequences.split(",") if item.strip())
        if not sequences:
            raise AttributeDataError(f"属性 {name} 没有任何序列")
        if len(set(sequences)) != len(sequences):
            raise AttributeDataError(f"属性 {name} 中存在重复序列")
        attributes[name] = sequences

    if not attributes:
        raise AttributeDataError(f"属性字典为空：{path}")
    return attributes


def read_attribute_definitions(source: Path) -> tuple[AttributeDefinition, ...]:
    """Read the legacy text format into ordered domain objects."""
    return tuple(
        AttributeDefinition(name=name, sequences=sequences)
        for name, sequences in read_attribute_dictionary(source).items()
    )
