from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from hsi_plotter.models.dataset import Modality, SequenceKey, parse_sequence_identity


class CsvBundleError(ValueError):
    """Raised when a combined Ground Truth or tracker-result CSV is invalid."""


@dataclass(frozen=True, slots=True)
class CsvBundle:
    source_path: Path
    files: dict[SequenceKey, Path]
    frame_count: int

    @property
    def sequence_count(self) -> int:
        return len(self.files)


_ID_PATTERN = re.compile(r"^(?P<sequence>.+)_(?P<frame>[1-9]\d*)$")
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_MODALITY_ALIASES = {
    "vis": Modality.VIS,
    "visible": Modality.VIS,
    "nir": Modality.NIR,
    "infrared": Modality.NIR,
    "rednir": Modality.REDNIR,
    "red-nir": Modality.REDNIR,
    "unspecified": Modality.UNSPECIFIED,
    "": Modality.UNSPECIFIED,
}


def _parse_sequence(value: str) -> tuple[Modality, str]:
    modality, name = parse_sequence_identity(value)
    if modality is not Modality.UNSPECIFIED:
        return modality, name
    lowered = value.casefold()
    for prefix, candidate in (
        ("rednir_", Modality.REDNIR),
        ("nir_", Modality.NIR),
        ("vis_", Modality.VIS),
    ):
        if lowered.startswith(prefix):
            return candidate, value[len(prefix) :]
    return Modality.UNSPECIFIED, value


def _parse_float(path: Path, row_number: int, column: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise CsvBundleError(f"{path} 第 {row_number} 行字段 {column} 不是数值") from exc
    if math.isinf(parsed):
        raise CsvBundleError(f"{path} 第 {row_number} 行字段 {column} 包含无穷值")
    return parsed


def _parse_modality(path: Path, row_number: int, value: str) -> Modality:
    normalized = value.strip().casefold()
    try:
        return _MODALITY_ALIASES[normalized]
    except KeyError as exc:
        raise CsvBundleError(
            f"{path} 第 {row_number} 行包含不支持的模态：{value!r}"
        ) from exc


def _cache_directory(source: Path, cache_root: Path | None) -> Path:
    digest = hashlib.sha256()
    digest.update(b"hsi-csv-bundle-v1\0")
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if cache_root is None:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        cache_root = base / "HSITrackingPlotStudio" / "csv-cache"
    return Path(cache_root) / digest.hexdigest()


def _format_number(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return format(value, ".17g")


def load_csv_bundle(
    source: Path,
    *,
    dataset: str,
    cache_root: Path | None = None,
) -> CsvBundle:
    """Parse one combined CSV and expose it as deterministic per-sequence files.

    Supported schemas are ``ID,x,y,width,height`` and the canonical
    ``dataset,modality,sequence,frame,x,y,width,height`` form.  ``dataset`` and
    ``modality`` are optional in the latter form.
    """

    path = Path(source).resolve(strict=False)
    if not path.is_file() or path.suffix.casefold() != ".csv":
        raise CsvBundleError(f"CSV 文件不存在或扩展名不是 .csv：{path}")

    rows_by_key: dict[SequenceKey, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise CsvBundleError(f"CSV 文件缺少表头：{path}")
            headers = {name.strip().casefold(): name for name in reader.fieldnames if name}
            required = {"x", "y", "width", "height"}
            missing = sorted(required - set(headers))
            if missing:
                raise CsvBundleError(f"CSV 缺少必需字段：{', '.join(missing)}")
            has_id = "id" in headers
            has_sequence_frame = "sequence" in headers and "frame" in headers
            if not has_id and not has_sequence_frame:
                raise CsvBundleError("CSV 必须包含 ID，或同时包含 sequence 与 frame 字段")

            for row_number, row in enumerate(reader, start=2):
                if not any((value or "").strip() for value in row.values()):
                    continue
                if has_sequence_frame:
                    sequence = (row.get(headers["sequence"]) or "").strip()
                    frame_text = (row.get(headers["frame"]) or "").strip()
                else:
                    identifier = (row.get(headers["id"]) or "").strip()
                    match = _ID_PATTERN.fullmatch(identifier)
                    if match is None:
                        raise CsvBundleError(
                            f"{path} 第 {row_number} 行 ID 必须以 _帧号 结尾：{identifier!r}"
                        )
                    sequence = match.group("sequence")
                    frame_text = match.group("frame")
                if not sequence:
                    raise CsvBundleError(f"{path} 第 {row_number} 行序列名为空")
                try:
                    frame = int(frame_text)
                except ValueError as exc:
                    raise CsvBundleError(f"{path} 第 {row_number} 行帧号不是整数") from exc
                if frame < 1:
                    raise CsvBundleError(f"{path} 第 {row_number} 行帧号必须从 1 开始")

                inferred_modality, sequence_name = _parse_sequence(sequence)
                if "modality" in headers:
                    explicit = _parse_modality(
                        path, row_number, row.get(headers["modality"]) or ""
                    )
                    modality = (
                        explicit
                        if explicit is not Modality.UNSPECIFIED
                        else inferred_modality
                    )
                else:
                    modality = inferred_modality
                key = SequenceKey(dataset=dataset, modality=modality, name=sequence_name)
                if frame in rows_by_key[key]:
                    raise CsvBundleError(
                        f"{path} 第 {row_number} 行重复定义序列 {sequence} 的第 {frame} 帧"
                    )

                values = tuple(
                    _parse_float(path, row_number, column, row.get(headers[column]) or "")
                    for column in ("x", "y", "width", "height")
                )
                nan_count = sum(math.isnan(value) for value in values)
                if nan_count not in (0, 4):
                    raise CsvBundleError(f"{path} 第 {row_number} 行存在部分 NaN 边界框")
                rows_by_key[key][frame] = values  # type: ignore[assignment]
    except (OSError, UnicodeError, csv.Error) as exc:
        if isinstance(exc, CsvBundleError):
            raise
        raise CsvBundleError(f"无法读取 CSV 文件 {path}：{exc}") from exc

    if not rows_by_key:
        raise CsvBundleError(f"CSV 文件不包含任何边界框记录：{path}")

    output_dir = _cache_directory(path, cache_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[SequenceKey, Path] = {}
    manifest_rows: list[dict[str, object]] = []
    total_frames = 0
    ordered_keys = sorted(rows_by_key, key=lambda item: item.identifier.casefold())
    for index, key in enumerate(ordered_keys):
        frames = rows_by_key[key]
        expected = list(range(1, max(frames) + 1))
        actual = sorted(frames)
        if actual != expected:
            missing_frames = sorted(set(expected) - set(actual))
            preview = ", ".join(map(str, missing_frames[:10]))
            raise CsvBundleError(
                f"序列 {key.name} 的帧号不连续，缺少：{preview}"
            )
        safe_name = _SAFE_NAME_PATTERN.sub("_", key.name).strip("._") or "sequence"
        output = output_dir / f"{index:04d}_{key.modality.value}_{safe_name}.txt"
        lines = [
            ",".join(_format_number(value) for value in frames[frame])
            for frame in actual
        ]
        expected_text = "\n".join(lines) + "\n"
        if not output.is_file() or output.read_text(encoding="utf-8") != expected_text:
            output.write_text(expected_text, encoding="utf-8")
        files[key] = output.resolve(strict=False)
        total_frames += len(actual)
        manifest_rows.append(
            {
                "dataset": key.dataset,
                "modality": key.modality.value,
                "sequence": key.name,
                "frames": len(actual),
                "file": output.name,
            }
        )

    manifest = {
        "source": str(path),
        "sequence_count": len(files),
        "frame_count": total_frames,
        "sequences": manifest_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return CsvBundle(source_path=path, files=files, frame_count=total_frames)
