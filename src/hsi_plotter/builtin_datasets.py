from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BuiltinDataset:
    dataset_id: str
    label: str
    csv_path: Path
    sequence_count: int
    frame_count: int


def groundtruth_root() -> Path:
    if getattr(sys, "frozen", False):
        external_root = Path(sys.executable).resolve().parent / "Groundtruth"
        if (external_root / "datasets.json").is_file():
            return external_root
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        return bundle_root / "Groundtruth"
    return Path(__file__).resolve().parents[2] / "resources" / "Groundtruth"


def built_in_datasets() -> tuple[BuiltinDataset, ...]:
    root = groundtruth_root()
    manifest_path = root / "datasets.json"
    if not manifest_path.is_file():
        return ()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ()
    datasets: list[BuiltinDataset] = []
    for item in payload.get("datasets", []):
        csv_path = (root / str(item["path"])).resolve(strict=False)
        if not csv_path.is_file():
            continue
        datasets.append(
            BuiltinDataset(
                dataset_id=str(item["id"]),
                label=str(item.get("label", item["id"])),
                csv_path=csv_path,
                sequence_count=int(item.get("sequence_count", 0)),
                frame_count=int(item.get("frame_count", 0)),
            )
        )
    return tuple(datasets)
