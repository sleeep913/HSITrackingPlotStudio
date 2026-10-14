from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from hsi_plotter.io.sequence_discovery import discover_sequence_files
from hsi_plotter.io.tracking_reader import read_bounding_boxes

DATASETS = ("HOTC2020", "HOTC2023", "HOTC2024")


def _number(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return format(float(value), ".17g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_dataset(source: Path, destination: Path, dataset: str) -> dict[str, object]:
    discovery = discover_sequence_files(source, dataset=dataset)
    if discovery.report.has_errors or not discovery.files:
        messages = "; ".join(issue.message for issue in discovery.report.issues)
        raise RuntimeError(f"{dataset} Ground Truth 扫描失败：{messages}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    modalities: set[str] = set()
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("dataset", "modality", "sequence", "frame", "x", "y", "width", "height"))
        for key in sorted(discovery.files, key=lambda item: item.identifier.casefold()):
            boxes = read_bounding_boxes(discovery.files[key], allow_missing_rows=False)
            modalities.add(key.modality.value)
            for frame, values in enumerate(boxes.values, start=1):
                writer.writerow(
                    (
                        dataset,
                        key.modality.value,
                        key.name,
                        frame,
                        *(_number(value) for value in values),
                    )
                )
            frame_count += boxes.frame_count

    return {
        "id": dataset,
        "label": dataset,
        "path": f"{dataset}/groundtruth.csv",
        "sequence_count": len(discovery.files),
        "frame_count": frame_count,
        "modalities": sorted(modalities),
        "sha256": _sha256(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    args = parser.parse_args()

    datasets = []
    for dataset in DATASETS:
        output = args.destination_root / dataset / "groundtruth.csv"
        datasets.append(
            generate_dataset(args.source_root / dataset, output, dataset)
        )
    manifest = {"schema_version": 1, "datasets": datasets}
    (args.destination_root / "datasets.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
