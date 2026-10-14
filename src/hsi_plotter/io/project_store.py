from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from hsi_plotter.io.cache_store import read_json_cache, write_json_cache
from hsi_plotter.models.cache import CacheManifest, CacheRecord
from hsi_plotter.models.project import Project, ProjectSources
from hsi_plotter.models.style import PlotStyle, PlotStylePreset, TrackerStyle
from hsi_plotter.models.tracker import Tracker
from hsi_plotter.models.tracker_import import ResultLayout

SCHEMA_VERSION = 4
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3, 4)


class ProjectFormatError(ValueError):
    """Raised when a project file is unreadable or has an unsupported schema."""


def _encode_path(path: Path | None, base_directory: Path) -> str | None:
    if path is None:
        return None
    normalized = path.resolve(strict=False)
    try:
        relative = os.path.relpath(normalized, base_directory)
    except ValueError:
        return normalized.as_posix()
    return Path(relative).as_posix()


def _decode_path(value: Any, base_directory: Path, field_name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProjectFormatError(f"字段 {field_name} 必须是非空路径字符串或 null")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base_directory / candidate
    return candidate.resolve(strict=False)


def _project_to_payload(project: Project, base_directory: Path) -> dict[str, Any]:
    sources = project.sources
    return {
        "schema_version": SCHEMA_VERSION,
        "name": project.name,
        "sources": {
            "ground_truth": _encode_path(sources.ground_truth, base_directory),
            "results_root": _encode_path(sources.results_root, base_directory),
            "image_root": _encode_path(sources.image_root, base_directory),
            "attribute_file": _encode_path(sources.attribute_file, base_directory),
            "export_directory": _encode_path(sources.export_directory, base_directory),
            "hsi_files": [_encode_path(path, base_directory) for path in sources.hsi_files],
        },
        "trackers": [
            {
                "tracker_id": tracker.tracker_id,
                "display_name": tracker.display_name,
                "results_path": _encode_path(tracker.results_path, base_directory),
                "layout": tracker.layout.value,
                "style": {
                    "color": tracker.style.color,
                    "line_style": tracker.style.line_style,
                    "line_width": tracker.style.line_width,
                },
            }
            for tracker in project.trackers
        ],
        "recipe_files": [_encode_path(path, base_directory) for path in project.recipe_files],
        "plot_style": _plot_style_payload(project.plot_style),
        "cache_manifest": _cache_manifest_payload(project.cache_manifest),
    }


def _cache_manifest_payload(manifest: CacheManifest) -> dict[str, object]:
    return {
        "records": {
            module: {
                "module": record.module,
                "cache_schema_version": record.cache_schema_version,
                "algorithm_version": record.algorithm_version,
                "created_at": record.created_at,
                "input_fingerprint": record.input_fingerprint,
                "payload_kind": record.payload_kind,
                "payload": record.payload,
                "relative_path": record.relative_path,
            }
            for module, record in manifest.records.items()
        }
    }


def _plot_style_payload(style: PlotStyle) -> dict[str, object]:
    return {
        "preset": style.preset.value,
        "font_family": style.font_family,
        "font_size": style.font_size,
        "title_size": style.title_size,
        "legend_size": style.legend_size,
        "legend_scale": style.legend_scale,
        "text_color": style.text_color,
        "background_color": style.background_color,
        "axes_color": style.axes_color,
        "grid_enabled": style.grid_enabled,
        "grid_color": style.grid_color,
        "grid_alpha": style.grid_alpha,
        "axes_line_width": style.axes_line_width,
        "default_line_width": style.default_line_width,
        "marker": style.marker,
        "legend_location": style.legend_location,
        "legend_columns": style.legend_columns,
        "width_inches": style.width_inches,
        "height_inches": style.height_inches,
        "dpi": style.dpi,
        "margin_left": style.margin_left,
        "margin_right": style.margin_right,
        "margin_bottom": style.margin_bottom,
        "margin_top": style.margin_top,
    }


def _decode_plot_style(value: Any) -> PlotStyle:
    if value is None:
        return PlotStyle()
    payload = _require_mapping(value, "plot_style")
    try:
        style = PlotStyle(
            **{
                **payload,
                "preset": PlotStylePreset(
                    payload.get("preset", PlotStylePreset.PAPER_DOUBLE_COLUMN)
                ),
            }
        )
        style.validate()
    except (TypeError, ValueError) as exc:
        raise ProjectFormatError(f"字段 plot_style 无效：{exc}") from exc
    return style


def _decode_cache_manifest(value: Any) -> CacheManifest:
    if value is None:
        return CacheManifest()
    try:
        root = _require_mapping(value, "cache_manifest")
        raw_records = _require_mapping(root.get("records", {}), "cache_manifest.records")
    except ProjectFormatError:
        return CacheManifest()
    records: dict[str, CacheRecord] = {}
    for module, raw_record in raw_records.items():
        try:
            item = _require_mapping(raw_record, f"cache_manifest.records.{module}")
            payload = item.get("payload")
            if payload is not None and not isinstance(payload, dict):
                raise ProjectFormatError("缓存 payload 必须是 JSON 对象或 null")
            record = CacheRecord(
                module=_require_string(item.get("module"), "cache.module"),
                cache_schema_version=int(item.get("cache_schema_version")),
                algorithm_version=_require_string(
                    item.get("algorithm_version"), "cache.algorithm_version"
                ),
                created_at=_require_string(item.get("created_at"), "cache.created_at"),
                input_fingerprint=_require_string(
                    item.get("input_fingerprint"), "cache.input_fingerprint"
                ),
                payload=payload,
                payload_kind=str(item.get("payload_kind", "embedded_json")),
                relative_path=item.get("relative_path"),
            )
        except (ProjectFormatError, TypeError, ValueError):
            continue
        records[str(module)] = record
    return CacheManifest(records=records)


def save_project(project: Project, destination: Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    for module, record in tuple(project.cache_manifest.records.items()):
        if record.payload_kind == "sidecar_json":
            payload = record.payload
            if (
                payload is None
                and project.storage_path is not None
                and record.relative_path is not None
            ):
                try:
                    payload = read_json_cache(project.storage_path, record.relative_path)
                except (OSError, ValueError):
                    payload = None
            if payload is None:
                continue
            relative_path = write_json_cache(path, module, payload)
            project.cache_manifest.records[module] = replace(
                record, payload=None, relative_path=relative_path
            )
    project.storage_path = path.resolve(strict=False)
    payload = _project_to_payload(project, path.parent.resolve(strict=False))
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProjectFormatError(f"字段 {field_name} 必须是 JSON 对象")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectFormatError(f"字段 {field_name} 必须是非空字符串")
    return value


def load_project(source: Path) -> Project:
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectFormatError(f"项目文件不是有效 JSON：{path}；{exc}") from exc

    root = _require_mapping(payload, "root")
    version = root.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ProjectFormatError(
            f"不支持的 schema_version：{version!r}，当前支持 {SUPPORTED_SCHEMA_VERSIONS}"
        )

    base_directory = path.parent.resolve(strict=False)
    sources_payload = _require_mapping(root.get("sources", {}), "sources")
    hsi_values = sources_payload.get("hsi_files", [])
    if not isinstance(hsi_values, list):
        raise ProjectFormatError("字段 sources.hsi_files 必须是数组")

    sources = ProjectSources(
        ground_truth=_decode_path(
            sources_payload.get("ground_truth"), base_directory, "sources.ground_truth"
        ),
        results_root=_decode_path(
            sources_payload.get("results_root"), base_directory, "sources.results_root"
        ),
        image_root=_decode_path(
            sources_payload.get("image_root"), base_directory, "sources.image_root"
        ),
        attribute_file=_decode_path(
            sources_payload.get("attribute_file"), base_directory, "sources.attribute_file"
        ),
        export_directory=_decode_path(
            sources_payload.get("export_directory"),
            base_directory,
            "sources.export_directory",
        ),
        hsi_files=tuple(
            path_value
            for index, value in enumerate(hsi_values)
            if (path_value := _decode_path(value, base_directory, f"sources.hsi_files[{index}]"))
            is not None
        ),
    )

    tracker_values = root.get("trackers", [])
    if not isinstance(tracker_values, list):
        raise ProjectFormatError("字段 trackers 必须是数组")
    trackers: list[Tracker] = []
    for index, value in enumerate(tracker_values):
        tracker_payload = _require_mapping(value, f"trackers[{index}]")
        style_payload = _require_mapping(tracker_payload.get("style"), f"trackers[{index}].style")
        results_path = _decode_path(
            tracker_payload.get("results_path"),
            base_directory,
            f"trackers[{index}].results_path",
        )
        if results_path is None:
            raise ProjectFormatError(f"trackers[{index}].results_path 不能为空")
        try:
            line_width = float(style_payload.get("line_width"))
        except (TypeError, ValueError) as exc:
            raise ProjectFormatError(f"trackers[{index}].style.line_width 必须是数值") from exc
        raw_layout = tracker_payload.get("layout")
        if raw_layout is None:
            layout = ResultLayout.FLAT
        else:
            try:
                layout = ResultLayout(str(raw_layout))
            except ValueError as exc:
                raise ProjectFormatError(
                    f"trackers[{index}].layout 不是支持的结果布局：{raw_layout!r}"
                ) from exc
        trackers.append(
            Tracker(
                tracker_id=_require_string(
                    tracker_payload.get("tracker_id"), f"trackers[{index}].tracker_id"
                ),
                display_name=_require_string(
                    tracker_payload.get("display_name"),
                    f"trackers[{index}].display_name",
                ),
                results_path=results_path,
                layout=layout,
                style=TrackerStyle(
                    color=_require_string(
                        style_payload.get("color"), f"trackers[{index}].style.color"
                    ),
                    line_style=_require_string(
                        style_payload.get("line_style"),
                        f"trackers[{index}].style.line_style",
                    ),
                    line_width=line_width,
                ),
            )
        )

    recipe_values = root.get("recipe_files", [])
    if not isinstance(recipe_values, list):
        raise ProjectFormatError("字段 recipe_files 必须是数组")
    recipe_files = [
        decoded
        for index, value in enumerate(recipe_values)
        if (decoded := _decode_path(value, base_directory, f"recipe_files[{index}]")) is not None
    ]

    return Project(
        name=_require_string(root.get("name"), "name"),
        sources=sources,
        trackers=trackers,
        recipe_files=recipe_files,
        plot_style=_decode_plot_style(root.get("plot_style")),
        cache_manifest=_decode_cache_manifest(root.get("cache_manifest")),
        storage_path=path.resolve(strict=False),
    )
