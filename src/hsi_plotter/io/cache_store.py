"""Safe atomic storage for large project sidecar cache payloads."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def cache_directory(project_path: Path) -> Path:
    path = Path(project_path).resolve(strict=False)
    return path.parent / f"{path.stem}.hsicache"


def _safe_module(module: str) -> str:
    normalized = "".join(char for char in module if char.isalnum() or char in "-_")
    if not normalized or normalized != module:
        raise ValueError("缓存模块名包含非法字符")
    return normalized


def write_json_cache(project_path: Path, module: str, payload: dict[str, Any]) -> str:
    root = cache_directory(project_path)
    target = root / f"{_safe_module(module)}.json"
    root.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target.relative_to(Path(project_path).resolve(strict=False).parent).as_posix()


def read_json_cache(project_path: Path, relative_path: str) -> dict[str, Any]:
    base = Path(project_path).resolve(strict=False).parent
    target = (base / relative_path).resolve(strict=False)
    root = cache_directory(project_path).resolve(strict=False)
    if target.parent != root:
        raise ValueError("缓存路径超出项目缓存目录")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("缓存文件根节点必须是 JSON 对象")
    return value


def clear_sidecar_cache(project_path: Path) -> None:
    root = cache_directory(project_path).resolve(strict=False)
    expected_parent = Path(project_path).resolve(strict=False).parent
    if root.parent != expected_parent or root.name != f"{Path(project_path).stem}.hsicache":
        raise ValueError("拒绝清除非项目缓存目录")
    if root.exists():
        shutil.rmtree(root)


__all__ = [
    "cache_directory",
    "clear_sidecar_cache",
    "read_json_cache",
    "write_json_cache",
]
