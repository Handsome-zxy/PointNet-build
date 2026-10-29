"""Configuration loading with a small `extends` merge mechanism."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base` (returns a new dict)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, _seen: set[str] | None = None) -> Dict[str, Any]:
    """Load a JSON config; `extends` is resolved relative to the config file."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if _seen is None:
        _seen = set()
    if str(path) in _seen:
        raise ValueError(f"circular extends: {path}")
    _seen.add(str(path))

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    parent = cfg.get("extends")
    if parent:
        parent_cfg = load_config(path.parent / parent, _seen)
        cfg = _deep_merge(parent_cfg, cfg)

    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve_path(cfg: Dict[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(value)
    if p.is_absolute():
        return p
    return (Path(cfg["_project_root"]) / p).resolve()


def data_dir(cfg: Dict[str, Any]) -> Path:
    return resolve_path(cfg, cfg["data"]["raw_dir"])


def cache_dir(cfg: Dict[str, Any]) -> Path:
    p = resolve_path(cfg, cfg["data"]["cache_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
