"""Run directories, JSON/CSV logging and environment capture."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_files(paths: Iterable[str | Path]) -> Dict[str, str]:
    return {str(Path(p).resolve()): sha256_file(p) for p in paths}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(root: Path, arm: str, seed: int, suffix: str = "") -> Path:
    name = f"{timestamp()}_{arm}_s{seed}"
    if suffix:
        name += f"_{suffix}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
        f.write("\n")


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


class CsvLogger:
    def __init__(self, path: Path, fieldnames: Iterable[str]):
        self.path = path
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames, extrasaction="ignore")
        self._w.writeheader()
        self._f.flush()

    def write(self, row: Dict[str, Any]) -> None:
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)

    def __call__(self, msg: str) -> None:
        print(msg)
        self._f.write(msg + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def collect_env(cfg: Dict[str, Any], command: str) -> Dict[str, Any]:
    import tensorflow as tf

    env = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "command": command,
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [],
        "config_path": cfg.get("_config_path"),
    }
    try:
        env["gpus"] = [
            {
                "name": d.name,
                "device_type": d.device_type,
            }
            for d in tf.config.list_physical_devices("GPU")
        ]
    except Exception as exc:  # pragma: no cover - diagnostics only
        env["gpus_error"] = repr(exc)
    return env
