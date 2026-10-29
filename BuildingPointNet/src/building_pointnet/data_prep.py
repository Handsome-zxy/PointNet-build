"""Build the point cache, spatial split, window index and training schedule."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .config import cache_dir, data_dir, resolve_path
from .las_utils import LasHeader, iter_chunks, point_keys, read_header

SUBSET_TRAIN = 0
SUBSET_VAL = 1
SUBSET_BUFFER = 2
SUBSET_IGNORE = 3

SUBSET_NAMES = {
    SUBSET_TRAIN: "train",
    SUBSET_VAL: "val",
    SUBSET_BUFFER: "buffer",
    SUBSET_IGNORE: "ignore",
}


def _raw_bounds(lo: float, hi: float, offset: float, scale: float) -> Tuple[int, int]:
    return (
        int(math.ceil((lo - offset) / scale - 1e-9)),
        int(math.ceil((hi - offset) / scale - 1e-9)),
    )


def _starts(vmin: float, vmax: float, size: float, stride: float) -> List[float]:
    if vmax - vmin <= size:
        return [float(vmin)]
    out: List[float] = []
    s = float(vmin)
    while s + size < vmax - 1e-9:
        out.append(s)
        s += stride
    last = max(float(vmin), float(vmax - size))
    if not out or last > out[-1] + 1e-9:
        out.append(last)
    return out


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _load_npy(path: Path, mmap: bool = True) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None)


def _box_for_cfg(cfg: Dict[str, Any]) -> Tuple[float, float, float, float]:
    vx0, vx1, vy0, vy1 = [float(x) for x in cfg["data"]["val_box"]]
    m = float(cfg["data"]["buffer_margin"])
    return vx0 - m, vx1 + m, vy0 - m, vy1 + m


def _assign_subsets(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    hdr: LasHeader,
    cfg: Dict[str, Any],
) -> np.ndarray:
    vx0, vx1, vy0, vy1 = [float(x) for x in cfg["data"]["val_box"]]
    bx0, bx1, by0, by1 = _box_for_cfg(cfg)

    x0r, x1r = _raw_bounds(vx0, vx1, hdr.offsets[0], hdr.scales[0])
    y0r, y1r = _raw_bounds(vy0, vy1, hdr.offsets[1], hdr.scales[1])
    bx0r, bx1r = _raw_bounds(bx0, bx1, hdr.offsets[0], hdr.scales[0])
    by0r, by1r = _raw_bounds(by0, by1, hdr.offsets[1], hdr.scales[1])

    in_val = (x_raw >= x0r) & (x_raw < x1r) & (y_raw >= y0r) & (y_raw < y1r)
    in_buffer = (
        (x_raw >= bx0r)
        & (x_raw < bx1r)
        & (y_raw >= by0r)
        & (y_raw < by1r)
        & (~in_val)
    )
    subset = np.full(len(x_raw), SUBSET_TRAIN, dtype=np.uint8)
    subset[in_buffer] = SUBSET_BUFFER
    subset[in_val] = SUBSET_VAL
    return subset


def _read_small_file(
    path: Path,
    out: Dict[str, np.ndarray],
    start: int,
    label: int,
    source: int,
    cfg: Dict[str, Any],
    store_keys: bool,
) -> Tuple[LasHeader, np.ndarray | None]:
    hdr = read_header(path)
    n = hdr.n_points
    end = start + n
    if end > len(out["label"]):
        raise ValueError("small cache allocation too small")

    keys = np.empty(n, dtype=np.uint64) if store_keys else None
    pos = start
    for chunk in iter_chunks(path):
        m = len(chunk["x"])
        sl = slice(pos, pos + m)
        out["x_raw"][sl] = chunk["x"]
        out["y_raw"][sl] = chunk["y"]
        out["z_raw"][sl] = chunk["z"]
        out["rgb"][sl] = chunk["rgb"]
        out["label"][sl] = label
        out["source"][sl] = source
        out["row"][sl] = chunk["row_offset"]
        if keys is not None:
            keys[pos - start:pos - start + m] = point_keys(chunk["x"], chunk["y"], chunk["z"])
        pos += m
    if pos != end:
        raise IOError(f"{path}: read {pos - start} points, expected {n}")
    return hdr, keys


def _read_test_file(
    path: Path,
    out: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> LasHeader:
    hdr = read_header(path)
    pos = 0
    for chunk in iter_chunks(path):
        m = len(chunk["x"])
        sl = slice(pos, pos + m)
        out["x_raw"][sl] = chunk["x"]
        out["y_raw"][sl] = chunk["y"]
        out["z_raw"][sl] = chunk["z"]
        out["rgb"][sl] = chunk["rgb"]
        flag = np.rint(chunk["extra_f32"]).astype(np.int16)
        valid = (flag == 0) | (flag == 1)
        # 0 -> building (1); 1 -> non-building (0); anything else -> ignore
        label = np.full(m, -1, dtype=np.int8)
        label[valid] = (1 - flag[valid]).astype(np.int8)
        out["label"][sl] = label
        out["source"][sl] = np.where(valid, flag, -1).astype(np.int8)
        out["row"][sl] = chunk["row_offset"]
        pos += m
    if pos != hdr.n_points:
        raise IOError(f"{path}: read {pos}, expected {hdr.n_points}")
    return hdr


def _build_windows(
    cfg: Dict[str, Any],
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    hdr: LasHeader,
    indices: np.ndarray,
    set_name: str,
    min_points: int,
) -> Dict[str, Any]:
    size = float(cfg["data"]["window_size"])
    stride = float(cfg["data"]["window_stride"])

    xs = np.asarray(x_raw[indices], dtype=np.int32)
    ys = np.asarray(y_raw[indices], dtype=np.int32)
    xmin = xs.min() * hdr.scales[0] + hdr.offsets[0]
    xmax = xs.max() * hdr.scales[0] + hdr.offsets[0]
    ymin = ys.min() * hdr.scales[1] + hdr.offsets[1]
    ymax = ys.max() * hdr.scales[1] + hdr.offsets[1]

    starts_x = _starts(xmin, xmax + hdr.scales[0], size, stride)
    starts_y = _starts(ymin, ymax + hdr.scales[1], size, stride)

    all_idx: List[np.ndarray] = []
    lengths: List[int] = []
    meta: List[Dict[str, Any]] = []

    def add_window(x0: float, y0: float) -> bool:
        y0r, y1r = _raw_bounds(y0, y0 + size, hdr.offsets[1], hdr.scales[1])
        x0r, x1r = _raw_bounds(x0, x0 + size, hdr.offsets[0], hdr.scales[0])
        mask = (xs >= x0r) & (xs < x1r) & (ys >= y0r) & (ys < y1r)
        local = np.flatnonzero(mask)
        n = int(len(local))
        if n == 0:
            return False
        if set_name == "train" and n < min_points:
            return False
        global_idx = indices[local].astype(np.int64)
        all_idx.append(global_idx)
        lengths.append(n)
        meta.append(
            {
                "id": len(meta),
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x0 + size),
                "y1": float(y0 + size),
                "cx": float(x0 + size / 2.0),
                "cy": float(y0 + size / 2.0),
                "n_points": n,
            }
        )
        return True

    for y0 in starts_y:
        for x0 in starts_x:
            add_window(x0, y0)

    if not all_idx:
        raise RuntimeError(f"no windows for set {set_name}")

    concat = np.concatenate(all_idx).astype(np.int32)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)

    # coverage check
    covered = np.zeros(len(indices), dtype=bool)
    for a, b in zip(offsets[:-1], offsets[1:]):
        global_idx = concat[a:b]
        # map global point index to local index in `indices`
        # indices is sorted? train/val from flatnonzero are sorted; test is arange.
        local = np.searchsorted(indices, global_idx)
        covered[local] = True
    coverage = float(covered.mean())
    if set_name in ("val", "test") and coverage < 1.0:
        # A few points can fall into a sub-millimetre sliver at a bbox edge.
        # Add one centered window per uncovered point, then re-check.
        missing_local = np.flatnonzero(~covered)
        for li in missing_local:
            g = int(indices[li])
            xm = x_raw[g] * hdr.scales[0] + hdr.offsets[0]
            ym = y_raw[g] * hdr.scales[1] + hdr.offsets[1]
            add_window(xm - size / 2.0, ym - size / 2.0)

        concat = np.concatenate(all_idx).astype(np.int32)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        covered[:] = False
        for a, b in zip(offsets[:-1], offsets[1:]):
            global_idx = concat[a:b]
            local = np.searchsorted(indices, global_idx)
            covered[local] = True
        coverage = float(covered.mean())
        if coverage < 1.0:
            missing = int((~covered).sum())
            raise RuntimeError(f"{set_name}: {missing} points are not covered by any window")

    return {
        "indices": concat,
        "offsets": offsets,
        "meta": meta,
        "coverage": coverage,
        "n_windows": len(meta),
        "total_sampled": int(len(concat)),
    }


def _save_windows(root: Path, set_name: str, data: Dict[str, Any]) -> None:
    d = root / "windows"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{set_name}_indices.npy", data["indices"])
    np.save(d / f"{set_name}_offsets.npy", data["offsets"])
    with open(d / f"{set_name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_windows": data["n_windows"],
                "coverage": data["coverage"],
                "total_sampled": data["total_sampled"],
                "windows": data["meta"],
            },
            f,
            indent=2,
        )


def _make_schedule(
    root: Path,
    n_windows: int,
    max_steps: int,
    batch_size: int,
    seed: int,
) -> Dict[str, int]:
    n_blocks = int(max_steps) * int(batch_size)
    rng = np.random.RandomState(seed)
    window_id = rng.randint(0, n_windows, size=n_blocks).astype(np.int32)
    block_seed = rng.randint(0, 2**31 - 1, size=n_blocks).astype(np.uint32)
    d = root / "windows"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "schedule.npz", window_id=window_id, block_seed=block_seed)
    return {"n_blocks": n_blocks, "seed": int(seed)}


def _write_conflicts(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rows: np.ndarray,
    hdr: LasHeader,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "noise_row"])
        for i in range(len(x)):
            w.writerow(
                [
                    f"{x[i] * hdr.scales[0] + hdr.offsets[0]:.2f}",
                    f"{y[i] * hdr.scales[1] + hdr.offsets[1]:.2f}",
                    f"{z[i] * hdr.scales[2] + hdr.offsets[2]:.2f}",
                    int(rows[i]),
                ]
            )


def prepare_all(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw_dir = data_dir(cfg)
    root = cache_dir(cfg)
    (root / "small").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)

    build_path = raw_dir / cfg["data"]["build_small"]
    noise_path = raw_dir / cfg["data"]["noise_small"]
    test_path = raw_dir / cfg["data"]["test"]

    build_hdr = read_header(build_path)
    noise_hdr = read_header(noise_path)
    test_hdr = read_header(test_path)

    n_build = build_hdr.n_points
    n_noise = noise_hdr.n_points
    n_small = n_build + n_noise

    small: Dict[str, np.ndarray] = {
        "x_raw": np.empty(n_small, dtype=np.int32),
        "y_raw": np.empty(n_small, dtype=np.int32),
        "z_raw": np.empty(n_small, dtype=np.int32),
        "rgb": np.empty((n_small, 3), dtype=np.uint16),
        "label": np.empty(n_small, dtype=np.uint8),
        "source": np.empty(n_small, dtype=np.uint8),
        "row": np.empty(n_small, dtype=np.int32),
        "subset": np.empty(n_small, dtype=np.uint8),
    }

    print(f"reading build_small ({n_build:,}) ...", flush=True)
    hdr_b, build_keys = _read_small_file(build_path, small, 0, 1, 1, cfg, store_keys=True)
    print(f"reading noise_small ({n_noise:,}) ...", flush=True)
    hdr_n, noise_keys = _read_small_file(noise_path, small, n_build, 0, 0, cfg, store_keys=True)

    if hdr_b.scales != hdr_n.scales or hdr_b.offsets != hdr_n.offsets:
        raise ValueError("build_small and noise_small must share scales/offsets")

    print("assigning train/val/buffer ...", flush=True)
    small["subset"][:] = _assign_subsets(small["x_raw"], small["y_raw"], hdr_b, cfg)

    print("checking cross-class coordinate conflicts ...", flush=True)
    build_unique = np.unique(build_keys)
    conflict_noise = np.isin(noise_keys, build_unique)
    n_conflict = int(conflict_noise.sum())
    conflict_pos = n_build + np.flatnonzero(conflict_noise)
    small["subset"][conflict_pos] = SUBSET_IGNORE
    _write_conflicts(
        resolve_path(cfg, "reports/conflicting_noise_points.csv"),
        small["x_raw"][conflict_pos],
        small["y_raw"][conflict_pos],
        small["z_raw"][conflict_pos],
        small["row"][conflict_pos],
        hdr_b,
    )
    del build_keys, noise_keys, build_unique, conflict_noise

    # duplicate statistics within each class (reported, not removed)
    dup_build = int(n_build - len(np.unique(point_keys(
        small["x_raw"][:n_build], small["y_raw"][:n_build], small["z_raw"][:n_build]))))
    dup_noise = int(n_noise - len(np.unique(point_keys(
        small["x_raw"][n_build:], small["y_raw"][n_build:], small["z_raw"][n_build:]))))

    # save small arrays
    for key in ("x_raw", "y_raw", "z_raw", "rgb", "label", "source", "row", "subset"):
        _save_npy(root / "small" / f"{key}.npy", small[key])
    small_hdr = {
        "scales": list(hdr_b.scales),
        "offsets": list(hdr_b.offsets),
        "point_format": hdr_b.point_format,
        "record_len": hdr_b.record_len,
    }
    with open(root / "small" / "header.json", "w", encoding="utf-8") as f:
        json.dump(small_hdr, f, indent=2)

    train_mask = small["subset"] == SUBSET_TRAIN
    val_mask = small["subset"] == SUBSET_VAL
    buffer_mask = small["subset"] == SUBSET_BUFFER
    ignore_mask = small["subset"] == SUBSET_IGNORE
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise RuntimeError("empty train or val split")

    z_ref_raw = float(np.median(small["z_raw"][train_mask]))
    z_ref = z_ref_raw * hdr_b.scales[2] + hdr_b.offsets[2]

    train_counts = np.bincount(small["label"][train_mask], minlength=2).astype(np.float64)
    mode = str(cfg["train"].get("class_weight_mode", "none")).lower()
    if mode == "inverse":
        class_weights = float(train_mask.sum()) / (2.0 * train_counts)
        class_weights = class_weights / class_weights.mean()
    elif mode == "none":
        # Pilot at 2.5 epochs showed that inverse-frequency weighting
        # over-corrects: building AP was 0.9976 but recall at 0.5 was 0.004.
        # Building is the majority in train/val, and Dice already handles the
        # imbalance, so v1 uses unweighted CE + Dice.
        class_weights = np.ones(2, dtype=np.float64)
    else:
        raise ValueError(f"unknown class_weight_mode: {mode}")

    split_counts = {
        name: {
            "total": int((small["subset"] == sid).sum()),
            "building": int(((small["subset"] == sid) & (small["label"] == 1)).sum()),
            "nonbuilding": int(((small["subset"] == sid) & (small["label"] == 0)).sum()),
        }
        for sid, name in SUBSET_NAMES.items()
    }

    # test cache
    n_test = test_hdr.n_points
    test: Dict[str, np.ndarray] = {
        "x_raw": np.empty(n_test, dtype=np.int32),
        "y_raw": np.empty(n_test, dtype=np.int32),
        "z_raw": np.empty(n_test, dtype=np.int32),
        "rgb": np.empty((n_test, 3), dtype=np.uint16),
        "label": np.empty(n_test, dtype=np.int8),
        "source": np.empty(n_test, dtype=np.int8),
        "row": np.empty(n_test, dtype=np.int32),
    }
    print(f"reading test ({n_test:,}) ...", flush=True)
    _read_test_file(test_path, test, cfg)
    for key in ("x_raw", "y_raw", "z_raw", "rgb", "label", "source", "row"):
        _save_npy(root / "test" / f"{key}.npy", test[key])
    with open(root / "test" / "header.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scales": list(test_hdr.scales),
                "offsets": list(test_hdr.offsets),
                "point_format": test_hdr.point_format,
                "record_len": test_hdr.record_len,
            },
            f,
            indent=2,
        )
    test_label_counts = {
        "building": int((test["label"] == 1).sum()),
        "nonbuilding": int((test["label"] == 0).sum()),
        "ignore": int((test["label"] == -1).sum()),
    }

    # windows
    print("building train windows ...", flush=True)
    train_idx = np.flatnonzero(small["subset"] == SUBSET_TRAIN)
    win_train = _build_windows(cfg, small["x_raw"], small["y_raw"], hdr_b, train_idx, "train",
                               int(cfg["data"]["min_window_points"]))
    print(f"  train windows={win_train['n_windows']}, coverage={win_train['coverage']:.4f}", flush=True)
    _save_windows(root, "train", win_train)

    print("building val windows ...", flush=True)
    val_idx = np.flatnonzero(small["subset"] == SUBSET_VAL)
    win_val = _build_windows(cfg, small["x_raw"], small["y_raw"], hdr_b, val_idx, "val", 0)
    print(f"  val windows={win_val['n_windows']}, coverage={win_val['coverage']:.4f}", flush=True)
    _save_windows(root, "val", win_val)

    print("building test windows ...", flush=True)
    test_idx = np.arange(n_test, dtype=np.int64)
    win_test = _build_windows(cfg, test["x_raw"], test["y_raw"], test_hdr, test_idx, "test", 0)
    print(f"  test windows={win_test['n_windows']}, coverage={win_test['coverage']:.4f}", flush=True)
    _save_windows(root, "test", win_test)

    sched = _make_schedule(
        root,
        win_train["n_windows"],
        int(cfg["train"]["max_steps"]),
        int(cfg["train"]["batch_size"]),
        int(cfg["train"]["schedule_seed"]),
    )

    manifest = {
        "counts": {
            "build_small": int(n_build),
            "noise_small": int(n_noise),
            "small_total": int(n_small),
            "test": int(n_test),
            "conflicting_noise_removed": int(n_conflict),
            "duplicate_build_within_class": int(dup_build),
            "duplicate_noise_within_class": int(dup_noise),
        },
        "split_counts": split_counts,
        "test_label_counts": test_label_counts,
        "val_box": [float(x) for x in cfg["data"]["val_box"]],
        "buffer_box": [float(x) for x in _box_for_cfg(cfg)],
        "z_ref": float(z_ref),
        "rgb_scale": float(cfg["data"]["rgb_scale"]),
        "class_weights": [float(x) for x in class_weights],
        "class_weight_mode": mode,
        "train_class_counts": [int(x) for x in train_counts],
        "headers": {
            "small": small_hdr,
            "test": {
                "scales": list(test_hdr.scales),
                "offsets": list(test_hdr.offsets),
            },
        },
        "windows": {
            "train": {"n_windows": win_train["n_windows"], "coverage": win_train["coverage"],
                      "total_sampled": win_train["total_sampled"]},
            "val": {"n_windows": win_val["n_windows"], "coverage": win_val["coverage"],
                    "total_sampled": win_val["total_sampled"]},
            "test": {"n_windows": win_test["n_windows"], "coverage": win_test["coverage"],
                     "total_sampled": win_test["total_sampled"]},
        },
        "schedule": sched,
        "config": {
            "window_size": cfg["data"]["window_size"],
            "window_stride": cfg["data"]["window_stride"],
            "points_per_block": cfg["data"]["points_per_block"],
        },
    }
    with open(root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _write_data_report(cfg, root, manifest)
    return manifest


def _write_data_report(cfg: Dict[str, Any], root: Path, manifest: Dict[str, Any]) -> None:
    report = resolve_path(cfg, "reports/data_report.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    c = manifest["counts"]
    s = manifest["split_counts"]
    t = manifest["test_label_counts"]
    lines = [
        "# BuildingPointNet 数据核验报告",
        "",
        "## 点数",
        "",
        f"- build_small: {c['build_small']:,}",
        f"- noise_small: {c['noise_small']:,}",
        f"- small 合计: {c['small_total']:,}",
        f"- 删除的跨类冲突 noise 点: {c['conflicting_noise_removed']:,}",
        f"- build_small 类内重复坐标: {c['duplicate_build_within_class']:,}",
        f"- noise_small 类内重复坐标: {c['duplicate_noise_within_class']:,}",
        "",
        "## 划分",
        "",
        "| 集合 | 总数 | 建筑 | 非建筑 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("train", "val", "buffer", "ignore"):
        lines.append(
            f"| {name} | {s[name]['total']:,} | {s[name]['building']:,} | {s[name]['nonbuilding']:,} |"
        )
    lines += [
        "",
        f"- val_box: {manifest['val_box']}",
        f"- buffer_box: {manifest['buffer_box']}",
        f"- z_ref (train median): {manifest['z_ref']:.4f} m",
        f"- class_weights: {manifest['class_weights']}",
        f"- train class counts: {manifest['train_class_counts']}",
        "",
        "## 测试集标签（来自 Original cloud index 坐标追溯）",
        "",
        f"- building: {t['building']:,}",
        f"- non-building: {t['nonbuilding']:,}",
        f"- ignore: {t['ignore']:,}",
        "",
        "## 窗口",
        "",
        "| 集合 | 窗口数 | 覆盖率 | 采样点次 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("train", "val", "test"):
        w = manifest["windows"][name]
        lines.append(f"| {name} | {w['n_windows']} | {w['coverage']:.4f} | {w['total_sampled']:,} |")
    lines += [
        "",
        f"- window_size={cfg['data']['window_size']} m, stride={cfg['data']['window_stride']} m, "
        f"points_per_block={cfg['data']['points_per_block']}",
        f"- schedule: {manifest['schedule']}",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
