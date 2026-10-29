"""Cache-backed dataset, training sampler and deterministic evaluation batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

from .config import cache_dir


class CacheSet:
    """Point arrays for one split (small or test)."""

    def __init__(self, root: Path, name: str):
        self.name = name
        d = root / name
        self.x_raw = np.load(d / "x_raw.npy")
        self.y_raw = np.load(d / "y_raw.npy")
        self.z_raw = np.load(d / "z_raw.npy")
        self.rgb = np.load(d / "rgb.npy")
        self.label = np.load(d / "label.npy")
        self.row = np.load(d / "row.npy")
        subset_path = d / "subset.npy"
        self.subset = np.load(subset_path) if subset_path.exists() else None
        with open(d / "header.json", "r", encoding="utf-8") as f:
            hdr = json.load(f)
        self.scales = tuple(float(x) for x in hdr["scales"])
        self.offsets = tuple(float(x) for x in hdr["offsets"])
        self._x_m: np.ndarray | None = None
        self._y_m: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.x_raw)

    @property
    def x_m(self) -> np.ndarray:
        if self._x_m is None:
            self._x_m = self.x_raw.astype(np.float64) * self.scales[0] + self.offsets[0]
        return self._x_m

    @property
    def y_m(self) -> np.ndarray:
        if self._y_m is None:
            self._y_m = self.y_raw.astype(np.float64) * self.scales[1] + self.offsets[1]
        return self._y_m


def window_permutation(
    n: int,
    cx: float,
    cy: float,
    seed: int,
) -> np.ndarray:
    """Deterministic, label-independent permutation for one evaluation window.

    The cache stores all building points before all noise points.  Slicing a
    window in cache order therefore produces mostly pure-class 4096-point
    blocks, unlike the random sampling used in training.  The same rule must be
    used by validation and arbitrary-LAS inference, so the seed is derived from
    the window center rather than its enumeration index.
    """
    ix = int(round(float(cx) * 100.0))
    iy = int(round(float(cy) * 100.0))
    h = (ix * 73856093) ^ (iy * 19349663) ^ int(seed)
    rng = np.random.RandomState(np.uint32(h & 0xFFFFFFFF))
    return rng.permutation(int(n))


class WindowSet:
    def __init__(self, root: Path, name: str):
        d = root / "windows"
        self.name = name
        self.indices = np.load(d / f"{name}_indices.npy")
        self.offsets = np.load(d / f"{name}_offsets.npy")
        with open(d / f"{name}_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.meta: List[Dict[str, Any]] = meta["windows"]
        self.coverage = float(meta["coverage"])
        self.n_windows = int(meta["n_windows"])

    def __len__(self) -> int:
        return self.n_windows

    def get(self, i: int) -> np.ndarray:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.indices[a:b]

    def center(self, i: int) -> Tuple[float, float]:
        m = self.meta[i]
        return float(m["cx"]), float(m["cy"])


class BuildingData:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.root = cache_dir(cfg)
        with open(self.root / "manifest.json", "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.z_ref = float(self.manifest["z_ref"])
        self.rgb_scale = float(self.manifest["rgb_scale"])
        self.points_per_block = int(cfg["data"]["points_per_block"])
        self.window_size = float(cfg["data"]["window_size"])
        self.eval_perm_seed = int(cfg.get("eval", {}).get("perm_seed", 20261028))
        self._small: CacheSet | None = None
        self._test: CacheSet | None = None

    @property
    def small(self) -> CacheSet:
        if self._small is None:
            self._small = CacheSet(self.root, "small")
        return self._small

    @property
    def test(self) -> CacheSet:
        if self._test is None:
            self._test = CacheSet(self.root, "test")
        return self._test

    def windows(self, name: str) -> WindowSet:
        return WindowSet(self.root, name)

    def schedule(self) -> Tuple[np.ndarray, np.ndarray]:
        d = np.load(self.root / "windows" / "schedule.npz")
        return d["window_id"], d["block_seed"]


def _normalize(
    cache: CacheSet,
    indices: np.ndarray,
    cx: float,
    cy: float,
    z_ref: float,
    rgb_scale: float,
    window_size: float,
    rng: np.random.RandomState | None = None,
    translate_m: float = 0.0,
    rotate_z: bool = False,
    jitter_m: float = 0.0,
) -> np.ndarray:
    """Return [N,6] features for `indices` in one window."""
    x = cache.x_raw[indices].astype(np.float64) * cache.scales[0] + cache.offsets[0]
    y = cache.y_raw[indices].astype(np.float64) * cache.scales[1] + cache.offsets[1]
    z = cache.z_raw[indices].astype(np.float64) * cache.scales[2] + cache.offsets[2]

    dx = x - cx
    dy = y - cy
    dz = z - z_ref

    if rng is not None:
        if translate_m > 0:
            dx += rng.uniform(-translate_m, translate_m)
            dy += rng.uniform(-translate_m, translate_m)
        if rotate_z:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            dx, dy = dx * c - dy * s, dx * s + dy * c
        if jitter_m > 0:
            dx += rng.normal(0.0, jitter_m, size=dx.shape)
            dy += rng.normal(0.0, jitter_m, size=dy.shape)
            dz += rng.normal(0.0, jitter_m, size=dz.shape)

    feat = np.empty((len(indices), 6), dtype=np.float32)
    feat[:, 0] = dx / window_size
    feat[:, 1] = dy / window_size
    feat[:, 2] = dz / window_size
    feat[:, 3:6] = cache.rgb[indices].astype(np.float32) / rgb_scale
    return feat


class TrainSampler:
    """Deterministic schedule: (window_id, block_seed) -> one 4096-point block."""

    def __init__(self, data: BuildingData):
        self.data = data
        self.cfg = data.cfg
        self.windows = data.windows("train")
        self.window_id, self.block_seed = data.schedule()
        self.N = data.points_per_block
        self.window_size = data.window_size
        aug = self.cfg["train"]["augment"]
        self.rotate_z = bool(aug.get("rotate_z", True))
        self.translate_m = float(aug.get("translate_xy_m", 0.0))
        self.jitter_m = float(aug.get("jitter_m", 0.0))

    def __len__(self) -> int:
        return len(self.window_id)

    def sample(self, block_idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        wid = int(self.window_id[block_idx])
        seed = int(self.block_seed[block_idx])
        idx = self.windows.get(wid)
        rng = np.random.RandomState(seed)

        if len(idx) >= self.N:
            sel = rng.choice(idx, self.N, replace=False)
        else:
            sel = rng.choice(idx, self.N, replace=True)
        sel = sel.astype(np.int64, copy=False)

        cx, cy = self.windows.center(wid)
        feat = _normalize(
            self.data.small,
            sel,
            cx,
            cy,
            self.data.z_ref,
            self.data.rgb_scale,
            self.window_size,
            rng=rng,
            translate_m=self.translate_m,
            rotate_z=self.rotate_z,
            jitter_m=self.jitter_m,
        )
        labels = self.data.small.label[sel].astype(np.int32)
        return feat, labels, wid


def iter_eval_batches(
    data: BuildingData,
    set_name: str,
    batch_blocks: int = 8,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (features, labels, point_indices, valid_lengths) for a whole split.

    Each point appears once per window; overlapping windows are averaged by the
    caller.  The last chunk/batch is padded and `valid_lengths` tells the
    caller which rows to aggregate.
    """
    cache = data.test if set_name == "test" else data.small
    windows = data.windows(set_name)
    N = data.points_per_block

    for wi in range(len(windows)):
        idx = windows.get(wi)
        cx, cy = windows.center(wi)
        # Fixed, label-independent permutation before chunking.  Without it
        # the cache's class-contiguous order produces mostly pure-class blocks.
        idx = idx[window_permutation(len(idx), cx, cy, data.eval_perm_seed)]
        n = len(idx)
        start = 0
        while start < n:
            chunks: List[np.ndarray] = []
            valid: List[int] = []
            for b in range(batch_blocks):
                a = start + b * N
                if a >= n:
                    break
                chunk = idx[a:a + N].astype(np.int64, copy=False)
                valid.append(len(chunk))
                if len(chunk) < N:
                    pad = np.full(N - len(chunk), chunk[-1], dtype=chunk.dtype)
                    chunk = np.concatenate([chunk, pad])
                chunks.append(chunk)
            if not chunks:
                break
            while len(chunks) < batch_blocks:
                chunks.append(chunks[-1].copy())
                valid.append(0)

            feats = np.stack([
                _normalize(
                    cache, c, cx, cy,
                    data.z_ref, data.rgb_scale, data.window_size,
                    rng=None,
                )
                for c in chunks
            ])
            labels = np.stack([cache.label[c].astype(np.int32) for c in chunks])
            point_idx = np.stack(chunks)
            yield feats, labels, point_idx, np.asarray(valid, dtype=np.int32)
            start += batch_blocks * N
