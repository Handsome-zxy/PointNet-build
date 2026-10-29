"""Predict an arbitrary LAS/LAZ point cloud with a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from .config import load_config, resolve_path
from .dataset import BuildingData, window_permutation
from .evaluate import build_inference_ops, make_session
from .logging_utils import TeeLogger, write_json


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


def _window_indices(
    x: np.ndarray,
    y: np.ndarray,
    starts_x: List[float],
    starts_y: List[float],
    size: float,
) -> Tuple[List[np.ndarray], List[dict]]:
    windows: List[np.ndarray] = []
    meta: List[dict] = []
    for y0 in starts_y:
        for x0 in starts_x:
            mask = (x >= x0) & (x < x0 + size) & (y >= y0) & (y < y0 + size)
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                continue
            windows.append(idx)
            meta.append(
                {
                    "x0": float(x0),
                    "y0": float(y0),
                    "x1": float(x0 + size),
                    "y1": float(y0 + size),
                    "cx": float(x0 + size / 2.0),
                    "cy": float(y0 + size / 2.0),
                    "n_points": int(len(idx)),
                }
            )
    return windows, meta


def _ensure_coverage(
    x: np.ndarray,
    y: np.ndarray,
    windows: List[np.ndarray],
    meta: List[dict],
    size: float,
) -> None:
    covered = np.zeros(len(x), dtype=bool)
    for idx in windows:
        covered[idx] = True
    missing = np.flatnonzero(~covered)
    for li in missing:
        x0 = float(x[li] - size / 2.0)
        y0 = float(y[li] - size / 2.0)
        mask = (x >= x0) & (x < x0 + size) & (y >= y0) & (y < y0 + size)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            idx = np.array([li], dtype=np.int64)
        windows.append(idx)
        meta.append(
            {
                "x0": x0,
                "y0": y0,
                "x1": x0 + size,
                "y1": y0 + size,
                "cx": x0 + size / 2.0,
                "cy": y0 + size / 2.0,
                "n_points": int(len(idx)),
            }
        )
        covered[idx] = True
    if not covered.all():
        raise RuntimeError(f"{int((~covered).sum())} points remain uncovered")


def _predict(
    sess: tf.Session,
    ops,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rgb: np.ndarray,
    windows: List[np.ndarray],
    meta: List[dict],
    num_point: int,
    batch_size: int,
    window_size: float,
    z_ref: float,
    rgb_scale: float,
    perm_seed: int,
    log,
) -> np.ndarray:
    prob_sum = np.zeros(len(x), dtype=np.float64)
    count = np.zeros(len(x), dtype=np.int32)

    for wi, idx in enumerate(windows):
        cx = meta[wi]["cx"]
        cy = meta[wi]["cy"]
        # Same fixed, label-independent regrouping as validation.
        idx = idx[window_permutation(len(idx), cx, cy, perm_seed)]
        n = len(idx)
        for start in range(0, n, num_point * batch_size):
            chunks: List[np.ndarray] = []
            valid: List[int] = []
            for b in range(batch_size):
                a = start + b * num_point
                if a >= n:
                    break
                chunk = idx[a:a + num_point]
                valid.append(len(chunk))
                if len(chunk) < num_point:
                    pad = np.full(num_point - len(chunk), chunk[-1], dtype=chunk.dtype)
                    chunk = np.concatenate([chunk, pad])
                chunks.append(chunk)
            if not chunks:
                break
            while len(chunks) < batch_size:
                chunks.append(chunks[-1].copy())
                valid.append(0)

            feats = np.empty((len(chunks), num_point, 6), dtype=np.float32)
            for b, chunk in enumerate(chunks):
                feats[b, :, 0] = (x[chunk] - cx) / window_size
                feats[b, :, 1] = (y[chunk] - cy) / window_size
                feats[b, :, 2] = (z[chunk] - z_ref) / window_size
                feats[b, :, 3:6] = rgb[chunk].astype(np.float32) / rgb_scale

            probs = sess.run(
                ops["prob"],
                feed_dict={
                    ops["pointclouds_pl"]: feats,
                    ops["is_training_pl"]: False,
                },
            )
            for b, length in enumerate(valid):
                if length == 0:
                    continue
                ids = chunks[b][:length]
                prob_sum[ids] += probs[b, :length, 1]
                count[ids] += 1
        if (wi + 1) % 10 == 0 or wi == len(windows) - 1:
            log(f"window {wi + 1}/{len(windows)} n={n:,}")

    if np.any(count == 0):
        raise RuntimeError(f"{int((count == 0).sum())} points were not predicted")
    return (prob_sum / count).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--rgb-scale", type=float, default=None)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    import laspy

    cfg = load_config(args.config)
    data = BuildingData(cfg)
    z_ref = data.z_ref
    rgb_scale = float(args.rgb_scale if args.rgb_scale is not None else cfg["data"]["rgb_scale"])
    window_size = float(cfg["data"]["window_size"])
    stride = float(cfg["data"]["window_stride"])
    num_point = int(cfg["data"]["points_per_block"])
    output = resolve_path(cfg, args.output)
    log_dir = Path(args.log_dir) if args.log_dir else output.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log = TeeLogger(log_dir / "predict.log")
    started = time.time()

    las = laspy.read(args.input)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    if hasattr(las, "red"):
        rgb = np.stack([las.red, las.green, las.blue], axis=1).astype(np.uint16)
    else:
        rgb = np.zeros((len(x), 3), dtype=np.uint16)
        log("WARNING: input has no RGB; using zeros")

    starts_x = _starts(float(x.min()), float(x.max()) + 1e-6, window_size, stride)
    starts_y = _starts(float(y.min()), float(y.max()) + 1e-6, window_size, stride)
    windows, meta = _window_indices(x, y, starts_x, starts_y, window_size)
    _ensure_coverage(x, y, windows, meta, window_size)
    log(f"input points={len(x):,} windows={len(windows)}")

    with tf.Graph().as_default():
        ops = build_inference_ops(cfg, args.batch_size, num_point, gpu=args.gpu)
        sess = make_session(cfg)
        try:
            ops["saver"].restore(sess, args.checkpoint)
            probs = _predict(
                sess, ops, x, y, z, rgb, windows, meta,
                num_point, args.batch_size, window_size, z_ref, rgb_scale,
                int(cfg.get("eval", {}).get("perm_seed", 20261028)), log,
            )
        finally:
            sess.close()

    classification = (probs >= args.threshold).astype(np.uint8)
    las.classification = classification.astype(np.uint8)
    existing = {dim.name for dim in las.point_format.extra_dimensions}
    if "pred_prob" not in existing:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="pred_prob", type=np.float32,
                description="PointNet building probability",
            )
        )
    las["pred_prob"] = probs
    if "pred_building" not in existing:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="pred_building", type=np.uint8,
                description="1=building, 0=non-building",
            )
        )
    las["pred_building"] = classification
    output.parent.mkdir(parents=True, exist_ok=True)
    las.write(output)

    summary = {
        "input": str(Path(args.input).resolve()),
        "output": str(output),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "threshold": args.threshold,
        "n_points": int(len(x)),
        "n_building": int(classification.sum()),
        "n_nonbuilding": int((classification == 0).sum()),
        "prob_min": float(probs.min()),
        "prob_mean": float(probs.mean()),
        "prob_max": float(probs.max()),
        "windows": len(windows),
        "runtime_sec": time.time() - started,
    }
    write_json(output.with_suffix(".json"), summary)
    log(f"wrote {output}  building={summary['n_building']:,}  nonbuilding={summary['n_nonbuilding']:,}")
    log.close()


if __name__ == "__main__":
    main()
