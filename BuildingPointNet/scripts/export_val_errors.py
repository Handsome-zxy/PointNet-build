#!/usr/bin/env python
"""Export validation errors as a LAS and a top-down PNG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from building_pointnet.config import load_config, resolve_path  # noqa: E402
from building_pointnet.dataset import BuildingData, window_permutation  # noqa: E402
from building_pointnet.evaluate import (  # noqa: E402
    build_inference_ops,
    make_session,
    predict_split,
)
from building_pointnet.logging_utils import sha256_file, write_json  # noqa: E402
from building_pointnet.metrics import best_threshold, binary_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve_path(cfg, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = BuildingData(cfg)
    N = int(cfg["data"]["points_per_block"])

    with tf.Graph().as_default():
        ops = build_inference_ops(cfg, args.batch_size, N, gpu=args.gpu)
        sess = make_session(cfg)
        try:
            ops["saver"].restore(sess, args.checkpoint)
            eval_idx, probs, count = predict_split(sess, ops, data, "val", args.batch_size)
        finally:
            sess.close()

    labels = data.small.label[eval_idx].astype(np.int32)
    tuned_t, tuned = best_threshold(labels, probs, metric="miou")
    threshold = float(tuned_t)
    metrics = binary_metrics(labels, probs, threshold=threshold)
    pred = (probs >= threshold).astype(np.uint8)

    err = np.zeros(len(labels), dtype=np.uint8)
    err[(labels == 0) & (pred == 1)] = 1  # false positive
    err[(labels == 1) & (pred == 0)] = 2  # false negative

    # Error LAS, using the small-file header for CRS/scale/offset.
    import laspy

    src = laspy.open(resolve_path(cfg, Path(cfg["data"]["raw_dir"]) / cfg["data"]["build_small"])).header
    hdr = laspy.LasHeader(point_format=7, version="1.4")
    hdr.scales = src.scales
    hdr.offsets = src.offsets
    for vlr in src.vlrs:
        if getattr(vlr, "user_id", None) == "LASF_Projection":
            hdr.vlrs.append(vlr)
    las = laspy.LasData(hdr)
    xraw = data.small.x_raw[eval_idx]
    yraw = data.small.y_raw[eval_idx]
    zraw = data.small.z_raw[eval_idx]
    rgb = data.small.rgb[eval_idx]
    las.x = xraw.astype(np.float64) * data.small.scales[0] + data.small.offsets[0]
    las.y = yraw.astype(np.float64) * data.small.scales[1] + data.small.offsets[1]
    las.z = zraw.astype(np.float64) * data.small.scales[2] + data.small.offsets[2]
    las.red = rgb[:, 0]; las.green = rgb[:, 1]; las.blue = rgb[:, 2]
    for name, dtype, desc in [
        ("label", np.uint8, "0=non-building, 1=building"),
        ("pred_label", np.uint8, "0=non-building, 1=building"),
        ("pred_prob", np.float32, "building probability"),
        ("error_code", np.uint8, "0=correct, 1=FP, 2=FN"),
    ]:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype, description=desc))
    las["label"] = labels.astype(np.uint8)
    las["pred_label"] = pred
    las["pred_prob"] = probs.astype(np.float32)
    las["error_code"] = err
    las_path = out_dir / "val_errors.las"
    las.write(las_path)

    # Per-window errors.
    windows = data.windows("val")
    pos = np.full(len(data.small), -1, dtype=np.int64)
    pos[eval_idx] = np.arange(len(eval_idx), dtype=np.int64)
    per_window = []
    for wi in range(len(windows)):
        idx = windows.get(wi)
        idx = idx[window_permutation(len(idx), windows.center(wi)[0], windows.center(wi)[1], data.eval_perm_seed)]
        local = pos[idx]
        e = err[local]
        per_window.append({
            "window": wi,
            "points": int(len(idx)),
            "fp": int((e == 1).sum()),
            "fn": int((e == 2).sum()),
            "error_rate": float((e > 0).mean()),
        })

    # Top-down image: gray=correct, red=FP, blue=FN.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xm = las.x; ym = las.y
    x0, x1 = float(np.min(xm)), float(np.max(xm))
    y0, y1 = float(np.min(ym)), float(np.max(ym))
    res = 0.25
    nx = int(np.ceil((x1 - x0) / res)) + 1
    ny = int(np.ceil((y1 - y0) / res)) + 1
    ix = ((xm - x0) / res).astype(int)
    iy = ((ym - y0) / res).astype(int)
    counts = np.zeros((3, ny, nx), dtype=np.float64)
    for code in (0, 1, 2):
        m = err == code
        np.add.at(counts[code], (iy[m], ix[m]), 1)
    img = np.zeros((ny, nx, 3))
    gray = np.clip(np.log1p(counts[0]) / max(1.0, np.log1p(counts[0]).max()), 0, 1) * 0.55
    img[..., 0] = gray
    img[..., 1] = gray
    img[..., 2] = gray
    img[..., 0] = np.maximum(img[..., 0], np.clip(np.log1p(counts[1]) / max(1.0, np.log1p(counts[1]).max()), 0, 1))
    img[..., 2] = np.maximum(img[..., 2], np.clip(np.log1p(counts[2]) / max(1.0, np.log1p(counts[2]).max()), 0, 1))
    plt.figure(figsize=(12, 9))
    plt.imshow(img, origin="lower")
    plt.title(
        f"validation errors @ t={threshold:.2f}: red=FP ({int((err==1).sum()):,}), "
        f"blue=FN ({int((err==2).sum()):,}), gray=correct"
    )
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    png_path = out_dir / "val_errors.png"
    plt.savefig(png_path, dpi=150)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "threshold": threshold,
        "metrics": metrics,
        "n_points": int(len(labels)),
        "fp": int((err == 1).sum()),
        "fn": int((err == 2).sum()),
        "error_rate": float((err > 0).mean()),
        "per_window": per_window,
        "las_path": str(las_path),
        "las_sha256": sha256_file(las_path),
        "png_path": str(png_path),
    }
    write_json(out_dir / "val_errors.json", summary)
    (out_dir / "val_errors.md").write_text(
        "\n".join([
            "# Validation error export",
            "",
            f"- checkpoint: `{args.checkpoint}`",
            f"- threshold (val mIoU): {threshold:.2f}",
            f"- points: {len(labels):,}",
            f"- FP: {summary['fp']:,}; FN: {summary['fn']:,}; error rate: {summary['error_rate']:.6f}",
            f"- mIoU: {metrics['miou']:.6f}; building F1: {metrics['building_f1']:.6f}; non-building F1: {metrics['nonbuilding_f1']:.6f}",
            f"- LAS: `{las_path}`",
            f"- PNG: `{png_path}`",
            "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_window"}, indent=2))


if __name__ == "__main__":
    main()
