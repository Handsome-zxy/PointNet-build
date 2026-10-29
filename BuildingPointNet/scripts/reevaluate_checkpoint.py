#!/usr/bin/env python
"""Re-evaluate a saved checkpoint with the corrected grouping and AP."""

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

from building_pointnet.config import load_config, save_config  # noqa: E402
from building_pointnet.dataset import BuildingData  # noqa: E402
from building_pointnet.evaluate import (  # noqa: E402
    build_inference_ops,
    make_session,
    predict_split,
)
from building_pointnet.logging_utils import collect_env, hash_files, write_json  # noqa: E402
from building_pointnet.metrics import best_threshold, binary_metrics  # noqa: E402


def _constant_baseline(labels: np.ndarray, value: int) -> dict:
    probs = np.full(len(labels), float(value), dtype=np.float32)
    return binary_metrics(labels, probs, threshold=0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (Path(cfg["_project_root"]) / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = BuildingData(cfg)
    num_point = int(cfg["data"]["points_per_block"])

    with tf.Graph().as_default():
        ops = build_inference_ops(cfg, args.batch_size, num_point, gpu=args.gpu)
        sess = make_session(cfg)
        try:
            ops["saver"].restore(sess, args.checkpoint)
            eval_idx, probs, count = predict_split(
                sess, ops, data, args.split, args.batch_size
            )
        finally:
            sess.close()

    cache = data.test if args.split == "test" else data.small
    labels = cache.label[eval_idx].astype(np.int32)
    metrics_05 = binary_metrics(labels, probs, threshold=args.threshold)

    tuned = None
    if args.split == "val":
        tuned_t, tuned_metrics = best_threshold(labels, probs, metric="miou")
        # AP is threshold-independent; keep it in the tuned metric row too.
        tuned_metrics["building_ap"] = metrics_05["building_ap"]
        tuned = {"threshold": float(tuned_t), "metrics": tuned_metrics}

    baseline_building = _constant_baseline(labels, 1)
    baseline_nonbuilding = _constant_baseline(labels, 0)

    np.save(out_dir / "prob.npy", probs)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "eval_indices.npy", eval_idx)
    save_config(cfg, out_dir / "config.json")
    write_json(out_dir / "env.json", collect_env(cfg, " ".join(sys.argv)))
    write_json(
        out_dir / "code_hashes.json",
        hash_files(
            [Path(cfg["_config_path"]), Path(cfg["_project_root"]) / "cache" / "manifest.json"]
            + sorted((Path(cfg["_project_root"]) / "src" / "building_pointnet").glob("*.py"))
        ),
    )

    result = {
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "perm_seed": data.eval_perm_seed,
        "n_eval_points": int(len(eval_idx)),
        "avg_hits_per_point": float(count.mean()),
        "metrics_at_threshold": metrics_05,
        "threshold": args.threshold,
        "tuned_on_val": tuned,
        "constant_building_baseline": baseline_building,
        "constant_nonbuilding_baseline": baseline_nonbuilding,
    }
    write_json(out_dir / "reeval.json", result)

    lines = [
        f"# Re-evaluation: {args.split}",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- perm_seed: {data.eval_perm_seed}",
        f"- points: {len(eval_idx):,}",
        "",
        "| protocol | threshold | OA | building F1 | building IoU | non-building F1 | non-building IoU | mIoU | building AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = [("0.5", args.threshold, metrics_05),
            ("constant building", 0.5, baseline_building),
            ("constant non-building", 0.5, baseline_nonbuilding)]
    if tuned:
        rows.insert(1, ("tuned mIoU", tuned["threshold"], tuned["metrics"]))
    for name, t, m in rows:
        lines.append(
            f"| {name} | {t:.2f} | {m['oa']:.6f} | {m['building_f1']:.6f} | "
            f"{m['building_iou']:.6f} | {m['nonbuilding_f1']:.6f} | "
            f"{m['nonbuilding_iou']:.6f} | {m['miou']:.6f} | "
            f"{m.get('building_ap', float('nan')):.6f} |"
        )
    lines += ["", "Note: AP uses sklearn / tie-grouped average precision; evaluation chunks use the fixed, label-independent window permutation.", ""]
    (out_dir / "reeval.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
