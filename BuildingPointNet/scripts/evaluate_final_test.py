#!/usr/bin/env python
"""Evaluate a frozen final-test LAS with a frozen checkpoint/threshold.

This script never tunes a threshold on the test set.  The threshold must come
from validation.  It runs `building_pointnet.predict` to produce per-point
predictions, then scores them against the candidate LAS `label` field.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from building_pointnet.config import load_config, resolve_path  # noqa: E402
from building_pointnet.logging_utils import sha256_file, write_json  # noqa: E402
from building_pointnet.metrics import binary_metrics, confusion_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-las", required=True)
    parser.add_argument("--candidate-meta", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--allow-unfrozen", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve_path(cfg, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.candidate_meta)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not meta.get("frozen", False) and not args.allow_unfrozen:
        raise SystemExit(
            f"candidate is not frozen ({meta_path}); freeze it first or pass --allow-unfrozen"
        )

    candidate = str(resolve_path(cfg, args.candidate_las))
    pred_path = out_dir / "final_test_pred.las"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        sys.executable, "-m", "building_pointnet.predict",
        "--config", str(Path(cfg["_config_path"])),
        "--checkpoint", args.checkpoint,
        "--input", candidate,
        "--output", str(pred_path),
        "--threshold", str(args.threshold),
        "--batch-size", str(args.batch_size),
        "--gpu", str(args.gpu),
    ]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    import laspy

    inp = laspy.read(candidate)
    pred_las = laspy.read(pred_path)
    labels = np.asarray(inp["label"]).astype(np.int32)
    probs = np.asarray(pred_las["pred_prob"]).astype(np.float32)
    if len(labels) != len(probs):
        raise RuntimeError("candidate and prediction point counts differ")
    if "pred_building" in {d.name for d in pred_las.point_format.extra_dimensions}:
        pred = np.asarray(pred_las["pred_building"]).astype(np.int32)
    else:
        pred = (probs >= args.threshold).astype(np.int32)

    metrics_main = binary_metrics(labels, probs, threshold=args.threshold)
    metrics_05 = binary_metrics(labels, probs, threshold=0.5)
    cm = confusion_matrix(labels, probs, threshold=args.threshold)
    const_b = binary_metrics(labels, np.ones_like(probs), threshold=0.5)

    np.save(out_dir / "final_test_prob.npy", probs)
    np.save(out_dir / "final_test_labels.npy", labels)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header="pred_0,pred_1", comments="")
    result = {
        "candidate_las": candidate,
        "candidate_sha256": sha256_file(candidate),
        "prediction_las": str(pred_path),
        "prediction_sha256": sha256_file(pred_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "threshold_from_validation": args.threshold,
        "metrics_at_validation_threshold": metrics_main,
        "metrics_at_0.5": metrics_05,
        "constant_building_baseline": const_b,
        "confusion_matrix": cm.tolist(),
        "candidate_meta": str(meta_path.resolve()),
        "candidate_frozen": bool(meta.get("frozen", False)),
    }
    write_json(out_dir / "final_test_metrics.json", result)
    lines = [
        "# Final test evaluation",
        "",
        f"- candidate: `{candidate}`",
        f"- sha256: `{result['candidate_sha256']}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- threshold from validation: {args.threshold:.2f}",
        f"- points: {len(labels):,}",
        "",
        "| protocol | OA | building F1 | building IoU | non-building F1 | non-building IoU | mIoU | building AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in [("validation threshold", metrics_main), ("0.5", metrics_05), ("constant building", const_b)]:
        lines.append(
            f"| {name} | {m['oa']:.6f} | {m['building_f1']:.6f} | {m['building_iou']:.6f} | "
            f"{m['nonbuilding_f1']:.6f} | {m['nonbuilding_iou']:.6f} | {m['miou']:.6f} | {m['building_ap']:.6f} |"
        )
    lines += ["", "Threshold is not tuned on this test set.", ""]
    (out_dir / "final_test_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("metrics_at_validation_threshold", "metrics_at_0.5", "constant_building_baseline")}, indent=2))


if __name__ == "__main__":
    main()
