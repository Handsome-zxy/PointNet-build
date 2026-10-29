#!/usr/bin/env python
"""Freeze evaluation/training protocol and source/config/data hashes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from building_pointnet.config import load_config  # noqa: E402
from building_pointnet.logging_utils import hash_files, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="2026-10-28")
    parser.add_argument("--out-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()

    base = load_config(ROOT / "configs/base.json")
    manifest_path = ROOT / "cache" / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    paths = [
        ROOT / "configs/base.json",
        ROOT / "configs/tnet_on.json",
        ROOT / "configs/tnet_off.json",
        manifest_path,
        ROOT / "DESIGN.md",
        ROOT / "README.md",
        ROOT / "reports/test_set_usage.md",
    ]
    paths += sorted((ROOT / "src/building_pointnet").glob("*.py"))
    paths += sorted((ROOT / "scripts").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("*.py"))

    freeze = {
        "protocol_version": "BuildingPointNet-v1-review-corrected",
        "created_at": datetime.now().isoformat(),
        "project_root": str(ROOT),
        "training": {
            "max_steps": int(base["train"]["max_steps"]),
            "batch_size": int(base["train"]["batch_size"]),
            "points_per_block": int(base["data"]["points_per_block"]),
            "window_size_m": float(base["data"]["window_size"]),
            "window_stride_m": float(base["data"]["window_stride"]),
            "class_weight_mode": base["train"].get("class_weight_mode"),
            "optimizer": "Adam",
            "learning_rate": float(base["train"]["learning_rate"]),
            "warmup_steps": int(base["train"]["warmup_steps"]),
            "end_learning_rate": float(base["train"]["end_learning_rate"]),
            "val_every_steps": int(base["train"]["val_every_steps"]),
            "early_stop_patience": int(base["train"]["early_stop_patience"]),
        },
        "evaluation": {
            "perm_seed": int(base["eval"].get("perm_seed", 20261028)),
            "chunking": "fixed label-independent permutation per window, then 4096-point chunks",
            "threshold_grid": [0.01, 0.99, 0.01],
            "threshold_selection": "validation mIoU",
            "checkpoint_selection": "validation mIoU at tuned threshold",
            "ap": "sklearn average_precision_score / tie-grouped fallback",
            "report_metrics": ["OA", "balanced_accuracy", "building F1/IoU", "non-building F1/IoU", "macro F1", "mIoU", "AP", "confusion matrix"],
        },
        "data": {
            "manifest_hash": hash_files([manifest_path])[str(manifest_path.resolve())],
            "counts": manifest["counts"],
            "split_counts": manifest["split_counts"],
            "test_label_counts": manifest["test_label_counts"],
            "z_ref": manifest["z_ref"],
            "class_weights": manifest["class_weights"],
            "class_weight_mode": manifest.get("class_weight_mode"),
            "val_box": manifest["val_box"],
            "buffer_box": manifest["buffer_box"],
        },
        "test_usage": {
            "strict_unseen": False,
            "development_seen_runs": ["20261028_135939_tnet_on_s1234_pilot"],
            "policy": "training does not evaluate test by default; final test only once after this freeze, not for tuning",
            "note": "A strict unseen test would require freezing a new independent region; pending user approval.",
        },
        "code_hashes": hash_files(paths),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"protocol_freeze_{args.tag}.json"
    write_json(json_path, freeze)

    lines = [
        f"# Protocol freeze {args.tag}",
        "",
        f"- protocol: `{freeze['protocol_version']}`",
        f"- created: {freeze['created_at']}",
        f"- eval perm seed: {freeze['evaluation']['perm_seed']}",
        f"- checkpoint selection: {freeze['evaluation']['checkpoint_selection']}",
        f"- threshold selection: {freeze['evaluation']['threshold_selection']}",
        f"- test strict unseen: {freeze['test_usage']['strict_unseen']}",
        f"- data manifest hash: `{freeze['data']['manifest_hash']}`",
        "",
        "## Code hashes",
        "",
        "| file | sha256 |",
        "| --- | --- |",
    ]
    for p, h in sorted(freeze["code_hashes"].items()):
        lines.append(f"| `{Path(p).relative_to(ROOT)}` | `{h}` |")
    lines.append("")
    (out_dir / f"protocol_freeze_{args.tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
