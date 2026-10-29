#!/usr/bin/env python
"""Run the pilot or the full T-Net on/off experiment matrix."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = Path("/mnt/whu_stg/zxy/projects/pointcloud/PointNet/.venv")


def _env(gpu: int) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TF_DETERMINISTIC_OPS"] = "1"
    libs = glob.glob(str(VENV / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib"))
    if libs:
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(libs + ([old] if old else []))
    return env


def _run(cmd: list[str], gpu: int, dry_run: bool) -> int:
    print("$", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT, env=_env(gpu)).returncode


def _summarize(run_dirs: list[Path]) -> None:
    rows = []
    for d in run_dirs:
        p = d / "summary.json"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
        test = s.get("test_metrics") or {}
        val = s.get("best_val_metrics") or {}
        rows.append(
            {
                "run": d.name,
                "arm": s.get("arm"),
                "seed": s.get("seed"),
                "best_step": s.get("best_step"),
                "params": s.get("n_parameters"),
                "runtime_sec": s.get("runtime_sec"),
                "val_building_iou": val.get("building_iou"),
                "val_building_f1": val.get("building_f1"),
                "val_miou": val.get("miou"),
                "test_oa": test.get("oa"),
                "test_building_f1": test.get("building_f1"),
                "test_building_iou": test.get("building_iou"),
                "test_nonbuilding_f1": test.get("nonbuilding_f1"),
                "test_miou": test.get("miou"),
                "test_building_ap": test.get("building_ap"),
            }
        )
    if not rows:
        print("no summaries found")
        return
    out_csv = ROOT / "runs" / "summary.csv"
    out_md = ROOT / "runs" / "summary.md"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    keys = ["test_oa", "test_building_f1", "test_building_iou", "test_nonbuilding_f1", "test_miou", "test_building_ap"]
    lines = ["# Experiment summary", "", "| run | arm | seed | val bIoU | " + " | ".join(keys) + " |",
             "| --- | --- | ---: | ---: | " + " | ".join(["---:"] * len(keys)) + " |"]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['arm']} | {r['seed']} | {r['val_building_iou']} | "
            + " | ".join(str(r[k]) for k in keys) + " |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_csv} and {out_md}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--arms", nargs="+", default=["tnet_on", "tnet_off"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.pilot:
        seeds = args.seeds or [1234]
    else:
        seeds = args.seeds or [1234, 2345, 3456]

    started = []
    for arm in args.arms:
        for seed in seeds:
            config = ROOT / "configs" / f"{arm}.json"
            if not config.exists():
                raise FileNotFoundError(config)
            cmd = [
                str(VENV / "bin" / "python"), "-m", "building_pointnet.train",
                "--config", str(config),
                "--seed", str(seed),
                "--gpu", "0",
                "--run-root", str(ROOT / "runs"),
            ]
            if args.pilot:
                cmd.append("--pilot")
            if args.max_steps is not None:
                cmd.extend(["--max-steps", str(args.max_steps)])
            if args.val_every is not None:
                cmd.extend(["--val-every", str(args.val_every)])
            if args.no_test:
                cmd.append("--no-test")
            rc = _run(cmd, args.gpu, args.dry_run)
            if rc != 0:
                print(f"run failed: arm={arm} seed={seed} rc={rc}", file=sys.stderr)
                return
            started.append(ROOT / "runs")

    # summarize all run dirs that contain summary.json
    run_dirs = sorted([p for p in (ROOT / "runs").iterdir() if p.is_dir() and (p / "summary.json").exists()])
    _summarize(run_dirs)


if __name__ == "__main__":
    main()
