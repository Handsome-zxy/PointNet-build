#!/usr/bin/env python
"""Audit class composition of evaluation chunks before/after regrouping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from building_pointnet.config import load_config  # noqa: E402
from building_pointnet.dataset import BuildingData, window_permutation  # noqa: E402


def audit_split(data: BuildingData, split: str) -> dict:
    cache = data.test if split == "test" else data.small
    windows = data.windows(split)
    N = data.points_per_block
    result = {"split": split, "windows": len(windows), "points_per_block": N}
    for mode in ("cache_order", "fixed_permutation"):
        pure_b = pure_n = mixed = chunks = 0
        building_frac = []
        for wi in range(len(windows)):
            idx = windows.get(wi)
            cx, cy = windows.center(wi)
            if mode == "fixed_permutation":
                idx = idx[window_permutation(len(idx), cx, cy, data.eval_perm_seed)]
            labels = cache.label[idx]
            for a in range(0, len(idx), N):
                lab = labels[a:a + N]
                lab = lab[lab >= 0]
                if len(lab) == 0:
                    continue
                chunks += 1
                nb = int((lab == 1).sum())
                building_frac.append(nb / len(lab))
                if nb == len(lab):
                    pure_b += 1
                elif nb == 0:
                    pure_n += 1
                else:
                    mixed += 1
        result[mode] = {
            "chunks": chunks,
            "pure_building": pure_b,
            "pure_nonbuilding": pure_n,
            "mixed": mixed,
            "mean_building_fraction": float(np.mean(building_frac)) if building_frac else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/base.json"))
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--out", default=str(ROOT / "reports/eval_grouping_audit.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = BuildingData(cfg)
    out = {"perm_seed": data.eval_perm_seed, "splits": {}}
    for split in args.splits:
        out["splits"][split] = audit_split(data, split)
        print(split, json.dumps(out["splits"][split], indent=2))

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
