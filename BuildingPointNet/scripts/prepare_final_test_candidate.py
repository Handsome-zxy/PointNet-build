#!/usr/bin/env python
"""Build and verify a spatially independent final-test candidate.

Candidate (original coordinates):
    X [405185, 405262)
    Y [3112900, 3112968)

It is south of the big_validation area (X starts 405543.57) and far north of
the small training tile (Y max 3112732.19).  Labels come from the big files:
build_big = 1, noise_big = 0.  The candidate is NOT frozen here; the metadata
records the verification and `frozen=false`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label as cc_label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from building_pointnet.config import load_config, resolve_path  # noqa: E402
from building_pointnet.las_utils import iter_chunks, point_keys, read_header  # noqa: E402
from building_pointnet.logging_utils import sha256_file, write_json  # noqa: E402

BOX = (405185.0, 405262.0, 3112900.0, 3112968.0)
ANALYSIS = (405155.0, 405292.0, 3112870.0, 3112998.0)
TRAIN_SMALL_YMAX = 3112732.19
OLD_TEST_XMIN = 405543.57


def load_build_analysis(path: Path):
    h = read_header(path)
    xs, ys, zs, rgb = [], [], [], []
    for ch in iter_chunks(path):
        xm = ch["x"].astype(np.float64) * h.scales[0] + h.offsets[0]
        ym = ch["y"].astype(np.float64) * h.scales[1] + h.offsets[1]
        keep = (
            (xm >= ANALYSIS[0]) & (xm < ANALYSIS[1])
            & (ym >= ANALYSIS[2]) & (ym < ANALYSIS[3])
        )
        if keep.any():
            xs.append(ch["x"][keep]); ys.append(ch["y"][keep])
            zs.append(ch["z"][keep]); rgb.append(ch["rgb"][keep])
    return h, *(np.concatenate(v) for v in (xs, ys, zs, rgb))


def load_noise_candidate(path: Path):
    h = read_header(path)
    xs, ys, zs, rgb = [], [], [], []
    for ch in iter_chunks(path):
        xm = ch["x"].astype(np.float64) * h.scales[0] + h.offsets[0]
        ym = ch["y"].astype(np.float64) * h.scales[1] + h.offsets[1]
        keep = (xm >= BOX[0]) & (xm < BOX[1]) & (ym >= BOX[2]) & (ym < BOX[3])
        if keep.any():
            xs.append(ch["x"][keep]); ys.append(ch["y"][keep])
            zs.append(ch["z"][keep]); rgb.append(ch["rgb"][keep])
    return h, *(np.concatenate(v) for v in (xs, ys, zs, rgb))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/base.json"))
    parser.add_argument("--out-las", default=str(ROOT / "cache/final_test_candidate_2026-10-28.las"))
    parser.add_argument("--out-meta", default=str(ROOT / "reports/final_test_candidate_2026-10-28.json"))
    parser.add_argument("--out-md", default=str(ROOT / "reports/final_test_candidate_2026-10-28.md"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = resolve_path(cfg, Path(cfg["data"]["raw_dir"]))
    build_path = raw / cfg["data"]["build_big"]
    noise_path = raw / cfg["data"]["noise_big"]

    print("loading build_big analysis region ...", flush=True)
    hb, xb, yb, zb, rgbb = load_build_analysis(build_path)
    xbm = xb.astype(np.float64) * hb.scales[0] + hb.offsets[0]
    ybm = yb.astype(np.float64) * hb.scales[1] + hb.offsets[1]
    in_box_build = (
        (xbm >= BOX[0]) & (xbm < BOX[1]) & (ybm >= BOX[2]) & (ybm < BOX[3])
    )

    print("loading noise_big candidate region ...", flush=True)
    hn, xn, yn, zn, rgbn = load_noise_candidate(noise_path)

    # 0.5 m building occupancy and connected components in the analysis region.
    res = 0.5
    nx = int(np.ceil((ANALYSIS[1] - ANALYSIS[0]) / res))
    ny = int(np.ceil((ANALYSIS[3] - ANALYSIS[2]) / res))
    mask = np.zeros((ny, nx), dtype=bool)
    ix = ((xbm - ANALYSIS[0]) / res).astype(int)
    iy = ((ybm - ANALYSIS[2]) / res).astype(int)
    mask[iy, ix] = True
    lab, ncomp = cc_label(mask, structure=np.ones((3, 3), dtype=int))

    components = []
    cuts = []
    for cid in range(1, ncomp + 1):
        sel = lab[iy, ix] == cid
        if not sel.any():
            continue
        xs = xbm[sel]; ys = ybm[sel]; zs = zb[sel].astype(np.float64) * hb.scales[2] + hb.offsets[2]
        inside = (xs >= BOX[0]) & (xs < BOX[1]) & (ys >= BOX[2]) & (ys < BOX[3])
        rec = {
            "component": int(cid),
            "points": int(len(xs)),
            "grid_cells": int((lab == cid).sum()),
            "bbox": [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())],
            "z_range": [float(zs.min()), float(zs.max())],
            "inside_candidate": int(inside.sum()),
            "outside_candidate": int((~inside).sum()),
        }
        components.append(rec)
        if inside.any() and (~inside).any():
            cuts.append(rec)

    # Duplicates and cross-class coordinate conflicts in the candidate.
    kb = point_keys(xb[in_box_build], yb[in_box_build], zb[in_box_build])
    kn = point_keys(xn, yn, zn)
    build_unique = np.unique(kb)
    cross = int(np.isin(kn, build_unique).sum())
    stats = {
        "candidate_box": list(BOX),
        "analysis_box": list(ANALYSIS),
        "building_points": int(in_box_build.sum()),
        "nonbuilding_points": int(len(xn)),
        "total_points": int(in_box_build.sum() + len(xn)),
        "building_fraction": float(in_box_build.sum() / max(1, in_box_build.sum() + len(xn))),
        "building_unique": int(len(build_unique)),
        "building_duplicates": int(len(kb) - len(build_unique)),
        "noise_unique": int(len(np.unique(kn))),
        "noise_duplicates": int(len(kn) - len(np.unique(kn))),
        "cross_class_conflicts": cross,
        "components_total": int(ncomp),
        "components_intersecting_candidate": int(sum(1 for c in components if c["inside_candidate"] > 0)),
        "components_cut": cuts,
        "spatial_gap_to_train_small_m": float(BOX[2] - TRAIN_SMALL_YMAX),
        "spatial_gap_to_old_test_m": float(OLD_TEST_XMIN - BOX[1]),
        "label_source": {"building": str(build_path.name), "nonbuilding": str(noise_path.name)},
    }
    checks = {
        "both_classes_present": stats["building_points"] > 0 and stats["nonbuilding_points"] > 0,
        "no_cut_buildings": len(cuts) == 0,
        "no_cross_class_conflicts": cross == 0,
        "spatially_disjoint_from_train": stats["spatial_gap_to_train_small_m"] > 0,
        "spatially_disjoint_from_old_test": stats["spatial_gap_to_old_test_m"] > 0,
    }
    ready = all(checks.values())
    stats["checks"] = checks
    stats["ready_for_freeze"] = bool(ready)
    stats["frozen"] = False

    # Write candidate LAS (building first, then noise), preserving CRS/scale.
    import laspy

    src_header = laspy.open(build_path).header
    hdr = laspy.LasHeader(point_format=7, version="1.4")
    hdr.scales = src_header.scales
    hdr.offsets = src_header.offsets
    # Copy the projection VLR without requiring pyproj; copying all VLRs would
    # also copy the source ExtraBytes VLR and conflict with our new fields.
    for vlr in src_header.vlrs:
        if getattr(vlr, "user_id", None) == "LASF_Projection":
            hdr.vlrs.append(vlr)
    las = laspy.LasData(hdr)

    x_all = np.concatenate([xb[in_box_build], xn])
    y_all = np.concatenate([yb[in_box_build], yn])
    z_all = np.concatenate([zb[in_box_build], zn])
    rgb_all = np.concatenate([rgbb[in_box_build], rgbn])
    label_all = np.concatenate([
        np.ones(int(in_box_build.sum()), dtype=np.uint8),
        np.zeros(len(xn), dtype=np.uint8),
    ])
    source_all = np.concatenate([
        np.zeros(int(in_box_build.sum()), dtype=np.uint8),
        np.ones(len(xn), dtype=np.uint8),
    ])
    las.x = x_all.astype(np.float64) * hb.scales[0] + hb.offsets[0]
    las.y = y_all.astype(np.float64) * hb.scales[1] + hb.offsets[1]
    las.z = z_all.astype(np.float64) * hb.scales[2] + hb.offsets[2]
    las.red = rgb_all[:, 0]
    las.green = rgb_all[:, 1]
    las.blue = rgb_all[:, 2]
    las.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.uint8, description="0=non-building, 1=building"))
    las.add_extra_dim(laspy.ExtraBytesParams(name="source", type=np.uint8, description="0=build 1=noise"))
    las["label"] = label_all
    las["source"] = source_all
    out_las = Path(args.out_las)
    out_las.parent.mkdir(parents=True, exist_ok=True)
    las.write(out_las)
    stats["las_path"] = str(out_las)
    stats["las_sha256"] = sha256_file(out_las)

    write_json(Path(args.out_meta), stats)

    lines = [
        "# Final test candidate 2026-10-28",
        "",
        f"- candidate box: X [{BOX[0]:.2f}, {BOX[1]:.2f}) Y [{BOX[2]:.2f}, {BOX[3]:.2f})",
        f"- points: {stats['total_points']:,} (building {stats['building_points']:,}, non-building {stats['nonbuilding_points']:,})",
        f"- building fraction: {stats['building_fraction']:.4f}",
        f"- spatial gap to small training tile: {stats['spatial_gap_to_train_small_m']:.2f} m",
        f"- spatial gap to old big_validation: {stats['spatial_gap_to_old_test_m']:.2f} m",
        f"- duplicates: building {stats['building_duplicates']:,}, noise {stats['noise_duplicates']:,}",
        f"- cross-class conflicts: {stats['cross_class_conflicts']:,}",
        f"- components intersecting candidate: {stats['components_intersecting_candidate']}",
        f"- cut components: {len(cuts)}",
        f"- ready_for_freeze: {ready}",
        f"- frozen: false",
        "",
        "## Checks",
        "",
    ]
    for k, v in checks.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Components", "", "| component | points | bbox X | bbox Y | Z range | inside | outside |", "| ---: | ---: | --- | --- | --- | ---: | ---: |"]
    for c in components:
        if c["inside_candidate"] == 0:
            continue
        lines.append(
            f"| {c['component']} | {c['points']:,} | [{c['bbox'][0]:.1f}, {c['bbox'][1]:.1f}] | "
            f"[{c['bbox'][2]:.1f}, {c['bbox'][3]:.1f}] | [{c['z_range'][0]:.1f}, {c['z_range'][1]:.1f}] | "
            f"{c['inside_candidate']:,} | {c['outside_candidate']:,} |"
        )
    lines += ["", f"- LAS: `{out_las}`", f"- SHA256: `{stats['las_sha256']}`", ""]
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if not ready:
        raise SystemExit("candidate checks failed; adjust the box before freezing")


if __name__ == "__main__":
    main()
