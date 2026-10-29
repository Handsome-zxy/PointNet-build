#!/usr/bin/env python
"""Freeze the verified final-test candidate metadata.

Run only after all models/thresholds are selected.  The script does not modify
the candidate LAS; it records the frozen status and hashes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from building_pointnet.logging_utils import sha256_file, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--las", required=True)
    parser.add_argument("--tag", default="2026-10-28")
    args = parser.parse_args()

    meta_path = Path(args.meta)
    las_path = Path(args.las)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not meta.get("ready_for_freeze", False):
        raise SystemExit("candidate is not ready_for_freeze")
    if meta.get("frozen", False):
        raise SystemExit("candidate is already frozen")

    meta["frozen"] = True
    meta["status"] = "frozen"
    meta["frozen_at"] = datetime.now().isoformat()
    meta["las_sha256"] = sha256_file(las_path)
    write_json(meta_path, meta)

    md = Path(str(meta_path).replace(".json", "_FROZEN.md"))
    md.write_text(
        "\n".join([
            "# Final test candidate frozen",
            "",
            f"- frozen_at: {meta['frozen_at']}",
            f"- candidate: `{las_path}`",
            f"- sha256: `{meta['las_sha256']}`",
            f"- box: {meta['candidate_box']}",
            f"- points: {meta['total_points']:,}",
            f"- building fraction: {meta['building_fraction']:.4f}",
            f"- checks: {json.dumps(meta['checks'], ensure_ascii=False)}",
            "",
            "Do not use this test set for model/threshold selection.",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"frozen {meta_path}; sha256={meta['las_sha256']}")


if __name__ == "__main__":
    main()
