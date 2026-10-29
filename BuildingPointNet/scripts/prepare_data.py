#!/usr/bin/env python
"""Build the cache, spatial split, window index and training schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from building_pointnet.config import load_config, save_config  # noqa: E402
from building_pointnet.data_prep import prepare_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.json"))
    parser.add_argument("--save-config", default=str(ROOT / "cache" / "resolved_config.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_config(cfg, args.save_config)
    manifest = prepare_all(cfg)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
