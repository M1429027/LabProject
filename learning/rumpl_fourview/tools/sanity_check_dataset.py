from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate rumpl_fourview generated samples and manifest shapes.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("entries", [])
    if not entries:
        raise ValueError("Manifest is empty.")
    checked = 0
    for entry in entries[: args.limit]:
        sample = np.load(Path(entry["path"]))
        ray_tokens = sample["ray_tokens"]
        view_mask = sample["view_mask"]
        target_3d = sample["target_3d"]
        if ray_tokens.shape != (17, 4, 7):
            raise ValueError(f"Unexpected ray_tokens shape for {entry['id']}: {ray_tokens.shape}")
        if view_mask.shape != (4,):
            raise ValueError(f"Unexpected view_mask shape for {entry['id']}: {view_mask.shape}")
        if target_3d.shape != (17, 3):
            raise ValueError(f"Unexpected target_3d shape for {entry['id']}: {target_3d.shape}")
        checked += 1
    print(f"Sanity check passed for {checked} samples from {args.manifest}")


if __name__ == "__main__":
    main()
