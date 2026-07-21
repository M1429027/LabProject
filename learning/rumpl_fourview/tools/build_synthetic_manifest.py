from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manifest from generated rumpl_fourview samples.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--train-prefix", default="train")
    parser.add_argument("--val-prefix", default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    entries = []
    for sample_path in sorted(input_dir.rglob("*.npz")):
        split = "train"
        if args.val_prefix and args.val_prefix in sample_path.parts:
            split = "val"
        entry = {
            "id": sample_path.stem,
            "split": split,
            "path": str(sample_path),
            "num_views": 4,
            "num_joints": 17,
            "feature_dim": 7,
        }
        entries.append(entry)
    payload = {
        "schema_version": "rumpl_fourview_manifest_v1",
        "entries": entries,
    }
    with open(args.output_manifest, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote manifest with {len(entries)} entries to {args.output_manifest}")


if __name__ == "__main__":
    main()
