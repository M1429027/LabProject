"""Sanity check Stage A dataset manifests and dataloader shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import KarateStageADataset, build_stage_a_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Stage A dataset manifest and batch shapes.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = {
        "data": {
            "manifest_path": str(manifest_path),
            "batch_size": int(args.batch_size),
            "num_workers": 0,
            "train_split": "train",
            "val_split": "val",
        }
    }
    train_loader, val_loader = build_stage_a_dataloaders(config)
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_count = 0
    try:
        test_count = len(KarateStageADataset(manifest_path, split="test"))
    except ValueError:
        test_count = 0
    report = {
        "manifest": str(manifest_path),
        "summary": manifest.get("summary", {}),
        "train_count": len(train_loader.dataset),
        "val_count": len(val_loader.dataset),
        "test_count": test_count,
        "train_batch_shapes": {
            key: list(value.shape)
            for key, value in train_batch.items()
            if hasattr(value, "shape")
        },
        "val_batch_shapes": {
            key: list(value.shape)
            for key, value in val_batch.items()
            if hasattr(value, "shape")
        },
        "example_train_sample": train_batch["sample_id"][0],
        "example_val_sample": val_batch["sample_id"][0],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
