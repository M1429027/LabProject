from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PrecomputedRayDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: Optional[str] = None):
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        entries = manifest.get("entries", [])
        if split is not None:
            entries = [entry for entry in entries if entry.get("split") == split]
        if not entries:
            raise ValueError(f"No dataset entries found in {manifest_path} for split={split!r}.")
        self.manifest_path = manifest_path
        self.entries: List[Dict[str, object]] = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[index]
        sample_path = Path(str(entry["path"]))
        if not sample_path.is_absolute():
            sample_path = (Path.cwd() / sample_path).resolve()
        sample = np.load(sample_path, allow_pickle=False)
        ray_tokens = torch.from_numpy(sample["ray_tokens"]).float()
        view_mask = torch.from_numpy(sample["view_mask"].astype(np.bool_))
        target_3d = torch.from_numpy(sample["target_3d"]).float()
        batch = {
            "ray_tokens": ray_tokens,
            "view_mask": view_mask,
            "target_3d": target_3d,
            "sample_id": str(entry.get("id", index)),
        }
        if "observations_2d" in sample:
            batch["observations_2d"] = torch.from_numpy(sample["observations_2d"]).float()
        return batch


def collate_batch(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch: Dict[str, object] = {
        "ray_tokens": torch.stack([sample["ray_tokens"] for sample in samples], dim=0),
        "view_mask": torch.stack([sample["view_mask"] for sample in samples], dim=0),
        "target_3d": torch.stack([sample["target_3d"] for sample in samples], dim=0),
        "sample_id": [sample["sample_id"] for sample in samples],
    }
    if "observations_2d" in samples[0]:
        batch["observations_2d"] = torch.stack([sample["observations_2d"] for sample in samples], dim=0)
    return batch


def build_dataloaders(config: Dict[str, object]) -> Tuple[DataLoader, DataLoader]:
    data_cfg = config.get("data", {})
    manifest_path = data_cfg.get("manifest_path")
    if not manifest_path:
        raise ValueError("data.manifest_path is required.")
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    train_dataset = PrecomputedRayDataset(manifest_path, split=data_cfg.get("train_split", "train"))
    val_dataset = PrecomputedRayDataset(manifest_path, split=data_cfg.get("val_split", "val"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader

