"""PyTorch dataloader for Stage A precomputed karate ray-token samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class KarateStageADataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str | None = None):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        entries = list(manifest.get("entries", []))
        if split is not None:
            entries = [entry for entry in entries if entry.get("split") == split]
        if not entries:
            raise ValueError(f"No Stage A entries found for split={split!r} in {self.manifest_path}")
        self.manifest = manifest
        self.entries: list[dict[str, Any]] = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        path = Path(str(entry["path"]))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        sample = np.load(path, allow_pickle=False)
        return {
            "ray_tokens": torch.from_numpy(sample["ray_tokens"]).float(),
            "view_mask": torch.from_numpy(sample["view_mask"].astype(np.bool_)),
            "joint_view_mask": torch.from_numpy(sample["joint_view_mask"].astype(np.bool_)),
            "target_3d": torch.from_numpy(sample["target_3d"]).float(),
            "target_3d_root_relative": torch.from_numpy(sample["target_3d_root_relative"]).float(),
            "target_confidence": torch.from_numpy(sample["target_confidence"]).float(),
            "sample_id": str(entry["id"]),
            "sequence": str(entry["sequence"]),
            "frame_id": int(entry["frame_id"]),
            "person_id": str(entry["person_id"]),
        }


def collate_stage_a(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ray_tokens": torch.stack([sample["ray_tokens"] for sample in samples], dim=0),
        "view_mask": torch.stack([sample["view_mask"] for sample in samples], dim=0),
        "joint_view_mask": torch.stack([sample["joint_view_mask"] for sample in samples], dim=0),
        "target_3d": torch.stack([sample["target_3d"] for sample in samples], dim=0),
        "target_3d_root_relative": torch.stack([sample["target_3d_root_relative"] for sample in samples], dim=0),
        "target_confidence": torch.stack([sample["target_confidence"] for sample in samples], dim=0),
        "sample_id": [sample["sample_id"] for sample in samples],
        "sequence": [sample["sequence"] for sample in samples],
        "frame_id": [sample["frame_id"] for sample in samples],
        "person_id": [sample["person_id"] for sample in samples],
    }


def build_stage_a_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config.get("data", {})
    manifest_path = data_cfg["manifest_path"]
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    train_dataset = KarateStageADataset(manifest_path, split=data_cfg.get("train_split", "train"))
    val_dataset = KarateStageADataset(manifest_path, split=data_cfg.get("val_split", "val"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_stage_a,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_stage_a,
    )
    return train_loader, val_loader
