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
        triangulated_3d = sample["triangulated_3d"] if "triangulated_3d" in sample.files else np.zeros((17, 3), dtype=np.float32)
        triangulated_root = (
            sample["triangulated_3d_root_relative"]
            if "triangulated_3d_root_relative" in sample.files
            else np.zeros((17, 3), dtype=np.float32)
        )
        triangulated_quality = sample["triangulated_quality"] if "triangulated_quality" in sample.files else np.zeros((17,), dtype=np.float32)
        triangulated_valid = sample["triangulated_valid"] if "triangulated_valid" in sample.files else np.zeros((17,), dtype=np.bool_)
        pelvis_anchor = sample["pelvis_anchor"] if "pelvis_anchor" in sample.files else np.zeros((3,), dtype=np.float32)
        pelvis_anchor_quality = (
            np.asarray(sample["pelvis_anchor_quality"], dtype=np.float32).reshape(())
            if "pelvis_anchor_quality" in sample.files
            else np.asarray(0.0, dtype=np.float32)
        )
        pelvis_anchor_valid = (
            np.asarray(sample["pelvis_anchor_valid"], dtype=np.bool_).reshape(())
            if "pelvis_anchor_valid" in sample.files
            else np.asarray(False, dtype=np.bool_)
        )
        ray_origin_center = sample["ray_origin_center"] if "ray_origin_center" in sample.files else np.zeros((3,), dtype=np.float32)
        ray_origin_scale = (
            np.asarray(sample["ray_origin_scale"], dtype=np.float32).reshape(())
            if "ray_origin_scale" in sample.files
            else np.asarray(1.0, dtype=np.float32)
        )
        camera_origin_delta = sample["camera_origin_delta"] if "camera_origin_delta" in sample.files else np.zeros((sample["ray_tokens"].shape[1], 3), dtype=np.float32)
        camera_translation_delta = sample["camera_translation_delta"] if "camera_translation_delta" in sample.files else np.zeros((sample["ray_tokens"].shape[1], 3), dtype=np.float32)
        camera_rotation_delta = sample["camera_rotation_delta"] if "camera_rotation_delta" in sample.files else np.zeros((sample["ray_tokens"].shape[1], 3), dtype=np.float32)
        camera_scale_delta = sample["camera_scale_delta"] if "camera_scale_delta" in sample.files else np.zeros((sample["ray_tokens"].shape[1],), dtype=np.float32)
        camera_origin_delta_valid = sample["camera_origin_delta_valid"] if "camera_origin_delta_valid" in sample.files else np.zeros((sample["ray_tokens"].shape[1],), dtype=np.bool_)
        return {
            "ray_tokens": torch.from_numpy(sample["ray_tokens"]).float(),
            "view_mask": torch.from_numpy(sample["view_mask"].astype(np.bool_)),
            "joint_view_mask": torch.from_numpy(sample["joint_view_mask"].astype(np.bool_)),
            "target_3d": torch.from_numpy(sample["target_3d"]).float(),
            "target_3d_root_relative": torch.from_numpy(sample["target_3d_root_relative"]).float(),
            "target_confidence": torch.from_numpy(sample["target_confidence"]).float(),
            "triangulated_3d": torch.from_numpy(triangulated_3d).float(),
            "triangulated_3d_root_relative": torch.from_numpy(triangulated_root).float(),
            "triangulated_quality": torch.from_numpy(triangulated_quality).float(),
            "triangulated_valid": torch.from_numpy(triangulated_valid.astype(np.bool_)),
            "pelvis_anchor": torch.from_numpy(pelvis_anchor).float(),
            "pelvis_anchor_quality": torch.as_tensor(pelvis_anchor_quality).float(),
            "pelvis_anchor_valid": torch.as_tensor(pelvis_anchor_valid).bool(),
            "ray_origin_center": torch.from_numpy(ray_origin_center).float(),
            "ray_origin_scale": torch.as_tensor(ray_origin_scale).float(),
            "camera_origin_delta": torch.from_numpy(camera_origin_delta).float(),
            "camera_translation_delta": torch.from_numpy(camera_translation_delta).float(),
            "camera_rotation_delta": torch.from_numpy(camera_rotation_delta).float(),
            "camera_scale_delta": torch.from_numpy(camera_scale_delta).float(),
            "camera_origin_delta_valid": torch.from_numpy(camera_origin_delta_valid.astype(np.bool_)),
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
        "triangulated_3d": torch.stack([sample["triangulated_3d"] for sample in samples], dim=0),
        "triangulated_3d_root_relative": torch.stack([sample["triangulated_3d_root_relative"] for sample in samples], dim=0),
        "triangulated_quality": torch.stack([sample["triangulated_quality"] for sample in samples], dim=0),
        "triangulated_valid": torch.stack([sample["triangulated_valid"] for sample in samples], dim=0),
        "pelvis_anchor": torch.stack([sample["pelvis_anchor"] for sample in samples], dim=0),
        "pelvis_anchor_quality": torch.stack([sample["pelvis_anchor_quality"] for sample in samples], dim=0),
        "pelvis_anchor_valid": torch.stack([sample["pelvis_anchor_valid"] for sample in samples], dim=0),
        "ray_origin_center": torch.stack([sample["ray_origin_center"] for sample in samples], dim=0),
        "ray_origin_scale": torch.stack([sample["ray_origin_scale"] for sample in samples], dim=0),
        "camera_origin_delta": torch.stack([sample["camera_origin_delta"] for sample in samples], dim=0),
        "camera_translation_delta": torch.stack([sample["camera_translation_delta"] for sample in samples], dim=0),
        "camera_rotation_delta": torch.stack([sample["camera_rotation_delta"] for sample in samples], dim=0),
        "camera_scale_delta": torch.stack([sample["camera_scale_delta"] for sample in samples], dim=0),
        "camera_origin_delta_valid": torch.stack([sample["camera_origin_delta_valid"] for sample in samples], dim=0),
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
