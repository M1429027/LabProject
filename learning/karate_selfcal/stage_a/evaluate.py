"""Evaluate a trained Stage A model on a selected split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .dataset import KarateStageAClipDataset, KarateStageADataset, collate_stage_a
from .model import build_model
from .train import compute_extrinsic_refine_loss, compute_joint_finetune_loss, compute_loss, compute_motion_loss, compute_pelvis_loss, compute_triangulated_residual_loss, move_batch, resolve_device, select_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage A checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def mean_metrics(buckets: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(sum(values) / max(len(values), 1)) for key, values in buckets.items()}


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config: dict[str, Any] = checkpoint["config"]
    data_cfg = config.get("data", {})
    clip_length = int(data_cfg.get("clip_length", 1))
    if clip_length > 1:
        dataset = KarateStageAClipDataset(
            args.manifest,
            split=args.split,
            clip_length=clip_length,
            clip_stride=int(data_cfg.get("clip_stride", max(1, clip_length // 2))),
        )
    else:
        dataset = KarateStageADataset(args.manifest, split=args.split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_stage_a,
    )
    device = resolve_device(str(args.device))
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loss_cfg = config.get("loss", {})
    buckets: dict[str, list[float]] = {
        "loss": [],
        "absolute_mpjpe": [],
        "root_relative_mpjpe": [],
        "pose_root_relative_mpjpe": [],
        "pelvis_mpjpe": [],
        "endpoint_loss": [],
        "limb_extension_loss": [],
        "high_extension_ratio": [],
        "anchor_absolute_mpjpe": [],
        "anchor_root_relative_mpjpe": [],
        "corrected_anchor_absolute_mpjpe": [],
        "corrected_anchor_root_relative_mpjpe": [],
        "ray_consistency_loss": [],
        "camera_delta_loss": [],
        "camera_translation_delta_loss": [],
        "camera_rotation_delta_loss": [],
        "camera_rotation_geodesic_loss": [],
        "camera_rotation_relative_geodesic_loss": [],
        "rotation_corrected_geometry_loss": [],
        "camera_scale_delta_loss": [],
    }
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(
                batch["ray_tokens"],
                view_mask=batch["view_mask"],
                joint_view_mask=batch["joint_view_mask"],
            )
            target_space = str(loss_cfg.get("target_space", "absolute")).lower()
            if target_space in {"extrinsic_refine", "camera_extrinsic", "extrinsic_only"}:
                losses = compute_extrinsic_refine_loss(
                    outputs=outputs,
                    batch=batch,
                    loss_cfg=loss_cfg,
                )
            elif target_space in {"motion", "full_motion", "root_relative_plus_pelvis"}:
                losses = compute_motion_loss(
                    outputs=outputs,
                    batch=batch,
                    pose_weight=float(loss_cfg.get("root_relative_weight", 1.0)),
                    pelvis_weight=float(loss_cfg.get("absolute_weight", 1.0)),
                    final_weight=float(loss_cfg.get("final_weight", 0.25)),
                )
            elif target_space in {"joint_finetune", "a3_joint", "staged_joint"}:
                losses = compute_joint_finetune_loss(
                    outputs=outputs,
                    batch=batch,
                    loss_cfg=loss_cfg,
                )
            elif target_space in {"triangulated_residual", "geometry_residual", "anchor_residual"}:
                losses = compute_triangulated_residual_loss(
                    outputs=outputs,
                    batch=batch,
                    root_relative_weight=float(loss_cfg.get("root_relative_weight", 1.0)),
                    absolute_weight=float(loss_cfg.get("absolute_weight", 1.0)),
                    loss_cfg=loss_cfg,
                )
            elif target_space in {"pelvis", "global_pelvis", "translation"}:
                losses = compute_pelvis_loss(
                    outputs=outputs,
                    batch=batch,
                    pelvis_weight=float(loss_cfg.get("absolute_weight", 1.0)),
                )
            else:
                losses = compute_loss(
                    pred=outputs["pred_3d"],
                    target=select_target(batch, str(loss_cfg.get("target_space", "absolute"))),
                    confidence=batch["target_confidence"],
                    root_relative_weight=float(loss_cfg.get("root_relative_weight", 1.0)),
                    absolute_weight=float(loss_cfg.get("absolute_weight", 0.25)),
                    loss_cfg=loss_cfg,
                )
            for key in buckets:
                if key in losses:
                    buckets[key].append(float(losses[key].detach().cpu()))

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "split": args.split,
        "num_samples": len(dataset),
        "metrics": mean_metrics(buckets),
    }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
