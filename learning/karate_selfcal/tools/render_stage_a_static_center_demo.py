"""Evaluate sequence-static weighted center correction on a held-out Stage A sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from learning.karate_selfcal.stage_a.dataset import KarateStageADataset, collate_stage_a
from learning.karate_selfcal.stage_a.model import build_model
from learning.karate_selfcal.stage_a.train import (
    camera_corrected_triangulation,
    load_yaml,
    move_batch,
    resolve_device,
    root_relative,
)
from learning.karate_selfcal.tools.visualize_triangulation import estimate_axis_limits, render_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--axis-percentile", type=float, default=98.0)
    return parser.parse_args()


def frame_payload(frame_id: int, points: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    joints = []
    for joint_id, (point, valid) in enumerate(zip(points.detach().cpu().numpy(), mask.detach().cpu().numpy())):
        if not bool(valid) or not np.isfinite(point).all():
            continue
        joints.append(
            {
                "id": joint_id,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "confidence": 1.0,
            }
        )
    return {
        "frame": int(frame_id),
        "identities": [{"identity_id": 0, "person_id": "amass_person01", "joints": joints}],
    }


def masked_mpjpe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    error = torch.linalg.norm(pred - target, dim=-1)
    weight = mask.float()
    return float(((error * weight).sum() / weight.sum().clamp_min(1.0)).detach().cpu())


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config = load_yaml(config_path)
    dataset = KarateStageADataset(config["data"]["manifest_path"], split=args.split)
    indexes = [
        index
        for index, entry in enumerate(dataset.entries)
        if str(entry.get("sequence")) == args.sequence
        and int(entry.get("variant", 0)) == int(args.variant)
    ]
    if not indexes:
        raise ValueError(f"No held-out samples for {args.sequence}, variant {args.variant}")
    indexes.sort(key=lambda index: int(dataset.entries[index]["frame_id"]))
    entries = [dataset.entries[index] for index in indexes]
    samples = [dataset[index] for index in indexes]
    batch = collate_stage_a(samples)

    device = resolve_device(args.device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model = build_model(config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    batch_device = move_batch(batch, device)
    with torch.no_grad():
        outputs = model(
            batch_device["ray_tokens"],
            view_mask=batch_device["view_mask"],
            joint_view_mask=batch_device["joint_view_mask"],
        )

        joint_mask = batch_device["joint_view_mask"].bool()
        ray_confidence = batch_device["ray_tokens"][..., 8].clamp_min(0.0)
        frame_view_weight = (ray_confidence * joint_mask.float()).sum(dim=1)
        frame_view_weight = frame_view_weight / joint_mask.float().sum(dim=1).clamp_min(1.0)

        scale = batch_device["ray_origin_scale"].reshape(-1, 1, 1)
        predicted_delta_m = outputs["pred_camera_origin_delta"] * scale
        weighted_sum = (predicted_delta_m * frame_view_weight.unsqueeze(-1)).sum(dim=0)
        static_delta_m = weighted_sum / frame_view_weight.sum(dim=0).clamp_min(1e-6).unsqueeze(-1)
        static_delta_norm = static_delta_m.unsqueeze(0) / scale
        static_outputs = {
            "pred_camera_origin_delta": static_delta_norm.expand_as(outputs["pred_camera_origin_delta"])
        }
        corrected, corrected_valid = camera_corrected_triangulation(
            static_outputs,
            batch_device,
            apply_origin=True,
            apply_rotation=False,
        )

    target = batch_device["target_3d"]
    rough = batch_device["triangulated_3d"]
    target_valid = batch_device["target_confidence"] > 0.0
    rough_valid = target_valid & batch_device["triangulated_valid"].bool()
    corrected_mask = target_valid & corrected_valid
    shared_valid = rough_valid & corrected_mask
    frame_ids = [int(entry["frame_id"]) for entry in entries]

    gt_frames = [frame_payload(fid, target[i], target_valid[i]) for i, fid in enumerate(frame_ids)]
    rough_frames = [frame_payload(fid, rough[i], rough_valid[i]) for i, fid in enumerate(frame_ids)]
    corrected_frames = [frame_payload(fid, corrected[i], corrected_mask[i]) for i, fid in enumerate(frame_ids)]
    axis_limits = estimate_axis_limits(
        gt_frames + rough_frames + corrected_frames,
        up_axis="z",
        flip_up_axis=False,
        axis_percentile=float(args.axis_percentile),
    )

    output_path = Path(args.output_video).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (int(args.panel_width) * 3, int(args.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video: {output_path}")
    preview_path = output_path.with_name(f"{output_path.stem}_preview.png")
    try:
        for index, frame_id in enumerate(frame_ids):
            panels = [
                render_frame(
                    frame_data=frames[index],
                    axis_limits=axis_limits,
                    canvas_width=int(args.panel_width),
                    canvas_height=int(args.height),
                    title=f"{label} | frame {frame_id}",
                    up_axis="z",
                    flip_up_axis=False,
                    highlight_key_joints=True,
                )
                for label, frames in (
                    ("GT 3D", gt_frames),
                    ("Rough extrinsics", rough_frames),
                    ("Static center corrected", corrected_frames),
                )
            ]
            comparison = np.concatenate(panels, axis=1)
            writer.write(comparison)
            if index == len(frame_ids) // 2 and not cv2.imwrite(str(preview_path), comparison):
                raise RuntimeError(f"Failed to write preview image: {preview_path}")
    finally:
        writer.release()

    target_rr = root_relative(target)
    metrics = {
        "rough_raw_mpjpe": masked_mpjpe(rough, target, shared_valid),
        "center_corrected_raw_mpjpe": masked_mpjpe(corrected, target, shared_valid),
        "rough_root_relative_mpjpe": masked_mpjpe(root_relative(rough), target_rr, shared_valid),
        "center_corrected_root_relative_mpjpe": masked_mpjpe(
            root_relative(corrected), target_rr, shared_valid
        ),
    }
    summary = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "split": args.split,
        "sequence": args.sequence,
        "variant": int(args.variant),
        "num_frames": len(frame_ids),
        "aggregation": "confidence_weighted_mean",
        "static_center_delta_m": static_delta_m.detach().cpu().tolist(),
        "metrics_m": metrics,
        "coverage": {
            "rough": float(rough_valid.float().mean().detach().cpu()),
            "center_corrected": float(corrected_mask.float().mean().detach().cpu()),
            "shared": float(shared_valid.float().mean().detach().cpu()),
        },
        "output_video": str(output_path),
        "preview_image": str(preview_path),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
