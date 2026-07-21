"""Render GT, rough triangulation, and temporal camera-model correction side by side."""

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
from learning.karate_selfcal.tools.visualize_triangulation import (
    estimate_axis_limits,
    render_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--person-id", default=None, help="Optional identity filter for temporal single-person clips.")
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--axis-percentile", type=float, default=98.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _frame_payload(frame_id: int, points: torch.Tensor, confidence: torch.Tensor) -> dict[str, Any]:
    joints = []
    points_np = points.detach().cpu().numpy()
    confidence_np = confidence.detach().cpu().numpy()
    for joint_id, (point, score) in enumerate(zip(points_np, confidence_np)):
        if float(score) <= 0.0 or not np.isfinite(point).all():
            continue
        joints.append(
            {
                "id": int(joint_id),
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "confidence": float(score),
            }
        )
    return {
        "frame": int(frame_id),
        "identities": [{"identity_id": 0, "person_id": "amass_person01", "joints": joints}],
    }


def _masked_mpjpe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    error = torch.linalg.norm(pred - target, dim=-1)
    weights = mask.float()
    return float(((error * weights).sum() / weights.sum().clamp_min(1.0)).detach().cpu())


def _load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model = build_model(config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def _centered_clip(samples: list[dict[str, Any]], center_index: int, length: int) -> dict[str, torch.Tensor]:
    half = length // 2
    indexes = [min(max(center_index + offset - half, 0), len(samples) - 1) for offset in range(length)]
    selected = [samples[index] for index in indexes]
    return {
        key: torch.stack([sample[key] for sample in selected], dim=0)
        for key in ("ray_tokens", "view_mask", "joint_view_mask")
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config = load_yaml(config_path)
    manifest_path = Path(config["data"]["manifest_path"])
    dataset = KarateStageADataset(manifest_path, split=args.split)

    indexes = [
        index
        for index, entry in enumerate(dataset.entries)
        if str(entry.get("sequence")) == args.sequence
        and int(entry.get("variant", 0)) == int(args.variant)
        and (args.person_id is None or str(entry.get("person_id")) == args.person_id)
    ]
    if not indexes:
        raise ValueError(
            f"No samples for split={args.split}, sequence={args.sequence}, variant={args.variant}"
        )
    indexes.sort(key=lambda index: int(dataset.entries[index]["frame_id"]))
    entries = [dataset.entries[index] for index in indexes]
    samples = [dataset[index] for index in indexes]
    noise_sources = sorted({str(entry.get("noise_source", "unknown")) for entry in entries})

    clip_length = int(config.get("data", {}).get("clip_length", 1))
    if clip_length <= 1:
        raise ValueError("This demo expects a temporal model with data.clip_length > 1")
    clip_samples = [_centered_clip(samples, index, clip_length) for index in range(len(samples))]
    center_batch = collate_stage_a(samples)
    clip_batch = {
        key: torch.stack([clip[key] for clip in clip_samples], dim=0)
        for key in ("ray_tokens", "view_mask", "joint_view_mask")
    }

    requested_device = args.device or str(config.get("training", {}).get("device", "auto"))
    device = resolve_device(requested_device)
    model = _load_model(config, checkpoint_path, device)
    with torch.no_grad():
        clip_on_device = move_batch(clip_batch, device)
        center_on_device = move_batch(center_batch, device)
        outputs = model(
            clip_on_device["ray_tokens"],
            view_mask=clip_on_device["view_mask"],
            joint_view_mask=clip_on_device["joint_view_mask"],
        )
        corrected, corrected_valid = camera_corrected_triangulation(
            outputs,
            center_on_device,
            apply_origin=True,
            apply_rotation=True,
        )

    target = center_on_device["target_3d"]
    rough = center_on_device["triangulated_3d"]
    target_valid = center_on_device["target_confidence"] > 0.0
    rough_valid = target_valid & center_on_device["triangulated_valid"].bool()
    model_valid = target_valid & corrected_valid
    shared_valid = rough_valid & model_valid

    frame_ids = [int(entry["frame_id"]) for entry in entries]
    gt_frames = [
        _frame_payload(frame_id, target[index], target_valid[index].float())
        for index, frame_id in enumerate(frame_ids)
    ]
    rough_frames = [
        _frame_payload(frame_id, rough[index], rough_valid[index].float())
        for index, frame_id in enumerate(frame_ids)
    ]
    corrected_frames = [
        _frame_payload(frame_id, corrected[index], model_valid[index].float())
        for index, frame_id in enumerate(frame_ids)
    ]
    all_frames = gt_frames + rough_frames + corrected_frames
    axis_limits = estimate_axis_limits(
        all_frames,
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
    try:
        for index, frame_id in enumerate(frame_ids):
            panels = []
            for label, frames in (
                ("GT 3D", gt_frames),
                ("Rough extrinsics", rough_frames),
                ("Temporal camera model", corrected_frames),
            ):
                panels.append(
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
                )
            comparison = np.concatenate(panels, axis=1)
            writer.write(comparison)
            if index == len(frame_ids) // 2:
                preview_path = output_path.with_name(f"{output_path.stem}_preview.png")
                if not cv2.imwrite(str(preview_path), comparison):
                    raise RuntimeError(f"Failed to write preview image: {preview_path}")
    finally:
        writer.release()

    rough_rr = root_relative(rough)
    corrected_rr = root_relative(corrected)
    target_rr = root_relative(target)
    summary = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "manifest": str(manifest_path.resolve()),
        "split": args.split,
        "sequence": args.sequence,
        "person_id": args.person_id,
        "variant": int(args.variant),
        "noise_sources": noise_sources,
        "frames": len(frame_ids),
        "clip_length": clip_length,
        "device": str(device),
        "corrections_applied": ["camera_origin_delta", "camera_rotation_delta"],
        "metrics_m": {
            "rough_raw_mpjpe": _masked_mpjpe(rough, target, shared_valid),
            "model_raw_mpjpe": _masked_mpjpe(corrected, target, shared_valid),
            "rough_root_relative_mpjpe": _masked_mpjpe(rough_rr, target_rr, shared_valid),
            "model_root_relative_mpjpe": _masked_mpjpe(corrected_rr, target_rr, shared_valid),
        },
        "coverage": {
            "rough": float(rough_valid.float().mean().detach().cpu()),
            "model": float(model_valid.float().mean().detach().cpu()),
            "shared": float(shared_valid.float().mean().detach().cpu()),
        },
        "output_video": str(output_path),
        "preview_image": str(output_path.with_name(f"{output_path.stem}_preview.png")),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
