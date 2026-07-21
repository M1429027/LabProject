"""Render aligned two-person Harmony4D GT, rough, and camera-corrected poses."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from learning.karate_selfcal.stage_a.dataset import KarateStageADataset, collate_stage_a
from learning.karate_selfcal.stage_a.model import build_model
from learning.karate_selfcal.stage_a.train import camera_corrected_triangulation, load_yaml, move_batch, resolve_device
from learning.karate_selfcal.tools.visualize_triangulation import estimate_axis_limits, render_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gt-zip", required=True)
    parser.add_argument("--gt-sequence", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--panel-size", type=int, default=900)
    parser.add_argument("--axis-percentile", type=float, default=95.0)
    return parser.parse_args()


def payload(frame_id: int, identities: list[tuple[int, str, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    people = []
    for identity_id, person_id, points, confidence in identities:
        joints = [
            {"id": index, "x": float(point[0]), "y": float(point[1]), "z": float(point[2]), "confidence": float(score)}
            for index, (point, score) in enumerate(zip(points, confidence))
            if score > 0.0 and np.isfinite(point).all()
        ]
        people.append({"identity_id": identity_id, "person_id": person_id, "joints": joints})
    return {"frame": int(frame_id), "identities": people}


def centered_clips(samples: list[dict[str, Any]], length: int) -> dict[str, torch.Tensor]:
    half = length // 2
    clips = []
    for index in range(len(samples)):
        indices = [min(max(index + offset - half, 0), len(samples) - 1) for offset in range(length)]
        clips.append({key: torch.stack([samples[item][key] for item in indices]) for key in ("ray_tokens", "view_mask", "joint_view_mask")})
    return {key: torch.stack([clip[key] for clip in clips]) for key in ("ray_tokens", "view_mask", "joint_view_mask")}


def similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit one global rotation, scale, and translation from source to target."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    u, _singular, vt = np.linalg.svd(centered_source.T @ centered_target)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    rotated_source = centered_source @ rotation.T
    denominator = float(np.square(rotated_source).sum())
    if denominator <= 1e-12:
        raise ValueError("Cannot estimate a similarity transform from degenerate points.")
    scale = float((rotated_source * centered_target).sum() / denominator)
    translation = target_center - scale * source_center @ rotation.T
    return rotation, scale, translation


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dataset = KarateStageADataset(args.manifest, split="test")
    device = resolve_device(str(config.get("training", {}).get("device", "cuda")))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    records: dict[str, dict[int, dict[str, Any]]] = {}
    for person_id, identity_id in (("karate_person01", 0), ("karate_person02", 1)):
        indexes = [index for index, entry in enumerate(dataset.entries) if str(entry["person_id"]) == person_id]
        indexes.sort(key=lambda index: int(dataset.entries[index]["frame_id"]))
        samples = [dataset[index] for index in indexes]
        entries = [dataset.entries[index] for index in indexes]
        center_batch = collate_stage_a(samples)
        clips = centered_clips(samples, int(config["data"]["clip_length"]))
        with torch.no_grad():
            output = model(**move_batch(clips, device))
            corrected, valid = camera_corrected_triangulation(output, move_batch(center_batch, device), apply_origin=True, apply_rotation=True)
        records[person_id] = {
            int(entry["frame_id"]): {
                "rough": center_batch["triangulated_3d"][item].numpy(),
                "rough_valid": (center_batch["triangulated_valid"][item] & (center_batch["target_confidence"][item] > 0)).numpy(),
                "model": corrected[item].cpu().numpy(),
                "model_valid": (valid[item].cpu() & (center_batch["target_confidence"][item] > 0)).cpu().numpy(),
            }
            for item, entry in enumerate(entries)
        }

    gt: dict[int, dict[str, np.ndarray]] = {}
    prefix = f"{args.gt_sequence}/processed_data/poses3d/"
    with zipfile.ZipFile(args.gt_zip) as archive:
        for name in archive.namelist():
            if name.startswith(prefix) and name.endswith(".npy"):
                gt[int(Path(name).stem)] = np.load(io.BytesIO(archive.read(name)), allow_pickle=True).item()

    common = sorted(set(gt) & set(records["karate_person01"]) & set(records["karate_person02"]))
    gt_points, rough_points = [], []
    for frame_id in common:
        for person_id, gt_id in (("karate_person01", "aria01"), ("karate_person02", "aria02")):
            valid = records[person_id][frame_id]["rough_valid"]
            gt_points.append(np.asarray(gt[frame_id][gt_id], dtype=np.float64)[:17][valid, :3])
            rough_points.append(records[person_id][frame_id]["rough"][valid])
    rotation, scale, translation = similarity_transform(
        np.concatenate(rough_points),
        np.concatenate(gt_points),
    )

    gt_frames, rough_frames, model_frames = [], [], []
    rough_errors, model_errors = [], []
    for frame_id in common:
        panels = [[], [], []]
        for person_id, identity_id, gt_id in (("karate_person01", 0, "aria01"), ("karate_person02", 1, "aria02")):
            record = records[person_id][frame_id]
            gt_pose = np.asarray(gt[frame_id][gt_id], dtype=np.float64)[:17, :3]
            rough_pose = record["rough"] @ rotation.T * scale + translation
            model_pose = record["model"] @ rotation.T * scale + translation
            confidence = np.ones(17, dtype=np.float32)
            panels[0].append((identity_id, person_id, gt_pose, confidence))
            panels[1].append((identity_id, person_id, rough_pose, record["rough_valid"].astype(np.float32)))
            panels[2].append((identity_id, person_id, model_pose, record["model_valid"].astype(np.float32)))
            shared = record["rough_valid"] & record["model_valid"]
            rough_errors.append(np.linalg.norm(rough_pose[shared] - gt_pose[shared], axis=1))
            model_errors.append(np.linalg.norm(model_pose[shared] - gt_pose[shared], axis=1))
        gt_frames.append(payload(frame_id, panels[0]))
        rough_frames.append(payload(frame_id, panels[1]))
        model_frames.append(payload(frame_id, panels[2]))

    axis_limits = estimate_axis_limits(gt_frames + rough_frames + model_frames, up_axis="z", flip_up_axis=False, axis_percentile=args.axis_percentile)
    output = Path(args.output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.panel_size * 3, args.panel_size))
    for index, frame_id in enumerate(common):
        image = np.concatenate([
            render_frame(frame, axis_limits, args.panel_size, args.panel_size, title, up_axis="z", flip_up_axis=False, highlight_key_joints=True)
            for frame, title in zip((gt_frames[index], rough_frames[index], model_frames[index]), ("Aligned GT 3D", "Rough extrinsics", "V2 corrected triangulation"))
        ], axis=1)
        writer.write(image)
        if index == len(common) // 2:
            cv2.imwrite(str(output.with_name(f"{output.stem}_preview.png")), image)
    writer.release()
    summary = {
        "frames": len(common),
        "rough_similarity_aligned_mpjpe_m": float(np.concatenate(rough_errors).mean()),
        "model_similarity_aligned_mpjpe_m": float(np.concatenate(model_errors).mean()),
        "similarity_scale": scale,
        "alignment": "one global similarity transform applied equally to rough and model over both people and all frames",
        "output_video": str(output),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
