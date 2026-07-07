"""Optimization-based 3D pose refinement entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .priors import COCO_SKELETON_TREE, estimate_bone_priors, joint_confidence_weight


NUM_COCO_JOINTS = 17


@dataclass(frozen=True)
class PoseRefinementConfig:
    """Hyperparameters for pose-only sequence refinement."""

    data_weight: float = 1.0
    bone_weight: float = 2.0
    symmetry_weight: float = 0.4
    smoothness_weight: float = 0.15
    reprojection_scale_px: float = 20.0
    min_bone_samples: int = 8
    max_nfev: int = 80
    robust_loss: str = "soft_l1"
    robust_f_scale: float = 0.08
    window_size: int = 12
    window_stride: int = 6
    learning_rate: float = 0.03
    bone_outlier_ratio: float = 1.6
    bone_outlier_downweight: float = 0.2


def extract_identity_sequences(
    payload: dict[str, Any],
    reprojection_scale_px: float,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    """Convert JSON payload into dense per-identity arrays."""

    frame_numbers = sorted({int(frame["frame"]) for frame in payload.get("frames", [])})
    frame_to_index = {frame: idx for idx, frame in enumerate(frame_numbers)}
    sequences: dict[int, dict[str, Any]] = {}
    for frame in payload.get("frames", []):
        frame_idx = frame_to_index[int(frame["frame"])]
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            if identity_id not in sequences:
                sequences[identity_id] = {
                    "joints": np.zeros((len(frame_numbers), NUM_COCO_JOINTS, 3), dtype=np.float64),
                    "mask": np.zeros((len(frame_numbers), NUM_COCO_JOINTS), dtype=bool),
                    "weights": np.zeros((len(frame_numbers), NUM_COCO_JOINTS), dtype=np.float64),
                    "meta": {},
                }
            sequence = sequences[identity_id]
            for joint in identity.get("joints", []):
                joint_id = int(joint["id"])
                if joint_id < 0 or joint_id >= NUM_COCO_JOINTS:
                    continue
                sequence["joints"][frame_idx, joint_id] = np.array(
                    [float(joint["x"]), float(joint["y"]), float(joint["z"])],
                    dtype=np.float64,
                )
                sequence["mask"][frame_idx, joint_id] = True
                sequence["weights"][frame_idx, joint_id] = joint_confidence_weight(
                    joint,
                    reprojection_scale_px=reprojection_scale_px,
                )
                sequence["meta"][(frame_idx, joint_id)] = {
                    key: value
                    for key, value in joint.items()
                    if key not in {"x", "y", "z"}
                }
    return frame_numbers, sequences


def pack_observed(joints: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Pack observed joint coordinates into an optimizer vector."""

    indices = [(int(frame_idx), int(joint_idx)) for frame_idx, joint_idx in np.argwhere(mask)]
    vector = np.asarray([joints[frame_idx, joint_idx] for frame_idx, joint_idx in indices], dtype=np.float64)
    return vector.reshape(-1), indices


def unpack_observed(
    vector: np.ndarray,
    original: np.ndarray,
    indices: list[tuple[int, int]],
) -> np.ndarray:
    """Unpack optimizer vector back into a dense joint array."""

    current = np.asarray(original, dtype=np.float64).copy()
    points = np.asarray(vector, dtype=np.float64).reshape(-1, 3)
    for point, (frame_idx, joint_idx) in zip(points, indices):
        current[frame_idx, joint_idx] = point
    return current


def optimize_identity_sequence(
    original: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    config: PoseRefinementConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize one identity sequence with pose-level constraints."""

    if int(np.sum(mask)) == 0:
        return original.copy(), {"status": "empty_identity"}

    priors = estimate_bone_priors(
        joints=original,
        mask=mask,
        min_samples=int(config.min_bone_samples),
    )
    effective_weights = downweight_bone_outlier_anchors(
        joints=original,
        mask=mask,
        weights=weights,
        priors=priors,
        max_ratio=float(config.bone_outlier_ratio),
        downweight=float(config.bone_outlier_downweight),
    )

    device = torch.device("cpu")
    original_t = torch.tensor(original, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
    weights_t = torch.tensor(effective_weights, dtype=torch.float32, device=device)
    current = torch.nn.Parameter(original_t.clone())
    optimizer = torch.optim.Adam([current], lr=float(config.learning_rate))

    def _loss() -> torch.Tensor:
        data_delta = current - original_t
        data_loss = torch.sum((data_delta[mask_t] ** 2) * weights_t[mask_t].unsqueeze(1))

        bone_terms = []
        for joint_a, joint_b in COCO_SKELETON_TREE:
            target = priors.bone_lengths.get((joint_a, joint_b))
            valid = mask_t[:, joint_a] & mask_t[:, joint_b]
            if target is None or not bool(torch.any(valid)):
                continue
            length = torch.linalg.norm(current[valid, joint_b] - current[valid, joint_a], dim=1)
            bone_terms.append(torch.mean((length - float(target)) ** 2))
        bone_loss = torch.stack(bone_terms).mean() if bone_terms else torch.tensor(0.0, device=device)

        symmetry_pairs = [((5, 7), (6, 8)), ((7, 9), (8, 10)), ((11, 13), (12, 14)), ((13, 15), (14, 16))]
        symmetry_terms = []
        for (la, lb), (ra, rb) in symmetry_pairs:
            valid = mask_t[:, la] & mask_t[:, lb] & mask_t[:, ra] & mask_t[:, rb]
            if not bool(torch.any(valid)):
                continue
            left = torch.linalg.norm(current[valid, lb] - current[valid, la], dim=1)
            right = torch.linalg.norm(current[valid, rb] - current[valid, ra], dim=1)
            symmetry_terms.append(torch.mean((left - right) ** 2))
        symmetry_loss = torch.stack(symmetry_terms).mean() if symmetry_terms else torch.tensor(0.0, device=device)

        smooth_terms = []
        if current.shape[0] >= 3:
            valid = mask_t[:-2] & mask_t[1:-1] & mask_t[2:]
            acceleration = current[:-2] - 2.0 * current[1:-1] + current[2:]
            if bool(torch.any(valid)):
                smooth_terms.append(torch.mean(acceleration[valid] ** 2))
        smooth_loss = torch.stack(smooth_terms).mean() if smooth_terms else torch.tensor(0.0, device=device)

        return (
            float(config.data_weight) * data_loss
            + float(config.bone_weight) * bone_loss
            + float(config.symmetry_weight) * symmetry_loss
            + float(config.smoothness_weight) * smooth_loss
        )

    initial_cost = float(_loss().detach().cpu().item())
    for _ in range(int(config.max_nfev)):
        optimizer.zero_grad()
        loss = _loss()
        loss.backward()
        with torch.no_grad():
            current.grad[~mask_t] = 0.0
        optimizer.step()
    final_cost = float(_loss().detach().cpu().item())
    refined = current.detach().cpu().numpy().astype(np.float64)
    refined[~mask] = original[~mask]
    diagnostics = {
        "status": "ok",
        "success": True,
        "optimizer": "torch_adam",
        "num_variables": int(np.sum(mask) * 3),
        "num_bone_priors": int(len(priors.bone_lengths)),
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "iterations": int(config.max_nfev),
    }
    return refined, diagnostics


def downweight_bone_outlier_anchors(
    joints: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    priors: Any,
    max_ratio: float,
    downweight: float,
) -> np.ndarray:
    """Reduce data-anchor trust for joints attached to implausibly long bones."""

    adjusted = np.asarray(weights, dtype=np.float64).copy()
    max_ratio = float(max(max_ratio, 1.0))
    downweight = float(np.clip(downweight, 0.0, 1.0))
    for frame_idx in range(joints.shape[0]):
        for joint_a, joint_b in COCO_SKELETON_TREE:
            target = priors.bone_lengths.get((joint_a, joint_b))
            if target is None or float(target) <= 1e-9:
                continue
            if not (mask[frame_idx, joint_a] and mask[frame_idx, joint_b]):
                continue
            length = float(np.linalg.norm(joints[frame_idx, joint_b] - joints[frame_idx, joint_a]))
            if not np.isfinite(length):
                continue
            if length / float(target) > max_ratio:
                adjusted[frame_idx, joint_b] *= downweight
    return adjusted


def optimize_identity_sequence_windowed(
    original: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    config: PoseRefinementConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize one identity sequence in overlapping temporal windows."""

    num_frames = int(original.shape[0])
    window_size = max(int(config.window_size), 3)
    window_stride = max(int(config.window_stride), 1)
    if num_frames <= window_size:
        return optimize_identity_sequence(original, mask, weights, config)

    refined_sum = np.zeros_like(original, dtype=np.float64)
    refined_count = np.zeros(mask.shape, dtype=np.float64)
    window_diagnostics = []
    starts = list(range(0, num_frames, window_stride))
    if starts and starts[-1] + window_size < num_frames:
        starts.append(max(num_frames - window_size, 0))
    elif not starts:
        starts = [0]

    for start in starts:
        end = min(start + window_size, num_frames)
        if end - start < 3:
            continue
        window_original = original[start:end]
        window_mask = mask[start:end]
        window_weights = weights[start:end]
        if int(np.sum(window_mask)) == 0:
            continue
        window_refined, diagnostics = optimize_identity_sequence(
            original=window_original,
            mask=window_mask,
            weights=window_weights,
            config=config,
        )
        window_diagnostics.append({"start": int(start), "end": int(end), **diagnostics})
        for local_frame, frame_idx in enumerate(range(start, end)):
            valid_joints = window_mask[local_frame]
            refined_sum[frame_idx, valid_joints] += window_refined[local_frame, valid_joints]
            refined_count[frame_idx, valid_joints] += 1.0

    refined = original.copy()
    valid = refined_count > 0
    refined[valid] = refined_sum[valid] / refined_count[..., None][valid]
    diagnostics = {
        "status": "ok",
        "mode": "windowed",
        "window_size": window_size,
        "window_stride": window_stride,
        "num_windows": len(window_diagnostics),
        "window_diagnostics": window_diagnostics,
        "initial_cost": float(sum(item.get("initial_cost", 0.0) for item in window_diagnostics)),
        "final_cost": float(sum(item.get("final_cost", 0.0) for item in window_diagnostics)),
    }
    return refined, diagnostics


def compute_sequence_metrics(
    joints: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Compute simple pose-quality metrics for one identity sequence."""

    displacement = np.linalg.norm(joints[mask] - original[mask], axis=1) if np.any(mask) else np.asarray([])
    bone_lengths = []
    temporal_acc = []
    for frame_idx in range(joints.shape[0]):
        for joint_a, joint_b in COCO_SKELETON_TREE:
            if mask[frame_idx, joint_a] and mask[frame_idx, joint_b]:
                bone_lengths.append(float(np.linalg.norm(joints[frame_idx, joint_b] - joints[frame_idx, joint_a])))
    for frame_idx in range(1, joints.shape[0] - 1):
        for joint_idx in range(joints.shape[1]):
            if mask[frame_idx - 1, joint_idx] and mask[frame_idx, joint_idx] and mask[frame_idx + 1, joint_idx]:
                acc = joints[frame_idx - 1, joint_idx] - 2.0 * joints[frame_idx, joint_idx] + joints[frame_idx + 1, joint_idx]
                temporal_acc.append(float(np.linalg.norm(acc)))
    return {
        "mean_displacement_from_input": float(np.mean(displacement)) if displacement.size else None,
        "median_displacement_from_input": float(np.median(displacement)) if displacement.size else None,
        "p95_displacement_from_input": float(np.percentile(displacement, 95)) if displacement.size else None,
        "median_bone_length": float(np.median(bone_lengths)) if bone_lengths else None,
        "p95_bone_length": float(np.percentile(bone_lengths, 95)) if bone_lengths else None,
        "mean_temporal_acceleration": float(np.mean(temporal_acc)) if temporal_acc else None,
        "median_temporal_acceleration": float(np.median(temporal_acc)) if temporal_acc else None,
    }


def build_refined_payload(
    input_payload: dict[str, Any],
    frame_numbers: list[int],
    sequences: dict[int, dict[str, Any]],
    refined_by_identity: dict[int, np.ndarray],
    config: PoseRefinementConfig,
) -> dict[str, Any]:
    """Build JSON payload from optimized dense sequences."""

    frames_out = []
    for frame_array_idx, frame_number in enumerate(frame_numbers):
        identities_out = []
        for identity_id in sorted(refined_by_identity):
            sequence = sequences[identity_id]
            mask = sequence["mask"]
            joints = refined_by_identity[identity_id]
            joints_out = []
            for joint_id in range(NUM_COCO_JOINTS):
                if not mask[frame_array_idx, joint_id]:
                    continue
                point = joints[frame_array_idx, joint_id]
                meta = dict(sequence["meta"].get((frame_array_idx, joint_id), {}))
                joints_out.append(
                    {
                        **meta,
                        "id": int(joint_id),
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "z": float(point[2]),
                        "pose_refinement": "optimization_pose_only_v1",
                    }
                )
            if joints_out:
                identities_out.append(
                    {
                        "identity_id": int(identity_id),
                        "num_joints": len(joints_out),
                        "joints": joints_out,
                    }
                )
        if identities_out:
            frames_out.append({"frame": int(frame_number), "identities": identities_out})

    return {
        "metadata": {
            **dict(input_payload.get("metadata", {})),
            "stage": "optimization_pose_refinement",
            "pose_refinement": {
                "version": "pose_only_v1",
                "data_weight": float(config.data_weight),
                "bone_weight": float(config.bone_weight),
                "symmetry_weight": float(config.symmetry_weight),
                "smoothness_weight": float(config.smoothness_weight),
                "robust_loss": str(config.robust_loss),
                "robust_f_scale": float(config.robust_f_scale),
                "learning_rate": float(config.learning_rate),
                "window_size": int(config.window_size),
                "window_stride": int(config.window_stride),
                "bone_outlier_ratio": float(config.bone_outlier_ratio),
                "bone_outlier_downweight": float(config.bone_outlier_downweight),
                "note": "Optimizes 3D joints only; cameras and 2D reprojection are not optimized in this version.",
            },
        },
        "frames": frames_out,
    }


def optimize_pose_payload(
    payload: dict[str, Any],
    config: PoseRefinementConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine all identities in a triangulated 3D payload."""

    frame_numbers, sequences = extract_identity_sequences(
        payload,
        reprojection_scale_px=float(config.reprojection_scale_px),
    )
    refined_by_identity: dict[int, np.ndarray] = {}
    diagnostics_by_identity: dict[str, Any] = {}
    before_metrics: dict[str, Any] = {}
    after_metrics: dict[str, Any] = {}
    for identity_id, sequence in sequences.items():
        original = sequence["joints"]
        mask = sequence["mask"]
        weights = sequence["weights"]
        refined, diagnostics = optimize_identity_sequence_windowed(
            original=original,
            mask=mask,
            weights=weights,
            config=config,
        )
        refined_by_identity[identity_id] = refined
        diagnostics_by_identity[str(identity_id)] = diagnostics
        before_metrics[str(identity_id)] = compute_sequence_metrics(original, original, mask)
        after_metrics[str(identity_id)] = compute_sequence_metrics(refined, original, mask)

    refined_payload = build_refined_payload(
        input_payload=payload,
        frame_numbers=frame_numbers,
        sequences=sequences,
        refined_by_identity=refined_by_identity,
        config=config,
    )
    metrics = {
        "stage": "optimization_pose_refinement",
        "version": "pose_only_v1",
        "config": config.__dict__,
        "identity_diagnostics": diagnostics_by_identity,
        "before": before_metrics,
        "after": after_metrics,
    }
    refined_payload["summary"] = {
        "num_identities": len(refined_by_identity),
        "identity_diagnostics": diagnostics_by_identity,
    }
    return refined_payload, metrics
