"""Loss definitions for post-triangulation pose refinement."""

from __future__ import annotations

import numpy as np

from .priors import COCO_SKELETON_TREE, SYMMETRIC_BONES, SkeletonPriors


def data_residuals(
    current: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    loss_weight: float,
) -> list[float]:
    """Keep refined joints close to the triangulated input, weighted by reliability."""

    residuals: list[float] = []
    scale = float(np.sqrt(max(loss_weight, 0.0)))
    for frame_idx, joint_idx in np.argwhere(mask):
        joint_weight = float(np.sqrt(max(weights[frame_idx, joint_idx], 1e-6)))
        delta = current[frame_idx, joint_idx] - original[frame_idx, joint_idx]
        residuals.extend((scale * joint_weight * delta).tolist())
    return residuals


def bone_length_residuals(
    current: np.ndarray,
    mask: np.ndarray,
    priors: SkeletonPriors,
    loss_weight: float,
) -> list[float]:
    """Encourage each bone to stay near its identity-level median length."""

    residuals: list[float] = []
    scale = float(np.sqrt(max(loss_weight, 0.0)))
    if scale <= 0.0:
        return residuals
    for frame_idx in range(current.shape[0]):
        for joint_a, joint_b in COCO_SKELETON_TREE:
            target = priors.bone_lengths.get((joint_a, joint_b))
            if target is None or not (mask[frame_idx, joint_a] and mask[frame_idx, joint_b]):
                continue
            length = float(np.linalg.norm(current[frame_idx, joint_b] - current[frame_idx, joint_a]))
            residuals.append(scale * (length - float(target)))
    return residuals


def symmetry_residuals(
    current: np.ndarray,
    mask: np.ndarray,
    loss_weight: float,
) -> list[float]:
    """Encourage left/right limb lengths to remain mutually consistent."""

    residuals: list[float] = []
    scale = float(np.sqrt(max(loss_weight, 0.0)))
    if scale <= 0.0:
        return residuals
    for frame_idx in range(current.shape[0]):
        for left_edge, right_edge in SYMMETRIC_BONES:
            la, lb = left_edge
            ra, rb = right_edge
            if not (mask[frame_idx, la] and mask[frame_idx, lb] and mask[frame_idx, ra] and mask[frame_idx, rb]):
                continue
            left_length = float(np.linalg.norm(current[frame_idx, lb] - current[frame_idx, la]))
            right_length = float(np.linalg.norm(current[frame_idx, rb] - current[frame_idx, ra]))
            residuals.append(scale * (left_length - right_length))
    return residuals


def temporal_smoothness_residuals(
    current: np.ndarray,
    mask: np.ndarray,
    loss_weight: float,
) -> list[float]:
    """Penalize second-order temporal acceleration for each observed joint."""

    residuals: list[float] = []
    scale = float(np.sqrt(max(loss_weight, 0.0)))
    if scale <= 0.0 or current.shape[0] < 3:
        return residuals
    for frame_idx in range(1, current.shape[0] - 1):
        for joint_idx in range(current.shape[1]):
            if not (mask[frame_idx - 1, joint_idx] and mask[frame_idx, joint_idx] and mask[frame_idx + 1, joint_idx]):
                continue
            acceleration = current[frame_idx - 1, joint_idx] - 2.0 * current[frame_idx, joint_idx] + current[frame_idx + 1, joint_idx]
            residuals.extend((scale * acceleration).tolist())
    return residuals


def build_pose_refinement_residuals(
    current: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    priors: SkeletonPriors,
    data_weight: float,
    bone_weight: float,
    symmetry_weight: float,
    smoothness_weight: float,
) -> np.ndarray:
    """Build one residual vector for pose-only optimization."""

    residuals: list[float] = []
    residuals.extend(data_residuals(current, original, mask, weights, data_weight))
    residuals.extend(bone_length_residuals(current, mask, priors, bone_weight))
    residuals.extend(symmetry_residuals(current, mask, symmetry_weight))
    residuals.extend(temporal_smoothness_residuals(current, mask, smoothness_weight))
    return np.asarray(residuals, dtype=np.float64)
