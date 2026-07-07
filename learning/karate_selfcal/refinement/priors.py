"""Pose prior helpers for optimization-based refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


COCO_SKELETON_TREE = [
    (11, 12),
    (11, 5),
    (12, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 0),
    (6, 0),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

SYMMETRIC_BONES = [
    ((5, 7), (6, 8)),
    ((7, 9), (8, 10)),
    ((11, 13), (12, 14)),
    ((13, 15), (14, 16)),
    ((5, 11), (6, 12)),
    ((5, 0), (6, 0)),
]


@dataclass(frozen=True)
class SkeletonPriors:
    """Per-identity skeleton priors estimated from the input sequence."""

    bone_lengths: dict[tuple[int, int], float]


def estimate_bone_priors(
    joints: np.ndarray,
    mask: np.ndarray,
    min_samples: int,
) -> SkeletonPriors:
    """Estimate median bone lengths from one identity sequence."""

    bone_lengths: dict[tuple[int, int], float] = {}
    for joint_a, joint_b in COCO_SKELETON_TREE:
        valid = mask[:, joint_a] & mask[:, joint_b]
        if int(np.sum(valid)) < int(min_samples):
            continue
        lengths = np.linalg.norm(joints[valid, joint_b] - joints[valid, joint_a], axis=1)
        lengths = lengths[np.isfinite(lengths) & (lengths > 1e-9)]
        if len(lengths) < int(min_samples):
            continue
        bone_lengths[(joint_a, joint_b)] = float(np.median(lengths))
    return SkeletonPriors(bone_lengths=bone_lengths)


def joint_confidence_weight(joint: dict[str, Any], reprojection_scale_px: float) -> float:
    """Build a soft reliability weight from confidence, reprojection error, and view count."""

    raw_confidence = joint.get("mean_confidence", joint.get("confidence", 1.0))
    if raw_confidence is None:
        raw_confidence = 0.5
    confidence = float(raw_confidence)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    raw_num_views = joint.get("num_views", 2.0)
    if raw_num_views is None:
        raw_num_views = 1.0
    num_views = max(float(raw_num_views), 1.0)
    reprojection_error = joint.get("mean_reprojection_error_px")
    if reprojection_error is None:
        reprojection_weight = 1.0
    else:
        reprojection_weight = 1.0 / (1.0 + max(float(reprojection_error), 0.0) / float(reprojection_scale_px))
    view_weight = min(num_views / 4.0, 1.0)
    return float(max(confidence * reprojection_weight * (0.5 + 0.5 * view_weight), 1e-3))
