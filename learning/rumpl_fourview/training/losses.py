from __future__ import annotations

import torch


def pelvis_center(joints_3d: torch.Tensor) -> torch.Tensor:
    left_hip = joints_3d[:, 11, :]
    right_hip = joints_3d[:, 12, :]
    return (left_hip + right_hip) / 2.0


def root_relative_joints(joints_3d: torch.Tensor) -> torch.Tensor:
    pelvis = pelvis_center(joints_3d).unsqueeze(1)
    return joints_3d - pelvis


def mpjpe(pred_3d: torch.Tensor, target_3d: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred_3d - target_3d, dim=-1).mean()


def compute_losses(pred_3d: torch.Tensor, target_3d: torch.Tensor, root_relative: bool = True) -> dict[str, torch.Tensor]:
    absolute = mpjpe(pred_3d, target_3d)
    if root_relative:
        pred_rr = root_relative_joints(pred_3d)
        target_rr = root_relative_joints(target_3d)
        root_loss = mpjpe(pred_rr, target_rr)
    else:
        root_loss = absolute
    total = root_loss + (0.25 * absolute)
    return {
        "loss": total,
        "absolute_mpjpe": absolute,
        "root_relative_mpjpe": root_loss,
    }
