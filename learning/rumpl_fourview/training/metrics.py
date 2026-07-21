from __future__ import annotations

from learning.rumpl_fourview.training.losses import mpjpe, root_relative_joints


def compute_metrics(pred_3d, target_3d) -> dict[str, float]:
    absolute = float(mpjpe(pred_3d, target_3d).detach().cpu())
    root_relative = float(mpjpe(root_relative_joints(pred_3d), root_relative_joints(target_3d)).detach().cpu())
    return {
        "mpjpe": absolute,
        "root_relative_mpjpe": root_relative,
    }
