"""Train Stage A supervised multi-view 2D-to-3D warm-up model."""

from __future__ import annotations

import argparse
import math
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .dataset import build_stage_a_dataloaders
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage A supervised warm-up model.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required for Stage A training config.")
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pelvis_center(joints: torch.Tensor) -> torch.Tensor:
    return 0.5 * (joints[:, 11, :] + joints[:, 12, :])


def root_relative(joints: torch.Tensor) -> torch.Tensor:
    return joints - pelvis_center(joints).unsqueeze(1)


def mpjpe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    err = torch.linalg.norm(pred - target, dim=-1)
    if mask is not None:
        weights = mask.float()
        return (err * weights).sum() / weights.sum().clamp_min(1.0)
    return err.mean()


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = mask.float()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def endpoint_extension_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Extra supervision for attack endpoints that MPJPE tends to average away."""

    endpoint_weight = float(loss_cfg.get("endpoint_weight", 0.0))
    extension_weight = float(loss_cfg.get("limb_extension_weight", 0.0))
    high_extension_weight = float(loss_cfg.get("high_extension_weight", 0.0))
    if endpoint_weight <= 0.0 and extension_weight <= 0.0:
        zero = pred.new_tensor(0.0)
        return {
            "endpoint_loss": zero,
            "limb_extension_loss": zero,
            "high_extension_ratio": zero,
            "weighted_endpoint_loss": zero,
            "weighted_limb_extension_loss": zero,
        }

    endpoint_indices = loss_cfg.get("endpoint_joints", [9, 10, 15, 16])
    endpoint_indices = [int(index) for index in endpoint_indices]
    endpoint_tensor = torch.as_tensor(endpoint_indices, dtype=torch.long, device=pred.device)
    pred_endpoints = pred.index_select(dim=1, index=endpoint_tensor)
    target_endpoints = target.index_select(dim=1, index=endpoint_tensor)
    endpoint_mask = confidence.index_select(dim=1, index=endpoint_tensor) > 0.0

    endpoint_err = torch.linalg.norm(pred_endpoints - target_endpoints, dim=-1)

    pred_pelvis = pelvis_center(pred)
    target_pelvis = pelvis_center(target)
    pred_extension = torch.linalg.norm(pred_endpoints - pred_pelvis.unsqueeze(1), dim=-1)
    target_extension = torch.linalg.norm(target_endpoints - target_pelvis.unsqueeze(1), dim=-1)
    extension_err = torch.abs(pred_extension - target_extension)

    high_extension_threshold = float(loss_cfg.get("high_extension_threshold", 0.65))
    high_extension_mask = target_extension > high_extension_threshold
    high_extension_ratio = high_extension_mask.float().mean()
    weights = torch.ones_like(endpoint_err)
    if high_extension_weight > 0.0:
        weights = weights + float(high_extension_weight) * high_extension_mask.float()

    endpoint_loss = masked_mean(endpoint_err * weights, endpoint_mask)
    extension_loss = masked_mean(extension_err * weights, endpoint_mask)
    return {
        "endpoint_loss": endpoint_loss,
        "limb_extension_loss": extension_loss,
        "high_extension_ratio": high_extension_ratio,
        "weighted_endpoint_loss": endpoint_loss * endpoint_weight,
        "weighted_limb_extension_loss": extension_loss * extension_weight,
    }


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    root_relative_weight: float,
    absolute_weight: float,
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    mask = confidence > 0.0
    abs_loss = mpjpe(pred, target, mask=mask)
    rr_loss = mpjpe(root_relative(pred), root_relative(target), mask=mask)
    extra_losses = endpoint_extension_losses(pred=pred, target=target, confidence=confidence, loss_cfg=loss_cfg)
    total = (
        float(root_relative_weight) * rr_loss
        + float(absolute_weight) * abs_loss
        + extra_losses["weighted_endpoint_loss"]
        + extra_losses["weighted_limb_extension_loss"]
    )
    return {
        "loss": total,
        "absolute_mpjpe": abs_loss,
        "root_relative_mpjpe": rr_loss,
        "endpoint_loss": extra_losses["endpoint_loss"],
        "limb_extension_loss": extra_losses["limb_extension_loss"],
        "high_extension_ratio": extra_losses["high_extension_ratio"],
    }


def ray_reprojection_consistency_loss(
    pred: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> torch.Tensor:
    """Point-to-ray distance used as a ray-space reprojection proxy."""

    ray_tokens = batch["ray_tokens"]
    joint_view_mask = batch["joint_view_mask"].bool()
    if "ray_origin_center" not in batch or "ray_origin_scale" not in batch:
        return pred.new_tensor(0.0)
    origin_norm = ray_tokens[..., 0:3]
    direction = torch.nn.functional.normalize(ray_tokens[..., 3:6], dim=-1)
    confidence = ray_tokens[..., 8].clamp_min(0.0)
    scale = batch["ray_origin_scale"].reshape(-1, 1, 1, 1)
    center = batch["ray_origin_center"].reshape(-1, 1, 1, 3)
    origin = origin_norm * scale + center
    if "pred_camera_origin_delta" in outputs:
        origin = origin + outputs["pred_camera_origin_delta"].unsqueeze(1) * scale
    offset = pred.unsqueeze(2) - origin
    parallel = (offset * direction).sum(dim=-1, keepdim=True) * direction
    distance = torch.linalg.norm(offset - parallel, dim=-1)
    weights = confidence * joint_view_mask.float()
    return (distance * weights).sum() / weights.sum().clamp_min(1.0)


def camera_origin_delta_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    if "pred_camera_origin_delta" not in outputs or "camera_origin_delta" not in batch:
        return batch["ray_tokens"].new_tensor(0.0)
    mask = batch.get("camera_origin_delta_valid")
    if mask is None:
        mask = batch["view_mask"]
    err = torch.linalg.norm(outputs["pred_camera_origin_delta"] - batch["camera_origin_delta"], dim=-1)
    weights = mask.float()
    return (err * weights).sum() / weights.sum().clamp_min(1.0)


def camera_translation_delta_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    if "pred_camera_translation_delta" not in outputs or "camera_translation_delta" not in batch:
        return batch["ray_tokens"].new_tensor(0.0)
    mask = batch.get("camera_origin_delta_valid")
    if mask is None:
        mask = batch["view_mask"]
    err = torch.linalg.norm(outputs["pred_camera_translation_delta"] - batch["camera_translation_delta"], dim=-1)
    weights = mask.float()
    return (err * weights).sum() / weights.sum().clamp_min(1.0)


def camera_rotation_delta_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    if "pred_camera_rotation_delta" not in outputs or "camera_rotation_delta" not in batch:
        return batch["ray_tokens"].new_tensor(0.0)
    mask = batch.get("camera_origin_delta_valid")
    if mask is None:
        mask = batch["view_mask"]
    err = torch.linalg.norm(outputs["pred_camera_rotation_delta"] - batch["camera_rotation_delta"], dim=-1)
    weights = mask.float()
    return (err * weights).sum() / weights.sum().clamp_min(1.0)


def skew_symmetric(vectors: torch.Tensor) -> torch.Tensor:
    """Build batched skew-symmetric matrices for SO(3) operations."""

    x, y, z = vectors.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(vectors.shape[:-1] + (3, 3))


def rotation_vector_to_matrix(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Map axis-angle vectors to rotation matrices with a stable exponential map."""

    return torch.matrix_exp(skew_symmetric(rotation_vector))


def rotation_geodesic_angle(estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the SO(3) geodesic angle in radians."""

    relative = estimated @ target.transpose(-1, -2)
    skew_vector = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.norm(skew_vector, dim=-1)
    cosine = 0.5 * (
        relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2] - 1.0
    )
    return torch.atan2(sine, cosine.clamp(min=-1.0, max=1.0))


def camera_rotation_geodesic_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    anchor_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise absolute and camera-to-anchor relative rotation corrections."""

    if "pred_camera_rotation_delta" not in outputs or "camera_rotation_delta" not in batch:
        zero = batch["ray_tokens"].new_tensor(0.0)
        return zero, zero
    mask = batch.get("camera_origin_delta_valid", batch["view_mask"]).bool()
    predicted = rotation_vector_to_matrix(outputs["pred_camera_rotation_delta"])
    target = rotation_vector_to_matrix(batch["camera_rotation_delta"])
    absolute_loss = masked_mean(rotation_geodesic_angle(predicted, target), mask)

    anchor_index = int(max(0, min(anchor_index, predicted.shape[1] - 1)))
    predicted_anchor = predicted[:, anchor_index : anchor_index + 1]
    target_anchor = target[:, anchor_index : anchor_index + 1]
    predicted_relative = predicted_anchor.transpose(-1, -2) @ predicted
    target_relative = target_anchor.transpose(-1, -2) @ target
    relative_mask = mask & mask[:, anchor_index : anchor_index + 1]
    relative_loss = masked_mean(
        rotation_geodesic_angle(predicted_relative, target_relative),
        relative_mask,
    )
    return absolute_loss, relative_loss


def camera_scale_delta_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
    if "pred_camera_scale_delta" not in outputs or "camera_scale_delta" not in batch:
        return batch["ray_tokens"].new_tensor(0.0)
    mask = batch.get("camera_origin_delta_valid")
    if mask is None:
        mask = batch["view_mask"]
    err = torch.abs(outputs["pred_camera_scale_delta"] - batch["camera_scale_delta"])
    weights = mask.float()
    return (err * weights).sum() / weights.sum().clamp_min(1.0)


def camera_correction_gate_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Supervise when each camera should remain unchanged or apply a correction."""

    zero = batch["ray_tokens"].new_tensor(0.0)
    required = (
        "pred_camera_noop_probability" in outputs
        and "pred_camera_correction_gate" in outputs
        and "camera_origin_delta" in batch
    )
    if not required:
        return {
            "noop_gate_loss": zero,
            "correction_gate_loss": zero,
            "mean_noop_probability": zero,
            "mean_correction_gate": zero,
            "mean_effective_gate": zero,
        }

    valid = batch.get("camera_origin_delta_valid", batch["view_mask"]).bool()
    origin_error = torch.linalg.norm(batch["camera_origin_delta"], dim=-1)
    translation_error = torch.linalg.norm(batch["camera_translation_delta"], dim=-1)
    rotation_error = torch.linalg.norm(batch["camera_rotation_delta"], dim=-1)
    scale_error = torch.abs(batch["camera_scale_delta"])
    origin_threshold = float(loss_cfg.get("camera_gate_origin_threshold", 0.03))
    translation_threshold = float(loss_cfg.get("camera_gate_translation_threshold", 0.03))
    rotation_threshold = math.radians(float(loss_cfg.get("camera_gate_rotation_threshold_deg", 1.5)))
    scale_threshold = float(loss_cfg.get("camera_gate_scale_threshold", 0.03))
    required_strength = torch.maximum(
        torch.maximum(origin_error / max(origin_threshold, 1e-6), translation_error / max(translation_threshold, 1e-6)),
        torch.maximum(rotation_error / max(rotation_threshold, 1e-6), scale_error / max(scale_threshold, 1e-6)),
    )
    correction_target = required_strength.clamp(0.0, 1.0)
    noop_target = (required_strength <= float(loss_cfg.get("camera_noop_strength_threshold", 0.15))).float()
    noop_prediction = outputs["pred_camera_noop_probability"].clamp(1e-5, 1.0 - 1e-5)
    correction_prediction = outputs["pred_camera_correction_gate"]
    noop_loss = masked_mean(
        torch.nn.functional.binary_cross_entropy(noop_prediction, noop_target, reduction="none"), valid
    )
    correction_loss = masked_mean(
        torch.nn.functional.smooth_l1_loss(correction_prediction, correction_target, reduction="none"), valid
    )
    effective = outputs.get("pred_camera_effective_gate", correction_prediction * (1.0 - noop_prediction))
    return {
        "noop_gate_loss": noop_loss,
        "correction_gate_loss": correction_loss,
        "mean_noop_probability": masked_mean(noop_prediction, valid),
        "mean_correction_gate": masked_mean(correction_prediction, valid),
        "mean_effective_gate": masked_mean(effective, valid),
    }


def compute_extrinsic_refine_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Train camera correction heads directly toward oracle/clean extrinsics."""

    origin_loss = camera_origin_delta_loss(outputs=outputs, batch=batch)
    translation_loss = camera_translation_delta_loss(outputs=outputs, batch=batch)
    rotation_loss = camera_rotation_delta_loss(outputs=outputs, batch=batch)
    rotation_geodesic_loss, rotation_relative_geodesic_loss = camera_rotation_geodesic_losses(
        outputs=outputs,
        batch=batch,
        anchor_index=int(loss_cfg.get("rotation_anchor_index", 0)),
    )
    scale_loss = camera_scale_delta_loss(outputs=outputs, batch=batch)
    raw_outputs = dict(outputs)
    for key in (
        "pred_camera_origin_delta",
        "pred_camera_translation_delta",
        "pred_camera_rotation_delta",
        "pred_camera_scale_delta",
    ):
        raw_outputs[key] = outputs["raw_" + key]
    raw_origin_loss = camera_origin_delta_loss(outputs=raw_outputs, batch=batch)
    raw_translation_loss = camera_translation_delta_loss(outputs=raw_outputs, batch=batch)
    raw_rotation_loss = camera_rotation_delta_loss(outputs=raw_outputs, batch=batch)
    raw_rotation_geodesic_loss, _ = camera_rotation_geodesic_losses(
        outputs=raw_outputs,
        batch=batch,
        anchor_index=int(loss_cfg.get("rotation_anchor_index", 0)),
    )
    gate_losses = camera_correction_gate_losses(outputs=outputs, batch=batch, loss_cfg=loss_cfg)
    corrected_geometry_loss = batch["ray_tokens"].new_tensor(0.0)
    if bool(loss_cfg.get("use_rotation_corrected_triangulation", False)):
        corrected, corrected_valid = camera_corrected_triangulation(
            outputs=outputs,
            batch=batch,
            apply_origin=bool(loss_cfg.get("apply_origin_correction", False)),
            apply_rotation=True,
        )
        corrected_mask = corrected_valid & (batch["target_confidence"] > 0.0)
        corrected_geometry_loss = mpjpe(corrected, batch["target_3d"], mask=corrected_mask)
    total = (
        float(loss_cfg.get("camera_delta_weight", 1.0)) * origin_loss
        + float(loss_cfg.get("camera_translation_delta_weight", 0.25)) * translation_loss
        + float(loss_cfg.get("camera_rotation_delta_weight", 1.0)) * rotation_loss
        + float(loss_cfg.get("camera_rotation_geodesic_weight", 0.0)) * rotation_geodesic_loss
        + float(loss_cfg.get("camera_rotation_relative_geodesic_weight", 0.0)) * rotation_relative_geodesic_loss
        + float(loss_cfg.get("rotation_corrected_geometry_weight", 0.0)) * corrected_geometry_loss
        + float(loss_cfg.get("camera_scale_delta_weight", 0.5)) * scale_loss
        + float(loss_cfg.get("camera_noop_gate_weight", 0.0)) * gate_losses["noop_gate_loss"]
        + float(loss_cfg.get("camera_correction_gate_weight", 0.0)) * gate_losses["correction_gate_loss"]
        + float(loss_cfg.get("raw_camera_delta_weight", 0.0)) * raw_origin_loss
        + float(loss_cfg.get("raw_camera_translation_weight", 0.0)) * raw_translation_loss
        + float(loss_cfg.get("raw_camera_rotation_weight", 0.0)) * raw_rotation_loss
        + float(loss_cfg.get("raw_camera_rotation_geodesic_weight", 0.0)) * raw_rotation_geodesic_loss
    )
    zero = batch["ray_tokens"].new_tensor(0.0)
    return {
        "loss": total,
        "absolute_mpjpe": zero,
        "root_relative_mpjpe": zero,
        "pose_root_relative_mpjpe": zero,
        "pelvis_mpjpe": zero,
        "endpoint_loss": zero,
        "limb_extension_loss": zero,
        "high_extension_ratio": zero,
        "anchor_absolute_mpjpe": zero,
        "anchor_root_relative_mpjpe": zero,
        "corrected_anchor_absolute_mpjpe": zero,
        "corrected_anchor_root_relative_mpjpe": zero,
        "ray_consistency_loss": zero,
        "camera_delta_loss": origin_loss,
        "camera_translation_delta_loss": translation_loss,
        "camera_rotation_delta_loss": rotation_loss,
        "camera_rotation_geodesic_loss": rotation_geodesic_loss,
        "camera_rotation_relative_geodesic_loss": rotation_relative_geodesic_loss,
        "rotation_corrected_geometry_loss": corrected_geometry_loss,
        "camera_scale_delta_loss": scale_loss,
        "raw_camera_delta_loss": raw_origin_loss,
        "raw_camera_translation_loss": raw_translation_loss,
        "raw_camera_rotation_loss": raw_rotation_loss,
        "raw_camera_rotation_geodesic_loss": raw_rotation_geodesic_loss,
        **gate_losses,
    }

def camera_corrected_triangulation(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    apply_origin: bool = True,
    apply_rotation: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triangulate joints again after applying predicted camera-center correction.

    The dataset stores ray origins normalized by ``ray_origin_center`` and
    ``ray_origin_scale``. The camera-origin head predicts the correction in the
    same normalized space, so it can directly move ray origins before solving
    the multi-view least-squares ray intersection.
    """

    ray_tokens = batch["ray_tokens"]
    joint_view_mask = batch["joint_view_mask"].bool()
    if ray_tokens.ndim == 5:
        # The temporal model predicts one correction for the center frame.
        center_index = ray_tokens.shape[1] // 2
        ray_tokens = ray_tokens[:, center_index]
        joint_view_mask = joint_view_mask[:, center_index]
    origin_norm = ray_tokens[..., 0:3]
    direction = torch.nn.functional.normalize(ray_tokens[..., 3:6], dim=-1)
    confidence = ray_tokens[..., 8].clamp_min(0.0)
    scale = batch["ray_origin_scale"].reshape(-1, 1, 1, 1)
    center = batch["ray_origin_center"].reshape(-1, 1, 1, 3)
    origin = origin_norm * scale + center
    if apply_origin and "pred_camera_origin_delta" in outputs:
        origin = origin + outputs["pred_camera_origin_delta"].unsqueeze(1) * scale
    if apply_rotation and "pred_camera_rotation_delta" in outputs:
        correction = rotation_vector_to_matrix(outputs["pred_camera_rotation_delta"]).unsqueeze(1)
        direction = (correction @ direction.unsqueeze(-1)).squeeze(-1)
        direction = torch.nn.functional.normalize(direction, dim=-1)

    weights = (confidence * joint_view_mask.float()).clamp_min(0.0)
    eye = torch.eye(3, dtype=ray_tokens.dtype, device=ray_tokens.device).reshape(1, 1, 1, 3, 3)
    direction_outer = direction.unsqueeze(-1) * direction.unsqueeze(-2)
    projector = eye - direction_outer
    weighted_projector = projector * weights.unsqueeze(-1).unsqueeze(-1)
    lhs = weighted_projector.sum(dim=2)
    rhs = (weighted_projector @ origin.unsqueeze(-1)).sum(dim=2).squeeze(-1)

    valid = weights.gt(0.0).sum(dim=2) >= 2
    regularizer = 1e-4 * eye.reshape(1, 1, 3, 3)
    corrected = torch.linalg.solve(lhs + regularizer, rhs.unsqueeze(-1)).squeeze(-1)
    fallback = batch["triangulated_3d"]
    corrected = torch.where(valid.unsqueeze(-1), corrected, fallback)
    return corrected, valid


def compute_triangulated_residual_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    root_relative_weight: float,
    absolute_weight: float,
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Refine a geometry triangulation anchor instead of predicting 3D from scratch."""

    target = batch["target_3d"]
    confidence = batch["target_confidence"]
    anchor = batch["triangulated_3d"]
    anchor_valid = batch["triangulated_valid"]
    corrected_anchor = anchor
    corrected_anchor_valid = anchor_valid
    if bool(loss_cfg.get("use_camera_corrected_anchor", False)):
        corrected_anchor, corrected_anchor_valid = camera_corrected_triangulation(outputs=outputs, batch=batch)
    joint_mask = (confidence > 0.0) & anchor_valid
    corrected_joint_mask = (confidence > 0.0) & corrected_anchor_valid

    # Reuse the pose head as a per-joint residual head. The final prediction is
    # corrected geometry anchor + learned correction. This keeps global position grounded.
    pred = corrected_anchor + outputs["pred_pose_root_relative"]

    abs_loss = mpjpe(pred, target, mask=corrected_joint_mask)
    rr_loss = mpjpe(root_relative(pred), root_relative(target), mask=corrected_joint_mask)
    extra_losses = endpoint_extension_losses(
        pred=pred,
        target=target,
        confidence=corrected_joint_mask.float(),
        loss_cfg=loss_cfg,
    )
    ray_loss = ray_reprojection_consistency_loss(pred=pred, outputs=outputs, batch=batch)
    camera_loss = camera_origin_delta_loss(outputs=outputs, batch=batch)
    translation_loss = camera_translation_delta_loss(outputs=outputs, batch=batch)
    rotation_loss = camera_rotation_delta_loss(outputs=outputs, batch=batch)
    scale_loss = camera_scale_delta_loss(outputs=outputs, batch=batch)
    total = (
        float(absolute_weight) * abs_loss
        + float(root_relative_weight) * rr_loss
        + extra_losses["weighted_endpoint_loss"]
        + extra_losses["weighted_limb_extension_loss"]
        + float(loss_cfg.get("ray_consistency_weight", loss_cfg.get("reprojection_weight", 0.0))) * ray_loss
        + float(loss_cfg.get("camera_delta_weight", 0.0)) * camera_loss
        + float(loss_cfg.get("camera_translation_delta_weight", 0.0)) * translation_loss
        + float(loss_cfg.get("camera_rotation_delta_weight", 0.0)) * rotation_loss
        + float(loss_cfg.get("camera_scale_delta_weight", 0.0)) * scale_loss
    )
    return {
        "loss": total,
        "absolute_mpjpe": abs_loss,
        "root_relative_mpjpe": rr_loss,
        "anchor_absolute_mpjpe": mpjpe(anchor, target, mask=joint_mask),
        "anchor_root_relative_mpjpe": mpjpe(root_relative(anchor), root_relative(target), mask=joint_mask),
        "corrected_anchor_absolute_mpjpe": mpjpe(corrected_anchor, target, mask=corrected_joint_mask),
        "corrected_anchor_root_relative_mpjpe": mpjpe(root_relative(corrected_anchor), root_relative(target), mask=corrected_joint_mask),
        "endpoint_loss": extra_losses["endpoint_loss"],
        "limb_extension_loss": extra_losses["limb_extension_loss"],
        "high_extension_ratio": extra_losses["high_extension_ratio"],
        "ray_consistency_loss": ray_loss,
        "camera_delta_loss": camera_loss,
        "camera_translation_delta_loss": translation_loss,
        "camera_rotation_delta_loss": rotation_loss,
        "camera_scale_delta_loss": scale_loss,
    }


def compute_motion_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    pose_weight: float,
    pelvis_weight: float,
    final_weight: float,
) -> dict[str, torch.Tensor]:
    """Jointly supervise root-relative pose and global pelvis motion."""

    target_3d = batch["target_3d"]
    target_root_relative = batch["target_3d_root_relative"]
    confidence = batch["target_confidence"]
    joint_mask = confidence > 0.0

    pred_pose = outputs["pred_pose_root_relative"]
    pred_pelvis = outputs["pred_pelvis"]
    pred_3d = outputs["pred_3d"]
    target_pelvis = pelvis_center(target_3d)

    pose_loss = mpjpe(pred_pose, target_root_relative, mask=joint_mask)
    final_loss = mpjpe(pred_3d, target_3d, mask=joint_mask)
    pelvis_loss = torch.linalg.norm(pred_pelvis - target_pelvis, dim=-1).mean()
    rr_loss = mpjpe(root_relative(pred_3d), root_relative(target_3d), mask=joint_mask)
    total = float(pose_weight) * pose_loss + float(pelvis_weight) * pelvis_loss + float(final_weight) * final_loss
    return {
        "loss": total,
        "absolute_mpjpe": final_loss,
        "root_relative_mpjpe": rr_loss,
        "pose_root_relative_mpjpe": pose_loss,
        "pelvis_mpjpe": pelvis_loss,
    }


def compute_pelvis_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    pelvis_weight: float,
) -> dict[str, torch.Tensor]:
    """Supervise only the global pelvis translation branch."""

    target_3d = batch["target_3d"]
    confidence = batch["target_confidence"]
    joint_mask = confidence > 0.0
    target_pelvis = pelvis_center(target_3d)
    pred_pelvis = outputs["pred_pelvis"]
    pred_3d = outputs["pred_3d"]

    pelvis_loss = torch.linalg.norm(pred_pelvis - target_pelvis, dim=-1).mean()
    abs_loss = mpjpe(pred_3d, target_3d, mask=joint_mask)
    rr_loss = mpjpe(root_relative(pred_3d), root_relative(target_3d), mask=joint_mask)
    return {
        "loss": float(pelvis_weight) * pelvis_loss,
        "absolute_mpjpe": abs_loss,
        "root_relative_mpjpe": rr_loss,
        "pelvis_mpjpe": pelvis_loss,
    }


def compute_joint_finetune_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """A3 loss that balances pose shape and global pelvis during joint fine-tuning."""

    target_3d = batch["target_3d"]
    target_root_relative = batch["target_3d_root_relative"]
    confidence = batch["target_confidence"]
    joint_mask = confidence > 0.0

    pred_pose = outputs["pred_pose_root_relative"]
    pred_pelvis = outputs["pred_pelvis"]
    pred_3d = outputs["pred_3d"]
    target_pelvis = pelvis_center(target_3d)

    pose_loss = mpjpe(pred_pose, target_root_relative, mask=joint_mask)
    pelvis_loss = torch.linalg.norm(pred_pelvis - target_pelvis, dim=-1).mean()
    full_loss = mpjpe(pred_3d, target_3d, mask=joint_mask)
    rr_loss = mpjpe(root_relative(pred_3d), root_relative(target_3d), mask=joint_mask)
    extra_losses = endpoint_extension_losses(
        pred=pred_pose,
        target=target_root_relative,
        confidence=confidence,
        loss_cfg=loss_cfg,
    )

    pose_weight = float(loss_cfg.get("pose_weight", loss_cfg.get("root_relative_weight", 1.0)))
    pelvis_weight = float(loss_cfg.get("pelvis_weight", loss_cfg.get("absolute_weight", 0.75)))
    full_weight = float(loss_cfg.get("full_3d_weight", loss_cfg.get("final_weight", 0.15)))
    total = (
        pose_weight * pose_loss
        + pelvis_weight * pelvis_loss
        + full_weight * full_loss
        + extra_losses["weighted_endpoint_loss"]
        + extra_losses["weighted_limb_extension_loss"]
    )
    return {
        "loss": total,
        "absolute_mpjpe": full_loss,
        "root_relative_mpjpe": rr_loss,
        "pose_root_relative_mpjpe": pose_loss,
        "pelvis_mpjpe": pelvis_loss,
        "endpoint_loss": extra_losses["endpoint_loss"],
        "limb_extension_loss": extra_losses["limb_extension_loss"],
        "high_extension_ratio": extra_losses["high_extension_ratio"],
    }

def select_target(batch: dict[str, Any], target_space: str) -> torch.Tensor:
    """Select the supervised target space for this experiment."""

    if str(target_space).lower() in {"root_relative", "root-relative", "relative"}:
        return batch["target_3d_root_relative"]
    return batch["target_3d"]


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def mean_metrics(buckets: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(sum(values) / max(len(values), 1)) for key, values in buckets.items()}


def run_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    root_relative_weight: float,
    absolute_weight: float,
    final_weight: float,
    grad_clip_norm: float | None,
    target_space: str,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
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
        "noop_gate_loss": [],
        "correction_gate_loss": [],
        "mean_noop_probability": [],
        "mean_correction_gate": [],
        "mean_effective_gate": [],
        "raw_camera_delta_loss": [],
        "raw_camera_translation_loss": [],
        "raw_camera_rotation_loss": [],
        "raw_camera_rotation_geodesic_loss": [],
    }
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(
                batch["ray_tokens"],
                view_mask=batch["view_mask"],
                joint_view_mask=batch["joint_view_mask"],
            )
            target_space_key = str(target_space).lower()
            if target_space_key in {"extrinsic_refine", "camera_extrinsic", "extrinsic_only"}:
                losses = compute_extrinsic_refine_loss(
                    outputs=outputs,
                    batch=batch,
                    loss_cfg=loss_cfg,
                )
            elif target_space_key in {"motion", "full_motion", "root_relative_plus_pelvis"}:
                losses = compute_motion_loss(
                    outputs=outputs,
                    batch=batch,
                    pose_weight=root_relative_weight,
                    pelvis_weight=absolute_weight,
                    final_weight=final_weight,
                )
            elif target_space_key in {"joint_finetune", "a3_joint", "staged_joint"}:
                losses = compute_joint_finetune_loss(
                    outputs=outputs,
                    batch=batch,
                    loss_cfg=loss_cfg,
                )
            elif target_space_key in {"triangulated_residual", "geometry_residual", "anchor_residual"}:
                losses = compute_triangulated_residual_loss(
                    outputs=outputs,
                    batch=batch,
                    root_relative_weight=root_relative_weight,
                    absolute_weight=absolute_weight,
                    loss_cfg=loss_cfg,
                )
            elif target_space_key in {"pelvis", "global_pelvis", "translation"}:
                losses = compute_pelvis_loss(
                    outputs=outputs,
                    batch=batch,
                    pelvis_weight=absolute_weight,
                )
            else:
                losses = compute_loss(
                    pred=outputs["pred_3d"],
                    target=select_target(batch, target_space),
                    confidence=batch["target_confidence"],
                    root_relative_weight=root_relative_weight,
                    absolute_weight=absolute_weight,
                    loss_cfg=loss_cfg,
                )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()
        for key in buckets:
            if key in losses:
                buckets[key].append(float(losses[key].detach().cpu()))
    return mean_metrics(buckets)


def resolve_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def load_a3_pretrained(model: torch.nn.Module, training_cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Initialize A3 from validated A1/A2 checkpoints."""

    pose_checkpoint_path = training_cfg.get("pose_checkpoint")
    pelvis_checkpoint_path = training_cfg.get("pelvis_checkpoint")
    if not pose_checkpoint_path and not pelvis_checkpoint_path:
        return {"loaded": False}

    init_mode = str(training_cfg.get("a3_init_mode", "pose_backbone")).lower()
    summary: dict[str, Any] = {"loaded": True, "a3_init_mode": init_mode}
    pose_checkpoint = torch.load(str(pose_checkpoint_path), map_location=device) if pose_checkpoint_path else None
    pelvis_checkpoint = torch.load(str(pelvis_checkpoint_path), map_location=device) if pelvis_checkpoint_path else None

    if init_mode == "pelvis_backbone_pose_head":
        if pelvis_checkpoint is None or pose_checkpoint is None:
            raise ValueError("a3_init_mode=pelvis_backbone_pose_head requires both pose_checkpoint and pelvis_checkpoint.")
        state = dict(pelvis_checkpoint["model_state"])
        pose_head_state = {
            key: value
            for key, value in pose_checkpoint["model_state"].items()
            if key.startswith("head.") and key in state and state[key].shape == value.shape
        }
        state.update(pose_head_state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        summary["pose_loaded_keys"] = sorted(pose_head_state.keys())
        summary["base_checkpoint"] = str(Path(str(pelvis_checkpoint_path)).resolve())
        summary["pose_checkpoint"] = str(Path(str(pose_checkpoint_path)).resolve())
        summary["missing_keys"] = list(missing)
        summary["unexpected_keys"] = list(unexpected)
        return summary

    if pose_checkpoint is not None:
        missing, unexpected = model.load_state_dict(pose_checkpoint["model_state"], strict=False)
        summary["pose_checkpoint"] = str(Path(str(pose_checkpoint_path)).resolve())
        summary["pose_missing_keys"] = list(missing)
        summary["pose_unexpected_keys"] = list(unexpected)

    if pelvis_checkpoint is not None:
        model_state = model.state_dict()
        pelvis_state = {
            key: value
            for key, value in pelvis_checkpoint["model_state"].items()
            if key.startswith("pelvis_head.") and key in model_state and model_state[key].shape == value.shape
        }
        model_state.update(pelvis_state)
        model.load_state_dict(model_state, strict=True)
        summary["pelvis_checkpoint"] = str(Path(str(pelvis_checkpoint_path)).resolve())
        summary["pelvis_loaded_keys"] = sorted(pelvis_state.keys())
    return summary


def set_trainable_stage(model: torch.nn.Module, training_cfg: dict[str, Any], epoch: int) -> dict[str, Any]:
    """Apply A3 staged training schedule and report trainable parameter counts."""

    trainable_prefixes = [str(value) for value in training_cfg.get("trainable_prefixes", [])]
    if trainable_prefixes:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)
        return {
            "stage": "selected_prefixes",
            "trainable_prefixes": trainable_prefixes,
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
        }

    warmup_epochs = int(training_cfg.get("head_warmup_epochs", 0) or 0)
    head_only = epoch <= warmup_epochs
    for name, parameter in model.named_parameters():
        if head_only:
            parameter.requires_grad = name.startswith("head.") or name.startswith("pelvis_head.")
        else:
            parameter.requires_grad = True
    return {
        "stage": "head_only" if head_only else "full_model",
        "head_warmup_epochs": warmup_epochs,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }


def build_optimizer(model: torch.nn.Module, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

def load_optional_pretrained(model: torch.nn.Module, training_cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Load an optional warm-start checkpoint and optionally freeze pose layers."""

    checkpoint_path = training_cfg.get("pretrained_checkpoint")
    if not checkpoint_path:
        return {"loaded": False}

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    freeze_pose = bool(training_cfg.get("freeze_pose_backbone", False))
    if freeze_pose:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("pelvis_head.")
    return {
        "loaded": True,
        "checkpoint": str(Path(str(checkpoint_path)).resolve()),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "freeze_pose_backbone": freeze_pose,
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    experiment_cfg = config.get("experiment", {})
    training_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})

    set_seed(int(experiment_cfg.get("seed", 42)))
    output_dir = Path(str(experiment_cfg.get("output_dir", "outputs/karate_selfcal/stage_a_run"))).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    train_loader, val_loader = build_stage_a_dataloaders(config)
    device = resolve_device(str(training_cfg.get("device", "auto")))
    model = build_model(config).to(device)
    if training_cfg.get("pose_checkpoint") or training_cfg.get("pelvis_checkpoint"):
        pretrained_summary = load_a3_pretrained(model, training_cfg, device)
    else:
        pretrained_summary = load_optional_pretrained(model, training_cfg, device)
    if pretrained_summary.get("loaded"):
        print(json.dumps({"stage": "load_pretrained", **pretrained_summary}, ensure_ascii=False, indent=2))
    optimizer = build_optimizer(model, training_cfg)

    epochs = int(training_cfg.get("epochs", 20))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))
    root_relative_weight = float(loss_cfg.get("root_relative_weight", 1.0))
    absolute_weight = float(loss_cfg.get("absolute_weight", 0.25))
    final_weight = float(loss_cfg.get("final_weight", 0.25))
    target_space = str(loss_cfg.get("target_space", "absolute"))
    history = []
    best_val = float("inf")
    early_stop_patience = training_cfg.get("early_stop_patience")
    early_stop_patience = int(early_stop_patience) if early_stop_patience is not None else None
    early_stop_min_delta = float(training_cfg.get("early_stop_min_delta", 0.0))
    selection_metric = str(training_cfg.get("selection_metric", "root_relative_mpjpe"))
    epochs_without_improvement = 0
    stopped_early = False
    optimizer_stage: str | None = None

    for epoch in range(1, epochs + 1):
        trainable_summary = set_trainable_stage(model, training_cfg, epoch)
        if trainable_summary["stage"] != optimizer_stage:
            optimizer = build_optimizer(model, training_cfg)
            optimizer_stage = str(trainable_summary["stage"])
            print(json.dumps({"stage": "trainable_schedule", "epoch": epoch, **trainable_summary}, ensure_ascii=False, indent=2))
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            root_relative_weight=root_relative_weight,
            absolute_weight=absolute_weight,
            final_weight=final_weight,
            grad_clip_norm=grad_clip_norm,
            target_space=target_space,
            loss_cfg=loss_cfg,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                root_relative_weight=root_relative_weight,
                absolute_weight=absolute_weight,
                final_weight=final_weight,
                grad_clip_norm=None,
                target_space=target_space,
                loss_cfg=loss_cfg,
            )
        summary = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        state = {
            "epoch": epoch,
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train": train_metrics,
            "val": val_metrics,
        }
        torch.save(state, checkpoint_dir / "latest.pt")
        current_val = float(val_metrics.get(selection_metric, val_metrics["root_relative_mpjpe"]))
        if current_val < best_val - early_stop_min_delta:
            best_val = current_val
            epochs_without_improvement = 0
            torch.save(state, checkpoint_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        if early_stop_patience is not None and epochs_without_improvement >= early_stop_patience:
            stopped_early = True
            print(
                json.dumps(
                    {
                        "stage": "early_stop",
                        "epoch": epoch,
                        "best_val_metric": best_val,
                        "selection_metric": selection_metric,
                        "patience": early_stop_patience,
                        "min_delta": early_stop_min_delta,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            break
    save_json(
        output_dir / "history.json",
        {
            "epochs": history,
            "best_val_metric": best_val,
            "selection_metric": selection_metric,
            "stopped_early": stopped_early,
        },
    )


if __name__ == "__main__":
    main()
