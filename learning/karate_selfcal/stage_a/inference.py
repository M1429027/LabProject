"""Inference helpers for shared-camera refinement and non-degradation checks."""

from __future__ import annotations

import math
from typing import Any

import torch


def camera_observation_weights(batch: dict[str, Any]) -> torch.Tensor:
    """Estimate per-camera support from visible joints in a batch or clip batch."""

    joint_view_mask = batch.get("joint_view_mask")
    view_mask = batch.get("view_mask")
    if joint_view_mask is not None:
        weights = joint_view_mask.float()
        if weights.ndim == 4:
            return weights.mean(dim=(1, 2))
        if weights.ndim == 3:
            return weights.mean(dim=1)
        raise ValueError(f"joint_view_mask must be [B,J,V] or [B,T,J,V], got {tuple(weights.shape)}")
    if view_mask is not None:
        weights = view_mask.float()
        if weights.ndim == 3:
            return weights.mean(dim=1)
        if weights.ndim == 2:
            return weights
        raise ValueError(f"view_mask must be [B,V] or [B,T,V], got {tuple(weights.shape)}")
    raise KeyError("batch must contain joint_view_mask or view_mask for shared-camera aggregation.")


def aggregate_camera_tokens(
    camera_token_batches: list[torch.Tensor],
    observation_weight_batches: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool per-sample camera tokens into one sequence-level shared camera token."""

    if not camera_token_batches or not observation_weight_batches:
        raise ValueError("camera_token_batches and observation_weight_batches must be non-empty.")
    tokens = torch.cat(camera_token_batches, dim=0)
    weights = torch.cat(observation_weight_batches, dim=0)
    if tokens.ndim != 3 or weights.ndim != 2:
        raise ValueError(
            f"Expected tokens [N,V,D] and weights [N,V], got {tuple(tokens.shape)} and {tuple(weights.shape)}"
        )
    if tokens.shape[:2] != weights.shape:
        raise ValueError(
            f"Camera tokens and weights must agree on [N,V], got {tuple(tokens.shape)} and {tuple(weights.shape)}"
        )
    weight_sum = weights.sum(dim=0, keepdim=True)
    weighted = (tokens * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
    fallback = tokens.mean(dim=0, keepdim=True)
    shared = torch.where(
        weight_sum.unsqueeze(-1) > 0.0,
        weighted / weight_sum.unsqueeze(-1).clamp_min(1.0),
        fallback,
    )
    return shared, weight_sum


def predict_sequence_shared_camera(
    model: torch.nn.Module,
    camera_token_batches: list[torch.Tensor],
    observation_weight_batches: list[torch.Tensor],
    reference_ray_tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Predict one shared camera correction for a whole sequence."""

    shared_tokens, weight_sum = aggregate_camera_tokens(
        camera_token_batches=camera_token_batches,
        observation_weight_batches=observation_weight_batches,
    )
    outputs = model._predict_camera(shared_tokens, reference_ray_tokens[:1])  # noqa: SLF001
    outputs["shared_camera_tokens"] = shared_tokens
    outputs["shared_camera_weight_sum"] = weight_sum
    return outputs


def broadcast_camera_outputs(
    outputs: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Expand one shared camera prediction across every sample in the batch."""

    broadcast: dict[str, torch.Tensor] = {}
    for key, value in outputs.items():
        if not torch.is_tensor(value):
            broadcast[key] = value
        elif value.ndim > 0 and value.shape[0] == 1:
            broadcast[key] = value.expand(batch_size, *value.shape[1:])
        else:
            broadcast[key] = value
    return broadcast


def camera_output_diagnostics(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    """Summarize gate behavior and correction magnitudes for reporting."""

    diagnostics: dict[str, float] = {}
    if "pred_camera_origin_delta" in outputs:
        diagnostics["mean_camera_origin_delta_m"] = float(
            outputs["pred_camera_origin_delta"].norm(dim=-1).mean().detach().cpu()
        )
    if "pred_camera_rotation_delta" in outputs:
        diagnostics["mean_camera_rotation_delta_deg"] = float(
            torch.rad2deg(outputs["pred_camera_rotation_delta"].norm(dim=-1)).mean().detach().cpu()
        )
    if "pred_camera_scale_delta" in outputs:
        diagnostics["mean_camera_scale_delta"] = float(
            outputs["pred_camera_scale_delta"].abs().mean().detach().cpu()
        )
    for key in (
        "pred_camera_noop_probability",
        "pred_camera_correction_gate",
        "pred_camera_effective_gate",
    ):
        if key in outputs:
            diagnostics[f"mean_{key.removeprefix('pred_camera_')}"] = float(
                outputs[key].mean().detach().cpu()
            )
    if "shared_camera_weight_sum" in outputs:
        diagnostics["mean_shared_camera_support"] = float(
            outputs["shared_camera_weight_sum"].mean().detach().cpu()
        )
    return diagnostics


def non_degradation_summary(
    rough_score: float,
    corrected_score: float,
    tolerance: float = 0.0,
) -> dict[str, float | bool]:
    """Report whether corrected output safely beats the rough baseline."""

    improvement = float(rough_score - corrected_score)
    return {
        "rough_score": float(rough_score),
        "corrected_score": float(corrected_score),
        "improvement_m": improvement,
        "improvement_pct": 100.0 * improvement / max(abs(float(rough_score)), 1e-12),
        "fallback_to_rough": not (float(corrected_score) + float(tolerance) < float(rough_score)),
        "tolerance_m": float(tolerance),
    }
