from __future__ import annotations

from typing import Dict

import torch

from learning.rumpl_fourview.training.losses import compute_losses
from learning.rumpl_fourview.training.metrics import compute_metrics


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _mean_metrics(metric_buckets: Dict[str, list[float]]) -> Dict[str, float]:
    return {key: float(sum(values) / max(len(values), 1)) for key, values in metric_buckets.items()}


def run_epoch(model, loader, optimizer, device, root_relative: bool, grad_clip_norm: float | None = None) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    metrics: Dict[str, list[float]] = {
        "loss": [],
        "absolute_mpjpe": [],
        "root_relative_mpjpe": [],
        "mpjpe": [],
    }
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch["ray_tokens"], view_mask=batch["view_mask"])
            losses = compute_losses(outputs["pred_3d"], batch["target_3d"], root_relative=root_relative)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        step_metrics = compute_metrics(outputs["pred_3d"].detach(), batch["target_3d"].detach())
        metrics["loss"].append(float(losses["loss"].detach().cpu()))
        metrics["absolute_mpjpe"].append(float(losses["absolute_mpjpe"].detach().cpu()))
        metrics["root_relative_mpjpe"].append(float(losses["root_relative_mpjpe"].detach().cpu()))
        metrics["mpjpe"].append(step_metrics["mpjpe"])
    return _mean_metrics(metrics)
