"""Train Stage A supervised multi-view 2D-to-3D warm-up model."""

from __future__ import annotations

import argparse
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
            if target_space_key in {"motion", "full_motion", "root_relative_plus_pelvis"}:
                losses = compute_motion_loss(
                    outputs=outputs,
                    batch=batch,
                    pose_weight=root_relative_weight,
                    pelvis_weight=absolute_weight,
                    final_weight=final_weight,
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
    pretrained_summary = load_optional_pretrained(model, training_cfg, device)
    if pretrained_summary.get("loaded"):
        print(json.dumps({"stage": "load_pretrained", **pretrained_summary}, ensure_ascii=False, indent=2))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

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

    for epoch in range(1, epochs + 1):
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
