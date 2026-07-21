from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.rumpl_fourview.datasets.ray_dataset import build_dataloaders
from learning.rumpl_fourview.models.vft_model import build_model_from_config
from learning.rumpl_fourview.training.engine import run_epoch
from learning.rumpl_fourview.utils import ensure_dir, load_yaml, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the four-view RUMPL-style transformer.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    experiment_cfg = config.get("experiment", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    set_seed(int(experiment_cfg.get("seed", 42)))
    output_dir = ensure_dir(experiment_cfg.get("output_dir", "outputs/learning/rumpl_fourview/run"))
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    save_json(output_dir / "resolved_config.json", config)

    train_loader, val_loader = build_dataloaders(config)

    device = torch.device(training_cfg.get("device", "cpu"))
    model = build_model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(training_cfg.get("epochs", 5))
    root_relative = bool(data_cfg.get("root_relative", True))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))

    history = []
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            root_relative=root_relative,
            grad_clip_norm=grad_clip_norm,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                root_relative=root_relative,
                grad_clip_norm=None,
            )
        epoch_summary = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, indent=2))

        state = {
            "epoch": epoch,
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(state, checkpoints_dir / "latest.pt")
        if val_metrics["root_relative_mpjpe"] < best_val:
            best_val = val_metrics["root_relative_mpjpe"]
            torch.save(state, checkpoints_dir / "best.pt")

    save_json(output_dir / "history.json", {"epochs": history})
    print(f"Training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
