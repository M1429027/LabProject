from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from learning.rumpl_fourview.models.vft_model import build_model_from_config


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not include an embedded config.")
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config, checkpoint
