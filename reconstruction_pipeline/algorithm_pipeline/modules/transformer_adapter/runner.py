from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from learning.rumpl_fourview.inference.checkpoint_loader import load_checkpoint


class TransformerRunner:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.config, self.checkpoint = load_checkpoint(checkpoint_path, device=device)

    def predict(self, ray_tokens: np.ndarray, view_mask: np.ndarray, batch_size: int = 16) -> np.ndarray:
        ray_tensor = torch.from_numpy(ray_tokens).float().to(self.device)
        view_mask_tensor = torch.from_numpy(view_mask.astype(np.bool_)).to(self.device)
        preds = []
        with torch.no_grad():
            for start in range(0, ray_tensor.shape[0], batch_size):
                end = start + batch_size
                outputs = self.model(ray_tensor[start:end], view_mask=view_mask_tensor[start:end])
                preds.append(outputs["pred_3d"].cpu())
        return torch.cat(preds, dim=0).numpy()
