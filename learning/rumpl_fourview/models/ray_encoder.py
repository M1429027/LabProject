from __future__ import annotations

import torch.nn as nn


class RayTokenEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, ray_tokens):
        return self.net(ray_tokens)
