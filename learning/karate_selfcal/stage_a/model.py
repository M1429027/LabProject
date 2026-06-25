"""Small supervised Stage A model for multi-view 2D-to-3D warm-up."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class StageARayFusionModel(nn.Module):
    """Fuse four view tokens per joint, then model the 17-joint skeleton."""

    def __init__(
        self,
        input_dim: int = 7,
        num_joints: int = 17,
        max_views: int = 4,
        d_model: int = 96,
        num_heads: int = 8,
        view_depth: int = 3,
        joint_depth: int = 3,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_joints = int(num_joints)
        self.max_views = int(max_views)
        self.d_model = int(d_model)
        self.token_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.fused_view_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.view_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(view_depth)]
        )
        self.joint_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(joint_depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.pelvis_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.fused_view_token, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        ray_tokens: torch.Tensor,
        view_mask: torch.Tensor | None = None,
        joint_view_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if ray_tokens.ndim != 4:
            raise ValueError(f"ray_tokens must be [B,J,V,F], got {tuple(ray_tokens.shape)}")
        batch, joints, views, features = ray_tokens.shape
        if joints != self.num_joints or views > self.max_views or features != self.input_dim:
            raise ValueError(
                f"Expected [B,{self.num_joints},<={self.max_views},{self.input_dim}], "
                f"got {tuple(ray_tokens.shape)}"
            )

        encoded = self.token_encoder(ray_tokens).reshape(batch * joints, views, self.d_model)
        fused = self.fused_view_token.expand(batch * joints, -1, -1)
        view_tokens = torch.cat([fused, encoded], dim=1)

        key_padding_mask = None
        if joint_view_mask is not None:
            if joint_view_mask.shape != (batch, joints, views):
                raise ValueError(
                    f"joint_view_mask must be [B,J,V], got {tuple(joint_view_mask.shape)}"
                )
            inactive = ~joint_view_mask.bool().reshape(batch * joints, views)
            fused_mask = torch.zeros((batch * joints, 1), dtype=torch.bool, device=ray_tokens.device)
            key_padding_mask = torch.cat([fused_mask, inactive], dim=1)
        elif view_mask is not None:
            if view_mask.shape != (batch, views):
                raise ValueError(f"view_mask must be [B,V], got {tuple(view_mask.shape)}")
            inactive = ~view_mask.bool().repeat_interleave(joints, dim=0)
            fused_mask = torch.zeros((batch * joints, 1), dtype=torch.bool, device=ray_tokens.device)
            key_padding_mask = torch.cat([fused_mask, inactive], dim=1)

        for block in self.view_blocks:
            view_tokens = block(view_tokens, key_padding_mask=key_padding_mask)

        joint_tokens = view_tokens[:, 0, :].reshape(batch, joints, self.d_model)
        for block in self.joint_blocks:
            joint_tokens = block(joint_tokens)

        pred_pose_root_relative = self.head(joint_tokens)
        global_token = joint_tokens.mean(dim=1)
        pred_pelvis = self.pelvis_head(global_token)
        pred_3d = pred_pose_root_relative + pred_pelvis.unsqueeze(1)
        return {
            "pred_3d": pred_3d,
            "pred_pose_root_relative": pred_pose_root_relative,
            "pred_pelvis": pred_pelvis,
            "joint_tokens": joint_tokens,
            "global_token": global_token,
        }


def build_model(config: dict[str, Any]) -> StageARayFusionModel:
    cfg = config.get("model", {})
    return StageARayFusionModel(
        input_dim=int(cfg.get("input_dim", 7)),
        num_joints=int(cfg.get("num_joints", 17)),
        max_views=int(cfg.get("max_views", 4)),
        d_model=int(cfg.get("d_model", 96)),
        num_heads=int(cfg.get("num_heads", 8)),
        view_depth=int(cfg.get("view_depth", 3)),
        joint_depth=int(cfg.get("joint_depth", 3)),
        mlp_ratio=float(cfg.get("mlp_ratio", 2.0)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
