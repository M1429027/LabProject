from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from learning.rumpl_fourview import FEATURE_DIM, MAX_VIEWS, MODEL_JOINT_NAMES
from learning.rumpl_fourview.models.heads import JointRegressionHead
from learning.rumpl_fourview.models.ray_encoder import RayTokenEncoder
from learning.rumpl_fourview.models.transformer_blocks import TransformerBlock


class RumplFourViewTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        max_views: int = MAX_VIEWS,
        d_model: int = 64,
        num_heads: int = 8,
        view_fusion_depth: int = 4,
        joint_transformer_depth: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_views = max_views
        self.d_model = d_model
        self.num_joints = len(MODEL_JOINT_NAMES)

        self.ray_encoder = RayTokenEncoder(input_dim=input_dim, d_model=d_model, dropout=dropout)
        self.fused_ray_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.view_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(view_fusion_depth)
            ]
        )
        self.joint_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(joint_transformer_depth)
            ]
        )
        self.head = JointRegressionHead(d_model=d_model, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.fused_ray_token, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, ray_tokens: torch.Tensor, view_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if ray_tokens.ndim != 4:
            raise ValueError(f"Expected ray_tokens with shape (B, J, V, F), got {tuple(ray_tokens.shape)}.")
        batch_size, num_joints, num_views, _ = ray_tokens.shape
        if num_joints != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {num_joints}.")
        if num_views > self.max_views:
            raise ValueError(f"Expected <= {self.max_views} views, got {num_views}.")

        encoded = self.ray_encoder(ray_tokens)
        encoded = encoded.reshape(batch_size * num_joints, num_views, self.d_model)

        fused = self.fused_ray_token.expand(batch_size * num_joints, -1, -1)
        view_tokens = torch.cat([fused, encoded], dim=1)

        key_padding_mask = None
        if view_mask is not None:
            if view_mask.ndim != 2 or view_mask.shape != (batch_size, num_views):
                raise ValueError(
                    f"view_mask must have shape (B, V) matching the input. Got {tuple(view_mask.shape)} for {(batch_size, num_views)}."
                )
            inactive_mask = ~view_mask.bool()
            inactive_mask = inactive_mask.repeat_interleave(num_joints, dim=0)
            fused_mask = torch.zeros((inactive_mask.shape[0], 1), dtype=torch.bool, device=inactive_mask.device)
            key_padding_mask = torch.cat([fused_mask, inactive_mask], dim=1)

        for block in self.view_blocks:
            view_tokens = block(view_tokens, key_padding_mask=key_padding_mask)

        fused_joints = view_tokens[:, 0, :].reshape(batch_size, num_joints, self.d_model)
        joint_tokens = fused_joints
        for block in self.joint_blocks:
            joint_tokens = block(joint_tokens)

        pred_3d = self.head(joint_tokens)
        return {
            "pred_3d": pred_3d,
            "joint_tokens": joint_tokens,
        }


def build_model_from_config(config: Dict[str, object]) -> RumplFourViewTransformer:
    model_cfg = config.get("model", {})
    return RumplFourViewTransformer(
        input_dim=int(model_cfg.get("input_dim", FEATURE_DIM)),
        max_views=int(model_cfg.get("max_views", MAX_VIEWS)),
        d_model=int(model_cfg.get("d_model", 64)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        view_fusion_depth=int(model_cfg.get("view_fusion_depth", 4)),
        joint_transformer_depth=int(model_cfg.get("joint_transformer_depth", 4)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 2.0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
