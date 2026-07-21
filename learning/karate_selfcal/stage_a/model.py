"""Small supervised Stage A model for multi-view 2D-to-3D warm-up."""

from __future__ import annotations

import math
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
        temporal_depth: int = 0,
        camera_depth: int = 0,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        camera_anchor_index: int = 0,
        fix_camera_anchor: bool = True,
        minimum_active_correction: float = 0.15,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_joints = int(num_joints)
        self.max_views = int(max_views)
        self.d_model = int(d_model)
        self.camera_anchor_index = int(camera_anchor_index)
        self.fix_camera_anchor = bool(fix_camera_anchor)
        self.minimum_active_correction = float(min(max(minimum_active_correction, 0.0), 1.0))
        self.temporal_depth = int(temporal_depth)
        self.camera_depth = int(camera_depth)
        if not 0 <= self.camera_anchor_index < self.max_views:
            raise ValueError(
                f"camera_anchor_index must be in [0, {self.max_views}), "
                f"got {self.camera_anchor_index}"
            )
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
        self.temporal_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(self.temporal_depth)]
        )
        self.camera_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(self.camera_depth)]
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
        self.camera_delta_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.camera_translation_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.camera_rotation_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.camera_scale_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.camera_noop_gate_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.camera_correction_gate_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.fused_view_token, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Start conservatively: the model must earn a non-zero camera correction.
        self.camera_noop_gate_head[-1].bias.data.fill_(1.0)
        self.camera_correction_gate_head[-1].bias.data.fill_(-0.5)

    def forward(
        self,
        ray_tokens: torch.Tensor,
        view_mask: torch.Tensor | None = None,
        joint_view_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if ray_tokens.ndim == 5:
            return self._forward_clip(ray_tokens, view_mask, joint_view_mask)
        if ray_tokens.ndim != 4:
            raise ValueError(f"ray_tokens must be [B,J,V,F] or [B,T,J,V,F], got {tuple(ray_tokens.shape)}")
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

        per_view_tokens = view_tokens[:, 1:, :].reshape(batch, joints, views, self.d_model)
        if joint_view_mask is not None:
            view_joint_weights = joint_view_mask.bool().float().permute(0, 2, 1).unsqueeze(-1)
            per_view_for_head = per_view_tokens.permute(0, 2, 1, 3)
            camera_tokens = (per_view_for_head * view_joint_weights).sum(dim=2) / view_joint_weights.sum(dim=2).clamp_min(1.0)
        elif view_mask is not None:
            camera_tokens = per_view_tokens.mean(dim=1)
            camera_tokens = camera_tokens * view_mask.bool().float().unsqueeze(-1)
        else:
            camera_tokens = per_view_tokens.mean(dim=1)
        camera_outputs = self._predict_camera(camera_tokens, ray_tokens)
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
            **camera_outputs,
            "joint_tokens": joint_tokens,
            "global_token": global_token,
            "camera_tokens": camera_tokens,
        }

    def _predict_camera(
        self,
        camera_tokens: torch.Tensor,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        views = camera_tokens.shape[1]
        raw_outputs = {
            "pred_camera_origin_delta": self.camera_delta_head(camera_tokens),
            "pred_camera_translation_delta": self.camera_translation_head(camera_tokens),
            "pred_camera_rotation_delta": self.camera_rotation_head(camera_tokens),
            "pred_camera_scale_delta": self.camera_scale_head(camera_tokens).squeeze(-1),
        }
        noop_probability = torch.sigmoid(self.camera_noop_gate_head(camera_tokens).squeeze(-1))
        correction_gate = torch.sigmoid(self.camera_correction_gate_head(camera_tokens).squeeze(-1))
        active_gate = self.minimum_active_correction + (1.0 - self.minimum_active_correction) * correction_gate
        effective_gate = (1.0 - noop_probability) * active_gate
        outputs = {"raw_" + key: value for key, value in raw_outputs.items()}
        outputs.update({
            "pred_camera_origin_delta": raw_outputs["pred_camera_origin_delta"] * effective_gate.unsqueeze(-1),
            "pred_camera_translation_delta": raw_outputs["pred_camera_translation_delta"] * effective_gate.unsqueeze(-1),
            "pred_camera_rotation_delta": raw_outputs["pred_camera_rotation_delta"] * effective_gate.unsqueeze(-1),
            "pred_camera_scale_delta": raw_outputs["pred_camera_scale_delta"] * effective_gate,
            "pred_camera_noop_probability": noop_probability,
            "pred_camera_correction_gate": correction_gate,
            "pred_camera_effective_gate": effective_gate,
        })
        if not self.fix_camera_anchor:
            return outputs
        if self.camera_anchor_index >= views:
            raise ValueError(
                f"camera_anchor_index {self.camera_anchor_index} is unavailable for {views} views"
            )
        mask = torch.ones((1, views, 1), dtype=reference.dtype, device=reference.device)
        mask[:, self.camera_anchor_index] = 0.0
        for key in (
            "pred_camera_origin_delta",
            "pred_camera_translation_delta",
            "pred_camera_rotation_delta",
        ):
            outputs[key] = outputs[key] * mask
        outputs["pred_camera_scale_delta"] = outputs["pred_camera_scale_delta"] * mask.squeeze(-1)
        outputs["pred_camera_noop_probability"] = (
            outputs["pred_camera_noop_probability"] * mask.squeeze(-1) + (1.0 - mask.squeeze(-1))
        )
        outputs["pred_camera_correction_gate"] = outputs["pred_camera_correction_gate"] * mask.squeeze(-1)
        outputs["pred_camera_effective_gate"] = outputs["pred_camera_effective_gate"] * mask.squeeze(-1)
        return outputs

    def _temporal_position_encoding(
        self,
        length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / self.d_model)
        )
        encoding = torch.zeros((length, self.d_model), dtype=torch.float32, device=device)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=dtype)

    def _forward_clip(
        self,
        ray_tokens: torch.Tensor,
        view_mask: torch.Tensor | None,
        joint_view_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch, frames, joints, views, features = ray_tokens.shape
        if self.temporal_depth <= 0:
            raise ValueError("5D clip input requires model.temporal_depth > 0")
        if view_mask is not None and view_mask.shape != (batch, frames, views):
            raise ValueError(f"clip view_mask must be [B,T,V], got {tuple(view_mask.shape)}")
        if joint_view_mask is not None and joint_view_mask.shape != (batch, frames, joints, views):
            raise ValueError(
                f"clip joint_view_mask must be [B,T,J,V], got {tuple(joint_view_mask.shape)}"
            )

        flat_outputs = self.forward(
            ray_tokens.reshape(batch * frames, joints, views, features),
            None if view_mask is None else view_mask.reshape(batch * frames, views),
            None
            if joint_view_mask is None
            else joint_view_mask.reshape(batch * frames, joints, views),
        )
        frame_camera_tokens = flat_outputs["camera_tokens"].reshape(
            batch, frames, views, self.d_model
        )
        temporal_tokens = frame_camera_tokens.permute(0, 2, 1, 3).reshape(
            batch * views, frames, self.d_model
        )
        temporal_tokens = temporal_tokens + self._temporal_position_encoding(
            frames, temporal_tokens.dtype, temporal_tokens.device
        ).unsqueeze(0)

        if joint_view_mask is not None:
            temporal_valid = joint_view_mask.bool().any(dim=2)
            temporal_weight = joint_view_mask.float().mean(dim=2)
        elif view_mask is not None:
            temporal_valid = view_mask.bool()
            temporal_weight = temporal_valid.float()
        else:
            temporal_valid = torch.ones(
                (batch, frames, views), dtype=torch.bool, device=ray_tokens.device
            )
            temporal_weight = temporal_valid.float()
        temporal_valid_flat = temporal_valid.permute(0, 2, 1).reshape(batch * views, frames)
        temporal_padding = ~temporal_valid_flat
        all_invalid = temporal_padding.all(dim=1)
        if all_invalid.any():
            temporal_padding = temporal_padding.clone()
            temporal_padding[all_invalid, 0] = False
        for block in self.temporal_blocks:
            temporal_tokens = block(temporal_tokens, key_padding_mask=temporal_padding)

        weights = temporal_weight.permute(0, 2, 1).reshape(batch * views, frames, 1)
        clip_camera_tokens = (temporal_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        clip_camera_tokens = clip_camera_tokens.reshape(batch, views, self.d_model)
        camera_valid = temporal_valid.any(dim=1)
        camera_padding = ~camera_valid
        all_cameras_invalid = camera_padding.all(dim=1)
        if all_cameras_invalid.any():
            camera_padding = camera_padding.clone()
            camera_padding[all_cameras_invalid, 0] = False
        for block in self.camera_blocks:
            clip_camera_tokens = block(clip_camera_tokens, key_padding_mask=camera_padding)
        clip_camera_tokens = clip_camera_tokens * camera_valid.float().unsqueeze(-1)

        center_index = frames // 2
        center_outputs: dict[str, torch.Tensor] = {}
        for key, value in flat_outputs.items():
            if key.startswith("pred_camera_") or key == "camera_tokens":
                continue
            center_outputs[key] = value.reshape(batch, frames, *value.shape[1:])[:, center_index]
        return {
            **center_outputs,
            **self._predict_camera(clip_camera_tokens, ray_tokens),
            "camera_tokens": clip_camera_tokens,
            "frame_camera_tokens": frame_camera_tokens,
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
        temporal_depth=int(cfg.get("temporal_depth", 0)),
        camera_depth=int(cfg.get("camera_depth", 0)),
        mlp_ratio=float(cfg.get("mlp_ratio", 2.0)),
        dropout=float(cfg.get("dropout", 0.1)),
        camera_anchor_index=int(cfg.get("camera_anchor_index", 0)),
        fix_camera_anchor=bool(cfg.get("fix_camera_anchor", True)),
    )
