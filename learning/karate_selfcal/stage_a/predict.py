"""Export Stage A model predictions into the shared 3D skeleton JSON format."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import KarateStageADataset, collate_stage_a
from .model import build_model
from .train import load_yaml, move_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage A predictions for review rendering.")
    parser.add_argument("--config", required=True, help="Stage A training config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, usually checkpoints/best.pt.")
    parser.add_argument(
        "--pelvis-checkpoint",
        default=None,
        help="Optional checkpoint used only for the pelvis/global translation head.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to export.")
    parser.add_argument("--sequence", default=None, help="Optional sequence filter, e.g. 11_karate3/008_karate3.")
    parser.add_argument("--output-json", required=True, help="Output prediction JSON path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size.")
    parser.add_argument(
        "--identity-spacing",
        type=float,
        default=1.2,
        help="Display spacing used when exporting root-relative predictions.",
    )
    return parser.parse_args()


def identity_id(person_id: str) -> int:
    normalized = person_id.lower()
    if normalized.endswith("01"):
        return 0
    if normalized.endswith("02"):
        return 1
    digits = "".join(ch for ch in normalized if ch.isdigit())
    return max(int(digits) - 1, 0) if digits else 0


def joint_payload(
    prediction: torch.Tensor,
    confidence: torch.Tensor,
    display_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[dict[str, float | int]]:
    joints = []
    offset = torch.tensor(display_offset, dtype=prediction.dtype)
    for joint_id, point in enumerate(prediction.detach().cpu().tolist()):
        if float(confidence[joint_id].detach().cpu()) <= 0.0:
            continue
        shifted = torch.tensor(point, dtype=prediction.dtype) + offset
        joints.append(
            {
                "id": int(joint_id),
                "x": float(shifted[0]),
                "y": float(shifted[1]),
                "z": float(shifted[2]),
                "confidence": float(confidence[joint_id].detach().cpu()),
            }
        )
    return joints


def load_checkpoint_model(config: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    target_space = str(loss_cfg.get("target_space", "absolute")).lower()

    dataset = KarateStageADataset(data_cfg["manifest_path"], split=args.split)
    if args.sequence is not None:
        dataset.entries = [entry for entry in dataset.entries if str(entry.get("sequence")) == args.sequence]
        if not dataset.entries:
            raise ValueError(f"No entries found for split={args.split!r}, sequence={args.sequence!r}")

    batch_size = int(args.batch_size or data_cfg.get("batch_size", 64))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_stage_a,
    )

    device = resolve_device(str(training_cfg.get("device", "auto")))
    model = load_checkpoint_model(config, args.checkpoint, device)
    pelvis_model = (
        load_checkpoint_model(config, args.pelvis_checkpoint, device)
        if args.pelvis_checkpoint is not None
        else None
    )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            batch_on_device = move_batch(batch, device)
            outputs = model(
                batch_on_device["ray_tokens"],
                view_mask=batch_on_device["view_mask"],
                joint_view_mask=batch_on_device["joint_view_mask"],
            )
            if pelvis_model is not None:
                pelvis_outputs = pelvis_model(
                    batch_on_device["ray_tokens"],
                    view_mask=batch_on_device["view_mask"],
                    joint_view_mask=batch_on_device["joint_view_mask"],
                )
                predictions = (
                    outputs["pred_pose_root_relative"] + pelvis_outputs["pred_pelvis"].unsqueeze(1)
                ).cpu()
            else:
                predictions = outputs["pred_3d"].cpu()
            confidences = batch["target_confidence"].cpu()
            for index in range(predictions.shape[0]):
                sequence = str(batch["sequence"][index])
                frame_id = int(batch["frame_id"][index])
                person_id = str(batch["person_id"][index])
                ident_id = identity_id(person_id)
                # Only separate identities for pure root-relative display exports.
                # If a pelvis checkpoint is provided, predictions already include global translation.
                use_display_spacing = pelvis_model is None and target_space in {
                    "root_relative",
                    "root-relative",
                    "relative",
                }
                display_offset = (ident_id * float(args.identity_spacing), 0.0, 0.0) if use_display_spacing else (0.0, 0.0, 0.0)
                grouped[(sequence, frame_id)].append(
                    {
                        "identity_id": ident_id,
                        "person_id": person_id,
                        "joints": joint_payload(predictions[index], confidences[index], display_offset=display_offset),
                    }
                )

    frames = []
    for (sequence, frame_id), identities in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        frames.append(
            {
                "sequence": sequence,
                "frame": int(frame_id),
                "identities": sorted(identities, key=lambda item: int(item["identity_id"])),
            }
        )

    payload = {
        "metadata": {
            "stage": "stage_a_prediction",
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "pelvis_checkpoint": str(Path(args.pelvis_checkpoint).resolve()) if args.pelvis_checkpoint else None,
            "manifest": str(Path(data_cfg["manifest_path"]).resolve()),
            "split": args.split,
            "sequence": args.sequence,
            "target_space": target_space,
            "identity_spacing": float(args.identity_spacing),
            "num_frames": len(frames),
        },
        "frames": frames,
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "num_frames": len(frames)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
