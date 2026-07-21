"""Diagnose camera-head supervision, generalization, bias, and stability."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from learning.karate_selfcal.stage_a.dataset import (
    KarateStageAClipDataset,
    KarateStageADataset,
    collate_stage_a,
)
from learning.karate_selfcal.stage_a.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--head", choices=["rotation", "center"])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def cosine_similarity(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sum(prediction * target, axis=-1)
    denominator = np.linalg.norm(prediction, axis=-1) * np.linalg.norm(target, axis=-1)
    return numerator / np.maximum(denominator, 1e-9)


def summarize_noise(manifest: dict[str, Any]) -> dict[str, Any]:
    rotation = []
    center = []
    for record in manifest.get("metadata", {}).get("noise", []):
        for view_index, view in enumerate(record.get("views", [])):
            if view_index == int(manifest.get("metadata", {}).get("anchor_index", 0)):
                continue
            rotation.append(float(view.get("rotation_noise_deg", 0.0)))
            center.append(float(view.get("center_noise_m", 0.0)))
    return {
        "rotation_noise_deg": {
            "mean": safe_mean(rotation),
            "p90": float(np.percentile(rotation, 90)) if rotation else 0.0,
            "max": max(rotation, default=0.0),
        },
        "center_noise_m": {
            "mean": safe_mean(center),
            "p90": float(np.percentile(center, 90)) if center else 0.0,
            "max": max(center, default=0.0),
        },
    }


def split_integrity(manifest: dict[str, Any]) -> dict[str, Any]:
    sequences_by_split: dict[str, set[str]] = defaultdict(set)
    samples_by_split: dict[str, int] = defaultdict(int)
    for entry in manifest.get("entries", []):
        split = str(entry.get("split"))
        sequences_by_split[split].add(str(entry.get("sequence")))
        samples_by_split[split] += 1
    overlaps = {}
    split_names = sorted(sequences_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlaps[f"{left}_{right}"] = sorted(sequences_by_split[left] & sequences_by_split[right])
    return {
        "samples": dict(samples_by_split),
        "sequences": {key: len(value) for key, value in sequences_by_split.items()},
        "sequence_overlaps": overlaps,
        "has_sequence_leakage": any(overlaps.values()),
    }


def variant_from_sample_id(sample_id: str) -> int:
    match = re.search(r"_v(\d+)_", sample_id)
    return int(match.group(1)) if match else 0


def evaluate_split(
    model: torch.nn.Module,
    manifest_path: str,
    split: str,
    mode: str,
    anchor_index: int,
    batch_size: int,
    device: torch.device,
    data_cfg: dict[str, Any],
) -> dict[str, Any]:
    clip_length = int(data_cfg.get("clip_length", 1))
    if clip_length > 1:
        dataset = KarateStageAClipDataset(
            manifest_path,
            split=split,
            clip_length=clip_length,
            clip_stride=int(data_cfg.get("clip_stride", max(1, clip_length // 2))),
        )
        sources = np.asarray([
            str(dataset.frames.entries[indexes[0]].get("noise_source", mode))
            for _, indexes in dataset.clips
        ])
    else:
        dataset = KarateStageADataset(manifest_path, split=split)
        sources = np.asarray([str(entry.get("noise_source", mode)) for entry in dataset.entries])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_stage_a,
    )
    prediction_key = "pred_camera_rotation_delta" if mode == "rotation" else "pred_camera_origin_delta"
    target_key = "camera_rotation_delta" if mode == "rotation" else "camera_origin_delta"
    all_predictions = []
    all_targets = []
    all_masks = []
    all_scales = []
    group_records: dict[tuple[str, int], dict[str, list[np.ndarray]]] = defaultdict(lambda: {"pred": [], "target": [], "scale": []})

    model.eval()
    with torch.no_grad():
        for batch in loader:
            ray_tokens = batch["ray_tokens"].to(device)
            outputs = model(
                ray_tokens,
                view_mask=batch["view_mask"].to(device),
                joint_view_mask=batch["joint_view_mask"].to(device),
            )
            prediction = outputs[prediction_key].cpu().numpy()
            target = batch[target_key].numpy()
            mask = batch["camera_origin_delta_valid"].numpy().astype(bool)
            scales = batch["ray_origin_scale"].numpy()
            all_predictions.append(prediction)
            all_targets.append(target)
            all_masks.append(mask)
            all_scales.append(scales)
            for index, sample_id in enumerate(batch["sample_id"]):
                key = (str(batch["sequence"][index]), variant_from_sample_id(str(sample_id)))
                group_records[key]["pred"].append(prediction[index])
                group_records[key]["target"].append(target[index])
                group_records[key]["scale"].append(np.asarray(scales[index]))

    prediction = np.concatenate(all_predictions)
    target = np.concatenate(all_targets)
    mask = np.concatenate(all_masks)
    scales = np.concatenate(all_scales)
    non_anchor_mask = mask.copy()
    non_anchor_mask[:, anchor_index] = False
    error = np.linalg.norm(prediction - target, axis=-1)
    zero_error = np.linalg.norm(target, axis=-1)
    pred_norm = np.linalg.norm(prediction, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    cosine = cosine_similarity(prediction, target)

    per_view = []
    for view_index in range(prediction.shape[1]):
        view_mask = mask[:, view_index]
        per_view.append({
            "view_index": view_index,
            "is_anchor": view_index == anchor_index,
            "zero_error": safe_mean(zero_error[:, view_index][view_mask].tolist()),
            "model_error": safe_mean(error[:, view_index][view_mask].tolist()),
            "prediction_norm": safe_mean(pred_norm[:, view_index][view_mask].tolist()),
            "target_norm": safe_mean(target_norm[:, view_index][view_mask].tolist()),
            "cosine": safe_mean(cosine[:, view_index][view_mask & (target_norm[:, view_index] > 1e-8)].tolist()),
        })

    aggregated_errors = []
    frame_stability = []
    target_stability = []
    aggregated_errors_metric = []
    for records in group_records.values():
        pred_frames = np.stack(records["pred"])
        target_frames = np.stack(records["target"])
        pred_mean = pred_frames.mean(axis=0)
        target_mean = target_frames.mean(axis=0)
        aggregated_errors.extend(np.linalg.norm(pred_mean - target_mean, axis=-1)[1:].tolist())
        frame_stability.extend(np.linalg.norm(pred_frames - pred_mean, axis=-1)[:, 1:].reshape(-1).tolist())
        target_stability.extend(np.linalg.norm(target_frames - target_mean, axis=-1)[:, 1:].reshape(-1).tolist())
        if mode == "center":
            scale = float(np.mean(records["scale"]))
            aggregated_errors_metric.extend((np.linalg.norm(pred_mean - target_mean, axis=-1)[1:] * scale).tolist())

    zero_non_anchor = safe_mean(zero_error[non_anchor_mask].tolist())
    model_non_anchor = safe_mean(error[non_anchor_mask].tolist())
    component_correlation = []
    for component in range(3):
        pred_component = prediction[..., component][non_anchor_mask]
        target_component = target[..., component][non_anchor_mask]
        if np.std(pred_component) < 1e-12 or np.std(target_component) < 1e-12:
            component_correlation.append(0.0)
        else:
            component_correlation.append(float(np.corrcoef(pred_component, target_component)[0, 1]))

    per_source = {}
    for source in sorted(set(sources.tolist())):
        source_mask = non_anchor_mask & (sources == source)[:, None]
        source_zero = safe_mean(zero_error[source_mask].tolist())
        source_model = safe_mean(error[source_mask].tolist())
        source_report = {
            "num_samples": int(np.sum(sources == source)),
            "zero_error_non_anchor": source_zero,
            "model_error_non_anchor": source_model,
            "improvement_percent": 100.0 * (source_zero - source_model) / max(source_zero, 1e-9),
        }
        if mode == "center":
            scale_matrix = np.broadcast_to(scales[:, None], error.shape)
            source_report["zero_error_m"] = safe_mean((zero_error * scale_matrix)[source_mask].tolist())
            source_report["model_error_m"] = safe_mean((error * scale_matrix)[source_mask].tolist())
        per_source[source] = source_report

    return {
        "num_samples": len(dataset),
        "zero_error_non_anchor": zero_non_anchor,
        "model_error_non_anchor": model_non_anchor,
        "improvement_percent": 100.0 * (zero_non_anchor - model_non_anchor) / max(zero_non_anchor, 1e-9),
        "prediction_target_cosine_non_anchor": safe_mean(cosine[non_anchor_mask & (target_norm > 1e-8)].tolist()),
        "component_correlation": component_correlation,
        "anchor_prediction_norm": safe_mean(pred_norm[:, anchor_index][mask[:, anchor_index]].tolist()),
        "sequence_aggregated_error_non_anchor": safe_mean(aggregated_errors),
        "sequence_aggregated_error_m": safe_mean(aggregated_errors_metric),
        "prediction_frame_std": safe_mean(frame_stability),
        "target_frame_std": safe_mean(target_stability),
        "per_view": per_view,
        "per_source": per_source,
        "mean_ray_origin_scale_m": float(np.mean(scales)),
    }


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    mode = str(args.head or manifest.get("metadata", {}).get("mode"))
    if mode not in {"rotation", "center"}:
        raise ValueError("Mixed manifests require --head rotation or --head center.")
    anchor_index = int(manifest.get("metadata", {}).get("anchor_index", 0))
    device = torch.device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "mode": mode,
        "anchor_index": anchor_index,
        "split_integrity": split_integrity(manifest),
        "noise_distribution": summarize_noise(manifest),
        "splits": {},
    }
    for split in args.splits:
        report["splits"][split] = evaluate_split(
            model,
            args.manifest,
            split,
            mode,
            anchor_index,
            args.batch_size,
            device,
            config.get("data", {}),        )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
