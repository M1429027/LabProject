from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from learning.karate_selfcal.evaluation.compare_camera_geometry import build_comparison, relative_centers
from learning.karate_selfcal.evaluation.diagnose_static_extrinsic_refinement import (
    aggregate_camera_deltas,
    compute_method_metrics,
    corrected_payload_from_center_deltas,
    export_prediction_json,
    load_gt,
    load_json,
    save_json,
    triangulate_with_static_delta,
)
from learning.karate_selfcal.stage_a.dataset import KarateStageADataset, collate_stage_a
from learning.karate_selfcal.stage_a.model import build_model
from learning.karate_selfcal.stage_a.train import move_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep static camera-center correction aggregation/scale choices.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rough-extrinsics-json", required=True)
    parser.add_argument("--reference-extrinsics-json", required=True)
    parser.add_argument("--gt-zip", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--anchor-view", default="karate004_cam02")
    parser.add_argument("--view-ids", nargs="+", default=["karate004_cam02", "karate004_cam03", "karate004_cam07", "karate004_cam13"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trim-ratio", type=float, default=0.15)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=["mean", "median", "trimmed_mean", "weighted_mean", "weighted_trimmed_mean"],
    )
    parser.add_argument("--selection-metric", choices=["mean_center_error", "raw_mpjpe"], default="mean_center_error")
    return parser.parse_args()


def identity_id(person_id: str) -> int:
    normalized = str(person_id).lower()
    if normalized.endswith("01"):
        return 0
    if normalized.endswith("02"):
        return 1
    digits = "".join(ch for ch in normalized if ch.isdigit())
    return max(int(digits) - 1, 0) if digits else 0


def collect_model_outputs(args: argparse.Namespace) -> tuple[dict[int, list[np.ndarray]], dict[int, list[float]], list[dict[str, Any]], dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    dataset = KarateStageADataset(args.manifest, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_stage_a)
    device = resolve_device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    delta_by_view: dict[int, list[np.ndarray]] = defaultdict(list)
    weight_by_view: dict[int, list[float]] = defaultdict(list)
    batches: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch(batch, device)
            outputs = model(batch_dev["ray_tokens"], view_mask=batch_dev["view_mask"], joint_view_mask=batch_dev["joint_view_mask"])
            deltas = (outputs["pred_camera_origin_delta"] * batch_dev["ray_origin_scale"].reshape(-1, 1, 1)).detach().cpu().numpy()
            view_mask = batch_dev["view_mask"].detach().cpu().numpy().astype(bool)
            joint_view_mask = batch_dev["joint_view_mask"].detach().cpu().numpy().astype(bool)
            ray_conf = batch_dev["ray_tokens"][..., 8].detach().cpu().numpy()
            sample_weights = (ray_conf * joint_view_mask).sum(axis=1) / joint_view_mask.sum(axis=1).clip(min=1)
            for sample_index in range(deltas.shape[0]):
                for view_index in range(deltas.shape[1]):
                    if view_mask[sample_index, view_index]:
                        delta_by_view[view_index].append(deltas[sample_index, view_index].astype(np.float64))
                        weight_by_view[view_index].append(float(sample_weights[sample_index, view_index]))
            batches.append({"batch": batch, "batch_dev": batch_dev, "outputs": outputs})
    return delta_by_view, weight_by_view, batches, config


def build_static_delta(args: argparse.Namespace, delta_by_view: dict[int, list[np.ndarray]], weight_by_view: dict[int, list[float]], method: str, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    static_np = []
    stats = {}
    for view_index, view_id in enumerate(args.view_ids):
        arr = np.asarray(delta_by_view.get(view_index, []), dtype=np.float64)
        weights = np.asarray(weight_by_view.get(view_index, []), dtype=np.float64)
        value, view_stats = aggregate_camera_deltas(arr, weights, method, args.trim_ratio, 0.0)
        value = value * float(alpha)
        view_stats["alpha_scaled_delta_xyz_m"] = value.tolist()
        stats[view_id] = view_stats
        static_np.append(value)
    return np.asarray(static_np, dtype=np.float64), stats


def evaluate_candidate(args: argparse.Namespace, batches: list[dict[str, Any]], static_np: np.ndarray, gt: dict[int, dict[int, np.ndarray]], rough_payload: dict[str, Any], reference_payload: dict[str, Any]) -> dict[str, Any]:
    device = resolve_device(args.device)
    static_delta = torch.as_tensor(static_np.astype(np.float32), device=device)
    methods: dict[str, dict[tuple[int, int], np.ndarray]] = {"rough_anchor": {}, "static_corrected_anchor_only": {}, "static_corrected_anchor_plus_residual": {}}
    confidences: dict[tuple[int, int], np.ndarray] = {}
    with torch.no_grad():
        for item in batches:
            batch = item["batch"]
            batch_dev = item["batch_dev"]
            outputs = item["outputs"]
            static_anchor, _ = triangulate_with_static_delta(batch_dev, static_delta)
            final = static_anchor + outputs["pred_pose_root_relative"]
            rough = batch_dev["triangulated_3d"].detach().cpu().numpy()
            static_anchor_np = static_anchor.detach().cpu().numpy()
            final_np = final.detach().cpu().numpy()
            conf_np = batch_dev["target_confidence"].detach().cpu().numpy()
            for i in range(rough.shape[0]):
                key = (int(batch["frame_id"][i]), identity_id(str(batch["person_id"][i])))
                methods["rough_anchor"][key] = rough[i]
                methods["static_corrected_anchor_only"][key] = static_anchor_np[i]
                methods["static_corrected_anchor_plus_residual"][key] = final_np[i]
                confidences[key] = conf_np[i]
    corrected_payload = corrected_payload_from_center_deltas(rough_payload, args.view_ids, {v: static_np[i] for i, v in enumerate(args.view_ids)})
    rough_geometry = build_comparison(relative_centers(rough_payload, args.anchor_view), relative_centers(reference_payload, args.anchor_view), args.anchor_view)
    corrected_geometry = build_comparison(relative_centers(corrected_payload, args.anchor_view), relative_centers(reference_payload, args.anchor_view), args.anchor_view)
    return {
        "methods": methods,
        "confidences": confidences,
        "corrected_payload": corrected_payload,
        "rough_geometry": rough_geometry,
        "corrected_geometry": corrected_geometry,
        "gt_metrics": compute_method_metrics(methods, gt),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_by_view, weight_by_view, batches, _ = collect_model_outputs(args)
    gt = load_gt(args.gt_zip, args.sequence)
    rough_payload = load_json(args.rough_extrinsics_json)
    reference_payload = load_json(args.reference_extrinsics_json)

    candidates = []
    best = None
    best_eval = None
    for method in args.aggregations:
        for alpha in args.alphas:
            static_np, stats = build_static_delta(args, delta_by_view, weight_by_view, method, alpha)
            evaluated = evaluate_candidate(args, batches, static_np, gt, rough_payload, reference_payload)
            geom = evaluated["corrected_geometry"]["summary"]
            gt_all = evaluated["gt_metrics"]["static_corrected_anchor_plus_residual"]["all_joints_0_16"]
            row = {
                "aggregation": method,
                "alpha": float(alpha),
                "mean_center_error": geom.get("mean_center_error"),
                "median_center_error": geom.get("median_center_error"),
                "mean_direction_angle_error_deg": geom.get("mean_direction_angle_error_deg"),
                "raw_mpjpe_m": gt_all.get("raw_mpjpe_m"),
                "pelvis_aligned_mpjpe_m": gt_all.get("pelvis_aligned_mpjpe_m"),
                "similarity_aligned_mpjpe_m": gt_all.get("similarity_aligned_mpjpe_m"),
                "static_camera_correction_stats": stats,
            }
            candidates.append(row)
            score = row[args.selection_metric]
            if score is not None and (best is None or score < best[args.selection_metric]):
                best = row
                best_eval = evaluated
    if best is None or best_eval is None:
        raise RuntimeError("No valid center-correction candidate was evaluated.")

    best_name = f"{best['aggregation']}_alpha_{best['alpha']:.2f}".replace(".", "p")
    corrected_extrinsics_path = output_dir / f"static_center_sweep_best_{best_name}_extrinsics.json"
    prediction_json = output_dir / f"static_center_sweep_best_{best_name}_prediction.json"
    rough_json = output_dir / f"static_center_sweep_best_{best_name}_rough_anchor.json"
    center_only_json = output_dir / f"static_center_sweep_best_{best_name}_center_corrected_anchor.json"
    residual_json = output_dir / f"static_center_sweep_best_{best_name}_center_corrected_plus_residual.json"
    save_json(corrected_extrinsics_path, best_eval["corrected_payload"])
    common_metadata = {
        "stage": "static_center_correction_sweep_best",
        "best": best,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "rough_extrinsics": str(Path(args.rough_extrinsics_json).resolve()),
    }
    export_prediction_json(
        rough_json,
        best_eval["methods"]["rough_anchor"],
        best_eval["confidences"],
        sequence_name="harmony4d/karate004_rough_anchor",
        metadata={**common_metadata, "method": "rough_extrinsics_triangulation"},
    )
    export_prediction_json(
        center_only_json,
        best_eval["methods"]["static_corrected_anchor_only"],
        best_eval["confidences"],
        sequence_name="harmony4d/karate004_center_corrected_anchor",
        metadata={**common_metadata, "method": "center_corrected_extrinsics_triangulation"},
    )
    export_prediction_json(
        residual_json,
        best_eval["methods"]["static_corrected_anchor_plus_residual"],
        best_eval["confidences"],
        sequence_name="harmony4d/karate004_center_corrected_plus_residual",
        metadata={**common_metadata, "method": "center_corrected_plus_residual_model"},
    )
    # Backward-compatible alias used by previous runs.
    export_prediction_json(
        prediction_json,
        best_eval["methods"]["static_corrected_anchor_plus_residual"],
        best_eval["confidences"],
        sequence_name="harmony4d/karate004_static_center_sweep_best",
        metadata=common_metadata,
    )
    report = {
        "stage": "static_center_correction_sweep",
        "selection_metric": args.selection_metric,
        "rough_geometry_summary": best_eval["rough_geometry"]["summary"],
        "best_candidate": best,
        "best_gt_metrics": best_eval["gt_metrics"],
        "candidates": sorted(candidates, key=lambda x: (float("inf") if x[args.selection_metric] is None else x[args.selection_metric])),
        "paths": {
            "best_prediction_json": str(prediction_json),
            "rough_anchor_json": str(rough_json),
            "center_corrected_anchor_json": str(center_only_json),
            "center_corrected_plus_residual_json": str(residual_json),
            "best_corrected_extrinsics": str(corrected_extrinsics_path),
            "sweep_report": str(output_dir / "static_center_correction_sweep.json"),
        },
    }
    save_json(output_dir / "static_center_correction_sweep.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
