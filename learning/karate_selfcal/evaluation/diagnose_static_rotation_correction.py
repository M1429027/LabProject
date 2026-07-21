from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from learning.karate_selfcal.evaluation.diagnose_static_full_extrinsic_refinement import (
    ident,
    load_gt,
    rotvec_to_matrix,
    umeyama,
)
from learning.karate_selfcal.evaluation.diagnose_static_extrinsic_refinement import export_prediction_json
from learning.karate_selfcal.stage_a.dataset import KarateStageADataset, collate_stage_a
from learning.karate_selfcal.stage_a.model import build_model
from learning.karate_selfcal.stage_a.train import move_batch, resolve_device


COMPOSITIONS = (
    "right_transpose",
    "right",
    "left",
    "left_transpose",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a static camera-rotation head independently of center and scale correction."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rough-extrinsics-json", required=True)
    parser.add_argument("--reference-extrinsics-json", required=True)
    parser.add_argument("--gt-zip", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--anchor-view", default="karate004_cam02")
    parser.add_argument(
        "--view-ids",
        nargs="+",
        default=["karate004_cam02", "karate004_cam03", "karate004_cam07", "karate004_cam13"],
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--aggregations",
        nargs="+",
        choices=["mean", "median", "weighted_mean"],
        default=["mean", "median", "weighted_mean"],
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rotation_angle_deg(estimated: np.ndarray, reference: np.ndarray) -> float:
    relative = project_to_rotation(estimated) @ project_to_rotation(reference).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def relative_rotations(payload: dict[str, Any], anchor_view: str) -> dict[str, np.ndarray]:
    rotations = {
        view_id: project_to_rotation(meta["rotation"])
        for view_id, meta in payload["extrinsics_by_view"].items()
    }
    anchor = rotations[anchor_view]
    return {view_id: rotation @ anchor.T for view_id, rotation in rotations.items()}


def aggregate(values: np.ndarray, weights: np.ndarray, method: str) -> np.ndarray:
    if method == "median":
        return np.median(values, axis=0)
    if method == "weighted_mean":
        safe_weights = np.clip(weights, 0.0, None)
        if float(safe_weights.sum()) > 1e-8:
            return np.average(values, axis=0, weights=safe_weights)
    return np.mean(values, axis=0)


def axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(rotvec, dtype=torch.float64).reshape(1, 3)
    return rotvec_to_matrix(tensor)[0].cpu().numpy()


def compose_rotation(rough_rotation: np.ndarray, correction: np.ndarray, mode: str) -> np.ndarray:
    if mode == "right_transpose":
        return rough_rotation @ correction.T
    if mode == "right":
        return rough_rotation @ correction
    if mode == "left":
        return correction @ rough_rotation
    if mode == "left_transpose":
        return correction.T @ rough_rotation
    raise ValueError(f"Unknown composition: {mode}")


def corrected_rotations(
    rough_relative: dict[str, np.ndarray],
    view_ids: list[str],
    rotvecs: np.ndarray,
    alpha: float,
    composition: str,
    anchor_view: str,
) -> dict[str, np.ndarray]:
    absolute = {}
    for index, view_id in enumerate(view_ids):
        correction = axis_angle_to_matrix(rotvecs[index] * float(alpha))
        absolute[view_id] = project_to_rotation(
            compose_rotation(rough_relative[view_id], correction, composition)
        )
    # Rotation has a global gauge. Compare every result in the corrected anchor frame.
    anchor = absolute[anchor_view]
    return {view_id: rotation @ anchor.T for view_id, rotation in absolute.items()}


def rotation_metrics(
    estimated: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    view_ids: list[str],
    anchor_view: str,
) -> dict[str, Any]:
    per_view = {}
    errors = []
    for view_id in view_ids:
        error = rotation_angle_deg(estimated[view_id], reference[view_id])
        per_view[view_id] = error
        if view_id != anchor_view:
            errors.append(error)
    return {
        "per_view_rotation_error_deg": per_view,
        "mean_non_anchor_rotation_error_deg": float(np.mean(errors)),
        "median_non_anchor_rotation_error_deg": float(np.median(errors)),
        "max_non_anchor_rotation_error_deg": float(np.max(errors)),
    }


def triangulate_with_rotations(
    batch: dict[str, Any],
    rough_relative: dict[str, np.ndarray],
    corrected_relative: dict[str, np.ndarray],
    view_ids: list[str],
) -> torch.Tensor:
    ray_tokens = batch["ray_tokens"]
    mask = batch["joint_view_mask"].bool()
    origin_norm = ray_tokens[..., 0:3]
    direction = torch.nn.functional.normalize(ray_tokens[..., 3:6], dim=-1)
    confidence = ray_tokens[..., 8].clamp_min(0.0)
    scale = batch["ray_origin_scale"].reshape(-1, 1, 1, 1)
    center = batch["ray_origin_center"].reshape(-1, 1, 1, 3)
    origin = origin_norm * scale + center

    direction_corrections = []
    for view_id in view_ids:
        # d_corrected = R_corrected.T R_rough d_rough
        direction_corrections.append(corrected_relative[view_id].T @ rough_relative[view_id])
    correction = torch.as_tensor(
        np.stack(direction_corrections), dtype=direction.dtype, device=direction.device
    ).reshape(1, 1, len(view_ids), 3, 3)
    direction = (correction @ direction.unsqueeze(-1)).squeeze(-1)
    direction = torch.nn.functional.normalize(direction, dim=-1)

    weights = (confidence * mask.float()).clamp_min(0.0)
    eye = torch.eye(3, dtype=ray_tokens.dtype, device=ray_tokens.device).reshape(1, 1, 1, 3, 3)
    projector = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    weighted_projector = projector * weights.unsqueeze(-1).unsqueeze(-1)
    lhs = weighted_projector.sum(dim=2)
    rhs = (weighted_projector @ origin.unsqueeze(-1)).sum(dim=2).squeeze(-1)
    valid = weights.gt(0.0).sum(dim=2) >= 2
    corrected = torch.linalg.solve(
        lhs + 1e-4 * eye.reshape(1, 1, 3, 3), rhs.unsqueeze(-1)
    ).squeeze(-1)
    return torch.where(valid.unsqueeze(-1), corrected, batch["triangulated_3d"])


def pelvis(joints: np.ndarray) -> np.ndarray:
    return 0.5 * (joints[11] + joints[12])


def summarize_predictions(
    predictions: dict[tuple[int, int], np.ndarray],
    gt: dict[int, dict[int, np.ndarray]],
) -> dict[str, Any]:
    raw_errors = []
    pelvis_errors = []
    similarity_errors = []
    person_frames = 0
    for (frame_id, person_id), prediction in sorted(predictions.items()):
        gt_frame = gt.get(frame_id + 1)
        if gt_frame is None or person_id not in gt_frame:
            continue
        target = gt_frame[person_id]
        valid = np.isfinite(prediction).all(axis=1) & np.isfinite(target).all(axis=1)
        if int(valid.sum()) < 4:
            continue
        pred_valid = prediction[valid]
        target_valid = target[valid]
        raw_errors.extend(np.linalg.norm(pred_valid - target_valid, axis=1).tolist())
        pred_pelvis = prediction - pelvis(prediction)
        target_pelvis = target - pelvis(target)
        pelvis_errors.extend(np.linalg.norm(pred_pelvis[valid] - target_pelvis[valid], axis=1).tolist())
        aligned = umeyama(pred_valid, target_valid)
        similarity_errors.extend(np.linalg.norm(aligned - target_valid, axis=1).tolist())
        person_frames += 1
    return {
        "num_person_frames": person_frames,
        "raw_mpjpe_m": float(np.mean(raw_errors)) if raw_errors else None,
        "pelvis_aligned_mpjpe_m": float(np.mean(pelvis_errors)) if pelvis_errors else None,
        "similarity_aligned_mpjpe_m": float(np.mean(similarity_errors)) if similarity_errors else None,
    }


def export_extrinsics(
    rough_payload: dict[str, Any],
    corrected_relative: dict[str, np.ndarray],
    view_ids: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(rough_payload))
    for view_id in view_ids:
        meta = payload["extrinsics_by_view"][view_id]
        rotation = corrected_relative[view_id]
        translation = np.asarray(meta["translation"], dtype=np.float64).reshape(3)
        old_rotation = project_to_rotation(meta["rotation"])
        center = -old_rotation.T @ translation
        new_translation = -rotation @ center
        meta["rotation"] = rotation.tolist()
        meta["translation"] = new_translation.tolist()
        meta["camera_center_world"] = center.tolist()
        if "projection_matrix" in meta:
            projection = np.asarray(meta["projection_matrix"], dtype=np.float64)
            if projection.shape == (3, 4):
                projection[:, :3] = rotation
                projection[:, 3] = new_translation
                meta["projection_matrix"] = projection.tolist()
    payload["stage"] = "rotation_only_audit"
    payload["rotation_audit"] = metadata
    return payload


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    dataset = KarateStageADataset(args.manifest, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_stage_a,
    )

    predicted_by_view: dict[int, list[np.ndarray]] = defaultdict(list)
    weight_by_view: dict[int, list[float]] = defaultdict(list)
    cached_batches: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch_device = move_batch(batch, device)
            outputs = model(
                batch_device["ray_tokens"],
                view_mask=batch_device["view_mask"],
                joint_view_mask=batch_device["joint_view_mask"],
            )
            predicted = outputs["pred_camera_rotation_delta"].cpu().numpy()
            view_mask = batch_device["view_mask"].cpu().numpy().astype(bool)
            joint_mask = batch_device["joint_view_mask"].bool()
            confidence = batch_device["ray_tokens"][..., 8].clamp_min(0.0)
            sample_weights = (
                (confidence * joint_mask.float()).sum(dim=1)
                / joint_mask.float().sum(dim=1).clamp_min(1.0)
            ).cpu().numpy()
            for batch_index in range(predicted.shape[0]):
                for view_index in range(predicted.shape[1]):
                    if view_mask[batch_index, view_index]:
                        predicted_by_view[view_index].append(predicted[batch_index, view_index])
                        weight_by_view[view_index].append(float(sample_weights[batch_index, view_index]))
            cached_batches.append(batch)

    rough_payload = load_json(args.rough_extrinsics_json)
    reference_payload = load_json(args.reference_extrinsics_json)
    rough_relative = relative_rotations(rough_payload, args.anchor_view)
    reference_relative = relative_rotations(reference_payload, args.anchor_view)
    gt = load_gt(args.gt_zip, args.sequence)
    view_ids = list(args.view_ids)

    rough_metrics = rotation_metrics(rough_relative, reference_relative, view_ids, args.anchor_view)
    rough_predictions = {}
    confidence_by_key = {}
    for batch in cached_batches:
        points = batch["triangulated_3d"].cpu().numpy()
        confidences = batch["target_confidence"].cpu().numpy()
        for index in range(points.shape[0]):
            key = (int(batch["frame_id"][index]), ident(batch["person_id"][index]))
            rough_predictions[key] = points[index]
            confidence_by_key[key] = confidences[index]
    candidates = []
    candidate_rotations: dict[tuple[str, str, float], dict[str, np.ndarray]] = {}
    aggregated_by_method = {}
    for aggregation in args.aggregations:
        rotvecs = np.vstack([
            aggregate(
                np.asarray(predicted_by_view[index]),
                np.asarray(weight_by_view[index]),
                aggregation,
            )
            for index in range(len(view_ids))
        ]).astype(np.float64)
        aggregated_by_method[aggregation] = rotvecs
        for composition in COMPOSITIONS:
            for alpha in args.alphas:
                corrected = corrected_rotations(
                    rough_relative,
                    view_ids,
                    rotvecs,
                    alpha,
                    composition,
                    args.anchor_view,
                )
                metrics = rotation_metrics(corrected, reference_relative, view_ids, args.anchor_view)
                key = (aggregation, composition, float(alpha))
                candidate_rotations[key] = corrected
                candidates.append({
                    "aggregation": aggregation,
                    "composition": composition,
                    "alpha": float(alpha),
                    "rotation_metrics": metrics,
                })

    candidates.sort(key=lambda item: item["rotation_metrics"]["mean_non_anchor_rotation_error_deg"])
    # Triangulation is expensive, so evaluate the best geometry candidates plus the rough anchor.
    finalists = candidates[: min(8, len(candidates))]
    finalist_predictions = {}
    for finalist in finalists:
        key = (
            finalist["aggregation"],
            finalist["composition"],
            finalist["alpha"],
        )
        corrected = candidate_rotations[key]
        predictions = {}
        with torch.no_grad():
            for batch in cached_batches:
                batch_device = move_batch(batch, device)
                points = triangulate_with_rotations(
                    batch_device,
                    rough_relative,
                    corrected,
                    view_ids,
                ).cpu().numpy()
                for index in range(points.shape[0]):
                    pred_key = (int(batch["frame_id"][index]), ident(batch["person_id"][index]))
                    predictions[pred_key] = points[index]
        candidate_key = (finalist["aggregation"], finalist["composition"], finalist["alpha"])
        finalist_predictions[candidate_key] = predictions
        finalist["triangulation_metrics"] = summarize_predictions(predictions, gt)

    best_rotation = candidates[0]
    best_triangulation = min(
        finalists,
        key=lambda item: (
            item["triangulation_metrics"]["similarity_aligned_mpjpe_m"],
            item["rotation_metrics"]["mean_non_anchor_rotation_error_deg"],
        ),
    )
    best_key = (
        best_rotation["aggregation"],
        best_rotation["composition"],
        best_rotation["alpha"],
    )
    best_rotations = candidate_rotations[best_key]
    corrected_payload = export_extrinsics(
        rough_payload,
        best_rotations,
        view_ids,
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "aggregation": best_rotation["aggregation"],
            "composition": best_rotation["composition"],
            "alpha": best_rotation["alpha"],
        },
    )
    corrected_path = output_dir / "rotation_audit_best_rotation_extrinsics.json"
    save_json(corrected_path, corrected_payload)

    rough_prediction_path = output_dir / "rough_prediction.json"
    best_rotation_prediction_path = output_dir / "best_rotation_prediction.json"
    best_triangulation_prediction_path = output_dir / "best_triangulation_prediction.json"
    common_metadata = {
        "stage": "static_rotation_correction_audit",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "anchor_view": args.anchor_view,
        "view_ids": view_ids,
    }
    export_prediction_json(
        rough_prediction_path,
        rough_predictions,
        confidence_by_key,
        args.sequence,
        {**common_metadata, "method": "rough"},
    )
    export_prediction_json(
        best_rotation_prediction_path,
        finalist_predictions[best_key],
        confidence_by_key,
        args.sequence,
        {**common_metadata, "method": "best_rotation", **best_rotation},
    )
    best_triangulation_key = (
        best_triangulation["aggregation"],
        best_triangulation["composition"],
        best_triangulation["alpha"],
    )
    export_prediction_json(
        best_triangulation_prediction_path,
        finalist_predictions[best_triangulation_key],
        confidence_by_key,
        args.sequence,
        {**common_metadata, "method": "best_triangulation", **best_triangulation},
    )

    oracle_target_rotvec = {}
    for view_id in view_ids:
        target_correction = reference_relative[view_id].T @ rough_relative[view_id]
        cosine = float(np.clip((np.trace(target_correction) - 1.0) * 0.5, -1.0, 1.0))
        oracle_target_rotvec[view_id] = {
            "required_correction_angle_deg": math.degrees(math.acos(cosine)),
        }

    report = {
        "stage": "static_rotation_correction_audit",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_loss_config": checkpoint.get("config", {}).get("loss", {}),
        "num_samples": len(dataset),
        "rough_rotation_metrics": rough_metrics,
        "rough_triangulation_metrics": summarize_predictions(rough_predictions, gt),
        "oracle_required_rotation": oracle_target_rotvec,
        "aggregated_prediction_axis_angle_rad": {
            method: {
                view_id: aggregated_by_method[method][index].tolist()
                for index, view_id in enumerate(view_ids)
            }
            for method in args.aggregations
        },
        "best_rotation_candidate": best_rotation,
        "best_triangulation_candidate": best_triangulation,
        "top_candidates": finalists,
        "all_rotation_candidates": candidates,
        "paths": {
            "best_rotation_extrinsics": str(corrected_path),
            "rough_prediction": str(rough_prediction_path),
            "best_rotation_prediction": str(best_rotation_prediction_path),
            "best_triangulation_prediction": str(best_triangulation_prediction_path),
        },
    }
    report_path = output_dir / "rotation_correction_audit.json"
    save_json(report_path, report)
    print(json.dumps({
        "report": str(report_path),
        "rough_rotation_metrics": rough_metrics,
        "rough_triangulation_metrics": report["rough_triangulation_metrics"],
        "best_rotation_candidate": best_rotation,
        "best_triangulation_candidate": best_triangulation,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
