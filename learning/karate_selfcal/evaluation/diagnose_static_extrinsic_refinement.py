from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from learning.karate_selfcal.evaluation.compare_camera_geometry import (
    build_comparison,
    camera_center_from_extrinsics,
    relative_centers,
)
from learning.karate_selfcal.stage_a.dataset import KarateStageADataset, collate_stage_a
from learning.karate_selfcal.stage_a.model import build_model
from learning.karate_selfcal.stage_a.train import move_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequence-level static extrinsic correction diagnostic/export.")
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
    parser.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def identity_id(person_id: str) -> int:
    normalized = str(person_id).lower()
    if normalized.endswith("01"):
        return 0
    if normalized.endswith("02"):
        return 1
    digits = "".join(ch for ch in normalized if ch.isdigit())
    return max(int(digits) - 1, 0) if digits else 0


def load_gt(zip_path: str | Path, sequence: str) -> dict[int, dict[int, np.ndarray]]:
    out: dict[int, dict[int, np.ndarray]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        prefix = f"{sequence}/processed_data/poses3d/"
        for name in archive.namelist():
            if not name.startswith(prefix) or not name.endswith(".npy"):
                continue
            frame_id = int(Path(name).stem)
            payload = np.load(io.BytesIO(archive.read(name)), allow_pickle=True).item()
            people = {}
            for person_id, joints in payload.items():
                ident = 0 if str(person_id).endswith("01") else 1
                people[ident] = np.asarray(joints, dtype=np.float64)[:17, :3]
            out[frame_id] = people
    return out


def pelvis(points: np.ndarray) -> np.ndarray:
    return 0.5 * (points[11] + points[12])


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = src_centered.T @ dst_centered / max(len(src), 1)
    u, singular, vt = np.linalg.svd(cov)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    variance = float((src_centered**2).sum() / max(len(src), 1))
    scale = float(singular.sum() / max(variance, 1e-8))
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale * (src @ rotation.T) + translation


def add_metric_values(bucket: dict[str, list[float]], pred: np.ndarray, gt: np.ndarray, joints: list[int]) -> bool:
    pred_sel = pred[joints]
    gt_sel = gt[joints]
    valid = np.isfinite(pred_sel).all(axis=1) & np.isfinite(gt_sel).all(axis=1)
    if int(valid.sum()) < 3:
        return False
    pred_sel = pred_sel[valid]
    gt_sel = gt_sel[valid]
    bucket["raw"].extend(np.linalg.norm(pred_sel - gt_sel, axis=1).tolist())
    if np.isfinite(pred[[11, 12]]).all() and np.isfinite(gt[[11, 12]]).all():
        bucket["pelvis_aligned"].extend(np.linalg.norm((pred_sel - pelvis(pred)) - (gt_sel - pelvis(gt)), axis=1).tolist())
    if int(valid.sum()) >= 4:
        aligned = umeyama_similarity(pred_sel, gt_sel)
        bucket["similarity_aligned"].extend(np.linalg.norm(aligned - gt_sel, axis=1).tolist())
    return True


def summarize_bucket(bucket: dict[str, list[float]], person_frames: int) -> dict[str, Any]:
    return {
        "num_person_frames": int(person_frames),
        "raw_mpjpe_m": float(np.mean(bucket["raw"])) if bucket["raw"] else None,
        "pelvis_aligned_mpjpe_m": float(np.mean(bucket["pelvis_aligned"])) if bucket["pelvis_aligned"] else None,
        "similarity_aligned_mpjpe_m": float(np.mean(bucket["similarity_aligned"])) if bucket["similarity_aligned"] else None,
    }


def triangulate_with_static_delta(batch: dict[str, Any], static_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ray_tokens = batch["ray_tokens"]
    joint_view_mask = batch["joint_view_mask"].bool()
    origin_norm = ray_tokens[..., 0:3]
    direction = torch.nn.functional.normalize(ray_tokens[..., 3:6], dim=-1)
    confidence = ray_tokens[..., 8].clamp_min(0.0)
    scale = batch["ray_origin_scale"].reshape(-1, 1, 1, 1)
    center = batch["ray_origin_center"].reshape(-1, 1, 1, 3)
    origin = origin_norm * scale + center + static_delta.reshape(1, 1, -1, 3)

    weights = (confidence * joint_view_mask.float()).clamp_min(0.0)
    eye = torch.eye(3, dtype=ray_tokens.dtype, device=ray_tokens.device).reshape(1, 1, 1, 3, 3)
    direction_outer = direction.unsqueeze(-1) * direction.unsqueeze(-2)
    projector = eye - direction_outer
    weighted_projector = projector * weights.unsqueeze(-1).unsqueeze(-1)
    lhs = weighted_projector.sum(dim=2)
    rhs = (weighted_projector @ origin.unsqueeze(-1)).sum(dim=2).squeeze(-1)
    valid = weights.gt(0.0).sum(dim=2) >= 2
    regularizer = 1e-4 * eye.reshape(1, 1, 3, 3)
    points = torch.linalg.solve(lhs + regularizer, rhs.unsqueeze(-1)).squeeze(-1)
    points = torch.where(valid.unsqueeze(-1), points, batch["triangulated_3d"])
    return points, valid


def corrected_payload_from_center_deltas(rough_payload: dict[str, Any], view_ids: list[str], center_delta_by_view: dict[str, np.ndarray]) -> dict[str, Any]:
    payload = json.loads(json.dumps(rough_payload))
    for view_id in view_ids:
        meta = payload["extrinsics_by_view"][view_id]
        rotation = np.asarray(meta["rotation"], dtype=np.float64).reshape(3, 3)
        old_center = camera_center_from_extrinsics(meta)
        new_center = old_center + center_delta_by_view.get(view_id, np.zeros(3, dtype=np.float64))
        new_translation = -rotation @ new_center
        meta["translation"] = new_translation.tolist()
        meta["camera_center_world"] = new_center.tolist()
        if "projection_matrix" in meta:
            proj = np.asarray(meta["projection_matrix"], dtype=np.float64)
            if proj.shape == (3, 4):
                proj[:, 3] = new_translation
                meta["projection_matrix"] = proj.tolist()
    payload["stage"] = "stage_a_static_camera_center_correction"
    return payload


def joint_payload(points: np.ndarray, confidence: np.ndarray) -> list[dict[str, float | int]]:
    joints = []
    for joint_id, point in enumerate(points):
        conf = float(confidence[joint_id]) if joint_id < len(confidence) else 1.0
        if conf <= 0.0 or not np.isfinite(point).all():
            continue
        joints.append({"id": int(joint_id), "x": float(point[0]), "y": float(point[1]), "z": float(point[2]), "confidence": conf})
    return joints


def export_prediction_json(path: Path, method_arrays: dict[tuple[int, int], np.ndarray], confidences: dict[tuple[int, int], np.ndarray], sequence_name: str, metadata: dict[str, Any]) -> None:
    frames = []
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (frame_id, ident), points in sorted(method_arrays.items()):
        by_frame[frame_id].append({
            "identity_id": int(ident),
            "person_id": f"karate_person{ident + 1:02d}",
            "joints": joint_payload(points, confidences[(frame_id, ident)]),
        })
    for frame_id, identities in sorted(by_frame.items()):
        frames.append({"sequence": sequence_name, "frame": int(frame_id), "identities": sorted(identities, key=lambda x: x["identity_id"])})
    save_json(path, {"metadata": metadata, "frames": frames})


def compute_method_metrics(method_arrays: dict[str, dict[tuple[int, int], np.ndarray]], gt: dict[int, dict[int, np.ndarray]]) -> dict[str, Any]:
    report = {}
    for method_name, items in method_arrays.items():
        method_report = {}
        for label, joints in {"all_joints_0_16": list(range(17)), "body_joints_5_16": list(range(5, 17))}.items():
            bucket = {"raw": [], "pelvis_aligned": [], "similarity_aligned": []}
            person_frames = 0
            for (frame_id, ident), pred in sorted(items.items()):
                gt_frame = gt.get(frame_id + 1)
                if gt_frame is None or ident not in gt_frame:
                    continue
                if add_metric_values(bucket, pred, gt_frame[ident], joints):
                    person_frames += 1
            method_report[label] = summarize_bucket(bucket, person_frames)
        report[method_name] = method_report
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    dataset = KarateStageADataset(args.manifest, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_stage_a)
    device = resolve_device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    # First pass: collect a single static correction per camera over the whole sequence.
    delta_by_view: dict[int, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch(batch, device)
            outputs = model(batch_dev["ray_tokens"], view_mask=batch_dev["view_mask"], joint_view_mask=batch_dev["joint_view_mask"])
            deltas = (outputs["pred_camera_origin_delta"] * batch_dev["ray_origin_scale"].reshape(-1, 1, 1)).detach().cpu().numpy()
            view_mask = batch_dev["view_mask"].detach().cpu().numpy().astype(bool)
            for sample_index in range(deltas.shape[0]):
                for view_index in range(deltas.shape[1]):
                    if view_mask[sample_index, view_index]:
                        delta_by_view[view_index].append(deltas[sample_index, view_index].astype(np.float64))

    static_np = []
    static_stats = {}
    for view_index, view_id in enumerate(args.view_ids):
        arr = np.asarray(delta_by_view.get(view_index, []), dtype=np.float64)
        if arr.size == 0:
            value = np.zeros(3, dtype=np.float64)
            static_stats[view_id] = {"num_samples": 0}
        else:
            value = np.median(arr, axis=0) if args.aggregation == "median" else np.mean(arr, axis=0)
            norms = np.linalg.norm(arr, axis=1)
            static_stats[view_id] = {
                "num_samples": int(arr.shape[0]),
                "static_delta_xyz_m": value.tolist(),
                "std_delta_xyz_m": arr.std(axis=0).tolist(),
                "mean_delta_norm_m": float(norms.mean()),
                "median_delta_norm_m": float(np.median(norms)),
                "max_delta_norm_m": float(norms.max()),
            }
        static_np.append(value)
    static_delta = torch.as_tensor(np.asarray(static_np, dtype=np.float32), device=device)

    method_arrays: dict[str, dict[tuple[int, int], np.ndarray]] = {
        "rough_anchor": {},
        "static_corrected_anchor_only": {},
        "static_corrected_anchor_plus_residual": {},
    }
    confidences: dict[tuple[int, int], np.ndarray] = {}
    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch(batch, device)
            outputs = model(batch_dev["ray_tokens"], view_mask=batch_dev["view_mask"], joint_view_mask=batch_dev["joint_view_mask"])
            static_anchor, _ = triangulate_with_static_delta(batch_dev, static_delta)
            final = static_anchor + outputs["pred_pose_root_relative"]
            rough = batch_dev["triangulated_3d"].detach().cpu().numpy()
            static_anchor_np = static_anchor.detach().cpu().numpy()
            final_np = final.detach().cpu().numpy()
            conf_np = batch_dev["target_confidence"].detach().cpu().numpy()
            for i in range(rough.shape[0]):
                frame_id = int(batch["frame_id"][i])
                ident = identity_id(str(batch["person_id"][i]))
                key = (frame_id, ident)
                method_arrays["rough_anchor"][key] = rough[i]
                method_arrays["static_corrected_anchor_only"][key] = static_anchor_np[i]
                method_arrays["static_corrected_anchor_plus_residual"][key] = final_np[i]
                confidences[key] = conf_np[i]

    gt = load_gt(args.gt_zip, args.sequence)
    metrics = compute_method_metrics(method_arrays, gt)

    rough_payload = load_json(args.rough_extrinsics_json)
    reference_payload = load_json(args.reference_extrinsics_json)
    center_delta_by_view = {view_id: static_np[i] for i, view_id in enumerate(args.view_ids)}
    corrected_payload = corrected_payload_from_center_deltas(rough_payload, args.view_ids, center_delta_by_view)
    corrected_extrinsics_path = output_dir / f"static_{args.aggregation}_corrected_extrinsics.json"
    save_json(corrected_extrinsics_path, corrected_payload)
    rough_geometry = build_comparison(relative_centers(rough_payload, args.anchor_view), relative_centers(reference_payload, args.anchor_view), args.anchor_view)
    corrected_geometry = build_comparison(relative_centers(corrected_payload, args.anchor_view), relative_centers(reference_payload, args.anchor_view), args.anchor_view)

    prediction_json = output_dir / f"static_{args.aggregation}_corrected_prediction.json"
    export_prediction_json(
        prediction_json,
        method_arrays["static_corrected_anchor_plus_residual"],
        confidences,
        sequence_name="harmony4d/karate004_static_corrected",
        metadata={
            "stage": "stage_a_static_camera_correction_prediction",
            "aggregation": args.aggregation,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "manifest": str(Path(args.manifest).resolve()),
            "rough_extrinsics": str(Path(args.rough_extrinsics_json).resolve()),
        },
    )

    def pct_change(before: float | None, after: float | None) -> float | None:
        if before is None or after is None or abs(before) < 1e-9:
            return None
        return float((before - after) / before * 100.0)

    report = {
        "stage": "static_extrinsic_refinement_diagnostic",
        "aggregation": args.aggregation,
        "static_camera_correction_stats": static_stats,
        "camera_geometry": {
            "rough_vs_oracle": rough_geometry,
            "static_corrected_vs_oracle": corrected_geometry,
            "summary_delta": {
                "mean_center_error_change_pct": pct_change(rough_geometry["summary"].get("mean_center_error"), corrected_geometry["summary"].get("mean_center_error")),
                "median_center_error_change_pct": pct_change(rough_geometry["summary"].get("median_center_error"), corrected_geometry["summary"].get("median_center_error")),
                "mean_direction_angle_error_change_pct": pct_change(rough_geometry["summary"].get("mean_direction_angle_error_deg"), corrected_geometry["summary"].get("mean_direction_angle_error_deg")),
            },
        },
        "gt_metrics": metrics,
        "paths": {
            "prediction_json": str(prediction_json),
            "corrected_extrinsics": str(corrected_extrinsics_path),
            "diagnostic_json": str(output_dir / "static_extrinsic_refinement_diagnostic.json"),
        },
    }
    save_json(output_dir / "static_extrinsic_refinement_diagnostic.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
