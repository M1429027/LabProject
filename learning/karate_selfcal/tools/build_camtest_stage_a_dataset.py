"""Build Stage A ray-token samples from real camtest tracked 2D keypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COCO_NUM_JOINTS = 17
RAY_FEATURE_DIM = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real camtest Stage A inference samples.")
    parser.add_argument("--tracking-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view-ids", nargs="+", default=["cam1", "cam2", "cam3", "cam4"])
    parser.add_argument("--calib-root", default="camera_system/camera_calibration/charuco/calib_charuco_v2")
    parser.add_argument(
        "--calib-output-suffix",
        default="",
        help="Optional suffix after calib_out_camN, e.g. _recalc_2026_06_30.",
    )
    parser.add_argument("--sequence", default="camtest/aligned")
    parser.add_argument("--person-id", default="camtest_person01")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--triangulation-confidence-threshold", type=float, default=0.05)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_calibration(calib_root: Path, view_id: str, calib_output_suffix: str = "") -> dict[str, Any]:
    out_dir = calib_root / f"calib_out_{view_id}{calib_output_suffix}"
    intr_path = out_dir / f"{view_id}_intrinsics.npz"
    extr_path = out_dir / f"{view_id}_extrinsics.npz"
    if not extr_path.exists() and view_id in {"cam3", "cam4"} and not calib_output_suffix:
        extr_path = out_dir / f"{view_id}_latest_out_extrinsics.npz"
    if not intr_path.exists():
        raise FileNotFoundError(intr_path)
    if not extr_path.exists():
        raise FileNotFoundError(extr_path)
    intr = np.load(intr_path, allow_pickle=True)
    extr = np.load(extr_path, allow_pickle=True)
    return {
        "K": np.asarray(intr["K"], dtype=np.float64).reshape(3, 3),
        "dist": np.asarray(intr["dist"], dtype=np.float64).reshape(-1),
        "image_size": tuple(int(v) for v in np.asarray(intr["image_size"]).reshape(-1)[:2]),
        "R": np.asarray(extr["R"], dtype=np.float64).reshape(3, 3),
        "t": np.asarray(extr["t"], dtype=np.float64).reshape(3),
        "intrinsics_path": str(intr_path),
        "extrinsics_path": str(extr_path),
    }


def camera_origin_world_m(calib: dict[str, Any]) -> np.ndarray:
    # Calibration stores world(board)-to-camera: X_cam = R X_world + t, with translation in mm.
    origin_mm = -calib["R"].T @ calib["t"]
    return (origin_mm / 1000.0).astype(np.float32)


def pixel_to_world_ray(x: float, y: float, calib: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    point = np.array([[[float(x), float(y)]]], dtype=np.float64)
    normalized = cv2.undistortPoints(point, calib["K"], calib["dist"].reshape(-1)).reshape(2)
    direction_camera = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    direction_camera /= max(float(np.linalg.norm(direction_camera)), 1e-9)
    direction_world = calib["R"].T @ direction_camera
    direction_world /= max(float(np.linalg.norm(direction_world)), 1e-9)
    return camera_origin_world_m(calib), direction_world.astype(np.float32)


def root_relative(points: np.ndarray) -> np.ndarray:
    pelvis = 0.5 * (points[11] + points[12])
    return (points - pelvis.reshape(1, 3)).astype(np.float32)


def triangulate_weighted_rays(
    origins: list[np.ndarray],
    directions: list[np.ndarray],
    weights: list[float],
) -> tuple[np.ndarray, float]:
    if len(origins) < 2:
        return np.zeros(3, dtype=np.float32), 0.0
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    weight_sum = 0.0
    for origin, direction, weight in zip(origins, directions, weights):
        d = np.asarray(direction, dtype=np.float64)
        d /= max(float(np.linalg.norm(d)), 1e-9)
        projection = eye - np.outer(d, d)
        w = max(float(weight), 1e-6)
        a += w * projection
        b += w * projection @ np.asarray(origin, dtype=np.float64)
        weight_sum += w
    try:
        point = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float32), 0.0

    distances = []
    for origin, direction in zip(origins, directions):
        d = np.asarray(direction, dtype=np.float64)
        d /= max(float(np.linalg.norm(d)), 1e-9)
        offset = point - np.asarray(origin, dtype=np.float64)
        distances.append(float(np.linalg.norm(offset - np.dot(offset, d) * d)))
    mean_distance = float(np.mean(distances)) if distances else 1e6
    quality = float(len(origins)) * float(weight_sum / max(len(origins), 1)) / (1.0 + mean_distance)
    return point.astype(np.float32), quality


def triangulate_from_keypoints(
    frame_id: int,
    per_view_keypoints: dict[str, dict[int, list[dict[str, Any]]]],
    view_ids: list[str],
    calibrations: dict[str, dict[str, Any]],
    confidence_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_3d = np.zeros((COCO_NUM_JOINTS, 3), dtype=np.float32)
    quality = np.zeros((COCO_NUM_JOINTS,), dtype=np.float32)
    valid = np.zeros((COCO_NUM_JOINTS,), dtype=np.bool_)
    for joint_id in range(COCO_NUM_JOINTS):
        origins: list[np.ndarray] = []
        directions: list[np.ndarray] = []
        weights: list[float] = []
        for view_id in view_ids:
            points = per_view_keypoints[view_id].get(frame_id)
            if not points:
                continue
            point_by_id = {int(p["id"]): p for p in points}
            point = point_by_id.get(joint_id)
            if not point:
                continue
            x = float(point.get("x", np.nan))
            y = float(point.get("y", np.nan))
            conf = float(point.get("confidence", 0.0))
            if conf < confidence_threshold or not np.isfinite(x) or not np.isfinite(y):
                continue
            origin, direction = pixel_to_world_ray(x, y, calibrations[view_id])
            origins.append(origin.astype(np.float32))
            directions.append(direction.astype(np.float32))
            weights.append(conf * conf)
        point_3d, score = triangulate_weighted_rays(origins, directions, weights)
        points_3d[joint_id] = point_3d
        quality[joint_id] = np.float32(score)
        valid[joint_id] = len(origins) >= 2 and score > 0.0
    return points_3d, quality, valid


def keypoints_by_frame(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out = {}
    for frame in payload.get("frames", []):
        people = frame.get("people", [])
        if not people:
            continue
        # camtest aligned has one person per view; pick the longest/current track if multiple exist.
        people = sorted(people, key=lambda p: (int(p.get("track_id", p.get("person_id", 0))), -len(p.get("keypoints", []))))
        out[int(frame["frame"])] = people[0].get("keypoints", [])
    return out


def make_tokens(
    frame_id: int,
    per_view_keypoints: dict[str, dict[int, list[dict[str, Any]]]],
    view_ids: list[str],
    calibrations: dict[str, dict[str, Any]],
    origin_center: np.ndarray,
    origin_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tokens = np.zeros((COCO_NUM_JOINTS, len(view_ids), RAY_FEATURE_DIM), dtype=np.float32)
    view_mask = np.zeros((len(view_ids),), dtype=np.bool_)
    joint_view_mask = np.zeros((COCO_NUM_JOINTS, len(view_ids)), dtype=np.bool_)
    confidence = np.zeros((COCO_NUM_JOINTS,), dtype=np.float32)

    for view_index, view_id in enumerate(view_ids):
        points = per_view_keypoints[view_id].get(frame_id)
        if not points:
            continue
        point_by_id = {int(p["id"]): p for p in points}
        width, height = calibrations[view_id]["image_size"]
        view_mask[view_index] = True
        for joint_id in range(COCO_NUM_JOINTS):
            point = point_by_id.get(joint_id)
            if not point:
                continue
            x = float(point.get("x", np.nan))
            y = float(point.get("y", np.nan))
            conf = float(point.get("confidence", 0.0))
            if not np.isfinite(x) or not np.isfinite(y) or conf <= 0.0:
                continue
            origin, direction = pixel_to_world_ray(x, y, calibrations[view_id])
            origin_norm = (origin - origin_center) / max(float(origin_scale), 1e-6)
            x_norm = (x / float(width)) * 2.0 - 1.0
            y_norm = (y / float(height)) * 2.0 - 1.0
            tokens[joint_id, view_index] = [
                float(origin_norm[0]),
                float(origin_norm[1]),
                float(origin_norm[2]),
                float(direction[0]),
                float(direction[1]),
                float(direction[2]),
                float(x_norm),
                float(y_norm),
                float(conf),
                view_index / max(len(view_ids) - 1, 1),
                joint_id / max(COCO_NUM_JOINTS - 1, 1),
            ]
            joint_view_mask[joint_id, view_index] = True
            confidence[joint_id] = max(confidence[joint_id], np.float32(conf))
    return tokens, view_mask, joint_view_mask, confidence


def main() -> None:
    args = parse_args()
    tracking_dir = Path(args.tracking_dir)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    calib_root = Path(args.calib_root)
    view_ids = [str(v) for v in args.view_ids]

    calibrations = {view_id: load_calibration(calib_root, view_id, args.calib_output_suffix) for view_id in view_ids}
    origins = np.vstack([camera_origin_world_m(calibrations[v]) for v in view_ids])
    origin_center = origins.mean(axis=0).astype(np.float32)
    origin_scale = float(np.linalg.norm(origins - origin_center.reshape(1, 3), axis=1).max())
    origin_scale = max(origin_scale, 1.0)

    per_view_keypoints = {}
    frame_sets = []
    metadata_views = {}
    for view_id in view_ids:
        path = tracking_dir / f"tracks_{view_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = load_json(path)
        per_view_keypoints[view_id] = keypoints_by_frame(payload)
        frame_sets.append(set(per_view_keypoints[view_id].keys()))
        metadata_views[view_id] = {
            "tracking_json": str(path),
            "num_frames_with_keypoints": len(per_view_keypoints[view_id]),
            "intrinsics_path": calibrations[view_id]["intrinsics_path"],
            "extrinsics_path": calibrations[view_id]["extrinsics_path"],
            "image_size": list(calibrations[view_id]["image_size"]),
        }

    common_frames = sorted(set.intersection(*frame_sets)) if frame_sets else []
    if args.max_frames is not None:
        common_frames = common_frames[: int(args.max_frames)]

    entries = []
    dummy_target = np.zeros((COCO_NUM_JOINTS, 3), dtype=np.float32)
    for frame_id in common_frames:
        tokens, view_mask, joint_view_mask, confidence = make_tokens(
            frame_id=frame_id,
            per_view_keypoints=per_view_keypoints,
            view_ids=view_ids,
            calibrations=calibrations,
            origin_center=origin_center,
            origin_scale=origin_scale,
        )
        triangulated_3d, triangulated_quality, triangulated_valid = triangulate_from_keypoints(
            frame_id=frame_id,
            per_view_keypoints=per_view_keypoints,
            view_ids=view_ids,
            calibrations=calibrations,
            confidence_threshold=float(args.triangulation_confidence_threshold),
        )
        pelvis_anchor = 0.5 * (triangulated_3d[11] + triangulated_3d[12])
        pelvis_anchor_valid = bool(triangulated_valid[11] and triangulated_valid[12])
        pelvis_anchor_quality = (
            np.float32(0.5 * (triangulated_quality[11] + triangulated_quality[12]))
            if pelvis_anchor_valid
            else np.float32(0.0)
        )
        sample_id = f"camtest_aligned_frame_{frame_id:06d}_{args.person_id}"
        rel_path = Path("samples") / f"{sample_id}.npz"
        np.savez(
            output_dir / rel_path,
            ray_tokens=tokens,
            view_mask=view_mask,
            joint_view_mask=joint_view_mask,
            target_3d=dummy_target,
            target_3d_root_relative=dummy_target,
            target_confidence=confidence,
            triangulated_3d=triangulated_3d.astype(np.float32),
            triangulated_3d_root_relative=root_relative(triangulated_3d),
            triangulated_quality=triangulated_quality.astype(np.float32),
            triangulated_valid=triangulated_valid.astype(np.bool_),
            pelvis_anchor=pelvis_anchor.astype(np.float32),
            pelvis_anchor_quality=np.asarray(pelvis_anchor_quality, dtype=np.float32),
            pelvis_anchor_valid=np.asarray(pelvis_anchor_valid, dtype=np.bool_),
            ray_origin_center=origin_center.astype(np.float32),
            ray_origin_scale=np.asarray(origin_scale, dtype=np.float32),
        )
        entries.append(
            {
                "id": sample_id,
                "path": str(output_dir / rel_path),
                "split": args.split,
                "sequence": args.sequence,
                "frame_id": int(frame_id),
                "person_id": args.person_id,
                "active_joint_views": int(joint_view_mask.sum()),
                "triangulated_valid_joints": int(triangulated_valid.sum()),
                "pelvis_anchor_valid": pelvis_anchor_valid,
            }
        )

    manifest = {
        "metadata": {
            "stage": "camtest_stage_a_inference_dataset",
            "sequence": args.sequence,
            "view_ids": view_ids,
            "num_entries": len(entries),
            "origin_center_m": origin_center.tolist(),
            "origin_scale_m": origin_scale,
            "triangulation_confidence_threshold": float(args.triangulation_confidence_threshold),
            "views": metadata_views,
        },
        "entries": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "num_entries": len(entries)}, indent=2))


if __name__ == "__main__":
    main()