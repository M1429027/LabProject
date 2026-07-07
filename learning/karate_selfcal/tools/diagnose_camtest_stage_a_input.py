"""Diagnose real-camera Stage A inputs before running the ray Transformer."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .build_camtest_stage_a_dataset import (
    COCO_NUM_JOINTS,
    camera_origin_world_m,
    keypoints_by_frame,
    load_calibration,
    load_json,
    pixel_to_world_ray,
)


JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

BONES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose camtest Stage A real-camera inputs.")
    parser.add_argument("--tracking-dir", required=True)
    parser.add_argument("--video-dir", default="camtest")
    parser.add_argument("--calib-root", default="camera_system/camera_calibration/charuco/calib_charuco_v2")
    parser.add_argument("--calib-output-suffix", default="")
    parser.add_argument("--view-ids", nargs="+", default=["cam1", "cam2", "cam3", "cam4"])
    parser.add_argument("--video-suffix", default="_aligned.mp4")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    return parser.parse_args()


def video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"exists": path.exists(), "opened": False}
    meta = {
        "exists": True,
        "opened": True,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    return meta


def summarize(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p99": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def closest_ray_distance(o1: np.ndarray, d1: np.ndarray, o2: np.ndarray, d2: np.ndarray) -> float:
    """Shortest distance between two 3D rays, treated as infinite lines for diagnosis."""

    w0 = o1 - o2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        return float(np.linalg.norm(np.cross(w0, d1)))
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    return float(np.linalg.norm(p1 - p2))


def triangulate_rays(origins: list[np.ndarray], directions: list[np.ndarray]) -> np.ndarray | None:
    if len(origins) < 2:
        return None
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for origin, direction in zip(origins, directions):
        d = direction.reshape(3, 1)
        projection = eye - d @ d.T
        a += projection
        b += projection @ origin
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None


def point_by_joint(points: list[dict[str, Any]], confidence_threshold: float) -> dict[int, dict[str, Any]]:
    out = {}
    for point in points:
        joint_id = int(point.get("id", -1))
        if not 0 <= joint_id < COCO_NUM_JOINTS:
            continue
        if float(point.get("confidence", 0.0)) < confidence_threshold:
            continue
        out[joint_id] = point
    return out


def main() -> None:
    args = parse_args()
    tracking_dir = Path(args.tracking_dir)
    video_dir = Path(args.video_dir)
    calib_root = Path(args.calib_root)
    output_json = Path(args.output_json)
    view_ids = [str(view_id) for view_id in args.view_ids]

    calibrations = {view_id: load_calibration(calib_root, view_id, args.calib_output_suffix) for view_id in view_ids}
    camera_origins = {view_id: camera_origin_world_m(calibrations[view_id]) for view_id in view_ids}
    per_view = {}
    view_reports = {}
    frame_sets = []

    for view_id in view_ids:
        tracking_path = tracking_dir / f"tracks_{view_id}.json"
        payload = load_json(tracking_path)
        frames = keypoints_by_frame(payload)
        frame_sets.append(set(frames.keys()))
        video_path = video_dir / f"{view_id}{args.video_suffix}"
        video_meta = video_metadata(video_path)
        calib_size = calibrations[view_id]["image_size"]
        confidence_values = []
        joints_per_frame = []
        low_conf_counts = []
        for points in frames.values():
            confidence_values.extend(float(point.get("confidence", 0.0)) for point in points)
            joints_per_frame.append(len(points))
            low_conf_counts.append(sum(float(point.get("confidence", 0.0)) < args.confidence_threshold for point in points))
        view_reports[view_id] = {
            "tracking_json": str(tracking_path),
            "video_path": str(video_path),
            "video": video_meta,
            "calibration_image_size": list(calib_size),
            "video_matches_calibration_size": bool(
                video_meta.get("opened")
                and int(video_meta.get("width", -1)) == int(calib_size[0])
                and int(video_meta.get("height", -1)) == int(calib_size[1])
            ),
            "frames_with_keypoints": len(frames),
            "joints_per_frame": summarize(joints_per_frame),
            "low_conf_joints_per_frame": summarize(low_conf_counts),
            "confidence": summarize(confidence_values),
            "camera_origin_m": camera_origins[view_id].astype(float).tolist(),
            "intrinsics_path": calibrations[view_id]["intrinsics_path"],
            "extrinsics_path": calibrations[view_id]["extrinsics_path"],
        }
        per_view[view_id] = frames

    common_frames = sorted(set.intersection(*frame_sets)) if frame_sets else []
    sampled_frames = common_frames[: args.max_frames] if args.max_frames else common_frames

    pair_distances: dict[str, list[float]] = {f"{a}-{b}": [] for a, b in combinations(view_ids, 2)}
    joint_distances: dict[str, list[float]] = {str(joint_id): [] for joint_id in range(COCO_NUM_JOINTS)}
    triangulated_body_spans: list[float] = []
    triangulated_bone_lengths: dict[str, list[float]] = {f"{a}-{b}": [] for a, b in BONES}
    rays_per_joint: list[int] = []

    for frame_id in sampled_frames:
        joint_points = {view_id: point_by_joint(per_view[view_id].get(frame_id, []), args.confidence_threshold) for view_id in view_ids}
        triangulated = np.full((COCO_NUM_JOINTS, 3), np.nan, dtype=np.float64)
        for joint_id in range(COCO_NUM_JOINTS):
            rays = []
            for view_id in view_ids:
                point = joint_points[view_id].get(joint_id)
                if point is None:
                    continue
                origin, direction = pixel_to_world_ray(float(point["x"]), float(point["y"]), calibrations[view_id])
                rays.append((view_id, origin.astype(np.float64), direction.astype(np.float64)))
            rays_per_joint.append(len(rays))
            for (view_a, origin_a, direction_a), (view_b, origin_b, direction_b) in combinations(rays, 2):
                dist = closest_ray_distance(origin_a, direction_a, origin_b, direction_b)
                pair_distances[f"{view_a}-{view_b}"].append(dist)
                joint_distances[str(joint_id)].append(dist)
            point_3d = triangulate_rays([ray[1] for ray in rays], [ray[2] for ray in rays])
            if point_3d is not None:
                triangulated[joint_id] = point_3d
        if np.isfinite(triangulated[[11, 12]]).all():
            pelvis = 0.5 * (triangulated[11] + triangulated[12])
            valid = np.isfinite(triangulated).all(axis=1)
            if valid.any():
                triangulated_body_spans.append(float(np.nanmax(np.linalg.norm(triangulated[valid] - pelvis.reshape(1, 3), axis=1))))
            for bone_a, bone_b in BONES:
                if np.isfinite(triangulated[[bone_a, bone_b]]).all():
                    triangulated_bone_lengths[f"{bone_a}-{bone_b}"].append(float(np.linalg.norm(triangulated[bone_a] - triangulated[bone_b])))

    camera_baselines = {}
    for view_a, view_b in combinations(view_ids, 2):
        camera_baselines[f"{view_a}-{view_b}"] = float(np.linalg.norm(camera_origins[view_a] - camera_origins[view_b]))

    report = {
        "stage": "camtest_stage_a_input_diagnostic",
        "tracking_dir": str(tracking_dir),
        "calib_root": str(calib_root),
        "view_ids": view_ids,
        "confidence_threshold": float(args.confidence_threshold),
        "common_frames": len(common_frames),
        "diagnosed_frames": len(sampled_frames),
        "views": view_reports,
        "camera_baselines_m": camera_baselines,
        "rays_per_joint": summarize(rays_per_joint),
        "ray_pair_distance_m": {key: summarize(values) for key, values in pair_distances.items()},
        "ray_joint_distance_m": {
            f"{joint_id}_{JOINT_NAMES[joint_id]}": summarize(joint_distances[str(joint_id)])
            for joint_id in range(COCO_NUM_JOINTS)
        },
        "linear_ray_triangulation_body_span_m": summarize(triangulated_body_spans),
        "linear_ray_triangulation_bone_lengths_m": {
            key: summarize(values) for key, values in triangulated_bone_lengths.items()
        },
        "notes": [
            "Ray pair distance is the shortest distance between same-joint rays from different views; large values indicate bad calibration, wrong sync, wrong identity, or wrong 2D keypoints.",
            "Linear triangulation is diagnostic only; the Stage A Transformer does not use these triangulated points as input.",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "common_frames": len(common_frames), "diagnosed_frames": len(sampled_frames)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()