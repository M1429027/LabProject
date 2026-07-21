from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from real_world_pipeline import (
    KEYPOINT_NAMES,
    compute_fixed_axis,
    frames_to_video,
    load_camera_calibration,
    triangulate_ransac_refine,
    visualize_3d_skeleton,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Four-camera real-world 3D reconstruction pipeline.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--yolo-model", default="yolo11l-pose.pt")
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=25.0)
    parser.add_argument("--output-fps", type=float, default=-1.0)
    parser.add_argument("--skip-render", action="store_true")

    for camera_index in range(1, 5):
        parser.add_argument(f"--cam{camera_index}-video", required=True)
        parser.add_argument(f"--cam{camera_index}-intrinsics", required=True)
        parser.add_argument(f"--cam{camera_index}-extrinsics", required=True)

    return parser.parse_args()


def load_all_cameras(args: argparse.Namespace) -> Tuple[Dict[str, dict], Dict[str, np.ndarray]]:
    cameras: Dict[str, dict] = {}
    projection_matrices: Dict[str, np.ndarray] = {}

    for camera_index in range(1, 5):
        camera_name = f"cam{camera_index}"
        intrinsics = getattr(args, f"cam{camera_index}_intrinsics")
        extrinsics = getattr(args, f"cam{camera_index}_extrinsics")
        K, dist, R, t, P = load_camera_calibration(intrinsics, extrinsics)
        cameras[camera_name] = {
            "K": K,
            "dist": dist,
            "R": R,
            "t": t,
            "P": P,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }
        projection_matrices[camera_name] = P
    return cameras, projection_matrices


def run_yolo_on_video(model: YOLO, video_path: str, conf_thresh: float, output_dir: Path, camera_name: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    annotated_path = output_dir / f"yolo_{camera_name}.mp4"
    if width > 0 and height > 0 and fps > 0:
        writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames: List[dict] = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=conf_thresh)
        frame_data = {"frame": frame_id, "people": []}
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
            for person_id, person_keypoints in enumerate(keypoints):
                frame_data["people"].append(
                    {
                        "person_id": int(person_id),
                        "keypoints": [
                            {
                                "id": int(joint_id),
                                "x": float(kp[0]),
                                "y": float(kp[1]),
                                "confidence": float(kp[2]),
                            }
                            for joint_id, kp in enumerate(person_keypoints)
                        ],
                    }
                )
        frames.append(frame_data)

        if writer is not None:
            writer.write(results[0].plot())

        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    return {
        "metadata": {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": len(frames),
            "source_video": video_path,
        },
        "keypoint_names": KEYPOINT_NAMES,
        "frames": frames,
    }


def reconstruct_multi_camera(
    yolo_results: Dict[str, dict],
    projection_matrices: Dict[str, np.ndarray],
    confidence_threshold: float,
    reproj_threshold: float,
) -> List[dict]:
    camera_names = list(yolo_results.keys())
    total_frames = min(len(yolo_results[camera]["frames"]) for camera in camera_names)
    all_frames_3d: List[dict] = []

    for frame_id in range(total_frames):
        frame_result = {"frame": frame_id, "keypoints_3d": [], "valid_count": 0}
        current_keypoints = {}
        for camera_name in camera_names:
            people = yolo_results[camera_name]["frames"][frame_id]["people"]
            if people:
                current_keypoints[camera_name] = people[0]["keypoints"]

        if len(current_keypoints) < 2:
            all_frames_3d.append(frame_result)
            continue

        for joint_id in range(len(KEYPOINT_NAMES)):
            projections = []
            observed_points = []
            cameras_used = []
            observations = []

            for camera_name, keypoints in current_keypoints.items():
                if joint_id >= len(keypoints):
                    continue
                kp = keypoints[joint_id]
                if kp["confidence"] < confidence_threshold:
                    continue
                projections.append(projection_matrices[camera_name])
                observed_points.append([kp["x"], kp["y"]])
                cameras_used.append(camera_name)
                observations.append(
                    {
                        "camera": camera_name,
                        "x": float(kp["x"]),
                        "y": float(kp["y"]),
                        "confidence": float(kp["confidence"]),
                    }
                )

            if len(projections) >= 2:
                point_3d = triangulate_ransac_refine(projections, observed_points, reproj_threshold)
                if point_3d is not None and not np.any(np.isnan(point_3d)):
                    frame_result["keypoints_3d"].append(
                        {
                            "id": joint_id,
                            "name": KEYPOINT_NAMES[joint_id],
                            "position": point_3d.tolist(),
                            "valid": True,
                            "cameras_used": cameras_used,
                            "observations": observations,
                        }
                    )
                    frame_result["valid_count"] += 1
                    continue

            frame_result["keypoints_3d"].append(
                {
                    "id": joint_id,
                    "name": KEYPOINT_NAMES[joint_id],
                    "position": None,
                    "valid": False,
                    "cameras_used": [],
                    "observations": [],
                }
            )

        all_frames_3d.append(frame_result)

    return all_frames_3d


def project_point(point_3d: np.ndarray, projection: np.ndarray) -> np.ndarray:
    homogeneous = projection @ np.append(point_3d, 1.0)
    return homogeneous[:2] / homogeneous[2]


def compute_reprojection_metrics(all_frames_3d: List[dict], projection_matrices: Dict[str, np.ndarray]) -> dict:
    errors_by_camera: Dict[str, List[float]] = {camera_name: [] for camera_name in projection_matrices}

    for frame in all_frames_3d:
        for keypoint in frame["keypoints_3d"]:
            if not keypoint["valid"] or keypoint["position"] is None:
                continue
            point_3d = np.array(keypoint["position"], dtype=np.float64)
            for observation in keypoint.get("observations", []):
                camera_name = observation["camera"]
                projected = project_point(point_3d, projection_matrices[camera_name])
                observed = np.array([observation["x"], observation["y"]], dtype=np.float64)
                error = float(np.linalg.norm(projected - observed))
                errors_by_camera[camera_name].append(error)

    def summarize(errors: List[float]) -> dict | None:
        if not errors:
            return None
        arr = np.array(errors, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "rms": float(np.sqrt(np.mean(np.square(arr)))),
            "p95": float(np.percentile(arr, 95)),
        }

    metrics = {camera_name: summarize(errors) for camera_name, errors in errors_by_camera.items()}
    all_errors: List[float] = []
    for errors in errors_by_camera.values():
        all_errors.extend(errors)
    metrics["overall"] = summarize(all_errors)
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cameras, projection_matrices = load_all_cameras(args)

    model = YOLO(args.yolo_model)
    yolo_results: Dict[str, dict] = {}
    for camera_index in range(1, 5):
        camera_name = f"cam{camera_index}"
        video_path = getattr(args, f"cam{camera_index}_video")
        yolo_results[camera_name] = run_yolo_on_video(
            model=model,
            video_path=video_path,
            conf_thresh=args.confidence_threshold,
            output_dir=output_dir,
            camera_name=camera_name,
        )
        with open(output_dir / f"keypoints_{camera_name}.json", "w", encoding="utf-8") as handle:
            json.dump(yolo_results[camera_name], handle)

    all_frames_3d = reconstruct_multi_camera(
        yolo_results=yolo_results,
        projection_matrices=projection_matrices,
        confidence_threshold=args.confidence_threshold,
        reproj_threshold=args.ransac_reproj_threshold,
    )

    with open(output_dir / "skeleton_3d.json", "w", encoding="utf-8") as handle:
        json.dump({"frames": all_frames_3d}, handle, indent=2)

    reprojection_metrics = compute_reprojection_metrics(all_frames_3d, projection_matrices)
    with open(output_dir / "reprojection_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(reprojection_metrics, handle, indent=2)

    camera_params = {}
    for camera_name, camera in cameras.items():
        camera_params[camera_name] = {
            "intrinsics": camera["intrinsics"],
            "extrinsics": camera["extrinsics"],
            "K": camera["K"].tolist(),
            "dist": camera["dist"].tolist(),
            "R": camera["R"].tolist(),
            "t": camera["t"].flatten().tolist(),
            "P": camera["P"].tolist(),
        }
    with open(output_dir / "camera_params.json", "w", encoding="utf-8") as handle:
        json.dump(camera_params, handle, indent=2)

    valid_joint_total = sum(frame["valid_count"] for frame in all_frames_3d)
    if not args.skip_render:
        fixed_axis = compute_fixed_axis(all_frames_3d)
        for frame in all_frames_3d:
            if frame["valid_count"] > 0:
                visualize_3d_skeleton(frame, str(frames_dir), fixed_axis)
        fps = args.output_fps if args.output_fps > 0 else yolo_results["cam1"]["metadata"].get("fps", 30.0)
        frames_to_video(str(frames_dir), str(output_dir / "skeleton_3d.mp4"), fps)

    summary = {
        "num_frames": len(all_frames_3d),
        "camera_names": list(yolo_results.keys()),
        "confidence_threshold": args.confidence_threshold,
        "ransac_reproj_threshold": args.ransac_reproj_threshold,
        "valid_joint_total": int(valid_joint_total),
        "mean_valid_joints_per_frame": float(valid_joint_total / max(len(all_frames_3d), 1)),
        "reprojection_metrics": reprojection_metrics,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved four-camera reconstruction outputs to {output_dir}")


if __name__ == "__main__":
    main()
