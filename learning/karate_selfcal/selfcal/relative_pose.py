"""Relative pose estimation utilities for Stage 4A self-calibration."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .hypothesis_geometry import (
    build_track_frame_index,
    collect_correspondences_for_view_pair,
)


def load_intrinsics(path: str | Path) -> dict[str, Any]:
    """Load camera intrinsics from a report JSON or NPZ file."""

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        intrinsics_path = payload.get("intrinsics_path")
        if not intrinsics_path:
            raise ValueError(f"JSON file does not contain intrinsics_path: {path}")
        npz_path = Path(intrinsics_path)
        if not npz_path.is_absolute():
            npz_path = (path.parent / npz_path).resolve()
        return load_intrinsics(npz_path)

    if path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported intrinsics format: {path}")

    data = np.load(path, allow_pickle=True)
    camera_matrix = np.asarray(data["K"] if "K" in data.files else data["mtx"], dtype=np.float64)
    if "dist_coeffs" in data.files:
        dist_source = data["dist_coeffs"]
    elif "dist" in data.files:
        dist_source = data["dist"]
    else:
        dist_source = np.zeros(5, dtype=np.float64)
    dist_coeffs = np.asarray(dist_source, dtype=np.float64).reshape(-1)
    image_size = tuple(int(v) for v in np.asarray(data["image_size"]).reshape(-1)[:2]) if "image_size" in data.files else None
    rms = float(np.asarray(data["rms"]).reshape(())) if "rms" in data.files else None

    return {
        "source_path": str(path),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_size": image_size,
        "rms": rms,
    }


def approximate_intrinsics(width: int, height: int, focal_length_scale: float = 1.2) -> dict[str, Any]:
    """Create a simple pinhole intrinsics approximation from image size."""

    focal = float(max(width, height) * focal_length_scale)
    camera_matrix = np.array(
        [
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return {
        "source_path": "approximate_pinhole",
        "camera_matrix": camera_matrix,
        "dist_coeffs": np.zeros(5, dtype=np.float64),
        "image_size": (int(width), int(height)),
        "rms": None,
    }


def load_selected_hypothesis(path: str | Path) -> dict[str, Any]:
    """Load the selected hypothesis payload."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload.get("selected_hypothesis")
    if not selected:
        raise ValueError(f"No selected_hypothesis found in: {path}")
    return selected


def _prepare_normalized_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Undistort pixel coordinates into normalized camera coordinates."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(points, camera_matrix, dist_coeffs)
    return normalized.reshape(-1, 2)


def estimate_relative_pose(
    matched_keypoints_a: np.ndarray,
    matched_keypoints_b: np.ndarray,
    camera_matrix_a: np.ndarray,
    camera_matrix_b: np.ndarray,
    dist_coeffs_a: np.ndarray | None = None,
    dist_coeffs_b: np.ndarray | None = None,
    ransac_threshold: float = 1e-3,
    confidence: float = 0.999,
) -> dict[str, Any]:
    """Estimate a baseline relative pose using an essential matrix."""

    if len(matched_keypoints_a) < 8 or len(matched_keypoints_b) < 8:
        return {
            "status": "not_enough_points",
            "num_points": int(len(matched_keypoints_a)),
        }

    dist_coeffs_a = np.zeros(5, dtype=np.float64) if dist_coeffs_a is None else np.asarray(dist_coeffs_a, dtype=np.float64)
    dist_coeffs_b = np.zeros(5, dtype=np.float64) if dist_coeffs_b is None else np.asarray(dist_coeffs_b, dtype=np.float64)

    normalized_a = _prepare_normalized_points(matched_keypoints_a, camera_matrix_a, dist_coeffs_a)
    normalized_b = _prepare_normalized_points(matched_keypoints_b, camera_matrix_b, dist_coeffs_b)

    essential, essential_mask = cv2.findEssentialMat(
        normalized_a,
        normalized_b,
        cameraMatrix=np.eye(3, dtype=np.float64),
        method=cv2.RANSAC,
        prob=float(confidence),
        threshold=float(ransac_threshold),
    )
    if essential is None or essential_mask is None:
        return {
            "status": "essential_fit_failed",
            "num_points": int(len(matched_keypoints_a)),
        }

    if essential.shape[0] > 3:
        essential = essential[:3, :]

    inlier_mask = essential_mask.ravel().astype(bool)
    num_pose_inliers, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        normalized_a,
        normalized_b,
        cameraMatrix=np.eye(3, dtype=np.float64),
        mask=essential_mask,
    )
    pose_mask = pose_mask.ravel().astype(bool)

    return {
        "status": "ok",
        "num_points": int(len(matched_keypoints_a)),
        "num_essential_inliers": int(np.sum(inlier_mask)),
        "essential_inlier_ratio": float(np.mean(inlier_mask)) if len(inlier_mask) else 0.0,
        "num_pose_inliers": int(num_pose_inliers),
        "pose_inlier_ratio": float(np.mean(pose_mask)) if len(pose_mask) else 0.0,
        "essential_matrix": essential.tolist(),
        "rotation": rotation.tolist(),
        "translation_unit": translation.reshape(-1).tolist(),
    }


def estimate_all_relative_poses(
    selected_hypothesis: dict[str, Any],
    track_payloads: dict[str, dict[str, Any]],
    intrinsics_by_view: dict[str, dict[str, Any]],
    view_ids: list[str],
    min_confidence: float = 0.3,
    frame_step: int = 1,
    max_points_per_pair: int | None = 2000,
    essential_ransac_threshold: float = 1e-3,
) -> dict[str, Any]:
    """Estimate pairwise relative poses for all view pairs under one hypothesis."""

    frame_indexes = {
        view_id: build_track_frame_index(track_payloads[view_id])
        for view_id in view_ids
    }

    pair_results = []
    for view_a, view_b in combinations(view_ids, 2):
        points_a, points_b, metadata = collect_correspondences_for_view_pair(
            groups=list(selected_hypothesis.get("groups", [])),
            frame_indexes=frame_indexes,
            view_a=view_a,
            view_b=view_b,
            min_confidence=min_confidence,
            frame_step=frame_step,
            max_points=max_points_per_pair,
        )
        pose_result = estimate_relative_pose(
            matched_keypoints_a=points_a,
            matched_keypoints_b=points_b,
            camera_matrix_a=np.asarray(intrinsics_by_view[view_a]["camera_matrix"], dtype=np.float64),
            camera_matrix_b=np.asarray(intrinsics_by_view[view_b]["camera_matrix"], dtype=np.float64),
            dist_coeffs_a=np.asarray(intrinsics_by_view[view_a]["dist_coeffs"], dtype=np.float64),
            dist_coeffs_b=np.asarray(intrinsics_by_view[view_b]["dist_coeffs"], dtype=np.float64),
            ransac_threshold=essential_ransac_threshold,
        )
        pair_results.append(
            {
                "view_a": view_a,
                "view_b": view_b,
                "num_correspondences": int(len(metadata)),
                "intrinsics_a": {
                    "source_path": intrinsics_by_view[view_a]["source_path"],
                    "image_size": intrinsics_by_view[view_a]["image_size"],
                    "rms": intrinsics_by_view[view_a]["rms"],
                },
                "intrinsics_b": {
                    "source_path": intrinsics_by_view[view_b]["source_path"],
                    "image_size": intrinsics_by_view[view_b]["image_size"],
                    "rms": intrinsics_by_view[view_b]["rms"],
                },
                **pose_result,
            }
        )

    valid_pairs = [pair for pair in pair_results if pair.get("status") == "ok"]
    mean_pose_inlier_ratio = float(np.mean([pair["pose_inlier_ratio"] for pair in valid_pairs])) if valid_pairs else 0.0
    return {
        "selected_hypothesis": selected_hypothesis,
        "num_valid_pairs": len(valid_pairs),
        "mean_pose_inlier_ratio": mean_pose_inlier_ratio,
        "pair_relative_poses": pair_results,
    }
