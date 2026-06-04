"""Relative pose estimation utilities for Stage 4A self-calibration."""

from __future__ import annotations

import json
import re
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .hypothesis_geometry import (
    build_track_frame_index,
    collect_correspondences_for_view_pair,
)


def _open_text_source(path: str | Path):
    """Open a plain text file or a `zip::inner/path.txt` source."""

    source = str(path)
    if "::" not in source:
        return Path(source).open("r", encoding="utf-8")

    archive_path, inner_path = source.split("::", 1)
    archive = zipfile.ZipFile(Path(archive_path).resolve())
    handle = archive.open(inner_path, "r")
    return _ZipTextHandle(archive=archive, handle=handle)


class _ZipTextHandle:
    """Context manager for text inside a zip archive."""

    def __init__(self, archive: zipfile.ZipFile, handle: Any) -> None:
        self._archive = archive
        self._handle = handle

    def __enter__(self):
        import io

        self._text = io.TextIOWrapper(self._handle, encoding="utf-8")
        return self._text

    def __exit__(self, exc_type, exc, tb) -> None:
        self._text.close()
        self._archive.close()


def infer_colmap_camera_id(view_id: str) -> int:
    """Infer COLMAP camera id from a view id such as `karate004_cam16`."""

    match = re.search(r"cam(\d+)$", str(view_id))
    if not match:
        raise ValueError(f"Could not infer COLMAP camera id from view id: {view_id}")
    return int(match.group(1))


def load_colmap_intrinsics(
    cameras_txt_path: str | Path,
    camera_id: int,
) -> dict[str, Any]:
    """Load one camera intrinsics entry from COLMAP `cameras.txt`."""

    target_id = int(camera_id)
    with _open_text_source(cameras_txt_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if int(parts[0]) != target_id:
                continue

            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(value) for value in parts[4:]]
            if model == "OPENCV_FISHEYE":
                if len(params) != 8:
                    raise ValueError(f"Unexpected OPENCV_FISHEYE params for camera {camera_id}")
                fx, fy, cx, cy, k1, k2, k3, k4 = params
                camera_matrix = np.array(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                dist_coeffs = np.array([k1, k2, k3, k4], dtype=np.float64)
            elif model in {"PINHOLE", "SIMPLE_PINHOLE"}:
                if model == "PINHOLE":
                    fx, fy, cx, cy = params[:4]
                else:
                    f, cx, cy = params[:3]
                    fx = fy = f
                camera_matrix = np.array(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                dist_coeffs = np.zeros(5, dtype=np.float64)
            else:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")

            return {
                "source_path": f"{cameras_txt_path}::{camera_id}",
                "camera_id": target_id,
                "model": model,
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
                "image_size": (width, height),
                "rms": None,
            }

    raise ValueError(f"Camera id {camera_id} not found in COLMAP file: {cameras_txt_path}")


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
        "model": "PINHOLE",
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
        "model": "PINHOLE",
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
    model: str = "PINHOLE",
) -> np.ndarray:
    """Undistort pixel coordinates into normalized camera coordinates."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if str(model).upper() == "OPENCV_FISHEYE":
        normalized = cv2.fisheye.undistortPoints(points, camera_matrix, dist_coeffs.reshape(-1, 1))
    else:
        normalized = cv2.undistortPoints(points, camera_matrix, dist_coeffs)
    return normalized.reshape(-1, 2)


def estimate_relative_pose(
    matched_keypoints_a: np.ndarray,
    matched_keypoints_b: np.ndarray,
    camera_matrix_a: np.ndarray,
    camera_matrix_b: np.ndarray,
    dist_coeffs_a: np.ndarray | None = None,
    dist_coeffs_b: np.ndarray | None = None,
    model_a: str = "PINHOLE",
    model_b: str = "PINHOLE",
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

    normalized_a = _prepare_normalized_points(
        matched_keypoints_a,
        camera_matrix_a,
        dist_coeffs_a,
        model=model_a,
    )
    normalized_b = _prepare_normalized_points(
        matched_keypoints_b,
        camera_matrix_b,
        dist_coeffs_b,
        model=model_b,
    )

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
            model_a=str(intrinsics_by_view[view_a].get("model", "PINHOLE")),
            model_b=str(intrinsics_by_view[view_b].get("model", "PINHOLE")),
            ransac_threshold=essential_ransac_threshold,
        )
        pair_results.append(
            {
                "view_a": view_a,
                "view_b": view_b,
                "num_correspondences": int(len(metadata)),
                "intrinsics_a": {
                    "source_path": intrinsics_by_view[view_a]["source_path"],
                    "model": intrinsics_by_view[view_a].get("model", "PINHOLE"),
                    "image_size": intrinsics_by_view[view_a]["image_size"],
                    "rms": intrinsics_by_view[view_a]["rms"],
                },
                "intrinsics_b": {
                    "source_path": intrinsics_by_view[view_b]["source_path"],
                    "model": intrinsics_by_view[view_b].get("model", "PINHOLE"),
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


def _invert_transform(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert a camera-to-camera rigid transform."""

    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    inv_rotation = rotation.T
    inv_translation = -inv_rotation @ translation
    return inv_rotation, inv_translation


def _compose_transform(
    rotation_ab: np.ndarray,
    translation_ab: np.ndarray,
    rotation_bw: np.ndarray,
    translation_bw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose transforms when X_b = R_ab X_a + t_ab and X_a = R_aw X_w + t_aw."""

    rotation_ab = np.asarray(rotation_ab, dtype=np.float64).reshape(3, 3)
    translation_ab = np.asarray(translation_ab, dtype=np.float64).reshape(3)
    rotation_bw = np.asarray(rotation_bw, dtype=np.float64).reshape(3, 3)
    translation_bw = np.asarray(translation_bw, dtype=np.float64).reshape(3)
    rotation = rotation_ab @ rotation_bw
    translation = rotation_ab @ translation_bw + translation_ab
    return rotation, translation


def build_rough_extrinsics(
    relative_pose_results: dict[str, Any],
    intrinsics_by_view: dict[str, dict[str, Any]],
    view_ids: list[str],
    anchor_view: str | None = None,
) -> dict[str, Any]:
    """Assemble rough camera extrinsics from pairwise relative poses.

    The current baseline prefers direct anchor-to-view estimates. If a direct
    edge is missing, it falls back to a simple BFS composition over available
    pairwise transforms. Translation scale remains relative because essential
    matrix recovery only provides unit translation.
    """

    if not view_ids:
        raise ValueError("At least one view is required to build rough extrinsics.")

    anchor_view = anchor_view or str(view_ids[0])
    if anchor_view not in view_ids:
        raise ValueError(f"Anchor view not found: {anchor_view}")

    edges: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for pair in relative_pose_results.get("pair_relative_poses", []):
        if pair.get("status") != "ok":
            continue
        view_a = str(pair["view_a"])
        view_b = str(pair["view_b"])
        rotation = np.asarray(pair["rotation"], dtype=np.float64)
        translation = np.asarray(pair["translation_unit"], dtype=np.float64).reshape(3)
        edges[(view_a, view_b)] = (rotation, translation, pair)
        inv_rotation, inv_translation = _invert_transform(rotation, translation)
        edges[(view_b, view_a)] = (inv_rotation, inv_translation, pair)

    extrinsics_by_view: dict[str, dict[str, Any]] = {
        anchor_view: {
            "view_id": anchor_view,
            "is_anchor": True,
            "rotation": np.eye(3, dtype=np.float64).tolist(),
            "translation": np.zeros(3, dtype=np.float64).tolist(),
            "projection_matrix": (
                np.asarray(intrinsics_by_view[anchor_view]["camera_matrix"], dtype=np.float64)
                @ np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
            ).tolist(),
            "intrinsics": {
                "source_path": intrinsics_by_view[anchor_view]["source_path"],
                "model": intrinsics_by_view[anchor_view].get("model", "PINHOLE"),
                "camera_matrix": np.asarray(intrinsics_by_view[anchor_view]["camera_matrix"], dtype=np.float64).tolist(),
                "dist_coeffs": np.asarray(intrinsics_by_view[anchor_view]["dist_coeffs"], dtype=np.float64).reshape(-1).tolist(),
                "image_size": intrinsics_by_view[anchor_view]["image_size"],
                "rms": intrinsics_by_view[anchor_view]["rms"],
            },
            "path_from_anchor": [anchor_view],
        }
    }

    queue: list[str] = [anchor_view]
    while queue:
        current = queue.pop(0)
        current_rotation = np.asarray(extrinsics_by_view[current]["rotation"], dtype=np.float64)
        current_translation = np.asarray(extrinsics_by_view[current]["translation"], dtype=np.float64)

        for target in view_ids:
            if target in extrinsics_by_view:
                continue
            edge = edges.get((current, target))
            if edge is None:
                continue
            rotation_ct, translation_ct, pair_meta = edge
            rotation_tw, translation_tw = _compose_transform(
                rotation_ct,
                translation_ct,
                current_rotation,
                current_translation,
            )
            projection = (
                np.asarray(intrinsics_by_view[target]["camera_matrix"], dtype=np.float64)
                @ np.hstack([rotation_tw, translation_tw.reshape(3, 1)])
            )
            extrinsics_by_view[target] = {
                "view_id": target,
                "is_anchor": False,
                "rotation": rotation_tw.tolist(),
                "translation": translation_tw.tolist(),
                "projection_matrix": projection.tolist(),
                "intrinsics": {
                    "source_path": intrinsics_by_view[target]["source_path"],
                    "model": intrinsics_by_view[target].get("model", "PINHOLE"),
                    "camera_matrix": np.asarray(intrinsics_by_view[target]["camera_matrix"], dtype=np.float64).tolist(),
                    "dist_coeffs": np.asarray(intrinsics_by_view[target]["dist_coeffs"], dtype=np.float64).reshape(-1).tolist(),
                    "image_size": intrinsics_by_view[target]["image_size"],
                    "rms": intrinsics_by_view[target]["rms"],
                },
                "source_pair": {
                    "view_a": pair_meta["view_a"],
                    "view_b": pair_meta["view_b"],
                },
                "path_from_anchor": [*extrinsics_by_view[current]["path_from_anchor"], target],
            }
            queue.append(target)

    unresolved_views = [view_id for view_id in view_ids if view_id not in extrinsics_by_view]
    return {
        "anchor_view": anchor_view,
        "translation_scale_note": "Translations come from essential-matrix recovery and remain in relative unit scale.",
        "num_resolved_views": len(extrinsics_by_view),
        "unresolved_views": unresolved_views,
        "extrinsics_by_view": extrinsics_by_view,
    }
