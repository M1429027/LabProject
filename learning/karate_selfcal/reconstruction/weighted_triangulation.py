"""Weighted triangulation utilities."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def undistort_point(
    x: float,
    y: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    model: str = "PINHOLE",
) -> np.ndarray:
    """Convert one pixel observation into normalized camera coordinates."""

    points = np.array([[[float(x), float(y)]]], dtype=np.float64)
    if str(model).upper() == "OPENCV_FISHEYE":
        normalized = cv2.fisheye.undistortPoints(points, camera_matrix, dist_coeffs.reshape(-1, 1))
    else:
        normalized = cv2.undistortPoints(points, camera_matrix, dist_coeffs)
    return normalized.reshape(2)


def sampson_error(
    point_a: np.ndarray,
    point_b: np.ndarray,
    fundamental_matrix: np.ndarray,
) -> float:
    """Compute the Sampson error for one pixel-space correspondence."""

    fundamental = np.asarray(fundamental_matrix, dtype=np.float64).reshape(3, 3)
    pa = np.array([float(point_a[0]), float(point_a[1]), 1.0], dtype=np.float64)
    pb = np.array([float(point_b[0]), float(point_b[1]), 1.0], dtype=np.float64)
    fpa = fundamental @ pa
    ftpb = fundamental.T @ pb
    numerator = float((pb.T @ fundamental @ pa) ** 2)
    denominator = float(fpa[0] ** 2 + fpa[1] ** 2 + ftpb[0] ** 2 + ftpb[1] ** 2)
    if denominator <= 1e-12:
        return float("inf")
    return numerator / denominator


def triangulate_weighted(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Triangulate one 3D point from multi-view weighted observations.

    Each observation must contain:
    - `projection_matrix`: 3x4 camera matrix in the same coordinate space as (`x`, `y`)
    - `x`, `y`: 2D image or normalized coordinates
    - `weight`: positive confidence-like weight
    """

    if len(observations) < 2:
        return {
            "status": "not_enough_views",
            "num_views": len(observations),
        }

    rows = []
    for obs in observations:
        projection = np.asarray(obs["projection_matrix"], dtype=np.float64).reshape(3, 4)
        x = float(obs["x"])
        y = float(obs["y"])
        weight = max(float(obs.get("weight", 1.0)), 1e-6)
        rows.append(weight * (x * projection[2] - projection[0]))
        rows.append(weight * (y * projection[2] - projection[1]))

    design = np.asarray(rows, dtype=np.float64)
    try:
        _, _, vh = np.linalg.svd(design)
    except np.linalg.LinAlgError:
        return {
            "status": "svd_failed",
            "num_views": len(observations),
        }

    homogeneous = vh[-1]
    if abs(float(homogeneous[3])) < 1e-9:
        return {
            "status": "degenerate_solution",
            "num_views": len(observations),
        }

    point_3d = homogeneous[:3] / homogeneous[3]
    return {
        "status": "ok",
        "num_views": len(observations),
        "point_3d": point_3d.tolist(),
    }


def project_point(
    projection_matrix: np.ndarray,
    point_3d: np.ndarray,
) -> np.ndarray:
    """Project one 3D point into one camera."""

    projection_matrix = np.asarray(projection_matrix, dtype=np.float64).reshape(3, 4)
    point_h = np.hstack([np.asarray(point_3d, dtype=np.float64).reshape(3), 1.0])
    pixel_h = projection_matrix @ point_h
    if abs(float(pixel_h[2])) < 1e-9:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return pixel_h[:2] / pixel_h[2]


def project_point_pixels(
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    point_3d: np.ndarray,
    model: str = "PINHOLE",
) -> np.ndarray:
    """Project one world-space 3D point to pixel coordinates with distortion."""

    object_points = np.asarray(point_3d, dtype=np.float64).reshape(1, 1, 3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rotation)
    if str(model).upper() == "OPENCV_FISHEYE":
        image_points, _ = cv2.fisheye.projectPoints(
            object_points,
            rvec,
            translation,
            np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
        )
    else:
        image_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            translation,
            np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
        )
    return image_points.reshape(2)


def reprojection_errors(
    point_3d: np.ndarray,
    observations: list[dict[str, Any]],
) -> list[float]:
    """Return per-view reprojection errors in pixels."""

    errors = []
    for obs in observations:
        projected = project_point(obs["projection_matrix"], point_3d)
        measured = np.array([float(obs["x"]), float(obs["y"])], dtype=np.float64)
        if np.any(np.isnan(projected)):
            errors.append(float("nan"))
            continue
        errors.append(float(np.linalg.norm(projected - measured)))
    return errors


def pixel_reprojection_errors(
    point_3d: np.ndarray,
    observations: list[dict[str, Any]],
) -> list[float]:
    """Return per-view pixel reprojection errors using camera intrinsics/distortion."""

    errors = []
    for obs in observations:
        projected = project_point_pixels(
            rotation=np.asarray(obs["rotation"], dtype=np.float64),
            translation=np.asarray(obs["translation"], dtype=np.float64),
            camera_matrix=np.asarray(obs["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(obs["dist_coeffs"], dtype=np.float64),
            point_3d=point_3d,
            model=str(obs.get("camera_model", "PINHOLE")),
        )
        measured = np.array([float(obs["pixel_x"]), float(obs["pixel_y"])], dtype=np.float64)
        if np.any(np.isnan(projected)):
            errors.append(float("nan"))
            continue
        errors.append(float(np.linalg.norm(projected - measured)))
    return errors
