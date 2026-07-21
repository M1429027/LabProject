from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from learning.rumpl_fourview import FEATURE_DIM, MAX_VIEWS
from learning.rumpl_fourview.datasets.camera_sampler import camera_center_from_rt


def load_joints_from_npz(path: str | Path) -> np.ndarray:
    data = np.load(path)
    preferred_keys = ["joints_3d", "joints3d", "skeleton_3d", "positions_3d"]
    for key in preferred_keys:
        if key in data:
            joints = np.asarray(data[key], dtype=np.float32)
            break
    else:
        joints = None
        for key in data.files:
            candidate = np.asarray(data[key])
            if candidate.ndim >= 2 and candidate.shape[-1] == 3:
                joints = candidate.astype(np.float32)
                break
    if joints is None:
        raise ValueError(f"No joints_3d-style array found in {path}.")
    if joints.ndim == 2:
        joints = joints[None, ...]
    if joints.shape[-2] != 17 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints shaped (T, 17, 3), got {joints.shape} from {path}.")
    return joints


def project_points(points_3d: np.ndarray, camera: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    points_3d = np.asarray(points_3d, dtype=np.float64)
    R = camera["R"]
    t = camera["t"].reshape(3, 1)
    K = camera["K"]
    cam_points = (R @ points_3d.T) + t
    depths = cam_points[2].copy()
    pixels_h = K @ cam_points
    pixels = (pixels_h[:2] / np.clip(pixels_h[2:], 1e-6, None)).T
    return pixels.astype(np.float32), depths.astype(np.float32)


def backproject_ray(camera: Dict[str, np.ndarray], pixel_xy: np.ndarray) -> np.ndarray:
    K = camera["K"]
    R = camera["R"]
    pixel_h = np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
    cam_dir = np.linalg.inv(K) @ pixel_h
    world_dir = R.T @ cam_dir
    world_dir /= np.linalg.norm(world_dir) + 1e-8
    return world_dir.astype(np.float32)


def synthetic_noise_confidence(
    image_point: np.ndarray,
    image_size: np.ndarray,
    rng: np.random.Generator,
    pixel_noise_std: float,
    dropout_probability: float,
    confidence_floor: float,
    out_of_frame_confidence: float,
) -> Tuple[np.ndarray, float, bool]:
    noisy = np.asarray(image_point, dtype=np.float32).copy()
    noisy += rng.normal(0.0, pixel_noise_std, size=2).astype(np.float32)

    if rng.random() < dropout_probability:
        return noisy, 0.0, False

    width, height = int(image_size[0]), int(image_size[1])
    in_frame = 0.0 <= noisy[0] < width and 0.0 <= noisy[1] < height
    if not in_frame:
        return noisy, float(out_of_frame_confidence), False

    confidence = max(confidence_floor, 1.0 - min(1.0, pixel_noise_std / 20.0))
    return noisy, float(confidence), True


def create_synthetic_sample(
    joints_3d: np.ndarray,
    cameras: List[Dict[str, np.ndarray]],
    active_view_indices: List[int],
    rng: np.random.Generator,
    noise_cfg: Dict[str, float],
    max_views: int = MAX_VIEWS,
) -> Dict[str, np.ndarray]:
    if joints_3d.shape != (17, 3):
        raise ValueError(f"Expected joints_3d with shape (17, 3), got {joints_3d.shape}.")

    ray_tokens = np.zeros((17, max_views, FEATURE_DIM), dtype=np.float32)
    view_mask = np.zeros((max_views,), dtype=bool)
    observations_2d = np.zeros((17, max_views, 2), dtype=np.float32)
    confidences = np.zeros((17, max_views), dtype=np.float32)

    for view_slot, camera_index in enumerate(active_view_indices):
        camera = cameras[camera_index]
        view_mask[view_slot] = True
        pixels, depths = project_points(joints_3d, camera)
        camera_center = camera_center_from_rt(camera["R"], camera["t"]).astype(np.float32)
        for joint_index, (pixel, depth) in enumerate(zip(pixels, depths)):
            noisy_pixel, confidence, is_valid = synthetic_noise_confidence(
                image_point=pixel,
                image_size=camera["image_size"],
                rng=rng,
                pixel_noise_std=float(noise_cfg.get("pixel_noise_std", 4.0)),
                dropout_probability=float(noise_cfg.get("dropout_probability", 0.08)),
                confidence_floor=float(noise_cfg.get("confidence_floor", 0.2)),
                out_of_frame_confidence=float(noise_cfg.get("out_of_frame_confidence", 0.0)),
            )
            observations_2d[joint_index, view_slot] = noisy_pixel
            confidences[joint_index, view_slot] = confidence
            if depth <= 0.0 or not is_valid:
                continue
            direction = backproject_ray(camera, noisy_pixel)
            ray_tokens[joint_index, view_slot, :3] = camera_center
            ray_tokens[joint_index, view_slot, 3:6] = direction
            ray_tokens[joint_index, view_slot, 6] = confidence

    return {
        "ray_tokens": ray_tokens,
        "view_mask": view_mask,
        "target_3d": joints_3d.astype(np.float32),
        "observations_2d": observations_2d,
        "confidences": confidences,
        "active_view_indices": np.asarray(active_view_indices, dtype=np.int32),
    }
