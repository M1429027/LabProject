from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class CameraDistributionConfig:
    horizontal_translation_m: float = 0.3
    vertical_translation_m: float = 0.15
    yaw_degrees: float = 10.0
    pitch_degrees: float = 8.0
    roll_degrees: float = 3.0
    calibration_translation_noise_m: float = 0.03
    calibration_rotation_noise_deg: float = 2.0

    @classmethod
    def from_mapping(cls, payload: Dict[str, float] | None) -> "CameraDistributionConfig":
        payload = payload or {}
        return cls(**{k: payload.get(k, getattr(cls, k)) for k in cls.__annotations__})


def _rotation_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def apply_euler_jitter(base_rotation: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    jitter = (
        _rotation_z(np.deg2rad(yaw_deg))
        @ _rotation_y(np.deg2rad(pitch_deg))
        @ _rotation_x(np.deg2rad(roll_deg))
    )
    return jitter @ base_rotation


def camera_center_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    return (-rotation.T @ translation).reshape(3)


def make_camera(name: str, K: np.ndarray, R: np.ndarray, t: np.ndarray, image_size: Sequence[int]) -> Dict[str, np.ndarray]:
    return {
        "name": name,
        "K": np.asarray(K, dtype=np.float64),
        "R": np.asarray(R, dtype=np.float64),
        "t": np.asarray(t, dtype=np.float64).reshape(3, 1),
        "image_size": np.asarray(image_size, dtype=np.int32),
    }


def sample_camera_rig(
    base_cameras: Sequence[Dict[str, np.ndarray]],
    rng: np.random.Generator,
    distribution: CameraDistributionConfig,
) -> List[Dict[str, np.ndarray]]:
    sampled: List[Dict[str, np.ndarray]] = []
    for camera in base_cameras:
        base_R = np.asarray(camera["R"], dtype=np.float64)
        base_t = np.asarray(camera["t"], dtype=np.float64).reshape(3, 1)
        base_center = camera_center_from_rt(base_R, base_t)

        center_jitter = np.array(
            [
                rng.uniform(-distribution.horizontal_translation_m, distribution.horizontal_translation_m),
                rng.uniform(-distribution.horizontal_translation_m, distribution.horizontal_translation_m),
                rng.uniform(-distribution.vertical_translation_m, distribution.vertical_translation_m),
            ],
            dtype=np.float64,
        )
        noisy_center = base_center + center_jitter

        sampled_R = apply_euler_jitter(
            base_R,
            yaw_deg=rng.uniform(-distribution.yaw_degrees, distribution.yaw_degrees),
            pitch_deg=rng.uniform(-distribution.pitch_degrees, distribution.pitch_degrees),
            roll_deg=rng.uniform(-distribution.roll_degrees, distribution.roll_degrees),
        )
        sampled_t = (-sampled_R @ noisy_center.reshape(3, 1)).astype(np.float64)

        sampled_t += rng.normal(
            0.0,
            distribution.calibration_translation_noise_m,
            size=(3, 1),
        )
        sampled_R = apply_euler_jitter(
            sampled_R,
            yaw_deg=rng.normal(0.0, distribution.calibration_rotation_noise_deg),
            pitch_deg=rng.normal(0.0, distribution.calibration_rotation_noise_deg),
            roll_deg=rng.normal(0.0, distribution.calibration_rotation_noise_deg),
        )

        sampled.append(
            make_camera(
                name=camera["name"],
                K=camera["K"],
                R=sampled_R,
                t=sampled_t,
                image_size=camera["image_size"],
            )
        )
    return sampled


def sample_active_view_indices(num_cameras: int, rng: np.random.Generator, min_views: int = 2) -> List[int]:
    if num_cameras < min_views:
        raise ValueError(f"Need at least {min_views} cameras, got {num_cameras}.")
    count = int(rng.integers(min_views, num_cameras + 1))
    indices = np.arange(num_cameras)
    rng.shuffle(indices)
    return sorted(indices[:count].tolist())
