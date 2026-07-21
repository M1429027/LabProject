from __future__ import annotations

import math

import numpy as np

from learning.karate_selfcal.stage_a.build_amass_yolo_dataset import (
    axis_angle_rotation,
    camera_rotation_delta_axis_angle,
)
from learning.karate_selfcal.stage_a.derive_camera_head_dataset import (
    apply_scaled_error_template,
)


def test_scaled_template_reaches_requested_moderate_range() -> None:
    positions = (
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 4.0]),
        np.array([4.0, 0.0, 4.0]),
    )
    clean_cameras = [
        {
            "id": str(index),
            "R": np.eye(3),
            "position": position,
            "t": -position,
            "K": np.eye(3),
        }
        for index, position in enumerate(positions)
    ]
    tiny_angles = (0.0, 0.01, 0.02, 0.03)
    template = {
        "rotation_errors": np.stack([
            axis_angle_rotation(np.array([0.0, 1.0, 0.0]), math.radians(angle))
            for angle in tiny_angles
        ]),
        "center_errors": np.array([
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.0, 0.002, 0.0],
            [0.0, 0.0, 0.003],
        ]),
    }

    noisy, metadata = apply_scaled_error_template(
        clean_cameras,
        template,
        anchor_index=0,
        rotation_range_deg=(2.0, 2.0001),
        center_range_m=(0.4, 0.4001),
        rng=np.random.default_rng(7),
    )

    rotation_errors = [
        np.linalg.norm(camera_rotation_delta_axis_angle(clean, perturbed))
        * (180.0 / math.pi)
        for clean, perturbed in zip(clean_cameras, noisy)
    ]
    center_errors = [
        np.linalg.norm(np.asarray(perturbed["position"]) - np.asarray(clean["position"]))
        for clean, perturbed in zip(clean_cameras, noisy)
    ]
    assert np.allclose(noisy[0]["R"], clean_cameras[0]["R"])
    assert np.allclose(noisy[0]["position"], clean_cameras[0]["position"])
    assert 2.0 <= max(rotation_errors) <= 2.0001
    assert 0.4 <= max(center_errors) <= 0.4001
    assert 2.0 <= metadata["structured_target_rotation_max_deg"] <= 2.0001
    assert 0.4 <= metadata["structured_target_center_max_m"] <= 0.4001


if __name__ == "__main__":
    test_scaled_template_reaches_requested_moderate_range()
