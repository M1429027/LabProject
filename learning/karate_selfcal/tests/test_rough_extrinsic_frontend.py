from __future__ import annotations

import math

import numpy as np

from learning.karate_selfcal.stage_a.build_amass_yolo_dataset import (
    axis_angle_rotation,
)
from learning.karate_selfcal.stage_a.derive_camera_head_dataset import (
    BODY_LEFT_RIGHT_PERMUTATION,
    _rotation_error_deg,
    globally_consistent_rotations,
    stabilize_temporal_left_right,
)


def test_temporal_stabilization_repairs_one_frame_label_flip() -> None:
    keypoints = np.zeros((5, 1, 17, 3), dtype=np.float64)
    base = np.stack([
        np.array([20.0 + index * 8.0, 30.0 + index * 3.0, 0.9])
        for index in range(12)
    ])
    for frame in range(5):
        keypoints[frame, 0, 5:] = base + np.array([frame * 2.0, 0.0, 0.0])
    expected = keypoints.copy()
    keypoints[2, 0, 5:] = keypoints[2, 0, 5:][BODY_LEFT_RIGHT_PERMUTATION]

    stabilized, diagnostic = stabilize_temporal_left_right(
        keypoints, confidence_threshold=0.2, switch_penalty=0.05
    )

    assert np.allclose(stabilized[:, 0, 5:], expected[:, 0, 5:])
    assert diagnostic["views"][0]["num_flipped_frames"] == 1
    assert diagnostic["views"][0]["num_state_changes"] == 2


def test_global_rotation_graph_rejects_one_bad_pair() -> None:
    view_ids = ["cam0", "cam1", "cam2", "cam3"]
    rotations = {
        view: axis_angle_rotation(
            np.array([0.0, 1.0, 0.0]), math.radians(index * 30.0)
        )
        for index, view in enumerate(view_ids)
    }
    pairs = []
    for index_a in range(len(view_ids)):
        for index_b in range(index_a + 1, len(view_ids)):
            view_a, view_b = view_ids[index_a], view_ids[index_b]
            relative = rotations[view_b] @ rotations[view_a].T
            if (view_a, view_b) == ("cam0", "cam3"):
                relative = (
                    axis_angle_rotation(
                        np.array([1.0, 0.0, 0.0]), math.radians(100.0)
                    )
                    @ relative
                )
            pairs.append({
                "view_a": view_a,
                "view_b": view_b,
                "status": "ok",
                "rotation": relative.tolist(),
                "pose_inlier_ratio": 0.9,
            })

    solution = globally_consistent_rotations(pairs, view_ids, anchor_view="cam0")

    assert solution is not None
    for view in view_ids:
        assert _rotation_error_deg(solution["rotations"][view], rotations[view]) < 1e-5
    assert solution["median_residual_deg"] < 1e-5
    assert solution["max_residual_deg"] > 90.0


if __name__ == "__main__":
    test_temporal_stabilization_repairs_one_frame_label_flip()
    test_global_rotation_graph_rejects_one_bad_pair()

