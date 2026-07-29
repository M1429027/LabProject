"""Regression tests for calibrated two-person cross-view identity selection."""

from __future__ import annotations

import numpy as np

from camera_system.camera_calibration.charuco.calib_charuco_v2.run_demo_two_person_pipeline import (
    normalize_bbox_dict,
    select_identity_hypothesis,
)


def _camera(center: list[float]) -> dict[str, np.ndarray]:
    intrinsic = np.asarray(
        [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation = np.eye(3, dtype=np.float64)
    camera_center = np.asarray(center, dtype=np.float64)
    translation = -rotation @ camera_center
    projection = intrinsic @ np.hstack([rotation, translation.reshape(3, 1)])
    return {
        "K": intrinsic,
        "dist": np.zeros(5, dtype=np.float64),
        "R": rotation,
        "t": translation,
        "P": projection,
    }


def _project(camera: dict[str, np.ndarray], point: np.ndarray) -> np.ndarray:
    projected = camera["P"] @ np.append(point, 1.0)
    return projected[:2] / projected[2]


def _person(
    track_id: int,
    camera: dict[str, np.ndarray],
    base_xy: tuple[float, float],
    frame_idx: int,
) -> dict:
    keypoints = []
    for joint_id in range(17):
        point = np.asarray(
            [
                base_xy[0] + 0.012 * joint_id + 0.002 * frame_idx,
                base_xy[1] + 0.01 * (joint_id % 4),
                5.0 + 0.005 * (joint_id % 3),
            ]
        )
        pixel = _project(camera, point)
        keypoints.append(
            {
                "id": joint_id,
                "x": float(pixel[0]),
                "y": float(pixel[1]),
                "confidence": 0.95,
            }
        )
    return {
        "track_id": track_id,
        "tracking_score": 0.95,
        "bbox": {"x1": 0.0, "y1": 0.0, "x2": 200.0, "y2": 400.0, "confidence": 0.9},
        "keypoints": keypoints,
    }


def _payload(view_id: str, frames: list[dict], track_ids: list[int]) -> dict:
    return {
        "metadata": {
            "view_id": view_id,
            "fps": 30.0,
            "width": 1280,
            "height": 720,
            "total_frames": len(frames),
        },
        "tracks": [
            {
                "track_id": track_id,
                "start_frame": 0,
                "end_frame": len(frames) - 1,
                "num_detections": len(frames),
            }
            for track_id in track_ids
        ],
        "frames": frames,
    }


def test_bbox_list_adapter_preserves_tracker_schema() -> None:
    payload = {
        "frames": [
            {
                "frame": 0,
                "people": [{"bbox": [1, 2, 11, 22], "bbox_score": 0.8, "keypoints": []}],
            }
        ]
    }
    normalized = normalize_bbox_dict(payload)
    assert normalized["frames"][0]["people"][0]["bbox"] == {
        "x1": 1.0,
        "y1": 2.0,
        "x2": 11.0,
        "y2": 22.0,
        "confidence": 0.8,
    }


def test_known_geometry_recovers_swapped_local_track_order() -> None:
    cameras = {
        "cam1demo": _camera([0.0, 0.0, 0.0]),
        "cam2demo": _camera([1.0, 0.0, 0.0]),
    }
    frames_1 = []
    frames_2 = []
    for frame_idx in range(24):
        frames_1.append(
            {
                "frame": frame_idx,
                "people": [
                    _person(10, cameras["cam1demo"], (-0.4, -0.5), frame_idx),
                    _person(20, cameras["cam1demo"], (0.5, 0.7), frame_idx),
                ],
            }
        )
        # The second view intentionally assigns the opposite local track order.
        frames_2.append(
            {
                "frame": frame_idx,
                "people": [
                    _person(7, cameras["cam2demo"], (0.5, 0.7), frame_idx),
                    _person(9, cameras["cam2demo"], (-0.4, -0.5), frame_idx),
                ],
            }
        )

    tracks = {
        "cam1demo": _payload("cam1demo", frames_1, [10, 20]),
        "cam2demo": _payload("cam2demo", frames_2, [7, 9]),
    }
    sync_report = {
        "reference_interval_frames": [0, 23],
        "affine_local_frame=a_ref_frame_plus_b": {
            "cam1demo": {"a": 1.0, "b": 0.0},
            "cam2demo": {"a": 1.0, "b": 0.0},
        },
    }
    selected = select_identity_hypothesis(
        tracks=tracks,
        cameras=cameras,
        sync_report=sync_report,
        cam_ids=["cam1demo", "cam2demo"],
        frame_interval=(0, 23),
        matching_config={"weights": {}, "matching": {"min_keypoint_confidence": 0.1}},
        frame_step=1,
        min_conf=0.3,
    )["selected_hypothesis"]

    assert selected["assignments"]["cam1demo"]["identity_0"] == 10
    assert selected["assignments"]["cam2demo"]["identity_0"] == 9
    assert selected["assignments"]["cam1demo"]["identity_1"] == 20
    assert selected["assignments"]["cam2demo"]["identity_1"] == 7
    assert selected["known_geometry"]["median_epipolar_px"] < 1e-6
