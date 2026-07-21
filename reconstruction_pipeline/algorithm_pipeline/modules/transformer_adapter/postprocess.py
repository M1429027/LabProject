from __future__ import annotations

from typing import List

import numpy as np

from learning.rumpl_fourview import MODEL_JOINT_NAMES


def predictions_to_skeleton_frames(
    predictions: np.ndarray,
    ray_tokens: np.ndarray,
    frame_indices: List[int],
    camera_names_per_frame: List[List[str]],
) -> List[dict]:
    frames = []
    for batch_index, frame_index in enumerate(frame_indices):
        keypoints_3d = []
        valid_count = 0
        for joint_index, joint_name in enumerate(MODEL_JOINT_NAMES):
            confidences = ray_tokens[batch_index, joint_index, :, 6]
            cameras_used = [
                camera_names_per_frame[batch_index][view_index]
                for view_index, confidence in enumerate(confidences[: len(camera_names_per_frame[batch_index])])
                if confidence > 0.0
            ]
            valid = bool(np.any(confidences > 0.0))
            if valid:
                valid_count += 1
                position = predictions[batch_index, joint_index].tolist()
            else:
                position = None
            payload = {
                "id": joint_index,
                "name": joint_name,
                "position": position,
                "valid": valid,
            }
            if cameras_used:
                payload["cameras_used"] = cameras_used
            keypoints_3d.append(payload)
        frames.append(
            {
                "frame": int(frame_index),
                "keypoints_3d": keypoints_3d,
                "valid_count": valid_count,
            }
        )
    return frames
