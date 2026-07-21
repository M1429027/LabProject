from __future__ import annotations

import json
from typing import Dict, List, Tuple

import cv2
import numpy as np

from learning.rumpl_fourview import FEATURE_DIM, MAX_VIEWS, MODEL_JOINT_NAMES
from reconstruction_pipeline.algorithm_pipeline.modules.transformer_adapter.joint_mapping import person_keypoints_to_model_array, select_primary_person


def load_keypoints_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "frames" not in payload:
        raise ValueError(f"Invalid keypoints JSON (missing frames): {path}")
    return payload


def load_camera_calibration(intrin_path: str, extrin_path: str) -> Dict[str, np.ndarray]:
    intrin = np.load(intrin_path)
    extrin = np.load(extrin_path)

    K = None
    for key in ["mtx", "camera_matrix", "K", "intrinsic_matrix", "cameraMatrix"]:
        if key in intrin:
            K = np.asarray(intrin[key], dtype=np.float64)
            break
    if K is None:
        raise ValueError(f"Could not find intrinsic matrix in {intrin_path}")

    dist = None
    for key in ["dist", "dist_coeffs", "distCoeffs", "distortion_coefficients", "d"]:
        if key in intrin:
            dist = np.asarray(intrin[key], dtype=np.float64).reshape(-1)
            break
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)

    R = None
    for key in ["R", "rotation_matrix", "rvec", "rvecs", "rotation_vector"]:
        if key in extrin:
            value = np.asarray(extrin[key], dtype=np.float64)
            if value.shape == (3, 3):
                R = value
            else:
                R, _ = cv2.Rodrigues(value.reshape(3))
            break
    if R is None:
        raise ValueError(f"Could not find rotation in {extrin_path}")

    t = None
    for key in ["tvec", "tvecs", "t", "translation", "T", "translation_vector"]:
        if key in extrin:
            t = np.asarray(extrin[key], dtype=np.float64).reshape(3, 1)
            break
    if t is None:
        raise ValueError(f"Could not find translation in {extrin_path}")

    image_size = intrin["image_size"] if "image_size" in intrin else np.asarray([1920, 1080], dtype=np.int32)
    return {
        "K": K,
        "dist": dist,
        "R": R,
        "t": t,
        "image_size": np.asarray(image_size, dtype=np.int32),
    }


def camera_center_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
    return (-rotation.T @ translation).reshape(3)


def backproject_world_ray(camera: Dict[str, np.ndarray], point_2d: np.ndarray) -> np.ndarray:
    pixel_h = np.array([point_2d[0], point_2d[1], 1.0], dtype=np.float64)
    cam_direction = np.linalg.inv(camera["K"]) @ pixel_h
    world_direction = camera["R"].T @ cam_direction
    world_direction /= np.linalg.norm(world_direction) + 1e-8
    return world_direction.astype(np.float32)


def build_frame_ray_tokens(
    frame_payloads: List[Tuple[str, Dict[str, object], Dict[str, np.ndarray]]],
    confidence_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ray_tokens = np.zeros((len(MODEL_JOINT_NAMES), MAX_VIEWS, FEATURE_DIM), dtype=np.float32)
    view_mask = np.zeros((MAX_VIEWS,), dtype=bool)
    camera_names: List[str] = []

    for view_index, (camera_name, frame, camera) in enumerate(frame_payloads[:MAX_VIEWS]):
        view_mask[view_index] = True
        camera_names.append(camera_name)
        camera_center = camera_center_from_rt(camera["R"], camera["t"]).astype(np.float32)
        person = select_primary_person(frame)
        if person is None:
            continue
        joints = person_keypoints_to_model_array(person)
        for joint_index in range(joints.shape[0]):
            confidence = float(joints[joint_index, 2])
            if confidence < confidence_threshold:
                continue
            direction = backproject_world_ray(camera, joints[joint_index, :2])
            ray_tokens[joint_index, view_index, :3] = camera_center
            ray_tokens[joint_index, view_index, 3:6] = direction
            ray_tokens[joint_index, view_index, 6] = confidence

    return ray_tokens, view_mask, camera_names


def load_camera_streams(camera_inputs: List[Dict[str, str]]) -> List[Dict[str, object]]:
    streams: List[Dict[str, object]] = []
    for item in camera_inputs:
        if not item.get("keypoints_json"):
            continue
        stream = load_keypoints_json(item["keypoints_json"])
        calibration = load_camera_calibration(item["intrinsics"], item["extrinsics"])
        streams.append(
            {
                "name": item["name"],
                "stream": stream,
                "calibration": calibration,
            }
        )
    if len(streams) < 2:
        raise ValueError("Transformer inference requires at least two cameras.")
    return streams


def build_sequence_ray_tensors(streams: List[Dict[str, object]], confidence_threshold: float) -> Tuple[np.ndarray, np.ndarray, List[int], List[List[str]]]:
    num_frames = min(len(item["stream"]["frames"]) for item in streams)
    ray_tokens = np.zeros((num_frames, len(MODEL_JOINT_NAMES), MAX_VIEWS, FEATURE_DIM), dtype=np.float32)
    view_masks = np.zeros((num_frames, MAX_VIEWS), dtype=bool)
    frame_indices: List[int] = []
    camera_names_per_frame: List[List[str]] = []

    for frame_index in range(num_frames):
        frame_payloads = [
            (item["name"], item["stream"]["frames"][frame_index], item["calibration"])
            for item in streams
        ]
        frame_ray_tokens, frame_view_mask, camera_names = build_frame_ray_tokens(
            frame_payloads=frame_payloads,
            confidence_threshold=confidence_threshold,
        )
        ray_tokens[frame_index] = frame_ray_tokens
        view_masks[frame_index] = frame_view_mask
        frame_indices.append(int(frame_payloads[0][1].get("frame", frame_index)))
        camera_names_per_frame.append(camera_names)

    return ray_tokens, view_masks, frame_indices, camera_names_per_frame
