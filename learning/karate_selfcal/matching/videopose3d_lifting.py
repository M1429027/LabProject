"""Stage 3A learned single-view lifting with MMPose VideoPose3D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
from mmengine.structures import InstanceData
from mmpose.apis import (
    convert_keypoint_definition,
    extract_pose_sequence,
    inference_pose_lifter_model,
    init_model,
)
from mmpose.structures import PoseDataSample


H36M_KEYPOINT_NAMES = {
    0: "root",
    1: "right_hip",
    2: "right_knee",
    3: "right_foot",
    4: "left_hip",
    5: "left_knee",
    6: "left_foot",
    7: "spine",
    8: "thorax",
    9: "neck_base",
    10: "head",
    11: "left_shoulder",
    12: "left_elbow",
    13: "left_wrist",
    14: "right_shoulder",
    15: "right_elbow",
    16: "right_wrist",
}

COCO_TO_H36M_DIRECT_INDEX = {
    1: 12,   # right_hip <- right_hip
    2: 14,   # right_knee <- right_knee
    3: 16,   # right_foot <- right_ankle
    4: 11,   # left_hip <- left_hip
    5: 13,   # left_knee <- left_knee
    6: 15,   # left_foot <- left_ankle
    9: 0,    # neck_base <- nose
    11: 5,   # left_shoulder <- left_shoulder
    12: 7,   # left_elbow <- left_elbow
    13: 9,   # left_wrist <- left_wrist
    14: 6,   # right_shoulder <- right_shoulder
    15: 8,   # right_elbow <- right_elbow
    16: 10,  # right_wrist <- right_wrist
}


@dataclass
class PoseLiftingConfig:
    """Configuration required to run learned 2D-to-3D lifting."""

    config_path: Path
    checkpoint_path: Path
    device: str = "cuda:0"
    detector_dataset_name: str = "coco"
    norm_pose_2d: bool = True
    rebase_keypoint: bool = True


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON payload from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _midpoint(a: dict[str, float] | None, b: dict[str, float] | None) -> dict[str, float] | None:
    """Return the midpoint between two 2D keypoints."""

    if a is None or b is None:
        return None
    return {
        "x": (float(a["x"]) + float(b["x"])) * 0.5,
        "y": (float(a["y"]) + float(b["y"])) * 0.5,
        "confidence": (float(a["confidence"]) + float(b["confidence"])) * 0.5,
    }


def _kp_map(person: dict[str, Any]) -> dict[int, dict[str, float]]:
    """Map track JSON keypoints by joint id."""

    out: dict[int, dict[str, float]] = {}
    for point in person.get("keypoints", []):
        out[int(point["id"])] = {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "confidence": float(point.get("confidence", 0.0)),
        }
    return out


def _body_root(person: dict[str, Any]) -> dict[str, float]:
    """Choose a stable 2D root anchor for visualization."""

    points = _kp_map(person)
    hips = _midpoint(points.get(11), points.get(12))
    if hips is not None:
        return hips

    shoulders = _midpoint(points.get(5), points.get(6))
    if shoulders is not None:
        return shoulders

    bbox = person.get("bbox")
    if bbox:
        return {
            "x": (float(bbox["x1"]) + float(bbox["x2"])) * 0.5,
            "y": (float(bbox["y1"]) + float(bbox["y2"])) * 0.5,
            "confidence": float(bbox.get("confidence", 0.0)),
        }

    return {"x": 0.0, "y": 0.0, "confidence": 0.0}


def _build_pose_sample(
    person: dict[str, Any],
    pose_det_dataset_name: str,
    pose_lift_dataset_name: str,
) -> PoseDataSample:
    """Convert one tracked 2D person into a PoseDataSample for MMPose."""

    keypoints = np.zeros((1, 17, 2), dtype=np.float32)
    coco_keypoint_scores = np.zeros((17,), dtype=np.float32)
    for point in person.get("keypoints", []):
        joint_id = int(point["id"])
        keypoints[0, joint_id, 0] = float(point["x"])
        keypoints[0, joint_id, 1] = float(point["y"])
        coco_keypoint_scores[joint_id] = float(point.get("confidence", 0.0))

    converted = convert_keypoint_definition(
        keypoints,
        pose_det_dataset_name,
        pose_lift_dataset_name,
    ).astype(np.float32)

    keypoint_scores = np.zeros((1, 17), dtype=np.float32)
    if pose_det_dataset_name == "coco" and pose_lift_dataset_name == "h36m":
        keypoint_scores[0, 0] = float((coco_keypoint_scores[11] + coco_keypoint_scores[12]) * 0.5)
        keypoint_scores[0, 8] = float((coco_keypoint_scores[5] + coco_keypoint_scores[6]) * 0.5)
        keypoint_scores[0, 7] = float((keypoint_scores[0, 0] + keypoint_scores[0, 8]) * 0.5)
        keypoint_scores[0, 10] = float((coco_keypoint_scores[1] + coco_keypoint_scores[2]) * 0.5)
        for h36m_idx, coco_idx in COCO_TO_H36M_DIRECT_INDEX.items():
            keypoint_scores[0, h36m_idx] = coco_keypoint_scores[coco_idx]
    else:
        keypoint_scores[0] = coco_keypoint_scores

    bbox = person.get("bbox", {})
    x1 = float(bbox.get("x1", 0.0))
    y1 = float(bbox.get("y1", 0.0))
    x2 = float(bbox.get("x2", x1))
    y2 = float(bbox.get("y2", y1))
    area = max((x2 - x1) * (y2 - y1), 1.0)

    pred_instances = InstanceData()
    pred_instances.keypoints = converted
    pred_instances.keypoint_scores = keypoint_scores
    pred_instances.bboxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    pred_instances.areas = np.array([area], dtype=np.float32)

    sample = PoseDataSample()
    sample.pred_instances = pred_instances
    sample.gt_instances = InstanceData()
    sample.set_field(int(person["track_id"]), "track_id")
    return sample


def _normalize(vec: np.ndarray) -> list[float]:
    """Normalize a 3D vector into a JSON-friendly list."""

    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return [0.0, 0.0, 0.0]
    return [float(value / norm) for value in vec]


def _postprocess_keypoints(
    keypoints: np.ndarray,
    rebase_keypoint: bool,
) -> np.ndarray:
    """Match the official MMPose demo 3D keypoint post-processing."""

    processed = np.asarray(keypoints, dtype=np.float32)
    if processed.ndim == 4:
        processed = np.squeeze(processed, axis=1)
    if processed.ndim == 3:
        processed = np.squeeze(processed, axis=0)
    processed = processed[..., [0, 2, 1]]
    processed[..., 0] = -processed[..., 0]
    processed[..., 2] = -processed[..., 2]
    if rebase_keypoint:
        processed[..., 2] -= np.min(processed[..., 2], axis=-1, keepdims=True)
    return processed.astype(np.float32)


def _build_vectors(
    keypoints_3d: np.ndarray,
) -> tuple[list[float], list[float], list[float], float]:
    """Compute spine, shoulder, and forward descriptors from 3D joints."""

    left_shoulder = keypoints_3d[11]
    right_shoulder = keypoints_3d[14]
    root = keypoints_3d[0]
    thorax = keypoints_3d[8]

    shoulder_vector = right_shoulder - left_shoulder
    spine_vector = thorax - root
    if np.linalg.norm(shoulder_vector) < 1e-6:
        shoulder_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if np.linalg.norm(spine_vector) < 1e-6:
        spine_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward_vector = np.cross(shoulder_vector, spine_vector)
    if np.linalg.norm(forward_vector) < 1e-6:
        forward_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    shoulder_norm = _normalize(shoulder_vector)
    spine_norm = _normalize(spine_vector)
    forward_norm = _normalize(forward_vector)
    frontality = float(np.clip(forward_norm[2], -1.0, 1.0))
    return spine_norm, shoulder_norm, forward_norm, frontality


def load_pose_lifter(cfg: PoseLiftingConfig):
    """Initialize the MMPose VideoPose3D model."""

    return init_model(
        str(cfg.config_path),
        str(cfg.checkpoint_path),
        device=cfg.device,
    )


def lift_tracked_view_with_videopose3d(
    track_payload: dict[str, Any],
    cfg: PoseLiftingConfig,
) -> dict[str, Any]:
    """Run learned single-view lifting on one tracked view payload."""

    model = load_pose_lifter(cfg)
    pose_lift_dataset = model.cfg.test_dataloader.dataset
    pose_lift_dataset_name = str(model.dataset_meta["dataset_name"])
    seq_len = int(pose_lift_dataset.get("seq_len", 1))
    causal = bool(pose_lift_dataset.get("causal", False))
    seq_step = int(pose_lift_dataset.get("seq_step", 1))

    pose_results_list: list[list[PoseDataSample]] = []
    frame_roots: list[dict[int, dict[str, float]]] = []
    for frame in track_payload.get("frames", []):
        frame_samples: list[PoseDataSample] = []
        roots_for_frame: dict[int, dict[str, float]] = {}
        for person in frame.get("people", []):
            track_id = int(person["track_id"])
            roots_for_frame[track_id] = _body_root(person)
            frame_samples.append(
                _build_pose_sample(
                    person,
                    pose_det_dataset_name=cfg.detector_dataset_name,
                    pose_lift_dataset_name=pose_lift_dataset_name,
                )
            )
        pose_results_list.append(frame_samples)
        frame_roots.append(roots_for_frame)

    metadata = track_payload.get("metadata", {})
    image_size = (int(metadata.get("height", 1)), int(metadata.get("width", 1)))

    frames_out: list[dict[str, Any]] = []
    track_forward_vectors: dict[int, list[list[float]]] = {}
    track_frontality: dict[int, list[float]] = {}

    for frame_idx, frame in enumerate(track_payload.get("frames", [])):
        pose_seq_2d = extract_pose_sequence(
            pose_results_list,
            frame_idx=frame_idx,
            causal=causal,
            seq_len=seq_len,
            step=seq_step,
        )
        pose_lift_results = inference_pose_lifter_model(
            model,
            pose_seq_2d,
            image_size=image_size,
            norm_pose_2d=cfg.norm_pose_2d,
        )

        people_out: list[dict[str, Any]] = []
        current_people = frame.get("people", [])
        for person, pose_lift_result in zip(current_people, pose_lift_results):
            track_id = int(person["track_id"])
            keypoints_3d = _postprocess_keypoints(
                pose_lift_result.pred_instances.keypoints,
                rebase_keypoint=cfg.rebase_keypoint,
            )
            keypoint_scores = np.asarray(
                pose_lift_result.pred_instances.keypoint_scores,
                dtype=np.float32,
            )
            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            if keypoint_scores.ndim == 2:
                keypoint_scores = np.squeeze(keypoint_scores, axis=0)

            spine_vector, shoulder_vector, forward_vector, frontality = _build_vectors(keypoints_3d)
            root_2d = frame_roots[frame_idx].get(track_id, {"x": 0.0, "y": 0.0})

            lifted_keypoints = []
            for joint_id in range(keypoints_3d.shape[0]):
                point = keypoints_3d[joint_id]
                lifted_keypoints.append(
                    {
                        "id": int(joint_id),
                        "name": H36M_KEYPOINT_NAMES.get(joint_id, f"joint_{joint_id}"),
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "z": float(point[2]),
                        "confidence": float(keypoint_scores[joint_id]),
                    }
                )

            track_forward_vectors.setdefault(track_id, []).append(forward_vector)
            track_frontality.setdefault(track_id, []).append(frontality)
            people_out.append(
                {
                    "track_id": track_id,
                    "root": {
                        "x": float(root_2d["x"]),
                        "y": float(root_2d["y"]),
                        "z": 0.0,
                    },
                    "lifted_keypoints": lifted_keypoints,
                    "spine_vector": spine_vector,
                    "shoulder_vector": shoulder_vector,
                    "forward_vector": forward_vector,
                    "frontality": frontality,
                }
            )

        frames_out.append(
            {
                "frame": int(frame["frame"]),
                "people": people_out,
            }
        )

    track_descriptors = []
    for track_id in sorted(track_forward_vectors):
        forward_vectors = np.asarray(track_forward_vectors[track_id], dtype=np.float32)
        mean_forward_vector = _normalize(np.mean(forward_vectors, axis=0))
        track_descriptors.append(
            {
                "track_id": int(track_id),
                "mean_forward_vector": mean_forward_vector,
                "mean_frontality": float(np.mean(track_frontality[track_id])),
                "num_lifted_frames": int(len(track_forward_vectors[track_id])),
            }
        )

    return {
        "metadata": {
            "view_id": metadata.get("view_id"),
            "fps": metadata.get("fps"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "total_frames": metadata.get("total_frames"),
            "source_detection_backend": metadata.get("source_detection_backend"),
            "source_tracking_backend": "pose_single_view",
            "lifting_method": "mmpose_videopose3d_h36m",
            "pose_det_dataset_name": cfg.detector_dataset_name,
            "pose_lift_dataset_name": pose_lift_dataset_name,
            "pose_lift_config": str(cfg.config_path),
            "pose_lift_checkpoint": str(cfg.checkpoint_path),
            "seq_len": seq_len,
            "causal": causal,
            "seq_step": seq_step,
            "norm_pose_2d": bool(cfg.norm_pose_2d),
            "rebase_keypoint": bool(cfg.rebase_keypoint),
            "device": cfg.device,
        },
        "keypoint_names": [H36M_KEYPOINT_NAMES[idx] for idx in sorted(H36M_KEYPOINT_NAMES)],
        "frames": frames_out,
        "track_descriptors": track_descriptors,
    }
