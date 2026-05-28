"""Coarse single-view pseudo-3D lifting for track-level direction descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LEFT_IDS = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_IDS = {2, 4, 6, 8, 10, 12, 14, 16}
CENTER_IDS = {0}


@dataclass
class LiftedTrackFrame:
    """One frame of lifted pseudo-3D pose."""

    frame: int
    track_id: int
    lifted_keypoints: list[dict[str, float]]
    root: dict[str, float]
    spine_vector: list[float]
    shoulder_vector: list[float]
    forward_vector: list[float]
    frontality: float


def _kp_map(person: dict[str, Any], min_conf: float) -> dict[int, dict[str, float]]:
    """Convert keypoints into an id-indexed dict filtered by confidence."""

    out: dict[int, dict[str, float]] = {}
    for point in person.get("keypoints", []):
        confidence = float(point.get("confidence", 0.0))
        if confidence < min_conf:
            continue
        out[int(point["id"])] = {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "confidence": confidence,
        }
    return out


def _midpoint(a: dict[str, float] | None, b: dict[str, float] | None) -> dict[str, float] | None:
    """Compute midpoint between two keypoints."""

    if a is None or b is None:
        return None
    return {
        "x": (float(a["x"]) + float(b["x"])) * 0.5,
        "y": (float(a["y"]) + float(b["y"])) * 0.5,
        "confidence": (float(a["confidence"]) + float(b["confidence"])) * 0.5,
    }


def _body_root(points: dict[int, dict[str, float]], bbox: dict[str, Any] | None) -> dict[str, float]:
    """Choose a robust body root for root-centered coordinates."""

    hips = _midpoint(points.get(11), points.get(12))
    if hips is not None:
        return hips
    shoulders = _midpoint(points.get(5), points.get(6))
    if shoulders is not None:
        return shoulders
    if bbox:
        return {
            "x": (float(bbox["x1"]) + float(bbox["x2"])) * 0.5,
            "y": (float(bbox["y1"]) + float(bbox["y2"])) * 0.5,
            "confidence": float(bbox.get("confidence", 0.0)),
        }
    return {"x": 0.0, "y": 0.0, "confidence": 0.0}


def _body_scale(points: dict[int, dict[str, float]], bbox: dict[str, Any] | None) -> float:
    """Estimate person scale from torso geometry or bbox size."""

    shoulders = _midpoint(points.get(5), points.get(6))
    hips = _midpoint(points.get(11), points.get(12))
    scales = []
    if shoulders is not None and hips is not None:
        scales.append(
            ((float(shoulders["x"]) - float(hips["x"])) ** 2 + (float(shoulders["y"]) - float(hips["y"])) ** 2)
            ** 0.5
        )
    if points.get(5) and points.get(6):
        scales.append(abs(float(points[6]["x"]) - float(points[5]["x"])))
    if points.get(11) and points.get(12):
        scales.append(abs(float(points[12]["x"]) - float(points[11]["x"])))
    if bbox:
        scales.append(
            max(
                float(bbox["x2"]) - float(bbox["x1"]),
                float(bbox["y2"]) - float(bbox["y1"]),
            )
        )
    scales = [value for value in scales if value > 1.0]
    if not scales:
        return 100.0
    return max(float(np.mean(scales)), 50.0)


def _frontality_score(points: dict[int, dict[str, float]], root: dict[str, float], scale: float) -> float:
    """Estimate coarse facing direction from facial and shoulder asymmetry."""

    scores = []
    nose = points.get(0)
    shoulders = _midpoint(points.get(5), points.get(6))
    if nose is not None and shoulders is not None:
        scores.append(np.tanh((float(nose["x"]) - float(shoulders["x"])) / max(scale * 0.35, 1.0)))

    left_face_conf = np.mean([points[idx]["confidence"] for idx in (1, 3) if idx in points]) if any(
        idx in points for idx in (1, 3)
    ) else 0.0
    right_face_conf = np.mean([points[idx]["confidence"] for idx in (2, 4) if idx in points]) if any(
        idx in points for idx in (2, 4)
    ) else 0.0
    face_total = left_face_conf + right_face_conf
    if face_total > 0.0:
        scores.append((right_face_conf - left_face_conf) / face_total)

    if not scores:
        return 0.0
    return float(np.clip(np.mean(scores), -1.0, 1.0))


def _joint_side_sign(joint_id: int) -> float:
    """Map COCO joint ids into a left/right/center sign."""

    if joint_id in LEFT_IDS:
        return -1.0
    if joint_id in RIGHT_IDS:
        return 1.0
    if joint_id in CENTER_IDS:
        return 0.0
    return 0.0


def _lift_person_to_pseudo3d(
    person: dict[str, Any],
    frame_index: int,
    min_conf: float,
) -> LiftedTrackFrame | None:
    """Lift one 2D person detection into a coarse pseudo-3D skeleton."""

    points = _kp_map(person, min_conf=min_conf)
    if len(points) < 5:
        return None

    bbox = person.get("bbox")
    root = _body_root(points, bbox)
    scale = _body_scale(points, bbox)
    frontality = _frontality_score(points, root, scale)

    lifted_keypoints = []
    for joint_id, point in sorted(points.items()):
        x = (float(point["x"]) - float(root["x"])) / scale
        y = (float(root["y"]) - float(point["y"])) / scale
        side_sign = _joint_side_sign(joint_id)
        articulation = 0.25 * np.tanh(x)
        z = float(np.clip(side_sign * frontality + articulation * side_sign, -1.0, 1.0))
        lifted_keypoints.append(
            {
                "id": int(joint_id),
                "x": float(x),
                "y": float(y),
                "z": z,
                "confidence": float(point["confidence"]),
            }
        )

    lifted_map = {item["id"]: item for item in lifted_keypoints}
    left_shoulder = lifted_map.get(5)
    right_shoulder = lifted_map.get(6)
    left_hip = lifted_map.get(11)
    right_hip = lifted_map.get(12)

    if left_shoulder and right_shoulder:
        shoulder_center = np.array(
            [
                (left_shoulder["x"] + right_shoulder["x"]) * 0.5,
                (left_shoulder["y"] + right_shoulder["y"]) * 0.5,
                (left_shoulder["z"] + right_shoulder["z"]) * 0.5,
            ],
            dtype=np.float32,
        )
        shoulder_vector = np.array(
            [
                right_shoulder["x"] - left_shoulder["x"],
                right_shoulder["y"] - left_shoulder["y"],
                right_shoulder["z"] - left_shoulder["z"],
            ],
            dtype=np.float32,
        )
    else:
        shoulder_center = np.zeros(3, dtype=np.float32)
        shoulder_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    if left_hip and right_hip:
        hip_center = np.array(
            [
                (left_hip["x"] + right_hip["x"]) * 0.5,
                (left_hip["y"] + right_hip["y"]) * 0.5,
                (left_hip["z"] + right_hip["z"]) * 0.5,
            ],
            dtype=np.float32,
        )
        spine_vector = shoulder_center - hip_center
    else:
        hip_center = np.zeros(3, dtype=np.float32)
        spine_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    if np.linalg.norm(shoulder_vector) < 1e-6:
        shoulder_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if np.linalg.norm(spine_vector) < 1e-6:
        spine_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward_vector = np.cross(shoulder_vector, spine_vector)
    if np.linalg.norm(forward_vector) < 1e-6:
        forward_vector = np.array([frontality, 0.0, 1.0], dtype=np.float32)

    def _normalize(vec: np.ndarray) -> list[float]:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return [0.0, 0.0, 0.0]
        return [float(value / norm) for value in vec]

    return LiftedTrackFrame(
        frame=int(frame_index),
        track_id=int(person["track_id"]),
        lifted_keypoints=lifted_keypoints,
        root={
            "x": float(root["x"]),
            "y": float(root["y"]),
            "z": 0.0,
        },
        spine_vector=_normalize(spine_vector),
        shoulder_vector=_normalize(shoulder_vector),
        forward_vector=_normalize(forward_vector),
        frontality=float(frontality),
    )


def lift_track_frames_to_pseudo3d(
    track_payload: dict[str, Any],
    min_keypoint_confidence: float = 0.1,
) -> dict[str, Any]:
    """Lift all frames in one tracked view payload into coarse pseudo-3D."""

    frames_out = []
    track_forward_vectors: dict[int, list[list[float]]] = {}
    track_frontality: dict[int, list[float]] = {}

    for frame in track_payload.get("frames", []):
        frame_idx = int(frame["frame"])
        lifted_people = []
        for person in frame.get("people", []):
            lifted = _lift_person_to_pseudo3d(
                person=person,
                frame_index=frame_idx,
                min_conf=min_keypoint_confidence,
            )
            if lifted is None:
                continue
            lifted_people.append(
                {
                    "track_id": lifted.track_id,
                    "root": lifted.root,
                    "spine_vector": lifted.spine_vector,
                    "shoulder_vector": lifted.shoulder_vector,
                    "forward_vector": lifted.forward_vector,
                    "frontality": lifted.frontality,
                    "lifted_keypoints": lifted.lifted_keypoints,
                }
            )
            track_forward_vectors.setdefault(lifted.track_id, []).append(lifted.forward_vector)
            track_frontality.setdefault(lifted.track_id, []).append(float(lifted.frontality))

        frames_out.append({"frame": frame_idx, "people": lifted_people})

    track_descriptors = []
    for track in track_payload.get("tracks", []):
        track_id = int(track["track_id"])
        vectors = np.asarray(track_forward_vectors.get(track_id, []), dtype=np.float32)
        mean_forward = [0.0, 0.0, 0.0]
        if vectors.size > 0:
            mean_vec = np.mean(vectors, axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm > 1e-6:
                mean_forward = [float(value / norm) for value in mean_vec]
        frontality_values = track_frontality.get(track_id, [])
        track_descriptors.append(
            {
                "track_id": track_id,
                "start_frame": int(track.get("start_frame", 0)),
                "end_frame": int(track.get("end_frame", 0)),
                "num_detections": int(track.get("num_detections", 0)),
                "mean_forward_vector": mean_forward,
                "mean_frontality": float(np.mean(frontality_values)) if frontality_values else 0.0,
                "num_lifted_frames": len(track_forward_vectors.get(track_id, [])),
            }
        )

    return {
        "metadata": {
            **track_payload.get("metadata", {}),
            "lifting_method": "heuristic_pseudo3d_v1",
            "min_keypoint_confidence": float(min_keypoint_confidence),
        },
        "keypoint_names": track_payload.get("keypoint_names", []),
        "frames": frames_out,
        "track_descriptors": sorted(track_descriptors, key=lambda item: int(item["track_id"])),
    }
