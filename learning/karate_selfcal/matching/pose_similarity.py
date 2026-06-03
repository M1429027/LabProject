"""Pose-shape similarity helpers for cross-view association."""

from __future__ import annotations

from typing import Any

import numpy as np


def _midpoint(a: dict[str, float] | None, b: dict[str, float] | None) -> tuple[float, float] | None:
    """Return the midpoint between two keypoints."""

    if a is None or b is None:
        return None
    return ((float(a["x"]) + float(b["x"])) * 0.5, (float(a["y"]) + float(b["y"])) * 0.5)


def _keypoint_map(person: dict[str, Any], min_conf: float) -> dict[int, dict[str, float]]:
    """Index visible keypoints by joint id."""

    out: dict[int, dict[str, float]] = {}
    for point in person.get("keypoints", []):
        conf = float(point.get("confidence", 0.0))
        if conf < min_conf:
            continue
        out[int(point["id"])] = {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "confidence": conf,
        }
    return out


def _body_root(points: dict[int, dict[str, float]], bbox: dict[str, Any] | None) -> tuple[float, float]:
    """Choose a stable body root for normalization."""

    hips = _midpoint(points.get(11), points.get(12))
    if hips is not None:
        return hips
    shoulders = _midpoint(points.get(5), points.get(6))
    if shoulders is not None:
        return shoulders
    if bbox:
        return (
            (float(bbox["x1"]) + float(bbox["x2"])) * 0.5,
            (float(bbox["y1"]) + float(bbox["y2"])) * 0.5,
        )
    return (0.0, 0.0)


def _body_scale(points: dict[int, dict[str, float]], bbox: dict[str, Any] | None) -> float:
    """Estimate person scale from torso geometry or bounding box."""

    scales = []
    shoulders = _midpoint(points.get(5), points.get(6))
    hips = _midpoint(points.get(11), points.get(12))
    if shoulders is not None and hips is not None:
        scales.append(
            ((shoulders[0] - hips[0]) ** 2 + (shoulders[1] - hips[1]) ** 2) ** 0.5
        )
    if 5 in points and 6 in points:
        scales.append(abs(float(points[6]["x"]) - float(points[5]["x"])))
    if 11 in points and 12 in points:
        scales.append(abs(float(points[12]["x"]) - float(points[11]["x"])))
    if bbox:
        scales.append(
            max(
                float(bbox["x2"]) - float(bbox["x1"]),
                float(bbox["y2"]) - float(bbox["y1"]),
            )
        )
    scales = [value for value in scales if value > 1.0]
    return max(float(np.mean(scales)) if scales else 100.0, 50.0)


def _normalized_pose(person: dict[str, Any], min_conf: float) -> dict[int, np.ndarray]:
    """Convert one 2D skeleton into root-centered normalized coordinates."""

    points = _keypoint_map(person, min_conf=min_conf)
    root_x, root_y = _body_root(points, person.get("bbox"))
    scale = _body_scale(points, person.get("bbox"))
    normalized: dict[int, np.ndarray] = {}
    for point_id, point in points.items():
        normalized[point_id] = np.array(
            [
                (float(point["x"]) - root_x) / scale,
                (float(point["y"]) - root_y) / scale,
            ],
            dtype=np.float32,
        )
    return normalized


def score_pose_similarity(
    person_a: dict[str, Any],
    person_b: dict[str, Any],
    min_conf: float = 0.1,
) -> float:
    """Compare two 2D skeletons in a view-invariant normalized pose space."""

    pose_a = _normalized_pose(person_a, min_conf=min_conf)
    pose_b = _normalized_pose(person_b, min_conf=min_conf)
    common = sorted(set(pose_a) & set(pose_b))
    if len(common) < 5:
        return 0.0

    distances = []
    for point_id in common:
        distances.append(float(np.linalg.norm(pose_a[point_id] - pose_b[point_id])))

    mean_dist = float(np.mean(distances))
    score = 1.0 - min(mean_dist / 1.5, 1.0)
    return max(0.0, score)


def score_visibility_similarity(
    person_a: dict[str, Any],
    person_b: dict[str, Any],
    min_conf: float = 0.1,
) -> float:
    """Compare which joints are visible in two pose detections."""

    joints = set()
    visibility_a = {}
    visibility_b = {}
    for point in person_a.get("keypoints", []):
        point_id = int(point["id"])
        joints.add(point_id)
        visibility_a[point_id] = float(point.get("confidence", 0.0)) >= min_conf
    for point in person_b.get("keypoints", []):
        point_id = int(point["id"])
        joints.add(point_id)
        visibility_b[point_id] = float(point.get("confidence", 0.0)) >= min_conf

    if not joints:
        return 0.0

    matches = 0
    for point_id in joints:
        if visibility_a.get(point_id, False) == visibility_b.get(point_id, False):
            matches += 1
    return float(matches / len(joints))
