"""Geometry checks for Stage 3B global identity hypotheses."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import cv2
import numpy as np


def build_track_frame_index(track_payload: dict[str, Any]) -> dict[int, dict[int, dict[str, Any]]]:
    """Index a tracking payload as frame -> track_id -> person."""

    frame_index: dict[int, dict[int, dict[str, Any]]] = {}
    for frame in track_payload.get("frames", []):
        frame_idx = int(frame["frame"])
        frame_index[frame_idx] = {
            int(person["track_id"]): person
            for person in frame.get("people", [])
            if "track_id" in person
        }
    return frame_index


def _keypoints_by_id(person: dict[str, Any], min_confidence: float) -> dict[int, tuple[float, float, float]]:
    """Return visible keypoints as joint_id -> (x, y, confidence)."""

    points = {}
    for point in person.get("keypoints", []):
        confidence = float(point.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        points[int(point["id"])] = (
            float(point["x"]),
            float(point["y"]),
            confidence,
        )
    return points


def parse_group_nodes(group: dict[str, Any]) -> dict[str, int]:
    """Parse group nodes like view_id:track_id into a view -> track map."""

    out = {}
    for node in group.get("nodes", []):
        view_id, track_id = str(node).rsplit(":", 1)
        out[view_id] = int(track_id)
    return out


def collect_correspondences_for_view_pair(
    groups: list[dict[str, Any]],
    frame_indexes: dict[str, dict[int, dict[int, dict[str, Any]]]],
    view_a: str,
    view_b: str,
    min_confidence: float,
    frame_step: int = 1,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Collect same-identity, same-frame, same-joint 2D correspondences."""

    points_a: list[tuple[float, float]] = []
    points_b: list[tuple[float, float]] = []
    metadata: list[dict[str, Any]] = []
    frame_step = max(int(frame_step), 1)

    for group in groups:
        group_id = int(group["group_id"])
        group_tracks = parse_group_nodes(group)
        if view_a not in group_tracks or view_b not in group_tracks:
            continue

        track_a = group_tracks[view_a]
        track_b = group_tracks[view_b]
        frames = sorted(set(frame_indexes[view_a]) & set(frame_indexes[view_b]))
        for frame_idx in frames[::frame_step]:
            person_a = frame_indexes[view_a].get(frame_idx, {}).get(track_a)
            person_b = frame_indexes[view_b].get(frame_idx, {}).get(track_b)
            if person_a is None or person_b is None:
                continue
            kpts_a = _keypoints_by_id(person_a, min_confidence)
            kpts_b = _keypoints_by_id(person_b, min_confidence)
            for joint_id in sorted(set(kpts_a) & set(kpts_b)):
                points_a.append((kpts_a[joint_id][0], kpts_a[joint_id][1]))
                points_b.append((kpts_b[joint_id][0], kpts_b[joint_id][1]))
                metadata.append(
                    {
                        "group_id": group_id,
                        "frame": frame_idx,
                        "joint_id": joint_id,
                        "track_a": track_a,
                        "track_b": track_b,
                        "confidence_a": kpts_a[joint_id][2],
                        "confidence_b": kpts_b[joint_id][2],
                    }
                )

    if max_points is not None and len(points_a) > max_points:
        # Deterministic uniform subsampling keeps long clips from dominating runtime.
        indices = np.linspace(0, len(points_a) - 1, int(max_points), dtype=np.int64)
        points_a = [points_a[int(idx)] for idx in indices]
        points_b = [points_b[int(idx)] for idx in indices]
        metadata = [metadata[int(idx)] for idx in indices]

    return (
        np.asarray(points_a, dtype=np.float32),
        np.asarray(points_b, dtype=np.float32),
        metadata,
    )


def sampson_errors(points_a: np.ndarray, points_b: np.ndarray, fundamental: np.ndarray) -> np.ndarray:
    """Compute Sampson approximation of geometric reprojection error."""

    ones = np.ones((points_a.shape[0], 1), dtype=np.float64)
    x1 = np.hstack([points_a.astype(np.float64), ones])
    x2 = np.hstack([points_b.astype(np.float64), ones])
    fx1 = fundamental @ x1.T
    ftx2 = fundamental.T @ x2.T
    numerator = np.sum(x2 * (fundamental @ x1.T).T, axis=1) ** 2
    denominator = fx1[0] ** 2 + fx1[1] ** 2 + ftx2[0] ** 2 + ftx2[1] ** 2
    return numerator / np.maximum(denominator, 1e-12)


def score_fundamental_geometry(
    points_a: np.ndarray,
    points_b: np.ndarray,
    ransac_threshold: float = 3.0,
    confidence: float = 0.99,
) -> dict[str, Any]:
    """Estimate a fundamental matrix and summarize robust epipolar consistency."""

    if len(points_a) < 8 or len(points_b) < 8:
        return {
            "status": "not_enough_points",
            "num_points": int(len(points_a)),
            "inlier_ratio": 0.0,
            "median_sampson_error": None,
            "mean_sampson_error": None,
            "geometry_score": 0.0,
        }

    fundamental, mask = cv2.findFundamentalMat(
        points_a,
        points_b,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=float(ransac_threshold),
        confidence=float(confidence),
    )
    if fundamental is None or mask is None:
        return {
            "status": "fit_failed",
            "num_points": int(len(points_a)),
            "inlier_ratio": 0.0,
            "median_sampson_error": None,
            "mean_sampson_error": None,
            "geometry_score": 0.0,
        }

    if fundamental.shape[0] > 3:
        fundamental = fundamental[:3, :]
    inlier_mask = mask.ravel().astype(bool)
    errors = sampson_errors(points_a, points_b, fundamental)
    inlier_errors = errors[inlier_mask] if np.any(inlier_mask) else errors
    median_error = float(np.median(inlier_errors)) if len(inlier_errors) else None
    mean_error = float(np.mean(inlier_errors)) if len(inlier_errors) else None
    inlier_ratio = float(np.mean(inlier_mask)) if len(inlier_mask) else 0.0
    error_term = 1.0 / (1.0 + float(median_error or 1e6))

    return {
        "status": "ok",
        "num_points": int(len(points_a)),
        "num_inliers": int(np.sum(inlier_mask)),
        "inlier_ratio": inlier_ratio,
        "median_sampson_error": median_error,
        "mean_sampson_error": mean_error,
        "geometry_score": float(inlier_ratio * error_term),
        "fundamental_matrix": fundamental.tolist(),
    }


def evaluate_hypothesis_geometry(
    hypothesis: dict[str, Any],
    frame_indexes: dict[str, dict[int, dict[int, dict[str, Any]]]],
    view_ids: list[str],
    min_confidence: float = 0.3,
    frame_step: int = 1,
    max_points_per_pair: int | None = None,
    ransac_threshold: float = 3.0,
) -> dict[str, Any]:
    """Evaluate one global identity hypothesis with pairwise epipolar geometry."""

    pair_results = []
    for view_a, view_b in combinations(view_ids, 2):
        points_a, points_b, metadata = collect_correspondences_for_view_pair(
            groups=list(hypothesis.get("groups", [])),
            frame_indexes=frame_indexes,
            view_a=view_a,
            view_b=view_b,
            min_confidence=min_confidence,
            frame_step=frame_step,
            max_points=max_points_per_pair,
        )
        geometry = score_fundamental_geometry(
            points_a,
            points_b,
            ransac_threshold=ransac_threshold,
        )
        pair_results.append(
            {
                "view_a": view_a,
                "view_b": view_b,
                "num_correspondences": int(len(metadata)),
                **geometry,
            }
        )

    valid_pairs = [pair for pair in pair_results if pair.get("status") == "ok"]
    mean_inlier_ratio = float(np.mean([pair["inlier_ratio"] for pair in valid_pairs])) if valid_pairs else 0.0
    mean_geometry_score = float(np.mean([pair["geometry_score"] for pair in valid_pairs])) if valid_pairs else 0.0
    median_error_values = [
        float(pair["median_sampson_error"])
        for pair in valid_pairs
        if pair.get("median_sampson_error") is not None
    ]
    mean_median_sampson_error = float(np.mean(median_error_values)) if median_error_values else None

    return {
        "rank": int(hypothesis["rank"]),
        "hypothesis_id": int(hypothesis["hypothesis_id"]),
        "original_mean_pair_score": float(hypothesis.get("mean_pair_score", 0.0)),
        "original_min_pair_score": float(hypothesis.get("min_pair_score", 0.0)),
        "assignments": hypothesis.get("assignments", {}),
        "groups": hypothesis.get("groups", []),
        "num_valid_view_pairs": len(valid_pairs),
        "mean_inlier_ratio": mean_inlier_ratio,
        "mean_median_sampson_error": mean_median_sampson_error,
        "mean_geometry_score": mean_geometry_score,
        "pair_results": pair_results,
    }
