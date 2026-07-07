"""Robust all-joint triangulation with triangulation-level RANSAC scoring."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from ..selfcal.hypothesis_geometry import build_track_frame_index, parse_group_nodes
from .run_triangulation import (
    build_fundamental_by_pair,
    enrich_observation,
    filter_observations_by_geometry,
    keypoints_by_id,
    triangulation_angles_deg,
)
from .weighted_triangulation import pixel_reprojection_errors, triangulate_weighted


COCO_BONES = [
    (11, 12),
    (11, 5),
    (12, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 0),
    (6, 0),
]

PARENT_BY_JOINT = {
    12: 11,
    5: 11,
    6: 12,
    7: 5,
    9: 7,
    8: 6,
    10: 8,
    13: 11,
    15: 13,
    14: 12,
    16: 14,
    0: 5,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
}

JOINT_SELECTION_ORDER = [
    11,
    12,
    5,
    6,
    0,
    13,
    14,
    7,
    8,
    15,
    16,
    9,
    10,
    1,
    2,
    3,
    4,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robust all-joint triangulation with RANSAC-style observation selection."
    )
    parser.add_argument("--selected-hypothesis-json", required=True)
    parser.add_argument("--rough-extrinsics-json", required=True)
    parser.add_argument("--track-jsons", nargs="+", required=True)
    parser.add_argument("--view-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--geometry-sampson-threshold", type=float, default=1.0)
    parser.add_argument("--min-pair-support", type=int, default=1)
    parser.add_argument("--ransac-reprojection-threshold", type=float, default=25.0)
    parser.add_argument("--max-mean-reprojection-error", type=float, default=60.0)
    parser.add_argument("--min-triangulation-angle-deg", type=float, default=1.0)
    parser.add_argument("--reprojection-scale-px", type=float, default=25.0)
    parser.add_argument("--temporal-scale-m", type=float, default=0.35)
    parser.add_argument("--bone-ratio-soft-limit", type=float, default=0.55)
    parser.add_argument("--bone-ratio-hard-limit", type=float, default=1.25)
    parser.add_argument("--min-bone-prior-samples", type=int, default=8)
    parser.add_argument(
        "--preferred-min-inlier-views",
        type=int,
        default=3,
        help="Prefer candidates with at least this many inlier views when available.",
    )
    parser.add_argument(
        "--two-view-penalty",
        type=float,
        default=0.35,
        help="Quality penalty for two-view-only candidates when more observations exist.",
    )
    parser.add_argument(
        "--max-point-norm",
        type=float,
        default=8.0,
        help="Reject candidates whose world-space norm is above this value; set <=0 to disable.",
    )
    parser.add_argument(
        "--max-root-distance-ratio",
        type=float,
        default=4.0,
        help="Softly penalize candidates too far from the current torso anchor.",
    )
    parser.add_argument(
        "--root-distance-penalty",
        type=float,
        default=0.45,
        help="Quality penalty strength for candidates beyond max-root-distance-ratio.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_joint_observations(
    frame_idx: int,
    group: dict[str, Any],
    frame_indexes: dict[str, dict[int, dict[int, dict[str, Any]]]],
    extrinsics_by_view: dict[str, Any],
    resolved_views: set[str],
    min_confidence: float,
) -> dict[int, list[dict[str, Any]]]:
    """Collect all per-joint observations for one global identity."""

    group_tracks = parse_group_nodes(group)
    joint_observations: dict[int, list[dict[str, Any]]] = {}
    for view_id, track_id in group_tracks.items():
        if view_id not in resolved_views:
            continue
        person = frame_indexes.get(view_id, {}).get(frame_idx, {}).get(track_id)
        if person is None:
            continue
        visible = keypoints_by_id(person, min_confidence=min_confidence)
        for joint_id, point in visible.items():
            joint_observations.setdefault(joint_id, []).append(
                enrich_observation(
                    view_id=view_id,
                    point=point,
                    extrinsics_by_view=extrinsics_by_view,
                )
            )
    return joint_observations


def ransac_candidates(
    observations: list[dict[str, Any]],
    min_views: int,
    ransac_reprojection_threshold: float,
    max_point_norm: float,
) -> list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]]:
    """Generate triangulation candidates and inlier sets from view subsets."""

    candidates = []
    if len(observations) < int(min_views):
        return candidates

    num_candidates = 0
    for subset_size in range(int(min_views), len(observations) + 1):
        for indices in combinations(range(len(observations)), subset_size):
            subset = [observations[idx] for idx in indices]
            seed = triangulate_weighted(subset)
            num_candidates += 1
            if seed.get("status") != "ok":
                continue

            seed_point = np.asarray(seed["point_3d"], dtype=np.float64)
            if float(max_point_norm) > 0.0 and float(np.linalg.norm(seed_point)) > float(max_point_norm):
                continue
            all_errors = pixel_reprojection_errors(seed_point, observations)
            inlier_indices = [
                idx
                for idx, error in enumerate(all_errors)
                if np.isfinite(error) and float(error) <= float(ransac_reprojection_threshold)
            ]
            if len(inlier_indices) < int(min_views):
                continue

            inlier_observations = [observations[idx] for idx in inlier_indices]
            refined = triangulate_weighted(inlier_observations)
            if refined.get("status") != "ok":
                continue
            point_3d = np.asarray(refined["point_3d"], dtype=np.float64)
            if float(max_point_norm) > 0.0 and float(np.linalg.norm(point_3d)) > float(max_point_norm):
                continue
            inlier_errors = pixel_reprojection_errors(point_3d, inlier_observations)
            all_refined_errors = pixel_reprojection_errors(point_3d, observations)
            finite_inlier_errors = [float(err) for err in inlier_errors if np.isfinite(err)]
            if not finite_inlier_errors:
                continue

            outlier_views = [
                observations[idx]["view_id"]
                for idx, error in enumerate(all_refined_errors)
                if not np.isfinite(error) or float(error) > float(ransac_reprojection_threshold)
            ]
            angles = triangulation_angles_deg(point_3d, inlier_observations)
            diagnostics = {
                "seed_views": [obs["view_id"] for obs in subset],
                "inlier_views": [obs["view_id"] for obs in inlier_observations],
                "outlier_views": outlier_views,
                "candidate_num_views": len(observations),
                "num_inlier_views": len(inlier_observations),
                "num_outlier_views": len(outlier_views),
                "mean_pixel_reprojection_error": float(np.mean(finite_inlier_errors)),
                "max_pixel_reprojection_error": float(np.max(finite_inlier_errors)),
                "mean_confidence": float(np.mean([obs["confidence"] for obs in inlier_observations])),
                "triangulation_angles_deg": angles,
                "min_triangulation_angle_deg": float(np.min(angles)) if angles else 0.0,
                "num_ransac_subset_candidates": num_candidates,
            }
            refined["pixel_reprojection_errors"] = inlier_errors
            refined["triangulation_angles_deg"] = angles
            candidates.append((refined, inlier_observations, diagnostics))

    return candidates


def bone_score(
    identity_points: dict[int, np.ndarray],
    joint_id: int,
    point_3d: np.ndarray,
    bone_priors: dict[tuple[int, int], float],
    soft_limit: float,
    hard_limit: float,
) -> tuple[float, dict[str, Any] | None]:
    """Score one candidate by parent-child bone consistency."""

    parent_id = PARENT_BY_JOINT.get(int(joint_id))
    if parent_id is None or parent_id not in identity_points:
        return 1.0, None
    parent_point = identity_points[parent_id]
    key = (parent_id, int(joint_id))
    reverse_key = (int(joint_id), parent_id)
    prior_length = bone_priors.get(key, bone_priors.get(reverse_key))
    if prior_length is None or prior_length <= 1e-9:
        return 1.0, None

    length = float(np.linalg.norm(point_3d - parent_point))
    ratio = length / float(prior_length)
    deviation = abs(ratio - 1.0)
    if deviation <= float(soft_limit):
        score = 1.0
    elif deviation >= float(hard_limit):
        score = 0.05
    else:
        span = max(float(hard_limit) - float(soft_limit), 1e-6)
        score = 1.0 - 0.95 * ((deviation - float(soft_limit)) / span)
    return float(np.clip(score, 0.05, 1.0)), {
        "parent_joint_id": parent_id,
        "bone_length": length,
        "prior_length": float(prior_length),
        "length_over_prior": ratio,
        "bone_score": float(np.clip(score, 0.05, 1.0)),
    }


def temporal_score(
    previous_points: dict[tuple[int, int], np.ndarray],
    identity_id: int,
    joint_id: int,
    point_3d: np.ndarray,
    temporal_scale_m: float,
) -> tuple[float, float | None]:
    """Score one candidate by temporal displacement from the previous accepted point."""

    previous = previous_points.get((int(identity_id), int(joint_id)))
    if previous is None:
        return 1.0, None
    displacement = float(np.linalg.norm(point_3d - previous))
    score = 1.0 / (1.0 + displacement / max(float(temporal_scale_m), 1e-6))
    return float(np.clip(score, 0.05, 1.0)), displacement


def current_torso_anchor(identity_points: dict[int, np.ndarray]) -> tuple[np.ndarray | None, float | None, int]:
    """Estimate a provisional torso anchor from already accepted torso joints."""

    torso_points = [
        identity_points[joint_id]
        for joint_id in (5, 6, 11, 12)
        if joint_id in identity_points
    ]
    if not torso_points:
        return None, None, 0
    center = np.mean(np.vstack(torso_points), axis=0)
    torso_lengths = []
    for joint_a, joint_b in ((5, 6), (11, 12), (5, 11), (6, 12)):
        if joint_a in identity_points and joint_b in identity_points:
            torso_lengths.append(float(np.linalg.norm(identity_points[joint_b] - identity_points[joint_a])))
    if torso_lengths:
        scale = float(np.median([length for length in torso_lengths if length > 1e-9]))
    else:
        scale = None
    return center, scale, len(torso_points)


def score_candidate(
    tri: dict[str, Any],
    observations: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    identity_points: dict[int, np.ndarray],
    previous_points: dict[tuple[int, int], np.ndarray],
    identity_id: int,
    joint_id: int,
    bone_priors: dict[tuple[int, int], float],
    min_triangulation_angle_deg: float,
    reprojection_scale_px: float,
    temporal_scale_m: float,
    bone_ratio_soft_limit: float,
    bone_ratio_hard_limit: float,
    preferred_min_inlier_views: int,
    two_view_penalty: float,
    max_root_distance_ratio: float,
    root_distance_penalty: float,
) -> tuple[float, float, dict[str, Any]]:
    """Return sortable score and normalized quality for one candidate."""

    point_3d = np.asarray(tri["point_3d"], dtype=np.float64)
    mean_error = float(diagnostics["mean_pixel_reprojection_error"])
    max_error = float(diagnostics["max_pixel_reprojection_error"])
    mean_confidence = float(diagnostics["mean_confidence"])
    num_inliers = int(diagnostics["num_inlier_views"])
    candidate_views = int(diagnostics["candidate_num_views"])
    min_angle = float(diagnostics["min_triangulation_angle_deg"])

    reproj_quality = 1.0 / (1.0 + mean_error / max(float(reprojection_scale_px), 1e-6))
    angle_quality = min(min_angle / max(float(min_triangulation_angle_deg), 1e-6), 1.0)
    view_quality = num_inliers / max(candidate_views, 1)
    view_count_penalty = 0.0
    if candidate_views >= int(preferred_min_inlier_views) and num_inliers < int(preferred_min_inlier_views):
        view_count_penalty = float(two_view_penalty)
    confidence_quality = float(np.clip(mean_confidence, 0.0, 1.0))
    bone_quality, bone_diag = bone_score(
        identity_points=identity_points,
        joint_id=joint_id,
        point_3d=point_3d,
        bone_priors=bone_priors,
        soft_limit=bone_ratio_soft_limit,
        hard_limit=bone_ratio_hard_limit,
    )
    temp_quality, displacement = temporal_score(
        previous_points=previous_points,
        identity_id=identity_id,
        joint_id=joint_id,
        point_3d=point_3d,
        temporal_scale_m=temporal_scale_m,
    )
    anchor_center, anchor_scale, anchor_joint_count = current_torso_anchor(identity_points)
    root_distance_ratio = None
    root_distance_quality = 1.0
    root_distance_penalty_value = 0.0
    if anchor_center is not None and anchor_scale is not None and anchor_scale > 1e-9:
        root_distance_ratio = float(np.linalg.norm(point_3d - anchor_center) / anchor_scale)
        root_distance_quality = 1.0 / (1.0 + max(0.0, root_distance_ratio - 1.0))
        # Only enforce this once at least two torso joints have established a usable anchor.
        if anchor_joint_count >= 2 and root_distance_ratio > float(max_root_distance_ratio):
            overflow = (root_distance_ratio - float(max_root_distance_ratio)) / max(float(max_root_distance_ratio), 1e-6)
            root_distance_penalty_value = float(root_distance_penalty) * min(max(overflow, 0.0), 1.0)
    quality = (
        0.30 * reproj_quality
        + 0.18 * confidence_quality
        + 0.14 * view_quality
        + 0.10 * angle_quality
        + 0.14 * bone_quality
        + 0.08 * temp_quality
        + 0.06 * root_distance_quality
    )
    quality = max(0.0, quality - view_count_penalty - root_distance_penalty_value)
    penalty = max(0.0, max_error - float(reprojection_scale_px)) * 0.04
    score = -quality + penalty
    extra = {
        "joint_quality": float(np.clip(quality, 0.0, 1.0)),
        "reprojection_quality": float(reproj_quality),
        "confidence_quality": float(confidence_quality),
        "view_quality": float(view_quality),
        "angle_quality": float(angle_quality),
        "bone_quality": float(bone_quality),
        "temporal_quality": float(temp_quality),
        "root_distance_quality": float(root_distance_quality),
        "root_distance_ratio": root_distance_ratio,
        "root_distance_penalty": float(root_distance_penalty_value),
        "torso_anchor_joint_count": int(anchor_joint_count),
        "view_count_penalty": float(view_count_penalty),
        "temporal_displacement": displacement,
        "bone_diagnostic": bone_diag,
        "robust_candidate_score": float(score),
    }
    return float(score), float(quality), extra


def estimate_bone_priors_from_frames(
    frames: list[dict[str, Any]],
    min_samples: int,
) -> dict[int, dict[tuple[int, int], float]]:
    """Estimate median bone lengths by identity from preliminary frames."""

    lengths: dict[int, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for frame in frames:
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            points = {
                int(joint["id"]): np.array([joint["x"], joint["y"], joint["z"]], dtype=np.float64)
                for joint in identity.get("joints", [])
            }
            for joint_a, joint_b in COCO_BONES:
                if joint_a not in points or joint_b not in points:
                    continue
                length = float(np.linalg.norm(points[joint_b] - points[joint_a]))
                if np.isfinite(length) and length > 1e-9:
                    lengths[identity_id][(joint_a, joint_b)].append(length)

    priors: dict[int, dict[tuple[int, int], float]] = {}
    for identity_id, bone_values in lengths.items():
        priors[identity_id] = {}
        for bone, values in bone_values.items():
            if len(values) >= int(min_samples):
                priors[identity_id][bone] = float(median(values))
    return priors


def robust_triangulate_sequence(
    selected_hypothesis: dict[str, Any],
    rough_extrinsics: dict[str, Any],
    track_payloads: dict[str, dict[str, Any]],
    view_ids: list[str],
    args: argparse.Namespace,
    bone_priors_by_identity: dict[int, dict[tuple[int, int], float]] | None = None,
) -> dict[str, Any]:
    """Triangulate one sequence with robust all-joint candidate selection."""

    frame_indexes = {
        view_id: build_track_frame_index(track_payloads[view_id])
        for view_id in view_ids
    }
    groups = list(selected_hypothesis.get("groups", []))
    extrinsics_by_view = rough_extrinsics.get("extrinsics_by_view", {})
    fundamental_by_pair = build_fundamental_by_pair(selected_hypothesis)
    resolved_views = {view_id for view_id in extrinsics_by_view}
    all_frames = sorted({
        frame_idx
        for frame_map in frame_indexes.values()
        for frame_idx in frame_map.keys()
    })[:: max(int(args.frame_step), 1)]

    output_frames = []
    previous_points: dict[tuple[int, int], np.ndarray] = {}
    summary_counts = defaultdict(int)
    reprojection_values: list[float] = []
    quality_values: list[float] = []

    for frame_idx in all_frames:
        identities_out = []
        for group in groups:
            identity_id = int(group["group_id"])
            joint_observations = collect_joint_observations(
                frame_idx=frame_idx,
                group=group,
                frame_indexes=frame_indexes,
                extrinsics_by_view=extrinsics_by_view,
                resolved_views=resolved_views,
                min_confidence=float(args.min_confidence),
            )
            identity_points: dict[int, np.ndarray] = {}
            joints_out = []
            bone_priors = (bone_priors_by_identity or {}).get(identity_id, {})

            ordered_joint_ids = [
                joint_id
                for joint_id in JOINT_SELECTION_ORDER
                if joint_id in joint_observations
            ]
            ordered_joint_ids.extend(
                sorted(joint_id for joint_id in joint_observations if joint_id not in set(JOINT_SELECTION_ORDER))
            )

            for joint_id in ordered_joint_ids:
                observations = joint_observations[joint_id]
                summary_counts["total_candidate_observations"] += len(observations)
                if len(observations) < int(args.min_views):
                    continue
                geometry_filtered, geometry_pairs = filter_observations_by_geometry(
                    observations=observations,
                    fundamental_by_pair=fundamental_by_pair,
                    sampson_threshold=float(args.geometry_sampson_threshold),
                    min_pair_support=int(args.min_pair_support),
                )
                summary_counts["total_geometry_kept_observations"] += len(geometry_filtered)
                if len(geometry_filtered) < int(args.min_views):
                    continue

                candidates = ransac_candidates(
                    observations=geometry_filtered,
                    min_views=int(args.min_views),
                    ransac_reprojection_threshold=float(args.ransac_reprojection_threshold),
                    max_point_norm=float(args.max_point_norm),
                )
                summary_counts["total_ransac_candidates"] += len(candidates)
                if not candidates:
                    continue

                best = None
                for tri, candidate_observations, diagnostics in candidates:
                    if float(diagnostics["mean_pixel_reprojection_error"]) > float(args.max_mean_reprojection_error):
                        continue
                    score, quality, extra = score_candidate(
                        tri=tri,
                        observations=candidate_observations,
                        diagnostics=diagnostics,
                        identity_points=identity_points,
                        previous_points=previous_points,
                        identity_id=identity_id,
                        joint_id=joint_id,
                        bone_priors=bone_priors,
                        min_triangulation_angle_deg=float(args.min_triangulation_angle_deg),
                        reprojection_scale_px=float(args.reprojection_scale_px),
                        temporal_scale_m=float(args.temporal_scale_m),
                        bone_ratio_soft_limit=float(args.bone_ratio_soft_limit),
                        bone_ratio_hard_limit=float(args.bone_ratio_hard_limit),
                        preferred_min_inlier_views=int(args.preferred_min_inlier_views),
                        two_view_penalty=float(args.two_view_penalty),
                        max_root_distance_ratio=float(args.max_root_distance_ratio),
                        root_distance_penalty=float(args.root_distance_penalty),
                    )
                    if best is None or score < best[0]:
                        best = (score, quality, extra, tri, candidate_observations, diagnostics)

                if best is None:
                    continue
                _, quality, extra, tri, final_observations, diagnostics = best
                point_3d = np.asarray(tri["point_3d"], dtype=np.float64)
                identity_points[int(joint_id)] = point_3d
                previous_points[(identity_id, int(joint_id))] = point_3d
                summary_counts["total_final_kept_observations"] += len(final_observations)
                summary_counts["total_triangulated_points"] += 1
                reprojection_values.append(float(diagnostics["mean_pixel_reprojection_error"]))
                quality_values.append(float(quality))
                joints_out.append(
                    {
                        "id": int(joint_id),
                        "x": float(point_3d[0]),
                        "y": float(point_3d[1]),
                        "z": float(point_3d[2]),
                        "num_views": len(final_observations),
                        "candidate_num_views": len(observations),
                        "geometry_filtered_num_views": len(geometry_filtered),
                        "mean_confidence": float(diagnostics["mean_confidence"]),
                        "mean_reprojection_error_px": float(diagnostics["mean_pixel_reprojection_error"]),
                        "max_reprojection_error_px": float(diagnostics["max_pixel_reprojection_error"]),
                        "views": [obs["view_id"] for obs in final_observations],
                        "inlier_views": list(diagnostics["inlier_views"]),
                        "outlier_views": list(diagnostics["outlier_views"]),
                        "geometry_pair_checks": geometry_pairs,
                        "triangulation_angles_deg": tri.get("triangulation_angles_deg"),
                        "robust_selection": {**diagnostics, **extra},
                    }
                )

            if joints_out:
                identities_out.append(
                    {
                        "identity_id": identity_id,
                        "num_joints": len(joints_out),
                        "joints": joints_out,
                    }
                )
        if identities_out:
            output_frames.append({"frame": frame_idx, "identities": identities_out})

    total_candidate_observations = int(summary_counts["total_candidate_observations"])
    total_geometry_kept = int(summary_counts["total_geometry_kept_observations"])
    total_final_kept = int(summary_counts["total_final_kept_observations"])
    return {
        "metadata": {
            "stage": "robust_all_joint_triangulation_v5",
            "anchor_view": rough_extrinsics.get("anchor_view"),
            "view_ids": list(view_ids),
            "min_confidence": float(args.min_confidence),
            "min_views": int(args.min_views),
            "geometry_sampson_threshold": float(args.geometry_sampson_threshold),
            "ransac_reprojection_threshold": float(args.ransac_reprojection_threshold),
            "max_mean_reprojection_error": float(args.max_mean_reprojection_error),
            "preferred_min_inlier_views": int(args.preferred_min_inlier_views),
            "two_view_penalty": float(args.two_view_penalty),
            "max_point_norm": float(args.max_point_norm),
            "max_root_distance_ratio": float(args.max_root_distance_ratio),
            "root_distance_penalty": float(args.root_distance_penalty),
        },
        "frames": output_frames,
        "summary": {
            "num_frames_with_3d": len(output_frames),
            "total_triangulated_points": int(summary_counts["total_triangulated_points"]),
            "mean_reprojection_error_px": float(np.mean(reprojection_values)) if reprojection_values else None,
            "mean_joint_quality": float(np.mean(quality_values)) if quality_values else None,
            "total_candidate_observations": total_candidate_observations,
            "total_geometry_kept_observations": total_geometry_kept,
            "total_final_kept_observations": total_final_kept,
            "total_ransac_candidates": int(summary_counts["total_ransac_candidates"]),
            "geometry_keep_ratio": (
                float(total_geometry_kept / total_candidate_observations)
                if total_candidate_observations
                else None
            ),
            "final_keep_ratio": (
                float(total_final_kept / total_candidate_observations)
                if total_candidate_observations
                else None
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if len(args.track_jsons) != len(args.view_ids):
        raise ValueError("track-jsons and view-ids must have the same length.")

    selected_payload = load_json(args.selected_hypothesis_json)
    selected_hypothesis = selected_payload["selected_hypothesis"]
    rough_extrinsics = load_json(args.rough_extrinsics_json)
    track_payloads = {
        view_id: load_json(track_json)
        for view_id, track_json in zip(args.view_ids, args.track_jsons)
    }

    preliminary = robust_triangulate_sequence(
        selected_hypothesis=selected_hypothesis,
        rough_extrinsics=rough_extrinsics,
        track_payloads=track_payloads,
        view_ids=list(args.view_ids),
        args=args,
        bone_priors_by_identity={},
    )
    priors = estimate_bone_priors_from_frames(
        preliminary.get("frames", []),
        min_samples=int(args.min_bone_prior_samples),
    )
    final = robust_triangulate_sequence(
        selected_hypothesis=selected_hypothesis,
        rough_extrinsics=rough_extrinsics,
        track_payloads=track_payloads,
        view_ids=list(args.view_ids),
        args=args,
        bone_priors_by_identity=priors,
    )
    final["metadata"]["bone_prior_counts"] = {
        str(identity_id): len(bones)
        for identity_id, bones in priors.items()
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "triangulated_3d.json").open("w", encoding="utf-8") as handle:
        json.dump(final, handle, ensure_ascii=False, indent=2)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(final["summary"], handle, ensure_ascii=False, indent=2)
    with (output_dir / "bone_priors.json").open("w", encoding="utf-8") as handle:
        serializable_priors = {
            str(identity_id): {
                f"{bone[0]}-{bone[1]}": length
                for bone, length in bones.items()
            }
            for identity_id, bones in priors.items()
        }
        json.dump(serializable_priors, handle, ensure_ascii=False, indent=2)
    print(json.dumps(final["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
