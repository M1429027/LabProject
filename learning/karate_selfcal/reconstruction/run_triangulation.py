"""CLI for rough triangulation from selected identity hypotheses and rough extrinsics."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ..selfcal.hypothesis_geometry import build_track_frame_index, parse_group_nodes
from .weighted_triangulation import (
    pixel_reprojection_errors,
    sampson_error,
    triangulate_weighted,
    undistort_point,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run rough triangulation from selected identities and rough extrinsics."
    )
    parser.add_argument("--selected-hypothesis-json", required=True, help="Path to selected_hypothesis.json")
    parser.add_argument("--rough-extrinsics-json", required=True, help="Path to rough_extrinsics.json")
    parser.add_argument("--track-jsons", nargs="+", required=True, help="Per-view tracking JSON files")
    parser.add_argument("--view-ids", nargs="+", required=True, help="View ids matching --track-jsons")
    parser.add_argument("--output-dir", required=True, help="Directory for triangulation outputs")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum keypoint confidence")
    parser.add_argument("--min-views", type=int, default=2, help="Minimum number of views required per joint")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame")
    parser.add_argument(
        "--geometry-sampson-threshold",
        type=float,
        default=1.0,
        help="Maximum Sampson error for a pairwise geometry agreement.",
    )
    parser.add_argument(
        "--min-pair-support",
        type=int,
        default=1,
        help="Minimum number of supporting pairwise inliers required per observation.",
    )
    parser.add_argument(
        "--max-pixel-reprojection-error",
        type=float,
        default=60.0,
        help="Iteratively drop the worst observation if reprojection error exceeds this threshold.",
    )
    parser.add_argument(
        "--use-view-subset-selection",
        action="store_true",
        help="Try candidate view subsets per joint and keep the best triangulation-quality subset.",
    )
    parser.add_argument(
        "--min-triangulation-angle-deg",
        type=float,
        default=1.0,
        help="Preferred minimum ray angle for view-subset scoring.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def keypoints_by_id(person: dict[str, Any], min_confidence: float) -> dict[int, dict[str, float]]:
    """Index visible 2D keypoints for one person."""

    out = {}
    for point in person.get("keypoints", []):
        confidence = float(point.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        out[int(point["id"])] = {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "confidence": confidence,
        }
    return out


def build_fundamental_by_pair(selected_hypothesis: dict[str, Any]) -> dict[tuple[str, str], np.ndarray]:
    """Index pairwise fundamental matrices for fast inlier checks."""

    lookup: dict[tuple[str, str], np.ndarray] = {}
    for pair in selected_hypothesis.get("pair_results", []):
        if pair.get("status") != "ok" or not pair.get("fundamental_matrix"):
            continue
        view_a = str(pair["view_a"])
        view_b = str(pair["view_b"])
        fundamental = np.asarray(pair["fundamental_matrix"], dtype=np.float64).reshape(3, 3)
        lookup[(view_a, view_b)] = fundamental
        lookup[(view_b, view_a)] = fundamental.T
    return lookup


def enrich_observation(
    view_id: str,
    point: dict[str, float],
    extrinsics_by_view: dict[str, Any],
) -> dict[str, Any]:
    """Attach intrinsics/extrinsics and normalized coordinates to one observation."""

    view_meta = extrinsics_by_view[view_id]
    intrinsics = view_meta["intrinsics"]
    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(intrinsics["dist_coeffs"], dtype=np.float64)
    rotation = np.asarray(view_meta["rotation"], dtype=np.float64)
    translation = np.asarray(view_meta["translation"], dtype=np.float64)
    normalized_xy = undistort_point(
        x=float(point["x"]),
        y=float(point["y"]),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        model=str(intrinsics.get("model", "PINHOLE")),
    )
    return {
        "view_id": view_id,
        "pixel_x": float(point["x"]),
        "pixel_y": float(point["y"]),
        "x": float(normalized_xy[0]),
        "y": float(normalized_xy[1]),
        "weight": float(point["confidence"]),
        "confidence": float(point["confidence"]),
        "projection_matrix": np.hstack([rotation, translation.reshape(3, 1)]).tolist(),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "camera_model": str(intrinsics.get("model", "PINHOLE")),
    }


def filter_observations_by_geometry(
    observations: list[dict[str, Any]],
    fundamental_by_pair: dict[tuple[str, str], np.ndarray],
    sampson_threshold: float,
    min_pair_support: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep observations supported by pairwise epipolar consistency."""

    if len(observations) < 2:
        return [], []

    support_counts = [0 for _ in observations]
    pair_checks = []
    for idx_a, obs_a in enumerate(observations):
        for idx_b in range(idx_a + 1, len(observations)):
            obs_b = observations[idx_b]
            fundamental = fundamental_by_pair.get((obs_a["view_id"], obs_b["view_id"]))
            if fundamental is None:
                continue
            err = sampson_error(
                point_a=np.array([obs_a["pixel_x"], obs_a["pixel_y"]], dtype=np.float64),
                point_b=np.array([obs_b["pixel_x"], obs_b["pixel_y"]], dtype=np.float64),
                fundamental_matrix=fundamental,
            )
            is_inlier = bool(np.isfinite(err) and err <= float(sampson_threshold))
            pair_checks.append(
                {
                    "view_a": obs_a["view_id"],
                    "view_b": obs_b["view_id"],
                    "sampson_error": float(err),
                    "is_inlier": is_inlier,
                }
            )
            if is_inlier:
                support_counts[idx_a] += 1
                support_counts[idx_b] += 1

    filtered = [
        {**obs, "pair_support": int(support_counts[idx])}
        for idx, obs in enumerate(observations)
        if support_counts[idx] >= int(min_pair_support)
    ]
    return filtered, pair_checks


def refine_triangulation_by_reprojection(
    observations: list[dict[str, Any]],
    min_views: int,
    max_pixel_reprojection_error: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Iteratively drop the worst-view outlier until reprojection is acceptable."""

    kept = list(observations)
    while len(kept) >= int(min_views):
        tri = triangulate_weighted(kept)
        if tri.get("status") != "ok":
            return tri, kept
        point_3d = np.asarray(tri["point_3d"], dtype=np.float64)
        pixel_errors = pixel_reprojection_errors(point_3d, kept)
        finite_errors = [float(err) for err in pixel_errors if np.isfinite(err)]
        if not finite_errors:
            return tri, kept
        worst_error = max(finite_errors)
        if worst_error <= float(max_pixel_reprojection_error) or len(kept) <= int(min_views):
            tri["pixel_reprojection_errors"] = pixel_errors
            return tri, kept
        worst_index = int(np.nanargmax(np.asarray(pixel_errors, dtype=np.float64)))
        kept.pop(worst_index)

    return {"status": "not_enough_views_after_reprojection_filter", "num_views": len(kept)}, kept


def camera_center_from_observation(observation: dict[str, Any]) -> np.ndarray:
    """Return world-space camera center from an observation."""

    rotation = np.asarray(observation["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(observation["translation"], dtype=np.float64).reshape(3)
    return -rotation.T @ translation


def triangulation_angles_deg(point_3d: np.ndarray, observations: list[dict[str, Any]]) -> list[float]:
    """Return pairwise ray angles for one triangulated point."""

    if len(observations) < 2:
        return []
    point = np.asarray(point_3d, dtype=np.float64).reshape(3)
    rays = []
    for obs in observations:
        center = camera_center_from_observation(obs)
        ray = point - center
        norm = float(np.linalg.norm(ray))
        if norm <= 1e-9:
            continue
        rays.append(ray / norm)
    angles = []
    for ray_a, ray_b in combinations(rays, 2):
        cosine = float(np.clip(np.dot(ray_a, ray_b), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cosine))))
    return angles


def select_best_observation_subset(
    observations: list[dict[str, Any]],
    min_views: int,
    max_pixel_reprojection_error: float,
    min_triangulation_angle_deg: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Try small view subsets and select the best-quality triangulation."""

    best: tuple[float, dict[str, Any], list[dict[str, Any]], dict[str, Any]] | None = None
    num_candidates = 0
    max_views = len(observations)
    for subset_size in range(int(min_views), max_views + 1):
        for indices in combinations(range(max_views), subset_size):
            subset = [observations[idx] for idx in indices]
            tri = triangulate_weighted(subset)
            if tri.get("status") != "ok":
                continue
            point_3d = np.asarray(tri["point_3d"], dtype=np.float64)
            pixel_errors = pixel_reprojection_errors(point_3d, subset)
            finite_errors = [float(err) for err in pixel_errors if np.isfinite(err)]
            if not finite_errors:
                continue
            angles = triangulation_angles_deg(point_3d, subset)
            min_angle = float(np.min(angles)) if angles else 0.0
            mean_angle = float(np.mean(angles)) if angles else 0.0
            mean_error = float(np.mean(finite_errors))
            max_error = float(np.max(finite_errors))
            mean_confidence = float(np.mean([obs["confidence"] for obs in subset]))
            angle_penalty = max(0.0, float(min_triangulation_angle_deg) - min_angle) * 15.0
            reprojection_penalty = max(0.0, max_error - float(max_pixel_reprojection_error)) * 0.5
            dropped_view_penalty = (max_views - subset_size) * 2.0
            confidence_reward = mean_confidence * 1.5
            score = mean_error + angle_penalty + reprojection_penalty + dropped_view_penalty - confidence_reward
            num_candidates += 1
            diagnostics = {
                "num_subset_candidates": num_candidates,
                "subset_size": subset_size,
                "mean_pixel_reprojection_error": mean_error,
                "max_pixel_reprojection_error": max_error,
                "min_triangulation_angle_deg": min_angle,
                "mean_triangulation_angle_deg": mean_angle,
                "mean_confidence": mean_confidence,
                "selected_views": [obs["view_id"] for obs in subset],
                "quality_score": float(score),
            }
            tri["pixel_reprojection_errors"] = pixel_errors
            tri["triangulation_angles_deg"] = angles
            if best is None or score < best[0]:
                best = (score, tri, subset, diagnostics)

    if best is None:
        return {"status": "no_valid_view_subset", "num_views": len(observations)}, [], {
            "num_subset_candidates": num_candidates,
            "status": "no_valid_view_subset",
        }
    _, tri, subset, diagnostics = best
    diagnostics["num_subset_candidates"] = num_candidates
    diagnostics["status"] = "ok"
    return tri, subset, diagnostics


def triangulate_sequence(
    selected_hypothesis: dict[str, Any],
    rough_extrinsics: dict[str, Any],
    track_payloads: dict[str, dict[str, Any]],
    view_ids: list[str],
    min_confidence: float,
    min_views: int,
    frame_step: int,
    geometry_sampson_threshold: float,
    min_pair_support: int,
    max_pixel_reprojection_error: float,
    use_view_subset_selection: bool = False,
    min_triangulation_angle_deg: float = 1.0,
) -> dict[str, Any]:
    """Triangulate all identities across the sequence."""

    frame_indexes = {
        view_id: build_track_frame_index(track_payloads[view_id])
        for view_id in view_ids
    }
    groups = list(selected_hypothesis.get("groups", []))
    extrinsics_by_view = rough_extrinsics.get("extrinsics_by_view", {})
    fundamental_by_pair = build_fundamental_by_pair(selected_hypothesis)
    resolved_views = {view_id for view_id in extrinsics_by_view}
    frame_step = max(int(frame_step), 1)

    all_frames = sorted({
        frame_idx
        for frame_map in frame_indexes.values()
        for frame_idx in frame_map.keys()
    })[::frame_step]

    output_frames = []
    total_triangulated_points = 0
    reprojection_values: list[float] = []
    total_candidate_observations = 0
    total_geometry_kept_observations = 0
    total_final_kept_observations = 0
    total_subset_candidates = 0
    total_subset_selected_observations = 0

    for frame_idx in all_frames:
        identities_out = []
        for group in groups:
            identity_id = int(group["group_id"])
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

            joints_out = []
            for joint_id, observations in sorted(joint_observations.items()):
                total_candidate_observations += len(observations)
                if len(observations) < int(min_views):
                    continue
                geometry_filtered, geometry_pairs = filter_observations_by_geometry(
                    observations=observations,
                    fundamental_by_pair=fundamental_by_pair,
                    sampson_threshold=float(geometry_sampson_threshold),
                    min_pair_support=int(min_pair_support),
                )
                total_geometry_kept_observations += len(geometry_filtered)
                if len(geometry_filtered) < int(min_views):
                    continue
                if use_view_subset_selection:
                    tri, final_observations, subset_diagnostics = select_best_observation_subset(
                        observations=geometry_filtered,
                        min_views=int(min_views),
                        max_pixel_reprojection_error=float(max_pixel_reprojection_error),
                        min_triangulation_angle_deg=float(min_triangulation_angle_deg),
                    )
                    total_subset_candidates += int(subset_diagnostics.get("num_subset_candidates", 0))
                    total_subset_selected_observations += len(final_observations)
                else:
                    tri, final_observations = refine_triangulation_by_reprojection(
                        observations=geometry_filtered,
                        min_views=int(min_views),
                        max_pixel_reprojection_error=float(max_pixel_reprojection_error),
                    )
                    subset_diagnostics = None
                total_final_kept_observations += len(final_observations)
                if tri.get("status") != "ok":
                    continue
                point_3d = np.asarray(tri["point_3d"], dtype=np.float64)
                errors = tri.get("pixel_reprojection_errors") or pixel_reprojection_errors(point_3d, final_observations)
                finite_errors = [float(err) for err in errors if np.isfinite(err)]
                mean_error = float(np.mean(finite_errors)) if finite_errors else None
                if mean_error is not None:
                    reprojection_values.append(mean_error)
                joints_out.append(
                    {
                        "id": joint_id,
                        "x": float(point_3d[0]),
                        "y": float(point_3d[1]),
                        "z": float(point_3d[2]),
                        "num_views": len(final_observations),
                        "candidate_num_views": len(observations),
                        "geometry_filtered_num_views": len(geometry_filtered),
                        "mean_confidence": float(np.mean([obs["confidence"] for obs in final_observations])),
                        "mean_reprojection_error_px": mean_error,
                        "views": [obs["view_id"] for obs in final_observations],
                        "geometry_pair_checks": geometry_pairs,
                        "triangulation_angles_deg": tri.get("triangulation_angles_deg"),
                        "view_subset_quality": subset_diagnostics,
                    }
                )
                total_triangulated_points += 1

            if joints_out:
                identities_out.append(
                    {
                        "identity_id": identity_id,
                        "num_joints": len(joints_out),
                        "joints": joints_out,
                    }
                )

        if identities_out:
            output_frames.append(
                {
                    "frame": frame_idx,
                    "identities": identities_out,
                }
            )

    return {
        "metadata": {
            "stage": "rough_triangulation",
            "anchor_view": rough_extrinsics.get("anchor_view"),
            "translation_scale_note": rough_extrinsics.get("translation_scale_note"),
            "view_ids": list(view_ids),
            "min_confidence": float(min_confidence),
            "min_views": int(min_views),
            "frame_step": int(frame_step),
            "geometry_sampson_threshold": float(geometry_sampson_threshold),
            "min_pair_support": int(min_pair_support),
            "max_pixel_reprojection_error": float(max_pixel_reprojection_error),
            "use_view_subset_selection": bool(use_view_subset_selection),
            "min_triangulation_angle_deg": float(min_triangulation_angle_deg),
        },
        "frames": output_frames,
        "summary": {
            "num_frames_with_3d": len(output_frames),
            "total_triangulated_points": total_triangulated_points,
            "mean_reprojection_error_px": float(np.mean(reprojection_values)) if reprojection_values else None,
            "total_candidate_observations": total_candidate_observations,
            "total_geometry_kept_observations": total_geometry_kept_observations,
            "total_final_kept_observations": total_final_kept_observations,
            "geometry_keep_ratio": (
                float(total_geometry_kept_observations / total_candidate_observations)
                if total_candidate_observations
                else None
            ),
            "final_keep_ratio": (
                float(total_final_kept_observations / total_candidate_observations)
                if total_candidate_observations
                else None
            ),
            "total_subset_candidates": total_subset_candidates,
            "total_subset_selected_observations": total_subset_selected_observations,
        },
    }


def main() -> None:
    """Run rough triangulation."""

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

    triangulated = triangulate_sequence(
        selected_hypothesis=selected_hypothesis,
        rough_extrinsics=rough_extrinsics,
        track_payloads=track_payloads,
        view_ids=list(args.view_ids),
        min_confidence=float(args.min_confidence),
        min_views=int(args.min_views),
        frame_step=int(args.frame_step),
        geometry_sampson_threshold=float(args.geometry_sampson_threshold),
        min_pair_support=int(args.min_pair_support),
        max_pixel_reprojection_error=float(args.max_pixel_reprojection_error),
        use_view_subset_selection=bool(args.use_view_subset_selection),
        min_triangulation_angle_deg=float(args.min_triangulation_angle_deg),
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "triangulated_3d.json").open("w", encoding="utf-8") as handle:
        json.dump(triangulated, handle, ensure_ascii=False, indent=2)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(triangulated["summary"], handle, ensure_ascii=False, indent=2)
    print(json.dumps(triangulated["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
