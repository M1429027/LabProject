"""Derive head-specific noisy camera datasets from clean observations."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .build_amass_yolo_dataset import (
    axis_angle_rotation,
    camera_origin,
    camera_rotation_delta_axis_angle,
    make_stage_a_tokens,
    rotation_matrix_to_axis_angle,
    root_relative,
    triangulate_from_yolo_views,
)
from ..selfcal.relative_pose import build_rough_extrinsics, estimate_relative_pose
from ..selfcal.run_translation_scale_refinement import (
    pair_direction_residuals,
    solve_scaled_centers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["rotation", "center", "mixed_rough"], required=True)
    parser.add_argument("--variants-per-sequence", type=int, default=2)
    parser.add_argument("--rotation-noise-max-deg", type=float, default=12.0)
    parser.add_argument("--center-noise-sigma-m", type=float, default=0.6)
    parser.add_argument("--center-noise-max-m", type=float, default=1.5)
    parser.add_argument("--anchor-index", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--pipeline-ratio", type=float, default=0.35)
    parser.add_argument("--structured-ratio", type=float, default=0.35)
    parser.add_argument("--random-ratio", type=float, default=0.15)
    parser.add_argument("--clean-ratio", type=float, default=0.15)
    parser.add_argument("--structured-rotation-min-deg", type=float, default=0.25)
    parser.add_argument("--structured-rotation-max-deg", type=float, default=5.0)
    parser.add_argument("--structured-center-min-m", type=float, default=0.05)
    parser.add_argument("--structured-center-max-m", type=float, default=0.8)
    parser.add_argument("--essential-ransac-threshold", type=float, default=1e-3)
    parser.add_argument("--pipeline-min-correspondences", type=int, default=24)
    parser.add_argument("--pipeline-frame-keep-min", type=float, default=0.65)
    parser.add_argument("--pipeline-temporal-lr-stabilization", action="store_true", default=True)
    parser.add_argument("--no-pipeline-temporal-lr-stabilization", dest="pipeline_temporal_lr_stabilization", action="store_false")
    parser.add_argument("--pipeline-global-lr-hypotheses", action="store_true", default=True)
    parser.add_argument("--no-pipeline-global-lr-hypotheses", dest="pipeline_global_lr_hypotheses", action="store_false")
    parser.add_argument("--pipeline-max-global-rotation-residual-deg", type=float, default=25.0)
    parser.add_argument("--pipeline-max-global-rotation-max-residual-deg", type=float, default=25.0)
    parser.add_argument("--pipeline-min-valid-pairs", type=int, default=5)
    parser.add_argument("--pipeline-max-center-condition-ratio", type=float, default=0.02)
    parser.add_argument("--pipeline-max-rotation-cycle-error-deg", type=float, default=5.0)
    parser.add_argument("--pipeline-max-direction-residual-deg", type=float, default=5.0)
    parser.add_argument("--pipeline-robust-pair-subset", action="store_true", default=True)
    parser.add_argument("--no-pipeline-robust-pair-subset", dest="pipeline_robust_pair_subset", action="store_false")
    parser.add_argument("--pipeline-subset-min-inlier-pairs", type=int, default=5)
    parser.add_argument("--pipeline-keypoint-source", choices=["yolo", "gt", "detector"], default="yolo")
    parser.add_argument("--pipeline-precompute-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_sequences(sequence_ids: list[str], seed: int) -> dict[str, str]:
    shuffled = list(sorted(sequence_ids))
    np.random.default_rng(seed).shuffle(shuffled)
    count = len(shuffled)
    val_count = max(1, int(round(count * 0.1))) if count >= 3 else 0
    test_count = max(1, int(round(count * 0.1))) if count >= 3 else 0
    train_count = max(1, count - val_count - test_count)
    split_by_sequence = {sequence: "train" for sequence in shuffled[:train_count]}
    split_by_sequence.update({sequence: "val" for sequence in shuffled[train_count : train_count + val_count]})
    split_by_sequence.update({sequence: "test" for sequence in shuffled[train_count + val_count :]})
    return split_by_sequence


def camera_dicts(sample: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": str(sample["views"][index]),
            "K": np.asarray(sample["camera_intrinsics"][index], dtype=np.float64),
            "R": np.asarray(sample["camera_rotations"][index], dtype=np.float64),
            "t": np.asarray(sample["camera_translations"][index], dtype=np.float64),
            "position": np.asarray(sample["camera_origins"][index], dtype=np.float64),
        }
        for index in range(len(sample["views"]))
    ]


def sample_sequence_noise(
    clean_cameras: list[dict[str, Any]],
    mode: str,
    anchor_index: int,
    rotation_noise_max_deg: float,
    center_noise_sigma_m: float,
    center_noise_max_m: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    noisy_cameras: list[dict[str, Any]] = []
    noise_records = []
    for index, clean in enumerate(clean_cameras):
        noisy = dict(clean)
        rotation = np.asarray(clean["R"], dtype=np.float64).copy()
        center = np.asarray(clean["position"], dtype=np.float64).copy()
        angle_deg = 0.0
        center_delta = np.zeros(3, dtype=np.float64)

        if index != anchor_index and mode == "rotation":
            angle_deg = float(rng.triangular(-rotation_noise_max_deg, 0.0, rotation_noise_max_deg))
            axis = rng.normal(0.0, 1.0, size=3)
            world_correction = axis_angle_rotation(axis, math.radians(angle_deg))
            rotation = rotation @ world_correction.T
        elif index != anchor_index and mode == "center":
            center_delta = rng.normal(0.0, center_noise_sigma_m, size=3)
            norm = float(np.linalg.norm(center_delta))
            if norm > center_noise_max_m:
                center_delta *= center_noise_max_m / max(norm, 1e-9)
            center = center + center_delta

        noisy["R"] = rotation
        noisy["position"] = center
        noisy["t"] = -rotation @ center
        noisy_cameras.append(noisy)
        noise_records.append({
            "view": str(clean["id"]),
            "rotation_noise_deg": abs(angle_deg),
            "center_noise_m": float(np.linalg.norm(center_delta)),
        })
    return noisy_cameras, {"views": noise_records}


CV_TO_RENDER = np.diag([1.0, -1.0, -1.0])
BODY_LEFT_RIGHT_PERMUTATION = np.asarray(
    [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], dtype=np.int64
)


def _rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = np.asarray(rotation_a) @ np.asarray(rotation_b).T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def stabilize_temporal_left_right(
    keypoints: np.ndarray,
    confidence_threshold: float,
    switch_penalty: float = 0.15,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve frame-local detector left/right flips using trajectory continuity."""

    output = np.asarray(keypoints, dtype=np.float64).copy()
    frame_count, view_count = output.shape[:2]
    diagnostics: list[dict[str, Any]] = []
    for view in range(view_count):
        body = output[:, view, 5:, :].copy()
        states = np.stack([body, body[:, BODY_LEFT_RIGHT_PERMUTATION]], axis=1)
        costs = np.full((frame_count, 2), np.inf, dtype=np.float64)
        parents = np.zeros((frame_count, 2), dtype=np.int64)
        costs[0] = 0.0
        for frame in range(1, frame_count):
            for current_state in range(2):
                current = states[frame, current_state]
                for previous_state in range(2):
                    previous = states[frame - 1, previous_state]
                    confidence = np.sqrt(np.clip(current[:, 2] * previous[:, 2], 0.0, 1.0))
                    valid = confidence >= confidence_threshold
                    if np.any(valid):
                        combined = np.concatenate(
                            [current[valid, :2], previous[valid, :2]], axis=0
                        )
                        scale = max(float(np.linalg.norm(np.ptp(combined, axis=0))), 1.0)
                        displacement = np.linalg.norm(
                            current[valid, :2] - previous[valid, :2], axis=1
                        )
                        motion_cost = float(
                            np.average(displacement / scale, weights=confidence[valid])
                        )
                    else:
                        motion_cost = 1.0
                    transition_cost = switch_penalty if current_state != previous_state else 0.0
                    candidate = costs[frame - 1, previous_state] + motion_cost + transition_cost
                    if candidate < costs[frame, current_state]:
                        costs[frame, current_state] = candidate
                        parents[frame, current_state] = previous_state

        selected = np.zeros(frame_count, dtype=np.int64)
        selected[-1] = int(np.argmin(costs[-1]))
        for frame in range(frame_count - 1, 0, -1):
            selected[frame - 1] = parents[frame, selected[frame]]
        for frame, state in enumerate(selected):
            output[frame, view, 5:, :] = states[frame, state]
        diagnostics.append({
            "view_index": view,
            "num_flipped_frames": int(np.sum(selected)),
            "num_state_changes": int(np.sum(selected[1:] != selected[:-1])),
        })
    return output, {"views": diagnostics}


def collect_quality_matched_points(
    keypoints: np.ndarray,
    view_a: int,
    view_b: int,
    confidence_threshold: float,
    frame_keep_min: float,
    global_flip_a: bool,
    global_flip_b: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select high-quality frames instead of randomly dropping observations."""

    body_a = np.asarray(keypoints[:, view_a, 5:, :], dtype=np.float64)
    body_b = np.asarray(keypoints[:, view_b, 5:, :], dtype=np.float64)
    if global_flip_a:
        body_a = body_a[:, BODY_LEFT_RIGHT_PERMUTATION]
    if global_flip_b:
        body_b = body_b[:, BODY_LEFT_RIGHT_PERMUTATION]

    records: list[tuple[float, int, np.ndarray]] = []
    for frame in range(len(keypoints)):
        confidence = np.sqrt(
            np.clip(body_a[frame, :, 2] * body_b[frame, :, 2], 0.0, 1.0)
        )
        valid = confidence >= confidence_threshold
        if int(np.sum(valid)) < 5:
            continue
        points = np.concatenate(
            [body_a[frame, valid, :2], body_b[frame, valid, :2]], axis=0
        )
        extent = float(np.linalg.norm(np.ptp(points, axis=0)))
        score = float(np.mean(confidence[valid]) * np.sqrt(max(extent, 1.0)))
        records.append((score, frame, valid))

    if not records:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty.copy(), {
            "num_candidate_frames": 0,
            "num_selected_frames": 0,
        }
    keep_ratio = float(rng.uniform(frame_keep_min, 1.0))
    keep_count = max(1, int(math.ceil(len(records) * keep_ratio)))
    selected = sorted(records, key=lambda record: record[0], reverse=True)[:keep_count]
    points_a = np.concatenate(
        [body_a[frame, valid, :2] for _, frame, valid in selected]
    )
    points_b = np.concatenate(
        [body_b[frame, valid, :2] for _, frame, valid in selected]
    )
    return points_a, points_b, {
        "num_candidate_frames": len(records),
        "num_selected_frames": len(selected),
        "mean_selected_frame_score": float(
            np.mean([record[0] for record in selected])
        ),
    }


def globally_consistent_rotations(
    pair_results: list[dict[str, Any]],
    view_ids: list[str],
    anchor_view: str,
) -> dict[str, Any] | None:
    """Select a robust spanning-tree rotation hypothesis for the camera graph."""

    valid = [pair for pair in pair_results if pair.get("status") == "ok"]
    if len(valid) < len(view_ids) - 1:
        return None

    best: dict[str, Any] | None = None
    for tree_edges in combinations(valid, len(view_ids) - 1):
        adjacency: dict[str, list[tuple[str, np.ndarray]]] = {
            view: [] for view in view_ids
        }
        for pair in tree_edges:
            view_a, view_b = str(pair["view_a"]), str(pair["view_b"])
            rotation = np.asarray(pair["rotation"], dtype=np.float64)
            adjacency[view_a].append((view_b, rotation))
            adjacency[view_b].append((view_a, rotation.T))
        rotations = {anchor_view: np.eye(3, dtype=np.float64)}
        queue = [anchor_view]
        while queue:
            source = queue.pop(0)
            for target, relative in adjacency[source]:
                if target in rotations:
                    continue
                rotations[target] = relative @ rotations[source]
                queue.append(target)
        if len(rotations) != len(view_ids):
            continue

        residuals = []
        weights = []
        for pair in valid:
            observed = np.asarray(pair["rotation"], dtype=np.float64)
            predicted = (
                rotations[str(pair["view_b"])]
                @ rotations[str(pair["view_a"])].T
            )
            residuals.append(_rotation_error_deg(observed, predicted))
            weights.append(max(float(pair.get("pose_inlier_ratio", 0.0)), 0.05))
        residuals_array = np.asarray(residuals, dtype=np.float64)
        robust_score = float(
            np.average(
                np.minimum(residuals_array, 45.0),
                weights=np.asarray(weights, dtype=np.float64),
            )
        )
        candidate = {
            "rotations": rotations,
            "residuals_deg": residuals,
            "robust_score_deg": robust_score,
            "median_residual_deg": float(np.median(residuals_array)),
            "max_residual_deg": float(np.max(residuals_array)),
        }
        if best is None or candidate["robust_score_deg"] < best["robust_score_deg"]:
            best = candidate
    return best

def estimate_global_pair_results(
    clean_cameras: list[dict[str, Any]],
    sequence_entries: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve detector label ambiguity before projecting rotations to one graph."""

    keypoint_frames = []
    for entry in sequence_entries:
        sample = np.load(entry["path"], allow_pickle=False)
        keypoint_frames.append(
            np.asarray(
                sample[f"{args.pipeline_keypoint_source}_keypoints_2d"],
                dtype=np.float64,
            )
        )
    keypoints = np.stack(keypoint_frames)
    if args.pipeline_temporal_lr_stabilization:
        keypoints, temporal_diagnostic = stabilize_temporal_left_right(
            keypoints, args.confidence_threshold
        )
    else:
        temporal_diagnostic = {"views": []}

    view_count = len(clean_cameras)
    assignment_count = 2 ** (view_count - 1) if args.pipeline_global_lr_hypotheses else 1
    selection_seed = int(rng.integers(0, 2**31 - 1))
    candidates = []
    for assignment_index in range(assignment_count):
        flips = [False]
        flips.extend(
            bool((assignment_index >> (view - 1)) & 1)
            for view in range(1, view_count)
        )
        pair_results = []
        for pair_index, (index_a, index_b) in enumerate(
            combinations(range(view_count), 2)
        ):
            points_a, points_b, point_diagnostic = collect_quality_matched_points(
                keypoints,
                index_a,
                index_b,
                args.confidence_threshold,
                args.pipeline_frame_keep_min,
                flips[index_a],
                flips[index_b],
                np.random.default_rng(selection_seed + pair_index * 7919),
            )
            if len(points_a) < args.pipeline_min_correspondences:
                result = {
                    "status": "not_enough_points",
                    "num_points": int(len(points_a)),
                }
            else:
                result = estimate_relative_pose(
                    points_a,
                    points_b,
                    clean_cameras[index_a]["K"],
                    clean_cameras[index_b]["K"],
                    ransac_threshold=args.essential_ransac_threshold,
                )
            pair_results.append({
                "view_a": str(clean_cameras[index_a]["id"]),
                "view_b": str(clean_cameras[index_b]["id"]),
                "num_correspondences": int(len(points_a)),
                "point_selection": point_diagnostic,
                **result,
            })
        global_solution = globally_consistent_rotations(
            pair_results,
            [str(camera["id"]) for camera in clean_cameras],
            str(clean_cameras[args.anchor_index]["id"]),
        )
        valid_pair_count = sum(pair.get("status") == "ok" for pair in pair_results)
        score = (
            float(global_solution["robust_score_deg"])
            if global_solution is not None
            else float("inf")
        )
        candidates.append({
            "assignment_index": assignment_index,
            "flips": flips,
            "pair_results": pair_results,
            "global_solution": global_solution,
            "num_valid_pairs": valid_pair_count,
            "score": score + 45.0 * (len(clean_cameras) - 1 - min(valid_pair_count, len(clean_cameras) - 1)),
        })

    selected = min(candidates, key=lambda candidate: candidate["score"])
    solution = selected["global_solution"]
    pair_results = selected["pair_results"]
    if solution is not None:
        for pair in pair_results:
            if pair.get("status") != "ok":
                continue
            pair["observed_rotation"] = pair["rotation"]
            rotation = (
                solution["rotations"][str(pair["view_b"])]
                @ solution["rotations"][str(pair["view_a"])].T
            )
            pair["rotation"] = rotation.tolist()

    return pair_results, {
        "temporal_left_right": temporal_diagnostic,
        "num_lr_hypotheses": assignment_count,
        "selected_lr_flips": selected["flips"],
        "global_rotation_robust_score_deg": (
            None if solution is None else solution["robust_score_deg"]
        ),
        "global_rotation_median_residual_deg": (
            None if solution is None else solution["median_residual_deg"]
        ),
        "global_rotation_max_residual_deg": (
            None if solution is None else solution["max_residual_deg"]
        ),
    }


def collect_matched_points(
    sequence_entries: list[dict[str, Any]],
    view_a: int,
    view_b: int,
    confidence_threshold: float,
    frame_keep_min: float,
    keypoint_source: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect matched single-person body joints after synthetic ID matching."""

    keep_probability = float(rng.uniform(frame_keep_min, 1.0))
    points_a: list[np.ndarray] = []
    points_b: list[np.ndarray] = []
    for entry in sequence_entries:
        if rng.random() > keep_probability:
            continue
        sample = np.load(entry["path"], allow_pickle=False)
        keypoints = np.asarray(sample[f"{keypoint_source}_keypoints_2d"], dtype=np.float64)
        valid = (
            (keypoints[view_a, 5:, 2] >= confidence_threshold)
            & (keypoints[view_b, 5:, 2] >= confidence_threshold)
        )
        if np.any(valid):
            points_a.append(keypoints[view_a, 5:, :2][valid])
            points_b.append(keypoints[view_b, 5:, :2][valid])
    if not points_a:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty.copy()
    return np.concatenate(points_a), np.concatenate(points_b)



def pair_label(pair: dict[str, Any]) -> str:
    return f"{pair['view_a']}->{pair['view_b']}"


def pair_graph_connected(
    pairs: list[dict[str, Any]],
    view_ids: list[str],
) -> bool:
    adjacency = {view_id: set() for view_id in view_ids}
    for pair in pairs:
        view_a = str(pair["view_a"])
        view_b = str(pair["view_b"])
        adjacency[view_a].add(view_b)
        adjacency[view_b].add(view_a)
    visited = {view_ids[0]}
    pending = [view_ids[0]]
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return len(visited) == len(view_ids)


def robust_solve_camera_centers(
    relative_results: dict[str, Any],
    rough: dict[str, Any],
    view_ids: list[str],
    anchor_view: str,
    max_condition_ratio: float,
    max_direction_residual_deg: float,
    min_inlier_pairs: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """RANSAC-style center solve using connected four-pair hypotheses."""

    valid_pairs = [
        pair
        for pair in relative_results["pair_relative_poses"]
        if pair.get("status") == "ok"
    ]
    hypotheses = []
    seen_inlier_sets: set[tuple[str, ...]] = set()
    for subset in combinations(valid_pairs, min(4, len(valid_pairs))):
        subset_pairs = list(subset)
        if not pair_graph_connected(subset_pairs, view_ids):
            continue
        subset_results = {
            "pair_relative_poses": subset_pairs,
            "num_valid_pairs": len(subset_pairs),
        }
        try:
            centers, _ = solve_scaled_centers(
                subset_results,
                rough,
                view_ids,
                anchor_view,
                "unit_median_distance",
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        residuals = pair_direction_residuals(centers, relative_results, rough)
        inlier_labels = tuple(sorted(
            f"{item['view_a']}->{item['view_b']}"
            for item in residuals
            if item.get("angle_error_deg") is not None
            and float(item["angle_error_deg"]) <= max_direction_residual_deg
        ))
        if len(inlier_labels) < min_inlier_pairs or inlier_labels in seen_inlier_sets:
            continue
        seen_inlier_sets.add(inlier_labels)
        inlier_pairs = [
            pair for pair in valid_pairs if pair_label(pair) in inlier_labels
        ]
        if not pair_graph_connected(inlier_pairs, view_ids):
            continue
        inlier_results = {
            "pair_relative_poses": inlier_pairs,
            "num_valid_pairs": len(inlier_pairs),
        }
        try:
            refined_centers, solve_diagnostic = solve_scaled_centers(
                inlier_results,
                rough,
                view_ids,
                anchor_view,
                "unit_median_distance",
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        condition_ratio = solve_diagnostic.get(
            "condition_ratio_smallest_over_largest"
        )
        if condition_ratio is None or condition_ratio > max_condition_ratio:
            continue
        refined_residuals = pair_direction_residuals(
            refined_centers, relative_results, rough
        )
        refined_inliers = [
            item for item in refined_residuals
            if item.get("angle_error_deg") is not None
            and float(item["angle_error_deg"]) <= max_direction_residual_deg
        ]
        if len(refined_inliers) < min_inlier_pairs:
            continue
        angles = [float(item["angle_error_deg"]) for item in refined_inliers]
        hypotheses.append({
            "centers": refined_centers,
            "solve_diagnostic": solve_diagnostic,
            "selected_pairs": [pair_label(pair) for pair in inlier_pairs],
            "dropped_pairs": [
                pair_label(pair)
                for pair in valid_pairs
                if pair_label(pair) not in {pair_label(item) for item in inlier_pairs}
            ],
            "num_inlier_pairs": len(refined_inliers),
            "median_direction_residual_deg": float(np.median(angles)),
            "max_direction_residual_deg": float(np.max(angles)),
            "all_pair_direction_residuals": refined_residuals,
        })

    if not hypotheses:
        return None, {
            "mode": "robust_connected_subset",
            "status": "no_consensus",
            "num_valid_pairs": len(valid_pairs),
            "num_connected_hypotheses": sum(
                pair_graph_connected(list(subset), view_ids)
                for subset in combinations(valid_pairs, min(4, len(valid_pairs)))
            ),
            "min_required_inlier_pairs": min_inlier_pairs,
        }

    selected = min(
        hypotheses,
        key=lambda item: (
            -int(item["num_inlier_pairs"]),
            float(item["median_direction_residual_deg"]),
            float(item["max_direction_residual_deg"]),
            float(item["solve_diagnostic"]["condition_ratio_smallest_over_largest"]),
        ),
    )
    diagnostic = {
        key: value for key, value in selected.items() if key != "centers"
    }
    diagnostic.update({
        "mode": "robust_connected_subset",
        "status": "ok",
        "num_candidates": len(hypotheses),
    })
    return selected["centers"], diagnostic


def estimate_pipeline_rough_cameras(
    clean_cameras: list[dict[str, Any]],
    sequence_entries: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Run the Stage 4A essential-matrix baseline and align its unavoidable gauge."""

    view_ids = [str(camera["id"]) for camera in clean_cameras]
    intrinsics_by_view = {
        str(camera["id"]): {
            "source_path": "amass_clean_cache",
            "model": "PINHOLE",
            "camera_matrix": np.asarray(camera["K"], dtype=np.float64),
            "dist_coeffs": np.zeros(5, dtype=np.float64),
            "image_size": None,
            "rms": None,
        }
        for camera in clean_cameras
    }
    pair_results = []
    for index_a, index_b in combinations(range(len(clean_cameras)), 2):
        points_a, points_b = collect_matched_points(
            sequence_entries,
            index_a,
            index_b,
            args.confidence_threshold,
            args.pipeline_frame_keep_min,
            args.pipeline_keypoint_source,
            rng,
        )
        if len(points_a) < args.pipeline_min_correspondences:
            result = {"status": "not_enough_points", "num_points": int(len(points_a))}
        else:
            result = estimate_relative_pose(
                points_a,
                points_b,
                clean_cameras[index_a]["K"],
                clean_cameras[index_b]["K"],
                ransac_threshold=args.essential_ransac_threshold,
            )
        pair_results.append({
            "view_a": view_ids[index_a],
            "view_b": view_ids[index_b],
            "num_correspondences": int(len(points_a)),
            **result,
        })

    frontend_diagnostic: dict[str, Any] = {"mode": "legacy_pairwise"}
    if (
        args.pipeline_temporal_lr_stabilization
        or args.pipeline_global_lr_hypotheses
    ):
        pair_results, frontend_diagnostic = estimate_global_pair_results(
            clean_cameras, sequence_entries, args, rng
        )
        frontend_diagnostic["mode"] = "quality_temporal_global"
        global_residual = frontend_diagnostic.get(
            "global_rotation_robust_score_deg"
        )
        global_max_residual = frontend_diagnostic.get(
            "global_rotation_max_residual_deg"
        )
        if (
            global_residual is None
            or global_residual
            > args.pipeline_max_global_rotation_residual_deg
            or global_max_residual is None
            or global_max_residual
            > args.pipeline_max_global_rotation_max_residual_deg
        ):
            return None, {
                "status": "global_rotation_inconsistent",
                "num_valid_pairs": sum(
                    pair.get("status") == "ok" for pair in pair_results
                ),
                "global_rotation_robust_score_deg": global_residual,
                "global_rotation_max_residual_deg": global_max_residual,
                "max_allowed_global_rotation_residual_deg": (
                    args.pipeline_max_global_rotation_residual_deg
                ),
                "max_allowed_global_rotation_max_residual_deg": (
                    args.pipeline_max_global_rotation_max_residual_deg
                ),
                "rough_frontend": frontend_diagnostic,
            }

    relative_results = {
        "pair_relative_poses": pair_results,
        "num_valid_pairs": sum(pair.get("status") == "ok" for pair in pair_results),
    }
    rough = build_rough_extrinsics(
        relative_results,
        intrinsics_by_view,
        view_ids,
        anchor_view=view_ids[args.anchor_index],
    )
    if rough["unresolved_views"]:
        return None, {
            "status": "unresolved_views",
            "unresolved_views": rough["unresolved_views"],
            "num_valid_pairs": relative_results["num_valid_pairs"],
        }
    if relative_results["num_valid_pairs"] < args.pipeline_min_valid_pairs:
        return None, {
            "status": "not_enough_valid_pairs",
            "num_valid_pairs": relative_results["num_valid_pairs"],
        }
    rotation_cycle_errors = []
    rough_views = rough["extrinsics_by_view"]
    for pair in pair_results:
        if pair.get("status") != "ok":
            continue
        rotation_a = np.asarray(rough_views[str(pair["view_a"])]["rotation"], dtype=np.float64)
        rotation_b = np.asarray(rough_views[str(pair["view_b"])]["rotation"], dtype=np.float64)
        graph_relative = rotation_b @ rotation_a.T
        observed_relative = np.asarray(pair["rotation"], dtype=np.float64)
        cosine = float(np.clip((np.trace(observed_relative.T @ graph_relative) - 1.0) * 0.5, -1.0, 1.0))
        rotation_cycle_errors.append(float(np.degrees(np.arccos(cosine))))
    max_rotation_cycle_error = max(rotation_cycle_errors, default=float("inf"))
    if max_rotation_cycle_error > args.pipeline_max_rotation_cycle_error_deg:
        return None, {
            "status": "rotation_cycle_inconsistent",
            "num_valid_pairs": relative_results["num_valid_pairs"],
            "max_rotation_cycle_error_deg": max_rotation_cycle_error,
            "max_allowed_rotation_cycle_error_deg": args.pipeline_max_rotation_cycle_error_deg,
        }

    pair_subset_diagnostic: dict[str, Any] = {"mode": "all_pairs"}
    if args.pipeline_robust_pair_subset:
        solved_centers_cv, pair_subset_diagnostic = robust_solve_camera_centers(
            relative_results,
            rough,
            view_ids,
            view_ids[args.anchor_index],
            args.pipeline_max_center_condition_ratio,
            args.pipeline_max_direction_residual_deg,
            args.pipeline_subset_min_inlier_pairs,
        )
        if solved_centers_cv is None:
            return None, {
                "status": "robust_pair_subset_failed",
                "num_valid_pairs": relative_results["num_valid_pairs"],
                "pair_subset_selection": pair_subset_diagnostic,
            }
        center_solve_diagnostic = pair_subset_diagnostic["solve_diagnostic"]
    else:
        try:
            solved_centers_cv, center_solve_diagnostic = solve_scaled_centers(
                relative_results,
                rough,
                view_ids,
                view_ids[args.anchor_index],
                "unit_median_distance",
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            return None, {"status": "center_scale_solve_failed", "error": str(error)}

        condition_ratio = center_solve_diagnostic.get(
            "condition_ratio_smallest_over_largest"
        )
        if (
            condition_ratio is None
            or condition_ratio > args.pipeline_max_center_condition_ratio
        ):
            return None, {
                "status": "ill_conditioned_center_graph",
                "num_valid_pairs": relative_results["num_valid_pairs"],
                "condition_ratio_smallest_over_largest": condition_ratio,
                "max_allowed_condition_ratio": (
                    args.pipeline_max_center_condition_ratio
                ),
            }

        direction_residuals = pair_direction_residuals(
            solved_centers_cv, relative_results, rough
        )
        direction_angles = [
            float(item["angle_error_deg"])
            for item in direction_residuals
            if item.get("angle_error_deg") is not None
        ]
        max_direction_error = max(direction_angles, default=float("inf"))
        if max_direction_error > args.pipeline_max_direction_residual_deg:
            return None, {
                "status": "baseline_direction_inconsistent",
                "num_valid_pairs": relative_results["num_valid_pairs"],
                "max_direction_residual_deg": max_direction_error,
                "max_allowed_direction_residual_deg": (
                    args.pipeline_max_direction_residual_deg
                ),
            }

    anchor = clean_cameras[args.anchor_index]
    anchor_rotation = np.asarray(anchor["R"], dtype=np.float64)
    anchor_center = np.asarray(anchor["position"], dtype=np.float64)
    rough_relative_rotations = []
    rough_relative_centers = []
    clean_relative_centers = []
    for camera in clean_cameras:
        view_id = str(camera["id"])
        rough_entry = rough["extrinsics_by_view"][view_id]
        rotation_cv = np.asarray(rough_entry["rotation"], dtype=np.float64)
        rotation_render = CV_TO_RENDER @ rotation_cv @ CV_TO_RENDER
        center_render = CV_TO_RENDER @ np.asarray(solved_centers_cv[view_id], dtype=np.float64)
        rough_relative_rotations.append(rotation_render)
        rough_relative_centers.append(center_render)
        clean_relative_centers.append(
            anchor_rotation @ (np.asarray(camera["position"], dtype=np.float64) - anchor_center)
        )

    rough_relative_centers = np.asarray(rough_relative_centers)
    clean_relative_centers = np.asarray(clean_relative_centers)
    active = np.arange(len(clean_cameras)) != args.anchor_index
    numerator = float(np.sum(rough_relative_centers[active] * clean_relative_centers[active]))
    if numerator < 0.0:
        rough_relative_centers *= -1.0
        numerator *= -1.0
    denominator = float(np.sum(rough_relative_centers[active] ** 2))
    if denominator <= 1e-9:
        return None, {"status": "degenerate_translation_scale"}
    translation_scale = numerator / denominator

    noisy_cameras = []
    for index, clean in enumerate(clean_cameras):
        if index == args.anchor_index:
            noisy_cameras.append(dict(clean))
            continue
        rotation = rough_relative_rotations[index] @ anchor_rotation
        center_relative = translation_scale * rough_relative_centers[index]
        center = anchor_center + anchor_rotation.T @ center_relative
        noisy = dict(clean)
        noisy["R"] = rotation
        noisy["position"] = center
        noisy["t"] = -rotation @ center
        noisy_cameras.append(noisy)

    gt_rotation_errors_deg = [
        float(
            np.linalg.norm(camera_rotation_delta_axis_angle(clean, noisy))
            * 180.0
            / math.pi
        )
        for clean, noisy in zip(clean_cameras, noisy_cameras)
    ]
    gt_center_errors_m = [
        float(
            np.linalg.norm(
                np.asarray(noisy["position"], dtype=np.float64)
                - np.asarray(clean["position"], dtype=np.float64)
            )
        )
        for clean, noisy in zip(clean_cameras, noisy_cameras)
    ]

    return noisy_cameras, {
        "status": "ok",
        "num_valid_pairs": relative_results["num_valid_pairs"],
        "translation_scale_alignment": float(translation_scale),
        "gt_rotation_errors_deg": gt_rotation_errors_deg,
        "gt_rotation_error_mean_deg": float(np.mean(gt_rotation_errors_deg)),
        "gt_rotation_error_max_deg": float(np.max(gt_rotation_errors_deg)),
        "gt_center_errors_m": gt_center_errors_m,
        "gt_center_error_mean_m": float(np.mean(gt_center_errors_m)),
        "gt_center_error_max_m": float(np.max(gt_center_errors_m)),
        "mean_pose_inlier_ratio": float(np.mean([
            pair.get("pose_inlier_ratio", 0.0)
            for pair in pair_results
            if pair.get("status") == "ok"
        ])) if relative_results["num_valid_pairs"] else 0.0,
        "rough_frontend": frontend_diagnostic,
        "center_scale_solve": center_solve_diagnostic,
        "pair_subset_selection": pair_subset_diagnostic,
    }


def make_error_template(
    clean_cameras: list[dict[str, Any]],
    noisy_cameras: list[dict[str, Any]],
    anchor_index: int,
) -> dict[str, np.ndarray]:
    anchor_rotation = np.asarray(clean_cameras[anchor_index]["R"], dtype=np.float64)
    anchor_center = np.asarray(clean_cameras[anchor_index]["position"], dtype=np.float64)
    clean_centers = np.stack([np.asarray(camera["position"]) for camera in clean_cameras])
    rig_scale = max(float(np.linalg.norm(clean_centers - anchor_center, axis=1).max()), 1.0)
    rotation_errors = []
    center_errors = []
    for clean, noisy in zip(clean_cameras, noisy_cameras):
        clean_relative_rotation = np.asarray(clean["R"]) @ anchor_rotation.T
        noisy_relative_rotation = np.asarray(noisy["R"]) @ anchor_rotation.T
        rotation_errors.append(noisy_relative_rotation @ clean_relative_rotation.T)
        clean_relative_center = anchor_rotation @ (np.asarray(clean["position"]) - anchor_center)
        noisy_relative_center = anchor_rotation @ (np.asarray(noisy["position"]) - anchor_center)
        center_errors.append((noisy_relative_center - clean_relative_center) / rig_scale)
    return {
        "rotation_errors": np.asarray(rotation_errors),
        "center_errors": np.asarray(center_errors),
    }


def apply_error_template(
    clean_cameras: list[dict[str, Any]],
    template: dict[str, np.ndarray],
    anchor_index: int,
) -> list[dict[str, Any]]:
    anchor_rotation = np.asarray(clean_cameras[anchor_index]["R"], dtype=np.float64)
    anchor_center = np.asarray(clean_cameras[anchor_index]["position"], dtype=np.float64)
    clean_centers = np.stack([np.asarray(camera["position"]) for camera in clean_cameras])
    rig_scale = max(float(np.linalg.norm(clean_centers - anchor_center, axis=1).max()), 1.0)
    output = []
    for index, clean in enumerate(clean_cameras):
        if index == anchor_index:
            output.append(dict(clean))
            continue
        clean_relative_rotation = np.asarray(clean["R"]) @ anchor_rotation.T
        noisy_relative_rotation = template["rotation_errors"][index] @ clean_relative_rotation
        clean_relative_center = anchor_rotation @ (np.asarray(clean["position"]) - anchor_center)
        noisy_relative_center = clean_relative_center + template["center_errors"][index] * rig_scale
        rotation = noisy_relative_rotation @ anchor_rotation
        center = anchor_center + anchor_rotation.T @ noisy_relative_center
        noisy = dict(clean)
        noisy["R"] = rotation
        noisy["position"] = center
        noisy["t"] = -rotation @ center
        output.append(noisy)
    return output


def apply_scaled_error_template(
    clean_cameras: list[dict[str, Any]],
    template: dict[str, np.ndarray],
    anchor_index: int,
    rotation_range_deg: tuple[float, float],
    center_range_m: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Rescale a correlated pipeline error pattern into a moderate-error case."""

    camera_count = len(clean_cameras)
    active = np.arange(camera_count) != anchor_index
    clean_centers = np.stack([np.asarray(camera["position"]) for camera in clean_cameras])
    anchor_center = np.asarray(clean_cameras[anchor_index]["position"])
    rig_scale = max(float(np.linalg.norm(clean_centers - anchor_center, axis=1).max()), 1.0)

    rotation_vectors = np.stack([
        rotation_matrix_to_axis_angle(rotation)
        for rotation in np.asarray(template["rotation_errors"])
    ]).astype(np.float64)
    rotation_vectors[anchor_index] = 0.0
    rotation_norms = np.linalg.norm(rotation_vectors[active], axis=-1)
    if not len(rotation_norms) or float(rotation_norms.max()) < 1e-8:
        rotation_vectors = rng.normal(size=(camera_count, 3))
        rotation_vectors[anchor_index] = 0.0
        rotation_norms = np.linalg.norm(rotation_vectors[active], axis=-1)
    target_rotation_deg = float(rng.uniform(*rotation_range_deg))
    rotation_scale = math.radians(target_rotation_deg) / max(
        float(rotation_norms.max()), 1e-8
    )
    rotation_vectors *= rotation_scale

    center_offsets_m = np.asarray(template["center_errors"], dtype=np.float64) * rig_scale
    center_offsets_m[anchor_index] = 0.0
    center_norms = np.linalg.norm(center_offsets_m[active], axis=-1)
    if not len(center_norms) or float(center_norms.max()) < 1e-8:
        center_offsets_m = rng.normal(size=(camera_count, 3))
        center_offsets_m[anchor_index] = 0.0
        center_norms = np.linalg.norm(center_offsets_m[active], axis=-1)
    target_center_m = float(rng.uniform(*center_range_m))
    center_scale = target_center_m / max(float(center_norms.max()), 1e-8)
    center_offsets_m *= center_scale

    scaled_template = {
        "rotation_errors": np.stack([
            axis_angle_rotation(vector, float(np.linalg.norm(vector)))
            for vector in rotation_vectors
        ]),
        "center_errors": center_offsets_m / rig_scale,
    }
    noisy = apply_error_template(clean_cameras, scaled_template, anchor_index)
    return noisy, {
        "structured_target_rotation_max_deg": target_rotation_deg,
        "structured_target_center_max_m": target_center_m,
    }


def sample_random_mixed_noise(
    clean_cameras: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    output = []
    for index, clean in enumerate(clean_cameras):
        if index == args.anchor_index:
            output.append(dict(clean))
            continue
        angle = math.radians(float(rng.triangular(
            -args.rotation_noise_max_deg, 0.0, args.rotation_noise_max_deg
        )))
        axis = rng.normal(size=3)
        rotation = np.asarray(clean["R"]) @ axis_angle_rotation(axis, angle).T
        center_delta = rng.normal(0.0, args.center_noise_sigma_m, size=3)
        norm = float(np.linalg.norm(center_delta))
        if norm > args.center_noise_max_m:
            center_delta *= args.center_noise_max_m / max(norm, 1e-9)
        center = np.asarray(clean["position"]) + center_delta
        noisy = dict(clean)
        noisy["R"] = rotation
        noisy["position"] = center
        noisy["t"] = -rotation @ center
        output.append(noisy)
    return output
def prepare_mixed_noise(
    entries_by_sequence: dict[str, list[dict[str, Any]]],
    split_by_sequence: dict[str, str],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    ratios = np.asarray(
        [args.pipeline_ratio, args.structured_ratio, args.random_ratio, args.clean_ratio],
        dtype=np.float64,
    )
    if np.any(ratios < 0.0) or not np.isclose(float(ratios.sum()), 1.0, atol=1e-6):
        raise ValueError("pipeline/structured/random/clean ratios must be non-negative and sum to 1.")

    structured_ranges = (
        (args.structured_rotation_min_deg, args.structured_rotation_max_deg),
        (args.structured_center_min_m, args.structured_center_max_m),
    )
    for minimum, maximum in structured_ranges:
        if minimum < 0.0 or maximum <= minimum:
            raise ValueError("structured noise ranges require 0 <= minimum < maximum.")

    keys_by_split: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    clean_by_sequence: dict[str, list[dict[str, Any]]] = {}
    pipeline_cameras: dict[tuple[str, int], list[dict[str, Any]]] = {}
    pipeline_diagnostics: dict[tuple[str, int], dict[str, Any]] = {}
    templates_by_split: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "val": [], "test": []}

    for sequence_index, sequence_id in enumerate(sorted(entries_by_sequence)):
        sequence_entries = sorted(
            entries_by_sequence[sequence_id], key=lambda item: int(item["frame_id"])
        )
        clean_cameras = camera_dicts(np.load(sequence_entries[0]["path"], allow_pickle=False))
        clean_by_sequence[sequence_id] = clean_cameras
        split = split_by_sequence[sequence_id]
        # A fixed-camera sequence has one observed pipeline estimate. Re-running
        # the stochastic front end for every synthetic variant is both expensive
        # and inconsistent with deployment, where extrinsics stay static.
        pipeline_seed = int(args.seed + sequence_index * 1009)
        noisy, diagnostic = estimate_pipeline_rough_cameras(
            clean_cameras,
            sequence_entries,
            args,
            np.random.default_rng(pipeline_seed),
        )
        if noisy is not None:
            templates_by_split[split].append(
                make_error_template(clean_cameras, noisy, args.anchor_index)
            )
        for variant in range(args.variants_per_sequence):
            key = (sequence_id, variant)
            keys_by_split[split].append(key)
            pipeline_diagnostics[key] = diagnostic
            if noisy is not None:
                pipeline_cameras[key] = noisy

    status_counts: dict[str, int] = {}
    for diagnostic in pipeline_diagnostics.values():
        status = str(diagnostic.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    print(json.dumps({
        "pipeline_rough_precompute": {
            "status_counts": status_counts,
            "successful_by_split": {
                split: sum(key in pipeline_cameras for key in keys)
                for split, keys in keys_by_split.items()
            },
            "total_by_split": {split: len(keys) for split, keys in keys_by_split.items()},
            "failed_examples": [
                {"sequence": key[0], "variant": key[1], **diagnostic}
                for key, diagnostic in pipeline_diagnostics.items()
                if diagnostic.get("status") != "ok"
            ][:12],
        }
    }, indent=2))

    if args.pipeline_precompute_only:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "pipeline_precompute_report.json"
        report = {
            "metadata": {
                "cache_manifest": args.cache_manifest,
                "pipeline_keypoint_source": args.pipeline_keypoint_source,
                "num_cases": len(pipeline_diagnostics),
                "status_counts": status_counts,
            },
            "cases": [
                {
                    "sequence": key[0],
                    "variant": key[1],
                    "split": split_by_sequence[key[0]],
                    **diagnostic,
                }
                for key, diagnostic in pipeline_diagnostics.items()
            ],
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"pipeline_precompute_report": str(report_path)}, indent=2))
        raise SystemExit(0)

    source_by_key: dict[tuple[str, int], str] = {}
    for split, keys in keys_by_split.items():
        rng = np.random.default_rng(args.seed + {"train": 101, "val": 202, "test": 303}[split])
        successful = [key for key in keys if key in pipeline_cameras]
        rng.shuffle(successful)
        pipeline_count = int(round(len(keys) * args.pipeline_ratio))
        structured_count = int(round(len(keys) * args.structured_ratio))
        clean_count = int(round(len(keys) * args.clean_ratio))
        # Do not manufacture invalid pipeline rough extrinsics just to satisfy
        # a ratio. Fall back to independent noise when geometry rejects a case.
        pipeline_count = min(pipeline_count, len(successful))
        if structured_count and not templates_by_split[split]:
            structured_count = 0

        pipeline_keys = set(successful[:pipeline_count])
        remaining = [key for key in keys if key not in pipeline_keys]
        rng.shuffle(remaining)
        clean_keys = set(remaining[:clean_count])
        remaining = [key for key in remaining if key not in clean_keys]
        structured_keys = set(remaining[:structured_count])
        for key in keys:
            if key in pipeline_keys:
                source_by_key[key] = "pipeline_rough"
            elif key in clean_keys:
                source_by_key[key] = "clean_oracle"
            elif key in structured_keys:
                source_by_key[key] = "empirical_structured"
            else:
                source_by_key[key] = "independent_random"

    noisy_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    metadata: list[dict[str, Any]] = []
    source_counts = {source: 0 for source in (
        "pipeline_rough", "clean_oracle", "empirical_structured", "independent_random"
    )}
    for sequence_index, sequence_id in enumerate(sorted(entries_by_sequence)):
        clean_cameras = clean_by_sequence[sequence_id]
        split = split_by_sequence[sequence_id]
        for variant in range(args.variants_per_sequence):
            key = (sequence_id, variant)
            variant_seed = int(args.seed + sequence_index * 1009 + variant * 9176)
            rng = np.random.default_rng(variant_seed + 500_000)
            source = source_by_key[key]
            source_metadata: dict[str, float] = {}
            if source == "pipeline_rough":
                noisy = pipeline_cameras[key]
            elif source == "clean_oracle":
                noisy = [dict(camera) for camera in clean_cameras]
            elif source == "empirical_structured":
                template_index = int(rng.integers(0, len(templates_by_split[split])))
                noisy, source_metadata = apply_scaled_error_template(
                    clean_cameras,
                    templates_by_split[split][template_index],
                    args.anchor_index,
                    (
                        args.structured_rotation_min_deg,
                        args.structured_rotation_max_deg,
                    ),
                    (args.structured_center_min_m, args.structured_center_max_m),
                    rng,
                )
            else:
                noisy = sample_random_mixed_noise(clean_cameras, args, rng)
            noisy_by_key[key] = noisy
            source_counts[source] += 1
            metadata.append({
                "sequence": sequence_id,
                "variant": variant,
                "seed": variant_seed,
                "split": split,
                "noise_source": source,
                "pipeline_diagnostic": pipeline_diagnostics[key],
                **source_metadata,
            })

    return noisy_by_key, metadata, source_counts
def main() -> None:
    args = parse_args()
    cache_manifest_path = Path(args.cache_manifest)
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    entries_by_sequence: dict[str, list[dict[str, Any]]] = {}
    for entry in cache_manifest.get("entries", []):
        entries_by_sequence.setdefault(str(entry["sequence"]), []).append(entry)
    split_by_sequence = split_sequences(list(entries_by_sequence), args.seed)
    output_entries = []
    noise_metadata = []
    mixed_noisy_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    noise_source_counts: dict[str, int] = {}
    noise_source_by_key: dict[tuple[str, int], str] = {}
    if args.mode == "mixed_rough":
        mixed_noisy_by_key, noise_metadata, noise_source_counts = prepare_mixed_noise(
            entries_by_sequence, split_by_sequence, args
        )
        noise_source_by_key = {
            (str(record["sequence"]), int(record["variant"])): str(record["noise_source"])
            for record in noise_metadata
        }


    for sequence_index, sequence_id in enumerate(sorted(entries_by_sequence)):
        sequence_entries = sorted(entries_by_sequence[sequence_id], key=lambda item: int(item["frame_id"]))
        first_sample = np.load(sequence_entries[0]["path"], allow_pickle=False)
        clean_cameras = camera_dicts(first_sample)
        for variant in range(args.variants_per_sequence):
            variant_seed = int(args.seed + sequence_index * 1009 + variant * 9176)
            if args.mode == "mixed_rough":
                noisy_cameras = mixed_noisy_by_key[(sequence_id, variant)]
            else:
                rng = np.random.default_rng(variant_seed)
                noisy_cameras, noise_record = sample_sequence_noise(
                    clean_cameras,
                    args.mode,
                    args.anchor_index,
                    args.rotation_noise_max_deg,
                    args.center_noise_sigma_m,
                    args.center_noise_max_m,
                    rng,
                )
                noise_metadata.append({
                    "sequence": sequence_id,
                    "variant": variant,
                    "seed": variant_seed,
                    **noise_record,
                })

            for cache_entry in sequence_entries:
                clean_sample = np.load(cache_entry["path"], allow_pickle=False)
                yolo_array = np.asarray(clean_sample["yolo_keypoints_2d"], dtype=np.float32)
                yolo_by_view = {
                    str(clean_sample["views"][index]): yolo_array[index]
                    for index in range(yolo_array.shape[0])
                }
                active_views = {str(view) for view in clean_sample["views"]}
                width, height = np.asarray(clean_sample["image_size"], dtype=np.int32).tolist()
                ray_tokens, view_mask, joint_view_mask, target_confidence = make_stage_a_tokens(
                    yolo_by_view,
                    noisy_cameras,
                    int(width),
                    int(height),
                    args.confidence_threshold,
                    use_face_targets=False,
                    active_view_ids=active_views,
                )
                triangulated, triangulated_quality, triangulated_valid = triangulate_from_yolo_views(
                    yolo_by_view,
                    noisy_cameras,
                    active_views,
                    args.confidence_threshold,
                )
                noisy_origins = np.stack([camera_origin(camera) for camera in noisy_cameras]).astype(np.float32)
                clean_origins = np.asarray(clean_sample["camera_origins"], dtype=np.float32)
                origin_center = noisy_origins.mean(axis=0).astype(np.float32)
                origin_scale = np.float32(max(float(np.linalg.norm(noisy_origins - origin_center, axis=1).max()), 1.0))
                clean_translations = np.asarray(clean_sample["camera_translations"], dtype=np.float32)
                noisy_translations = np.stack([camera["t"] for camera in noisy_cameras]).astype(np.float32)
                camera_origin_delta = (clean_origins - noisy_origins) / origin_scale
                camera_translation_delta = (clean_translations - noisy_translations) / origin_scale
                frame_clean_cameras = camera_dicts(clean_sample)
                camera_rotation_delta = np.stack([
                    camera_rotation_delta_axis_angle(clean, noisy)
                    for clean, noisy in zip(frame_clean_cameras, noisy_cameras)
                ]).astype(np.float32)
                pelvis_anchor = 0.5 * (triangulated[11] + triangulated[12])
                pelvis_valid = bool(triangulated_valid[11] and triangulated_valid[12])
                pelvis_quality = float(0.5 * (triangulated_quality[11] + triangulated_quality[12])) if pelvis_valid else 0.0
                sample_id = f"{cache_entry['id']}_v{variant:02d}_{args.mode}"
                sample_path = samples_dir / f"{sample_id}.npz"
                np.savez_compressed(
                    sample_path,
                    ray_tokens=ray_tokens.astype(np.float32),
                    view_mask=view_mask.astype(np.bool_),
                    joint_view_mask=joint_view_mask.astype(np.bool_),
                    target_3d=np.asarray(clean_sample["target_3d"], dtype=np.float32),
                    target_3d_root_relative=np.asarray(clean_sample["target_3d_root_relative"], dtype=np.float32),
                    target_confidence=target_confidence.astype(np.float32),
                    triangulated_3d=triangulated.astype(np.float32),
                    triangulated_3d_root_relative=root_relative(triangulated),
                    triangulated_quality=triangulated_quality.astype(np.float32),
                    triangulated_valid=triangulated_valid.astype(np.bool_),
                    pelvis_anchor=pelvis_anchor.astype(np.float32),
                    pelvis_anchor_quality=np.asarray(pelvis_quality, dtype=np.float32),
                    pelvis_anchor_valid=np.asarray(pelvis_valid, dtype=np.bool_),
                    ray_origin_center=origin_center,
                    ray_origin_scale=np.asarray(origin_scale, dtype=np.float32),
                    camera_origin_delta=camera_origin_delta.astype(np.float32),
                    camera_translation_delta=camera_translation_delta.astype(np.float32),
                    camera_rotation_delta=camera_rotation_delta.astype(np.float32),
                    camera_scale_delta=np.zeros((len(noisy_cameras),), dtype=np.float32),
                    camera_origin_delta_valid=view_mask.astype(np.bool_),
                    source_frame=np.asarray(int(cache_entry["frame_id"]), dtype=np.int32),
                    views=np.asarray([camera["id"] for camera in noisy_cameras]),
                )
                output_entries.append({
                    "id": sample_id,
                    "path": str(sample_path),
                    "split": split_by_sequence[sequence_id],
                    "sequence": sequence_id,
                    "frame_id": int(cache_entry["frame_id"]),
                    "person_id": str(cache_entry.get("person_id", "amass_person01")),
                    "variant": variant,
                    "mode": args.mode,
                    "noise_source": noise_source_by_key.get((sequence_id, variant), args.mode),
                })

    split_counts = {
        split: sum(entry["split"] == split for entry in output_entries)
        for split in ("train", "val", "test")
    }
    manifest = {
        "metadata": {
            "stage": "camera_head_only_dataset",
            "mode": args.mode,
            "cache_manifest": str(cache_manifest_path),
            "num_entries": len(output_entries),
            "num_sequences": len(entries_by_sequence),
            "variants_per_sequence": args.variants_per_sequence,
            "anchor_index": args.anchor_index,
            "rotation_noise_max_deg": args.rotation_noise_max_deg if args.mode in {"rotation", "mixed_rough"} else 0.0,
            "center_noise_sigma_m": args.center_noise_sigma_m if args.mode in {"center", "mixed_rough"} else 0.0,
            "center_noise_max_m": args.center_noise_max_m if args.mode in {"center", "mixed_rough"} else 0.0,
            "noise_static_per_sequence": True,
            "pipeline_keypoint_source": args.pipeline_keypoint_source,
            "pipeline_temporal_lr_stabilization": (
                args.pipeline_temporal_lr_stabilization
            ),
            "pipeline_global_lr_hypotheses": args.pipeline_global_lr_hypotheses,
            "pipeline_max_global_rotation_residual_deg": (
                args.pipeline_max_global_rotation_residual_deg
            ),
            "split_rule": "sequence-level 8/1/1",
            "split_counts": split_counts,
            "noise_source_ratios_requested": {
                "pipeline_rough": args.pipeline_ratio,
                "empirical_structured": args.structured_ratio,
                "independent_random": args.random_ratio,
            } if args.mode == "mixed_rough" else {},
            "structured_noise_ranges": {
                "rotation_deg": [
                    args.structured_rotation_min_deg,
                    args.structured_rotation_max_deg,
                ],
                "center_m": [args.structured_center_min_m, args.structured_center_max_m],
            } if args.mode == "mixed_rough" else {},
            "noise_source_counts": noise_source_counts,
            "noise": noise_metadata,
        },
        "entries": output_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "split_counts": split_counts}, indent=2))


if __name__ == "__main__":
    main()
