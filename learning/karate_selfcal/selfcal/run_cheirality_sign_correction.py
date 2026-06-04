"""Correct relative-pose translation signs with lightweight cheirality voting."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .hypothesis_geometry import build_track_frame_index, collect_correspondences_for_view_pair
from .relative_pose import _prepare_normalized_points, build_rough_extrinsics, load_selected_hypothesis


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Try t and -t for each pairwise relative pose and select by cheirality."
    )
    parser.add_argument("--relative-pose-json", required=True, help="Input relative_pose_results.json")
    parser.add_argument("--rough-extrinsics-json", required=True, help="Input rough_extrinsics.json for intrinsics")
    parser.add_argument("--selected-hypothesis-json", required=True, help="Path to selected_hypothesis.json")
    parser.add_argument("--track-jsons", nargs="+", required=True, help="Per-view tracking JSON files")
    parser.add_argument("--view-ids", nargs="+", required=True, help="View ids matching --track-jsons")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--anchor-view", default=None, help="Anchor view for corrected rough extrinsics")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum keypoint confidence")
    parser.add_argument("--frame-step", type=int, default=5, help="Frame step for lightweight voting")
    parser.add_argument("--max-points-per-pair", type=int, default=500, help="Maximum sampled correspondences per pair")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def intrinsics_from_rough_extrinsics(rough_extrinsics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract intrinsics from rough_extrinsics.json in build_rough_extrinsics format."""

    intrinsics_by_view = {}
    for view_id, view_meta in rough_extrinsics.get("extrinsics_by_view", {}).items():
        intrinsics = dict(view_meta["intrinsics"])
        intrinsics["camera_matrix"] = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
        intrinsics["dist_coeffs"] = np.asarray(intrinsics["dist_coeffs"], dtype=np.float64).reshape(-1)
        intrinsics_by_view[view_id] = intrinsics
    return intrinsics_by_view


def triangulate_normalized_points(
    points_a: np.ndarray,
    points_b: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Triangulate normalized correspondences with P1=[I|0], P2=[R|t]."""

    projection_a = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    projection_b = np.hstack([rotation.reshape(3, 3), translation.reshape(3, 1)])
    points_4d = cv2.triangulatePoints(
        projection_a,
        projection_b,
        np.asarray(points_a, dtype=np.float64).T,
        np.asarray(points_b, dtype=np.float64).T,
    )
    denom = points_4d[3]
    valid = np.abs(denom) > 1e-9
    points_3d = np.full((points_4d.shape[1], 3), np.nan, dtype=np.float64)
    points_3d[valid] = (points_4d[:3, valid] / denom[valid]).T
    return points_3d


def score_translation_sign(
    normalized_a: np.ndarray,
    normalized_b: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, Any]:
    """Score one translation sign by positive depth and depth stability."""

    points_3d = triangulate_normalized_points(
        points_a=normalized_a,
        points_b=normalized_b,
        rotation=rotation,
        translation=translation,
    )
    finite_mask = np.all(np.isfinite(points_3d), axis=1)
    if not np.any(finite_mask):
        return {
            "score": 0.0,
            "valid_ratio": 0.0,
            "positive_depth_ratio_a": 0.0,
            "positive_depth_ratio_b": 0.0,
            "positive_depth_ratio_both": 0.0,
            "median_depth_a": None,
            "median_depth_b": None,
            "depth_spread": None,
        }

    finite_points = points_3d[finite_mask]
    depth_a = finite_points[:, 2]
    cam_b_points = (rotation.reshape(3, 3) @ finite_points.T).T + translation.reshape(1, 3)
    depth_b = cam_b_points[:, 2]
    positive_a = depth_a > 1e-6
    positive_b = depth_b > 1e-6
    positive_both = positive_a & positive_b
    valid_ratio = float(np.mean(finite_mask))
    positive_both_ratio = float(np.mean(positive_both)) if len(positive_both) else 0.0
    median_depth_a = float(np.median(depth_a)) if len(depth_a) else None
    median_depth_b = float(np.median(depth_b)) if len(depth_b) else None
    all_depths = np.concatenate([depth_a, depth_b])
    depth_spread = float(np.percentile(all_depths, 90) - np.percentile(all_depths, 10)) if len(all_depths) else None
    stability = 1.0 / (1.0 + float(depth_spread or 1e6))

    return {
        "score": float(positive_both_ratio * 0.9 + stability * 0.1),
        "valid_ratio": valid_ratio,
        "positive_depth_ratio_a": float(np.mean(positive_a)) if len(positive_a) else 0.0,
        "positive_depth_ratio_b": float(np.mean(positive_b)) if len(positive_b) else 0.0,
        "positive_depth_ratio_both": positive_both_ratio,
        "median_depth_a": median_depth_a,
        "median_depth_b": median_depth_b,
        "depth_spread": depth_spread,
    }


def correct_pair_sign(
    pair: dict[str, Any],
    selected_hypothesis: dict[str, Any],
    frame_indexes: dict[str, dict[int, dict[int, dict[str, Any]]]],
    intrinsics_by_view: dict[str, dict[str, Any]],
    min_confidence: float,
    frame_step: int,
    max_points_per_pair: int,
) -> dict[str, Any]:
    """Return one pair result with cheirality-selected translation sign."""

    corrected = deepcopy(pair)
    if pair.get("status") != "ok":
        corrected["cheirality_correction"] = {"status": "skipped_non_ok_pair"}
        return corrected

    view_a = str(pair["view_a"])
    view_b = str(pair["view_b"])
    points_a, points_b, metadata = collect_correspondences_for_view_pair(
        groups=list(selected_hypothesis.get("groups", [])),
        frame_indexes=frame_indexes,
        view_a=view_a,
        view_b=view_b,
        min_confidence=float(min_confidence),
        frame_step=int(frame_step),
        max_points=int(max_points_per_pair),
    )
    if len(points_a) < 8:
        corrected["cheirality_correction"] = {
            "status": "not_enough_points",
            "num_points": int(len(points_a)),
        }
        return corrected

    intr_a = intrinsics_by_view[view_a]
    intr_b = intrinsics_by_view[view_b]
    normalized_a = _prepare_normalized_points(
        points_a,
        np.asarray(intr_a["camera_matrix"], dtype=np.float64),
        np.asarray(intr_a["dist_coeffs"], dtype=np.float64),
        model=str(intr_a.get("model", "PINHOLE")),
    )
    normalized_b = _prepare_normalized_points(
        points_b,
        np.asarray(intr_b["camera_matrix"], dtype=np.float64),
        np.asarray(intr_b["dist_coeffs"], dtype=np.float64),
        model=str(intr_b.get("model", "PINHOLE")),
    )

    rotation = np.asarray(pair["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(pair["translation_unit"], dtype=np.float64).reshape(3)
    positive_score = score_translation_sign(normalized_a, normalized_b, rotation, translation)
    negative_score = score_translation_sign(normalized_a, normalized_b, rotation, -translation)
    selected_sign = 1.0 if positive_score["score"] >= negative_score["score"] else -1.0
    corrected_translation = translation * selected_sign
    corrected["translation_unit"] = corrected_translation.tolist()
    corrected["cheirality_correction"] = {
        "status": "ok",
        "num_points": int(len(metadata)),
        "selected_sign": selected_sign,
        "flipped": bool(selected_sign < 0.0),
        "positive_t": positive_score,
        "negative_t": negative_score,
    }
    return corrected


def correct_relative_pose_results(
    relative_pose_results: dict[str, Any],
    selected_hypothesis: dict[str, Any],
    track_payloads: dict[str, dict[str, Any]],
    intrinsics_by_view: dict[str, dict[str, Any]],
    view_ids: list[str],
    min_confidence: float,
    frame_step: int,
    max_points_per_pair: int,
) -> dict[str, Any]:
    """Apply cheirality sign correction to all pairwise relative poses."""

    frame_indexes = {
        view_id: build_track_frame_index(track_payloads[view_id])
        for view_id in view_ids
    }
    corrected = deepcopy(relative_pose_results)
    corrected_pairs = []
    for pair in relative_pose_results.get("pair_relative_poses", []):
        corrected_pairs.append(
            correct_pair_sign(
                pair=pair,
                selected_hypothesis=selected_hypothesis,
                frame_indexes=frame_indexes,
                intrinsics_by_view=intrinsics_by_view,
                min_confidence=float(min_confidence),
                frame_step=int(frame_step),
                max_points_per_pair=int(max_points_per_pair),
            )
        )
    corrected["pair_relative_poses"] = corrected_pairs
    corrected["cheirality_sign_correction"] = {
        "method": "pairwise_t_vs_negative_t_depth_voting",
        "min_confidence": float(min_confidence),
        "frame_step": int(frame_step),
        "max_points_per_pair": int(max_points_per_pair),
        "num_flipped_pairs": sum(
            1
            for pair in corrected_pairs
            if pair.get("cheirality_correction", {}).get("flipped")
        ),
    }
    return corrected


def main() -> None:
    """Run cheirality sign correction from the CLI."""

    args = parse_args()
    if len(args.track_jsons) != len(args.view_ids):
        raise ValueError("track-jsons and view-ids must have the same length.")

    relative_pose_results = load_json(args.relative_pose_json)
    rough_extrinsics = load_json(args.rough_extrinsics_json)
    selected_hypothesis = load_selected_hypothesis(args.selected_hypothesis_json)
    track_payloads = {
        view_id: load_json(track_json)
        for view_id, track_json in zip(args.view_ids, args.track_jsons)
    }
    intrinsics_by_view = intrinsics_from_rough_extrinsics(rough_extrinsics)

    corrected_results = correct_relative_pose_results(
        relative_pose_results=relative_pose_results,
        selected_hypothesis=selected_hypothesis,
        track_payloads=track_payloads,
        intrinsics_by_view=intrinsics_by_view,
        view_ids=list(args.view_ids),
        min_confidence=float(args.min_confidence),
        frame_step=int(args.frame_step),
        max_points_per_pair=int(args.max_points_per_pair),
    )
    corrected_rough_extrinsics = build_rough_extrinsics(
        relative_pose_results=corrected_results,
        intrinsics_by_view=intrinsics_by_view,
        view_ids=list(args.view_ids),
        anchor_view=args.anchor_view or rough_extrinsics.get("anchor_view"),
    )
    corrected_rough_extrinsics["translation_scale_note"] = (
        "Translations are unit-scale essential-matrix outputs after pairwise cheirality sign correction."
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "corrected_relative_pose_results.json").open("w", encoding="utf-8") as handle:
        json.dump(corrected_results, handle, ensure_ascii=False, indent=2)
    with (output_dir / "corrected_rough_extrinsics.json").open("w", encoding="utf-8") as handle:
        json.dump(corrected_rough_extrinsics, handle, ensure_ascii=False, indent=2)

    pair_summaries = []
    for pair in corrected_results.get("pair_relative_poses", []):
        correction = pair.get("cheirality_correction", {})
        pair_summaries.append(
            {
                "view_a": pair.get("view_a"),
                "view_b": pair.get("view_b"),
                "status": pair.get("status"),
                "correction_status": correction.get("status"),
                "selected_sign": correction.get("selected_sign"),
                "flipped": correction.get("flipped"),
                "positive_depth_ratio_both_t": correction.get("positive_t", {}).get("positive_depth_ratio_both"),
                "positive_depth_ratio_both_negative_t": correction.get("negative_t", {}).get("positive_depth_ratio_both"),
            }
        )
    summary = {
        "stage": "cheirality_sign_correction",
        "num_flipped_pairs": corrected_results["cheirality_sign_correction"]["num_flipped_pairs"],
        "rough_extrinsics": {
            "anchor_view": corrected_rough_extrinsics["anchor_view"],
            "num_resolved_views": corrected_rough_extrinsics["num_resolved_views"],
            "unresolved_views": corrected_rough_extrinsics["unresolved_views"],
        },
        "pair_summaries": pair_summaries,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
