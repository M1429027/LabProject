"""CLI for Stage 4A relative pose estimation from the selected identity hypothesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .relative_pose import (
    approximate_intrinsics,
    build_rough_extrinsics,
    estimate_all_relative_poses,
    infer_colmap_camera_id,
    load_colmap_intrinsics,
    load_intrinsics,
    load_selected_hypothesis,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Estimate pairwise relative poses from a geometry-selected identity hypothesis."
    )
    parser.add_argument("--selected-hypothesis-json", required=True, help="Path to selected_hypothesis.json")
    parser.add_argument("--track-jsons", nargs="+", required=True, help="Per-view tracking JSON files")
    parser.add_argument("--view-ids", nargs="+", required=True, help="View ids matching --track-jsons")
    parser.add_argument("--output-dir", required=True, help="Directory for Stage 4A outputs")
    parser.add_argument("--anchor-view", default=None, help="Optional anchor view for rough extrinsics")
    parser.add_argument(
        "--intrinsics",
        nargs="*",
        default=None,
        help="Optional per-view intrinsics reports or NPZ files aligned with --track-jsons",
    )
    parser.add_argument(
        "--colmap-cameras-txt",
        default=None,
        help="Optional COLMAP cameras.txt source, or `archive.zip::inner/cameras.txt`.",
    )
    parser.add_argument(
        "--approx-focal-scale",
        type=float,
        default=1.2,
        help="If no intrinsics are provided for a view, approximate focal = scale * max(width, height).",
    )
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum keypoint confidence")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--max-points-per-pair", type=int, default=2000, help="Maximum correspondences per view pair")
    parser.add_argument(
        "--essential-ransac-threshold",
        type=float,
        default=1e-3,
        help="RANSAC threshold in normalized image coordinates for essential matrix estimation",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scale_intrinsics_to_track_frame(
    intrinsics: dict[str, Any],
    track_payload: dict[str, Any],
) -> dict[str, Any]:
    """Scale intrinsics when the 2D tracks were produced on resized frames."""

    metadata = track_payload.get("metadata", {})
    target_width = int(metadata.get("width") or 0)
    target_height = int(metadata.get("height") or 0)
    source_size = intrinsics.get("image_size") or []
    if target_width <= 0 or target_height <= 0 or len(source_size) != 2:
        return intrinsics

    source_width = float(source_size[0])
    source_height = float(source_size[1])
    if source_width <= 0 or source_height <= 0:
        return intrinsics
    if int(source_width) == target_width and int(source_height) == target_height:
        return intrinsics

    scale_x = float(target_width) / source_width
    scale_y = float(target_height) / source_height
    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64).copy()
    camera_matrix[0, 0] *= scale_x
    camera_matrix[0, 2] *= scale_x
    camera_matrix[1, 1] *= scale_y
    camera_matrix[1, 2] *= scale_y

    scaled = dict(intrinsics)
    scaled["camera_matrix"] = camera_matrix
    scaled["image_size"] = (target_width, target_height)
    scaled["scaled_from_image_size"] = (int(source_width), int(source_height))
    scaled["scale_xy"] = (scale_x, scale_y)
    return scaled


def resolve_intrinsics_by_view(
    track_payloads: dict[str, dict[str, Any]],
    view_ids: list[str],
    intrinsics_paths: list[str] | None,
    colmap_cameras_txt: str | None,
    approx_focal_scale: float,
) -> dict[str, dict[str, Any]]:
    """Load provided intrinsics or fall back to a simple pinhole approximation."""

    if intrinsics_paths is not None and len(intrinsics_paths) not in (0, len(view_ids)):
        raise ValueError("intrinsics count must match view-ids count when provided.")

    intrinsics_by_view = {}
    for idx, view_id in enumerate(view_ids):
        if intrinsics_paths and idx < len(intrinsics_paths) and intrinsics_paths[idx]:
            intrinsics_by_view[view_id] = scale_intrinsics_to_track_frame(
                load_intrinsics(intrinsics_paths[idx]),
                track_payloads[view_id],
            )
            continue
        if colmap_cameras_txt:
            intrinsics_by_view[view_id] = scale_intrinsics_to_track_frame(
                load_colmap_intrinsics(
                    colmap_cameras_txt,
                    camera_id=infer_colmap_camera_id(view_id),
                ),
                track_payloads[view_id],
            )
            continue

        metadata = track_payloads[view_id].get("metadata", {})
        width = int(metadata.get("width", 1920))
        height = int(metadata.get("height", 1080))
        intrinsics_by_view[view_id] = approximate_intrinsics(
            width=width,
            height=height,
            focal_length_scale=float(approx_focal_scale),
        )

    return intrinsics_by_view


def main() -> None:
    """Run Stage 4A relative pose estimation."""

    args = parse_args()
    if len(args.track_jsons) != len(args.view_ids):
        raise ValueError("track-jsons and view-ids must have the same length.")

    selected_hypothesis = load_selected_hypothesis(args.selected_hypothesis_json)
    track_payloads = {
        view_id: load_json(track_json)
        for view_id, track_json in zip(args.view_ids, args.track_jsons)
    }
    intrinsics_by_view = resolve_intrinsics_by_view(
        track_payloads=track_payloads,
        view_ids=list(args.view_ids),
        intrinsics_paths=list(args.intrinsics) if args.intrinsics is not None else None,
        colmap_cameras_txt=args.colmap_cameras_txt,
        approx_focal_scale=float(args.approx_focal_scale),
    )

    results = estimate_all_relative_poses(
        selected_hypothesis=selected_hypothesis,
        track_payloads=track_payloads,
        intrinsics_by_view=intrinsics_by_view,
        view_ids=list(args.view_ids),
        min_confidence=float(args.min_confidence),
        frame_step=int(args.frame_step),
        max_points_per_pair=int(args.max_points_per_pair),
        essential_ransac_threshold=float(args.essential_ransac_threshold),
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rough_extrinsics = build_rough_extrinsics(
        relative_pose_results=results,
        intrinsics_by_view=intrinsics_by_view,
        view_ids=list(args.view_ids),
        anchor_view=args.anchor_view,
    )

    with (output_dir / "relative_pose_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    with (output_dir / "rough_extrinsics.json").open("w", encoding="utf-8") as handle:
        json.dump(rough_extrinsics, handle, ensure_ascii=False, indent=2)

    summary = {
        "stage": "relative_pose_estimation",
        "selected_hypothesis": {
            "geometry_rank": selected_hypothesis["geometry_rank"],
            "original_rank": selected_hypothesis["rank"],
            "hypothesis_id": selected_hypothesis["hypothesis_id"],
            "mean_geometry_score": selected_hypothesis["mean_geometry_score"],
        },
        "rough_extrinsics": {
            "anchor_view": rough_extrinsics["anchor_view"],
            "num_resolved_views": rough_extrinsics["num_resolved_views"],
            "unresolved_views": rough_extrinsics["unresolved_views"],
        },
        "num_valid_pairs": results["num_valid_pairs"],
        "mean_pose_inlier_ratio": results["mean_pose_inlier_ratio"],
        "pair_summaries": [
            {
                "view_a": pair["view_a"],
                "view_b": pair["view_b"],
                "status": pair["status"],
                "num_correspondences": pair["num_correspondences"],
                "num_pose_inliers": pair.get("num_pose_inliers"),
                "pose_inlier_ratio": pair.get("pose_inlier_ratio"),
            }
            for pair in results["pair_relative_poses"]
        ],
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
