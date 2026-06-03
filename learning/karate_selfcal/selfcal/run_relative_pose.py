"""CLI for Stage 4A relative pose estimation from the selected identity hypothesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .relative_pose import (
    approximate_intrinsics,
    estimate_all_relative_poses,
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
    parser.add_argument(
        "--intrinsics",
        nargs="*",
        default=None,
        help="Optional per-view intrinsics reports or NPZ files aligned with --track-jsons",
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


def resolve_intrinsics_by_view(
    track_payloads: dict[str, dict[str, Any]],
    view_ids: list[str],
    intrinsics_paths: list[str] | None,
    approx_focal_scale: float,
) -> dict[str, dict[str, Any]]:
    """Load provided intrinsics or fall back to a simple pinhole approximation."""

    if intrinsics_paths is not None and len(intrinsics_paths) not in (0, len(view_ids)):
        raise ValueError("intrinsics count must match view-ids count when provided.")

    intrinsics_by_view = {}
    for idx, view_id in enumerate(view_ids):
        if intrinsics_paths and idx < len(intrinsics_paths) and intrinsics_paths[idx]:
            intrinsics_by_view[view_id] = load_intrinsics(intrinsics_paths[idx])
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

    with (output_dir / "relative_pose_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    summary = {
        "stage": "relative_pose_estimation",
        "selected_hypothesis": {
            "geometry_rank": selected_hypothesis["geometry_rank"],
            "original_rank": selected_hypothesis["rank"],
            "hypothesis_id": selected_hypothesis["hypothesis_id"],
            "mean_geometry_score": selected_hypothesis["mean_geometry_score"],
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
