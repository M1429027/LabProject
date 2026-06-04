"""Run the COLMAP oracle reference baseline for reconstruction validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..reconstruction.run_triangulation import load_json, triangulate_sequence
from ..reconstruction.temporal_completion import complete_payload
from ..selfcal.export_colmap_extrinsics import export_colmap_extrinsics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run reference reconstruction with dataset COLMAP intrinsics/extrinsics."
    )
    parser.add_argument("--selected-hypothesis-json", required=True, help="Path to selected_hypothesis.json")
    parser.add_argument("--track-jsons", nargs="+", required=True, help="Per-view tracking JSON files")
    parser.add_argument("--view-ids", nargs="+", required=True, help="View ids matching --track-jsons")
    parser.add_argument("--colmap-cameras-txt", required=True, help="COLMAP cameras.txt or archive.zip::inner path")
    parser.add_argument("--colmap-images-txt", required=True, help="COLMAP images.txt or archive.zip::inner path")
    parser.add_argument("--output-dir", required=True, help="Reference baseline output directory")
    parser.add_argument("--anchor-view", default=None, help="Anchor view id; defaults to first view")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum keypoint confidence")
    parser.add_argument("--min-views", type=int, default=2, help="Minimum views required per joint")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--geometry-sampson-threshold", type=float, default=1.0)
    parser.add_argument("--min-pair-support", type=int, default=1)
    parser.add_argument("--max-pixel-reprojection-error", type=float, default=60.0)
    parser.add_argument("--expected-joints", type=int, default=17)
    parser.add_argument("--completion-max-gap", type=int, default=5)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--render-video", action="store_true", help="Render raw and completed review videos")
    parser.add_argument("--fps", type=float, default=10.0, help="Review video FPS")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def scale_extrinsics_intrinsics_to_tracks(
    rough_extrinsics: dict[str, Any],
    track_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Scale COLMAP intrinsics when 2D tracks come from resized videos."""

    scaled = dict(rough_extrinsics)
    scaled["extrinsics_by_view"] = dict(rough_extrinsics.get("extrinsics_by_view", {}))
    for view_id, view_meta in rough_extrinsics.get("extrinsics_by_view", {}).items():
        track_meta = track_payloads.get(view_id, {}).get("metadata", {})
        target_width = int(track_meta.get("width") or 0)
        target_height = int(track_meta.get("height") or 0)
        if target_width <= 0 or target_height <= 0:
            continue

        intrinsics = dict(view_meta.get("intrinsics", {}))
        source_size = intrinsics.get("image_size") or []
        if len(source_size) != 2:
            continue
        source_width = float(source_size[0])
        source_height = float(source_size[1])
        if source_width <= 0 or source_height <= 0:
            continue
        if int(source_width) == target_width and int(source_height) == target_height:
            continue

        scale_x = float(target_width) / source_width
        scale_y = float(target_height) / source_height
        camera_matrix = [
            [float(v) for v in row]
            for row in intrinsics["camera_matrix"]
        ]
        camera_matrix[0][0] *= scale_x
        camera_matrix[0][2] *= scale_x
        camera_matrix[1][1] *= scale_y
        camera_matrix[1][2] *= scale_y
        intrinsics["camera_matrix"] = camera_matrix
        intrinsics["image_size"] = [target_width, target_height]
        intrinsics["scaled_from_image_size"] = [int(source_width), int(source_height)]
        intrinsics["scale_xy"] = [scale_x, scale_y]

        new_view_meta = dict(view_meta)
        new_view_meta["intrinsics"] = intrinsics
        scaled["extrinsics_by_view"][view_id] = new_view_meta

    scaled["intrinsics_scaling_note"] = (
        "COLMAP intrinsics are scaled to the per-view tracking coordinate system when needed."
    )
    return scaled


def render_review_video(input_json: Path, output_video: Path, fps: float) -> None:
    """Render one triangulation review video through the existing visualization CLI."""

    command = [
        sys.executable,
        "-m",
        "learning.karate_selfcal.tools.visualize_triangulation",
        "--input-json",
        str(input_json),
        "--output-video",
        str(output_video),
        "--fps",
        str(float(fps)),
        "--per-frame-axis",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    """Run the complete reference baseline branch."""

    args = parse_args()
    if len(args.track_jsons) != len(args.view_ids):
        raise ValueError("track-jsons and view-ids must have the same length.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_view = args.anchor_view or args.view_ids[0]

    track_payloads = {
        view_id: load_json(track_json)
        for view_id, track_json in zip(args.view_ids, args.track_jsons)
    }

    rough_extrinsics = export_colmap_extrinsics(
        colmap_cameras_txt=args.colmap_cameras_txt,
        colmap_images_txt=args.colmap_images_txt,
        view_ids=list(args.view_ids),
        anchor_view=anchor_view,
    )
    rough_extrinsics = scale_extrinsics_intrinsics_to_tracks(
        rough_extrinsics=rough_extrinsics,
        track_payloads=track_payloads,
    )
    rough_extrinsics_path = output_dir / "rough_extrinsics_colmap.json"
    write_json(rough_extrinsics_path, rough_extrinsics)

    selected_payload = load_json(args.selected_hypothesis_json)
    selected_hypothesis = selected_payload["selected_hypothesis"]

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
    )
    triangulated["metadata"]["stage"] = "reference_baseline_colmap_triangulation"
    triangulated_path = output_dir / "triangulated_3d_raw.json"
    write_json(triangulated_path, triangulated)
    write_json(output_dir / "triangulation_summary.json", triangulated["summary"])

    completed_payload, completion_metrics = complete_payload(
        payload=triangulated,
        expected_joints=int(args.expected_joints),
        max_gap=int(args.completion_max_gap),
        smooth_window=int(args.smooth_window),
    )
    completed_path = output_dir / "triangulated_3d_completed.json"
    write_json(completed_path, completed_payload)
    write_json(output_dir / "completion_metrics.json", completion_metrics)

    rendered_outputs = {}
    if args.render_video:
        raw_video = output_dir / "triangulated_3d_raw_review.mp4"
        completed_video = output_dir / "triangulated_3d_completed_review.mp4"
        render_review_video(triangulated_path, raw_video, fps=float(args.fps))
        render_review_video(completed_path, completed_video, fps=float(args.fps))
        rendered_outputs = {
            "raw_review_video": str(raw_video),
            "completed_review_video": str(completed_video),
        }

    summary = {
        "stage": "reference_baseline_colmap",
        "output_dir": str(output_dir),
        "rough_extrinsics_json": str(rough_extrinsics_path),
        "raw_triangulation_json": str(triangulated_path),
        "completed_triangulation_json": str(completed_path),
        "triangulation_summary": triangulated["summary"],
        "completion_metrics": {
            "original_observed_ratio": completion_metrics.get("original_observed_ratio"),
            "completed_observed_ratio": completion_metrics.get("completed_observed_ratio"),
            "absolute_ratio_gain": completion_metrics.get("absolute_ratio_gain"),
            "relative_observed_gain": completion_metrics.get("relative_observed_gain"),
        },
        **rendered_outputs,
    }
    write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
