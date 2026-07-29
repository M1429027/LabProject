"""Two-person extension for the calibrated four-camera demo pipeline.

The implementation deliberately reuses the repository's existing single-view
pose-aware tracker and two-person global assignment hypothesis generator.  The
final cross-view identity decision is scored with the already calibrated camera
geometry instead of re-estimating a fundamental matrix for every hypothesis.

Expected input is the *unfiltered* multi-person HRNet JSON produced by
``run_demo_hrnet_pose.py``.  A fixed synchronization report is required so
identity scoring and 3D reconstruction use the exact same frame mapping.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera_system.camera_calibration.charuco.calib_charuco_v2.diagnose_multiview_pose_failure import (
    fundamental,
    symmetric_epipolar_px,
    undistorted,
)
from camera_system.camera_calibration.charuco.calib_charuco_v2.filter_2d_tracks import (
    filter_payload,
)
from camera_system.camera_calibration.charuco.calib_charuco_v2.run_demo_robust_pose_pipeline import (
    load_npz_camera,
    run_robust,
)
from learning.karate_selfcal.matching.cross_view_matcher import (
    CrossViewMatcher,
    build_tracklet_descriptors,
)
from learning.karate_selfcal.tracking.run_tracking import run_tracking_on_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track, identify, and reconstruct two people in the calibrated demo."
    )
    parser.add_argument("--line-dir", default="camtest/line")
    parser.add_argument("--pose-jsons", nargs="+", required=True, help="Raw multi-person HRNet JSONs")
    parser.add_argument(
        "--cam-ids",
        nargs="+",
        default=["cam1demo", "cam2demo", "cam3demo", "cam4demo"],
    )
    parser.add_argument("--fixed-sync-report", required=True)
    parser.add_argument("--refined-extrinsics-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference", default="cam1demo")
    parser.add_argument("--tracking-config", default="learning/karate_selfcal/configs/tracking.yaml")
    parser.add_argument("--matching-config", default="learning/karate_selfcal/configs/matching.yaml")
    parser.add_argument("--min-track-coverage", type=float, default=0.25)
    parser.add_argument("--identity-frame-step", type=int, default=3)
    parser.add_argument("--identity-min-confidence", type=float, default=0.30)
    parser.add_argument("--filter-min-confidence", type=float, default=0.25)
    parser.add_argument("--filter-max-jump-px", type=float, default=170.0)
    parser.add_argument("--filter-window", type=int, default=9)
    parser.add_argument("--triangulation-min-confidence", type=float, default=0.30)
    parser.add_argument("--ransac-reprojection-px", type=float, default=35.0)
    parser.add_argument("--save-tracked-videos", action="store_true")
    parser.add_argument("--render-video", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def normalize_bbox_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Adapt demo bbox lists to the tracker bbox mapping schema."""

    out = copy.deepcopy(payload)
    for frame in out.get("frames", []):
        for person in frame.get("people", []):
            bbox = person.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                person["bbox"] = {
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                    "confidence": float(person.get("bbox_score", 1.0)),
                }
    return out


def keep_two_longest_tracks(
    payload: dict[str, Any],
    min_coverage: float,
) -> dict[str, Any]:
    """Keep exactly two dominant post-merge tracks for the two-person demo."""

    total_frames = max(int(payload.get("metadata", {}).get("total_frames", 0)), 1)
    ranked = sorted(
        payload.get("tracks", []),
        key=lambda item: (
            int(item.get("num_detections", 0)),
            int(item.get("end_frame", 0)) - int(item.get("start_frame", 0)),
        ),
        reverse=True,
    )
    eligible = [
        item
        for item in ranked
        if float(item.get("num_detections", 0)) / total_frames >= float(min_coverage)
    ]
    if len(eligible) < 2:
        details = [
            {
                "track_id": int(item["track_id"]),
                "detections": int(item.get("num_detections", 0)),
                "coverage": float(item.get("num_detections", 0)) / total_frames,
            }
            for item in ranked
        ]
        raise RuntimeError(
            "Two persistent people were not found in one view. "
            f"Required coverage={min_coverage:.3f}; tracks={details}"
        )

    selected = eligible[:2]
    selected_ids = {int(item["track_id"]) for item in selected}
    out = copy.deepcopy(payload)
    out["tracks"] = sorted(selected, key=lambda item: int(item["track_id"]))
    for frame in out.get("frames", []):
        frame["people"] = [
            person
            for person in frame.get("people", [])
            if int(person.get("track_id", -1)) in selected_ids
        ]
    out.setdefault("metadata", {})["selected_two_track_ids"] = sorted(selected_ids)
    out["metadata"]["two_person_selection"] = "two longest tracks above minimum coverage"
    return out


def build_fixed_sync(
    pose_payloads: dict[str, dict[str, Any]],
    cam_ids: list[str],
    reference: str,
    report: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], tuple[int, int]]:
    """Convert the saved affine/offset report to the robust pipeline schema."""

    ref_start, ref_end = [int(value) for value in report["reference_interval_frames"]]
    ref_fps = float(pose_payloads[reference]["metadata"].get("fps", 30.0))
    sync: dict[str, dict[str, float]] = {}
    events: dict[str, dict[str, float]] = {}
    affine = report.get("affine_local_frame=a_ref_frame_plus_b")
    if affine:
        for cam_id in cam_ids:
            local_fps = float(pose_payloads[cam_id]["metadata"].get("fps", 30.0))
            a = float(affine[cam_id]["a"])
            b = float(affine[cam_id]["b"])
            scale = float(local_fps / (ref_fps * a))
            offset_s = float(-b * scale / local_fps)
            sync[cam_id] = {
                "scale": scale,
                "offset_s": offset_s,
                "affine_a_local_per_ref": a,
                "affine_b_frames": b,
            }
            events[cam_id] = {
                "start_frame": int(round(a * ref_start + b)),
                "start_time_s": float((a * ref_start + b) / local_fps),
                "end_frame": int(round(a * ref_end + b)),
                "end_time_s": float((a * ref_end + b) / local_fps),
            }
    else:
        offsets = report["offsets_frames_local_minus_ref"]
        for cam_id in cam_ids:
            local_fps = float(pose_payloads[cam_id]["metadata"].get("fps", 30.0))
            offset = int(offsets.get(cam_id, 0))
            sync[cam_id] = {
                "scale": 1.0,
                "offset_s": -float(offset) / local_fps,
                "fixed_offset_frames_local_minus_ref": float(offset),
            }
            events[cam_id] = {
                "start_frame": ref_start + offset,
                "start_time_s": float((ref_start + offset) / local_fps),
                "end_frame": ref_end + offset,
                "end_time_s": float((ref_end + offset) / local_fps),
            }
    return sync, events, (ref_start, ref_end)


def local_frame_for_reference(
    report: dict[str, Any],
    cam_id: str,
    reference_frame: int,
) -> int:
    affine = report.get("affine_local_frame=a_ref_frame_plus_b")
    if affine:
        item = affine[cam_id]
        return int(round(float(item["a"]) * reference_frame + float(item["b"])))
    return int(reference_frame + int(report["offsets_frames_local_minus_ref"].get(cam_id, 0)))


def frame_track_index(payload: dict[str, Any]) -> dict[int, dict[int, dict[str, Any]]]:
    return {
        int(frame["frame"]): {
            int(person["track_id"]): person
            for person in frame.get("people", [])
        }
        for frame in payload.get("frames", [])
    }


def keypoints_by_id(person: dict[str, Any], min_conf: float) -> dict[int, dict[str, float]]:
    return {
        int(point["id"]): point
        for point in person.get("keypoints", [])
        if float(point.get("confidence", 0.0)) >= min_conf
    }


def evaluate_identity_hypothesis(
    hypothesis: dict[str, Any],
    tracks: dict[str, dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    sync_report: dict[str, Any],
    cam_ids: list[str],
    frame_interval: tuple[int, int],
    frame_step: int,
    min_conf: float,
) -> dict[str, Any]:
    """Score one global assignment with the known calibrated epipolar geometry."""

    indexes = {cam: frame_track_index(tracks[cam]) for cam in cam_ids}
    fmats = {(a, b): fundamental(cameras[a], cameras[b]) for a, b in combinations(cam_ids, 2)}
    pair_errors: dict[str, list[float]] = {f"{a}+{b}": [] for a, b in combinations(cam_ids, 2)}
    identity_errors: dict[str, list[float]] = {"0": [], "1": []}
    assignments = hypothesis["assignments"]
    start, end = frame_interval

    for ref_frame in range(start, end + 1, max(1, int(frame_step))):
        for cam_a, cam_b in combinations(cam_ids, 2):
            frame_a = local_frame_for_reference(sync_report, cam_a, ref_frame)
            frame_b = local_frame_for_reference(sync_report, cam_b, ref_frame)
            pair_key = f"{cam_a}+{cam_b}"
            for identity_id in (0, 1):
                track_a = int(assignments[cam_a][f"identity_{identity_id}"])
                track_b = int(assignments[cam_b][f"identity_{identity_id}"])
                person_a = indexes[cam_a].get(frame_a, {}).get(track_a)
                person_b = indexes[cam_b].get(frame_b, {}).get(track_b)
                if person_a is None or person_b is None:
                    continue
                points_a = keypoints_by_id(person_a, min_conf)
                points_b = keypoints_by_id(person_b, min_conf)
                for joint_id in set(points_a) & set(points_b):
                    pa = undistorted(
                        cameras[cam_a],
                        np.asarray([points_a[joint_id]["x"], points_a[joint_id]["y"]]),
                    )
                    pb = undistorted(
                        cameras[cam_b],
                        np.asarray([points_b[joint_id]["x"], points_b[joint_id]["y"]]),
                    )
                    error = symmetric_epipolar_px(fmats[(cam_a, cam_b)], pa, pb)
                    pair_errors[pair_key].append(error)
                    identity_errors[str(identity_id)].append(error)

    all_errors = [value for values in pair_errors.values() for value in values]
    median = float(np.median(all_errors)) if all_errors else float("inf")
    p90 = float(np.percentile(all_errors, 90)) if all_errors else float("inf")
    pair_summary = {
        key: {
            "count": len(values),
            "median_px": float(np.median(values)) if values else None,
            "p90_px": float(np.percentile(values, 90)) if values else None,
        }
        for key, values in pair_errors.items()
    }
    return {
        **hypothesis,
        "known_geometry": {
            "num_correspondences": len(all_errors),
            "median_epipolar_px": median if np.isfinite(median) else None,
            "p90_epipolar_px": p90 if np.isfinite(p90) else None,
            "pair_summary": pair_summary,
            "identity_summary": {
                key: {
                    "count": len(values),
                    "median_px": float(np.median(values)) if values else None,
                }
                for key, values in identity_errors.items()
            },
        },
        "known_geometry_rank_key": [median, p90, -len(all_errors)],
    }


def select_identity_hypothesis(
    tracks: dict[str, dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    sync_report: dict[str, Any],
    cam_ids: list[str],
    frame_interval: tuple[int, int],
    matching_config: dict[str, Any],
    frame_step: int,
    min_conf: float,
) -> dict[str, Any]:
    descriptors = {cam: build_tracklet_descriptors(tracks[cam]) for cam in cam_ids}
    matcher = CrossViewMatcher(
        weights=dict(matching_config.get("weights", {})),
        min_keypoint_confidence=float(
            matching_config.get("matching", {}).get("min_keypoint_confidence", 0.1)
        ),
    )
    matched = matcher.match(descriptors)
    hypotheses_payload = matched["global_assignment_hypotheses"]
    if hypotheses_payload.get("status") != "ok":
        raise RuntimeError(
            "Existing two-person matcher could not enumerate assignments: "
            f"{hypotheses_payload}"
        )
    evaluated = [
        evaluate_identity_hypothesis(
            hypothesis,
            tracks,
            cameras,
            sync_report,
            cam_ids,
            frame_interval,
            frame_step,
            min_conf,
        )
        for hypothesis in hypotheses_payload.get("hypotheses", [])
    ]
    evaluated.sort(key=lambda item: tuple(item["known_geometry_rank_key"]))
    for rank, item in enumerate(evaluated):
        item["known_geometry_rank"] = rank
    if not evaluated or evaluated[0]["known_geometry"]["num_correspondences"] == 0:
        raise RuntimeError("No synchronized cross-view keypoint correspondences were available.")
    return {
        "selection_method": "known_calibrated_epipolar_geometry",
        "selected_hypothesis": evaluated[0],
        "evaluated_hypotheses": evaluated,
        "pairwise_matcher": matched,
    }


def payload_for_track(payload: dict[str, Any], track_id: int) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    for frame in out.get("frames", []):
        frame["people"] = [
            person
            for person in frame.get("people", [])
            if int(person.get("track_id", -1)) == int(track_id)
        ]
    out.setdefault("metadata", {})["selected_track_id"] = int(track_id)
    return out


def load_demo_cameras(line_dir: Path, refined_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "cam1demo": load_npz_camera(
            line_dir / "calib_out_cam1_hvflip/cam1_intrinsics.npz",
            refined_dir / "cam1_line_extrinsics_refined.npz",
        ),
        "cam2demo": load_npz_camera(
            line_dir / "calib_out_cam2_hvflip_recalc/cam2_intrinsics.npz",
            refined_dir / "cam2_line_extrinsics_refined.npz",
        ),
        "cam3demo": load_npz_camera(
            line_dir / "calib_out_cam3_hvflip_fast/cam3_intrinsics.npz",
            refined_dir / "cam3_line_extrinsics_refined.npz",
        ),
        "cam4demo": load_npz_camera(
            line_dir / "calib_out_cam4_hvflip_newintr_20260727/cam4_intrinsics.npz",
            refined_dir / "cam4_line_extrinsics_refined.npz",
        ),
    }


def merge_identity_outputs(
    identity_dirs: list[Path],
    selection: dict[str, Any],
    output_dir: Path,
) -> Path:
    payloads = [
        load_json(path / "triangulated_enhanced_robust_renderer_format.json")
        for path in identity_dirs
    ]
    frame_count = min(len(payload.get("frames", [])) for payload in payloads)
    frames = []
    for frame_idx in range(frame_count):
        identities = []
        source_frames = payloads[0]["frames"][frame_idx].get("source_frames", {})
        for identity_id, payload in enumerate(payloads):
            source_identity = payload["frames"][frame_idx].get("identities", [{}])[0]
            item = copy.deepcopy(source_identity)
            item["identity_id"] = identity_id
            identities.append(item)
        frames.append(
            {
                "frame": frame_idx,
                "source_frames": source_frames,
                "identities": identities,
            }
        )

    selected = selection["selected_hypothesis"]
    summary = {
        "stage": "two_person_calibrated_demo",
        "num_frames": frame_count,
        "num_identities": 2,
        "identity_selection": "known_calibrated_epipolar_geometry",
        "identity_median_epipolar_px": selected["known_geometry"]["median_epipolar_px"],
        "identity_p90_epipolar_px": selected["known_geometry"]["p90_epipolar_px"],
        "assignments": selected["assignments"],
    }
    merged = {"metadata": summary, "summary": summary, "frames": frames}
    output_path = output_dir / "triangulated_two_person_renderer_format.json"
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "two_person_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def main() -> None:
    args = parse_args()
    if len(args.pose_jsons) != len(args.cam_ids):
        raise ValueError("--pose-jsons and --cam-ids must have the same length")
    if len(args.cam_ids) < 2:
        raise ValueError("At least two camera views are required")

    line_dir = Path(args.line_dir)
    output_dir = Path(args.output_dir)
    tracking_dir = output_dir / "tracking"
    filtered_dir = output_dir / "identity_pose_filtered"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    raw_poses = {
        cam: normalize_bbox_dict(load_json(path))
        for cam, path in zip(args.cam_ids, args.pose_jsons)
    }
    tracking_config = load_yaml(args.tracking_config)
    matching_config = load_yaml(args.matching_config)
    tracked: dict[str, dict[str, Any]] = {}
    for cam_id in args.cam_ids:
        source_video = line_dir / f"{cam_id}.mkv"
        payload = run_tracking_on_view(
            detection_payload=raw_poses[cam_id],
            view_id=cam_id,
            config=tracking_config,
            output_dir=tracking_dir,
            source_video=source_video if source_video.exists() else None,
            save_tracked_video=bool(args.save_tracked_videos),
        )
        tracked[cam_id] = keep_two_longest_tracks(payload, args.min_track_coverage)
        (tracking_dir / f"tracks_{cam_id}_top2.json").write_text(
            json.dumps(tracked[cam_id], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    refined_dir = Path(args.refined_extrinsics_dir)
    all_cameras = load_demo_cameras(line_dir, refined_dir)
    cameras = {cam: all_cameras[cam] for cam in args.cam_ids}
    sync_report = load_json(args.fixed_sync_report)
    sync, events, frame_interval = build_fixed_sync(
        raw_poses,
        list(args.cam_ids),
        args.reference,
        sync_report,
    )

    selection = select_identity_hypothesis(
        tracks=tracked,
        cameras=cameras,
        sync_report=sync_report,
        cam_ids=list(args.cam_ids),
        frame_interval=frame_interval,
        matching_config=matching_config,
        frame_step=args.identity_frame_step,
        min_conf=args.identity_min_confidence,
    )
    (output_dir / "identity_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assignments = selection["selected_hypothesis"]["assignments"]
    identity_dirs: list[Path] = []
    for identity_id in (0, 1):
        identity_poses: dict[str, dict[str, Any]] = {}
        for cam_id in args.cam_ids:
            track_id = int(assignments[cam_id][f"identity_{identity_id}"])
            selected_payload = payload_for_track(tracked[cam_id], track_id)
            filtered, metrics = filter_payload(
                selected_payload,
                min_conf=float(args.filter_min_confidence),
                max_joint_jump_px=float(args.filter_max_jump_px),
                smooth_window=int(args.filter_window),
            )
            filtered.setdefault("metadata", {})["global_identity_id"] = identity_id
            filtered["metadata"]["source_track_id"] = track_id
            identity_poses[cam_id] = filtered
            (filtered_dir / f"identity_{identity_id}_{cam_id}.json").write_text(
                json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (filtered_dir / f"identity_{identity_id}_{cam_id}_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        identity_dir = output_dir / f"identity_{identity_id}_3d"
        identity_dir.mkdir(parents=True, exist_ok=True)
        run_robust(
            identity_poses,
            cameras,
            sync,
            events,
            args.reference,
            identity_dir,
            min_conf=float(args.triangulation_min_confidence),
            ransac_px=float(args.ransac_reprojection_px),
        )
        identity_dirs.append(identity_dir)

    merged_json = merge_identity_outputs(identity_dirs, selection, output_dir)
    if args.render_video:
        output_video = output_dir / "triangulated_two_person.mp4"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "learning.karate_selfcal.tools.visualize_triangulation",
                "--input-json",
                str(merged_json),
                "--output-video",
                str(output_video),
                "--fps",
                "30",
                "--width",
                "1280",
                "--height",
                "720",
                "--up-axis",
                "z",
                "--axis-percentile",
                "95",
                "--highlight-key-joints",
            ],
            check=True,
            cwd=REPO_ROOT,
        )

    compact = {
        "stage": "two_person_calibrated_demo",
        "output_dir": str(output_dir.resolve()),
        "merged_json": str(merged_json.resolve()),
        "selected_assignments": assignments,
        "identity_geometry": selection["selected_hypothesis"]["known_geometry"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
