"""One-command fast calibrated four-camera single-person 3D demo pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.signal import find_peaks

from camera_system.camera_calibration.charuco.calib_charuco_v2.filter_2d_tracks import filter_payload
from camera_system.camera_calibration.charuco.calib_charuco_v2.render_fast_pose3d import render_fast
from camera_system.camera_calibration.charuco.calib_charuco_v2.run_demo_hrnet_pose import process_video
from camera_system.camera_calibration.charuco.calib_charuco_v2.run_demo_robust_pose_pipeline import (
    load_npz_camera,
    run_robust,
    swap_payload_lr,
)
from camera_system.camera_calibration.charuco.calib_charuco_v2.sync_by_motion_peaks import (
    build_signal,
    detect_eight_events,
    detect_event_pair,
    fit_affine,
)
from learning.karate_selfcal.detection.detectors import build_detector


CAM_IDS = ["cam1demo", "cam2demo", "cam3demo", "cam4demo"]
VIDEO_NAMES = ["cam1.mkv", "cam2.mkv", "cam3.mkv", "cam4.mkv"]


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _auto_pair_candidates(
    derivative: np.ndarray, lo_frac: float, hi_frac: float
) -> list[tuple[float, list[int]]]:
    n = len(derivative)
    lo, hi = max(1, int(n * lo_frac)), min(n - 2, int(n * hi_frac))
    peaks, _ = find_peaks(derivative, distance=12)
    candidates = [int(i) for i in peaks if lo <= i <= hi and derivative[i] > 0]
    ranked: list[tuple[float, int, int]] = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            gap = second - first
            if gap < 25 or gap > 90:
                continue
            try:
                open1, close1 = detect_event_pair(derivative, first)
                open2, close2 = detect_event_pair(derivative, second)
            except RuntimeError:
                continue
            frames = [
                open1["frame_subpixel"],
                close1["frame_subpixel"],
                open2["frame_subpixel"],
                close2["frame_subpixel"],
            ]
            if not all(b > a for a, b in zip(frames, frames[1:])):
                continue
            first_duration = frames[1] - frames[0]
            second_duration = frames[3] - frames[2]
            repetition_gap = frames[2] - frames[0]
            if not (
                12.0 <= first_duration <= 32.0
                and 12.0 <= second_duration <= 32.0
                and 28.0 <= repetition_gap <= 60.0
            ):
                continue
            strength = sum(
                float(event["strength_over_local_median"])
                for event in (open1, close1, open2, close2)
            )
            ranked.append((float(strength), first, second))
    if not ranked:
        raise RuntimeError(
            f"Could not find two arm-spread gestures in frames {lo}:{hi}. "
            "Record two fast open/close gestures at both the start and end."
        )
    return [
        (strength, [first, second])
        for strength, first, second in sorted(ranked, reverse=True)
    ]


def _auto_pair(derivative: np.ndarray, lo_frac: float, hi_frac: float) -> list[int]:
    return _auto_pair_candidates(derivative, lo_frac, hi_frac)[0][1]


def build_fast_sync(poses: dict[str, dict[str, Any]], output: Path) -> dict[str, Any]:
    signals = {cam: build_signal(poses[cam]) for cam in CAM_IDS}
    reference_id = CAM_IDS[0]
    anchors = {
        reference_id: {
            "start": _auto_pair(signals[reference_id][2], 0.04, 0.34),
            "end": _auto_pair(signals[reference_id][2], 0.66, 0.97),
        }
    }
    events = {
        reference_id: detect_eight_events(
            signals[reference_id][2], anchors[reference_id]
        )
    }
    reference_peaks = np.asarray(
        [event["frame_subpixel"] for event in events[reference_id]],
        dtype=np.float64,
    )
    affine = {reference_id: fit_affine(reference_peaks, reference_peaks)}
    for cam in CAM_IDS[1:]:
        start_candidates = _auto_pair_candidates(signals[cam][2], 0.04, 0.34)[:16]
        end_candidates = _auto_pair_candidates(signals[cam][2], 0.66, 0.97)[:16]
        valid = []
        for start_strength, start_anchors in start_candidates:
            for end_strength, end_anchors in end_candidates:
                picked = {"start": start_anchors, "end": end_anchors}
                try:
                    candidate_events = detect_eight_events(signals[cam][2], picked)
                except RuntimeError:
                    continue
                local = np.asarray(
                    [event["frame_subpixel"] for event in candidate_events],
                    dtype=np.float64,
                )
                candidate_fit = fit_affine(reference_peaks, local)
                if 0.97 <= float(candidate_fit["a"]) <= 1.03:
                    valid.append((
                        float(candidate_fit["residual_rms_frames"]),
                        -(start_strength + end_strength),
                        picked,
                        candidate_events,
                        candidate_fit,
                    ))
        if not valid:
            raise RuntimeError(f"{cam}: no motion-sync candidate has plausible clock drift")
        best = min(valid, key=lambda item: (item[0], item[1]))
        if best[0] > 3.0:
            raise RuntimeError(
                f"{cam}: motion-sync event RMS is too high ({best[0]:.2f} frames)"
            )
        anchors[cam], events[cam], affine[cam] = best[2], best[3], best[4]
    for cam in CAM_IDS:
        local = np.asarray(
            [event["frame_subpixel"] for event in events[cam]], dtype=np.float64
        )
        if cam not in affine:
            affine[cam] = fit_affine(reference_peaks, local)
    report = {
        "mode": "fast_auto_8_motion_peak_affine_sync",
        "definition": "local_frame = a * reference_frame + b",
        "reference": "cam1demo",
        "reference_interval_frames": [
            int(np.floor(reference_peaks[0])),
            int(np.ceil(reference_peaks[-1])),
        ],
        "auto_coarse_anchors": anchors,
        "detected_events": events,
        "affine_local_frame=a_ref_frame_plus_b": affine,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def sync_for_robust(
    poses: dict[str, dict[str, Any]], report: dict[str, Any]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    ref_start, ref_end = [int(value) for value in report["reference_interval_frames"]]
    ref_fps = float(poses["cam1demo"]["metadata"].get("fps", 30.0))
    affine = report["affine_local_frame=a_ref_frame_plus_b"]
    sync: dict[str, dict[str, float]] = {}
    events: dict[str, dict[str, float]] = {}
    for cam in CAM_IDS:
        local_fps = float(poses[cam]["metadata"].get("fps", 30.0))
        a, b = float(affine[cam]["a"]), float(affine[cam]["b"])
        scale = float(local_fps / (ref_fps * a))
        offset_s = float(-b * scale / local_fps)
        sync[cam] = {
            "scale": scale,
            "offset_s": offset_s,
            "affine_a_local_per_ref": a,
            "affine_b_frames": b,
        }
        events[cam] = {
            "start_frame": int(round(a * ref_start + b)),
            "start_time_s": float((a * ref_start + b) / local_fps),
            "end_frame": int(round(a * ref_end + b)),
            "end_time_s": float((a * ref_end + b) / local_fps),
        }
    return sync, events


def load_cameras(calibration_dir: Path) -> dict[str, dict[str, Any]]:
    cameras = {}
    for index, cam in enumerate(CAM_IDS, start=1):
        intr = (
            calibration_dir / "intrinsics" / f"cam{index}"
            / f"cam{index}_intrinsics.npz"
        )
        ext = calibration_dir / "extrinsics" / f"cam{index}_line_extrinsics.npz"
        if not intr.exists() or not ext.exists():
            raise FileNotFoundError(
                f"Missing calibration for cam{index}: {intr} / {ext}"
            )
        cameras[cam] = load_npz_camera(intr, ext)
    return cameras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast end-to-end single-person four-camera 3D pose demo."
    )
    parser.add_argument(
        "--input", required=True, help="Folder containing cam1.mkv ... cam4.mkv"
    )
    parser.add_argument("--output", default="", help="Default: input/fast3d_output")
    parser.add_argument(
        "--calibration", default="camtest/line/testdemo/calibration"
    )
    parser.add_argument(
        "--pose-config",
        default="learning/karate_selfcal/configs/detection_hrnet_w32.yaml",
    )
    parser.add_argument("--sync-report", default="", help="Optional reusable sync report")
    parser.add_argument("--force-pose", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--pose-batch-frames", type=int, default=64)
    parser.add_argument("--yolo-batch-size", type=int, default=16)
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Development/testing only"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    input_dir = Path(args.input).resolve()
    output_dir = (
        Path(args.output).resolve() if args.output
        else input_dir / "fast3d_output"
    )
    pose_dir = output_dir / "pose_cache"
    reconstruction_dir = output_dir / "reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    videos = [input_dir / name for name in VIDEO_NAMES]
    missing = [str(path) for path in videos if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input videos: " + ", ".join(missing))

    manifest_path = pose_dir / "cache_manifest.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {}
    )
    fingerprints = {
        cam: _fingerprint(video) for cam, video in zip(CAM_IDS, videos)
    }
    pose_paths = {
        cam: pose_dir / f"keypoints_{cam}.json" for cam in CAM_IDS
    }
    stale = [
        cam for cam in CAM_IDS
        if args.force_pose
        or not pose_paths[cam].exists()
        or previous.get("videos", {}).get(cam) != fingerprints[cam]
    ]

    pose_started = time.perf_counter()
    if stale:
        config = copy.deepcopy(
            yaml.safe_load(Path(args.pose_config).read_text(encoding="utf-8"))
        )
        config.setdefault("detector", {})["max_people"] = 1
        config.setdefault("pose", {})["return_heatmaps"] = False
        detector = build_detector(config, draw_annotations=False)
        for cam, video in zip(CAM_IDS, videos):
            if cam not in stale:
                continue
            if pose_paths[cam].exists():
                pose_paths[cam].unlink()
            process_video(
                video,
                detector,
                pose_dir,
                cam,
                save_annotated=False,
                max_frames=int(args.max_frames),
                batch_frames=int(args.pose_batch_frames),
                detection_batch_size=int(args.yolo_batch_size),
            )
        manifest_path.write_text(
            json.dumps({"videos": fingerprints}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    raw_poses = {
        cam: json.loads(pose_paths[cam].read_text(encoding="utf-8"))
        for cam in CAM_IDS
    }
    timings["pose_2d_s"] = time.perf_counter() - pose_started

    filter_started = time.perf_counter()
    filtered: dict[str, dict[str, Any]] = {}
    for cam in CAM_IDS:
        payload, _ = filter_payload(
            raw_poses[cam],
            min_conf=0.25,
            max_joint_jump_px=170.0,
            smooth_window=9,
        )
        filtered[cam] = (
            swap_payload_lr(payload)
            if cam in {"cam3demo", "cam4demo"} else payload
        )
    timings["filter_2d_s"] = time.perf_counter() - filter_started

    sync_started = time.perf_counter()
    sync_path = output_dir / "sync_report.json"
    if args.sync_report:
        report = json.loads(Path(args.sync_report).read_text(encoding="utf-8"))
        shutil.copy2(args.sync_report, sync_path)
    else:
        report = build_fast_sync(filtered, sync_path)
    sync, events = sync_for_robust(filtered, report)
    timings["sync_s"] = time.perf_counter() - sync_started

    triangulation_started = time.perf_counter()
    summary = run_robust(
        filtered,
        load_cameras(Path(args.calibration).resolve()),
        sync,
        events,
        "cam1demo",
        reconstruction_dir,
        min_conf=0.30,
        ransac_px=35.0,
        fast_single_person=True,
        write_intermediates=False,
    )
    timings["triangulation_s"] = time.perf_counter() - triangulation_started

    source_json = reconstruction_dir / "triangulated_enhanced_robust.json"
    renderer_json = (
        reconstruction_dir / "triangulated_enhanced_robust_renderer_format.json"
    )
    final_json = output_dir / "pose3d.json"
    shutil.copy2(source_json, final_json)
    final_video = output_dir / "pose3d.mp4"
    if not args.no_video:
        render_started = time.perf_counter()
        render_fast(renderer_json, final_video, fps=30.0)
        timings["render_s"] = time.perf_counter() - render_started

    timings["total_s"] = time.perf_counter() - started
    result = {
        "stage": "fast_single_person_3d_demo",
        "input": str(input_dir),
        "output": str(output_dir),
        "pose3d_json": str(final_json.resolve()),
        "pose3d_video": None if args.no_video else str(final_video.resolve()),
        "sync_report": str(sync_path.resolve()),
        "used_cached_pose": not bool(stale),
        "inference_batching": {
            "pose_batch_frames": int(args.pose_batch_frames),
            "yolo_batch_size": int(args.yolo_batch_size),
        },
        "timings_seconds": {
            key: round(value, 3) for key, value in timings.items()
        },
        "quality_summary": summary,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
