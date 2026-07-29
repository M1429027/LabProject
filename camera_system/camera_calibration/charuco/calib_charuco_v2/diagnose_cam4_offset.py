"""Search a fixed cam4 frame offset against a cam1-cam3 consensus.

The important diagnostic residual is deliberately asymmetric:

1. triangulate a joint from cam1-cam3 at the reference time;
2. project that 3D point into cam4;
3. compare it with cam4's 2D observation at each candidate offset.

This prevents RANSAC/view selection from making a bad cam4 offset look good by
silently rejecting cam4.  The script also reports unfiltered four-view geometry
metrics and writes review videos for the best candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera_system.camera_calibration.charuco.calib_charuco_v2.run_demo_robust_pose_pipeline import (
    BONES,
    LR_SWAP,
    kpt_array,
    load_npz_camera,
    project_point,
    sampled_person,
    svd_triangulate,
    swap_payload_lr,
    undistort_observation,
)


CAM_IDS = ["cam1demo", "cam2demo", "cam3demo", "cam4demo"]
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    line_dir = Path(args.line_dir)
    pose_dir = Path(args.pose_dir)
    poses = {}
    for cam_id in CAM_IDS:
        path = pose_dir / f"keypoints_{cam_id}_filtered.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if cam_id in {"cam3demo", "cam4demo"} and not bool(
            payload.get("metadata", {}).get("left_right_swapped_coco17", False)
        ):
            payload = swap_payload_lr(payload)
        poses[cam_id] = payload

    refined = Path(args.refined_extrinsics_dir)
    cameras = {
        "cam1demo": load_npz_camera(
            line_dir / "calib_out_cam1_hvflip/cam1_intrinsics.npz",
            refined / "cam1_line_extrinsics_refined.npz",
        ),
        "cam2demo": load_npz_camera(
            line_dir / "calib_out_cam2_hvflip_recalc/cam2_intrinsics.npz",
            refined / "cam2_line_extrinsics_refined.npz",
        ),
        "cam3demo": load_npz_camera(
            line_dir / "calib_out_cam3_hvflip_fast/cam3_intrinsics.npz",
            refined / "cam3_line_extrinsics_refined.npz",
        ),
        "cam4demo": load_npz_camera(
            line_dir / "calib_out_cam4_hvflip_newintr_20260727/cam4_intrinsics.npz",
            refined / "cam4_line_extrinsics_refined.npz",
        ),
    }
    return poses, cameras


def frame_kpts(payload: dict[str, Any], frame_index: int) -> np.ndarray:
    return kpt_array(sampled_person(payload, frame_index))


def observation(cam_id: str, kp: np.ndarray, camera: dict[str, Any], min_conf: float) -> dict[str, Any] | None:
    if not np.isfinite(kp[:2]).all() or float(kp[2]) < min_conf:
        return None
    return {
        "cam_id": cam_id,
        "point": undistort_observation(camera, float(kp[0]), float(kp[1])),
        "raw_point": kp[:2].copy(),
        "confidence": float(kp[2]),
    }


def normalized_arm_signal(kpts: np.ndarray) -> float:
    required = [5, 6, 9, 10]
    if any(not np.isfinite(kpts[j, :2]).all() or kpts[j, 2] < 0.20 for j in required):
        return float("nan")
    shoulder_width = float(np.linalg.norm(kpts[5, :2] - kpts[6, :2]))
    if shoulder_width < 5.0:
        return float("nan")
    wrist_distance = float(np.linalg.norm(kpts[9, :2] - kpts[10, :2]))
    return wrist_distance / shoulder_width


def robust_corr(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 20:
        return float("nan")
    av = a[good]
    bv = b[good]
    if float(np.std(av)) < 1e-8 or float(np.std(bv)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def evaluate_offset(
    offset4: int,
    poses: dict[str, dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    offsets: dict[str, int],
    ref_start: int,
    ref_end: int,
    stride: int,
    min_conf: float,
) -> dict[str, Any]:
    local_offsets = dict(offsets)
    local_offsets["cam4demo"] = int(offset4)
    roots: list[np.ndarray] = []
    poses4: list[np.ndarray] = []
    reproj4: list[float] = []
    bad_bones: list[int] = []
    consensus_signals: list[float] = []
    cam4_signals: list[float] = []
    evaluated_frames: list[int] = []

    for ref_frame in range(ref_start, ref_end + 1, stride):
        by_cam = {
            cam_id: frame_kpts(poses[cam_id], ref_frame + local_offsets[cam_id])
            for cam_id in CAM_IDS
        }
        reference_arm = [
            normalized_arm_signal(by_cam[c]) for c in ("cam1demo", "cam2demo", "cam3demo")
        ]
        valid_arm = [x for x in reference_arm if np.isfinite(x)]
        consensus_signals.append(float(np.median(valid_arm)) if valid_arm else float("nan"))
        cam4_signals.append(normalized_arm_signal(by_cam["cam4demo"]))
        evaluated_frames.append(ref_frame)

        pose = np.full((17, 3), np.nan, dtype=np.float64)
        for joint in range(17):
            ref_obs = []
            for cam_id in ("cam1demo", "cam2demo", "cam3demo"):
                obs = observation(cam_id, by_cam[cam_id][joint], cameras[cam_id], min_conf)
                if obs is not None:
                    ref_obs.append(obs)
            # Require all three consensus cameras so the cam4 residual cannot be
            # dominated by an unstable two-view point.
            if len(ref_obs) == 3:
                x_ref = svd_triangulate(cameras, ref_obs)
                obs4 = observation("cam4demo", by_cam["cam4demo"][joint], cameras["cam4demo"], min_conf)
                if x_ref is not None and obs4 is not None:
                    reproj4.append(
                        float(np.linalg.norm(project_point(cameras["cam4demo"], x_ref) - obs4["raw_point"]))
                    )

            all_obs = []
            for cam_id in CAM_IDS:
                obs = observation(cam_id, by_cam[cam_id][joint], cameras[cam_id], min_conf)
                if obs is not None:
                    all_obs.append(obs)
            if len(all_obs) >= 3:
                x4 = svd_triangulate(cameras, all_obs)
                if x4 is not None:
                    pose[joint] = x4

        if np.isfinite(pose[11]).all() and np.isfinite(pose[12]).all():
            roots.append(0.5 * (pose[11] + pose[12]))
        poses4.append(pose)
        count = 0
        for a, b, lo, hi in BONES:
            if np.isfinite(pose[a]).all() and np.isfinite(pose[b]).all():
                length = float(np.linalg.norm(pose[a] - pose[b]))
                count += int(length < lo or length > hi)
        bad_bones.append(count)

    root_array = np.asarray(roots, dtype=np.float64)
    pose_array = np.asarray(poses4, dtype=np.float64)
    root_steps: list[float] = []
    action_steps: list[float] = []
    for i in range(1, len(pose_array)):
        p0, p1 = pose_array[i - 1], pose_array[i]
        if all(np.isfinite(p[j]).all() for p in (p0, p1) for j in (11, 12)):
            r0 = 0.5 * (p0[11] + p0[12])
            r1 = 0.5 * (p1[11] + p1[12])
            root_steps.append(float(np.linalg.norm(r1 - r0)) / stride)
            valid = np.isfinite(p0[:, 0]) & np.isfinite(p1[:, 0])
            if int(valid.sum()) >= 7:
                local0 = p0[valid] - r0
                local1 = p1[valid] - r1
                action_steps.append(float(np.median(np.linalg.norm(local1 - local0, axis=1))) / stride)

    consensus = np.asarray(consensus_signals, dtype=np.float64)
    sig4 = np.asarray(cam4_signals, dtype=np.float64)
    signal_corr = robust_corr(consensus, sig4)
    derivative_corr = robust_corr(np.diff(consensus), np.diff(sig4))
    return {
        "cam4_offset_frames_local_minus_ref": int(offset4),
        "num_sampled_frames": len(evaluated_frames),
        "cam123_to_cam4_reprojection_px_median": percentile(reproj4, 50),
        "cam123_to_cam4_reprojection_px_p90": percentile(reproj4, 90),
        "cam123_to_cam4_reprojection_observations": len(reproj4),
        "root_range_m": (
            (np.nanmax(root_array, axis=0) - np.nanmin(root_array, axis=0)).tolist()
            if len(root_array) else None
        ),
        "root_step_m_per_frame_p90": percentile(root_steps, 90),
        "bad_bones_per_sampled_frame_mean": float(np.mean(bad_bones)) if bad_bones else None,
        "bad_bones_per_sampled_frame_p90": percentile([float(x) for x in bad_bones], 90),
        "action_motion_m_per_frame_median": percentile(action_steps, 50),
        "action_motion_m_per_frame_p90": percentile(action_steps, 90),
        "arm_signal_correlation": signal_corr,
        "arm_motion_derivative_correlation": derivative_corr,
    }


def rank_results(results: list[dict[str, Any]]) -> None:
    """Attach a robust normalized score; lower is better."""
    components = [
        ("cam123_to_cam4_reprojection_px_median", +0.38),
        ("cam123_to_cam4_reprojection_px_p90", +0.16),
        ("root_step_m_per_frame_p90", +0.12),
        ("bad_bones_per_sampled_frame_mean", +0.12),
        ("arm_signal_correlation", -0.08),
        ("arm_motion_derivative_correlation", -0.14),
    ]
    total = np.zeros(len(results), dtype=np.float64)
    details = [dict() for _ in results]
    for key, weight in components:
        values = np.asarray([
            float(r[key]) if r.get(key) is not None and np.isfinite(float(r[key])) else np.nan
            for r in results
        ])
        finite = np.isfinite(values)
        if not finite.any():
            continue
        fill = float(np.nanmedian(values))
        values[~finite] = fill
        lo, hi = np.percentile(values, [10, 90])
        scale = max(float(hi - lo), 1e-9)
        norm = np.clip((values - lo) / scale, 0.0, 1.0)
        total += abs(weight) * (norm if weight > 0 else 1.0 - norm)
        for i, value in enumerate(norm):
            details[i][key] = float(value)
    order = np.argsort(total)
    for rank, idx in enumerate(order, start=1):
        results[int(idx)]["diagnostic_score_lower_is_better"] = float(total[int(idx)])
        results[int(idx)]["rank"] = int(rank)
        results[int(idx)]["normalized_score_components"] = details[int(idx)]

    # Keep temporal sync and geometric compatibility as separate rankings. If
    # they disagree, hiding that disagreement in one weighted score would
    # produce a misleading "best" offset.
    for result in results:
        corr = result.get("arm_signal_correlation")
        dcorr = result.get("arm_motion_derivative_correlation")
        corr = float(corr) if corr is not None and np.isfinite(float(corr)) else -1.0
        dcorr = float(dcorr) if dcorr is not None and np.isfinite(float(dcorr)) else -1.0
        result["temporal_sync_score_higher_is_better"] = float(0.45 * corr + 0.55 * dcorr)
    for rank, item in enumerate(
        sorted(results, key=lambda x: x["temporal_sync_score_higher_is_better"], reverse=True),
        start=1,
    ):
        item["temporal_sync_rank"] = int(rank)
    for rank, item in enumerate(
        sorted(
            results,
            key=lambda x: (
                float(x["cam123_to_cam4_reprojection_px_median"])
                if x.get("cam123_to_cam4_reprojection_px_median") is not None
                else float("inf")
            ),
        ),
        start=1,
    ):
        item["geometric_reprojection_rank"] = int(rank)


def draw_pose(frame: np.ndarray, kpts: np.ndarray) -> None:
    for a, b in SKELETON:
        if np.isfinite(kpts[a, :2]).all() and np.isfinite(kpts[b, :2]).all():
            cv2.line(frame, tuple(np.rint(kpts[a, :2]).astype(int)), tuple(np.rint(kpts[b, :2]).astype(int)), (80, 220, 80), 2, cv2.LINE_AA)
    for j in range(17):
        if np.isfinite(kpts[j, :2]).all():
            cv2.circle(frame, tuple(np.rint(kpts[j, :2]).astype(int)), 4, (0, 180, 255), -1, cv2.LINE_AA)


def write_review_video(
    output_path: Path,
    line_dir: Path,
    poses: dict[str, dict[str, Any]],
    offsets: dict[str, int],
    offset4: int,
    ref_start: int,
    ref_end: int,
) -> None:
    local_offsets = dict(offsets)
    local_offsets["cam4demo"] = int(offset4)
    caps = {cam: cv2.VideoCapture(str(line_dir / f"{cam}.mkv")) for cam in CAM_IDS}
    for cam_id, cap in caps.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, ref_start + local_offsets[cam_id]))
    fps = float(poses["cam1demo"]["metadata"].get("fps", 30.0))
    tile_w, tile_h = 640, 360
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tile_w * 2, tile_h * 2)
    )
    for ref_frame in range(ref_start, ref_end + 1):
        canvas = np.zeros((tile_h * 2, tile_w * 2, 3), dtype=np.uint8)
        for idx, cam_id in enumerate(CAM_IDS):
            ok, frame = caps[cam_id].read()
            if not ok:
                frame = np.zeros((
                    int(poses[cam_id]["metadata"].get("height", 720)),
                    int(poses[cam_id]["metadata"].get("width", 1280)),
                    3,
                ), dtype=np.uint8)
            src_frame = ref_frame + local_offsets[cam_id]
            draw_pose(frame, frame_kpts(poses[cam_id], src_frame))
            cv2.putText(
                frame, f"{cam_id} src={src_frame}", (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
            )
            tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            x, y = (idx % 2) * tile_w, (idx // 2) * tile_h
            canvas[y:y + tile_h, x:x + tile_w] = tile
        cv2.putText(
            canvas, f"reference={ref_frame}  cam4 offset={offset4:+d} frames",
            (24, tile_h * 2 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()
    for cap in caps.values():
        cap.release()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search cam4 fixed frame offset using HRNet and 3D consistency.")
    ap.add_argument("--line-dir", default="camtest/line")
    ap.add_argument(
        "--pose-dir",
        default="camtest/line/demo_hrnet_enhanced_20260728/pose_filtered",
    )
    ap.add_argument(
        "--sync-report",
        default="camtest/line/demo_armspread_sync_triangulation/aligned_armspread_v1/sync_report.json",
    )
    ap.add_argument(
        "--refined-extrinsics-dir",
        default="camtest/line/line_refine_20260728",
    )
    ap.add_argument(
        "--output-dir",
        default="camtest/line/demo_hrnet_enhanced_20260728/cam4_offset_diagnostic",
    )
    ap.add_argument("--offset-min", type=int, default=-90)
    ap.add_argument("--offset-max", type=int, default=-65)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.30)
    ap.add_argument("--review-count", type=int, default=3)
    ap.add_argument("--baseline-offset", type=int, default=-77)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses, cameras = load_inputs(args)
    sync_report = json.loads(Path(args.sync_report).read_text(encoding="utf-8"))
    ref_start, ref_end = [int(x) for x in sync_report["reference_interval_frames"]]
    offsets = {k: int(v) for k, v in sync_report["offsets_frames_local_minus_ref"].items()}
    results = []
    for offset4 in range(int(args.offset_min), int(args.offset_max) + 1):
        result = evaluate_offset(
            offset4, poses, cameras, offsets, ref_start, ref_end,
            stride=max(1, int(args.stride)), min_conf=float(args.min_conf),
        )
        results.append(result)
        print(
            f"[OFFSET {offset4:+d}] reproj={result['cam123_to_cam4_reprojection_px_median']:.2f}px "
            f"deriv_corr={result['arm_motion_derivative_correlation']:.3f} "
            f"bad_bones={result['bad_bones_per_sampled_frame_mean']:.2f}",
            flush=True,
        )
    rank_results(results)
    ranked = sorted(results, key=lambda x: x["rank"])
    temporal_ranked = sorted(results, key=lambda x: x["temporal_sync_rank"])
    geometry_ranked = sorted(results, key=lambda x: x["geometric_reprojection_rank"])
    recommended = temporal_ranked[0]
    review_offsets = [
        int(x["cam4_offset_frames_local_minus_ref"])
        for x in temporal_ranked[: max(1, args.review_count)]
    ]
    if int(args.baseline_offset) not in review_offsets:
        review_offsets.append(int(args.baseline_offset))
    geometry_best_offset = int(geometry_ranked[0]["cam4_offset_frames_local_minus_ref"])
    if geometry_best_offset not in review_offsets:
        review_offsets.append(geometry_best_offset)
    for offset4 in review_offsets:
        video_path = output_dir / f"sync_2x2_cam4_offset_{offset4:+d}.mp4"
        print(f"[VIDEO] {video_path}", flush=True)
        write_review_video(
            video_path, Path(args.line_dir), poses, offsets, offset4, ref_start, ref_end
        )
    report = {
        "definition": "local_frame = reference_frame + offset_frames",
        "reference_interval_frames": [ref_start, ref_end],
        "fixed_offsets": {**offsets, "cam4demo": "searched"},
        "diagnostic_note": (
            "cam123_to_cam4 reprojection triangulates from all of cam1-cam3 first, "
            "then tests cam4 without allowing RANSAC to reject it."
        ),
        "recommended_cam4_offset_frames_local_minus_ref": recommended["cam4_offset_frames_local_minus_ref"],
        "recommendation_basis": (
            "arm-distance signal and arm-motion derivative alignment; geometry is "
            "reported separately because its absolute residual is abnormally large"
        ),
        "best_temporal_sync_candidate": recommended,
        "best_geometric_reprojection_candidate": geometry_ranked[0],
        "best_combined_diagnostic_candidate": ranked[0],
        "geometry_sync_disagreement_warning": bool(
            int(recommended["cam4_offset_frames_local_minus_ref"]) != geometry_best_offset
            or float(geometry_ranked[0]["cam123_to_cam4_reprojection_px_median"]) > 50.0
        ),
        "review_video_offsets": review_offsets,
        "results_ranked": ranked,
    }
    (output_dir / "cam4_offset_diagnostic.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "best_sync_report.json").write_text(
        json.dumps(
            {
                "sync_mode": "cam4_offset_search_diagnostic",
                "definition": "local_frame = reference_frame + offset_frames",
                "reference": sync_report.get("reference", "cam1demo"),
                "reference_interval_frames": [ref_start, ref_end],
                "offsets_frames_local_minus_ref": {
                    **offsets,
                    "cam4demo": int(recommended["cam4_offset_frames_local_minus_ref"]),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "recommended_temporal_sync": recommended,
        "best_geometry": geometry_ranked[0],
        "best_combined_diagnostic": ranked[0],
        "report": str((output_dir / "cam4_offset_diagnostic.json").resolve()),
        "review_offsets": review_offsets,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
