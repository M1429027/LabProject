"""Enhanced camtest/line demo pipeline.

Inputs are per-camera 2D pose JSONs in the existing demo schema.  The script
auto-syncs by the existing arm/clap pose signal, writes synced videos/keypoints,
then runs:

1. optional COCO left/right correction per camera,
2. pose-level temporal view selection,
3. joint-level inlier rejection inside the selected subset,
4. light 3D temporal smoothing,
5. skeleton optimization with bone-length and foot-ground priors.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera_system.camera_calibration.charuco.calib_charuco_v2.run_demo_pose_sync_triangulation import (
    KEYPOINT_NAMES,
    SKELETON_CONNECTIONS,
    build_sync,
    clap_pose_signal,
    detect_clap_events,
    load_npz_camera,
    refine_with_motion,
    select_main_person,
)


LR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
BONES = [
    (5, 6, 0.18, 0.85),
    (11, 12, 0.12, 0.75),
    (5, 7, 0.15, 0.85),
    (7, 9, 0.12, 0.80),
    (6, 8, 0.15, 0.85),
    (8, 10, 0.12, 0.80),
    (11, 13, 0.18, 1.00),
    (13, 15, 0.18, 1.00),
    (12, 14, 0.18, 1.00),
    (14, 16, 0.18, 1.00),
    (5, 11, 0.25, 1.20),
    (6, 12, 0.25, 1.20),
]


def kpt_array(person: dict[str, Any] | None) -> np.ndarray:
    arr = np.full((17, 3), np.nan, dtype=np.float64)
    if person is None:
        return arr
    for kp in person.get("keypoints", []):
        idx = int(kp["id"])
        if 0 <= idx < 17:
            arr[idx] = [float(kp["x"]), float(kp["y"]), float(kp.get("confidence", 0.0))]
    return arr


def swap_payload_lr(payload: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    for frame in out.get("frames", []):
        for person in frame.get("people", []):
            by_id = {int(k["id"]): dict(k) for k in person.get("keypoints", [])}
            swapped = []
            for new_id, old_id in enumerate(LR_SWAP):
                if old_id in by_id:
                    item = by_id[old_id]
                    item["id"] = new_id
                    swapped.append(item)
            person["keypoints"] = swapped
    meta = out.setdefault("metadata", {})
    meta["left_right_swapped_coco17"] = True
    return out


def sampled_person(payload: dict[str, Any], frame_index: int) -> dict[str, Any] | None:
    frames = payload["frames"]
    if frame_index < 0 or frame_index >= len(frames):
        return None
    return select_main_person(frames[frame_index])


def undistort_observation(camera: dict[str, Any], x: float, y: float) -> np.ndarray:
    pts = np.array([[[float(x), float(y)]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, camera["K"], camera["dist"], P=camera["K"])
    return und.reshape(2)


def project_point(camera: dict[str, Any], X: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(camera["R"])
    uv, _ = cv2.projectPoints(np.asarray([[X]], dtype=np.float64), rvec, camera["t"], camera["K"], camera["dist"])
    return uv.reshape(2)


def svd_triangulate(cameras: dict[str, dict[str, Any]], obs: list[dict[str, Any]]) -> np.ndarray | None:
    A = []
    for o in obs:
        P = cameras[o["cam_id"]]["P"]
        x, y = o["point"]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    if len(A) < 4:
        return None
    _, _, vh = np.linalg.svd(np.asarray(A, dtype=np.float64))
    X = vh[-1]
    if abs(X[3]) < 1e-12:
        return None
    X = X[:3] / X[3]
    if not np.isfinite(X).all():
        return None
    if abs(X[0]) > 10 or abs(X[1]) > 10 or X[2] < -1.5 or X[2] > 4.5:
        return None
    return X


def reproj_errors(cameras: dict[str, dict[str, Any]], X: np.ndarray, obs: list[dict[str, Any]]) -> list[float]:
    return [float(np.linalg.norm(project_point(cameras[o["cam_id"]], X) - o["raw_point"])) for o in obs]


def camera_center(camera: dict[str, Any]) -> np.ndarray:
    return (-camera["R"].T @ camera["t"].reshape(3, 1)).reshape(3)


def min_triangulation_angle_deg(cameras: dict[str, dict[str, Any]], X: np.ndarray, obs: list[dict[str, Any]]) -> float:
    if len(obs) < 2:
        return 0.0
    rays = []
    for o in obs:
        C = camera_center(cameras[o["cam_id"]])
        v = np.asarray(X, dtype=np.float64) - C
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            rays.append(v / n)
    vals = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            c = float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))
            vals.append(float(np.degrees(np.arccos(c))))
    return float(min(vals)) if vals else 0.0


def triangulate_joint_robust(
    cameras: dict[str, dict[str, Any]],
    obs: list[dict[str, Any]],
    min_views: int,
    threshold_px: float,
) -> tuple[np.ndarray | None, list[dict[str, Any]], dict[str, Any]]:
    """Strict joint RANSAC.

    3+ views are preferred whenever available.  Two-view solutions are only a
    fallback and must pass tighter reprojection/confidence/angle gates; this
    prevents two mutually-wrong views from dominating just because they agree.
    """
    if len(obs) < min_views:
        return None, [], {"status": "too_few_views"}
    candidates = []
    preferred_min = 3 if len(obs) >= 3 else 2
    for r in range(preferred_min, len(obs) + 1):
        for idxs in combinations(range(len(obs)), r):
            subset = [obs[i] for i in idxs]
            X = svd_triangulate(cameras, subset)
            if X is None:
                continue
            all_err = reproj_errors(cameras, X, obs)
            gate = float(threshold_px) if r >= 3 else min(14.0, float(threshold_px) * 0.45)
            inliers = [obs[i] for i, e in enumerate(all_err) if np.isfinite(e) and e <= gate]
            if len(inliers) < preferred_min:
                continue
            X2 = svd_triangulate(cameras, inliers)
            if X2 is None:
                continue
            err = reproj_errors(cameras, X2, inliers)
            angle = min_triangulation_angle_deg(cameras, X2, inliers)
            conf = float(np.mean([o["confidence"] for o in inliers]))
            if len(inliers) == 2 and (float(np.max(err)) > 14.0 or conf < 0.45 or angle < 5.0):
                continue
            if len(inliers) >= 3 and (float(np.percentile(err, 90)) > float(threshold_px) or angle < 2.0):
                continue
            score = float(np.median(err) + 0.35 * np.percentile(err, 90) - 3.0 * len(inliers) - 0.03 * angle - 1.2 * conf)
            if len(inliers) == 2:
                score += 18.0
            candidates.append((score, X2, inliers, err, angle, conf))
    # Last-resort two-view fallback only if no 3-view solution exists.
    if not candidates and len(obs) >= 2:
        for idxs in combinations(range(len(obs)), 2):
            subset = [obs[i] for i in idxs]
            X = svd_triangulate(cameras, subset)
            if X is None:
                continue
            err = reproj_errors(cameras, X, subset)
            angle = min_triangulation_angle_deg(cameras, X, subset)
            conf = float(np.mean([o["confidence"] for o in subset]))
            if float(np.max(err)) <= 10.0 and conf >= 0.55 and angle >= 7.0:
                score = float(np.median(err) + 24.0 - 0.04 * angle - conf)
                candidates.append((score, X, subset, err, angle, conf))
    if not candidates:
        return None, [], {"status": "no_strict_inlier_solution"}
    score, X, inliers, err, angle, conf = min(candidates, key=lambda x: x[0])
    return X, inliers, {
        "status": "ok",
        "strict_ransac": True,
        "inlier_views": [o["cam_id"] for o in inliers],
        "mean_reprojection_error_px": float(np.mean(err)),
        "median_reprojection_error_px": float(np.median(err)),
        "max_reprojection_error_px": float(np.max(err)),
        "min_triangulation_angle_deg": float(angle),
        "mean_confidence": float(conf),
        "candidate_score": float(score),
    }


def pose_candidate(
    f: int,
    subset: tuple[str, ...],
    kps_by_cam: dict[str, np.ndarray],
    cameras: dict[str, dict[str, Any]],
    min_conf: float,
    ransac_px: float,
) -> dict[str, Any] | None:
    pts = np.full((17, 3), np.nan, dtype=np.float64)
    valid = np.zeros(17, dtype=bool)
    joint_meta: list[dict[str, Any] | None] = [None] * 17
    med_errs = []
    effective_views: set[str] = set()
    inlier_view_counts: list[int] = []
    for j in range(17):
        obs = []
        for cam_id in subset:
            kp = kps_by_cam[cam_id][f, j]
            if np.isfinite(kp[0]) and kp[2] >= min_conf:
                obs.append(
                    {
                        "cam_id": cam_id,
                        "point": undistort_observation(cameras[cam_id], kp[0], kp[1]),
                        "raw_point": kp[:2],
                        "confidence": float(kp[2]),
                    }
                )
        X, inliers, meta = triangulate_joint_robust(cameras, obs, min_views=2, threshold_px=ransac_px)
        if X is None:
            continue
        pts[j] = X
        valid[j] = True
        joint_meta[j] = meta
        med_errs.append(float(meta["median_reprojection_error_px"]))
        joint_views = [str(cam_id) for cam_id in meta.get("inlier_views", [])]
        effective_views.update(joint_views)
        inlier_view_counts.append(len(joint_views))
    if valid.sum() < 7:
        return None

    root = None
    if valid[11] and valid[12]:
        root = 0.5 * (pts[11] + pts[12])
    elif valid[5] and valid[6]:
        root = 0.5 * (pts[5] + pts[6])

    bad_bones = 0
    bone_pen = 0.0
    for a, b, lo, hi in BONES:
        if valid[a] and valid[b]:
            L = float(np.linalg.norm(pts[a] - pts[b]))
            if L < lo:
                bone_pen += (lo - L) * 8.0
                bad_bones += 1
            elif L > hi:
                bone_pen += (L - hi) * 12.0
                bad_bones += 1
    med = float(np.median(med_errs)) if med_errs else 999.0
    p90 = float(np.percentile(med_errs, 90)) if med_errs else 999.0
    score = med * 0.55 + p90 * 0.08 + bone_pen + bad_bones * 2.2 + (17 - int(valid.sum())) * 1.4
    # Reward the number of cameras that joint RANSAC actually accepted, not
    # the size of the candidate pool.  Otherwise an all-camera candidate gets
    # a four-view bonus even when some cameras have no person observation.
    effective_view_count = int(round(float(np.median(inlier_view_counts)))) if inlier_view_counts else 0
    score += {2: 7.5, 3: -1.4, 4: -2.4}.get(effective_view_count, 12.0)
    score += 0.75 * max(0, len(subset) - len(effective_views))
    return {
        "subset": subset,
        "effective_subset": tuple(cam_id for cam_id in subset if cam_id in effective_views),
        "effective_view_count": effective_view_count,
        "pts": pts,
        "valid": valid,
        "root": root,
        "joint_meta": joint_meta,
        "score": float(score),
        "med_err": med,
        "p90_err": p90,
        "bad_bones": int(bad_bones),
    }


def temporal_joint_gate(pts: np.ndarray, valid: np.ndarray, max_root_jump_m: float = 0.65, max_joint_jump_m: float = 0.95) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gated = pts.copy()
    out_valid = valid.copy()
    rejected = 0
    prev_root = None
    prev_pts = np.full((17, 3), np.nan, dtype=np.float64)
    for f in range(gated.shape[0]):
        root = None
        if out_valid[f, 11] and out_valid[f, 12]:
            root = 0.5 * (gated[f, 11] + gated[f, 12])
        elif out_valid[f, 5] and out_valid[f, 6]:
            root = 0.5 * (gated[f, 5] + gated[f, 6])
        root_jump = 0.0 if root is None or prev_root is None else float(np.linalg.norm(root - prev_root))
        for j in range(17):
            if not out_valid[f, j] or not np.isfinite(gated[f, j]).all():
                continue
            if np.isfinite(prev_pts[j]).all():
                jump = float(np.linalg.norm(gated[f, j] - prev_pts[j]))
                limit = max_joint_jump_m + 0.35 * min(root_jump, max_root_jump_m)
                if jump > limit:
                    out_valid[f, j] = False
                    gated[f, j] = np.nan
                    rejected += 1
                    continue
        if root is not None:
            prev_root = root
        for j in range(17):
            if out_valid[f, j] and np.isfinite(gated[f, j]).all():
                prev_pts[j] = gated[f, j]
    return gated, out_valid, {"temporal_joint_gate_rejected": int(rejected), "max_joint_jump_m": float(max_joint_jump_m)}


def smooth_sequence(pts: np.ndarray, valid: np.ndarray, window: int = 9) -> np.ndarray:
    n = pts.shape[0]
    # Preserve true absence outside each joint's first/last multiview
    # observation.  np.interp extrapolates edge values by default, which made
    # a person appear before entering or remain after leaving the scene.
    out = np.full_like(pts, np.nan, dtype=np.float64)
    for j in range(17):
        m = valid[:, j] & np.isfinite(pts[:, j, 0])
        if m.sum() == 1:
            out[m, j] = pts[m, j]
            continue
        if m.sum() >= 2:
            observed = np.flatnonzero(m)
            lo, hi = int(observed[0]), int(observed[-1])
            segment_x = np.arange(lo, hi + 1)
            for c in range(3):
                out[lo : hi + 1, j, c] = np.interp(
                    segment_x, observed, pts[m, j, c]
                )
            segment_n = hi - lo + 1
            win = min(window, segment_n if segment_n % 2 == 1 else segment_n - 1)
            if win >= 5:
                for c in range(3):
                    out[lo : hi + 1, j, c] = savgol_filter(
                        out[lo : hi + 1, j, c], win, 2
                    )
    return out


def skeleton_optimize(pts: np.ndarray, valid: np.ndarray, max_nfev: int = 60) -> tuple[np.ndarray, dict[str, Any]]:
    """Fast demo skeleton optimizer.

    Uses median bone priors plus iterative per-frame bone projection.  This is
    much faster than full-sequence least-squares and is intended for demo review:
    it suppresses impossible limb stretching without pretending to solve full IK.
    """

    X = pts.copy()
    mask = np.isfinite(X[:, :, 0])
    bone_priors: dict[tuple[int, int], float] = {}
    for a, b, lo, hi in BONES:
        vals = np.linalg.norm(X[:, a] - X[:, b], axis=1)
        good = mask[:, a] & mask[:, b] & np.isfinite(vals) & (vals > lo) & (vals < hi)
        if good.sum() >= 8:
            bone_priors[(a, b)] = float(np.median(vals[good]))

    before_bad = 0
    after_bad = 0
    for f in range(X.shape[0]):
        for a, b, lo, hi in BONES:
            if mask[f, a] and mask[f, b]:
                L = float(np.linalg.norm(X[f, b] - X[f, a]))
                before_bad += int(L < lo or L > hi)
        # Anchor torso/root more than limbs; adjust both ends lightly so the
        # skeleton remains close to triangulation but loses extreme stretching.
        for _ in range(4):
            for (a, b), target in bone_priors.items():
                if not (mask[f, a] and mask[f, b]):
                    continue
                vec = X[f, b] - X[f, a]
                L = float(np.linalg.norm(vec))
                if L < 1e-6:
                    continue
                if 0.72 * target <= L <= 1.35 * target:
                    continue
                desired = vec / L * target
                mid = 0.5 * (X[f, a] + X[f, b])
                # Hips/shoulders are stronger anchors; distal joints move more.
                if a in (5, 6, 11, 12) and b not in (5, 6, 11, 12):
                    X[f, b] = X[f, a] + desired
                else:
                    X[f, a] = mid - 0.5 * desired
                    X[f, b] = mid + 0.5 * desired
        for a, b, lo, hi in BONES:
            if mask[f, a] and mask[f, b]:
                L = float(np.linalg.norm(X[f, b] - X[f, a]))
                after_bad += int(L < lo or L > hi)

    # Foot-ground contact: shift the whole body vertically so low ankle frames
    # are close to z=0.  This preserves relative pose more than clamping ankles.
    ankle_z = []
    for j in (15, 16):
        good = mask[:, j] & np.isfinite(X[:, j, 2])
        ankle_z.extend(X[good, j, 2].tolist())
    ground_shift = 0.0
    if len(ankle_z) >= 20:
        floor_est = float(np.percentile(ankle_z, 8))
        ground_shift = floor_est - 0.03
        X[:, :, 2] -= ground_shift

    return X, {
        "optimizer_success": True,
        "method": "fast_bone_projection_ground_shift",
        "bone_priors": {f"{a}-{b}": v for (a, b), v in bone_priors.items()},
        "bad_bone_count_before_projection": int(before_bad),
        "bad_bone_count_after_projection": int(after_bad),
        "global_ground_z_shift_m": float(ground_shift),
    }


def build_synced_payloads(
    poses: dict[str, dict[str, Any]],
    video_paths: dict[str, Path],
    output_dir: Path,
    reference_id: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    events = {}
    for cam_id, payload in poses.items():
        events[cam_id] = detect_clap_events(cam_id, video_paths[cam_id], payload, output_dir)
        print(
            f"[SYNC-EVENT] {cam_id}: start={events[cam_id]['start_frame']} "
            f"end={events[cam_id]['end_frame']}"
        )
    # The imported triangulation module uses this global in detect helpers.
    import camera_system.camera_calibration.charuco.calib_charuco_v2.run_demo_pose_sync_triangulation as old

    old.sync_source_events = events
    sync = build_sync(events, reference_id)
    report = {"events": events, "sync_to_reference": sync, "reference": reference_id}
    (output_dir / "auto_sync_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return sync, report


def write_synced_videos(video_paths: dict[str, Path], poses: dict[str, dict[str, Any]], sync: dict[str, Any], events: dict[str, Any], reference_id: str, output_dir: Path) -> None:
    ref_fps = float(poses[reference_id]["metadata"]["fps"])
    ref_start = int(round(events[reference_id]["start_time_s"] * ref_fps))
    ref_end = int(round(events[reference_id]["end_time_s"] * ref_fps))
    writers_info = []
    for cam_id, video_path in video_paths.items():
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or poses[cam_id]["metadata"]["width"])
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or poses[cam_id]["metadata"]["height"])
        writer = cv2.VideoWriter(str(output_dir / f"{cam_id}_synced.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), ref_fps, (width, height))
        for ref_frame in range(ref_start, ref_end + 1):
            ref_time = ref_frame / ref_fps
            local_time = (ref_time - sync[cam_id]["offset_s"]) / sync[cam_id]["scale"]
            local_frame = int(round(local_time * float(poses[cam_id]["metadata"]["fps"])))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, local_frame))
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, f"{cam_id} src_f={local_frame}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
        cap.release()
        writer.release()
        writers_info.append((cam_id, width, height))

    # 2x2 contact sheet video for quick sync review.
    caps = {cam_id: cv2.VideoCapture(str(output_dir / f"{cam_id}_synced.mp4")) for cam_id in video_paths}
    thumbs = []
    for cam_id, cap in caps.items():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
        thumbs.append((cam_id, w, h))
    tile_w, tile_h = 640, 360
    count = int(caps[reference_id].get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = cv2.VideoWriter(str(output_dir / "synced_4cam_2x2_review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), ref_fps, (tile_w * 2, tile_h * 2))
    for _ in range(count):
        canvas = np.zeros((tile_h * 2, tile_w * 2, 3), dtype=np.uint8)
        for idx, cam_id in enumerate(video_paths):
            ok, fr = caps[cam_id].read()
            if not ok:
                fr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            fr = cv2.resize(fr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            y = (idx // 2) * tile_h
            x = (idx % 2) * tile_w
            canvas[y : y + tile_h, x : x + tile_w] = fr
        writer.write(canvas)
    writer.release()
    for cap in caps.values():
        cap.release()


def run_robust(
    poses: dict[str, dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    sync: dict[str, Any],
    events: dict[str, Any],
    reference_id: str,
    output_dir: Path,
    min_conf: float,
    ransac_px: float,
    fast_single_person: bool = False,
    write_intermediates: bool = True,
) -> dict[str, Any]:
    ref_fps = float(poses[reference_id]["metadata"]["fps"])
    ref_start = int(round(events[reference_id]["start_time_s"] * ref_fps))
    ref_end = int(round(events[reference_id]["end_time_s"] * ref_fps))
    cam_ids = list(poses.keys())
    synced_frames_by_cam = {cam_id: [] for cam_id in cam_ids}
    kps_by_cam = {cam_id: [] for cam_id in cam_ids}
    source_frames = []

    for out_f, ref_frame in enumerate(range(ref_start, ref_end + 1)):
        ref_time = ref_frame / ref_fps
        src = {}
        for cam_id in cam_ids:
            local_fps = float(poses[cam_id]["metadata"]["fps"])
            local_time = (ref_time - sync[cam_id]["offset_s"]) / sync[cam_id]["scale"]
            local_frame = int(round(local_time * local_fps))
            src[cam_id] = local_frame
            person = sampled_person(poses[cam_id], local_frame)
            kps_by_cam[cam_id].append(kpt_array(person))
            synced_frames_by_cam[cam_id].append(
                {"frame": out_f, "source_frame": local_frame, "people": [{"person_id": 0, "keypoints": person.get("keypoints", [])}] if person else []}
            )
        source_frames.append(src)

    if write_intermediates:
        for cam_id, frames in synced_frames_by_cam.items():
            payload = {"metadata": {**poses[cam_id].get("metadata", {}), "synced_to": reference_id, "sync": sync[cam_id]}, "keypoint_names": KEYPOINT_NAMES, "frames": frames}
            (output_dir / f"keypoints_{cam_id}_synced_robust_input.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    kps_by_cam = {k: np.asarray(v, dtype=np.float64) for k, v in kps_by_cam.items()}
    n = len(source_frames)
    if fast_single_person:
        # Keep pose-level rejection of one globally bad view, which is important
        # around self-occlusion, but skip all six two-camera states.  The validated
        # single-person sequence only selected the four-view state or one of these
        # leave-one-out states, so this cuts 11 candidates/frame to five without
        # removing the quality safeguard that made the reconstruction stable.
        states = [tuple(cam_ids)]
        states.extend(tuple(c for c in cam_ids if c != excluded) for excluded in cam_ids)
    else:
        states = []
        for r in (4, 3, 2):
            for s in combinations(cam_ids, r):
                states.append(tuple(s))
    cands = []
    for f in range(n):
        row = [pose_candidate(f, s, kps_by_cam, cameras, min_conf=min_conf, ransac_px=ransac_px) for s in states]
        cands.append(row)
        if f % 50 == 0:
            print(f"[ROBUST] frame {f}/{n} valid states={sum(c is not None for c in row)}", flush=True)

    S = len(states)
    INF = 1e12
    dp = np.full((n, S), INF, dtype=np.float64)
    prev = np.full((n, S), -1, dtype=np.int32)
    for si, c in enumerate(cands[0]):
        if c is not None:
            dp[0, si] = c["score"]
    for f in range(1, n):
        for si, c in enumerate(cands[f]):
            if c is None:
                continue
            best = (INF, -1)
            for pj, pc in enumerate(cands[f - 1]):
                if pc is None or dp[f - 1, pj] >= INF:
                    continue
                trans = 0.0
                if states[si] != states[pj]:
                    jac = len(set(states[si]) & set(states[pj])) / len(set(states[si]) | set(states[pj]))
                    trans += 12.0 * (1.0 - jac) + 3.0
                if c["root"] is not None and pc["root"] is not None:
                    trans += min(120.0, float(np.linalg.norm(c["root"] - pc["root"])) * 75.0)
                both = c["valid"] & pc["valid"]
                if both.sum() >= 6:
                    md = float(np.median(np.linalg.norm(c["pts"][both] - pc["pts"][both], axis=1)))
                    trans += min(80.0, md * 25.0)
                val = dp[f - 1, pj] + c["score"] + trans
                if val < best[0]:
                    best = (val, pj)
            dp[f, si] = best[0]
            prev[f, si] = best[1]
    last = int(np.argmin(dp[-1]))
    path = [last]
    for f in range(n - 1, 0, -1):
        last = int(prev[f, last])
        if last < 0:
            last = int(np.argmin(dp[f - 1]))
        path.append(last)
    path = path[::-1]

    raw_pts = np.full((n, 17, 3), np.nan, dtype=np.float64)
    valid = np.zeros((n, 17), dtype=bool)
    meta = []
    for f, si in enumerate(path):
        c = cands[f][si]
        if c is None:
            choices = [x for x in cands[f] if x is not None]
            c = min(choices, key=lambda x: x["score"]) if choices else None
        if c is None:
            meta.append({"subset": []})
            continue
        raw_pts[f] = c["pts"]
        valid[f] = c["valid"]
        meta.append({
            "subset": list(c["effective_subset"]),
            "candidate_subset": list(c["subset"]),
            "joint_meta": c["joint_meta"],
            "score": c["score"],
            "med_err": c["med_err"],
            "p90_err": c["p90_err"],
            "bad_bones": c["bad_bones"],
        })

    gated_pts, gated_valid, temporal_gate_metrics = temporal_joint_gate(raw_pts, valid)
    smoothed = smooth_sequence(gated_pts, gated_valid, window=9)
    optimized, opt_metrics = skeleton_optimize(smoothed, np.isfinite(smoothed[:, :, 0]), max_nfev=50)
    opt_metrics.update(temporal_gate_metrics)

    frames = []
    renderer_frames = []
    subset_counts: dict[str, int] = {}
    valid_counts = []
    bad_counts = []
    root_vals = []
    for f in range(n):
        subset = meta[f].get("subset", [])
        candidate_subset = meta[f].get("candidate_subset", [])
        frame_joint_meta = meta[f].get("joint_meta", [None] * 17)
        subset_counts["+".join(subset)] = subset_counts.get("+".join(subset), 0) + 1
        k3 = []
        joints = []
        for j in range(17):
            X = optimized[f, j]
            ok = bool(np.isfinite(X).all())
            if ok:
                joint_info = frame_joint_meta[j] if j < len(frame_joint_meta) else None
                actual_views = (
                    list(joint_info.get("inlier_views", []))
                    if gated_valid[f, j] and isinstance(joint_info, dict)
                    else []
                )
                item = {
                    "id": j,
                    "name": KEYPOINT_NAMES[j],
                    "valid": True,
                    "position": [float(v) for v in X],
                    "num_views": len(actual_views),
                    "views": actual_views,
                    "review_imputed": not bool(actual_views),
                    "review_source": "hrnet_temporal2d_poselevel_joint_ransac_skeleton_opt",
                }
                joints.append({"id": j, "x": float(X[0]), "y": float(X[1]), "z": float(X[2]), "num_views": len(actual_views), "views": actual_views, "review_imputed": not bool(actual_views)})
            else:
                item = {"id": j, "name": KEYPOINT_NAMES[j], "valid": False, "position": [None, None, None], "num_views": 0}
            k3.append(item)
        valid_counts.append(len(joints))
        if np.isfinite(optimized[f, 11]).all() and np.isfinite(optimized[f, 12]).all():
            root_vals.append(0.5 * (optimized[f, 11] + optimized[f, 12]))
        bc = 0
        for a, b, lo, hi in BONES:
            if np.isfinite(optimized[f, a]).all() and np.isfinite(optimized[f, b]).all():
                L = float(np.linalg.norm(optimized[f, a] - optimized[f, b]))
                bc += int(L < lo or L > hi)
        bad_counts.append(bc)
        frames.append({"frame": f, "source_frames": source_frames[f], "selected_view_subset": subset, "candidate_view_subset": candidate_subset, "keypoints_3d": k3})
        renderer_frames.append({"frame": f, "source_frames": source_frames[f], "selected_view_subset": subset, "candidate_view_subset": candidate_subset, "identities": [{"identity_id": 0, "num_joints": len(joints), "joints": joints}]})

    root_vals_np = np.asarray(root_vals, dtype=np.float64) if root_vals else np.zeros((0, 3))
    summary = {
        "stage": "enhanced_demo_robust_pipeline",
        "triangulation_mode": "fast_single_person_pose_level_leave_one_out" if fast_single_person else "full_pose_level_subset_search",
        "num_frames": n,
        "valid_joints_per_frame_median": float(np.median(valid_counts)) if valid_counts else 0.0,
        "selected_subset_counts": subset_counts,
        "root_range_m": (np.nanmax(root_vals_np, axis=0) - np.nanmin(root_vals_np, axis=0)).tolist() if len(root_vals_np) else None,
        "root_std_m": np.nanstd(root_vals_np, axis=0).tolist() if len(root_vals_np) else None,
        "bad_bones_per_frame_median": float(np.median(bad_counts)) if bad_counts else None,
        "bad_bones_per_frame_p90": float(np.percentile(bad_counts, 90)) if bad_counts else None,
        "skeleton_optimization": opt_metrics,
    }
    raw_payload = {"metadata": summary, "keypoint_names": KEYPOINT_NAMES, "skeleton_connections": SKELETON_CONNECTIONS, "frames": frames, "summary": summary}
    ren_payload = {"metadata": {**summary, "converted_for": "visualize_triangulation"}, "frames": renderer_frames, "summary": summary}
    (output_dir / "triangulated_enhanced_robust.json").write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "triangulated_enhanced_robust_renderer_format.json").write_text(json.dumps(ren_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "enhanced_robust_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "enhanced_robust_triangulation", "summary": summary, "output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run enhanced robust 3D demo pipeline from per-camera 2D pose JSON.")
    ap.add_argument("--line-dir", default="camtest/line")
    ap.add_argument("--pose-jsons", nargs="+", required=True)
    ap.add_argument("--cam-ids", nargs="+", default=["cam1demo", "cam2demo", "cam3demo", "cam4demo"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reference", default="cam1demo")
    ap.add_argument("--swap-lr-cameras", nargs="*", default=["cam3demo", "cam4demo"])
    ap.add_argument("--min-conf", type=float, default=0.30)
    ap.add_argument("--ransac-reprojection-px", type=float, default=35.0)
    ap.add_argument("--skip-synced-videos", action="store_true")
    ap.add_argument("--refined-extrinsics-dir", default="", help="Optional directory containing camX_line_extrinsics_refined.npz files")
    ap.add_argument("--fixed-sync-report", default="", help="Use an existing arm-spread sync_report.json with frame offsets instead of auto event detection")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.pose_jsons) != len(args.cam_ids):
        raise ValueError("--pose-jsons and --cam-ids must have the same length")
    line_dir = Path(args.line_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = {}
    for cam_id, path in zip(args.cam_ids, args.pose_jsons):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if cam_id in set(args.swap_lr_cameras):
            payload = swap_payload_lr(payload)
        poses[cam_id] = payload
    video_paths = {cam_id: line_dir / f"{cam_id}.mkv" for cam_id in args.cam_ids}
    refined_dir = Path(args.refined_extrinsics_dir) if args.refined_extrinsics_dir else None
    def ext_path(cam_num: int, default_path: Path) -> Path:
        if refined_dir:
            candidate = refined_dir / f"cam{cam_num}_line_extrinsics_refined.npz"
            if candidate.exists():
                return candidate
        return default_path
    cameras = {
        "cam1demo": load_npz_camera(line_dir / "calib_out_cam1_hvflip/cam1_intrinsics.npz", ext_path(1, line_dir / "extrinsics_all_hvflip_retry/cam1_line_extrinsics.npz")),
        "cam2demo": load_npz_camera(line_dir / "calib_out_cam2_hvflip_recalc/cam2_intrinsics.npz", ext_path(2, line_dir / "extrinsics_all_hvflip_retry/cam2_line_extrinsics.npz")),
        "cam3demo": load_npz_camera(line_dir / "calib_out_cam3_hvflip_fast/cam3_intrinsics.npz", ext_path(3, line_dir / "extrinsics_all_hvflip_retry/cam3_line_extrinsics.npz")),
        "cam4demo": load_npz_camera(line_dir / "calib_out_cam4_hvflip_newintr_20260727/cam4_intrinsics.npz", ext_path(4, line_dir / "extrinsics_cam4_hvflip_newintr_20260727/cam4_line_extrinsics.npz")),
    }
    cameras = {k: cameras[k] for k in args.cam_ids}
    if args.fixed_sync_report:
        fixed = json.loads(Path(args.fixed_sync_report).read_text(encoding="utf-8"))
        ref_interval = fixed["reference_interval_frames"]
        ref_fps = float(poses[args.reference]["metadata"].get("fps", 30.0))
        sync = {}
        events = {}
        ref_start, ref_end = int(ref_interval[0]), int(ref_interval[1])
        if "affine_local_frame=a_ref_frame_plus_b" in fixed:
            affine = fixed["affine_local_frame=a_ref_frame_plus_b"]
            for cam_id in args.cam_ids:
                local_fps = float(poses[cam_id]["metadata"].get("fps", 30.0))
                a = float(affine[cam_id]["a"]); b = float(affine[cam_id]["b"])
                # local_frame = a * reference_frame + b.
                # Existing sampler uses local_time=(ref_time-offset_s)/scale, so:
                # local_frame = local_fps/ref_fps/scale * ref_frame - local_fps*offset_s/scale.
                scale = float(local_fps / (ref_fps * a))
                offset_s = float(-b * scale / local_fps)
                sync[cam_id] = {"scale": scale, "offset_s": offset_s, "affine_a_local_per_ref": a, "affine_b_frames": b}
                events[cam_id] = {
                    "start_frame": int(round(a * ref_start + b)),
                    "start_time_s": float((a * ref_start + b) / local_fps),
                    "end_frame": int(round(a * ref_end + b)),
                    "end_time_s": float((a * ref_end + b) / local_fps),
                }
            sync_mode = "affine_frame_sync"
        else:
            offsets = fixed["offsets_frames_local_minus_ref"]
            for cam_id in args.cam_ids:
                local_fps = float(poses[cam_id]["metadata"].get("fps", 30.0))
                offset_frames = int(offsets.get(cam_id, 0))
                sync[cam_id] = {"scale": 1.0, "offset_s": -float(offset_frames) / local_fps, "fixed_offset_frames_local_minus_ref": offset_frames}
                events[cam_id] = {
                    "start_frame": ref_start + offset_frames,
                    "start_time_s": float((ref_start + offset_frames) / local_fps),
                    "end_frame": ref_end + offset_frames,
                    "end_time_s": float((ref_end + offset_frames) / local_fps),
                }
            sync_mode = "fixed_frame_offset"
        sync_report = {"mode": sync_mode, "source": str(Path(args.fixed_sync_report).resolve()), "reference": args.reference, "events": events, "sync_to_reference": sync}
        (output_dir / "auto_sync_report.json").write_text(json.dumps(sync_report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        sync, sync_report = build_synced_payloads(poses, video_paths, output_dir, args.reference)
    if not args.skip_synced_videos:
        write_synced_videos(video_paths, poses, sync, sync_report["events"], args.reference, output_dir)
    run_robust(
        poses,
        cameras,
        sync,
        sync_report["events"],
        args.reference,
        output_dir,
        min_conf=float(args.min_conf),
        ransac_px=float(args.ransac_reprojection_px),
    )


if __name__ == "__main__":
    main()
