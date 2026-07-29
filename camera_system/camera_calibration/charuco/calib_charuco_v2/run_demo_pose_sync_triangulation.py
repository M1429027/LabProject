import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter
from ultralytics import YOLO


KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def load_npz_camera(intrinsics_path: Path, extrinsics_path: Path):
    intr = np.load(str(intrinsics_path), allow_pickle=True)
    ext = np.load(str(extrinsics_path), allow_pickle=True)

    K = None
    for key in ("camera_matrix", "mtx", "K", "intrinsic_matrix"):
        if key in intr:
            K = np.asarray(intr[key], dtype=np.float64)
            break
    if K is None:
        raise KeyError(f"Cannot find camera matrix in {intrinsics_path}")

    dist = None
    for key in ("dist_coeffs", "dist", "distortion_coefficients", "d"):
        if key in intr:
            dist = np.asarray(intr[key], dtype=np.float64).reshape(-1, 1)
            break
    if dist is None:
        dist = np.zeros((5, 1), dtype=np.float64)

    if "R" in ext:
        R = np.asarray(ext["R"], dtype=np.float64)
    elif "rotation_matrix" in ext:
        R = np.asarray(ext["rotation_matrix"], dtype=np.float64)
    elif "rvec" in ext:
        R, _ = cv2.Rodrigues(np.asarray(ext["rvec"], dtype=np.float64).reshape(3))
    else:
        raise KeyError(f"Cannot find rotation in {extrinsics_path}")

    if "tvec" in ext:
        t = np.asarray(ext["tvec"], dtype=np.float64).reshape(3, 1)
    elif "t" in ext:
        t = np.asarray(ext["t"], dtype=np.float64).reshape(3, 1)
    elif "translation" in ext:
        t = np.asarray(ext["translation"], dtype=np.float64).reshape(3, 1)
    else:
        raise KeyError(f"Cannot find translation in {extrinsics_path}")

    P = K @ np.hstack([R, t])
    return {"K": K, "dist": dist, "R": R, "t": t, "P": P}


def process_video_pose(video_path: Path, model, output_dir: Path, conf: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    json_path = output_dir / f"keypoints_{stem}.json"
    annotated_path = output_dir / f"yolo_{stem}.mp4"
    if json_path.exists():
        print(f"[POSE] reuse {json_path}")
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frames = []
    frame_id = 0
    print(f"[POSE] {video_path.name}: {frame_count} frames, {fps:.2f} fps, {width}x{height}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model(frame, verbose=False, conf=conf)
        people = []
        if results and results[0].keypoints is not None:
            kpts = results[0].keypoints.data.cpu().numpy()
            boxes = None
            if results[0].boxes is not None and results[0].boxes.xyxy is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
            for person_id, person_kpts in enumerate(kpts):
                kp_list = [
                    {
                        "id": int(i),
                        "x": float(kp[0]),
                        "y": float(kp[1]),
                        "confidence": float(kp[2]),
                    }
                    for i, kp in enumerate(person_kpts)
                ]
                person = {"person_id": int(person_id), "keypoints": kp_list}
                if boxes is not None and person_id < len(boxes):
                    x1, y1, x2, y2 = boxes[person_id]
                    person["bbox"] = [float(x1), float(y1), float(x2), float(y2)]
                    person["bbox_area"] = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                people.append(person)
        frames.append({"frame": int(frame_id), "people": people})
        writer.write(results[0].plot() if results else frame)
        if frame_id % 30 == 0:
            print(f"  frame {frame_id}/{frame_count}", end="\r")
        frame_id += 1

    cap.release()
    writer.release()
    payload = {
        "metadata": {
            "source_video": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": frame_id,
            "model": getattr(model, "model_name", None),
            "confidence_threshold": conf,
        },
        "keypoint_names": KEYPOINT_NAMES,
        "frames": frames,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"\n[POSE] wrote {json_path}")
    return payload


def select_main_person(frame):
    people = frame.get("people", [])
    if not people:
        return None
    def score(person):
        area = float(person.get("bbox_area", 0.0))
        conf_sum = sum(float(k.get("confidence", 0.0)) for k in person.get("keypoints", []))
        return area + conf_sum * 1000.0
    return max(people, key=score)


def kpt_array(person):
    arr = np.zeros((17, 3), dtype=np.float64)
    arr[:, :] = np.nan
    if not person:
        return arr
    for kp in person.get("keypoints", []):
        idx = int(kp["id"])
        if 0 <= idx < 17:
            arr[idx] = [float(kp["x"]), float(kp["y"]), float(kp["confidence"])]
    return arr


def valid(k, idx, conf):
    return np.isfinite(k[idx, 0]) and k[idx, 2] >= conf


def clap_pose_signal(payload, min_conf=0.25):
    meta = payload["metadata"]
    width = float(meta["width"])
    height = float(meta["height"])
    frames = payload["frames"]
    n = len(frames)
    pose = np.full((n, 17, 3), np.nan, dtype=np.float64)
    for i, frame in enumerate(frames):
        pose[i] = kpt_array(select_main_person(frame))

    signal = np.zeros(n, dtype=np.float64)
    wrist_mid = np.full((n, 2), np.nan, dtype=np.float64)
    for i, k in enumerate(pose):
        if not (valid(k, 9, min_conf) and valid(k, 10, min_conf)):
            continue
        lw = k[9, :2]
        rw = k[10, :2]
        wrist_mid[i] = 0.5 * (lw + rw)

        refs_y = []
        for idx in (0, 5, 6):
            if valid(k, idx, min_conf):
                refs_y.append(k[idx, 1])
        if not refs_y:
            continue
        head_or_shoulder_y = min(refs_y)
        shoulder_width = width * 0.14
        if valid(k, 5, min_conf) and valid(k, 6, min_conf):
            shoulder_width = max(20.0, float(np.linalg.norm(k[5, :2] - k[6, :2])))

        wrist_y = min(lw[1], rw[1])
        hands_distance = float(np.linalg.norm(lw - rw))
        above = max(0.0, (head_or_shoulder_y - wrist_y + 0.12 * height) / (0.35 * height))
        close = max(0.0, 1.0 - hands_distance / max(35.0, 1.6 * shoulder_width))
        conf_term = 0.5 * (k[9, 2] + k[10, 2])
        signal[i] = above * 1.8 + close * 1.2 + conf_term * 0.7

    # Add local wrist motion. This helps pick the actual clap frame inside an overhead-hands segment.
    motion = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if np.isfinite(wrist_mid[i]).all() and np.isfinite(wrist_mid[i - 1]).all():
            motion[i] = float(np.linalg.norm(wrist_mid[i] - wrist_mid[i - 1])) / max(width, height)
    if np.max(motion) > 0:
        motion = motion / (np.percentile(motion[motion > 0], 95) + 1e-9)
        motion = np.clip(motion, 0.0, 2.0)
    signal = signal + 0.55 * motion

    if n >= 11:
        win = min(11, n // 2 * 2 - 1)
        if win >= 5:
            smooth = savgol_filter(signal, win, 2)
        else:
            smooth = signal
    else:
        smooth = signal
    return signal, np.asarray(smooth, dtype=np.float64), pose


def refine_with_motion(video_path: Path, event_frame: int, search_radius=8):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start = max(1, int(event_frame) - search_radius)
    end = min(n - 1, int(event_frame) + search_radius)
    prev = None
    best_frame = int(event_frame)
    best_score = -1.0
    for frame_id in range(start - 1, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        crop = frame[: int(h * 0.75), int(w * 0.08): int(w * 0.92)]
        gray = cv2.cvtColor(cv2.resize(crop, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev is not None and frame_id >= start:
            diff = cv2.absdiff(gray, prev)
            score = float(np.mean(np.clip(diff.astype(np.float32) - 4, 0, 255)))
            if score > best_score:
                best_score = score
                best_frame = frame_id
        prev = gray
    cap.release()
    return best_frame, best_score


def detect_clap_events(cam_id, video_path: Path, payload, output_dir: Path):
    fps = float(payload["metadata"]["fps"])
    n = len(payload["frames"])
    raw, smooth, pose = clap_pose_signal(payload)
    windows = {
        "start": (int(round(0.0 * fps)), min(n, int(round(8.0 * fps)))),
        "end": (max(0, n - int(round(9.0 * fps))), n),
    }
    picks = {}
    candidates = {}
    for name, (a, b) in windows.items():
        seg = smooth[a:b]
        if len(seg) == 0:
            raise RuntimeError(f"{cam_id}: empty {name} clap search window")
        peaks, props = find_peaks(seg, distance=max(3, int(0.35 * fps)), prominence=max(0.08, float(np.std(seg) * 0.4)))
        if len(peaks) == 0:
            peak = int(np.argmax(seg)) + a
            peak_list = [(peak, float(smooth[peak]))]
        else:
            peak_list = sorted([(int(p) + a, float(smooth[int(p) + a])) for p in peaks], key=lambda x: x[1], reverse=True)
        pose_frame = peak_list[0][0]
        motion_frame, motion_score = refine_with_motion(video_path, pose_frame, search_radius=int(round(0.25 * fps)))
        picks[name] = {
            "pose_frame": int(pose_frame),
            "pose_time_s": float(pose_frame / fps),
            "motion_refined_frame": int(motion_frame),
            "motion_refined_time_s": float(motion_frame / fps),
            "motion_score": float(motion_score),
            "score": float(smooth[pose_frame]),
        }
        candidates[name] = [
            {"frame": int(f), "time_s": float(f / fps), "score": float(s)}
            for f, s in peak_list[:8]
        ]

    np.savez(str(output_dir / f"{cam_id}_clap_signal.npz"), raw=raw, smooth=smooth)
    return {
        "start_frame": picks["start"]["motion_refined_frame"],
        "start_time_s": picks["start"]["motion_refined_time_s"],
        "end_frame": picks["end"]["motion_refined_frame"],
        "end_time_s": picks["end"]["motion_refined_time_s"],
        "details": picks,
        "candidates": candidates,
    }


def build_sync(events, reference_id):
    ref = events[reference_id]
    ref_start = ref["start_time_s"]
    ref_end = ref["end_time_s"]
    if ref_end <= ref_start:
        raise RuntimeError("reference end event must be after start event")
    sync = {}
    for cam_id, ev in events.items():
        start = ev["start_time_s"]
        end = ev["end_time_s"]
        if end <= start:
            raise RuntimeError(f"{cam_id}: end event must be after start event")
        scale = (ref_end - ref_start) / (end - start)
        offset = ref_start - scale * start
        sync[cam_id] = {
            "scale": float(scale),
            "offset_s": float(offset),
            "local_time = (reference_time - offset_s) / scale": True,
        }
    return sync


def undistort_observation(camera, x, y):
    pts = np.array([[[float(x), float(y)]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, camera["K"], camera["dist"], P=camera["K"])
    return und.reshape(2)


def project_point(point_3d, P):
    homog = P @ np.append(point_3d, 1.0)
    return homog[:2] / homog[2]


def reprojection_residual(point_3d, Ps, points):
    residual = []
    for P, pt in zip(Ps, points):
        residual.extend(project_point(point_3d, P) - pt)
    return np.asarray(residual, dtype=np.float64)


def svd_triangulate(Ps, points):
    A = np.zeros((len(Ps) * 2, 4), dtype=np.float64)
    for i, (P, (x, y)) in enumerate(zip(Ps, points)):
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def triangulate_joint(Ps, points):
    if len(Ps) < 2:
        return None
    initial = svd_triangulate(Ps, points)
    if initial is None or not np.isfinite(initial).all():
        return None
    try:
        res = least_squares(reprojection_residual, initial, args=(Ps, points), method="lm")
        if np.isfinite(res.x).all():
            return res.x
    except Exception:
        pass
    return initial


def sampled_person(payload, frame_index):
    frames = payload["frames"]
    if frame_index < 0 or frame_index >= len(frames):
        return None
    return select_main_person(frames[frame_index])


def triangulate_synced(poses, cameras, sync, reference_id, output_dir, min_conf=0.3):
    ref_meta = poses[reference_id]["metadata"]
    fps = float(ref_meta["fps"])
    events_start = max(ev["start_time_s"] for ev in sync_source_events.values())
    events_end = min(ev["end_time_s"] for ev in sync_source_events.values())
    # Better: use the reference clap-to-clap interval; each local frame is sampled through sync mapping.
    ref_start = int(round(sync_source_events[reference_id]["start_time_s"] * fps))
    ref_end = int(round(sync_source_events[reference_id]["end_time_s"] * fps))
    if ref_end <= ref_start:
        raise RuntimeError("invalid reference clip interval")

    frames_out = []
    per_camera_synced_keypoints = {cam_id: [] for cam_id in poses}
    total_points = 0
    valid_frames = 0
    for out_frame_id, ref_frame in enumerate(range(ref_start, ref_end + 1)):
        ref_time = ref_frame / fps
        frame_3d = []
        local_indices = {}
        observations_by_joint = {j: [] for j in range(17)}

        for cam_id, payload in poses.items():
            local_fps = float(payload["metadata"]["fps"])
            local_time = (ref_time - sync[cam_id]["offset_s"]) / sync[cam_id]["scale"]
            local_frame = int(round(local_time * local_fps))
            local_indices[cam_id] = local_frame
            person = sampled_person(payload, local_frame)
            k = kpt_array(person)
            people = []
            if person is not None:
                people = [{"person_id": 0, "keypoints": person.get("keypoints", [])}]
            per_camera_synced_keypoints[cam_id].append({"frame": out_frame_id, "source_frame": local_frame, "people": people})
            for j in range(17):
                if np.isfinite(k[j, 0]) and k[j, 2] >= min_conf:
                    xy = undistort_observation(cameras[cam_id], k[j, 0], k[j, 1])
                    observations_by_joint[j].append({
                        "cam_id": cam_id,
                        "point": xy,
                        "confidence": float(k[j, 2]),
                    })

        joints = []
        for joint_id in range(17):
            obs = observations_by_joint[joint_id]
            if len(obs) >= 2:
                Ps = [cameras[o["cam_id"]]["P"] for o in obs]
                pts = [o["point"] for o in obs]
                X = triangulate_joint(Ps, pts)
            else:
                X = None
            if X is not None and np.isfinite(X).all():
                total_points += 1
                joints.append({
                    "id": joint_id,
                    "name": KEYPOINT_NAMES[joint_id],
                    "valid": True,
                    "position": [float(v) for v in X],
                    "num_views": len(obs),
                    "views": [o["cam_id"] for o in obs],
                })
            else:
                joints.append({"id": joint_id, "name": KEYPOINT_NAMES[joint_id], "valid": False, "position": [None, None, None], "num_views": len(obs)})
        if any(j["valid"] for j in joints):
            valid_frames += 1
        frames_out.append({
            "frame": out_frame_id,
            "reference_frame": int(ref_frame),
            "reference_time_s": float(ref_time),
            "source_frames": {k: int(v) for k, v in local_indices.items()},
            "keypoints_3d": joints,
        })

    for cam_id, frames in per_camera_synced_keypoints.items():
        meta = dict(poses[cam_id]["metadata"])
        meta.update({"synced_to": reference_id, "sync": sync[cam_id]})
        with (output_dir / f"keypoints_{cam_id}_synced.json").open("w", encoding="utf-8") as handle:
            json.dump({"metadata": meta, "keypoint_names": KEYPOINT_NAMES, "frames": frames}, handle, ensure_ascii=False, indent=2)

    payload = {
        "metadata": {
            "stage": "demo_pose_sync_triangulation",
            "reference_camera": reference_id,
            "fps": fps,
            "min_confidence": min_conf,
        },
        "keypoint_names": KEYPOINT_NAMES,
        "skeleton_connections": SKELETON_CONNECTIONS,
        "frames": frames_out,
        "summary": {
            "num_output_frames": len(frames_out),
            "valid_frames": valid_frames,
            "total_triangulated_points": total_points,
        },
    }
    with (output_dir / "triangulated_3d.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with (output_dir / "triangulation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["summary"], handle, ensure_ascii=False, indent=2)
    return payload


def write_synced_videos(video_paths, poses, sync, events, reference_id, output_dir):
    ref_fps = float(poses[reference_id]["metadata"]["fps"])
    ref_start = int(round(events[reference_id]["start_time_s"] * ref_fps))
    ref_end = int(round(events[reference_id]["end_time_s"] * ref_fps))
    for cam_id, video_path in video_paths.items():
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or poses[cam_id]["metadata"]["width"])
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or poses[cam_id]["metadata"]["height"])
        writer = cv2.VideoWriter(
            str(output_dir / f"{cam_id}_synced.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            ref_fps,
            (width, height),
        )
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


def parse_cam_paths(base_dir):
    return {
        "cam1demo": {
            "video": base_dir / "cam1demo.mkv",
            "intr": base_dir / "calib_out_cam1_hvflip/cam1_intrinsics.npz",
            "ext": base_dir / "extrinsics_all_hvflip_retry/cam1_line_extrinsics.npz",
        },
        "cam2demo": {
            "video": base_dir / "cam2demo.mkv",
            "intr": base_dir / "calib_out_cam2_hvflip_recalc/cam2_intrinsics.npz",
            "ext": base_dir / "extrinsics_all_hvflip_retry/cam2_line_extrinsics.npz",
        },
        "cam3demo": {
            "video": base_dir / "cam3demo.mkv",
            "intr": base_dir / "calib_out_cam3_hvflip_fast/cam3_intrinsics.npz",
            "ext": base_dir / "extrinsics_all_hvflip_retry/cam3_line_extrinsics.npz",
        },
        "cam4demo": {
            "video": base_dir / "cam4demo.mkv",
            "intr": base_dir / "calib_out_cam4_hvflip/cam4_intrinsics.npz",
            "ext": base_dir / "extrinsics_all_hvflip_retry/cam4_line_extrinsics.npz",
        },
    }


sync_source_events = {}


def main():
    parser = argparse.ArgumentParser(description="Run demo videos through YOLO pose, auto clap sync, and 4-view triangulation.")
    parser.add_argument("--line-dir", default="camtest/line")
    parser.add_argument("--output-dir", default="camtest/line/demo_pose_sync_triangulation")
    parser.add_argument("--model", default="yolo11l-pose.pt")
    parser.add_argument("--reference", default="cam1demo")
    parser.add_argument("--pose-conf", type=float, default=0.35)
    parser.add_argument("--triangulation-conf", type=float, default=0.30)
    parser.add_argument("--skip-synced-videos", action="store_true")
    args = parser.parse_args()

    line_dir = Path(args.line_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = output_dir / "pose"
    cams = parse_cam_paths(line_dir)
    for cam_id, paths in cams.items():
        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{cam_id} {key} not found: {path}")

    print(f"[MODEL] loading {args.model}")
    model = YOLO(args.model)

    poses = {}
    cameras = {}
    video_paths = {}
    events = {}
    for cam_id, paths in cams.items():
        video_paths[cam_id] = paths["video"]
        cameras[cam_id] = load_npz_camera(paths["intr"], paths["ext"])
        poses[cam_id] = process_video_pose(paths["video"], model, pose_dir, conf=args.pose_conf)
        events[cam_id] = detect_clap_events(cam_id, paths["video"], poses[cam_id], output_dir)
        print(
            f"[SYNC-EVENT] {cam_id}: start f={events[cam_id]['start_frame']} "
            f"t={events[cam_id]['start_time_s']:.3f}s | end f={events[cam_id]['end_frame']} "
            f"t={events[cam_id]['end_time_s']:.3f}s"
        )

    global sync_source_events
    sync_source_events = events
    sync = build_sync(events, args.reference)

    sync_report = {
        "events": events,
        "sync_to_reference": sync,
        "reference": args.reference,
    }
    with (output_dir / "auto_sync_report.json").open("w", encoding="utf-8") as handle:
        json.dump(sync_report, handle, ensure_ascii=False, indent=2)

    if not args.skip_synced_videos:
        print("[VIDEO] writing synced videos")
        write_synced_videos(video_paths, poses, sync, events, args.reference, output_dir)

    print("[TRIANGULATION] running synced 4-view triangulation")
    triangulated = triangulate_synced(
        poses=poses,
        cameras=cameras,
        sync=sync,
        reference_id=args.reference,
        output_dir=output_dir,
        min_conf=args.triangulation_conf,
    )

    print("[DONE]")
    print(json.dumps({
        "output_dir": str(output_dir),
        "auto_sync_report": str(output_dir / "auto_sync_report.json"),
        "triangulated_3d": str(output_dir / "triangulated_3d.json"),
        "summary": triangulated["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
