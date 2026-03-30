"""
real_world_pipeline.py

Two-camera 3D skeleton reconstruction pipeline.
Comments were rebuilt and legacy corrupted annotations were removed.
"""

import os
import cv2
import json
import glob
import copy
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
from scipy.optimize import least_squares
import itertools


                                                              
                                                              

CONFIG = {
    "cam1_video": "cam1video.mp4",
    "cam2_video": "cam2video.mp4",
    "cam1_intrinsics": "cam1_intrinsics.npz",
    "cam1_extrinsics": "cam1_extrinsics.npz",
    "cam2_intrinsics": "cam2_intrinsics.npz",
    "cam2_extrinsics": "cam2_extrinsics.npz",
    "yolo_model": "yolo11l-pose.pt",
    "confidence_threshold": 0.35,
    "ransac_reproj_threshold": 25.0,
    "output_dir": "real_world_output",
    "output_fps": -1,

                                                                       
    "keypoints_cam1_json": "",
    "keypoints_cam2_json": "",

                                                                        
    "frame_shift_cam1_minus_cam2": 0,
    "time_shift_report": "",
    "world_rotation": np.array([
        [-1, 0,  0],              
        [0, 0,  1],              
        [0, -1, 0],                                         
    ], dtype=np.float64),
}


                                                              
                                                              

def load_camera_calibration(intrin_path, extrin_path):
    
    intrin = np.load(intrin_path)
    keys_intrin = list(intrin.keys())
    print(f"  [Intrinsics] keys: {keys_intrin}")
    K = None
    for k in ['mtx', 'camera_matrix', 'K', 'intrinsic_matrix', 'cameraMatrix']:
        if k in intrin:
            K = np.array(intrin[k], dtype=np.float64)
            print(f"    K from key='{k}', shape={K.shape}")
            break
    if K is None:
        for k in keys_intrin:
            arr = np.array(intrin[k])
            if arr.shape == (3, 3):
                K = arr.astype(np.float64)
                print(f"    K guessed from key='{k}'")
                break
    if K is None:
        raise ValueError(f"Cannot find K in {intrin_path}. Keys: {keys_intrin}")
    dist = None
    for k in ['dist', 'dist_coeffs', 'distCoeffs', 'distortion_coefficients', 'd']:
        if k in intrin:
            dist = np.array(intrin[k], dtype=np.float64).flatten()
            print(f"    dist from key='{k}', shape={dist.shape}")
            break
    if dist is None:
        print("    [Warning] dist not found, assuming zero.")
        dist = np.zeros(5)
    extrin = np.load(extrin_path)
    keys_extrin = list(extrin.keys())
    print(f"  [Extrinsics] keys: {keys_extrin}")
    R = None
    for k in ['rvec', 'rvecs', 'rotation_vector', 'R', 'rotation_matrix']:
        if k in extrin:
            arr = np.array(extrin[k], dtype=np.float64)
            if arr.flatten().shape == (3,):
                R, _ = cv2.Rodrigues(arr.flatten())
                print(f"    R (from rvec) key='{k}'")
            elif arr.shape == (3, 3):
                R = arr
                print(f"    R (matrix) key='{k}'")
            if R is not None:
                break
    if R is None:
        raise ValueError(f"Cannot find R in {extrin_path}. Keys: {keys_extrin}")
    t = None
    for k in ['tvec', 'tvecs', 't', 'translation', 'T', 'translation_vector']:
        if k in extrin:
            t = np.array(extrin[k], dtype=np.float64).flatten().reshape(3, 1)
            print(f"    t from key='{k}', value={t.flatten()}")
            break
    if t is None:
        raise ValueError(f"Cannot find t in {extrin_path}. Keys: {keys_extrin}")
    P = K @ np.hstack([R, t])
    return K, dist, R, t, P


                                                              
                                                              

def project_point(point_3d, P):
    
    h = P @ np.append(point_3d, 1.0)
    return h[:2] / h[2]


def reprojection_error(point_3d, Ps, obs):
    
    residuals = []
    for P, o in zip(Ps, obs):
        residuals.extend(project_point(point_3d, P) - np.array(o))
    return np.array(residuals)


def svd_triangulate(Ps, pts):
    
    A = np.zeros((len(Ps) * 2, 4))
    for i, (P, (x, y)) in enumerate(zip(Ps, pts)):
        A[2*i]     = x * P[2] - P[0]
        A[2*i + 1] = y * P[2] - P[1]
    _, _, Vh = np.linalg.svd(A)
    X = Vh[-1]
    return X[:3] / X[3]


def triangulate_ransac_refine(projections, points, reproj_threshold=25.0):
    
    n = len(projections)

    if n < 2:
        return None

    if n == 2:
        initial = svd_triangulate(projections, points)
        if np.any(np.isnan(initial)) or np.any(np.isinf(initial)):
            return None
        result = least_squares(
            reprojection_error, initial,
            args=(projections, points),
            method='lm'
        )
        return result.x
    best_inliers = []
    best_pt = None
    max_inliers = -1
    min_err = float('inf')

    for i1, i2 in itertools.combinations(range(n), 2):
        cand = svd_triangulate(
            [projections[i1], projections[i2]],
            [points[i1], points[i2]]
        )
        if np.any(np.isnan(cand)) or np.any(np.isinf(cand)):
            continue

        inliers, err_sum = [], 0.0
        for i in range(n):
            err = np.linalg.norm(project_point(cand, projections[i]) - np.array(points[i]))
            if err < reproj_threshold:
                inliers.append(i)
                err_sum += err

        cnt = len(inliers)
        if cnt > max_inliers or (cnt == max_inliers and err_sum < min_err):
            max_inliers = cnt
            min_err = err_sum
            best_inliers = inliers
            best_pt = cand

    if best_pt is None or len(best_inliers) < 2:
        return svd_triangulate(projections, points)

    fin_P   = [projections[i] for i in best_inliers]
    fin_pts = [points[i]      for i in best_inliers]
    result = least_squares(
        reprojection_error, best_pt,
        args=(fin_P, fin_pts),
        method='lm'
    )
    return result.x


                                                              
                                                              

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]


def run_yolo_on_videos(cam1_path, cam2_path, model_path, conf_thresh, output_dir):
    
    model = YOLO(model_path)
    print(f"[YOLO] Model loaded: {model_path}")

    results_dict = {}
    for cam_id, video_path in [("cam1", cam1_path), ("cam2", cam2_path)]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\n[YOLO] Processing {cam_id}: {video_path}")
        print(f"       {total} frames, {fps:.1f} fps, {w}x{h}")
        out_video_path = os.path.join(output_dir, f"yolo_{cam_id}.mp4")
        writer = cv2.VideoWriter(
            out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
        )

        frames_data = []
        fid = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False, conf=conf_thresh)
            fdata = {"frame": fid, "people": []}

            if results[0].keypoints is not None:
                kpts_tensor = results[0].keypoints.data.cpu().numpy()                     
                for pid, person_kpts in enumerate(kpts_tensor):
                    person_data = {
                        "person_id": pid,
                        "keypoints": [
                            {"id": i, "x": float(kp[0]), "y": float(kp[1]), "confidence": float(kp[2])}
                            for i, kp in enumerate(person_kpts)
                        ]
                    }
                    fdata["people"].append(person_data)

            frames_data.append(fdata)
            annotated = results[0].plot()
            writer.write(annotated)

            if fid % 30 == 0:
                print(f"  Frame {fid}/{total} ...", end='\r')
            fid += 1

        cap.release()
        writer.release()
        print(f"\n  Done. {fid} frames -> {out_video_path}")
        results_dict[cam_id] = {
            "metadata": {"fps": fps, "width": w, "height": h, "total_frames": fid},
            "keypoint_names": KEYPOINT_NAMES,
            "frames": frames_data
        }

    return results_dict


                                                              
                                                              

def reconstruct_3d(yolo_results, projection_matrices, conf_thresh, reproj_threshold, cfg=None):
    
    cam_ids  = list(yolo_results.keys())                         
    n_frames = min(len(yolo_results[c]["frames"]) for c in cam_ids)
    print(f"\n[Triangulation] Processing {n_frames} frames with {len(cam_ids)} cameras...")

    all_frames_3d = []
    cam_label = {c: i for i, c in enumerate(cam_ids)}

    for fid in range(n_frames):
        if fid % 30 == 0:
            print(f"  Frame {fid}/{n_frames} ...", end='\r')

        frame_res = {"frame": fid, "keypoints_3d": [], "valid_count": 0}
        cam_kpts = {}
        for cid in cam_ids:
            people = yolo_results[cid]["frames"][fid]["people"]
            if people:
                cam_kpts[cid] = people[0]["keypoints"]

        if len(cam_kpts) < 2:
            all_frames_3d.append(frame_res)
            continue
        for jid in range(len(KEYPOINT_NAMES)):
            Ps, pts_2d, cams_used = [], [], []
            for cid, kpts in cam_kpts.items():
                if jid < len(kpts):
                    kp = kpts[jid]
                    if kp["confidence"] >= conf_thresh:
                        Ps.append(projection_matrices[cid])
                        pts_2d.append([kp["x"], kp["y"]])
                        cams_used.append(cid)

            if len(Ps) >= 2:
                pt3d = triangulate_ransac_refine(Ps, pts_2d, reproj_threshold)
                if pt3d is not None and not np.any(np.isnan(pt3d)):
                    world_R = cfg.get("world_rotation") if cfg else None
                    if world_R is not None:
                        pt3d = world_R @ pt3d
                    frame_res["keypoints_3d"].append({

                        "id": jid,
                        "name": KEYPOINT_NAMES[jid],
                        "position": pt3d.tolist(),
                        "valid": True,
                        "cameras_used": cams_used
                    })
                    frame_res["valid_count"] += 1
                    continue
            frame_res["keypoints_3d"].append({
                "id": jid,
                "name": KEYPOINT_NAMES[jid],
                "position": None,
                "valid": False
            })

        all_frames_3d.append(frame_res)

    print(f"\n  Done.")
    return all_frames_3d


                                                              
                                                              

def visualize_3d_skeleton(frame_data, output_dir, axis_limits):
    

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    kpts = frame_data["keypoints_3d"]
    xs, ys, zs_raw, valid_ids = [], [], [], []

    for kpt in kpts:
        if kpt["valid"]:
            p = kpt["position"]
            xs.append(p[0])
            ys.append(p[1])
            zs_raw.append(p[2])
            valid_ids.append(kpt["id"])

    if not xs:
        plt.close(fig)
        return
    zs = list(zs_raw)

    ax.scatter(xs, ys, zs, c='red', s=30, zorder=5)

    for i1, i2 in SKELETON_CONNECTIONS:
        if i1 in valid_ids and i2 in valid_ids:
            p1 = kpts[i1]["position"]
            p2 = kpts[i2]["position"]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],                                    
                'b-', lw=2, alpha=0.7
            )

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (up)')
    ax.set_title(f"Frame {frame_data['frame']:04d}  |  valid: {frame_data['valid_count']}/17")
    cx, cy, cz, r = axis_limits
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)

    ax.view_init(elev=15, azim=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'frame_{frame_data["frame"]:04d}.png'), dpi=80)
    plt.close(fig)


def frames_to_video(frames_dir, output_path, fps):
    images = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not images:
        print("[Video] No frames found, skipping video synthesis.")
        return
    img0 = cv2.imread(images[0])
    h, w = img0.shape[:2]
    out  = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for p in images:
        out.write(cv2.imread(p))
    out.release()
    print(f"[Video] Saved: {output_path}  ({len(images)} frames @ {fps:.1f} fps)")


                                                              
                          
                                                              

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam1-video", default="")
    ap.add_argument("--cam2-video", default="")
    ap.add_argument("--cam1-intrinsics", default="")
    ap.add_argument("--cam1-extrinsics", default="")
    ap.add_argument("--cam2-intrinsics", default="")
    ap.add_argument("--cam2-extrinsics", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--yolo-model", default="")
    ap.add_argument("--confidence-threshold", type=float, default=None)
    ap.add_argument("--ransac-reproj-threshold", type=float, default=None)
    ap.add_argument("--output-fps", type=float, default=None)
    ap.add_argument("--keypoints-cam1-json", default="")
    ap.add_argument("--keypoints-cam2-json", default="")
    ap.add_argument("--frame-shift-cam1-minus-cam2", type=int, default=None)
    ap.add_argument("--time-shift-report", default="")
    return ap.parse_args()


def build_runtime_config(args):
    cfg = dict(CONFIG)

    if args.cam1_video:
        cfg["cam1_video"] = args.cam1_video
    if args.cam2_video:
        cfg["cam2_video"] = args.cam2_video
    if args.cam1_intrinsics:
        cfg["cam1_intrinsics"] = args.cam1_intrinsics
    if args.cam1_extrinsics:
        cfg["cam1_extrinsics"] = args.cam1_extrinsics
    if args.cam2_intrinsics:
        cfg["cam2_intrinsics"] = args.cam2_intrinsics
    if args.cam2_extrinsics:
        cfg["cam2_extrinsics"] = args.cam2_extrinsics
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.yolo_model:
        cfg["yolo_model"] = args.yolo_model
    if args.confidence_threshold is not None:
        cfg["confidence_threshold"] = float(args.confidence_threshold)
    if args.ransac_reproj_threshold is not None:
        cfg["ransac_reproj_threshold"] = float(args.ransac_reproj_threshold)
    if args.output_fps is not None:
        cfg["output_fps"] = float(args.output_fps)

    if args.keypoints_cam1_json:
        cfg["keypoints_cam1_json"] = args.keypoints_cam1_json
    if args.keypoints_cam2_json:
        cfg["keypoints_cam2_json"] = args.keypoints_cam2_json

    if args.frame_shift_cam1_minus_cam2 is not None:
        cfg["frame_shift_cam1_minus_cam2"] = int(args.frame_shift_cam1_minus_cam2)
    if args.time_shift_report:
        cfg["time_shift_report"] = args.time_shift_report

    return cfg


def load_keypoints_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "frames" not in data:
        raise ValueError(f"Invalid keypoints JSON (missing frames): {path}")
    return data


def resolve_frame_shift(cfg):
    shift = int(cfg.get("frame_shift_cam1_minus_cam2", 0))
    report_path = cfg.get("time_shift_report", "")

    if report_path:
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"time_shift_report not found: {report_path}")
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        rep_shift = int(rep.get("selected_shift", 0))

        if shift == 0:
            shift = rep_shift
            src = f"report:{report_path}"
        else:
            src = f"config(overrides report:{report_path})"
    else:
        src = "config"

    return shift, src


def _reindex_frames(frames_slice):
    out = []
    for i, fr in enumerate(frames_slice):
        new_fr = copy.deepcopy(fr)
        old_idx = int(new_fr.get("frame", i))
        new_fr["orig_frame"] = old_idx
        new_fr["frame"] = i
        out.append(new_fr)
    return out


def align_results_by_shift(yolo_results, shift):
                                             
    if shift == 0:
        n = min(len(yolo_results["cam1"]["frames"]), len(yolo_results["cam2"]["frames"]))
        return yolo_results, {
            "shift": 0,
            "convention": "cam1_index = cam2_index + shift",
            "overlap_frames": n,
            "cam1_start": 0,
            "cam2_start": 0,
        }

    f1 = yolo_results["cam1"]["frames"]
    f2 = yolo_results["cam2"]["frames"]

    if shift >= 0:
        s1, s2 = shift, 0
    else:
        s1, s2 = 0, -shift

    n = min(len(f1) - s1, len(f2) - s2)
    if n <= 0:
        raise ValueError(
            f"No overlap after applying shift={shift}. "
            f"cam1_frames={len(f1)}, cam2_frames={len(f2)}"
        )

    out = {}
    out["cam1"] = copy.deepcopy(yolo_results["cam1"])
    out["cam2"] = copy.deepcopy(yolo_results["cam2"])

    out["cam1"]["frames"] = _reindex_frames(f1[s1:s1 + n])
    out["cam2"]["frames"] = _reindex_frames(f2[s2:s2 + n])

    for cid in ["cam1", "cam2"]:
        meta = dict(out[cid].get("metadata", {}))
        meta["total_frames"] = int(n)
        meta["time_aligned"] = True
        meta["time_shift_cam1_minus_cam2"] = int(shift)
        out[cid]["metadata"] = meta

    info = {
        "shift": int(shift),
        "convention": "cam1_index = cam2_index + shift",
        "overlap_frames": int(n),
        "cam1_start": int(s1),
        "cam1_end": int(s1 + n - 1),
        "cam2_start": int(s2),
        "cam2_end": int(s2 + n - 1),
    }
    return out, info


                                                              
       
                                                              

def load_cameras_and_projection(cfg, out_dir):
    print("\n[Step 1] Loading camera calibration...")
    cameras = {}
    for cam_id, intrin_key, extrin_key in [
        ("cam1", "cam1_intrinsics", "cam1_extrinsics"),
        ("cam2", "cam2_intrinsics", "cam2_extrinsics"),
    ]:
        print(f"\n  Camera: {cam_id}")
        K, dist, R, t, P = load_camera_calibration(cfg[intrin_key], cfg[extrin_key])
        cameras[cam_id] = {"K": K, "dist": dist, "R": R, "t": t, "P": P}
        print(f"  K =\n{K}")
        print(f"  t = {t.flatten()}")

    proj_mats = {cid: cameras[cid]["P"] for cid in cameras}

    calib_summary = {}
    for cid, c in cameras.items():
        calib_summary[cid] = {
            "K": c["K"].tolist(),
            "dist": c["dist"].tolist(),
            "R": c["R"].tolist(),
            "t": c["t"].flatten().tolist(),
            "P": c["P"].tolist(),
        }

    with open(os.path.join(out_dir, "camera_params.json"), "w", encoding="utf-8") as f:
        json.dump(calib_summary, f, indent=2)
    print(f"\n  Saved camera_params.json to {out_dir}/")

    return cameras, proj_mats


def load_or_run_keypoints(cfg, out_dir):
    if cfg.get("keypoints_cam1_json") and cfg.get("keypoints_cam2_json"):
        print("\n[Step 2] Loading precomputed keypoints...")
        yolo_results = {
            "cam1": load_keypoints_json(cfg["keypoints_cam1_json"]),
            "cam2": load_keypoints_json(cfg["keypoints_cam2_json"]),
        }
        print(f"  cam1 keypoints: {cfg['keypoints_cam1_json']}")
        print(f"  cam2 keypoints: {cfg['keypoints_cam2_json']}")
        raw_yolo_results = None
    else:
        print("\n[Step 2] Running YOLO pose estimation...")
        yolo_results = run_yolo_on_videos(
            cfg["cam1_video"],
            cfg["cam2_video"],
            cfg["yolo_model"],
            cfg["confidence_threshold"],
            out_dir,
        )
        raw_yolo_results = copy.deepcopy(yolo_results)

    return yolo_results, raw_yolo_results


def apply_time_alignment_and_save(yolo_results, cfg, out_dir):
    shift, shift_src = resolve_frame_shift(cfg)
    if shift != 0:
        print(f"\n[Step 2.5] Applying time alignment shift={shift} from {shift_src}")
        yolo_results, align_info = align_results_by_shift(yolo_results, shift)
        align_path = os.path.join(out_dir, "time_alignment.json")
        with open(align_path, "w", encoding="utf-8") as f:
            json.dump(align_info, f, indent=2)
        print(f"  Overlap frames: {align_info['overlap_frames']}")
        print(
            f"  cam1 [{align_info['cam1_start']}..{align_info['cam1_end']}], "
            f"cam2 [{align_info['cam2_start']}..{align_info['cam2_end']}]"
        )
        print(f"  Saved {align_path}")
    else:
        print("\n[Step 2.5] Time alignment shift=0 (disabled)")

    return yolo_results, shift


def save_keypoint_outputs(yolo_results, out_dir, raw_yolo_results=None, shift=0):
    for cid, data in yolo_results.items():
        kp_path = os.path.join(out_dir, f"keypoints_{cid}.json")
        with open(kp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"  Saved {kp_path}")

    if raw_yolo_results is not None and shift != 0:
        for cid, data in raw_yolo_results.items():
            kp_raw_path = os.path.join(out_dir, f"keypoints_{cid}_raw.json")
            with open(kp_raw_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"  Saved {kp_raw_path}")


def run_triangulation_and_save(yolo_results, proj_mats, cfg, out_dir):
    print("\n[Step 3] Triangulation (RANSAC + LM)...")
    all_frames_3d = reconstruct_3d(
        yolo_results,
        proj_mats,
        cfg["confidence_threshold"],
        cfg["ransac_reproj_threshold"],
        cfg=cfg,
    )

    skeleton_path = os.path.join(out_dir, "skeleton_3d.json")
    with open(skeleton_path, "w", encoding="utf-8") as f:
        json.dump({"frames": all_frames_3d}, f, indent=2)
    print(f"  Saved {skeleton_path}")

    return all_frames_3d


def compute_fixed_axis(all_frames_3d):
    all_pos = []
    for frame_data in all_frames_3d:
        for k in frame_data["keypoints_3d"]:
            if k["valid"] and k["position"] is not None:
                p = k["position"]
                all_pos.append([p[0], p[1], -p[2]])

    if not all_pos:
        print("  Warning: no valid positions found, using default axis.")
        return (0.0, 0.0, 0.0, 1000.0)

    pos_arr = np.array(all_pos)
    medians = np.median(pos_arr, axis=0)
    stds = np.std(pos_arr, axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)

    mask = np.all(np.abs(pos_arr - medians) <= 3.0 * stds, axis=1)
    clean = pos_arr[mask]

    n_removed = len(pos_arr) - len(clean)
    print(f"  Outlier filtering: removed {n_removed}/{len(pos_arr)} points (>3?)")

    if len(clean) == 0:
        clean = pos_arr

    cx = float(np.median(clean[:, 0]))
    cy = float(np.median(clean[:, 1]))
    cz = float(np.median(clean[:, 2]))
    r = float(
        np.max(
            [
                clean[:, 0].max() - clean[:, 0].min(),
                clean[:, 1].max() - clean[:, 1].min(),
                clean[:, 2].max() - clean[:, 2].min(),
            ]
        )
    ) / 2.0 * 1.3
    r = max(r, 200.0)

    fixed_axis = (cx, cy, cz, r)
    print(f"  Fixed axis center=({cx:.1f}, {cy:.1f}, {cz:.1f}), radius={r:.1f}")
    return fixed_axis


def render_frames(all_frames_3d, frames_dir, fixed_axis):
    valid_count_total = 0
    for frame_data in all_frames_3d:
        valid_count_total += frame_data["valid_count"]

        if frame_data["valid_count"] > 0:
            visualize_3d_skeleton(frame_data, frames_dir, fixed_axis)

        if frame_data["frame"] % 30 == 0:
            print(f"  Frame {frame_data['frame']} ...", end="\r")

    print(f"\n  Total valid joints across all frames: {valid_count_total}")
    return valid_count_total


def export_video_and_summary(out_dir, yolo_results, cfg):
    fps = cfg["output_fps"]
    if fps <= 0:
        fps = yolo_results["cam1"]["metadata"]["fps"]

    output_video = os.path.join(out_dir, "skeleton_3d.mp4")
    frames_to_video(os.path.join(out_dir, "frames"), output_video, fps)

    print("\n" + "=" * 60)
    print(f"  All done! Output saved to: {out_dir}/")
    print("    skeleton_3d.json    3D keypoints (all frames)")
    print("    skeleton_3d.mp4     3D skeleton visualization (fixed view)")
    print("    yolo_cam1.mp4       cam1 YOLO annotated video")
    print("    yolo_cam2.mp4       cam2 YOLO annotated video")
    print("    camera_params.json  camera calibration summary")
    print("=" * 60)


def main():
    args = parse_args()
    cfg = build_runtime_config(args)

    out_dir = cfg["output_dir"]
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("=" * 60)
    print("  Real-World 2-Camera 3D Skeleton Reconstruction")
    print("=" * 60)

    _, proj_mats = load_cameras_and_projection(cfg, out_dir)
    yolo_results, raw_yolo_results = load_or_run_keypoints(cfg, out_dir)
    yolo_results, shift = apply_time_alignment_and_save(yolo_results, cfg, out_dir)

    save_keypoint_outputs(
        yolo_results,
        out_dir,
        raw_yolo_results=raw_yolo_results,
        shift=shift,
    )

    all_frames_3d = run_triangulation_and_save(yolo_results, proj_mats, cfg, out_dir)

    print("\n[Step 4] Rendering 3D skeleton frames...")
    fixed_axis = compute_fixed_axis(all_frames_3d)
    render_frames(all_frames_3d, frames_dir, fixed_axis)

    export_video_and_summary(out_dir, yolo_results, cfg)


if __name__ == "__main__":
    main()
