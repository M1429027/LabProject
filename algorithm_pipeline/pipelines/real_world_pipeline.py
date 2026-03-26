"""
real_world_pipeline.py
======================
?祕銝??璈?3D 撉冽?遣 Pipeline

瘚?嚗?
  1. 霈??cam1_intrinsics.npz / cam2_intrinsics.npz  ???批? K, dist
  2. 霈??cam1_extrinsics.npz / cam2_extrinsics.npz  ??憭? R, t
  3. 蝯???Projection Matrix P = K @ [R | t]
  4. ??YOLOv11l-pose 撠?yolo_camera1.mp4 / yolo_camera2.mp4 ?葷?刻?
  5. RANSAC + LM 銝?皜祇? ??3D ??摨扳?
  6. 頛詨 skeleton_3d.json + 3D 撉冽?敶梁?
"""

import os
import cv2
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
from scipy.optimize import least_squares
import itertools


# ============================================================
#  閮剖??嚗?湔靽格嚗?
# ============================================================

CONFIG = {
    # 頛詨敶梁?
    "cam1_video": "cam1video.mp4",
    "cam2_video": "cam2video.mp4",

    # ?豢?璅?瑼?(.npz)
    "cam1_intrinsics": "cam1_intrinsics.npz",
    "cam1_extrinsics": "cam1_extrinsics.npz",
    "cam2_intrinsics": "cam2_intrinsics.npz",
    "cam2_extrinsics": "cam2_extrinsics.npz",

    # YOLO 璅∪?
    "yolo_model": "yolo11l-pose.pt",

    # YOLO ?刻?靽∪??瑼?
    "confidence_threshold": 0.35,

    # RANSAC ??敶梯炊撌桅?瑼?(??)
    "ransac_reproj_threshold": 25.0,

    # 頛詨?桅?
    "output_dir": "real_world_output",

    # 頛詨敶梁? FPS嚗身??-1 ??? cam1 霈??
    "output_fps": -1,

    # ??銝?摨扳?蝟餅?頧耨甇?
    # 銝?皜祇?敺??冽迨?拚嚗?撉冽?孵?甇?Ⅱ??
    # ?身嚗(頨恍?)?(銝?, Y?, Z?  嚗???澈擃窒X???膩嚗?
    # 憒???銝?嚗岫銝?嗡??賊?嚗?甈⊥銝????嚗?
    #   銝???嚗?憪撓?綽?: np.eye(3)
    #   頨恍???Y嚗霈?霈?Z: np.array([[1,0,0],[0,0,1],[0,-1,0]])
    #   蝯?憿?             ?函???蝷?撠?Z ?? -1
    #   蝯?撌血?∪?:         ?函???蝷?撠?X ?? -1
    "world_rotation": np.array([
        [-1, 0,  0],   # ??X = ??X
        [0, 0,  1],   # ??Y = ??Z
        [0, -1, 0],   # ??Z = -??Y  ??鞎?靽格迤銝?憿?
    ], dtype=np.float64),
}



# ============================================================
#  ?豢?璅?霈??
# ============================================================

def load_camera_calibration(intrin_path, extrin_path):
    """
    敺?.npz 霈?璈憭?嚗??? Projection Matrix??

    撌脩Ⅱ隤? key ?澆?嚗?
      intrinsics.npz : 'mtx' (3x3 K), 'dist' (distortion)
      extrinsics.npz : 'rvec' (Rodrigues), 'tvec' (translation)

    銋?游隞虜閬撘?蝔曹???fallback??
    """
    # --- ?批? ---
    intrin = np.load(intrin_path)
    keys_intrin = list(intrin.keys())
    print(f"  [Intrinsics] keys: {keys_intrin}")

    # ??K嚗??'mtx'嚗? fallback嚗?
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

    # ?曄霈??芸? 'dist'嚗?
    dist = None
    for k in ['dist', 'dist_coeffs', 'distCoeffs', 'distortion_coefficients', 'd']:
        if k in intrin:
            dist = np.array(intrin[k], dtype=np.float64).flatten()
            print(f"    dist from key='{k}', shape={dist.shape}")
            break
    if dist is None:
        print("    [Warning] dist not found, assuming zero.")
        dist = np.zeros(5)

    # --- 憭? ---
    extrin = np.load(extrin_path)
    keys_extrin = list(extrin.keys())
    print(f"  [Extrinsics] keys: {keys_extrin}")

    # ??R嚗??'rvec' Rodrigues vector嚗?
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

    # ??t嚗??'tvec'嚗?
    t = None
    for k in ['tvec', 'tvecs', 't', 'translation', 'T', 'translation_vector']:
        if k in extrin:
            t = np.array(extrin[k], dtype=np.float64).flatten().reshape(3, 1)
            print(f"    t from key='{k}', value={t.flatten()}")
            break
    if t is None:
        raise ValueError(f"Cannot find t in {extrin_path}. Keys: {keys_extrin}")

    # 蝯? P = K @ [R|t]
    P = K @ np.hstack([R, t])
    return K, dist, R, t, P


# ============================================================
#  RANSAC + LM 銝?皜祇?嚗??暹?蝔?靽?銝?湛?
# ============================================================

def project_point(point_3d, P):
    """撠?3D 暺?敶勗 2D"""
    h = P @ np.append(point_3d, 1.0)
    return h[:2] / h[2]


def reprojection_error(point_3d, Ps, obs):
    """Multi-view reprojection residuals."""
    residuals = []
    for P, o in zip(Ps, obs):
        residuals.extend(project_point(point_3d, P) - np.array(o))
    return np.array(residuals)


def svd_triangulate(Ps, pts):
    """?箇? DLT-SVD 銝?皜祇?"""
    A = np.zeros((len(Ps) * 2, 4))
    for i, (P, (x, y)) in enumerate(zip(Ps, pts)):
        A[2*i]     = x * P[2] - P[0]
        A[2*i + 1] = y * P[2] - P[1]
    _, _, Vh = np.linalg.svd(A)
    X = Vh[-1]
    return X[:3] / X[3]


def triangulate_ransac_refine(projections, points, reproj_threshold=25.0):
    """
    RANSAC + Levenberg-Marquardt 銝?皜祇?
    ???triangulation.py ?摩?詨?嚗?撠?2 ?豢??????
    """
    n = len(projections)

    if n < 2:
        return None

    if n == 2:
        # ?芣??拙?豢?嚗??SVD + LM ?芸?
        initial = svd_triangulate(projections, points)
        if np.any(np.isnan(initial)) or np.any(np.isinf(initial)):
            return None
        result = least_squares(
            reprojection_error, initial,
            args=(projections, points),
            method='lm'
        )
        return result.x

    # 銝隞乩?嚗ANSAC
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


# ============================================================
#  YOLO ?刻?嚗撟嚗?
# ============================================================

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
    """
    ?葷頝?YOLO嚗??喳?臬蔣??瘥? keypoints??
    ??頛詨撣嗆?撉冽璅釣?蔣? output_dir/yolo_cam1.mp4 / yolo_cam2.mp4??
    """
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

        # 撱箇?頛詨敶梁?嚗OLO 璅釣嚗?
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
                kpts_tensor = results[0].keypoints.data.cpu().numpy()  # (N_people, 17, 3)
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

            # 撖怠撣園爸?嗥?璅釣撟
            annotated = results[0].plot()
            writer.write(annotated)

            if fid % 30 == 0:
                print(f"  Frame {fid}/{total} ...", end='\r')
            fid += 1

        cap.release()
        writer.release()
        print(f"\n  Done. {fid} frames ??{out_video_path}")
        results_dict[cam_id] = {
            "metadata": {"fps": fps, "width": w, "height": h, "total_frames": fid},
            "keypoint_names": KEYPOINT_NAMES,
            "frames": frames_data
        }

    return results_dict


# ============================================================
#  3D ?遣
# ============================================================

def reconstruct_3d(yolo_results, projection_matrices, conf_thresh, reproj_threshold, cfg=None):
    """
    ?拍?璈?keypoints + Projection Matrix ??RANSAC 銝?皜祇???
    cfg: ?舫嚗??CONFIG dict 隞亙???world_rotation??
    """
    cam_ids  = list(yolo_results.keys())       # ['cam1', 'cam2']
    n_frames = min(len(yolo_results[c]["frames"]) for c in cam_ids)
    print(f"\n[Triangulation] Processing {n_frames} frames with {len(cam_ids)} cameras...")

    all_frames_3d = []
    cam_label = {c: i for i, c in enumerate(cam_ids)}

    for fid in range(n_frames):
        if fid % 30 == 0:
            print(f"  Frame {fid}/{n_frames} ...", end='\r')

        frame_res = {"frame": fid, "keypoints_3d": [], "valid_count": 0}

        # 瘥?豢??洵銝?犖 (person_id=0)
        cam_kpts = {}
        for cid in cam_ids:
            people = yolo_results[cid]["frames"][fid]["people"]
            if people:
                cam_kpts[cid] = people[0]["keypoints"]

        if len(cam_kpts) < 2:
            all_frames_3d.append(frame_res)
            continue

        # ??蝭銝?皜祇?
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
                    # ??憟銝?摨扳???靽格迤
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

            # ?⊥???
            frame_res["keypoints_3d"].append({
                "id": jid,
                "name": KEYPOINT_NAMES[jid],
                "position": None,
                "valid": False
            })

        all_frames_3d.append(frame_res)

    print(f"\n  Done.")
    return all_frames_3d


# ============================================================
#  閬死??
# ============================================================

def visualize_3d_skeleton(frame_data, output_dir, axis_limits):
    """
    axis_limits: (cx, cy, cz, r) ???箏?閬?銝剖???敺?敹‵嚗??蝮格??
    Z 頠詨歇?冽迨?賢??批???-z嚗?雿輸爸?嗆迤蝡????箸迤嚗?
    """

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

    # world_rotation 撌脣銝?皜祇??挾靽格迤摨扳?頠賂?Z ?湔雿輻嚗?????
    zs = list(zs_raw)

    ax.scatter(xs, ys, zs, c='red', s=30, zorder=5)

    for i1, i2 in SKELETON_CONNECTIONS:
        if i1 in valid_ids and i2 in valid_ids:
            p1 = kpts[i1]["position"]
            p2 = kpts[i2]["position"]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],    # Z 撌脩 world_rotation 靽格迤
                'b-', lw=2, alpha=0.7
            )

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (up)')
    ax.set_title(f"Frame {frame_data['frame']:04d}  |  valid: {frame_data['valid_count']}/17")

    # ???箏?閬?嚗??券爸?嗥葬??
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


# ============================================================
#  Main
# ============================================================

def main():
    cfg = CONFIG
    out_dir    = cfg["output_dir"]
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("=" * 60)
    print("  Real-World 2-Camera 3D Skeleton Reconstruction")
    print("=" * 60)

    # ----- Step 1: 霈?璈?摰?-----
    print("\n[Step 1] Loading camera calibration...")
    cameras = {}
    for cam_id, intrin_key, extrin_key in [
        ("cam1", "cam1_intrinsics", "cam1_extrinsics"),
        ("cam2", "cam2_intrinsics", "cam2_extrinsics"),
    ]:
        print(f"\n  Camera: {cam_id}")
        K, dist, R, t, P = load_camera_calibration(
            cfg[intrin_key], cfg[extrin_key]
        )
        cameras[cam_id] = {"K": K, "dist": dist, "R": R, "t": t, "P": P}
        print(f"  K =\n{K}")
        print(f"  t = {t.flatten()}")

    proj_mats = {cid: cameras[cid]["P"] for cid in cameras}

    # ?脣? calibration ??
    calib_summary = {}
    for cid, c in cameras.items():
        calib_summary[cid] = {
            "K": c["K"].tolist(),
            "dist": c["dist"].tolist(),
            "R": c["R"].tolist(),
            "t": c["t"].flatten().tolist(),
            "P": c["P"].tolist()
        }
    with open(os.path.join(out_dir, "camera_params.json"), 'w') as f:
        json.dump(calib_summary, f, indent=2)
    print(f"\n  Saved camera_params.json to {out_dir}/")

    # ----- Step 2: YOLO ?刻? -----
    print("\n[Step 2] Running YOLO pose estimation...")
    yolo_results = run_yolo_on_videos(
        cfg["cam1_video"], cfg["cam2_video"],
        cfg["yolo_model"], cfg["confidence_threshold"],
        out_dir
    )

    # ?脣? YOLO keypoints
    for cid, data in yolo_results.items():
        kp_path = os.path.join(out_dir, f"keypoints_{cid}.json")
        with open(kp_path, 'w') as f:
            json.dump(data, f)
        print(f"  Saved {kp_path}")

    # ----- Step 3: 銝?皜祇? -----
    print("\n[Step 3] Triangulation (RANSAC + LM)...")
    all_frames_3d = reconstruct_3d(
        yolo_results, proj_mats,
        cfg["confidence_threshold"],
        cfg["ransac_reproj_threshold"],
        cfg=cfg
    )

    # ?脣? JSON
    skeleton_path = os.path.join(out_dir, "skeleton_3d.json")
    with open(skeleton_path, 'w') as f:
        json.dump({"frames": all_frames_3d}, f, indent=2)
    print(f"  Saved {skeleton_path}")

    # ----- Step 4: 閬死??-----
    print("\n[Step 4] Rendering 3D skeleton frames...")
    valid_count_total = 0

    # ??敺?????蝯梯??典?摨扳?蝭?嚗摰???Z 撌脣???
    # ??median 簣 3? ?蕪?Ｙ黎?潘??踹??撟??????
    all_pos = []
    for frame_data in all_frames_3d:
        for k in frame_data["keypoints_3d"]:
            if k["valid"] and k["position"] is not None:
                p = k["position"]
                all_pos.append([p[0], p[1], -p[2]])   # Z ??

    if all_pos:
        pos_arr = np.array(all_pos)  # shape (N, 3)

        # ?遘閮? median ??std嚗?瞈曇???3? ?蝢日?
        medians = np.median(pos_arr, axis=0)   # (3,)
        stds    = np.std(pos_arr, axis=0)      # (3,)
        stds    = np.where(stds < 1e-6, 1.0, stds)  # ?脫迫?日

        mask = np.all(np.abs(pos_arr - medians) <= 3.0 * stds, axis=1)
        clean = pos_arr[mask]

        n_removed = len(pos_arr) - len(clean)
        print(f"  Outlier filtering: removed {n_removed}/{len(pos_arr)} points (>3?)")

        if len(clean) == 0:
            clean = pos_arr  # fallback嚗?券?舫蝢文停蝞?

        cx = float(np.median(clean[:, 0]))
        cy = float(np.median(clean[:, 1]))
        cz = float(np.median(clean[:, 2]))

        # 隞仿?瞈曉??????嚗ax_span / 2 * 1.3嚗?
        r = float(np.max([
            clean[:, 0].max() - clean[:, 0].min(),
            clean[:, 1].max() - clean[:, 1].min(),
            clean[:, 2].max() - clean[:, 2].min()
        ])) / 2.0 * 1.3
        r = max(r, 200.0)   # ?撠???200 ?雿?閬?

        fixed_axis = (cx, cy, cz, r)
        print(f"  Fixed axis center=({cx:.1f}, {cy:.1f}, {cz:.1f}), radius={r:.1f}")
    else:
        fixed_axis = (0.0, 0.0, 0.0, 1000.0)
        print("  Warning: no valid positions found, using default axis.")

    for frame_data in all_frames_3d:
        valid_count_total += frame_data["valid_count"]

        if frame_data["valid_count"] > 0:
            visualize_3d_skeleton(frame_data, frames_dir, fixed_axis)

        if frame_data["frame"] % 30 == 0:
            print(f"  Frame {frame_data['frame']} ...", end='\r')

    print(f"\n  Total valid joints across all frames: {valid_count_total}")

    # ----- Step 5: ??敶梁? -----
    fps = cfg["output_fps"]
    if fps <= 0:
        fps = yolo_results["cam1"]["metadata"]["fps"]

    output_video = os.path.join(out_dir, "skeleton_3d.mp4")
    frames_to_video(frames_dir, output_video, fps)

    print("\n" + "=" * 60)
    print(f"  All done! Output saved to: {out_dir}/")
    print(f"    skeleton_3d.json    ??3D keypoints (all frames)")
    print(f"    skeleton_3d.mp4     ??3D skeleton visualization (Z-flipped, fixed view)")
    print(f"    yolo_cam1.mp4       ??cam1 YOLO annotated video")
    print(f"    yolo_cam2.mp4       ??cam2 YOLO annotated video")
    print(f"    camera_params.json  ??camera calibration summary")
    print("=" * 60)


if __name__ == "__main__":
    main()


