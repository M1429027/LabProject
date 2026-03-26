import os
import json
import cv2
import numpy as np
import argparse

from common import (
    load_yaml,
    ensure_dir,
    make_board,
    apply_roi_mask,
    detect_charuco,
    compute_reproj_rms,
    rotmat_to_quat,
    quat_to_rotmat,
    average_quaternions,
    robust_filter_by_rms,
)


def iter_video(video_path, frame_step, max_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fid = -1
    used = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1
        if frame_step > 1 and (fid % frame_step) != 0:
            continue
        yield fid, frame
        used += 1
        if max_frames and used >= max_frames:
            break
    cap.release()


def load_intrinsics_compatible(path):
    with np.load(path, allow_pickle=True) as d:
        keys = list(d.keys())
        K = None
        for k in ["K", "mtx", "camera_matrix", "cameraMatrix", "intrinsic_matrix"]:
            if k in d:
                K = np.array(d[k], dtype=np.float64)
                break
        if K is None:
            raise ValueError(f"Cannot find intrinsic matrix in {path}. Keys={keys}")

        dist = None
        for k in ["dist", "dist_coeffs", "distCoeffs", "d", "distortion_coefficients"]:
            if k in d:
                dist = np.array(d[k], dtype=np.float64).reshape(-1)
                break
        if dist is None:
            dist = np.zeros(5, dtype=np.float64)
    return K, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="calib_charuco_v2/config_extrinsics.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    board_cfg = cfg["board"]
    board, dictionary = make_board(board_cfg)

    K, dist = load_intrinsics_compatible(cfg["intrinsics"]["path"])

    out_dir = cfg["output"]["out_dir"]
    prefix = cfg["output"].get("prefix", "cam")
    ensure_dir(out_dir)

    frame_step = int(cfg["input"].get("frame_step", 5))
    max_frames = int(cfg["input"].get("max_frames", 0))
    min_corners = int(cfg["pose"].get("min_corners", 12))
    max_rms_px = float(cfg["pose"].get("max_rms_px", 0))
    min_frames = int(cfg["pose"].get("min_frames", 10))

    video_path = cfg["input"]["video"]

    rvecs = []
    tvecs = []
    rms_list = []
    frame_ids = []
    best = None
    scanned_frames = 0

    for fid, frame in iter_video(video_path, frame_step, max_frames):
        scanned_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_mask, _ = apply_roi_mask(gray, cfg.get("roi", {}))
        corners, ids = detect_charuco(gray_mask, board, dictionary)
        if ids is None or len(ids) < min_corners:
            continue

        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < 4:
            continue

        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue

        rms = compute_reproj_rms(obj_pts, img_pts, K, dist, rvec, tvec)
        rvecs.append(rvec)
        tvecs.append(tvec)
        rms_list.append(rms)
        frame_ids.append(int(fid))

        if best is None or rms < best["rms"]:
            best = {"rms": rms, "frame": frame.copy()}

    if len(rvecs) < min_frames:
        raise RuntimeError(f"Not enough valid frames: {len(rvecs)} < {min_frames} (scanned {scanned_frames})")

    keep_idx = robust_filter_by_rms(rms_list, max_rms_px=max_rms_px)
    if keep_idx.size < min_frames:
        order = np.argsort(np.array(rms_list))
        keep_idx = order[:min_frames]

    rvecs_f = [rvecs[i] for i in keep_idx]
    tvecs_f = [tvecs[i] for i in keep_idx]
    rms_f = [rms_list[i] for i in keep_idx]
    frames_f = [frame_ids[i] for i in keep_idx]

    quats = []
    for r in rvecs_f:
        R, _ = cv2.Rodrigues(r)
        quats.append(rotmat_to_quat(R))

    q_avg = average_quaternions(quats)
    R_avg = quat_to_rotmat(q_avg)
    rvec_avg, _ = cv2.Rodrigues(R_avg)
    t_avg = np.mean(np.hstack(tvecs_f), axis=1, keepdims=True)

    extrin_path = os.path.join(out_dir, f"{prefix}_extrinsics.npz")
    np.savez(
        extrin_path,
        R=R_avg,
        rotation_matrix=R_avg,
        t=t_avg,
        T=t_avg,
        tvec=t_avg,
        translation_vector=t_avg,
        rvec=rvec_avg,
        rotation_vector=rvec_avg,
        rms_median=float(np.median(rms_f)),
        rms_mean=float(np.mean(rms_f)),
        n_frames_used=int(len(rms_f)),
        n_frames_total=int(len(rms_list)),
        scanned_frames=int(scanned_frames),
        frame_ids=np.array(frames_f, dtype=np.int32),
        roi=np.array(cfg.get("roi", {}), dtype=object),
        board=np.array(board_cfg, dtype=object),
    )

    report = {
        "rms_median": float(np.median(rms_f)),
        "rms_mean": float(np.mean(rms_f)),
        "n_frames_used": int(len(rms_f)),
        "n_frames_total": int(len(rms_list)),
        "scanned_frames": int(scanned_frames),
        "extrinsics_path": extrin_path,
        "board": board_cfg,
        "roi": cfg.get("roi", {}),
    }
    report_path = os.path.join(out_dir, f"{prefix}_extrinsics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if best is not None:
        vis = best["frame"].copy()
        cv2.drawFrameAxes(vis, K, dist, rvec_avg, t_avg, 100.0)
        vis_path = os.path.join(out_dir, f"{prefix}_extrinsics_axes.jpg")
        cv2.imwrite(vis_path, vis)

    print("[OK] Extrinsics done")
    print(f"  RMS median: {report['rms_median']:.4f} px")
    print(f"  Valid/scanned: {len(rms_list)}/{scanned_frames}")
    print(f"  Saved: {extrin_path}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
