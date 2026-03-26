import os
import glob
import json
import cv2
import numpy as np
import argparse

from common import load_yaml, ensure_dir, make_board, apply_roi_mask, detect_charuco


def iter_images(image_dir):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    for path in sorted(files):
        img = cv2.imread(path)
        if img is not None:
            yield path, img


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


def calibrate_charuco_compatible(charuco_corners_all, charuco_ids_all, board, image_size):
    # Preferred API when available.
    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charuco_corners_all,
            charuco_ids_all,
            board,
            image_size,
            None,
            None,
        )
        return rms, K, dist, rvecs, tvecs

    # Fallback: convert Charuco correspondences into object/image points and use calibrateCamera.
    board_pts = np.array(board.getChessboardCorners(), dtype=np.float32)
    objpoints = []
    imgpoints = []

    for corners, ids in zip(charuco_corners_all, charuco_ids_all):
        if corners is None or ids is None:
            continue
        ids_flat = ids.reshape(-1).astype(np.int32)
        if len(ids_flat) < 4:
            continue
        obj = board_pts[ids_flat].reshape(-1, 1, 3).astype(np.float32)
        img = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
        objpoints.append(obj)
        imgpoints.append(img)

    if len(objpoints) < 5:
        raise RuntimeError("Not enough valid object/image point sets for calibration fallback")

    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
        flags=flags,
        criteria=criteria,
    )
    return rms, K, dist, rvecs, tvecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="calib_charuco_v2/config_intrinsics.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    board_cfg = cfg["board"]
    board, dictionary = make_board(board_cfg)

    out_dir = cfg["output"]["out_dir"]
    prefix = cfg["output"].get("prefix", "cam")
    ensure_dir(out_dir)

    frame_step = int(cfg["input"].get("frame_step", 5))
    max_frames = int(cfg["input"].get("max_frames", 0))
    min_corners = int(cfg["detector"].get("min_corners", 12))

    charuco_corners_all = []
    charuco_ids_all = []
    used_frames = []
    sample_vis = None
    image_size = None
    scanned_frames = 0

    image_dir = cfg["input"].get("image_dir", "")
    video_path = cfg["input"].get("video", "")

    if image_dir:
        for path, frame in iter_images(image_dir):
            scanned_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_mask, _ = apply_roi_mask(gray, cfg.get("roi", {}))
            corners, ids = detect_charuco(gray_mask, board, dictionary)
            if ids is None or len(ids) < min_corners:
                continue
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])
            charuco_corners_all.append(corners)
            charuco_ids_all.append(ids)
            used_frames.append(path)
            if sample_vis is None:
                vis = frame.copy()
                cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
                sample_vis = vis
    else:
        if not video_path:
            raise ValueError("input.video or input.image_dir is required")
        for fid, frame in iter_video(video_path, frame_step, max_frames):
            scanned_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_mask, _ = apply_roi_mask(gray, cfg.get("roi", {}))
            corners, ids = detect_charuco(gray_mask, board, dictionary)
            if ids is None or len(ids) < min_corners:
                continue
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])
            charuco_corners_all.append(corners)
            charuco_ids_all.append(ids)
            used_frames.append(int(fid))
            if sample_vis is None:
                vis = frame.copy()
                cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
                sample_vis = vis

    if len(charuco_corners_all) < 5:
        raise RuntimeError(
            f"Not enough valid frames for calibration: {len(charuco_corners_all)} valid / {scanned_frames} scanned"
        )

    rms, K, dist, _, _ = calibrate_charuco_compatible(
        charuco_corners_all, charuco_ids_all, board, image_size
    )

    dist = np.array(dist, dtype=np.float64).reshape(-1)

    intrin_path = os.path.join(out_dir, f"{prefix}_intrinsics.npz")
    np.savez(
        intrin_path,
        K=K,
        mtx=K,
        dist=dist,
        dist_coeffs=dist,
        image_size=np.array(image_size, dtype=np.int32),
        rms=float(rms),
        n_frames=len(charuco_corners_all),
        scanned_frames=int(scanned_frames),
        roi=np.array(cfg.get("roi", {}), dtype=object),
        board=np.array(board_cfg, dtype=object),
    )

    report = {
        "rms": float(rms),
        "n_frames": int(len(charuco_corners_all)),
        "scanned_frames": int(scanned_frames),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "intrinsics_path": intrin_path,
        "board": board_cfg,
        "roi": cfg.get("roi", {}),
    }
    report_path = os.path.join(out_dir, f"{prefix}_intrinsics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if sample_vis is not None:
        sample_path = os.path.join(out_dir, f"{prefix}_charuco_sample.jpg")
        cv2.imwrite(sample_path, sample_vis)

    used_frames_path = os.path.join(out_dir, f"{prefix}_intrinsics_frames.json")
    with open(used_frames_path, "w", encoding="utf-8") as f:
        json.dump({"used_frames": used_frames}, f, indent=2)

    print("[OK] Intrinsics done")
    print(f"  RMS: {rms:.4f} px")
    print(f"  Valid/scanned: {len(charuco_corners_all)}/{scanned_frames}")
    print(f"  Saved: {intrin_path}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
