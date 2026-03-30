"""
outer.py  (Stereo Calibration 版)
===================================
用 cv2.stereoCalibrate 直接算兩台相機的相對位姿 (R, t)，
完全避開 findChessboardCorners 角點順序不一致的世界系問題。

輸出：
  cam1_extrinsics.npz  → R=I, t=zeros  (cam1 就是世界原點)
  cam2_extrinsics.npz  → R12, t12      (cam2 相對於 cam1 的位姿)

三角測量時直接用這兩組 P = K @ [R|t] 就好，不再爆炸。
"""

import cv2
import numpy as np
import os

# ============================================================
#  設定區
# ============================================================
CHECKERBOARD = (5, 8)     # 內角點數，與內參標定一致
SQUARE_SIZE  = 55.0       # mm

CAM1_VIDEO    = "outer1_camera1.mp4"
CAM2_VIDEO    = "outer1_camera2.mp4"
CAM1_INTRIN   = "cam1_intrinsics.npz"
CAM2_INTRIN   = "cam2_intrinsics.npz"
CAM1_OUT      = "cam1_extrinsics.npz"
CAM2_OUT      = "cam2_extrinsics.npz"

FRAME_STEP    = 5    # 每幾幀取樣一次
MIN_PAIRS     = 5    # 至少需要幾組有效的雙相機幀才能做 stereoCalibrate
# ============================================================

def get_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def scan_video(video_path, checkerboard):
    """掃描影片，回傳每一有效幀的 (frame_idx, corners_refined, sharpness)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {video_path}")

    results = {}   # frame_idx → (corners_refined, sharpness)
    fid = 0
    print(f"  掃描 {video_path} ...", end="", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1
        if fid % FRAME_STEP != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        if found:
            corners_ref = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            results[fid] = (corners_ref, get_sharpness(gray))

    cap.release()
    print(f" 找到 {len(results)} 幀有棋盤格")
    return results


def main():
    # ----- 讀取內參 -----
    print("\n[1] 讀取內參...")
    with np.load(CAM1_INTRIN) as d:
        K1   = d['mtx'].astype(np.float64)
        dist1 = d['dist'].astype(np.float64)
    with np.load(CAM2_INTRIN) as d:
        K2   = d['mtx'].astype(np.float64)
        dist2 = d['dist'].astype(np.float64)
    print(f"  K1[0,0]={K1[0,0]:.1f}  K2[0,0]={K2[0,0]:.1f}")

    # ----- 掃描兩支影片 -----
    print("\n[2] 掃描影片...")
    r1 = scan_video(CAM1_VIDEO, CHECKERBOARD)
    r2 = scan_video(CAM2_VIDEO, CHECKERBOARD)

    # ----- 找共同幀（兩台都偵測到的幀）-----
    #  由於板子靜止，我們不需要精確同幀：
    #  只要兩台各自找最清晰的幀就夠了（單幀 stereoCalibrate 也可以）
    #  但多幀更穩定，所以盡量找 overlap
    common_fids = sorted(set(r1.keys()) & set(r2.keys()))
    print(f"\n[3] 共同幀數 (兩台都偵測到): {len(common_fids)}")

    if len(common_fids) < MIN_PAIRS:
        print(f"  ⚠️  共同幀不足 {MIN_PAIRS}，改用各自最清晰幀組合...")
        # fallback：各取最清晰的 N 幀，兩兩配對
        best_c1 = sorted(r1.items(), key=lambda x: -x[1][1])[:MIN_PAIRS]
        best_c2 = sorted(r2.items(), key=lambda x: -x[1][1])[:MIN_PAIRS]
        n = min(len(best_c1), len(best_c2))
        if n < 1:
            raise RuntimeError("任一影片根本沒找到棋盤格，請檢查影片或 CHECKERBOARD 設定")
        pairs = [(best_c1[i][1][0], best_c2[i][1][0]) for i in range(n)]
        print(f"  使用 {n} 對（板子靜止，多幀視為獨立觀測）")
    else:
        # 從共同幀裡選最清晰的 MIN_PAIRS ~ 20 幀
        ranked = sorted(
            common_fids,
            key=lambda f: -(r1[f][1] + r2[f][1])   # 兩台清晰度之和
        )[:max(MIN_PAIRS, 20)]
        pairs = [(r1[f][0], r2[f][0]) for f in ranked]
        print(f"  使用 {len(pairs)} 對共同幀做 stereoCalibrate")

    # ----- 準備 3D 物體點 -----
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints  = [objp.copy() for _ in pairs]
    imgpoints1 = [p[0] for p in pairs]
    imgpoints2 = [p[1] for p in pairs]

    # 影像尺寸（從影片讀）
    cap = cv2.VideoCapture(CAM1_VIDEO)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    img_size = (w, h)

    # ----- stereoCalibrate -----
    print(f"\n[4] 執行 stereoCalibrate（{len(pairs)} 對）...")
    flags = (cv2.CALIB_FIX_INTRINSIC)   # 內參已知，只算相對位姿

    rms, K1_, dist1_, K2_, dist2_, R12, t12, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        K1, dist1, K2, dist2,
        img_size,
        flags=flags
    )
    print(f"  stereoCalibrate RMS = {rms:.4f} px")
    if rms > 5.0:
        print("  ⚠️  WARNING: RMS 偏高，建議多錄幾幀或確認板子清晰")
    else:
        print("  ✅ RMS 正常")

    print(f"\n  R12 (cam2 相對 cam1 的旋轉):\n{np.round(R12, 4)}")
    print(f"  t12 (cam2 相對 cam1 的位移, mm):\n{t12.flatten()}")
    baseline = np.linalg.norm(t12)
    print(f"  Baseline = {baseline:.1f} mm ({baseline/1000:.2f} m)")

    # ----- 儲存 -----
    # cam1 = 世界原點 → R=I, t=0
    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros((3, 1), dtype=np.float64)
    np.savez(CAM1_OUT, rvec=np.zeros((3,1)), tvec=t1)

    # cam2 相對 cam1
    rvec2, _ = cv2.Rodrigues(R12)
    np.savez(CAM2_OUT, rvec=rvec2, tvec=t12)

    print(f"\n  💾 {CAM1_OUT}  (cam1 = 世界原點，R=I, t=0)")
    print(f"  💾 {CAM2_OUT}  (cam2 相對 cam1 的位姿)")

    # ----- 視覺化驗證（每相機用 solvePnP 取實際位姿）-----
    print("\n[5] 視覺化驗證...")

    def draw_reproj_solvepnp(video_path, video_data_dict, K, dist, out_prefix):
        frame_keys = list(video_data_dict.keys())
        if not frame_keys:
            print(f"  [{out_prefix}] 沒有有效幀，跳過")
            return
        best_fid = sorted(frame_keys, key=lambda f: -video_data_dict[f][1])[0]
        corners_ref = video_data_dict[best_fid][0]

        # 用 solvePnP 取這幀的實際位姿
        ok, rvec_f, tvec_f = cv2.solvePnP(objp, corners_ref, K, dist)
        if not ok:
            print(f"  [{out_prefix}] solvePnP 失敗")
            return

        proj_pts, _ = cv2.projectPoints(objp, rvec_f, tvec_f, K, dist)
        errors = np.linalg.norm(
            proj_pts.reshape(-1, 2) - corners_ref.reshape(-1, 2), axis=1)
        rms_frame = float(np.sqrt(np.mean(errors**2)))
        print(f"  [{out_prefix}] 單幀重投影 RMS = {rms_frame:.2f} px")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_fid)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return

        vis = frame.copy()
        for pt in proj_pts.reshape(-1, 2):
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
        for pt in corners_ref.reshape(-1, 2):
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            cv2.circle(vis, (x, y), 8, (0, 255, 0), 2)
        out_path = f"check_{out_prefix}.jpg"
        cv2.imwrite(out_path, vis)
        print(f"  [{out_prefix}] 驗證圖: {out_path}  (綠=偵測, 紅=重投影)")

    draw_reproj_solvepnp(CAM1_VIDEO, r1, K1, dist1, "cam1")
    draw_reproj_solvepnp(CAM2_VIDEO, r2, K2, dist2, "cam2")

    print("\n" + "="*55)
    print("  完成！")
    print(f"  {CAM1_OUT}: cam1 為世界原點")
    print(f"  {CAM2_OUT}: cam2 的相對外參")
    print("  現在把這兩個檔交給 real_world_pipeline.py 就不會爆炸了")
    print("="*55)


if __name__ == "__main__":
    main()