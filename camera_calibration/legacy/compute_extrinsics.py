"""
compute_extrinsics.py
=====================
使用 ChArUco Board 計算每台相機的外參 (rvec, tvec)。

相較於原本的 findChessboardCorners，ChArUco 的優點：
  ● 每個角點有唯一 ID，不論從哪個視角看，ID 對應的物理角點永遠相同
  ● 兩台相機的世界原點自動一致 → 三角測量不爆炸

【設定區說明】
  SQUARES_X / SQUARES_Y：棋盤格的格子數（不是內角點！）
    例如你的板子 findChessboardCorners 設 (5, 8) 表示 5×8 個內角點
    → 格子數為 (6, 9)，所以 SQUARES_X=6, SQUARES_Y=9（短邊在前）
  SQUARE_SIZE：每格邊長 (mm)
  MARKER_SIZE：ArUco marker 邊長 (mm)，通常是 SQUARE_SIZE * 0.75 左右
  ARUCO_DICT  ：你標定內參時用的 ArUco 字典，不確定的話設 None 會自動偵測
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os

# ============================================================
#  ★ 設定區（請依你的實際板子修改）
# ============================================================

SQUARES_X   = 6        # 水平方向格子數（比內角點多 1）
SQUARES_Y   = 9        # 垂直方向格子數（比內角點多 1）
SQUARE_SIZE = 55.0     # 格子邊長 (mm)
MARKER_SIZE = 40.0     # ArUco marker 邊長 (mm)，通常 = SQUARE_SIZE * 0.72 左右

# ArUco 字典：如果你確定用哪種就填，例如 aruco.DICT_6X6_250
# 不確定就設 None，程式會自動從常用字典裡試到偵測到為止
ARUCO_DICT  = None     # 例：aruco.DICT_6X6_250 / aruco.DICT_4X4_50 / None

VIDEO_CONFIGS = [
    {
        "video":      "outer1_camera1.mp4",
        "intrinsics": "cam1_intrinsics.npz",
        "output":     "cam1_extrinsics.npz",
    },
    {
        "video":      "outer1_camera2.mp4",
        "intrinsics": "cam2_intrinsics.npz",
        "output":     "cam2_extrinsics.npz",
    },
    # 有幾台就加幾組
]

# ============================================================
#  工具函式
# ============================================================

CANDIDATE_DICTS = [
    aruco.DICT_6X6_250,
    aruco.DICT_6X6_1000,
    aruco.DICT_5X5_250,
    aruco.DICT_4X4_250,
    aruco.DICT_7X7_250,
    aruco.DICT_6X6_50,
    aruco.DICT_4X4_50,
    aruco.DICT_5X5_50,
]

def auto_detect_dict(gray, board_sx, board_sy, sq_sz, mk_sz):
    """在常用 dict 清單中自動找到能偵測到 marker 的那個。"""
    for d_id in CANDIDATE_DICTS:
        dictionary = aruco.getPredefinedDictionary(d_id)
        board = aruco.CharucoBoard((board_sx, board_sy), sq_sz, mk_sz, dictionary)
        detector = aruco.CharucoDetector(board)
        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is not None and len(ids) >= 4:
            print(f"  ✅ 自動偵測到 ArUco dict id={d_id}")
            return dictionary, board
    return None, None


def make_board(dictionary):
    """根據全域設定建立 CharucoBoard。"""
    return aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_SIZE,
        MARKER_SIZE,
        dictionary,
    )


def get_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ============================================================
#  主邏輯
# ============================================================

def process_single_camera(video_path, intrinsic_path, output_path):
    print(f"\n{'='*55}")
    print(f"  Camera: {video_path}")
    print(f"{'='*55}")

    # ----- 讀取內參 -----
    if not os.path.exists(intrinsic_path):
        print(f"  ❌ 找不到內參檔: {intrinsic_path}")
        return
    with np.load(intrinsic_path) as data:
        mtx  = data['mtx'].astype(np.float64)
        dist = data['dist'].astype(np.float64)
    print(f"  內參讀取OK  K[0,0]={mtx[0,0]:.1f}")

    # ----- 掃影片，找最清晰且 ChArUco 點數最多的幀 -----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ 無法開啟影片: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  影片共 {total_frames} 幀，每 5 幀取樣一次...")

    # 第一幀先偵測 dict（避免每幀都嘗試）
    resolved_dict = None
    board = None

    best_frame   = None
    best_corners = None
    best_ids     = None
    best_score   = -1       # score = 角點數 * 清晰度 (兼顧覆蓋率與清晰度)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 第一次需要決定 dict
        if resolved_dict is None:
            if ARUCO_DICT is not None:
                resolved_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
                board = make_board(resolved_dict)
                print(f"  使用指定 ARUCO_DICT={ARUCO_DICT}")
            else:
                resolved_dict, board = auto_detect_dict(
                    gray, SQUARES_X, SQUARES_Y, SQUARE_SIZE, MARKER_SIZE)
                if resolved_dict is None:
                    print("  ⚠️  前幾幀偵測不到 ArUco，繼續嘗試後續幀...")
                    continue

        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < 4:
            continue

        sharpness = get_sharpness(gray)
        score = len(charuco_ids) * sharpness

        if score > best_score:
            best_score   = score
            best_frame   = frame.copy()
            best_corners = charuco_corners
            best_ids     = charuco_ids

    cap.release()

    if best_frame is None:
        print("  ❌ 整支影片都找不到 ChArUco 角點，請確認:")
        print("     1. SQUARES_X / SQUARES_Y 設定是否與板子一致")
        print("     2. MARKER_SIZE 是否合理")
        print("     3. 影片中板子是否清晰可見")
        return

    n_corners = len(best_ids)
    print(f"  最佳幀偵測到 {n_corners} 個唯一角點，清晰度={best_score/n_corners:.1f}")

    # ----- SolvePnP（使用 board.matchImagePoints 確保 3D↔2D 對應正確）-----
    obj_pts, img_pts = board.matchImagePoints(best_corners, best_ids)

    if obj_pts is None or len(obj_pts) < 4:
        print("  ❌ matchImagePoints 失敗，角點不足")
        return

    retval, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, mtx, dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not retval:
        print("  ❌ solvePnP 失敗")
        return

    # ----- 重投影誤差（sanity check）-----
    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)
    errors = np.linalg.norm(proj_pts.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    print(f"  外參重投影 RMS = {rms:.3f} px  (建議 < 5px)")
    if rms > 10:
        print("  ⚠️  WARNING: RMS 偏高，建議重錄或確認板子設定")

    # ----- 視覺化（畫座標軸）-----
    axis_len = SQUARE_SIZE * 3
    axis_pts = np.float32([
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, -axis_len],
    ])
    img_axis, _ = cv2.projectPoints(axis_pts, rvec, tvec, mtx, dist)

    # 世界原點 = board corner ID=0 的投影
    origin_world = np.float32([[0, 0, 0]])
    origin_img, _ = cv2.projectPoints(origin_world, rvec, tvec, mtx, dist)
    origin_px = tuple(map(int, origin_img[0].ravel()))

    vis = best_frame.copy()
    vis = cv2.line(vis, origin_px, tuple(map(int, img_axis[0].ravel())), (0, 0, 255), 5)   # X 紅
    vis = cv2.line(vis, origin_px, tuple(map(int, img_axis[1].ravel())), (0, 255, 0), 5)   # Y 綠
    vis = cv2.line(vis, origin_px, tuple(map(int, img_axis[2].ravel())), (255, 0, 0), 5)   # Z 藍
    cv2.circle(vis, origin_px, 8, (0, 255, 255), -1)   # 世界原點（黃點，應在同一物理角點）

    vis_path = f"check_ext_{os.path.splitext(os.path.basename(video_path))[0]}.jpg"
    cv2.imwrite(vis_path, vis)
    print(f"  驗證圖已存: {vis_path}")
    print(f"  ★ 兩台相機的黃點應落在同一個物理角點（板子左上角 marker 旁）")

    # ----- 儲存 -----
    np.savez(output_path, rvec=rvec, tvec=tvec)
    print(f"  💾 外參已儲存: {output_path}")

    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec
    print(f"  相機在世界中的位置 (mm): {C.flatten()}")
    print(f"  t (board→cam, mm):      {tvec.flatten()}")


# ============================================================
#  Entry Point
# ============================================================

if __name__ == "__main__":
    print("ChArUco Extrinsics Calibration")
    print(f"Board: {SQUARES_X}×{SQUARES_Y} squares, "
          f"square={SQUARE_SIZE}mm, marker={MARKER_SIZE}mm")

    for cfg in VIDEO_CONFIGS:
        process_single_camera(cfg["video"], cfg["intrinsics"], cfg["output"])

    print("\n🎉 完成！")
    print("驗證步驟：")
    print("  1. 打開 check_ext_outer_camera1.jpg 和 check_ext_outer_camera2.jpg")
    print("  2. 兩張圖的【黃色圓點】應落在同一個物理角點（通常是板子的某個固定角落）")
    print("  3. 若黃點位置不同 → 先確認 SQUARES_X/Y 和 MARKER_SIZE 是否正確")
