"""
verify_world_origin.py
======================
從 outer1_camera1.mp4 / outer1_camera2.mp4 各抽一幀，
用 findChessboardCorners 偵測，然後把 corners[0]（objp 的世界原點）
用紅色大圓點畫出來，讓你對照兩張圖的「紅點」是否在同一個物理角點。

如果兩台相機的紅點在不同物理角點 → 世界系不一致 → 三角測量爆炸
如果兩台相機的紅點在相同物理角點 → 世界系一致 → 問題在別處
"""
import cv2
import numpy as np

CHECKERBOARD = (5, 8)
VIDEOS = [
    ("outer1_camera1.mp4", "verify_cam1.jpg"),
    ("outer1_camera2.mp4", "verify_cam2.jpg"),
]

def find_best_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    best_frame, best_corners, best_sharp = None, None, 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % 5 != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if found:
            s = cv2.Laplacian(gray, cv2.CV_64F).var()
            if s > best_sharp:
                best_sharp = s
                best_frame = frame.copy()
                best_corners = corners.copy()
    cap.release()
    return best_frame, best_corners

for video_path, out_path in VIDEOS:
    frame, corners = find_best_frame(video_path)
    if frame is None:
        print(f"[{video_path}] ❌ 找不到棋盤格")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners_ref = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    vis = frame.copy()
    # 畫所有角點（小藍點）
    cv2.drawChessboardCorners(vis, CHECKERBOARD, corners_ref, True)

    # 紅色大圓 = corners[0]  → 這是 objp 的 (0,0,0)，即「世界原點」
    origin_px = tuple(map(int, corners_ref[0].ravel()))
    cv2.circle(vis, origin_px, 20, (0, 0, 255), -1)
    cv2.putText(vis, "World Origin (0,0,0)", 
                (origin_px[0]+25, origin_px[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 綠色大方塊 = corners[-1] → 世界座標系的「對角」
    last_px = tuple(map(int, corners_ref[-1].ravel()))
    cv2.circle(last_px and vis, last_px, 20, (0, 255, 0), -1)
    cv2.putText(vis, "Last corner",
                (last_px[0]+25, last_px[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imwrite(out_path, vis)
    print(f"[{video_path}] ✅ 已存 {out_path}")
    print(f"  World Origin pixel = {origin_px}")
    print(f"  Last corner pixel  = {last_px}")

print("\n請打開 verify_cam1.jpg 和 verify_cam2.jpg")
print("對照兩張的【紅色大圓點】是否落在板子同一個物理角點（例如同一個角落）")
