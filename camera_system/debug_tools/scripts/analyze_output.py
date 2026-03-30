"""
analyze_output.py
=================
分析 real_world_output/skeleton_3d.json，印出統計資訊來診斷問題。
"""
import json
import numpy as np

JSON_PATH = "real_world_output/skeleton_3d.json"

with open(JSON_PATH) as f:
    data = json.load(f)

frames = data["frames"]
n_frames = len(frames)
print(f"總幀數: {n_frames}")

# ---- 每幀的有效關節數 ----
valid_counts = [fr["valid_count"] for fr in frames]
print(f"\n[有效關節數/幀]")
print(f"  平均: {np.mean(valid_counts):.1f} / 17")
print(f"  最大: {max(valid_counts)}")
print(f"  最小: {min(valid_counts)}")
print(f"  為0的幀數: {sum(1 for v in valid_counts if v == 0)}")

# ---- 蒐集所有有效 3D 點 ----
all_pts = []
for fr in frames:
    for kpt in fr["keypoints_3d"]:
        if kpt["valid"] and kpt["position"] is not None:
            all_pts.append(kpt["position"])

all_pts = np.array(all_pts)  # (N, 3)
print(f"\n[所有有效 3D 點總數]: {len(all_pts)}")

if len(all_pts) > 0:
    print(f"\n[座標範圍]")
    for i, ax in enumerate(["X", "Y", "Z"]):
        mn, mx = all_pts[:, i].min(), all_pts[:, i].max()
        med = np.median(all_pts[:, i])
        std = np.std(all_pts[:, i])
        print(f"  {ax}: min={mn:.1f}  max={mx:.1f}  median={med:.1f}  std={std:.1f}")

    # ---- 離群點偵測 (>5σ) ----
    medians = np.median(all_pts, axis=0)
    stds = np.std(all_pts, axis=0)
    outlier_mask = np.any(np.abs(all_pts - medians) > 5 * stds, axis=1)
    n_outliers = outlier_mask.sum()
    print(f"\n[離群點 >5σ]: {n_outliers} / {len(all_pts)}  ({100*n_outliers/len(all_pts):.1f}%)")

    if n_outliers > 0:
        print(f"  離群點座標範圍:")
        out_pts = all_pts[outlier_mask]
        for i, ax in enumerate(["X", "Y", "Z"]):
            print(f"    {ax}: {out_pts[:,i].min():.1f} ~ {out_pts[:,i].max():.1f}")

    # ---- 每個關節的統計 ----
    KEYPOINT_NAMES = [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ]
    print(f"\n[各關節有效率 & 平均位置]")
    for jid in range(17):
        pts_j = []
        for fr in frames:
            kpts = fr["keypoints_3d"]
            if jid < len(kpts) and kpts[jid]["valid"] and kpts[jid]["position"]:
                pts_j.append(kpts[jid]["position"])
        rate = len(pts_j) / n_frames * 100
        if pts_j:
            pts_j = np.array(pts_j)
            med = np.median(pts_j, axis=0)
            std_j = np.std(pts_j, axis=0)
            # 判斷這個關節的 Z 方向是否正常（Z 應該比其他軸小，代表在地板上方）
            note = ""
            if abs(med[2]) > 5000:
                note = "  ⚠️ Z 值偏離"
            print(f"  {jid:2d} {KEYPOINT_NAMES[jid]:<20s} valid={rate:5.1f}%  "
                  f"med=({med[0]:7.1f},{med[1]:7.1f},{med[2]:7.1f})  "
                  f"std=({std_j[0]:6.1f},{std_j[1]:6.1f},{std_j[2]:6.1f})"
                  f"{note}")
        else:
            print(f"  {jid:2d} {KEYPOINT_NAMES[jid]:<20s} valid=  0.0%  (no data)")

    # ---- 連續幀間位移（穩定性）----
    print(f"\n[幀間位移穩定性（取 nose 關節）]")
    nose_pts = []
    for fr in frames:
        kpts = fr["keypoints_3d"]
        if kpts and kpts[0]["valid"] and kpts[0]["position"]:
            nose_pts.append((fr["frame"], kpts[0]["position"]))
    if len(nose_pts) > 2:
        dists = []
        for i in range(1, len(nose_pts)):
            d = np.linalg.norm(np.array(nose_pts[i][1]) - np.array(nose_pts[i-1][1]))
            dists.append(d)
        print(f"  平均幀間位移: {np.mean(dists):.1f} mm")
        print(f"  最大幀間位移: {np.max(dists):.1f} mm  (突波=可能是爆炸幀)")
        print(f"  >500mm 的突波幀數: {sum(1 for d in dists if d > 500)}")
