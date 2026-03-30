import numpy as np
import cv2
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
import itertools

# ==========================================
# 核心演算法升級區
# ==========================================

def project_point(point_3d, P):
    """將 3D 點投影到 2D"""
    point_4d = np.append(point_3d, 1)
    uv_homo = P @ point_4d
    uv = uv_homo[:2] / uv_homo[2]
    return uv

def reprojection_error(point_3d, projection_matrices, observed_points):
    """計算重投影誤差 (用於非線性優化)"""
    residuals = []
    for i, P in enumerate(projection_matrices):
        projected = project_point(point_3d, P)
        observed = observed_points[i]
        residuals.extend(projected - observed)
    return np.array(residuals)

def triangulate_simple_svd(projections, points):
    """基礎 SVD 三角測量 (用於 RANSAC 內部生成假設)"""
    num_views = len(projections)
    A = np.zeros((num_views * 2, 4))
    for i in range(num_views):
        P = projections[i]
        x, y = points[i]
        A[2*i] = x * P[2, :] - P[0, :]
        A[2*i + 1] = y * P[2, :] - P[1, :]
    u, s, vh = np.linalg.svd(A)
    X = vh[-1]
    return X[:3] / X[3]

def triangulate_ransac_refine(projections, points, reproj_threshold=20.0):
    """
    升級版核心：RANSAC + 非線性優化
    1. RANSAC: 剔除誤差過大的相機 (Outliers)
    2. Refinement: 使用 LM 演算法微調座標
    """
    num_views = len(projections)
    
    # 如果相機少於 3 個，RANSAC 意義不大，直接用 SVD + Refine
    if num_views < 3:
        initial_pt = triangulate_simple_svd(projections, points)
        res = least_squares(reprojection_error, initial_pt, args=(projections, points))
        return res.x

    # --- RANSAC 流程 ---
    best_inliers = []
    best_point = None
    max_inliers_count = -1
    min_total_error = float('inf')

    # 窮舉所有 2 相機組合 (C(4,2)=6種，計算量很小)
    combinations = list(itertools.combinations(range(num_views), 2))
    
    for idx1, idx2 in combinations:
        # 1. 建立假設
        subset_P = [projections[idx1], projections[idx2]]
        subset_pts = [points[idx1], points[idx2]]
        pt_candidate = triangulate_simple_svd(subset_P, subset_pts)

        # 2. 驗證 (計算所有相機的重投影誤差)
        current_inliers = []
        current_error_sum = 0
        
        for i in range(num_views):
            proj_uv = project_point(pt_candidate, projections[i])
            err = np.linalg.norm(proj_uv - points[i])
            
            if err < reproj_threshold:
                current_inliers.append(i)
                current_error_sum += err
        
        # 3. 擇優
        count = len(current_inliers)
        if count > max_inliers_count:
            max_inliers_count = count
            min_total_error = current_error_sum
            best_inliers = current_inliers
            best_point = pt_candidate
        elif count == max_inliers_count:
            if current_error_sum < min_total_error:
                min_total_error = current_error_sum
                best_inliers = current_inliers
                best_point = pt_candidate

    # 如果 RANSAC 失敗 (很少見)，退回 SVD
    if best_point is None or len(best_inliers) < 2:
        return triangulate_simple_svd(projections, points)

    # --- Refinement (非線性優化) ---
    # 只使用 Inliers (好相機) 來進行最終優化
    final_projections = [projections[i] for i in best_inliers]
    final_points = [points[i] for i in best_inliers]
    
    # Levenberg-Marquardt 優化
    refined_result = least_squares(
        reprojection_error, 
        best_point, 
        args=(final_projections, final_points),
        method='lm'
    )
    
    return refined_result.x

# ==========================================
# 其他功能函式 (保持不變)
# ==========================================

def load_keypoints(json_path):
    if not os.path.exists(json_path):
        print(f"Warning: File not found {json_path}")
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_camera_params(calib_path):
    with open(calib_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    projection_matrices = {}
    K = np.array(params['camera_matrix'], dtype=np.float32)
    trans_correction = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    
    for cam_id, cam_data in params['cameras'].items():
        R_gl = np.array(cam_data['R'], dtype=np.float32)
        t_gl = np.array(cam_data['t'], dtype=np.float32).reshape(3, 1)
        R_cv = trans_correction @ R_gl
        t_cv = trans_correction @ t_gl
        projection_matrices[cam_id] = K @ np.hstack([R_cv, t_cv])
    return projection_matrices

def visualize_3d_skeleton(frame_data, skeleton_connections, output_dir, axis_limits=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    kpts = frame_data['keypoints_3d']
    xs, ys, zs, valid_indices = [], [], [], []
    
    for i, kpt in enumerate(kpts):
        if kpt['valid']:
            pos = kpt['position']
            xs.append(pos[0]); ys.append(pos[1]); zs.append(pos[2])
            valid_indices.append(i)
    
    if not xs: plt.close(fig); return

    ax.scatter(xs, ys, zs, c='red', marker='o', s=20)
    for i1, i2 in skeleton_connections:
        if i1 in valid_indices and i2 in valid_indices:
            p1, p2 = kpts[i1]['position'], kpts[i2]['position']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', lw=2, alpha=0.6)
            
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"Frame {frame_data['frame']:04d}")
    
    if axis_limits:
        mx, my, mz, r = axis_limits
        ax.set_xlim(mx-r, mx+r); ax.set_ylim(my-r, my+r); ax.set_zlim(mz-r, mz+r)
    else:
        # Auto range
        mx, my, mz = (np.max(xs)+np.min(xs))/2, (np.max(ys)+np.min(ys))/2, (np.max(zs)+np.min(zs))/2
        r = np.max([np.max(xs)-np.min(xs), np.max(ys)-np.min(ys), np.max(zs)-np.min(zs)]) / 2
        ax.set_xlim(mx-r, mx+r); ax.set_ylim(my-r, my+r); ax.set_zlim(mz-r, mz+r)

    ax.view_init(elev=20, azim=45)
    plt.savefig(os.path.join(output_dir, f'frame_{frame_data["frame"]:04d}.png'), dpi=100)
    plt.close(fig)

def images_to_video(image_folder, output_video_path, fps):
    print(f"Synthesizing video: {output_video_path} ...")
    images = sorted(glob.glob(os.path.join(image_folder, "frame_*.png")))
    if not images: return
    frame = cv2.imread(images[0])
    h, w, _ = frame.shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for img_path in images: out.write(cv2.imread(img_path))
    out.release()

def reconstruct_multi_view(keypoint_files_dict, calib_path, output_dir, confidence_threshold=0.4):
    print("Starting Advanced N-View Triangulation (RANSAC + Refinement)...")
    
    raw_data = {}
    valid_cams = []
    for cid, path in keypoint_files_dict.items():
        d = load_keypoints(path)
        if d: raw_data[cid] = d; valid_cams.append(cid)
    
    proj_matrices = load_camera_params(calib_path)
    avail_cams = [c for c in valid_cams if c in proj_matrices]
    
    first_cam = raw_data[avail_cams[0]]
    kpt_names = first_cam.get('keypoint_names', [])
    total_frames = len(first_cam['frames'])
    
    skeleton_connections = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    
    img_dir = os.path.join(output_dir, 'frames')
    os.makedirs(img_dir, exist_ok=True)
    
    all_skeletons = []
    fixed_axis = None
    
    print(f"Processing {total_frames} frames...")
    
    for f_idx in range(total_frames):
        if f_idx % 10 == 0: print(f" Frame {f_idx}/{total_frames}...", end='\r')
        
        frame_res = {'frame': f_idx, 'keypoints_3d': [], 'valid_count': 0}
        curr_kpts = {}
        for cid in avail_cams:
            try:
                ppl = raw_data[cid]['frames'][f_idx]['people']
                if ppl: curr_kpts[cid] = ppl[0]['keypoints']
            except: pass
            
        if len(curr_kpts) < 2:
            all_skeletons.append(frame_res)
            continue
            
        recon_joints = []
        for j_idx in range(len(kpt_names)):
            projs, pts_2d, cams = [], [], []
            for cid, kpts in curr_kpts.items():
                if j_idx < len(kpts):
                    pt = kpts[j_idx]
                    if pt.get('confidence', 0) >= confidence_threshold:
                        projs.append(proj_matrices[cid])
                        pts_2d.append([pt['x'], pt['y']])
                        cams.append(cid)
            
            if len(projs) >= 2:
                # 🔥 改用新的演算法
                pt_3d = triangulate_ransac_refine(projs, pts_2d)
                
                recon_joints.append({
                    'id': j_idx, 'name': kpt_names[j_idx],
                    'position': pt_3d.tolist(), 'valid': True, 'cameras_used': cams
                })
                frame_res['valid_count'] += 1
            else:
                recon_joints.append({'id': j_idx, 'name': kpt_names[j_idx], 'position': None, 'valid': False})
        
        frame_res['keypoints_3d'] = recon_joints
        all_skeletons.append(frame_res)
        
        if fixed_axis is None and frame_res['valid_count'] > 5:
            pos = [k['position'] for k in recon_joints if k['valid']]
            mid = np.mean(pos, axis=0)
            fixed_axis = (mid[0], mid[1], mid[2], 1.2)
            
        if frame_res['valid_count'] > 0:
            visualize_3d_skeleton(frame_res, skeleton_connections, img_dir, fixed_axis)
            
    with open(os.path.join(output_dir, 'skeleton_3d.json'), 'w', encoding='utf-8') as f:
        json.dump({'frames': all_skeletons}, f, indent=2)
        
    images_to_video(img_dir, os.path.join(output_dir, 'multi_view_reconstruction.mp4'), 60)

def main():
    keypoint_files = {
        'cam1': 'rendervideo/keypoints_camera1_front_left.json',
        'cam2': 'rendervideo/keypoints_camera2_back_left.json',
        'cam3': 'rendervideo/keypoints_camera3_back_right.json',
        'cam4': 'rendervideo/keypoints_camera4_front_right.json'
    }
    calib_file = 'visualization_output_4cams/opencv_calibration.json'
    output_dir = 'triangulation_output_4views_ransac' # 改個資料夾名以免混淆
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(calib_file):
        reconstruct_multi_view(keypoint_files, calib_file, output_dir)
    else:
        print("Calibration file missing.")

if __name__ == "__main__":
    main()