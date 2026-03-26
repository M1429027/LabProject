import numpy as np
import cv2
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 核心演算法 (純 SVD)
# ==========================================

def triangulate_simple_svd(projections, points):
    """
    基礎 SVD 三角測量 (Direct Linear Transformation)
    這是最原始的方法，沒有 RANSAC 抗噪也沒有 LM 優化。
    """
    num_views = len(projections)
    A = np.zeros((num_views * 2, 4))
    for i in range(num_views):
        P = projections[i]
        x, y = points[i]
        # 構建 DLT 方程組
        A[2*i]     = x * P[2, :] - P[0, :]
        A[2*i + 1] = y * P[2, :] - P[1, :]
        
    # 解這組線性方程
    u, s, vh = np.linalg.svd(A)
    X = vh[-1]
    
    # 齊次座標歸一化
    return X[:3] / X[3]

# ==========================================
# 其他功能函式
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
    ax.set_title(f"Frame {frame_data['frame']:04d} (Simple SVD)")
    
    if axis_limits:
        mx, my, mz, r = axis_limits
        ax.set_xlim(mx-r, mx+r); ax.set_ylim(my-r, my+r); ax.set_zlim(mz-r, mz+r)
    else:
        mx, my, mz = (np.max(xs)+np.min(xs))/2, (np.max(ys)+np.min(ys))/2, (np.max(zs)+np.min(zs))/2
        r = np.max([np.max(xs)-np.min(xs), np.max(ys)-np.min(ys), np.max(zs)-np.min(zs)]) / 2
        if r == 0: r = 0.5
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
    print("Starting SIMPLE N-View Triangulation (SVD only)...")
    
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
    
    # 🔥 恢復處理所有 Frame
    print(f"Processing all {total_frames} frames...")

    skeleton_connections = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    
    img_dir = os.path.join(output_dir, 'frames')
    os.makedirs(img_dir, exist_ok=True)
    
    all_skeletons = []
    fixed_axis = None
    
    for f_idx in range(total_frames):
        if f_idx % 50 == 0: print(f" Frame {f_idx}/{total_frames}...", end='\r')
        
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
                # 🔥 使用單純的 SVD
                pt_3d = triangulate_simple_svd(projs, pts_2d)
                
                recon_joints.append({
                    'id': j_idx, 'name': kpt_names[j_idx],
                    'position': pt_3d.tolist(), 'valid': True, 'cameras_used': cams
                })
                frame_res['valid_count'] += 1
            else:
                recon_joints.append({'id': j_idx, 'name': kpt_names[j_idx], 'position': None, 'valid': False})
        
        frame_res['keypoints_3d'] = recon_joints
        all_skeletons.append(frame_res)
        
        if fixed_axis is None and frame_res['valid_count'] > 10:
            pos = [k['position'] for k in recon_joints if k['valid']]
            mid = np.mean(pos, axis=0)
            fixed_axis = (mid[0], mid[1], mid[2], 1.2)
            
        if frame_res['valid_count'] > 0:
            visualize_3d_skeleton(frame_res, skeleton_connections, img_dir, fixed_axis)
            
    with open(os.path.join(output_dir, 'skeleton_3d.json'), 'w', encoding='utf-8') as f:
        json.dump({'frames': all_skeletons}, f, indent=2)
        
    images_to_video(img_dir, os.path.join(output_dir, 'multi_view_reconstruction_simple.mp4'), 60)
    print("\nDone!")

def main():
    keypoint_files = {
        'cam1': 'rendervideo/keypoints_camera1_front_left.json',
        'cam2': 'rendervideo/keypoints_camera2_back_left.json',
        'cam3': 'rendervideo/keypoints_camera3_back_right.json',
        'cam4': 'rendervideo/keypoints_camera4_front_right.json'
    }
    calib_file = 'visualization_output_4cams/opencv_calibration.json'
    
    # 資料夾名稱維持 simple，跟優化版區隔
    output_dir = 'triangulation_output_4views_simple' 
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(calib_file):
        reconstruct_multi_view(keypoint_files, calib_file, output_dir)
    else:
        print("Calibration file missing.")

if __name__ == "__main__":
    main()