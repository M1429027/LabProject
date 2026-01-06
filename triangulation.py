import numpy as np
import cv2
import json
import os
import glob
import matplotlib
# 設定 Agg 後端，避免在迴圈中不斷跳出視窗 (Headless mode)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_keypoints(json_path):
    """載入 YOLO 輸出的關鍵點資料"""
    if not os.path.exists(json_path):
        print(f"⚠️ 警告: 找不到檔案 {json_path}")
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_camera_params(calib_path):
    """載入相機校正參數 (包含座標系修正：OpenGL -> OpenCV)"""
    with open(calib_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    projection_matrices = {}
    
    # 讀取共用的相機矩陣 K
    K = np.array(params['camera_matrix'], dtype=np.float32)
    
    # === 修正矩陣：將 OpenGL 座標系轉為 OpenCV 座標系 ===
    # Pyrender (OpenGL): X右, Y上, Z後 (看-Z)
    # OpenCV:           X右, Y下, Z前 (看+Z)
    # 需要將 Y 和 Z 軸反轉 (乘上 -1)
    # 這相當於繞 X 軸旋轉 180 度
    trans_correction = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])
    
    # 遍歷所有相機
    for cam_id, cam_data in params['cameras'].items():
        # 原始讀入的 R (OpenGL style)
        R_gl = np.array(cam_data['R'], dtype=np.float32)
        t_gl = np.array(cam_data['t'], dtype=np.float32).reshape(3, 1)
        
        # === 關鍵修正步驟 ===
        # 1. 對旋轉矩陣 R 的 Row 1 (Y) 和 Row 2 (Z) 變號
        #    數學上等於: R_cv = correction_matrix @ R_gl
        R_cv = trans_correction @ R_gl
        
        # 2. 對平移向量 t 也要做同樣的變換
        #    數學上等於: t_cv = correction_matrix @ t_gl
        t_cv = trans_correction @ t_gl
        
        # 計算投影矩陣 P = K[R|t]
        RT = np.hstack([R_cv, t_cv])
        P = K @ RT
        projection_matrices[cam_id] = P
        
    return projection_matrices

def triangulate_n_views(projections, points):
    """
    使用 SVD 進行 N-View 三角測量 (DLT 演算法)
    
    Args:
        projections: list of 3x4 projection matrices (P)
        points: list of (u, v) coordinates
        
    Returns:
        point_3d: (3,) numpy array
    """
    # 構建矩陣 A
    # 對於每個視角 i，我們有 P_i * X = w_i * x_i
    # 這可以轉化為兩個線性方程:
    # x * (P_row3 * X) - (P_row1 * X) = 0
    # y * (P_row3 * X) - (P_row2 * X) = 0
    
    num_views = len(projections)
    A = np.zeros((num_views * 2, 4))
    
    for i in range(num_views):
        P = projections[i]
        x, y = points[i]
        
        # Row 1: x * P_row3 - P_row1
        A[2*i] = x * P[2, :] - P[0, :]
        # Row 2: y * P_row3 - P_row2
        A[2*i + 1] = y * P[2, :] - P[1, :]
        
    # 使用 SVD 求解 A * X = 0
    u, s, vh = np.linalg.svd(A)
    
    # 解是 V 的最後一行 (或是 VH 的最後一列)
    X = vh[-1]
    
    # 從齊次座標轉換回歐幾里得座標 (除以 w)
    X = X[:3] / X[3]
    
    return X

def visualize_3d_skeleton(frame_data, skeleton_connections, output_dir, axis_limits=None):
    """
    視覺化 3D 骨架 (單幀)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    keypoints_3d = frame_data['keypoints_3d']
    
    xs, ys, zs = [], [], []
    valid_indices = []
    
    for i, kpt in enumerate(keypoints_3d):
        if kpt['valid'] and kpt['position'] is not None:
            pos = kpt['position']
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
            valid_indices.append(i)
    
    if not xs:
        plt.close(fig)
        return

    # 繪製關鍵點
    ax.scatter(xs, ys, zs, c='red', marker='o', s=20, label='Joints')
    
    # 繪製骨架連接
    for conn in skeleton_connections:
        idx1, idx2 = conn
        if idx1 in valid_indices and idx2 in valid_indices:
            kpt1 = keypoints_3d[idx1]['position']
            kpt2 = keypoints_3d[idx2]['position']
            ax.plot([kpt1[0], kpt2[0]], 
                    [kpt1[1], kpt2[1]], 
                    [kpt1[2], kpt2[2]], 
                    'b-', linewidth=2, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Multi-View Reconstruction - Frame {frame_data['frame']:04d}")
    
    # --- 視角與範圍設定 ---
    # 如果有傳入固定的軸範圍，就使用固定的，這樣影片才不會抖動
    if axis_limits:
        mid_x, mid_y, mid_z, max_range = axis_limits
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        # 自動範圍 (只用於第一幀或測試)
        max_range = np.array([
            np.max(xs) - np.min(xs),
            np.max(ys) - np.min(ys),
            np.max(zs) - np.min(zs)
        ]).max() / 2.0
        mid_x = (np.max(xs) + np.min(xs)) * 0.5
        mid_y = (np.max(ys) + np.min(ys)) * 0.5
        mid_z = (np.max(zs) + np.min(zs)) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(elev=20, azim=45) 

    output_png = os.path.join(output_dir, f'frame_{frame_data["frame"]:04d}.png')
    plt.savefig(output_png, dpi=100)
    plt.close(fig)

def images_to_video(image_folder, output_video_path, fps=30):
    """合成影片"""
    print(f"🎬 正在合成影片: {output_video_path} ...")
    images = sorted(glob.glob(os.path.join(image_folder, "frame_*.png")))
    
    if not images:
        print("❌ 找不到圖片，無法製作影片")
        return

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    size = (width, height)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    
    for i, image_path in enumerate(images):
        if i % 50 == 0:
            print(f"  處理中... {i}/{len(images)}", end='\r')
        img = cv2.imread(image_path)
        out.write(img)
    
    out.release()
    print(f"\n✅ 影片製作完成！")

def reconstruct_multi_view(keypoint_files_dict, calib_path, output_dir, confidence_threshold=0.4):
    """
    主函數：多視角重建
    Args:
        keypoint_files_dict: dict, e.g. {'cam1': 'path/to/json1', 'cam2': 'path/to/json2'}
    """
    print("=" * 70)
    print("🔺 開始 N-View 三角測量 (SVD Method)")
    print("=" * 70)

    # 1. 載入資料
    raw_data = {}
    valid_cam_ids = []
    
    for cam_id, path in keypoint_files_dict.items():
        data = load_keypoints(path)
        if data:
            raw_data[cam_id] = data
            valid_cam_ids.append(cam_id)
            print(f"✅ 載入 {cam_id}: {len(data['frames'])} 幀")
    
    if len(valid_cam_ids) < 2:
        print("❌ 錯誤：至少需要 2 個有效的相機數據才能進行重建")
        return

    # 2. 載入相機參數 (投影矩陣 P)
    proj_matrices = load_camera_params(calib_path)
    
    # 確保參數檔裡面的 ID 跟 檔案 ID 對得上
    available_cams = [cid for cid in valid_cam_ids if cid in proj_matrices]
    if len(available_cams) < 2:
        print("❌ 錯誤：相機參數檔中的 ID 與輸入檔案不匹配")
        return
    print(f"📡 可用於重建的相機: {available_cams}")

    # 取得一些 meta data
    first_cam = raw_data[available_cams[0]]
    keypoint_names = first_cam.get('keypoint_names', [])
    total_frames = len(first_cam['frames']) # 假設所有檔案幀數相同
    
    # 定義骨架連接 (COCO 格式範例)
    skeleton_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), 
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), 
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    # 建立輸出目錄
    img_output_dir = os.path.join(output_dir, 'frames')
    os.makedirs(img_output_dir, exist_ok=True)
    
    all_3d_skeletons = []
    
    # --- 計算固定的軸範圍 (避免影片抖動) ---
    # 這裡我們簡單設定一個固定的空間範圍，假設人是在原點附近
    # 如果你的單位是公尺，範圍設 -1.5 ~ 1.5 應該夠
    # 如果是公分，可能要 -150 ~ 150
    # 為了自動化，我們可以在第一幀先跑一次動態計算
    fixed_axis_limits = None 

    print(f"🚀 開始逐幀重建 (共 {total_frames} 幀)...")

    for frame_idx in range(total_frames):
        if frame_idx % 10 == 0:
            print(f"  Processing frame {frame_idx}/{total_frames}...", end='\r')
            
        frame_result = {
            'frame': frame_idx,
            'keypoints_3d': [],
            'valid_count': 0
        }
        
        # 提取當前幀所有相機的 keypoints
        # 結構: current_frame_kpts['cam1'] = [List of 17 points]
        current_frame_kpts = {}
        for cid in available_cams:
            try:
                # 假設只有一個人 (people[0])
                people = raw_data[cid]['frames'][frame_idx]['people']
                if people:
                    current_frame_kpts[cid] = people[0]['keypoints']
            except IndexError:
                pass # 該幀該相機沒人或索引錯誤
        
        # 如果這一幀幾乎沒有相機拍到人，就跳過
        if len(current_frame_kpts) < 2:
            all_3d_skeletons.append(frame_result)
            continue

        # 針對每一個關節點進行 N-View Triangulation
        num_joints = len(keypoint_names)
        reconstructed_joints = []
        
        for j_idx in range(num_joints):
            
            valid_projections = []
            valid_points_2d = []
            contributing_cams = []
            
            # 檢查所有相機是否看到這個點 (j_idx)
            for cid, kpts in current_frame_kpts.items():
                if j_idx < len(kpts):
                    pt_data = kpts[j_idx]
                    # 信心度檢查
                    if pt_data.get('confidence', 0) >= confidence_threshold:
                        valid_projections.append(proj_matrices[cid])
                        valid_points_2d.append([pt_data['x'], pt_data['y']])
                        contributing_cams.append(cid)
            
            # 只有當至少 2 個相機看到此點時，才進行重建
            if len(valid_projections) >= 2:
                point_3d = triangulate_n_views(valid_projections, valid_points_2d)
                
                reconstructed_joints.append({
                    'id': j_idx,
                    'name': keypoint_names[j_idx],
                    'position': point_3d.tolist(),
                    'valid': True,
                    'cameras_used': contributing_cams
                })
                frame_result['valid_count'] += 1
            else:
                reconstructed_joints.append({
                    'id': j_idx,
                    'name': keypoint_names[j_idx],
                    'position': None,
                    'valid': False
                })
        
        frame_result['keypoints_3d'] = reconstructed_joints
        all_3d_skeletons.append(frame_result)
        
        # 第一次成功重建後，鎖定軸範圍
        if fixed_axis_limits is None and frame_result['valid_count'] > 5:
            # 快速計算當前幀的範圍
            positions = [k['position'] for k in reconstructed_joints if k['valid']]
            positions = np.array(positions)
            mid = np.mean(positions, axis=0)
            # 設定一個稍微大一點的固定半徑，例如 1.2 米 (假設單位是米)
            # 你可以根據你的場景單位調整這個 radius
            radius = 1.2 
            fixed_axis_limits = (mid[0], mid[1], mid[2], radius)
            print(f"\n🔒 鎖定視覺化範圍中心: {mid}, 半徑: {radius}")

        # 繪圖
        if frame_result['valid_count'] > 0:
            visualize_3d_skeleton(frame_result, skeleton_connections, img_output_dir, fixed_axis_limits)

    print(f"\n✅ 所有幀處理完成！")
    
    # 儲存 3D 骨架 JSON
    output_json = os.path.join(output_dir, 'skeleton_3d.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({'frames': all_3d_skeletons}, f, indent=2, ensure_ascii=False)
    
    # 合成影片
    video_path = os.path.join(output_dir, 'multi_view_reconstruction.mp4')
    images_to_video(img_output_dir, video_path, fps=30)

def main():
    # === 設定輸入檔案路徑 ===
    # 請根據你上一個步驟輸出的資料夾結構設定
    
    # 假設你的 YOLO 關鍵點輸出檔名如下 (你需要先跑 YOLO 產生這些檔)
    keypoint_files = {
        'cam1': 'rendervideo/output_frontR.json',
        'cam2': 'rendervideo/output_frontL.json',
        'cam3': 'rendervideo/output_backL.json',
        'cam4': 'rendervideo/output_backR.json'
    }
    
    # 上一步驟產生的 OpenCV 格式校正檔
    calib_file = 'visualization_output_4cams/opencv_calibration.json'
    
    output_dir = 'triangulation_output_4views'
    os.makedirs(output_dir, exist_ok=True)
    
    # 檢查校正檔是否存在
    if not os.path.exists(calib_file):
        print(f"❌ 找不到校正檔: {calib_file}")
        print("請確認你是否已經執行了上一步的相機生成程式。")
        return

    reconstruct_multi_view(keypoint_files, calib_file, output_dir)

if __name__ == "__main__":
    main()