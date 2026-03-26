import torch
import numpy as np
import json
import os
import glob
import cv2
from os import path as osp
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# ==========================================
# 1. 設定路徑與參數
# ==========================================
TRIANGULATION_RESULT_PATH = 'triangulation_output_4views_smoothed/skeleton_3d.json'
SUPPORT_DIR = 'support_data'
AMASS_NPZ_PATH = osp.join(SUPPORT_DIR, 'github_data/01_01_poses.npz')
OUTPUT_FRAMES_DIR = 'comparison_frames_output_aligned' # 改個資料夾名區分
OUTPUT_VIDEO_NAME = 'comparison_result_aligned_60fps.mp4'

# ... (get_amass_ground_truth 和 load_triangulation_result 函式保持不變) ...
# 為了節省篇幅，請確保你保留了這兩個函式
def get_amass_ground_truth():
    # ... (請複製之前的程式碼) ...
    if not osp.exists(AMASS_NPZ_PATH): raise FileNotFoundError(f"找不到: {AMASS_NPZ_PATH}")
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bdata = np.load(AMASS_NPZ_PATH)
    subject_gender = bdata['gender'].item() if isinstance(bdata['gender'], np.ndarray) else bdata['gender']
    if isinstance(subject_gender, bytes): subject_gender = subject_gender.decode('utf-8')
    bm_fname = osp.join(SUPPORT_DIR, f'body_models/smplh/{subject_gender}/model.npz')
    dmpl_fname = osp.join(SUPPORT_DIR, f'body_models/dmpls/{subject_gender}/model.npz')
    time_length = len(bdata['trans'])
    bm = BodyModel(bm_path=bm_fname, num_betas=16, model_type='smplh', batch_size=time_length, path_dmpl=dmpl_fname).to(comp_device)
    body_parms = {'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), 'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), 'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), 'trans': torch.Tensor(bdata['trans']).to(comp_device), 'betas': torch.Tensor(np.repeat(bdata['betas'][:16][np.newaxis], repeats=time_length, axis=0)).to(comp_device), 'dmpls': torch.Tensor(bdata['dmpls'][:, :8]).to(comp_device)}
    with torch.no_grad(): body_trans_root = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls', 'trans', 'root_orient']})
    return c2c(body_trans_root.Jtr)

def load_triangulation_result(json_path):
    if not osp.exists(json_path): raise FileNotFoundError(f"找不到: {json_path}")
    with open(json_path, 'r') as f: return json.load(f)['frames']

# ★★★ 新增：對齊函式 ★★★
def align_skeletons(pred_joints, gt_joints):
    """透過對齊質心 (Centroid) 來移除整體位移偏差"""
    if len(pred_joints) == 0 or len(gt_joints) == 0:
        return pred_joints
        
    # 計算兩個骨架的中心點 (質心)
    pred_center = np.mean(pred_joints, axis=0)
    gt_center = np.mean(gt_joints, axis=0)
    
    # 計算從 pred 到 gt 的平移向量
    translation = gt_center - pred_center
    
    # 將所有預測點加上這個平移向量
    aligned_pred = pred_joints + translation
    return aligned_pred

def save_frame_image(frame_idx, gt_data, pred_data, save_dir):
    if frame_idx >= len(gt_data) or frame_idx >= len(pred_data): return False

    # 1. 準備 GT 資料
    gt_frame = gt_data[frame_idx][:22] 
    
    # 2. 準備 Pred 資料 (包含 Y 軸修正)
    pred_frame_info = pred_data[frame_idx]
    pred_joints = []
    
    for kpt in pred_frame_info['keypoints_3d']:
        if kpt['valid'] and kpt['position'] is not None:
            raw_x, raw_y, raw_z = kpt['position']
            #翻轉Y軸對齊
            fixed_pos = [raw_x, -raw_y, raw_z] 
            
            pred_joints.append(fixed_pos)
            
    pred_joints = np.array(pred_joints)

    # 只有當兩個骨架都有資料時才對齊
    if len(pred_joints) > 0 and len(gt_frame) > 0:
        aligned_pred_joints = align_skeletons(pred_joints, gt_frame)
    else:
        aligned_pred_joints = pred_joints

    # 3. 繪圖
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 畫 GT (藍)
    ax.scatter(gt_frame[:, 0], gt_frame[:, 1], gt_frame[:, 2], 
               c='blue', marker='^', s=50, label='GT (SMPL Joint)')
    
    # 畫 Pred (紅)
    if len(aligned_pred_joints) > 0:
        ax.scatter(aligned_pred_joints[:, 0], aligned_pred_joints[:, 1], aligned_pred_joints[:, 2], 
                   c='red', marker='o', s=50, label='Pred (Aligned & Flipped)')
        
        # 標示頭部確認方向
        ax.text(aligned_pred_joints[0,0], aligned_pred_joints[0,1], aligned_pred_joints[0,2], "Nose", color='red')

    if len(gt_frame) > 0:
         ax.text(gt_frame[0,0], gt_frame[0,1], gt_frame[0,2], "Pelvis", color='blue')

    ax.set_title(f"Comparison - Frame {frame_idx:04d}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # 設定視角範圍 (跟隨 GT)
    if len(gt_frame) > 0:
        mid = np.mean(gt_frame, axis=0)
        max_range = 1.0
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.view_init(elev=15, azim=45)
    plt.savefig(os.path.join(save_dir, f"frame_{frame_idx:04d}.png"), dpi=80)
    plt.close(fig)
    return True

# ... (images_to_video 函式保持不變) ...
def images_to_video(image_folder, output_video_path, fps=60):
    print(f"🎬 合成影片 (FPS={fps}): {output_video_path} ...")
    images = sorted(glob.glob(os.path.join(image_folder, "frame_*.png")))
    if not images: print("❌ 無圖片"); return
    frame = cv2.imread(images[0]); height, width, _ = frame.shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in tqdm(images, desc="Writing Video"): out.write(cv2.imread(f))
    out.release(); print("🎉 完成！")

def main():
    print("🚀 開始處理...")
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
    gt_data = get_amass_ground_truth()
    pred_data = load_triangulation_result(TRIANGULATION_RESULT_PATH)
    total_frames = min(len(gt_data), len(pred_data))
    print(f"共 {total_frames} 幀")

    # 可以先跑前 100 幀測試
    # for i in tqdm(range(100), desc="Rendering"):
    for i in tqdm(range(total_frames), desc="Rendering"):
        save_frame_image(i, gt_data, pred_data, OUTPUT_FRAMES_DIR)
        
    images_to_video(OUTPUT_FRAMES_DIR, OUTPUT_VIDEO_NAME, fps=60)

if __name__ == "__main__":
    main()