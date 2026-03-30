import torch
import numpy as np
import json
import os
from os import path as osp
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# Settings
TRIANGULATION_RESULT_PATH = 'triangulation_output_4views_ransac/skeleton_3d.json'
SUPPORT_DIR = 'support_data'
AMASS_NPZ_PATH = osp.join(SUPPORT_DIR, 'github_data/02_01_poses.npz')

# Joint Map: (COCO_ID, SMPL_ID, Name)
JOINT_MAP = [
    (11, 1, "Left Hip"), (12, 2, "Right Hip"),
    (13, 4, "Left Knee"), (14, 5, "Right Knee"),
    (15, 7, "Left Ankle"), (16, 8, "Right Ankle"),
    (5, 16, "Left Shoulder"), (6, 17, "Right Shoulder"),
    (7, 18, "Left Elbow"), (8, 19, "Right Elbow"),
    (9, 20, "Left Wrist"), (10, 21, "Right Wrist")
]

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) to a set of 3D points S2,
    minimizing the mean squared error.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where K=U*D*V'
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    
    # Construct rotation matrix
    R = V.dot(U.T)

    # 5. Determine scale
    trace_TA = s.sum()
    scale = trace_TA / var1 if var1 > 1e-8 else 1.0

    # 6. Transform
    # S1_hat = scale * R * S1 + t
    t = mu2 - scale*(R.dot(mu1))
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    # Return transformed points and the calculated scale
    return S1_hat, scale

def get_amass_gt_joints():
    if not osp.exists(AMASS_NPZ_PATH):
        raise FileNotFoundError(f"Missing AMASS file: {AMASS_NPZ_PATH}")

    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bdata = np.load(AMASS_NPZ_PATH)
    
    subject_gender = bdata['gender'].item() if isinstance(bdata['gender'], np.ndarray) else bdata['gender']
    if isinstance(subject_gender, (bytes, np.bytes_)): subject_gender = subject_gender.decode('utf-8')

    bm_fname = osp.join(SUPPORT_DIR, f'body_models/smplh/{subject_gender}/model.npz')
    dmpl_fname = osp.join(SUPPORT_DIR, f'body_models/dmpls/{subject_gender}/model.npz')
    
    time_length = len(bdata['trans'])
    bm = BodyModel(bm_path=bm_fname, num_betas=16, model_type='smplh',
                   batch_size=time_length, path_dmpl=dmpl_fname).to(comp_device)
    
    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device),
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device),
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device),
        'trans': torch.Tensor(bdata['trans']).to(comp_device),
        'betas': torch.Tensor(np.repeat(bdata['betas'][:16][np.newaxis], repeats=time_length, axis=0)).to(comp_device),
        'dmpls': torch.Tensor(bdata['dmpls'][:, :8]).to(comp_device)
    }
    
    with torch.no_grad():
        body_trans_root = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls', 'trans', 'root_orient']})
    
    return c2c(body_trans_root.Jtr)

def load_and_process_pred_joints(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    num_frames = len(frames)
    pred_array = np.full((num_frames, 17, 3), np.nan)
    
    for i, frame in enumerate(frames):
        for kpt in frame['keypoints_3d']:
            k_id = kpt['id']
            if kpt['valid'] and kpt['position'] is not None and k_id < 17:
                x, y, z = kpt['position']
                
                # Applying Y-axis flip
                fixed_pos = [x, -y, z] 
                
                pred_array[i, k_id] = fixed_pos
                
    return pred_array

def calculate_metrics(gt_all, pred_all):
    coco_indices = [x[0] for x in JOINT_MAP]
    smpl_indices = [x[1] for x in JOINT_MAP]
    joint_names = [x[2] for x in JOINT_MAP]
    
    gt_subset = gt_all[:, smpl_indices, :]
    pred_subset = pred_all[:, coco_indices, :]
    
    # Filter valid frames
    valid_mask = ~np.isnan(pred_subset).any(axis=(1, 2))
    if np.sum(valid_mask) == 0:
        print("Error: No valid frames found.")
        return

    gt_subset = gt_subset[valid_mask]
    pred_subset = pred_subset[valid_mask]
    
    num_frames = len(gt_subset)
    print(f"Valid frames: {num_frames}")

    # 1. MPJPE (Root Aligned Only)
    gt_root = np.mean(gt_subset[:, 0:2, :], axis=1, keepdims=True)
    pred_root = np.mean(pred_subset[:, 0:2, :], axis=1, keepdims=True)
    
    gt_centered = gt_subset - gt_root
    pred_centered = pred_subset - pred_root
    
    mpjpe_raw = np.linalg.norm(pred_centered - gt_centered, axis=2) * 1000.0
    print(f"Raw MPJPE (Root Aligned): {np.mean(mpjpe_raw):.2f} mm")

    # 2. PA-MPJPE (Procrustes Aligned)
    pa_mpjpe_errors = []
    scale_factors = []
    
    for i in range(num_frames):
        S_gt = gt_subset[i]
        S_pred = pred_subset[i]
        
        # Calculate transform and scale
        S_pred_aligned, scale = compute_similarity_transform(S_pred, S_gt)
        
        # Calculate error
        diff = np.linalg.norm(S_pred_aligned - S_gt, axis=1) * 1000.0
        pa_mpjpe_errors.append(diff)
        scale_factors.append(scale)
    
    pa_mpjpe_errors = np.array(pa_mpjpe_errors)
    
    print("-" * 60)
    print("Evaluation Results (PA-MPJPE)")
    print("-" * 60)
    
    pa_mpjpe = np.mean(pa_mpjpe_errors)
    print(f"Overall PA-MPJPE: {pa_mpjpe:.2f} mm")
    print("-" * 60)
    
    mean_per_joint = np.mean(pa_mpjpe_errors, axis=0)
    for name, err in zip(joint_names, mean_per_joint):
        print(f"{name:<20} | {err:.2f}")
    
    # 3. Scale Factor Analysis
    print("-" * 60)
    print("Scale Factor Analysis (for Vicon compatibility)")
    print("-" * 60)
    
    avg_scale = np.mean(scale_factors)
    print(f"Average Scale Factor: {avg_scale:.4f}")
    
    if 0.9 <= avg_scale <= 1.1:
        print("Scale is HEALTHY (Close to 1.0)")
    else:
        print("Scale is WARNING (Check Camera Calibration)")
    print("-" * 60)

def main():
    print("Loading Data...")
    gt_joints = get_amass_gt_joints()
    pred_joints = load_and_process_pred_joints(TRIANGULATION_RESULT_PATH)
    
    min_frames = min(len(gt_joints), len(pred_joints))
    calculate_metrics(gt_joints[:min_frames], pred_joints[:min_frames])

if __name__ == "__main__":
    main()