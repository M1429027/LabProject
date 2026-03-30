import torch
import numpy as np
import os
from os import path as osp
import trimesh
import imageio
import argparse
import pyrender
import json
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from body_visualizer.tools.vis_tools import colors

def check_files_exist():
    """Check if required files exist"""
    support_dir = 'support_data'
    required_files = [
        osp.join(support_dir, 'github_data/01_01_poses.npz'),
        osp.join(support_dir, 'body_models/smplh/male/model.npz'),
        osp.join(support_dir, 'body_models/smplh/female/model.npz'),
        osp.join(support_dir, 'body_models/dmpls/male/model.npz'),
        osp.join(support_dir, 'body_models/dmpls/female/model.npz')
    ]
    
    missing_files = [f for f in required_files if not osp.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")

def make_model_axes_at_mesh(length=0.25, radius=0.02):
    """生成固定在世界原點的標準右手座標系 XYZ 軸"""
    def make_axis(color_rgba, axis):
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
        if axis == 'x':
            cyl.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0]))
            cyl.apply_translation([length / 2, 0, 0])
        elif axis == 'y':
            cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
            cyl.apply_translation([0, length / 2, 0])
        else:  # 'z'
            cyl.apply_translation([0, 0, length / 2])
        cyl.visual.vertex_colors = np.tile(
            np.array(color_rgba, dtype=np.uint8), (cyl.vertices.shape[0], 1)
        )
        return cyl

    x_axis = make_axis([255, 0, 0, 255], 'x')
    y_axis = make_axis([0, 255, 0, 255], 'y')
    z_axis = make_axis([0, 0, 255, 255], 'z')
    return [x_axis, y_axis, z_axis]

def compute_camera_intrinsics(img_width, img_height, fov_y):
    """計算相機內部參數矩陣"""
    f_y = (img_height / 2.0) / np.tan(fov_y / 2.0)
    f_x = f_y 
    c_x = img_width / 2.0
    c_y = img_height / 2.0
    
    K = np.array([
        [f_x,  0,   c_x],
        [0,   f_y,  c_y],
        [0,    0,    1]
    ])
    return K

def compute_camera_extrinsics(cam_pos, target, up):
    """計算相機外部參數矩陣 [R|t]"""
    cam_pos = cam_pos.astype(float)
    target = target.astype(float)
    up = up.astype(float)
    
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    
    true_up = np.cross(right, forward)
    
    R = np.array([
        right,
        true_up,
        -forward
    ])
    
    t = -R @ cam_pos
    RT = np.hstack([R, t.reshape(3, 1)])
    
    return R, t, RT

def render_mesh_pyrender(meshes, cam_pos, img_width=1600, img_height=1600, 
                          target=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    """使用 pyrender 渲染 mesh"""
    if not isinstance(meshes, list):
        meshes = [meshes]
    
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    
    for mesh in meshes:
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(py_mesh)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    
    cam_pos = cam_pos.astype(float)
    target = target.astype(float)
    up = up.astype(float)
    
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    
    true_up = np.cross(right, forward)
    
    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = true_up
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = cam_pos
    
    scene.add(camera, pose=camera_pose)
    light_pose = camera_pose.copy()
    scene.add(light, pose=light_pose)
    
    r = pyrender.OffscreenRenderer(img_width, img_height)
    color, depth = r.render(scene)
    r.delete()
    
    return color

def main():
    check_files_exist()
    
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {comp_device}")
    
    # 載入資料
    support_dir = 'support_data'
    amass_npz_fname = osp.join(support_dir, 'github_data/02_01_poses.npz')
    bdata = np.load(amass_npz_fname)
    
    subject_gender = bdata['gender']
    if isinstance(subject_gender, np.ndarray) and subject_gender.shape == ():
        subject_gender = subject_gender.item()
    if isinstance(subject_gender, (bytes, np.bytes_)):
        subject_gender = subject_gender.decode('utf-8')
    
    print('The subject of the mocap sequence is:', subject_gender)
    
    bm_fname = osp.join(support_dir, f'body_models/smplh/{subject_gender}/model.npz')
    dmpl_fname = osp.join(support_dir, f'body_models/dmpls/{subject_gender}/model.npz')
    
    num_betas = 16
    num_dmpls = 8
    time_length = len(bdata['trans'])
    bm = BodyModel(bm_path=bm_fname, num_betas=num_betas, model_type='smplh',
                   batch_size=time_length, path_dmpl=dmpl_fname).to(comp_device)
    faces = c2c(bm.f)
    
    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device),
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device),
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device),
        'trans': torch.Tensor(bdata['trans']).to(comp_device),
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], 
                                      repeats=time_length, axis=0)).to(comp_device),
        'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device)
    }
    
    body_trans_root = bm(**{k: v for k, v in body_parms.items() 
                            if k in ['pose_body', 'betas', 'pose_hand', 'dmpls', 'trans', 'root_orient']})

    # CLI 參數
    parser = argparse.ArgumentParser(description='生成三角測量用的四相機影片 (前後左右)')
    parser.add_argument('--all', action='store_true', help='處理所有幀')
    parser.add_argument('--start', type=int, default=0, help='起始幀')
    parser.add_argument('--end', type=int, default=None, help='結束幀')
    parser.add_argument('--step', type=int, default=1, help='幀間隔')
    parser.add_argument('--num', type=int, default=5, help='處理幀數')
    args = parser.parse_args()

    if args.all:
        frames_to_save = list(range(time_length))
    else:
        start = max(0, args.start)
        if args.end is None:
            end = min(time_length, start + args.num)
        else:
            end = min(time_length, args.end)
        frames_to_save = list(range(start, end, max(1, args.step)))

    # ========== 場景與相機設定 ==========
    
    first_frame_verts = c2c(body_trans_root.v[frames_to_save[0]])
    first_mesh_center = first_frame_verts.mean(axis=0)
    
    body_height = first_frame_verts[:, 2].max() - first_frame_verts[:, 2].min()
    body_width = first_frame_verts[:, 0].max() - first_frame_verts[:, 0].min()
    
    print("\n" + "=" * 70)
    print("📏 場景尺寸分析")
    print("=" * 70)
    print(f"人體身高: {body_height:.4f}")
    print(f"第一幀中心座標: {first_mesh_center}")
    
    if 1.5 < body_height < 2.0:
        unit = "meters"
        unit_zh = "公尺"
    elif 150 < body_height < 200:
        unit = "centimeters"
        unit_zh = "公分"
    else:
        unit = "unknown"
        unit_zh = f"未知"
    
    # 通用相機參數
    img_width = 1600
    img_height = 1600
    fov_y = np.pi / 3.0 
    camera_distance = body_height * 3.5
    camera_height = body_height * 0.15 
    up_vector = np.array([0, 0, 1])
    K = compute_camera_intrinsics(img_width, img_height, fov_y)

    # ---------------------------------------------------------
    # 定義 4 個相機配置 (修正版：基於觀測結果 X+為前)
    # ---------------------------------------------------------
    
    camera_configs = [
        {
            "id": "cam1",
            "name": "Front-Left (135°)",   # 左前
            "angle_signs": (1, 1),         # X(+), Y(+) (因為人物面向 X+, 左手在 Y+)
            "folder": "camera1_front_left"
        },
        {
            "id": "cam2",
            "name": "Back-Left (225°)",    # 左後
            "angle_signs": (-1, 1),        # X(-), Y(+) (X-是後, Y+是左)
            "folder": "camera2_back_left"
        },
        {
            "id": "cam3",
            "name": "Back-Right (315°)",   # 右後
            "angle_signs": (-1, -1),       # X(-), Y(-) (X-是後, Y-是右)
            "folder": "camera3_back_right"
        },
        {
            "id": "cam4",
            "name": "Front-Right (45°)",   # 右前
            "angle_signs": (1, -1),        # X(+), Y(-) (X+是前, Y-是右)
            "folder": "camera4_front_right"
        }
    ]

    # 初始化參數字典
    camera_params_export = {
        "metadata": {
            "unit": unit,
            "scene_center": first_mesh_center.tolist(),
            "image_resolution": {"width": img_width, "height": img_height},
            "fov": float(np.degrees(fov_y))
        },
        "intrinsic_matrix": {
            "K": K.tolist(),
            "dist_coeffs": [0, 0, 0, 0, 0]
        },
        "cameras": {}
    }
    
    opencv_params_export = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": [0, 0, 0, 0, 0],
        "image_size": [img_width, img_height],
        "cameras": {}
    }

    print("\n📸 計算相機參數...")
    
    # 建立輸出目錄並計算參數
    output_dir = 'visualization_output_4cams'
    
    for config in camera_configs:
        # 1. 計算位置
        sign_x, sign_y = config['angle_signs']
        # 45度角意味着 x 和 y 的絕對值偏移量相同
        offset_x = camera_distance * np.sin(np.pi/4) * sign_x
        offset_y = camera_distance * np.cos(np.pi/4) * sign_y
        
        pos = first_mesh_center + np.array([offset_x, offset_y, camera_height])
        target = first_mesh_center.copy()
        
        # 2. 計算 Extrinsics
        R, t, RT = compute_camera_extrinsics(pos, target, up_vector)
        
        # 3. 儲存參數到 Config 物件中 (供渲染迴圈使用)
        config['pos'] = pos
        config['target'] = target
        config['R'] = R
        config['t'] = t
        
        # 4. 建立資料夾
        save_path = osp.join(output_dir, config['folder'])
        os.makedirs(save_path, exist_ok=True)
        config['save_path'] = save_path
        
        # 5. 更新 Export 字典
        camera_params_export["cameras"][config['id']] = {
            "name": config['name'],
            "position": pos.tolist(),
            "target": target.tolist(),
            "extrinsic": {"R": R.tolist(), "t": t.tolist()}
        }
        
        opencv_params_export["cameras"][config['id']] = {
            "R": R.tolist(),
            "t": t.tolist()
        }
        
        print(f"✅ {config['name']} set at {pos}")

    # 儲存 JSON
    json_path = osp.join(output_dir, 'camera_calibration.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(camera_params_export, f, indent=2)
        
    cv_json_path = osp.join(output_dir, 'opencv_calibration.json')
    with open(cv_json_path, 'w', encoding='utf-8') as f:
        json.dump(opencv_params_export, f, indent=2)

    print(f"\n✅ 參數已儲存至: {output_dir}")
    
    # ========== 開始渲染 ==========
    
    print("=" * 70)
    print(f"🎬 開始渲染 {len(frames_to_save)} 幀 (共 4 個視角)...")
    print("=" * 70)
    
    for fId in frames_to_save:
        verts = c2c(body_trans_root.v[fId])
        # 使用灰色網格
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, 
                               vertex_colors=np.tile(colors['grey'], (verts.shape[0], 1)))
        
        axis_markers = make_model_axes_at_mesh(length=0.3, radius=0.05)
        
        # 對每個相機進行渲染
        for config in camera_configs:
            img = render_mesh_pyrender(
                meshes=[mesh] + axis_markers,
                cam_pos=config['pos'],
                img_width=img_width,
                img_height=img_height,
                target=config['target'],
                up=up_vector
            )
            
            # 存檔
            fname = osp.join(config['save_path'], f'frame_{fId:04d}.png')
            imageio.imwrite(fname, img)
        
        print(f'✅ Frame {fId:04d} rendered for all 4 cameras')

    print("\n" + "=" * 70)
    print("🎉 4視角渲染完成！")
    print(f"📁 輸出目錄: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()