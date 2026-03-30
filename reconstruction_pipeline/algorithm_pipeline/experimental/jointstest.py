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

# 嘗試匯入顏色，如果沒有安裝 body_visualizer 則使用預設值
try:
    from body_visualizer.tools.vis_tools import colors
    grey_color = colors['grey']
    red_color = colors['red']
except ImportError:
    grey_color = [0.7, 0.7, 0.7]
    red_color = [0.8, 0.1, 0.1]

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

# ==========================================
# Helper Function: 點轉球體 Mesh
# ==========================================
def points_to_spheres(points, radius=0.01, point_color=[200, 50, 50, 255]):
    """
    將 3D 點陣列轉換為球體 meshes 列表
    """
    spheres = []
    if len(point_color) == 3:
        point_color = list(point_color) + [255]
    elif len(point_color) == 4 and isinstance(point_color[0], float) and point_color[0] <= 1.0:
        point_color = (np.array(point_color) * 255).astype(np.uint8)

    # 取前 22 個主要關節 (軀幹+四肢)
    viz_points = points[:22] 
    
    for p in viz_points:
        s = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        s.apply_translation(p)
        s.visual.vertex_colors = np.tile(point_color, (s.vertices.shape[0], 1))
        spheres.append(s)
    return spheres

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
    cam_pos = cam_pos.astype(float)
    target = target.astype(float)
    up = up.astype(float)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    R = np.array([right, true_up, -forward])
    t = -R @ cam_pos
    RT = np.hstack([R, t.reshape(3, 1)])
    return R, t, RT

def render_mesh_pyrender(meshes, cam_pos, img_width=1600, img_height=1600, 
                         target=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    if not isinstance(meshes, list):
        meshes = [meshes]
    
    # 修改處：移除了 bg_color=[0.0, 0.0, 0.0]，變回預設背景
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
    # pyrender 預設背景其實是透明的 (Alpha=0)，如果你看圖軟體背景是黑的就會看到黑的
    color, depth = r.render(scene)
    r.delete()
    return color

def main():
    check_files_exist()
    
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {comp_device}")
    
    # 載入資料
    support_dir = 'support_data'
    amass_npz_fname = osp.join(support_dir, 'github_data/01_01_poses.npz')
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

    parser = argparse.ArgumentParser(description='生成三角測量用的雙相機影片 (Mesh + Joints)')
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

    # ========== 相機設定 ==========
    first_frame_verts = c2c(body_trans_root.v[frames_to_save[0]])
    first_mesh_center = first_frame_verts.mean(axis=0)
    body_height = first_frame_verts[:, 2].max() - first_frame_verts[:, 2].min()
    
    img_width = 1600
    img_height = 1600
    fov_y = np.pi / 3.0
    
    camera_distance = body_height * 2.5
    camera_height = body_height * 0.15
    
    cam1_pos = first_mesh_center + np.array([
        camera_distance * np.sin(np.pi/4),
        -camera_distance * np.cos(np.pi/4),
        camera_height
    ])
    cam1_target = first_mesh_center.copy()
    
    cam2_pos = first_mesh_center + np.array([
        camera_distance * np.sin(np.pi/4),
        camera_distance * np.cos(np.pi/4),
        camera_height
    ])
    cam2_target = first_mesh_center.copy()
    
    up_vector = np.array([0, 0, 1])
    
    K = compute_camera_intrinsics(img_width, img_height, fov_y)
    R1, t1, RT1 = compute_camera_extrinsics(cam1_pos, cam1_target, up_vector)
    R2, t2, RT2 = compute_camera_extrinsics(cam2_pos, cam2_target, up_vector)
    baseline = np.linalg.norm(cam1_pos - cam2_pos)
    
    camera_params = {
        "intrinsic_matrix": { "K": K.tolist() },
        "stereo_geometry": { "baseline": float(baseline) }
    }
    
    # 建立輸出目錄
    output_dir = 'visualization_output'
    output_dir_cam1 = osp.join(output_dir, 'camera1')
    output_dir_cam2 = osp.join(output_dir, 'camera2')
    # 新增兩個關節目錄
    output_dir_joints1 = osp.join(output_dir, 'joints_camera1')
    output_dir_joints2 = osp.join(output_dir, 'joints_camera2')
    
    os.makedirs(output_dir_cam1, exist_ok=True)
    os.makedirs(output_dir_cam2, exist_ok=True)
    os.makedirs(output_dir_joints1, exist_ok=True)
    os.makedirs(output_dir_joints2, exist_ok=True)
    
    params_path = osp.join(output_dir, 'camera_calibration.json')
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(camera_params, f, indent=2, ensure_ascii=False)
    
    # ========== 開始渲染 ==========
    
    print("=" * 70)
    print(f"🎬 開始渲染 {len(frames_to_save)} 幀 (含 Mesh 與 Joints)...")
    print("=" * 70)
    
    for fId in frames_to_save:
        # 1. 準備資料
        verts = c2c(body_trans_root.v[fId])
        joints_coord = c2c(body_trans_root.Jtr[fId])
        
        # 準備 Mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, 
                              vertex_colors=np.tile(grey_color, (verts.shape[0], 1)))
        
        # 準備 Joints Spheres
        joints_mesh_list = points_to_spheres(joints_coord, radius=0.015, point_color=red_color)
        
        # 準備座標軸
        axis_markers = make_model_axes_at_mesh(length=0.3, radius=0.05)
        
        # 2. 渲染 Mesh (Camera 1 & 2)
        img1 = render_mesh_pyrender(
            meshes=[mesh] + axis_markers,
            cam_pos=cam1_pos, img_width=img_width, img_height=img_height,
            target=cam1_target, up=up_vector
        )
        
        img2 = render_mesh_pyrender(
            meshes=[mesh] + axis_markers,
            cam_pos=cam2_pos, img_width=img_width, img_height=img_height,
            target=cam2_target, up=up_vector
        )
        
        # 3. 渲染 Joints (Camera 1 & 2)
        img_joints1 = render_mesh_pyrender(
            meshes=joints_mesh_list + axis_markers,
            cam_pos=cam1_pos, img_width=img_width, img_height=img_height,
            target=cam1_target, up=up_vector
        )
        
        img_joints2 = render_mesh_pyrender(
            meshes=joints_mesh_list + axis_markers,
            cam_pos=cam2_pos, img_width=img_width, img_height=img_height,
            target=cam2_target, up=up_vector
        )
        
        # 4. 存檔
        imageio.imwrite(osp.join(output_dir_cam1, f'frame_{fId:04d}.png'), img1)
        imageio.imwrite(osp.join(output_dir_cam2, f'frame_{fId:04d}.png'), img2)
        imageio.imwrite(osp.join(output_dir_joints1, f'frame_{fId:04d}.png'), img_joints1)
        imageio.imwrite(osp.join(output_dir_joints2, f'frame_{fId:04d}.png'), img_joints2)
        
        print(f'✅ Frame {fId:04d} saved: Mesh(1,2) & Joints(1,2)')

    print("\n" + "=" * 70)
    print("🎉 渲染完成！")
    print(f"📁 Mesh 相機1:   {output_dir_cam1}/")
    print(f"📁 Mesh 相機2:   {output_dir_cam2}/")
    print(f"📁 Joints 相機1: {output_dir_joints1}/")
    print(f"📁 Joints 相機2: {output_dir_joints2}/")
    print("=" * 70)

if __name__ == "__main__":
    main()