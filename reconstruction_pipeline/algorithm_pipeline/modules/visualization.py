import torch
import numpy as np
import os
from os import path as osp
import trimesh
import imageio
import argparse
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres

def check_files_exist():
    """Check if required files exist"""
    support_dir = 'support_data'
    required_files = [
        osp.join(support_dir, 'github_data/amass_sample.npz'),
        osp.join(support_dir, 'body_models/smplh/male/model.npz'),
        osp.join(support_dir, 'body_models/smplh/female/model.npz'),
        osp.join(support_dir, 'body_models/dmpls/male/model.npz'),
        osp.join(support_dir, 'body_models/dmpls/female/model.npz')
    ]
    
    missing_files = [f for f in required_files if not osp.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")

def make_model_axes_at_mesh(mesh=None, length=0.25, radius=0.02):
    """
    生成固定在世界原點 (0,0,0) 的標準右手座標系 XYZ 軸。
    X: 紅 / Y: 綠 / Z: 藍
    length: 軸的長度
    radius: 軸的半徑
    """
    def make_axis(color_rgba, axis):
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
        # trimesh 的 cylinder 預設沿 Z 軸
        if axis == 'x':
            # Z → X: -90° 沿 Y 軸
            cyl.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0]))
            cyl.apply_translation([length / 2, 0, 0])
        elif axis == 'y':
            # Z → Y: +90° 沿 X 軸
            cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
            cyl.apply_translation([0, length / 2, 0])
        else:  # 'z'
            # 保持原方向 Z 軸
            cyl.apply_translation([0, 0, length / 2])

        cyl.visual.vertex_colors = np.tile(
            np.array(color_rgba, dtype=np.uint8), (cyl.vertices.shape[0], 1)
        )
        return cyl

    # 各軸
    x_axis = make_axis([255, 0, 0, 255], 'x')
    y_axis = make_axis([0, 255, 0, 255], 'y')
    z_axis = make_axis([0, 0, 255, 255], 'z')

    return [x_axis, y_axis, z_axis]

def look_at_matrix(cam_pos, target=np.array([0,0,0]), up=np.array([0,0,1])):
    cam_pos = cam_pos.astype(float)
    target = target.astype(float)
    up = up.astype(float)

    forward = target - cam_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)

    mat = np.eye(4, dtype=float)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = forward
    mat[:3, 3] = cam_pos

    return mat


def main():
    # Check required files
    check_files_exist()
    
    # Initialize device
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #comp_device = torch.device('cpu')

    print(f"Using device: {comp_device}")
    
    # Setup paths and load data
    support_dir = 'support_data'
    amass_npz_fname = osp.join(support_dir, 'github_data/01_01_poses.npz')
    bdata = np.load(amass_npz_fname)
    # 讀取 gender（可能是 bytes、numpy.bytes_、numpy.ndarray scalar 或純 str）
    subject_gender = bdata['gender']

    # 如果是 numpy array scalar（shape = ()），先取出 python value
    if isinstance(subject_gender, np.ndarray) and subject_gender.shape == ():
        subject_gender = subject_gender.item()

    # 如果是 bytes，decode 成 str
    if isinstance(subject_gender, (bytes, np.bytes_)):
        subject_gender = subject_gender.decode('utf-8')

    # 現在 subject_gender 應該是普通的 str，例如 'female'
    print('The subject of the mocap sequence is:', subject_gender)

    
    print('Data keys available:', list(bdata.keys()))
    print('The subject of the mocap sequence is:', subject_gender)
    
    # Setup body model
    bm_fname = osp.join(support_dir, f'body_models/smplh/{subject_gender}/model.npz')
    dmpl_fname = osp.join(support_dir, f'body_models/dmpls/{subject_gender}/model.npz')
    
    num_betas = 16  # number of body parameters
    num_dmpls = 8   # number of DMPL parameters
    time_length = len(bdata['trans'])
    bm = BodyModel(bm_path=bm_fname, num_betas=num_betas,model_type='smplh',batch_size=time_length,
                  path_dmpl=dmpl_fname).to(comp_device)
    faces = c2c(bm.f)
    
    # Prepare body parameters
    
    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device),
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device),
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device),
        'trans': torch.Tensor(bdata['trans']).to(comp_device),
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], 
                                      repeats=time_length, axis=0)).to(comp_device),
        'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device)
    }
    
    print('Body parameter vector shapes:')
    for k, v in body_parms.items():
        print(f'{k}: {v.shape}')
    print(f'time_length = {time_length}')
    
    # Setup viewer
    imw, imh = 1600, 1600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    
    # Create output directory
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Precompute batched outputs (safe with different subsets)
    body_pose_beta = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas']})
    body_pose_hand = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand']})

    use_dmpls = True
    try:
        body_dmpls = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls']})
    except Exception as e:
        print('DMPL forward failed or not supported, skipping DMPL visualizations. Error:', e)
        use_dmpls = False
        body_dmpls = None

    body_trans_root = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls', 'trans', 'root_orient']})

    # CLI options to choose frames to save
    parser = argparse.ArgumentParser(description='Save rendered frames from AMASS params')
    parser.add_argument('--all', action='store_true', help='Save all frames')
    parser.add_argument('--start', type=int, default=0, help='Start frame (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='End frame (exclusive)')
    parser.add_argument('--step', type=int, default=1, help='Frame step')
    parser.add_argument('--num', type=int, default=5, help='If end not set, save start..start+num frames')
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

    print(f'Saving {len(frames_to_save)} frames: {frames_to_save[:5]}{"..." if len(frames_to_save)>5 else ""}')

    def save_mesh_render(meshes, basename, cam_pos=None):
        if not isinstance(meshes, list):
            meshes = [meshes]
        png_path = osp.join(output_dir, f'{basename}.png')

        if cam_pos is not None:
            mv.camera_transform = look_at_matrix(cam_pos, target=np.array([0,0,0]), up=np.array([0,0,1]))

        mv.set_static_meshes(meshes)
        img = mv.render(render_wireframe=False)
        imageio.imwrite(png_path, img)

    for fId in frames_to_save:
        """
        # 1. pose + betas
        verts = c2c(body_pose_beta.v[fId])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=np.tile(colors['grey'], (verts.shape[0], 1)))
        save_mesh_render(mesh, f'01_pose_beta_f{fId:04d}')

        # 2. pose + hand
        verts = c2c(body_pose_hand.v[fId])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=np.tile(colors['grey'], (verts.shape[0], 1)))
        save_mesh_render(mesh, f'02_pose_hand_f{fId:04d}')
        
        # 3. joints
        joints = c2c(body_pose_hand.Jtr[fId])
        joints_mesh = points_to_spheres(joints, point_color=colors['red'], radius=0.005)
        save_mesh_render(joints_mesh, f'03_joints_f{fId:04d}')

        # 4. dmpls (if available)
        if use_dmpls and body_dmpls is not None:
            verts = c2c(body_dmpls.v[fId])
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=np.tile(colors['grey'], (verts.shape[0], 1)))
            save_mesh_render(mesh, f'04_dmpls_f{fId:04d}')

        # 5. trans + root_orient
        verts = c2c(body_trans_root.v[fId])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=np.tile(colors['grey'], (verts.shape[0], 1)))
        save_mesh_render(mesh, f'05_trans_root_f{fId:04d}')
        """
        # 6. transformed view (rotate mesh before saving)
        """
        verts = c2c(body_trans_root.v[fId])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=np.tile(colors['grey'], (verts.shape[0], 1)))
        
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-180), (0, 0, 1)))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-60), (1, 0, 0)))
        mesh.apply_translation([1.0, -1.0, -2.0])
        
        axis_markers = make_model_axes(length=30, radius=0.01)
        save_mesh_render([mesh] + axis_markers, f'06_transformed_f{fId:04d}')
        """
        verts = c2c(body_trans_root.v[fId])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=np.tile(colors['grey'], (verts.shape[0], 1)))
        #mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-180), (0, 0, 1)))
        #mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-60), (1, 0, 0)))
        #mesh.apply_translation([1.0, -1.0, -2.0])
        axis_markers = make_model_axes_at_mesh(length=0.3, radius=0.05)
        print(mesh.vertices.mean(axis=0))
        print([a.vertices.mean(axis=0) for a in axis_markers])
        save_mesh_render([mesh] + axis_markers, f'frame_{fId:04d}', cam_pos=np.array([2,2,2]))


    print(f"All visualizations have been saved to the '{output_dir}' directory")


if __name__ == "__main__":
    main()
