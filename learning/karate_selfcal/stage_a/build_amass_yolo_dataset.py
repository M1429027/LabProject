"""Build Stage A noisy training samples from AMASS by rendering and running YOLO pose.

Pipeline:
AMASS .npz -> SMPLH/DMPL mesh + Jtr joints -> four virtual camera renders -> YOLO 2D pose -> Stage A ray tokens.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

# Must be set before importing pyrender/OpenGL on headless WSL machines.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from body_visualizer.tools.vis_tools import colors
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from pyrender import DirectionalLight, Mesh, OffscreenRenderer, PerspectiveCamera, Scene
from ultralytics import YOLO


COCO_NUM_JOINTS = 17
RAY_FEATURE_DIM = 11

# SMPLH/SMPL 22-body joint approximation to COCO-17.
# Eyes/ears are not available from the body model, so they are approximated by the head joint.
SMPLH_TO_COCO17 = {
    0: 15,   # nose/head approx
    1: 15,   # left_eye approx
    2: 15,   # right_eye approx
    3: 15,   # left_ear approx
    4: 15,   # right_ear approx
    5: 16,   # left_shoulder
    6: 17,   # right_shoulder
    7: 18,   # left_elbow
    8: 19,   # right_elbow
    9: 20,   # left_wrist
    10: 21,  # right_wrist
    11: 1,   # left_hip
    12: 2,   # right_hip
    13: 4,   # left_knee
    14: 5,   # right_knee
    15: 7,   # left_ankle
    16: 8,   # right_ankle
}

CAMERA_DEFS = [
    ("cam1", "camera1_front_left", (1.0, 1.0)),
    ("cam2", "camera2_back_left", (-1.0, 1.0)),
    ("cam3", "camera3_back_right", (-1.0, -1.0)),
    ("cam4", "camera4_front_right", (1.0, -1.0)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render AMASS sequences, run YOLO pose, and export Stage A samples.")
    parser.add_argument("--source-glob", required=True, help="Glob for AMASS *_poses.npz files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--support-dir", default="support_data")
    parser.add_argument("--yolo-model", default="yolo11l-pose.pt")
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--max-frames-per-sequence", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=960)
    parser.add_argument("--fov-degree", type=float, default=60.0)
    parser.add_argument("--distance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distance-jitter", type=float, default=0.25, help="Relative camera distance jitter, e.g. 0.25 = +/-25%.")
    parser.add_argument("--height-jitter", type=float, default=0.25, help="Relative camera height jitter against body height.")
    parser.add_argument("--azimuth-jitter-deg", type=float, default=15.0, help="Per-camera horizontal angle jitter in degrees.")
    parser.add_argument("--target-jitter", type=float, default=0.12, help="Target point jitter as a fraction of body height.")
    parser.add_argument("--fov-jitter-deg", type=float, default=8.0)
    parser.add_argument("--extrinsic-rotation-noise-deg", type=float, default=0.0, help="Calibration noise: random camera viewing-direction rotation in degrees used only for ray tokens/triangulation.")
    parser.add_argument("--extrinsic-translation-noise", type=float, default=0.0, help="Calibration noise: camera position Gaussian noise as a fraction of body height.")
    parser.add_argument("--extrinsic-scale-noise", type=float, default=0.0, help="Calibration noise: relative camera distance scale jitter around the body center.")
    parser.add_argument("--view-dropout-prob", type=float, default=0.0, help="Randomly disable a view while keeping fixed 4-view tensor slots.")
    parser.add_argument("--min-active-views", type=int, default=2)
    parser.add_argument("--camera-aug-per-frame", action="store_true", help="Sample a different camera rig per frame instead of per sequence.")
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--split", default="train")
    parser.add_argument("--save-rendered-frames", action="store_true")
    parser.add_argument("--save-annotated-frames", action="store_true")
    parser.add_argument(
        "--use-face-targets",
        action="store_true",
        help="Supervise COCO face joints 0-4. Default ignores them because SMPLH has no true eyes/ears/nose joints.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--yolo-device", default="auto", help="Ultralytics YOLO device. Use 0 for GPU 0, cpu for CPU, or auto to follow --device.")
    return parser.parse_args()


def compute_camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    fy = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def compute_extrinsics(position: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up is None else up.astype(np.float64)
    forward = target - position
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    true_up = np.cross(right, forward)
    rotation = np.array([right, true_up, -forward], dtype=np.float64)
    translation = -rotation @ position.reshape(3)
    return rotation, translation


def camera_pose_opengl(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    right /= max(float(np.linalg.norm(right)), 1e-9)
    true_up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = position
    return pose


def normalize_gender(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).lower()
    return value if value in {"male", "female", "neutral"} else "neutral"


def selected_frame_indices(num_frames: int, stride: int, max_frames: int) -> np.ndarray:
    stride = max(int(stride), 1)
    indices = np.arange(0, num_frames, stride, dtype=np.int64)
    if max_frames > 0:
        indices = indices[: int(max_frames)]
    return indices


def load_amass_body(npz_path: Path, frame_indices: np.ndarray, support_dir: Path, device: torch.device):
    data = np.load(npz_path, allow_pickle=True)
    poses = np.asarray(data["poses"], dtype=np.float32)[frame_indices]
    trans = np.asarray(data["trans"], dtype=np.float32)[frame_indices]
    dmpls = np.asarray(data["dmpls"], dtype=np.float32)[frame_indices]
    betas = np.asarray(data["betas"], dtype=np.float32).reshape(1, -1)[:, :16]
    betas = np.repeat(betas, len(frame_indices), axis=0)
    gender = normalize_gender(data["gender"])

    bm_path = support_dir / "body_models" / "smplh" / gender / "model.npz"
    dmpl_path = support_dir / "body_models" / "dmpls" / gender / "model.npz"
    if not bm_path.exists():
        bm_path = support_dir / "body_models" / "smplh" / "neutral" / "model.npz"
    if not dmpl_path.exists():
        dmpl_path = support_dir / "body_models" / "dmpls" / "neutral" / "model.npz"

    bm = BodyModel(
        bm_path=str(bm_path),
        num_betas=16,
        model_type="smplh",
        batch_size=len(frame_indices),
        path_dmpl=str(dmpl_path),
    ).to(device)
    body_parms = {
        "root_orient": torch.as_tensor(poses[:, :3], dtype=torch.float32, device=device),
        "pose_body": torch.as_tensor(poses[:, 3:66], dtype=torch.float32, device=device),
        "pose_hand": torch.as_tensor(poses[:, 66:156], dtype=torch.float32, device=device),
        "trans": torch.as_tensor(trans, dtype=torch.float32, device=device),
        "betas": torch.as_tensor(betas, dtype=torch.float32, device=device),
        "dmpls": torch.as_tensor(dmpls, dtype=torch.float32, device=device),
    }
    with torch.no_grad():
        body = bm(**body_parms)
    vertices = c2cpu_array(body.v)
    joints = c2cpu_array(body.Jtr)
    faces = c2c(bm.f)
    return vertices, joints, faces, gender


def c2cpu_array(value: Any) -> np.ndarray:
    return np.asarray(c2c(value), dtype=np.float32)


def smplh_to_coco17(joints: np.ndarray) -> np.ndarray:
    out = np.zeros((joints.shape[0], COCO_NUM_JOINTS, 3), dtype=np.float32)
    for coco_id, smpl_id in SMPLH_TO_COCO17.items():
        out[:, coco_id] = joints[:, smpl_id, :3]
    return out


def build_camera_rig(
    center: np.ndarray,
    height: float,
    width: int,
    image_height: int,
    fov_deg: float,
    distance_scale: float,
    rng: np.random.Generator | None = None,
    distance_jitter: float = 0.0,
    height_jitter: float = 0.0,
    azimuth_jitter_deg: float = 0.0,
    target_jitter: float = 0.0,
    fov_jitter_deg: float = 0.0,
):
    rng = np.random.default_rng(0) if rng is None else rng
    body_height = max(float(height), 1.0)
    base_dist = body_height * float(distance_scale)
    cameras = []
    base_angles = {
        "cam1": math.radians(45.0),
        "cam2": math.radians(135.0),
        "cam3": math.radians(225.0),
        "cam4": math.radians(315.0),
    }
    for view_id, name, _signs in CAMERA_DEFS:
        dist_mul = 1.0 + rng.uniform(-distance_jitter, distance_jitter)
        cam_dist = max(base_dist * dist_mul, body_height * 1.5)
        angle = base_angles[view_id] + math.radians(float(rng.uniform(-azimuth_jitter_deg, azimuth_jitter_deg)))
        z_offset = max(body_height * (0.15 + rng.uniform(-height_jitter, height_jitter)), 0.15)
        target = center + np.array(
            [
                rng.uniform(-target_jitter, target_jitter) * body_height,
                rng.uniform(-target_jitter, target_jitter) * body_height,
                rng.uniform(-target_jitter * 0.5, target_jitter * 0.5) * body_height,
            ],
            dtype=np.float64,
        )
        position = center + np.array([cam_dist * math.cos(angle), cam_dist * math.sin(angle), z_offset], dtype=np.float64)
        sample_fov = float(fov_deg) + float(rng.uniform(-fov_jitter_deg, fov_jitter_deg))
        sample_fov = float(np.clip(sample_fov, 35.0, 90.0))
        K = compute_camera_intrinsics(width, image_height, sample_fov)
        R, t = compute_extrinsics(position, target)
        cameras.append(
            {
                "id": view_id,
                "name": name,
                "K": K,
                "R": R,
                "t": t,
                "position": position,
                "target": target,
                "fov_degree": sample_fov,
            }
        )
    return cameras


def axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9 or abs(float(angle_rad)) < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    x, y, z = axis.tolist()
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float((np.trace(rotation) - 1.0) * 0.5)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ], dtype=np.float64) / max(2.0 * math.sin(angle), 1e-8)
    return (axis * angle).astype(np.float32)


def camera_rotation_delta_axis_angle(clean_camera: dict[str, Any], noisy_camera: dict[str, Any]) -> np.ndarray:
    clean_rotation = np.asarray(clean_camera["R"], dtype=np.float64).reshape(3, 3)
    noisy_rotation = np.asarray(noisy_camera["R"], dtype=np.float64).reshape(3, 3)
    # World-ray correction: direction_clean = (R_clean.T @ R_noisy) @ direction_noisy.
    correction = clean_rotation.T @ noisy_rotation
    return rotation_matrix_to_axis_angle(correction)


def camera_scale_delta_log(clean_camera: dict[str, Any], noisy_camera: dict[str, Any], center: np.ndarray) -> float:
    center = np.asarray(center, dtype=np.float64).reshape(3)
    clean_distance = float(np.linalg.norm(np.asarray(clean_camera["position"], dtype=np.float64).reshape(3) - center))
    noisy_distance = float(np.linalg.norm(np.asarray(noisy_camera["position"], dtype=np.float64).reshape(3) - center))
    return float(math.log(max(clean_distance, 1e-6) / max(noisy_distance, 1e-6)))


def perturb_camera_rig_for_noisy_extrinsics(
    cameras: list[dict[str, Any]],
    center: np.ndarray,
    body_height: float,
    rng: np.random.Generator,
    rotation_noise_deg: float = 0.0,
    translation_noise: float = 0.0,
    scale_noise: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Return noisy calibration cameras while preserving clean render cameras.

    The rendered images and 2D detections come from clean cameras. These noisy
    cameras are used only to build ray tokens and rough triangulation anchors,
    simulating rough extrinsics at inference time.
    """

    if rotation_noise_deg <= 0.0 and translation_noise <= 0.0 and scale_noise <= 0.0:
        return cameras, {
            "rotation_noise_deg": 0.0,
            "translation_noise_m": 0.0,
            "scale_noise": 0.0,
        }

    noisy_cameras: list[dict[str, Any]] = []
    rotation_samples: list[float] = []
    translation_samples: list[float] = []
    scale_samples: list[float] = []
    body_height = max(float(body_height), 1.0)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    for camera in cameras:
        noisy = dict(camera)
        clean_position = np.asarray(camera["position"], dtype=np.float64).reshape(3)
        clean_target = np.asarray(camera["target"], dtype=np.float64).reshape(3)
        clean_forward = clean_target - clean_position
        forward_norm = max(float(np.linalg.norm(clean_forward)), 1e-9)

        sampled_scale = 1.0 + float(rng.uniform(-scale_noise, scale_noise)) if scale_noise > 0.0 else 1.0
        scaled_position = center + (clean_position - center) * sampled_scale

        sampled_translation = (
            rng.normal(0.0, float(translation_noise) * body_height, size=3)
            if translation_noise > 0.0
            else np.zeros(3, dtype=np.float64)
        )
        noisy_position = scaled_position + sampled_translation

        sampled_angle = math.radians(float(rng.uniform(-rotation_noise_deg, rotation_noise_deg))) if rotation_noise_deg > 0.0 else 0.0
        sampled_axis = rng.normal(0.0, 1.0, size=3)
        rot = axis_angle_rotation(sampled_axis, sampled_angle)
        noisy_forward = rot @ (clean_forward / forward_norm)
        noisy_target = noisy_position + noisy_forward * forward_norm

        R, t = compute_extrinsics(noisy_position, noisy_target)
        noisy["position"] = noisy_position
        noisy["target"] = noisy_target
        noisy["R"] = R
        noisy["t"] = t
        noisy["extrinsic_noise"] = {
            "rotation_noise_deg": abs(math.degrees(sampled_angle)),
            "translation_noise_m": float(np.linalg.norm(sampled_translation)),
            "scale_noise": abs(sampled_scale - 1.0),
        }
        noisy_cameras.append(noisy)
        rotation_samples.append(abs(math.degrees(sampled_angle)))
        translation_samples.append(float(np.linalg.norm(sampled_translation)))
        scale_samples.append(abs(sampled_scale - 1.0))

    return noisy_cameras, {
        "rotation_noise_deg": float(np.mean(rotation_samples)) if rotation_samples else 0.0,
        "translation_noise_m": float(np.mean(translation_samples)) if translation_samples else 0.0,
        "scale_noise": float(np.mean(scale_samples)) if scale_samples else 0.0,
    }


def sample_active_view_ids(cameras: list[dict[str, Any]], rng: np.random.Generator, dropout_prob: float, min_active_views: int) -> set[str]:
    ids = [str(cam["id"]) for cam in cameras]
    if dropout_prob <= 0.0:
        return set(ids)
    active = [view_id for view_id in ids if rng.random() >= dropout_prob]
    min_active_views = max(1, min(int(min_active_views), len(ids)))
    if len(active) < min_active_views:
        inactive = [view_id for view_id in ids if view_id not in active]
        rng.shuffle(inactive)
        active.extend(inactive[: min_active_views - len(active)])
    return set(active)


def render_frame(renderer: OffscreenRenderer, scene: Scene, camera: dict[str, Any], verts: np.ndarray, faces: np.ndarray, fov_deg: float) -> np.ndarray:
    scene.clear()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=np.tile(colors["grey"], (verts.shape[0], 1)))
    scene.add(Mesh.from_trimesh(mesh, smooth=False))
    pose = camera_pose_opengl(camera["position"], camera["target"])
    cam_node = scene.add(PerspectiveCamera(yfov=math.radians(fov_deg)), pose=pose)
    light_node = scene.add(DirectionalLight([1.0, 1.0, 1.0], 3.0), pose=pose)
    color, _ = renderer.render(scene)
    scene.remove_node(cam_node)
    scene.remove_node(light_node)
    return color


def yolo_keypoints(model: YOLO, image_rgb: np.ndarray, yolo_device: str | int | None = None) -> np.ndarray:
    kwargs = {"verbose": False, "conf": 0.01}
    if yolo_device is not None:
        kwargs["device"] = yolo_device
    result = model(image_rgb, **kwargs)[0]
    out = np.zeros((COCO_NUM_JOINTS, 3), dtype=np.float32)
    if result.keypoints is None or result.keypoints.data is None or len(result.keypoints.data) == 0:
        return out
    keypoints = result.keypoints.data.detach().cpu().numpy()
    # Synthetic render has one subject; select the detection with highest summed confidence.
    scores = keypoints[:, :, 2].sum(axis=1)
    best = keypoints[int(np.argmax(scores)), :COCO_NUM_JOINTS, :3]
    out[: best.shape[0]] = best.astype(np.float32)
    return out


def camera_origin(camera: dict[str, Any]) -> np.ndarray:
    return (-camera["R"].T @ camera["t"].reshape(3)).astype(np.float32)


def backproject_direction(camera: dict[str, Any], x: float, y: float) -> np.ndarray:
    # pyrender uses an OpenGL camera convention: the camera looks along -Z and
    # image Y points downward. Convert pixels to that convention before rotating
    # the ray back to world coordinates.
    fx = float(camera["K"][0, 0])
    fy = float(camera["K"][1, 1])
    cx = float(camera["K"][0, 2])
    cy = float(camera["K"][1, 2])
    cam_dir = np.array([(x - cx) / fx, -(y - cy) / fy, -1.0], dtype=np.float64)
    cam_dir /= max(float(np.linalg.norm(cam_dir)), 1e-9)
    world_dir = camera["R"].T @ cam_dir
    world_dir /= max(float(np.linalg.norm(world_dir)), 1e-9)
    return world_dir.astype(np.float32)

def root_relative(points: np.ndarray) -> np.ndarray:
    pelvis = 0.5 * (points[11] + points[12])
    return (points - pelvis.reshape(1, 3)).astype(np.float32)


def triangulate_weighted_rays(origins: list[np.ndarray], directions: list[np.ndarray], weights: list[float]) -> tuple[np.ndarray, float]:
    if len(origins) < 2:
        return np.zeros(3, dtype=np.float32), 0.0
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    weight_sum = 0.0
    for origin, direction, weight in zip(origins, directions, weights):
        d = np.asarray(direction, dtype=np.float64)
        d /= max(float(np.linalg.norm(d)), 1e-9)
        projection = eye - np.outer(d, d)
        w = max(float(weight), 1e-6)
        a += w * projection
        b += w * projection @ np.asarray(origin, dtype=np.float64)
        weight_sum += w
    try:
        point = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float32), 0.0

    distances = []
    for origin, direction in zip(origins, directions):
        d = np.asarray(direction, dtype=np.float64)
        d /= max(float(np.linalg.norm(d)), 1e-9)
        offset = point - np.asarray(origin, dtype=np.float64)
        distances.append(float(np.linalg.norm(offset - np.dot(offset, d) * d)))
    mean_distance = float(np.mean(distances)) if distances else 1e6
    quality = float(len(origins)) * float(weight_sum / max(len(origins), 1)) / (1.0 + mean_distance)
    return point.astype(np.float32), quality


def triangulate_from_yolo_views(
    yolo_by_view: dict[str, np.ndarray],
    cameras: list[dict[str, Any]],
    active_view_ids: set[str],
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_3d = np.zeros((COCO_NUM_JOINTS, 3), dtype=np.float32)
    quality = np.zeros((COCO_NUM_JOINTS,), dtype=np.float32)
    valid = np.zeros((COCO_NUM_JOINTS,), dtype=np.bool_)
    for joint_id in range(COCO_NUM_JOINTS):
        origins: list[np.ndarray] = []
        directions: list[np.ndarray] = []
        weights: list[float] = []
        for camera in cameras:
            if str(camera["id"]) not in active_view_ids:
                continue
            x, y, conf = [float(v) for v in yolo_by_view[camera["id"]][joint_id]]
            if conf < confidence_threshold or not np.isfinite(x) or not np.isfinite(y):
                continue
            origins.append(camera_origin(camera).astype(np.float32))
            directions.append(backproject_direction(camera, x, y))
            weights.append(conf * conf)
        point, score = triangulate_weighted_rays(origins, directions, weights)
        points_3d[joint_id] = point
        quality[joint_id] = np.float32(score)
        valid[joint_id] = len(origins) >= 2 and score > 0.0
    return points_3d, quality, valid


def make_stage_a_tokens(
    yolo_by_view: dict[str, np.ndarray],
    cameras: list[dict[str, Any]],
    width: int,
    height: int,
    confidence_threshold: float,
    use_face_targets: bool = False,
    active_view_ids: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tokens = np.zeros((COCO_NUM_JOINTS, len(cameras), RAY_FEATURE_DIM), dtype=np.float32)
    active_view_ids = {str(cam["id"]) for cam in cameras} if active_view_ids is None else active_view_ids
    view_mask = np.zeros((len(cameras),), dtype=np.bool_)
    joint_view_mask = np.zeros((COCO_NUM_JOINTS, len(cameras)), dtype=np.bool_)
    target_confidence = np.zeros((COCO_NUM_JOINTS,), dtype=np.float32)
    origins = np.vstack([camera_origin(cam) for cam in cameras])
    origin_center = origins.mean(axis=0)
    origin_scale = max(float(np.linalg.norm(origins - origin_center.reshape(1, 3), axis=1).max()), 1.0)

    for view_index, camera in enumerate(cameras):
        if str(camera["id"]) not in active_view_ids:
            continue
        view_mask[view_index] = True
        points = yolo_by_view[camera["id"]]
        origin_norm = (camera_origin(camera) - origin_center) / origin_scale
        for joint_id in range(COCO_NUM_JOINTS):
            x, y, conf = [float(v) for v in points[joint_id]]
            if not np.isfinite(x) or not np.isfinite(y) or conf <= 0.0:
                continue
            direction = backproject_direction(camera, x, y)
            x_norm = (x / float(width)) * 2.0 - 1.0
            y_norm = (y / float(height)) * 2.0 - 1.0
            tokens[joint_id, view_index] = [
                float(origin_norm[0]),
                float(origin_norm[1]),
                float(origin_norm[2]),
                float(direction[0]),
                float(direction[1]),
                float(direction[2]),
                float(x_norm),
                float(y_norm),
                float(conf),
                view_index / max(len(cameras) - 1, 1),
                joint_id / max(COCO_NUM_JOINTS - 1, 1),
            ]
            if conf >= confidence_threshold and 0.0 <= x < width and 0.0 <= y < height:
                joint_view_mask[joint_id, view_index] = True
            target_confidence[joint_id] = max(target_confidence[joint_id], np.float32(conf))
    if not use_face_targets:
        target_confidence[:5] = 0.0
    return tokens, view_mask, joint_view_mask, target_confidence


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    rendered_dir = output_dir / "rendered_frames"
    annotated_dir = output_dir / "annotated_frames"
    samples_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rendered_frames:
        rendered_dir.mkdir(parents=True, exist_ok=True)
    if args.save_annotated_frames:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    source_paths = [Path(path) for path in sorted(glob.glob(args.source_glob, recursive=True))]
    if args.max_sequences > 0:
        source_paths = source_paths[: args.max_sequences]
    if not source_paths:
        raise FileNotFoundError(f"No AMASS files matched: {args.source_glob}")

    rng = np.random.default_rng(int(args.seed))
    yolo_device: str | int | None
    if str(args.yolo_device).lower() != "auto":
        yolo_device = int(args.yolo_device) if str(args.yolo_device).isdigit() else str(args.yolo_device)
    elif device.type == "cuda":
        yolo_device = 0
    else:
        yolo_device = "cpu"
    print(json.dumps({"torch_device": str(device), "yolo_device": yolo_device}, ensure_ascii=False))
    model = YOLO(args.yolo_model)
    renderer = OffscreenRenderer(args.resolution, args.resolution)
    scene = Scene(ambient_light=[0.3, 0.3, 0.3])
    entries: list[dict[str, Any]] = []
    metadata_sequences = []

    try:
        for seq_index, npz_path in enumerate(source_paths):
            raw = np.load(npz_path, allow_pickle=True)
            frame_indices = selected_frame_indices(len(raw["trans"]), args.frame_stride, args.max_frames_per_sequence)
            if len(frame_indices) == 0:
                continue
            vertices, joints_smplh, faces, gender = load_amass_body(npz_path, frame_indices, Path(args.support_dir), device)
            target_coco = smplh_to_coco17(joints_smplh)
            first_verts = vertices[0]
            center = first_verts.mean(axis=0).astype(np.float64)
            body_height = float(first_verts[:, 2].max() - first_verts[:, 2].min())
            sequence_cameras = build_camera_rig(
                center,
                body_height,
                args.resolution,
                args.resolution,
                args.fov_degree,
                args.distance_scale,
                rng=rng,
                distance_jitter=args.distance_jitter,
                height_jitter=args.height_jitter,
                azimuth_jitter_deg=args.azimuth_jitter_deg,
                target_jitter=args.target_jitter,
                fov_jitter_deg=args.fov_jitter_deg,
            )
            seq_name = f"amass_{npz_path.parent.name}_{npz_path.stem}"
            metadata_sequences.append({
                "sequence": seq_name,
                "source_npz": str(npz_path),
                "gender": gender,
                "num_frames": int(len(frame_indices)),
                "frame_indices": frame_indices.astype(int).tolist(),
            })

            for local_idx, source_frame in enumerate(frame_indices):
                clean_cameras = (
                    build_camera_rig(
                        center,
                        body_height,
                        args.resolution,
                        args.resolution,
                        args.fov_degree,
                        args.distance_scale,
                        rng=rng,
                        distance_jitter=args.distance_jitter,
                        height_jitter=args.height_jitter,
                        azimuth_jitter_deg=args.azimuth_jitter_deg,
                        target_jitter=args.target_jitter,
                        fov_jitter_deg=args.fov_jitter_deg,
                    )
                    if args.camera_aug_per_frame
                    else sequence_cameras
                )
                noisy_cameras, extrinsic_noise_summary = perturb_camera_rig_for_noisy_extrinsics(
                    clean_cameras,
                    center=center,
                    body_height=body_height,
                    rng=rng,
                    rotation_noise_deg=float(args.extrinsic_rotation_noise_deg),
                    translation_noise=float(args.extrinsic_translation_noise),
                    scale_noise=float(args.extrinsic_scale_noise),
                )
                active_view_ids = sample_active_view_ids(clean_cameras, rng, args.view_dropout_prob, args.min_active_views)
                yolo_by_view = {}
                for camera in clean_cameras:
                    if str(camera["id"]) not in active_view_ids:
                        yolo_by_view[camera["id"]] = np.zeros((COCO_NUM_JOINTS, 3), dtype=np.float32)
                        continue
                    image_rgb = render_frame(renderer, scene, camera, vertices[local_idx], faces, float(camera.get("fov_degree", args.fov_degree)))
                    kpts = yolo_keypoints(model, image_rgb, yolo_device=yolo_device)
                    yolo_by_view[camera["id"]] = kpts
                    if args.save_rendered_frames:
                        imageio.imwrite(rendered_dir / f"{seq_name}_frame_{int(source_frame):06d}_{camera['id']}.png", image_rgb)
                    if args.save_annotated_frames:
                        annotated = image_rgb.copy()
                        for x, y, conf in kpts:
                            if conf >= args.confidence_threshold:
                                cv2.circle(annotated, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
                        imageio.imwrite(annotated_dir / f"{seq_name}_frame_{int(source_frame):06d}_{camera['id']}.png", annotated)

                tokens, view_mask, joint_view_mask, target_conf = make_stage_a_tokens(
                    yolo_by_view=yolo_by_view,
                    cameras=noisy_cameras,
                    width=args.resolution,
                    height=args.resolution,
                    confidence_threshold=args.confidence_threshold,
                    use_face_targets=bool(args.use_face_targets),
                    active_view_ids=active_view_ids,
                )
                triangulated_3d, triangulated_quality, triangulated_valid = triangulate_from_yolo_views(
                    yolo_by_view=yolo_by_view,
                    cameras=noisy_cameras,
                    active_view_ids=active_view_ids,
                    confidence_threshold=args.confidence_threshold,
                )
                pelvis_anchor = 0.5 * (triangulated_3d[11] + triangulated_3d[12])
                pelvis_anchor_valid = bool(triangulated_valid[11] and triangulated_valid[12])
                pelvis_anchor_quality = np.float32(0.5 * (triangulated_quality[11] + triangulated_quality[12])) if pelvis_anchor_valid else np.float32(0.0)
                camera_origins = np.vstack([camera_origin(cam) for cam in noisy_cameras]).astype(np.float32)
                clean_camera_origins = np.vstack([camera_origin(cam) for cam in clean_cameras]).astype(np.float32)
                origin_center = camera_origins.mean(axis=0).astype(np.float32)
                origin_scale = np.float32(max(float(np.linalg.norm(camera_origins - origin_center.reshape(1, 3), axis=1).max()), 1.0))
                camera_origin_delta = ((clean_camera_origins - camera_origins) / max(float(origin_scale), 1e-6)).astype(np.float32)
                clean_translations = np.vstack([np.asarray(cam["t"], dtype=np.float32).reshape(3) for cam in clean_cameras])
                noisy_translations = np.vstack([np.asarray(cam["t"], dtype=np.float32).reshape(3) for cam in noisy_cameras])
                camera_translation_delta = ((clean_translations - noisy_translations) / max(float(origin_scale), 1e-6)).astype(np.float32)
                camera_rotation_delta = np.vstack([
                    camera_rotation_delta_axis_angle(clean_cam, noisy_cam)
                    for clean_cam, noisy_cam in zip(clean_cameras, noisy_cameras)
                ]).astype(np.float32)
                camera_scale_delta = np.asarray([
                    camera_scale_delta_log(clean_cam, noisy_cam, center)
                    for clean_cam, noisy_cam in zip(clean_cameras, noisy_cameras)
                ], dtype=np.float32)
                camera_origin_delta_valid = np.asarray([str(cam["id"]) in active_view_ids for cam in noisy_cameras], dtype=np.bool_)
                sample_id = f"{seq_name}_frame_{int(source_frame):06d}"
                sample_path = samples_dir / f"{sample_id}.npz"
                np.savez_compressed(
                    sample_path,
                    ray_tokens=tokens.astype(np.float32),
                    view_mask=view_mask.astype(np.bool_),
                    joint_view_mask=joint_view_mask.astype(np.bool_),
                    target_3d=target_coco[local_idx].astype(np.float32),
                    target_3d_root_relative=root_relative(target_coco[local_idx]),
                    target_confidence=target_conf.astype(np.float32),
                    triangulated_3d=triangulated_3d.astype(np.float32),
                    triangulated_3d_root_relative=root_relative(triangulated_3d).astype(np.float32),
                    triangulated_quality=triangulated_quality.astype(np.float32),
                    triangulated_valid=triangulated_valid.astype(np.bool_),
                    pelvis_anchor=pelvis_anchor.astype(np.float32),
                    pelvis_anchor_quality=np.asarray(pelvis_anchor_quality, dtype=np.float32),
                    pelvis_anchor_valid=np.asarray(pelvis_anchor_valid, dtype=np.bool_),
                    ray_origin_center=origin_center.astype(np.float32),
                    ray_origin_scale=np.asarray(origin_scale, dtype=np.float32),
                    camera_origin_delta=camera_origin_delta.astype(np.float32),
                    camera_translation_delta=camera_translation_delta.astype(np.float32),
                    camera_rotation_delta=camera_rotation_delta.astype(np.float32),
                    camera_scale_delta=camera_scale_delta.astype(np.float32),
                    camera_origin_delta_valid=camera_origin_delta_valid.astype(np.bool_),
                    source_frame=np.array(int(source_frame), dtype=np.int32),
                    views=np.asarray([cam["id"] for cam in noisy_cameras]),
                    extrinsic_noise_rotation_deg=np.asarray(extrinsic_noise_summary["rotation_noise_deg"], dtype=np.float32),
                    extrinsic_noise_translation_m=np.asarray(extrinsic_noise_summary["translation_noise_m"], dtype=np.float32),
                    extrinsic_noise_scale=np.asarray(extrinsic_noise_summary["scale_noise"], dtype=np.float32),
                )
                entries.append({
                    "id": sample_id,
                    "path": str(sample_path),
                    "split": args.split,
                    "sequence": seq_name,
                    "frame_id": int(source_frame),
                    "person_id": "amass_person01",
                    "source_npz": str(npz_path),
                    "active_joint_views": int(joint_view_mask.sum()),
                    "active_views": sorted(active_view_ids),
                    "extrinsic_noise": extrinsic_noise_summary,
                })
                print(json.dumps({"sample": sample_id, "active_views": sorted(active_view_ids), "active_joint_views": int(joint_view_mask.sum())}, ensure_ascii=False))
    finally:
        renderer.delete()

    manifest = {
        "metadata": {
            "stage": "amass_yolo_stage_a_dataset",
            "source_glob": args.source_glob,
            "num_entries": len(entries),
            "feature_dim": RAY_FEATURE_DIM,
            "num_joints": COCO_NUM_JOINTS,
            "max_views": len(CAMERA_DEFS),
            "resolution": [args.resolution, args.resolution],
            "fov_degree": args.fov_degree,
            "torch_device": str(device),
            "yolo_device": yolo_device,
            "augmentation": {
                "seed": int(args.seed),
                "distance_jitter": float(args.distance_jitter),
                "height_jitter": float(args.height_jitter),
                "azimuth_jitter_deg": float(args.azimuth_jitter_deg),
                "target_jitter": float(args.target_jitter),
                "fov_jitter_deg": float(args.fov_jitter_deg),
                "view_dropout_prob": float(args.view_dropout_prob),
                "min_active_views": int(args.min_active_views),
                "camera_aug_per_frame": bool(args.camera_aug_per_frame),
                "extrinsic_rotation_noise_deg": float(args.extrinsic_rotation_noise_deg),
                "extrinsic_translation_noise": float(args.extrinsic_translation_noise),
                "extrinsic_scale_noise": float(args.extrinsic_scale_noise),
            },
            "confidence_threshold": args.confidence_threshold,
            "use_face_targets": bool(args.use_face_targets),
            "ignored_target_joints": [] if args.use_face_targets else [0, 1, 2, 3, 4],
            "sequences": metadata_sequences,
        },
        "entries": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "num_entries": len(entries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
