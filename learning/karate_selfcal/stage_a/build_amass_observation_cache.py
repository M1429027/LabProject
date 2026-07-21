"""Build clean, fixed-rig AMASS observations for camera-head datasets."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from body_visualizer.tools.vis_tools import colors
from pyrender import DirectionalLight, Mesh, OffscreenRenderer, PerspectiveCamera, Scene
from ultralytics import YOLO
from ..detection.detectors import YOLOHRNetTopDownDetector

from .build_amass_yolo_dataset import (
    CAMERA_DEFS,
    COCO_NUM_JOINTS,
    backproject_direction,
    camera_origin,
    camera_pose_opengl,
    compute_camera_intrinsics,
    compute_extrinsics,
    load_amass_body,
    make_stage_a_tokens,
    root_relative,
    selected_frame_indices,
    smplh_to_coco17,

)


COCO_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)

RIG_PRESETS = {
    # A close-range baseline keeps the body large enough for top-down pose estimation.
    "near": {"radius": (3.5, 5.0), "height": (1.3, 2.2), "fov": (30.0, 42.0)},
    "standard": {"radius": (8.0, 15.0), "height": (1.5, 2.5), "fov": (35.0, 60.0)},
    "far": {"radius": (15.0, 25.0), "height": (2.0, 4.0), "fov": (20.0, 45.0)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--support-dir", default="support_data")
    parser.add_argument("--yolo-model", default="yolo11l-pose.pt")
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--sample-sequences-randomly", action="store_true")
    parser.add_argument("--max-frames-per-sequence", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--field-size-m", type=float, default=8.0)
    parser.add_argument("--field-position-jitter-m", type=float, default=1.5)
    parser.add_argument("--rig-preset", choices=["near", "standard", "far", "mixture"], default="standard")
    parser.add_argument(
        "--rig-layout",
        choices=["random", "diagonal45"],
        default="diagonal45",
        help="Place four fixed cameras randomly or at 45-degree diagonal offsets from the reference body heading.",
    )
    parser.add_argument("--rig-radius-range", type=float, nargs=2, default=None, metavar=("MIN_M", "MAX_M"))
    parser.add_argument("--rig-fov-range", type=float, nargs=2, default=None, metavar=("MIN_DEG", "MAX_DEG"))
    parser.add_argument("--azimuth-jitter-deg", type=float, default=6.0)
    parser.add_argument("--target-height-m", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument(
        "--min-person-pixel-height",
        type=float,
        default=0.0,
        help="Require this GT body height per qualifying view; 0 disables the gate.",
    )
    parser.add_argument(
        "--min-visible-body-joints",
        type=int,
        default=10,
        help="Minimum in-frame body joints (COCO 5-16) for one qualifying view.",
    )
    parser.add_argument(
        "--min-qualifying-views",
        type=int,
        default=4,
        help="Minimum views meeting the GT pixel-height and visibility gates.",
    )
    parser.add_argument(
        "--rig-resample-attempts",
        type=int,
        default=12,
        help="Fixed-rig proposals tried before rejecting a sequence.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-view-images", action="store_true")
    parser.add_argument("--preview-limit", type=int, default=12, help="Maximum number of four-view previews; negative saves all.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--yolo-device", default="auto")
    parser.add_argument("--occlusion-probability", type=float, default=0.0)
    parser.add_argument(
        "--occlusion-mode",
        choices=["physical3d", "legacy2d"],
        default="physical3d",
        help="Use a world-space blocker (recommended) or the legacy per-image overlay.",
    )
    parser.add_argument("--physical-occluder-attempts", type=int, default=64)
    parser.add_argument("--physical-occluder-min-views", type=int, default=2)
    parser.add_argument("--physical-occluder-max-views", type=int, default=3)
    parser.add_argument("--physical-occluder-min-body-joints", type=int, default=2)
    parser.add_argument("--physical-occluder-max-body-joints", type=int, default=8)
    parser.add_argument("--occlusion-min-width-ratio", type=float, default=0.10)
    parser.add_argument("--occlusion-max-width-ratio", type=float, default=0.22)
    parser.add_argument("--occlusion-min-height-ratio", type=float, default=0.30)
    parser.add_argument("--occlusion-max-height-ratio", type=float, default=0.55)
    parser.add_argument("--pose-backend", choices=["yolo_pose", "hrnet"], default="yolo_pose")
    parser.add_argument(
        "--hrnet-config",
        default="learning/karate_selfcal/checkpoints/hrnet_w32_coco_256x192/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
    )
    parser.add_argument(
        "--hrnet-checkpoint",
        default="learning/karate_selfcal/checkpoints/hrnet_w32_coco_256x192/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth",
    )
    parser.add_argument("--hrnet-bbox-scale", type=float, default=1.25)
    return parser.parse_args()


def choose_preset(name: str, rng: np.random.Generator) -> str:
    if name != "mixture":
        return name
    return str(rng.choice(["near", "standard", "far"], p=[0.2, 0.6, 0.2]))


def build_karate_rig(
    width: int,
    height: int,
    preset_name: str,
    azimuth_jitter_deg: float,
    target_height_m: float,
    rng: np.random.Generator,
    layout: str = "diagonal45",
    reference_facing_yaw_deg: float | None = None,
    radius_range: tuple[float, float] | None = None,
    fov_range: tuple[float, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preset = RIG_PRESETS[preset_name]
    radius_range = preset["radius"] if radius_range is None else radius_range
    fov_range = preset["fov"] if fov_range is None else fov_range
    base_radius = float(rng.uniform(*radius_range))
    base_height = float(rng.uniform(*preset["height"]))
    base_fov = float(rng.uniform(*fov_range))
    if layout == "diagonal45" and reference_facing_yaw_deg is not None:
        global_yaw_deg = float(reference_facing_yaw_deg + 45.0)
    else:
        global_yaw_deg = float(rng.uniform(0.0, 360.0))
    cameras: list[dict[str, Any]] = []

    for index, (view_id, view_name, _signs) in enumerate(CAMERA_DEFS):
        angle_deg = global_yaw_deg + index * 90.0 + float(rng.uniform(-azimuth_jitter_deg, azimuth_jitter_deg))
        angle = math.radians(angle_deg)
        radius = base_radius * float(rng.uniform(0.92, 1.08))
        camera_height = max(0.8, base_height + float(rng.uniform(-0.15, 0.15)))
        fov_deg = float(np.clip(base_fov + rng.uniform(-3.0, 3.0), 18.0, 75.0))
        position = np.array([radius * math.cos(angle), radius * math.sin(angle), camera_height], dtype=np.float64)
        target = np.array([0.0, 0.0, target_height_m + float(rng.uniform(-0.1, 0.1))], dtype=np.float64)
        intrinsic = compute_camera_intrinsics(width, height, fov_deg)
        rotation, translation = compute_extrinsics(position, target)
        cameras.append({
            "id": view_id,
            "name": view_name,
            "K": intrinsic,
            "R": rotation,
            "t": translation,
            "position": position,
            "target": target,
            "fov_degree": fov_deg,
            "azimuth_degree": angle_deg % 360.0,
        })

    rig_metadata = {
        "preset": preset_name,
        "layout": layout,
        "global_yaw_degree": global_yaw_deg,
        "reference_facing_yaw_degree": reference_facing_yaw_deg,
        "base_radius_m": base_radius,
        "base_height_m": base_height,
        "base_fov_degree": base_fov,
        "azimuth_jitter_degree": azimuth_jitter_deg,
        "radius_range_m": list(radius_range),
        "fov_range_degree": list(fov_range),
    }
    return cameras, rig_metadata


def place_sequence_in_field(
    vertices: np.ndarray,
    joints: np.ndarray,
    target_coco: np.ndarray,
    field_position_jitter_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pelvis = 0.5 * (target_coco[0, 11] + target_coco[0, 12])
    floor_z = float(vertices[0, :, 2].min())
    field_xy = rng.uniform(-field_position_jitter_m, field_position_jitter_m, size=2)
    offset = np.array([field_xy[0] - pelvis[0], field_xy[1] - pelvis[1], -floor_z], dtype=np.float32)
    return vertices + offset, joints + offset, target_coco + offset, offset


def reference_body_facing_yaw_deg(target_coco: np.ndarray) -> float:
    """Estimate a stable horizontal heading from shoulders, falling back to hips."""

    shoulder_axis = target_coco[6, :2] - target_coco[5, :2]
    if float(np.linalg.norm(shoulder_axis)) < 1e-5:
        shoulder_axis = target_coco[12, :2] - target_coco[11, :2]
    if float(np.linalg.norm(shoulder_axis)) < 1e-5:
        return 0.0
    # The normal gives the body forward/back axis. Its sign is irrelevant for
    # a four-camera diagonal rig because a 180-degree shift yields the same set.
    facing = np.asarray([-shoulder_axis[1], shoulder_axis[0]], dtype=np.float64)
    return float(math.degrees(math.atan2(facing[1], facing[0])) % 360.0)


def render_karate_frame(
    renderer: OffscreenRenderer,
    scene: Scene,
    camera: dict[str, Any],
    vertices: np.ndarray,
    faces: np.ndarray,
    field_size_m: float,
    physical_occluder: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    scene.clear()
    floor = trimesh.creation.box(extents=(field_size_m, field_size_m, 0.04))
    floor.apply_translation((0.0, 0.0, -0.02))
    floor.visual.vertex_colors = np.tile(np.array([52, 105, 76, 255], dtype=np.uint8), (len(floor.vertices), 1))
    scene.add(Mesh.from_trimesh(floor, smooth=False))

    body = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=np.tile(colors["grey"], (vertices.shape[0], 1)),
    )
    scene.add(Mesh.from_trimesh(body, smooth=False))
    if physical_occluder is not None:
        blocker = trimesh.creation.box(extents=physical_occluder["extents"])
        blocker.apply_translation(physical_occluder["center"])
        blocker.visual.vertex_colors = np.tile(
            np.asarray([151, 75, 43, 255], dtype=np.uint8), (len(blocker.vertices), 1)
        )
        scene.add(Mesh.from_trimesh(blocker, smooth=False))
    pose = camera_pose_opengl(camera["position"], camera["target"])
    camera_node = scene.add(PerspectiveCamera(yfov=math.radians(float(camera["fov_degree"]))), pose=pose)
    light_node = scene.add(DirectionalLight([1.0, 1.0, 1.0], 3.0), pose=pose)
    color, _depth = renderer.render(scene)
    scene.remove_node(camera_node)
    scene.remove_node(light_node)
    return color


def project_gt(points_3d: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    camera_points = (camera["R"] @ points_3d.T).T + camera["t"].reshape(1, 3)
    depth = -camera_points[:, 2]
    valid = depth > 1e-6
    projected = np.zeros((len(points_3d), 3), dtype=np.float32)
    projected[valid, 0] = camera["K"][0, 0] * camera_points[valid, 0] / depth[valid] + camera["K"][0, 2]
    projected[valid, 1] = camera["K"][1, 1] * (-camera_points[valid, 1]) / depth[valid] + camera["K"][1, 2]
    projected[valid, 2] = 1.0
    return projected


def projected_body_quality(
    projected_gt: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, int]:
    """Measure detector-relevant body scale before running the pose backend."""

    body = projected_gt[5:17]
    valid = (
        (body[:, 2] > 0.0)
        & (body[:, 0] >= 0.0)
        & (body[:, 0] < width)
        & (body[:, 1] >= 0.0)
        & (body[:, 1] < height)
    )
    visible = int(valid.sum())
    pixel_height = float(np.ptp(body[valid, 1])) if visible >= 2 else 0.0
    return pixel_height, visible


def view_quality(
    target_3d: np.ndarray,
    cameras: list[dict[str, Any]],
    width: int,
    height: int,
    min_pixel_height: float,
    min_visible_joints: int,
) -> tuple[list[np.ndarray], list[float], list[int], int]:
    """Return projected poses and the number of views suitable for pose estimation."""

    projections = [project_gt(target_3d, camera) for camera in cameras]
    qualities = [projected_body_quality(projection, width, height) for projection in projections]
    pixel_heights = [item[0] for item in qualities]
    visible_joints = [item[1] for item in qualities]
    qualifying = sum(
        visible >= int(min_visible_joints)
        and (float(min_pixel_height) <= 0.0 or pixels >= float(min_pixel_height))
        for pixels, visible in qualities
    )
    return projections, pixel_heights, visible_joints, int(qualifying)


def segment_intersects_aabb(
    origin: np.ndarray,
    target: np.ndarray,
    center: np.ndarray,
    extents: np.ndarray,
) -> bool:
    """Return whether the camera-to-joint segment crosses a world-space box."""

    direction = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    lower = np.asarray(center, dtype=np.float64) - 0.5 * np.asarray(extents, dtype=np.float64)
    upper = np.asarray(center, dtype=np.float64) + 0.5 * np.asarray(extents, dtype=np.float64)
    t_enter, t_exit = 0.0, 1.0
    for axis in range(3):
        if abs(float(direction[axis])) < 1e-8:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return False
            continue
        near = (lower[axis] - origin[axis]) / direction[axis]
        far = (upper[axis] - origin[axis]) / direction[axis]
        if near > far:
            near, far = far, near
        t_enter = max(t_enter, float(near))
        t_exit = min(t_exit, float(far))
        if t_enter > t_exit:
            return False
    # Ignore intersections at the camera itself or exactly on the labelled joint.
    return t_exit > 1e-3 and t_enter < 0.985


def physical_occlusion_joint_mask(
    target_3d: np.ndarray,
    cameras: list[dict[str, Any]],
    occluder: dict[str, np.ndarray],
) -> np.ndarray:
    mask = np.zeros((len(cameras), COCO_NUM_JOINTS), dtype=np.bool_)
    for view_index, camera in enumerate(cameras):
        for joint_index in range(5, COCO_NUM_JOINTS):
            mask[view_index, joint_index] = segment_intersects_aabb(
                camera_origin(camera), target_3d[joint_index], occluder["center"], occluder["extents"]
            )
    return mask


def sample_physical_occluder(
    target_3d: np.ndarray,
    cameras: list[dict[str, Any]],
    rng: np.random.Generator,
    attempts: int,
    min_views: int,
    max_views: int,
    min_body_joints: int,
    max_body_joints: int,
    required_view_index: int | None = None,
) -> tuple[dict[str, np.ndarray] | None, np.ndarray]:
    """Sample one upright world-space blocker that partially occludes 2--3 cameras."""

    torso = target_3d[[5, 6, 11, 12]].mean(axis=0)
    body_height = max(float(target_3d[5:17, 2].ptp()), 1.2)
    for _ in range(max(1, int(attempts))):
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        radial_offset = float(rng.uniform(0.28, 0.72))
        center = torso.copy()
        center[:2] += radial_offset * np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32)
        center[2] = float(np.clip(torso[2] + rng.uniform(-0.18, 0.18), 0.45, body_height - 0.15))
        extents = np.asarray([
            rng.uniform(0.32, 0.58),
            rng.uniform(0.32, 0.58),
            rng.uniform(0.72, min(1.35, body_height * 0.82)),
        ], dtype=np.float32)
        occluder = {"center": center.astype(np.float32), "extents": extents}
        joint_mask = physical_occlusion_joint_mask(target_3d, cameras, occluder)
        counts = joint_mask[:, 5:17].sum(axis=1)
        view_count = int((counts >= int(min_body_joints)).sum())
        if (
            int(min_views) <= view_count <= int(max_views)
            and int(counts.max()) <= int(max_body_joints)
            and (required_view_index is None or counts[int(required_view_index)] >= int(min_body_joints))
        ):
            return occluder, joint_mask
    return None, np.zeros((len(cameras), COCO_NUM_JOINTS), dtype=np.bool_)


def apply_local_occluder(
    image_rgb: np.ndarray,
    projected_gt: np.ndarray,
    rng: np.random.Generator,
    min_width_ratio: float,
    max_width_ratio: float,
    min_height_ratio: float,
    max_height_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Place one opaque, body-overlapping obstacle without covering most of the person."""

    height, width = image_rgb.shape[:2]
    valid = (
        (projected_gt[:, 2] > 0.0)
        & (projected_gt[:, 0] >= 0.0)
        & (projected_gt[:, 0] < width)
        & (projected_gt[:, 1] >= 0.0)
        & (projected_gt[:, 1] < height)
    )
    if int(valid.sum()) < 5:
        return image_rgb, np.full((4,), -1, dtype=np.int32), 0.0
    points = projected_gt[valid, :2]
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    body_width = max(float(x_max - x_min), 1.0)
    body_height = max(float(y_max - y_min), 1.0)
    obstacle_width = body_width * float(rng.uniform(min_width_ratio, max_width_ratio))
    obstacle_height = body_height * float(rng.uniform(min_height_ratio, max_height_ratio))
    center_x = float(rng.uniform(x_min + 0.25 * body_width, x_max - 0.25 * body_width))
    center_y = float(rng.uniform(y_min + 0.30 * body_height, y_max - 0.20 * body_height))
    left = int(np.clip(round(center_x - 0.5 * obstacle_width), 0, width - 1))
    right = int(np.clip(round(center_x + 0.5 * obstacle_width), left + 1, width))
    top = int(np.clip(round(center_y - 0.5 * obstacle_height), 0, height - 1))
    bottom = int(np.clip(round(center_y + 0.5 * obstacle_height), top + 1, height))

    output = image_rgb.copy()
    base = rng.integers(35, 135, size=3, dtype=np.uint8)
    output[top:bottom, left:right] = base
    stripe_step = max(5, (right - left) // 5)
    stripe = np.clip(base.astype(np.int16) + 24, 0, 255).astype(np.uint8)
    for stripe_x in range(left, right, stripe_step * 2):
        output[top:bottom, stripe_x : min(stripe_x + stripe_step, right)] = stripe
    cv2.rectangle(output, (left, top), (right - 1, bottom - 1), (20, 20, 20), 2)
    body_area = body_width * body_height
    area_ratio = float((right - left) * (bottom - top) / max(body_area, 1.0))
    return output, np.asarray([left, top, right, bottom], dtype=np.int32), area_ratio

def draw_pose(image_rgb: np.ndarray, keypoints: np.ndarray, threshold: float, label: str) -> np.ndarray:
    output = image_rgb.copy()
    for start, end in COCO_EDGES:
        if keypoints[start, 2] >= threshold and keypoints[end, 2] >= threshold:
            p1 = tuple(np.rint(keypoints[start, :2]).astype(int))
            p2 = tuple(np.rint(keypoints[end, :2]).astype(int))
            cv2.line(output, p1, p2, (255, 210, 40), 2, cv2.LINE_AA)
    for x, y, confidence in keypoints:
        if confidence >= threshold:
            cv2.circle(output, (int(round(x)), int(round(y))), 4, (255, 60, 60), -1, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (260, 34), (20, 20, 20), -1)
    cv2.putText(output, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return output

def yolo_keypoints_batch(model: YOLO, images_rgb: list[np.ndarray], yolo_device: str | int) -> list[np.ndarray]:
    results = model(images_rgb, verbose=False, conf=0.01, device=yolo_device)
    poses = []
    for result in results:
        output = np.zeros((COCO_NUM_JOINTS, 3), dtype=np.float32)
        if result.keypoints is not None and result.keypoints.data is not None and len(result.keypoints.data) > 0:
            candidates = result.keypoints.data.detach().cpu().numpy()
            best = candidates[int(np.argmax(candidates[:, :, 2].sum(axis=1))), :COCO_NUM_JOINTS, :3]
            output[: best.shape[0]] = best.astype(np.float32)
        poses.append(output)
    return poses
def hrnet_keypoints_batch(
    detector: YOLOHRNetTopDownDetector,
    images_rgb: list[np.ndarray],
) -> list[np.ndarray]:
    """Run top-down HRNet and keep the strongest person in each synthetic view."""

    poses = []
    for image_rgb in images_rgb:
        result = detector.run(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        output = np.zeros((COCO_NUM_JOINTS, 3), dtype=np.float32)
        if result.people:
            person = max(
                result.people,
                key=lambda item: sum(
                    float(point.get("confidence", 0.0))
                    for point in item.get("keypoints", [])
                ),
            )
            for point in person.get("keypoints", []):
                joint_id = int(point["id"])
                if 0 <= joint_id < COCO_NUM_JOINTS:
                    output[joint_id] = (
                        float(point["x"]),
                        float(point["y"]),
                        float(point.get("confidence", 0.0)),
                    )
        poses.append(output)
    return poses


def save_preview(images: list[np.ndarray], path: Path) -> None:
    if len(images) != 4:
        return
    top = np.concatenate(images[:2], axis=1)
    bottom = np.concatenate(images[2:], axis=1)
    imageio.imwrite(path, np.concatenate([top, bottom], axis=0))


def serialize_rig(cameras: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(camera["id"]): {
            "name": str(camera["name"]),
            "K": np.asarray(camera["K"]).tolist(),
            "R": np.asarray(camera["R"]).tolist(),
            "t": np.asarray(camera["t"]).reshape(3).tolist(),
            "position": np.asarray(camera["position"]).reshape(3).tolist(),
            "target": np.asarray(camera["target"]).reshape(3).tolist(),
            "fov_degree": float(camera["fov_degree"]),
            "azimuth_degree": float(camera["azimuth_degree"]),
        }
        for camera in cameras
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    previews_dir = output_dir / "previews"
    views_dir = output_dir / "views"
    samples_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    if args.save_view_images:
        views_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if str(args.yolo_device).lower() == "auto":
        yolo_device: str | int = 0 if device.type == "cuda" else "cpu"
    else:
        yolo_device = int(args.yolo_device) if str(args.yolo_device).isdigit() else str(args.yolo_device)

    source_paths = [Path(path) for path in sorted(glob.glob(args.source_glob, recursive=True))]
    if args.sample_sequences_randomly:
        source_rng = np.random.default_rng(args.seed)
        source_rng.shuffle(source_paths)
    if args.max_sequences > 0:
        source_paths = source_paths[: args.max_sequences]
    if not source_paths:
        raise FileNotFoundError(f"No AMASS files matched: {args.source_glob}")

    rng = np.random.default_rng(args.seed)
    model = None
    hrnet_detector = None
    if args.pose_backend == "hrnet":
        hrnet_detector = YOLOHRNetTopDownDetector(
            detector_model_path=args.yolo_model,
            pose_config_path=args.hrnet_config,
            pose_checkpoint_path=args.hrnet_checkpoint,
            conf_thresh=0.01,
            iou_thresh=0.5,
            max_people=1,
            device=str(yolo_device),
            return_heatmaps=False,
            bbox_scale=args.hrnet_bbox_scale,
        )
    else:
        model = YOLO(args.yolo_model)
    renderer = OffscreenRenderer(args.width, args.height)
    scene = Scene(ambient_light=[0.35, 0.35, 0.35], bg_color=[30, 30, 30, 255])
    entries: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    rejected_sequences = 0
    skipped_low_quality_frames = 0

    print(json.dumps({"torch_device": str(device), "yolo_device": yolo_device}, ensure_ascii=False))
    try:
        for sequence_index, npz_path in enumerate(source_paths):
            raw = np.load(npz_path, allow_pickle=True)
            frame_indices = selected_frame_indices(len(raw["trans"]), args.frame_stride, args.max_frames_per_sequence)
            if len(frame_indices) == 0:
                continue
            vertices, joints, faces, gender = load_amass_body(npz_path, frame_indices, Path(args.support_dir), device)
            target_coco = smplh_to_coco17(joints)
            vertices, joints, target_coco, scene_offset = place_sequence_in_field(
                vertices, joints, target_coco, args.field_position_jitter_m, rng
            )
            preset_name = choose_preset(args.rig_preset, rng)
            required_views = min(len(CAMERA_DEFS), max(1, int(args.min_qualifying_views)))
            reference_facing_yaw_deg = reference_body_facing_yaw_deg(target_coco[0])
            rig_attempts = max(1, int(args.rig_resample_attempts))
            cameras: list[dict[str, Any]] = []
            rig_metadata: dict[str, Any] = {}
            initial_pixel_heights: list[float] = []
            initial_visible_joints: list[int] = []
            initial_qualifying_views = 0
            for rig_attempt in range(rig_attempts):
                cameras, rig_metadata = build_karate_rig(
                    args.width,
                    args.height,
                    preset_name,
                    args.azimuth_jitter_deg,
                    args.target_height_m,
                    rng,
                    layout=args.rig_layout,
                    radius_range=(
                        (float(args.rig_radius_range[0]), float(args.rig_radius_range[1]))
                        if args.rig_radius_range is not None else None
                    ),
                    fov_range=(
                        (float(args.rig_fov_range[0]), float(args.rig_fov_range[1]))
                        if args.rig_fov_range is not None else None
                    ),
                    reference_facing_yaw_deg=reference_facing_yaw_deg,
                )
                _, initial_pixel_heights, initial_visible_joints, initial_qualifying_views = view_quality(
                    target_coco[0],
                    cameras,
                    args.width,
                    args.height,
                    args.min_person_pixel_height,
                    args.min_visible_body_joints,
                )
                if initial_qualifying_views >= required_views:
                    break
            else:
                rejected_sequences += 1
                print(json.dumps({
                    "sequence_rejected": str(npz_path),
                    "reason": "initial_frame_below_pixel_gate",
                    "qualifying_views": initial_qualifying_views,
                    "required_views": required_views,
                }, ensure_ascii=False))
                continue
            rig_metadata["quality_gate"] = {
                "attempts_used": rig_attempt + 1,
                "initial_person_pixel_height_per_view": initial_pixel_heights,
                "initial_visible_body_joints_per_view": initial_visible_joints,
                "initial_qualifying_views": initial_qualifying_views,
            }
            sequence_name = f"amass_{npz_path.parent.name}_{npz_path.stem}"
            sequence_rig = serialize_rig(cameras)
            sequences.append({
                "sequence": sequence_name,
                "source_npz": str(npz_path),
                "gender": gender,
                "frame_indices": frame_indices.astype(int).tolist(),
                "scene_offset": scene_offset.tolist(),
                "rig_metadata": rig_metadata,
                "cameras": sequence_rig,
            })

            for local_index, source_frame in enumerate(frame_indices):
                yolo_views: list[np.ndarray] = []
                gt_views: list[np.ndarray] = []
                ray_directions = np.zeros((4, COCO_NUM_JOINTS, 3), dtype=np.float32)
                preview_images: list[np.ndarray] = []
                occlusion_view_mask = np.zeros((4,), dtype=np.bool_)
                occlusion_boxes = np.full((4, 4), -1, dtype=np.int32)
                occlusion_body_area_ratio = np.zeros((4,), dtype=np.float32)
                occlusion_joint_mask = np.zeros((4, COCO_NUM_JOINTS), dtype=np.bool_)
                occluder_center = np.full((3,), np.nan, dtype=np.float32)
                occluder_extents = np.zeros((3,), dtype=np.float32)
                projected_views, pixel_heights, visible_body_joints, qualifying_views = view_quality(
                    target_coco[local_index],
                    cameras,
                    args.width,
                    args.height,
                    args.min_person_pixel_height,
                    args.min_visible_body_joints,
                )
                if qualifying_views < required_views:
                    skipped_low_quality_frames += 1
                    print(json.dumps({
                        "sample_skipped": f"{sequence_name}_frame_{int(source_frame):06d}",
                        "reason": "below_pixel_gate",
                        "person_pixel_height_per_view": pixel_heights,
                        "visible_body_joints_per_view": visible_body_joints,
                        "qualifying_views": qualifying_views,
                    }, ensure_ascii=False))
                    continue

                physical_occluder: dict[str, np.ndarray] | None = None
                if (
                    args.occlusion_mode == "physical3d"
                    and rng.random() < float(np.clip(args.occlusion_probability, 0.0, 1.0))
                ):
                    physical_occluder, occlusion_joint_mask = sample_physical_occluder(
                        target_coco[local_index], cameras, rng,
                        args.physical_occluder_attempts, args.physical_occluder_min_views,
                        args.physical_occluder_max_views, args.physical_occluder_min_body_joints,
                        args.physical_occluder_max_body_joints,
                        required_view_index=(sequence_index + local_index) % len(cameras),
                    )
                    if physical_occluder is not None:
                        occluder_center = physical_occluder["center"]
                        occluder_extents = physical_occluder["extents"]
                        occlusion_view_mask = (occlusion_joint_mask[:, 5:17].sum(axis=1) >= 1)
                        occlusion_body_area_ratio = occlusion_joint_mask[:, 5:17].sum(axis=1).astype(np.float32) / 12.0

                rendered_images = [
                    render_karate_frame(
                        renderer, scene, camera, vertices[local_index], faces, args.field_size_m, physical_occluder
                    )
                    for camera in cameras
                ]
                if (
                    args.occlusion_mode == "legacy2d"
                    and rng.random() < float(np.clip(args.occlusion_probability, 0.0, 1.0))
                ):
                    occluded_view = (sequence_index + local_index) % len(cameras)
                    projected_gt = projected_views[occluded_view]
                    occluded_image, box, area_ratio = apply_local_occluder(
                        rendered_images[occluded_view],
                        projected_gt,
                        rng,
                        args.occlusion_min_width_ratio,
                        args.occlusion_max_width_ratio,
                        args.occlusion_min_height_ratio,
                        args.occlusion_max_height_ratio,
                    )
                    if box[0] >= 0:
                        rendered_images[occluded_view] = occluded_image
                        occlusion_view_mask[occluded_view] = True
                        occlusion_boxes[occluded_view] = box
                        occlusion_body_area_ratio[occluded_view] = area_ratio
                if args.pose_backend == "hrnet":
                    assert hrnet_detector is not None
                    detected_poses = hrnet_keypoints_batch(
                        hrnet_detector, rendered_images
                    )
                else:
                    assert model is not None
                    detected_poses = yolo_keypoints_batch(
                        model, rendered_images, yolo_device
                    )

                for view_index, (camera, image_rgb, yolo_pose) in enumerate(zip(cameras, rendered_images, detected_poses)):
                    gt_pose = projected_views[view_index]
                    for joint_index, (x, y, confidence) in enumerate(yolo_pose):
                        if confidence > 0.0:
                            ray_directions[view_index, joint_index] = backproject_direction(camera, float(x), float(y))
                    yolo_views.append(yolo_pose)
                    gt_views.append(gt_pose)
                    label = (
                        f"{camera['id']}  az={camera['azimuth_degree']:.1f}  "
                        f"r={np.linalg.norm(camera['position'][:2]):.1f}m  fov={camera['fov_degree']:.1f}"
                    )
                    if occlusion_view_mask[view_index]:
                        label += f"  OCC={occlusion_body_area_ratio[view_index]:.2f}"
                    annotated = draw_pose(image_rgb, yolo_pose, args.confidence_threshold, label)
                    preview_images.append(annotated)
                    if args.save_view_images:
                        imageio.imwrite(
                            views_dir / f"{sequence_name}_frame_{int(source_frame):06d}_{camera['id']}.png",
                            annotated,
                        )

                yolo_array = np.stack(yolo_views).astype(np.float32)
                gt_array = np.stack(gt_views).astype(np.float32)
                yolo_by_view = {camera["id"]: yolo_array[index] for index, camera in enumerate(cameras)}
                clean_tokens, view_mask, joint_view_mask, target_confidence = make_stage_a_tokens(
                    yolo_by_view,
                    cameras,
                    args.width,
                    args.height,
                    args.confidence_threshold,
                    use_face_targets=False,
                    active_view_ids={camera["id"] for camera in cameras},
                )
                origins = np.stack([camera_origin(camera) for camera in cameras]).astype(np.float32)
                origin_center = origins.mean(axis=0).astype(np.float32)
                origin_scale = np.float32(max(float(np.linalg.norm(origins - origin_center, axis=1).max()), 1.0))
                sample_id = f"{sequence_name}_frame_{int(source_frame):06d}"
                sample_path = samples_dir / f"{sample_id}.npz"
                np.savez_compressed(
                    sample_path,
                    clean_ray_tokens=clean_tokens.astype(np.float32),
                    view_mask=view_mask.astype(np.bool_),
                    joint_view_mask=joint_view_mask.astype(np.bool_),
                    yolo_keypoints_2d=yolo_array,
                    detector_keypoints_2d=yolo_array,
                    gt_keypoints_2d=gt_array,
                    ray_directions=ray_directions,
                    target_3d=target_coco[local_index].astype(np.float32),
                    target_3d_root_relative=root_relative(target_coco[local_index]),
                    target_confidence=target_confidence.astype(np.float32),
                    camera_intrinsics=np.stack([camera["K"] for camera in cameras]).astype(np.float32),
                    camera_rotations=np.stack([camera["R"] for camera in cameras]).astype(np.float32),
                    camera_translations=np.stack([camera["t"] for camera in cameras]).astype(np.float32),
                    camera_origins=origins,
                    ray_origin_center=origin_center,
                    ray_origin_scale=np.asarray(origin_scale, dtype=np.float32),
                    image_size=np.asarray([args.width, args.height], dtype=np.int32),
                    field_size_m=np.asarray(args.field_size_m, dtype=np.float32),
                    source_frame=np.asarray(int(source_frame), dtype=np.int32),
                    views=np.asarray([camera["id"] for camera in cameras]),
                    occlusion_view_mask=occlusion_view_mask,
                    occlusion_boxes_xyxy=occlusion_boxes,
                    occlusion_body_area_ratio=occlusion_body_area_ratio,
                    occlusion_joint_mask=occlusion_joint_mask,
                    physical_occluder_center_xyz=occluder_center,
                    physical_occluder_extents_xyz=occluder_extents,
                    person_pixel_height_per_view=np.asarray(pixel_heights, dtype=np.float32),
                    visible_body_joints_per_view=np.asarray(visible_body_joints, dtype=np.int16),
                )
                save_this_preview = args.preview_limit < 0 or len(entries) < args.preview_limit
                preview_path = previews_dir / f"{sample_id}_fourview.png"
                if save_this_preview:
                    save_preview(preview_images, preview_path)
                entry = {
                    "id": sample_id,
                    "path": str(sample_path),
                    "sequence": sequence_name,
                    "frame_id": int(source_frame),
                    "person_id": "amass_person01",
                    "source_npz": str(npz_path),
                    "active_joint_views": int(joint_view_mask.sum()),
                    "rig_preset": preset_name,
                    "person_pixel_height_per_view": [round(value, 2) for value in pixel_heights],
                    "visible_body_joints_per_view": visible_body_joints,
                    "qualifying_views": qualifying_views,
                    "occluded_views": [
                        str(cameras[index]["id"])
                        for index in np.flatnonzero(occlusion_view_mask)
                    ],
                    "occlusion_mode": args.occlusion_mode,
                    "occluded_body_joints_per_view": (
                        occlusion_joint_mask[:, 5:17].sum(axis=1).astype(int).tolist()
                    ),
                    "physical_occluder_center_xyz": occluder_center.round(4).tolist(),
                    "physical_occluder_extents_xyz": occluder_extents.round(4).tolist(),
                    "occlusion_body_area_ratio": occlusion_body_area_ratio.tolist(),
                }
                if save_this_preview:
                    entry["preview"] = str(preview_path)
                entries.append(entry)
                print(json.dumps({"sample": sample_id, "active_joint_views": int(joint_view_mask.sum())}))
    finally:
        renderer.delete()

    manifest = {
        "metadata": {
            "stage": "amass_clean_observation_cache",
            "source_glob": args.source_glob,
            "num_entries": len(entries),
            "num_sequences": len(sequences),
            "resolution": [args.width, args.height],
            "field_size_m": args.field_size_m,
            "rig_preset": args.rig_preset,
            "rig_fixed_per_sequence": True,
            "extrinsic_noise": False,
            "occlusion": {
                "mode": args.occlusion_mode,
                "probability": float(args.occlusion_probability),
                "physical_target_view_rotation": args.occlusion_mode == "physical3d",
                "min_occluded_views": int(args.physical_occluder_min_views),
                "max_occluded_views": int(args.physical_occluder_max_views),
                "min_body_joints_per_view": int(args.physical_occluder_min_body_joints),
                "max_body_joints_per_view": int(args.physical_occluder_max_body_joints),
                "legacy_width_ratio": [float(args.occlusion_min_width_ratio), float(args.occlusion_max_width_ratio)],
                "legacy_height_ratio": [float(args.occlusion_min_height_ratio), float(args.occlusion_max_height_ratio)],
            },
            "seed": args.seed,
            "sample_sequences_randomly": bool(args.sample_sequences_randomly),
            "preview_limit": int(args.preview_limit),
            "torch_device": str(device),
            "yolo_device": yolo_device,
            "quality_gate": {
                "min_person_pixel_height": float(args.min_person_pixel_height),
                "min_visible_body_joints": int(args.min_visible_body_joints),
                "min_qualifying_views": int(args.min_qualifying_views),
                "rig_resample_attempts": int(args.rig_resample_attempts),
                "rejected_sequences": int(rejected_sequences),
                "skipped_low_quality_frames": int(skipped_low_quality_frames),
            },
            "pose_backend": args.pose_backend,
            "hrnet_bbox_scale": float(args.hrnet_bbox_scale),
            "sequences": sequences,
        },
        "entries": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "num_entries": len(entries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
