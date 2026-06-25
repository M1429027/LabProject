"""Build precomputed Stage A samples from Harmony4D karate annotations."""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COCO_NUM_JOINTS = 17
DEFAULT_IMAGE_WIDTH = 3840.0
DEFAULT_IMAGE_HEIGHT = 2160.0
RAY_FEATURE_DIM = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage A supervised warm-up dataset.")
    parser.add_argument("--selected-views-manifest", required=True)
    parser.add_argument("--zip-map-json", default=None, help="Optional sequence-prefix to zip path map.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-width", type=float, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=float, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed. Slower but smaller; default keeps dataset build fast.",
    )
    parser.add_argument("--val-sequences", nargs="*", default=["09_karate/004_karate"])
    parser.add_argument("--test-sequences", nargs="*", default=["11_karate3/008_karate3"])
    return parser.parse_args()


def default_zip_for_sequence(sequence: str) -> str:
    dataset = sequence.split("/", 1)[0]
    mapping = {
        "09_karate": "/mnt/d/09_karate.zip",
        "10_karate2": "/mnt/d/10_karate2.zip",
        "11_karate3": "/mnt/d/11_karate3.zip",
    }
    if dataset not in mapping:
        raise KeyError(f"No default zip mapping for sequence {sequence!r}.")
    return mapping[dataset]


def load_zip_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame_id_from_name(path: str) -> int:
    return int(Path(path).stem)


def sorted_frame_ids(archive: zipfile.ZipFile, sequence: str) -> list[int]:
    prefix = f"{sequence}/processed_data/poses3d/"
    ids = []
    for name in archive.namelist():
        if name.startswith(prefix) and name.endswith(".npy"):
            ids.append(frame_id_from_name(name))
    return sorted(ids)


def load_npy_dict(archive: zipfile.ZipFile, path: str) -> dict[str, np.ndarray]:
    return np.load(io.BytesIO(archive.read(path)), allow_pickle=True).item()


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP qvec [qw, qx, qy, qz] to a world-to-camera rotation matrix."""

    qw, qx, qy, qz = [float(value) for value in qvec]
    return np.array(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qz * qw, 2.0 * qx * qz + 2.0 * qy * qw],
            [2.0 * qx * qy + 2.0 * qz * qw, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qx * qw],
            [2.0 * qx * qz - 2.0 * qy * qw, 2.0 * qy * qz + 2.0 * qx * qw, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


def parse_colmap_cameras(archive: zipfile.ZipFile, sequence: str) -> dict[int, dict[str, Any]]:
    """Parse COLMAP camera intrinsics from a Harmony4D sequence."""

    path = f"{sequence}/colmap/workplace/cameras.txt"
    cameras = {}
    for raw_line in archive.read(path).decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = np.asarray([float(value) for value in parts[4:]], dtype=np.float64)
        fx, fy, cx, cy = params[:4]
        cameras[camera_id] = {
            "camera_id": camera_id,
            "model": model,
            "image_size": (width, height),
            "camera_matrix": np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "dist_coeffs": params[4:8].astype(np.float64),
        }
    return cameras


def parse_colmap_image_poses(archive: zipfile.ZipFile, sequence: str) -> dict[str, dict[str, Any]]:
    """Parse one fixed exo-camera pose per view from COLMAP images.txt."""

    path = f"{sequence}/colmap/workplace/images.txt"
    poses: dict[str, dict[str, Any]] = {}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for raw_line in archive.read(path).decode("utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 10 or not parts[-1].endswith(".jpg"):
            continue
        image_name = parts[-1]
        view_id = image_name.split("/", 1)[0]
        if not view_id.startswith("cam"):
            continue
        qvec = np.asarray([float(value) for value in parts[1:5]], dtype=np.float64)
        tvec = np.asarray([float(value) for value in parts[5:8]], dtype=np.float64)
        frame_number = int(Path(image_name).stem)
        candidates.setdefault(view_id, []).append(
            {
                "view_id": view_id,
                "image_name": image_name,
                "frame_number": frame_number,
                "camera_id": int(parts[8]),
                "rotation": qvec_to_rotmat(qvec),
                "translation": tvec,
            }
        )
    for view_id, rows in candidates.items():
        rows = sorted(rows, key=lambda row: (abs(int(row["frame_number"]) - 1), int(row["frame_number"])))
        poses[view_id] = rows[0]
    return poses


def load_processed_from_colmap_transform(archive: zipfile.ZipFile, sequence: str) -> np.ndarray:
    """Load the sequence transform used to express COLMAP rays in processed-data space."""

    path = f"{sequence}/colmap/workplace/scale.npy"
    if path not in set(archive.namelist()):
        return np.eye(4, dtype=np.float64)
    return np.asarray(np.load(io.BytesIO(archive.read(path)), allow_pickle=True), dtype=np.float64)


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([np.asarray(point, dtype=np.float64).reshape(3), np.ones(1, dtype=np.float64)])
    return (np.asarray(transform, dtype=np.float64).reshape(4, 4) @ homogeneous)[:3]


def build_ray_geometry(
    archive: zipfile.ZipFile,
    sequence: str,
    views: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Build per-view camera centers and ray transforms in processed-data space."""

    cameras = parse_colmap_cameras(archive, sequence)
    image_poses = parse_colmap_image_poses(archive, sequence)
    processed_from_colmap = load_processed_from_colmap_transform(archive, sequence)
    geometry = {}
    origins = []
    for view in views:
        if view not in image_poses:
            continue
        pose = image_poses[view]
        camera = cameras[int(pose["camera_id"])]
        rotation = np.asarray(pose["rotation"], dtype=np.float64)
        translation = np.asarray(pose["translation"], dtype=np.float64)
        origin_colmap = -rotation.T @ translation
        origin_processed = transform_point(processed_from_colmap, origin_colmap)
        geometry[view] = {
            **camera,
            "rotation": rotation,
            "translation": translation,
            "origin_colmap": origin_colmap,
            "origin_processed": origin_processed,
            "processed_from_colmap": processed_from_colmap,
        }
        origins.append(origin_processed)

    if origins:
        origin_array = np.vstack(origins)
        center = origin_array.mean(axis=0)
        radius = float(np.linalg.norm(origin_array - center.reshape(1, 3), axis=1).max())
        radius = max(radius, 1.0)
        for view_geometry in geometry.values():
            view_geometry["origin_center"] = center
            view_geometry["origin_scale"] = radius
    return geometry


def pixel_to_processed_ray(x: float, y: float, view_geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert one image point into a camera ray in processed-data coordinates."""

    point = np.array([[[float(x), float(y)]]], dtype=np.float64)
    camera_matrix = np.asarray(view_geometry["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(view_geometry["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    if str(view_geometry.get("model", "")).upper() == "OPENCV_FISHEYE":
        normalized = cv2.fisheye.undistortPoints(point, camera_matrix, dist_coeffs).reshape(2)
    else:
        normalized = cv2.undistortPoints(point, camera_matrix, dist_coeffs.reshape(-1)).reshape(2)

    direction_camera = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    direction_camera /= max(float(np.linalg.norm(direction_camera)), 1e-9)
    rotation = np.asarray(view_geometry["rotation"], dtype=np.float64).reshape(3, 3)
    direction_colmap = rotation.T @ direction_camera
    direction_colmap /= max(float(np.linalg.norm(direction_colmap)), 1e-9)

    origin_colmap = np.asarray(view_geometry["origin_colmap"], dtype=np.float64).reshape(3)
    processed_from_colmap = np.asarray(view_geometry["processed_from_colmap"], dtype=np.float64).reshape(4, 4)
    origin_processed = np.asarray(view_geometry["origin_processed"], dtype=np.float64).reshape(3)
    point_on_ray_processed = transform_point(processed_from_colmap, origin_colmap + direction_colmap)
    direction_processed = point_on_ray_processed - origin_processed
    direction_processed /= max(float(np.linalg.norm(direction_processed)), 1e-9)
    return origin_processed.astype(np.float32), direction_processed.astype(np.float32)


def make_tokens(
    poses2d_by_view: dict[str, dict[str, np.ndarray]],
    person_id: str,
    views: list[str],
    image_width: float,
    image_height: float,
    ray_geometry_by_view: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create [J,V,F] tokens from 2D annotations.

    Feature layout:
    0-2 camera ray origin in normalized processed-data coordinates
    3-5 camera ray direction in processed-data coordinates
    6-7 normalized xy in [-1, 1]
    8 confidence / visibility
    9 normalized view index
    10 normalized joint index
    """

    tokens = np.zeros((COCO_NUM_JOINTS, len(views), RAY_FEATURE_DIM), dtype=np.float32)
    view_mask = np.zeros((len(views),), dtype=np.bool_)
    joint_view_mask = np.zeros((COCO_NUM_JOINTS, len(views)), dtype=np.bool_)
    for view_index, view_id in enumerate(views):
        view_payload = poses2d_by_view.get(view_id)
        if not view_payload or person_id not in view_payload:
            continue
        points = np.asarray(view_payload[person_id], dtype=np.float32)
        if points.shape[0] < COCO_NUM_JOINTS:
            continue
        view_mask[view_index] = True
        for joint_id in range(COCO_NUM_JOINTS):
            x, y = points[joint_id, :2]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            x_norm = (float(x) / float(image_width)) * 2.0 - 1.0
            y_norm = (float(y) / float(image_height)) * 2.0 - 1.0
            origin = np.zeros(3, dtype=np.float32)
            direction = np.array([x_norm, y_norm, 1.0], dtype=np.float32)
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            if ray_geometry_by_view and view_id in ray_geometry_by_view:
                origin_raw, direction = pixel_to_processed_ray(x, y, ray_geometry_by_view[view_id])
                center = np.asarray(ray_geometry_by_view[view_id].get("origin_center", np.zeros(3)), dtype=np.float32)
                scale = float(ray_geometry_by_view[view_id].get("origin_scale", 1.0))
                origin = (origin_raw - center) / max(scale, 1e-6)
            tokens[joint_id, view_index] = [
                float(origin[0]),
                float(origin[1]),
                float(origin[2]),
                float(direction[0]),
                float(direction[1]),
                float(direction[2]),
                x_norm,
                y_norm,
                1.0,
                view_index / max(len(views) - 1, 1),
                joint_id / max(COCO_NUM_JOINTS - 1, 1),
            ]
            joint_view_mask[joint_id, view_index] = True
    return tokens, view_mask, joint_view_mask


def root_relative_target(target: np.ndarray) -> np.ndarray:
    pelvis = 0.5 * (target[11] + target[12])
    return target - pelvis.reshape(1, 3)


def split_for_sequence(sequence: str, val_sequences: set[str], test_sequences: set[str]) -> str:
    if sequence in test_sequences:
        return "test"
    if sequence in val_sequences:
        return "val"
    return "train"


def build_dataset(
    selected_manifest: dict[str, Any],
    zip_map: dict[str, str],
    output_dir: Path,
    image_width: float,
    image_height: float,
    val_sequences: set[str],
    test_sequences: set[str],
    max_samples: int | None,
    compressed: bool,
) -> dict[str, Any]:
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    zip_handles: dict[str, zipfile.ZipFile] = {}
    zip_name_sets: dict[str, set[str]] = {}

    try:
        for sequence_row in selected_manifest["sequences"]:
            sequence = str(sequence_row["sequence"])
            views = [str(view) for view in sequence_row["selected_views"]]
            zip_path = zip_map.get(sequence) or default_zip_for_sequence(sequence)
            if zip_path not in zip_handles:
                zip_handles[zip_path] = zipfile.ZipFile(zip_path)
                zip_name_sets[zip_path] = set(zip_handles[zip_path].namelist())
            archive = zip_handles[zip_path]
            archive_names = zip_name_sets[zip_path]
            frame_ids = sorted_frame_ids(archive, sequence)
            split = split_for_sequence(sequence, val_sequences, test_sequences)
            ray_geometry_by_view = build_ray_geometry(archive, sequence, views)

            for frame_id in frame_ids:
                pose3d_path = f"{sequence}/processed_data/poses3d/{frame_id:05d}.npy"
                if pose3d_path not in archive_names:
                    continue
                poses3d = load_npy_dict(archive, pose3d_path)
                poses2d_by_view = {}
                missing_view = False
                for view in views:
                    pose2d_path = f"{sequence}/processed_data/poses2d/{view}/{frame_id:05d}.npy"
                    if pose2d_path not in archive_names:
                        missing_view = True
                        break
                    poses2d_by_view[view] = load_npy_dict(archive, pose2d_path)
                if missing_view:
                    continue

                for person_id, joints3d in poses3d.items():
                    target = np.asarray(joints3d, dtype=np.float32)
                    if target.shape[0] < COCO_NUM_JOINTS:
                        continue
                    target_xyz = target[:COCO_NUM_JOINTS, :3]
                    target_conf = target[:COCO_NUM_JOINTS, 3] if target.shape[1] > 3 else np.ones(COCO_NUM_JOINTS)
                    tokens, view_mask, joint_view_mask = make_tokens(
                        poses2d_by_view=poses2d_by_view,
                        person_id=str(person_id),
                        views=views,
                        image_width=image_width,
                        image_height=image_height,
                        ray_geometry_by_view=ray_geometry_by_view,
                    )
                    if int(joint_view_mask.sum()) == 0:
                        continue
                    sample_id = f"{sequence.replace('/', '__')}__{frame_id:05d}__{person_id}"
                    sample_path = samples_dir / f"{sample_id}.npz"
                    sample_payload = {
                        "ray_tokens": tokens.astype(np.float32),
                        "view_mask": view_mask.astype(np.bool_),
                        "joint_view_mask": joint_view_mask.astype(np.bool_),
                        "target_3d": target_xyz.astype(np.float32),
                        "target_3d_root_relative": root_relative_target(target_xyz).astype(np.float32),
                        "target_confidence": target_conf.astype(np.float32),
                        "sequence": np.array(sequence),
                        "frame_id": np.array(frame_id, dtype=np.int32),
                        "person_id": np.array(str(person_id)),
                        "views": np.asarray(views),
                    }
                    if compressed:
                        np.savez_compressed(sample_path, **sample_payload)
                    else:
                        np.savez(sample_path, **sample_payload)
                    entries.append(
                        {
                            "id": sample_id,
                            "path": str(sample_path),
                            "sequence": sequence,
                            "frame_id": int(frame_id),
                            "person_id": str(person_id),
                            "split": split,
                        }
                    )
                    if max_samples is not None and len(entries) >= int(max_samples):
                        raise StopIteration
    except StopIteration:
        pass
    finally:
        for handle in zip_handles.values():
            handle.close()

    split_counts: dict[str, int] = {}
    for entry in entries:
        split_counts[entry["split"]] = split_counts.get(entry["split"], 0) + 1
    return {
        "stage": "karate_selfcal_stage_a_dataset",
        "selected_views": selected_manifest["selected_views"],
        "feature_dim": RAY_FEATURE_DIM,
        "feature_layout": [
            "ray_origin_x",
            "ray_origin_y",
            "ray_origin_z",
            "ray_dir_x",
            "ray_dir_y",
            "ray_dir_z",
            "x_norm",
            "y_norm",
            "confidence",
            "view_id_norm",
            "joint_id_norm",
        ],
        "num_joints": COCO_NUM_JOINTS,
        "image_size": [float(image_width), float(image_height)],
        "entries": entries,
        "summary": {
            "num_samples": len(entries),
            "split_counts": split_counts,
            "val_sequences": sorted(val_sequences),
            "test_sequences": sorted(test_sequences),
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_manifest = json.loads(Path(args.selected_views_manifest).read_text(encoding="utf-8"))
    manifest = build_dataset(
        selected_manifest=selected_manifest,
        zip_map=load_zip_map(args.zip_map_json),
        output_dir=output_dir,
        image_width=float(args.image_width),
        image_height=float(args.image_height),
        val_sequences=set(args.val_sequences),
        test_sequences=set(args.test_sequences),
        max_samples=args.max_samples,
        compressed=bool(args.compressed),
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
