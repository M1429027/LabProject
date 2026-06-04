"""Export fixed-camera COLMAP intrinsics/extrinsics into rough_extrinsics.json format."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .relative_pose import infer_colmap_camera_id, load_colmap_intrinsics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Export COLMAP camera poses as oracle rough extrinsics for debugging."
    )
    parser.add_argument(
        "--colmap-cameras-txt",
        required=True,
        help="Path to COLMAP cameras.txt, or archive.zip::inner/cameras.txt",
    )
    parser.add_argument(
        "--colmap-images-txt",
        required=True,
        help="Path to COLMAP images.txt, or archive.zip::inner/images.txt",
    )
    parser.add_argument("--view-ids", nargs="+", required=True, help="Target view ids, e.g. karate004_cam01")
    parser.add_argument("--anchor-view", default=None, help="Optional anchor view")
    parser.add_argument("--output-json", required=True, help="Output rough_extrinsics.json path")
    return parser.parse_args()


class _ZipTextHandle:
    """Context manager for text inside a zip archive."""

    def __init__(self, archive: zipfile.ZipFile, inner_path: str) -> None:
        self._archive = archive
        self._inner_path = inner_path
        self._raw_handle = None
        self._text_handle = None

    def __enter__(self):
        import io

        self._raw_handle = self._archive.open(self._inner_path, "r")
        self._text_handle = io.TextIOWrapper(self._raw_handle, encoding="utf-8")
        return self._text_handle

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._text_handle is not None:
            self._text_handle.close()
        self._archive.close()


def open_text_source(path: str | Path):
    """Open a plain text path or `archive.zip::inner/path.txt`."""

    source = str(path)
    if "::" not in source:
        return Path(source).open("r", encoding="utf-8")
    archive_path, inner_path = source.split("::", 1)
    archive = zipfile.ZipFile(Path(archive_path).resolve())
    return _ZipTextHandle(archive, inner_path)


def qvec_to_rotation(qvec: list[float]) -> np.ndarray:
    """Convert COLMAP quaternion [qw, qx, qy, qz] to a rotation matrix."""

    q = np.asarray(qvec, dtype=np.float64).reshape(4)
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def load_colmap_image_poses(
    images_txt_path: str | Path,
    view_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Load one representative COLMAP pose per requested camera view."""

    requested = {
        infer_colmap_camera_id(view_id): view_id
        for view_id in view_ids
    }
    found: dict[str, dict[str, Any]] = {}
    with open_text_source(images_txt_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10 or "/" not in parts[9]:
                continue
            camera_id = int(parts[8])
            if camera_id not in requested:
                continue
            view_id = requested[camera_id]
            if view_id in found:
                continue
            rotation = qvec_to_rotation([float(v) for v in parts[1:5]])
            translation = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
            found[view_id] = {
                "camera_id": camera_id,
                "image_name": parts[9],
                "rotation": rotation,
                "translation": translation,
            }
            if len(found) == len(view_ids):
                break

    missing = [view_id for view_id in view_ids if view_id not in found]
    if missing:
        raise ValueError(f"Missing COLMAP poses for views: {missing}")
    return found


def camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Return world-space camera center from COLMAP world-to-camera transform."""

    return -rotation.T @ translation.reshape(3)


def export_colmap_extrinsics(
    colmap_cameras_txt: str | Path,
    colmap_images_txt: str | Path,
    view_ids: list[str],
    anchor_view: str | None = None,
) -> dict[str, Any]:
    """Export COLMAP poses into the rough_extrinsics schema used by triangulation."""

    anchor_view = anchor_view or view_ids[0]
    poses_by_view = load_colmap_image_poses(
        images_txt_path=colmap_images_txt,
        view_ids=view_ids,
    )

    extrinsics_by_view = {}
    for view_id in view_ids:
        intrinsics = load_colmap_intrinsics(
            cameras_txt_path=colmap_cameras_txt,
            camera_id=infer_colmap_camera_id(view_id),
        )
        pose = poses_by_view[view_id]
        rotation = np.asarray(pose["rotation"], dtype=np.float64)
        translation = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
        extrinsics_by_view[view_id] = {
            "view_id": view_id,
            "is_anchor": view_id == anchor_view,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "projection_matrix": np.hstack([rotation, translation.reshape(3, 1)]).tolist(),
            "intrinsics": {
                "source_path": intrinsics["source_path"],
                "model": intrinsics["model"],
                "camera_matrix": np.asarray(intrinsics["camera_matrix"], dtype=np.float64).tolist(),
                "dist_coeffs": np.asarray(intrinsics["dist_coeffs"], dtype=np.float64).reshape(-1).tolist(),
                "image_size": intrinsics["image_size"],
                "rms": intrinsics["rms"],
            },
            "colmap_image_name": pose["image_name"],
            "camera_center_world": camera_center(rotation, translation).tolist(),
        }

    return {
        "anchor_view": anchor_view,
        "translation_scale_note": "Oracle COLMAP extrinsics from dataset; world scale follows COLMAP reconstruction.",
        "num_resolved_views": len(extrinsics_by_view),
        "unresolved_views": [],
        "extrinsics_by_view": extrinsics_by_view,
    }


def main() -> None:
    """Export COLMAP poses into the rough_extrinsics schema used by triangulation."""

    args = parse_args()
    payload = export_colmap_extrinsics(
        colmap_cameras_txt=args.colmap_cameras_txt,
        colmap_images_txt=args.colmap_images_txt,
        view_ids=list(args.view_ids),
        anchor_view=args.anchor_view,
    )
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output_json": str(output_path), "num_views": len(extrinsics_by_view)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
