"""Rotate each floor-calibrated camera into one shared physical XY frame.

The image observations and local planar PnP solution stay unchanged.  The
three user-selected anchors identify which old local-grid directions correspond
to physical global +X and +Y, allowing an exact coordinate-basis transform:

    X_old = basis_old_from_global @ X_global
    R_global_to_camera = R_old_to_camera @ basis_old_from_global
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INTRINSICS = {
    "cam1": "calib_out_cam1_hvflip/cam1_intrinsics.npz",
    "cam2": "calib_out_cam2_hvflip_recalc/cam2_intrinsics.npz",
    "cam3": "calib_out_cam3_hvflip_fast/cam3_intrinsics.npz",
    "cam4": "calib_out_cam4_hvflip_newintr_20260727/cam4_intrinsics.npz",
}


def npz_dict(path: Path) -> dict[str, Any]:
    data = np.load(str(path), allow_pickle=True)
    return {key: data[key] for key in data.files}


def camera_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    matrix = next(np.asarray(data[key], dtype=np.float64) for key in ("camera_matrix", "mtx", "K", "intrinsic_matrix") if key in data)
    distortion = next(
        (np.asarray(data[key], dtype=np.float64) for key in ("dist_coeffs", "dist", "distortion_coefficients", "d") if key in data),
        np.zeros((5, 1), dtype=np.float64),
    )
    return matrix, distortion


def selected_old_xy(camera_assignment: dict[str, Any], role: str) -> np.ndarray:
    item = camera_assignment[role]
    if item is None:
        raise ValueError(f"Missing {role} assignment")
    return np.asarray(item["previous_world_xyz_m"][:2], dtype=np.float64)


def basis_from_assignment(camera_assignment: dict[str, Any], spacing: float) -> np.ndarray:
    center = selected_old_xy(camera_assignment, "center")
    old_x = selected_old_xy(camera_assignment, "plus_x")
    old_y = selected_old_xy(camera_assignment, "plus_y")
    basis2 = np.column_stack(((old_x - center) / spacing, (old_y - center) / spacing))
    if not np.allclose(basis2.T @ basis2, np.eye(2), atol=1e-6):
        raise ValueError(f"Selected +X/+Y are not orthonormal in old grid: {basis2}")
    basis3 = np.eye(3, dtype=np.float64)
    basis3[:2, :2] = basis2
    if np.linalg.det(basis3) < 0:
        raise ValueError("Selected axes form a reflection; +X/+Y must be a right-handed floor frame")
    return basis3


def frame_from_source(points_json: Path, line_dir: Path) -> np.ndarray:
    payload = json.loads(points_json.read_text(encoding="utf-8"))
    meta = payload["metadata"]
    source = Path(meta["video_path"])
    if not source.is_absolute():
        source = line_dir / source.name
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(meta["frame_index"]))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame from {source}")
    return frame


def draw_axis_overlay(
    frame: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    output: Path,
) -> None:
    world = np.asarray(
        [[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(rotation)
    uv, _ = cv2.projectPoints(world, rvec, translation, matrix, distortion)
    pts = np.rint(uv.reshape(-1, 2)).astype(int)
    origin = tuple(pts[0])
    cv2.arrowedLine(frame, origin, tuple(pts[1]), (0, 0, 255), 8, cv2.LINE_AA, tipLength=0.12)
    cv2.arrowedLine(frame, origin, tuple(pts[2]), (0, 255, 0), 8, cv2.LINE_AA, tipLength=0.12)
    cv2.arrowedLine(frame, origin, tuple(pts[3]), (255, 0, 0), 8, cv2.LINE_AA, tipLength=0.12)
    cv2.putText(frame, "+X", tuple(pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "+Y", tuple(pts[2]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "+Z", tuple(pts[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imwrite(str(output), frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-dir", default="camtest/line")
    parser.add_argument("--assignments", default="camtest/line/global_axis_assignments.json")
    parser.add_argument("--source-extrinsics-dir", default="camtest/line/line_refine_20260728")
    parser.add_argument("--output-dir", default="camtest/line/line_refine_global_axes_20260729")
    args = parser.parse_args()
    line_dir = Path(args.line_dir)
    source_dir = Path(args.source_extrinsics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = json.loads(Path(args.assignments).read_text(encoding="utf-8"))
    spacing = float(assignments["spacing_m"])
    records = {}

    for cam in ("cam1", "cam2", "cam3", "cam4"):
        source_path = source_dir / f"{cam}_line_extrinsics_refined.npz"
        source = npz_dict(source_path)
        old_rotation = np.asarray(source["R"], dtype=np.float64)
        translation = np.asarray(source["tvec"], dtype=np.float64).reshape(3, 1)
        basis = basis_from_assignment(assignments["cameras"][cam], spacing)
        new_rotation = old_rotation @ basis
        new_rvec, _ = cv2.Rodrigues(new_rotation)
        old_center = (-old_rotation.T @ translation).reshape(3)
        new_center = (-new_rotation.T @ translation).reshape(3)
        old_forward = old_rotation.T @ np.asarray([0.0, 0.0, 1.0])
        new_forward = new_rotation.T @ np.asarray([0.0, 0.0, 1.0])

        updated = dict(source)
        for key in ("R", "rotation_matrix"):
            updated[key] = new_rotation
        for key in ("rvec", "rotation_vector"):
            updated[key] = new_rvec.reshape(3, 1)
        updated["global_axis_basis_old_from_new"] = basis
        updated["global_axis_assignment_file"] = str(Path(args.assignments).resolve())
        output_npz = output_dir / f"{cam}_line_extrinsics_refined.npz"
        np.savez(str(output_npz), **updated)

        records[cam] = {
            "basis_old_from_global": basis.tolist(),
            "determinant": float(np.linalg.det(basis)),
            "camera_center_old_m": old_center.tolist(),
            "camera_center_global_m": new_center.tolist(),
            "forward_old_world": old_forward.tolist(),
            "forward_global_world": new_forward.tolist(),
            "output_npz": str(output_npz.resolve()),
        }

        points_file = Path(assignments["cameras"][cam]["center"].get("points_file", ""))
        if not points_file.is_file():
            candidates = {
                "cam1": line_dir / "point_picks_cam4_hvflip_retry/cam1out_line_points.json",
                "cam2": line_dir / "point_picks_cam4_hvflip_retry/cam2out_line_points.json",
                "cam3": line_dir / "point_picks_cam4_hvflip_retry/cam3out_line_points.json",
                "cam4": line_dir / "point_picks_cam4_hvflip_retry/cam4_line_points.json",
            }
            points_file = candidates[cam]
        frame = frame_from_source(points_file, line_dir)
        matrix, distortion = camera_matrix(line_dir / INTRINSICS[cam])
        draw_axis_overlay(
            frame, matrix, distortion, new_rotation, translation,
            output_dir / f"{cam}_global_axis_overlay.jpg",
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, key, title in (
        (axes[0], "camera_center_old_m", "Before: image-relative XY"),
        (axes[1], "camera_center_global_m", "After: one physical global XY"),
    ):
        ax.add_patch(plt.Rectangle((-2.5, -2.5), 5.0, 5.0, fill=False, color="#888888", linewidth=1.5, linestyle="--"))
        for cam, rec in records.items():
            center = np.asarray(rec[key])
            forward_key = "forward_old_world" if "old" in key else "forward_global_world"
            forward = np.asarray(rec[forward_key])
            ax.scatter(center[0], center[1], s=70)
            ax.arrow(center[0], center[1], forward[0], forward[1], width=0.025, head_width=0.16, length_includes_head=True)
            ax.text(center[0] + 0.08, center[1] + 0.08, cam)
        ax.scatter([0], [0], marker="+", s=130, color="black")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.grid(alpha=0.25)
        ax.set_xlabel("global X (m)")
        ax.set_ylabel("global Y (m)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "camera_layout_before_after.png", dpi=170)
    plt.close(fig)

    pair_distances = {}
    for i, cam_a in enumerate(records):
        for cam_b in list(records)[i + 1:]:
            a = np.asarray(records[cam_a]["camera_center_global_m"])
            b = np.asarray(records[cam_b]["camera_center_global_m"])
            pair_distances[f"{cam_a}+{cam_b}"] = float(np.linalg.norm(a - b))
    report = {
        "stage": "apply_global_floor_axis_mapping",
        "assignment_file": str(Path(args.assignments).resolve()),
        "source_extrinsics_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "cameras": records,
        "pair_distances_global_m": pair_distances,
        "note": "This is an exact world-basis change; per-camera image reprojection is unchanged.",
    }
    (output_dir / "global_axis_transform_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
