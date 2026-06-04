"""Compare self-calibrated rough extrinsics against reference camera geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Compare estimated camera centers with reference camera centers."
    )
    parser.add_argument("--estimated-extrinsics-json", required=True, help="Selfcal rough_extrinsics.json")
    parser.add_argument("--reference-extrinsics-json", required=True, help="Reference rough_extrinsics_colmap.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for diagnostics")
    parser.add_argument("--anchor-view", default=None, help="Optional anchor view id")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON payload."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def camera_center_from_extrinsics(view_meta: dict[str, Any]) -> np.ndarray:
    """Return world-space camera center from a world-to-camera transform."""

    if "camera_center_world" in view_meta:
        return np.asarray(view_meta["camera_center_world"], dtype=np.float64).reshape(3)
    rotation = np.asarray(view_meta["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(view_meta["translation"], dtype=np.float64).reshape(3)
    return -rotation.T @ translation


def relative_centers(payload: dict[str, Any], anchor_view: str) -> dict[str, np.ndarray]:
    """Return camera centers relative to an anchor camera coordinate frame."""

    extrinsics = payload.get("extrinsics_by_view", {})
    centers = {
        view_id: camera_center_from_extrinsics(view_meta)
        for view_id, view_meta in extrinsics.items()
    }
    if anchor_view not in centers:
        raise ValueError(f"Anchor view not found in extrinsics: {anchor_view}")
    anchor_rotation = np.asarray(extrinsics[anchor_view]["rotation"], dtype=np.float64).reshape(3, 3)
    anchor = centers[anchor_view]
    return {
        view_id: anchor_rotation @ (center - anchor)
        for view_id, center in centers.items()
    }


def angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float | None:
    """Return angle in degrees between two vectors."""

    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return None
    cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def build_comparison(
    estimated: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    anchor_view: str,
) -> dict[str, Any]:
    """Build numeric comparison between estimated and reference centers."""

    common_views = sorted(set(estimated) & set(reference))
    rows = []
    center_errors = []
    angle_errors = []
    scale_ratios = []
    for view_id in common_views:
        est = estimated[view_id]
        ref = reference[view_id]
        center_error = float(np.linalg.norm(est - ref))
        ref_distance = float(np.linalg.norm(ref))
        est_distance = float(np.linalg.norm(est))
        scale_ratio = est_distance / ref_distance if ref_distance > 1e-9 else None
        angle_error = angle_between(est, ref)
        if view_id != anchor_view:
            center_errors.append(center_error)
            if angle_error is not None:
                angle_errors.append(angle_error)
            if scale_ratio is not None:
                scale_ratios.append(scale_ratio)
        rows.append(
            {
                "view_id": view_id,
                "estimated_center_relative": est.tolist(),
                "reference_center_relative": ref.tolist(),
                "estimated_distance_from_anchor": est_distance,
                "reference_distance_from_anchor": ref_distance,
                "distance_scale_ratio_est_over_ref": scale_ratio,
                "center_error": center_error,
                "direction_angle_error_deg": angle_error,
            }
        )

    return {
        "anchor_view": anchor_view,
        "common_views": common_views,
        "per_view": rows,
        "summary": {
            "mean_center_error": float(np.mean(center_errors)) if center_errors else None,
            "median_center_error": float(np.median(center_errors)) if center_errors else None,
            "mean_direction_angle_error_deg": float(np.mean(angle_errors)) if angle_errors else None,
            "median_direction_angle_error_deg": float(np.median(angle_errors)) if angle_errors else None,
            "mean_distance_scale_ratio_est_over_ref": float(np.mean(scale_ratios)) if scale_ratios else None,
            "median_distance_scale_ratio_est_over_ref": float(np.median(scale_ratios)) if scale_ratios else None,
        },
    }


def equal_axis_3d(ax: Any, points: np.ndarray) -> None:
    """Set equal 3D axis limits around all points."""

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = (mins + maxs) * 0.5
    half_span = max(float(np.max(maxs - mins) * 0.6), 0.5)
    ax.set_xlim(centers[0] - half_span, centers[0] + half_span)
    ax.set_ylim(centers[1] - half_span, centers[1] + half_span)
    ax.set_zlim(centers[2] - half_span, centers[2] + half_span)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def plot_centers(
    estimated: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot reference and estimated camera centers in the same coordinate frame."""

    common_views = comparison["common_views"]
    ref_points = np.asarray([reference[view_id] for view_id in common_views], dtype=np.float64)
    est_points = np.asarray([estimated[view_id] for view_id in common_views], dtype=np.float64)
    all_points = np.vstack([ref_points, est_points])

    fig = plt.figure(figsize=(10, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], c="#2ECC71", s=70, label="COLMAP oracle")
    ax.scatter(est_points[:, 0], est_points[:, 1], est_points[:, 2], c="#E74C3C", s=70, label="selfcal estimate")

    for view_id, ref, est in zip(common_views, ref_points, est_points):
        ax.plot([ref[0], est[0]], [ref[1], est[1]], [ref[2], est[2]], c="#555555", alpha=0.55)
        ax.text(ref[0], ref[1], ref[2], f"{view_id}\nref", fontsize=7, color="#1E8449")
        ax.text(est[0], est[1], est[2], f"{view_id}\nest", fontsize=7, color="#922B21")

    ax.set_title("Camera Geometry Diagnostic: Selfcal vs COLMAP")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    equal_axis_3d(ax, all_points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    """Run camera geometry diagnostics."""

    args = parse_args()
    estimated_payload = load_json(args.estimated_extrinsics_json)
    reference_payload = load_json(args.reference_extrinsics_json)
    anchor_view = args.anchor_view or estimated_payload.get("anchor_view") or reference_payload.get("anchor_view")
    if not anchor_view:
        raise ValueError("No anchor view provided or found in extrinsics payloads.")

    estimated = relative_centers(estimated_payload, anchor_view=anchor_view)
    reference = relative_centers(reference_payload, anchor_view=anchor_view)
    comparison = build_comparison(estimated=estimated, reference=reference, anchor_view=anchor_view)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "camera_geometry_comparison.json"
    plot_path = output_dir / "camera_geometry_comparison.png"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)
    plot_centers(
        estimated=estimated,
        reference=reference,
        comparison=comparison,
        output_path=plot_path,
    )
    print(
        json.dumps(
            {
                "summary_json": str(summary_path),
                "plot_png": str(plot_path),
                "summary": comparison["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
