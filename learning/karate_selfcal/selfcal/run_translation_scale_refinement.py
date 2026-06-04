"""Refine relative translation scales from pairwise camera-direction constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Estimate relative camera-center scales from pairwise relative-pose directions."
    )
    parser.add_argument("--relative-pose-json", required=True, help="Cheirality-corrected relative pose JSON")
    parser.add_argument("--rough-extrinsics-json", required=True, help="Cheirality-corrected rough extrinsics JSON")
    parser.add_argument("--view-ids", nargs="+", required=True, help="View ids in the global camera graph")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--anchor-view", default=None, help="Anchor view; defaults to rough extrinsics anchor")
    parser.add_argument(
        "--scale-mode",
        choices=["median_initial_distance", "unit_median_distance"],
        default="median_initial_distance",
        help="Gauge choice for the unavoidable global scale ambiguity.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON payload."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def camera_center_from_view_meta(view_meta: dict[str, Any]) -> np.ndarray:
    """Return camera center from world-to-camera extrinsics."""

    rotation = np.asarray(view_meta["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(view_meta["translation"], dtype=np.float64).reshape(3)
    return -rotation.T @ translation


def pair_direction_in_view_a(pair: dict[str, Any]) -> np.ndarray:
    """Return unit camera-center direction from view_a to view_b in view_a coordinates."""

    rotation_ab = np.asarray(pair["rotation"], dtype=np.float64).reshape(3, 3)
    translation_ab = np.asarray(pair["translation_unit"], dtype=np.float64).reshape(3)
    direction = -rotation_ab.T @ translation_ab
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError(f"Degenerate translation direction for pair {pair.get('view_a')}->{pair.get('view_b')}")
    return direction / norm


def skew(vector: np.ndarray) -> np.ndarray:
    """Return a skew-symmetric matrix for cross products."""

    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def build_linear_system(
    relative_pose_results: dict[str, Any],
    initial_extrinsics: dict[str, Any],
    view_ids: list[str],
    anchor_view: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    """Build homogeneous linear constraints for global camera centers."""

    variable_views = [view_id for view_id in view_ids if view_id != anchor_view]
    view_to_col = {view_id: idx * 3 for idx, view_id in enumerate(variable_views)}
    num_vars = len(variable_views) * 3
    extrinsics = initial_extrinsics["extrinsics_by_view"]
    rows = []
    row_meta = []

    for pair in relative_pose_results.get("pair_relative_poses", []):
        if pair.get("status") != "ok":
            continue
        view_a = str(pair["view_a"])
        view_b = str(pair["view_b"])
        if view_a not in view_ids or view_b not in view_ids:
            continue

        rotation_a = np.asarray(extrinsics[view_a]["rotation"], dtype=np.float64).reshape(3, 3)
        direction_a = pair_direction_in_view_a(pair)
        block = skew(direction_a) @ rotation_a

        row_block = np.zeros((3, num_vars), dtype=np.float64)
        if view_b != anchor_view:
            col_b = view_to_col[view_b]
            row_block[:, col_b : col_b + 3] += block
        if view_a != anchor_view:
            col_a = view_to_col[view_a]
            row_block[:, col_a : col_a + 3] -= block

        rows.append(row_block)
        row_meta.append(
            {
                "view_a": view_a,
                "view_b": view_b,
                "direction_in_view_a": direction_a.tolist(),
            }
        )

    if not rows:
        raise ValueError("No valid pairwise constraints found.")
    return np.vstack(rows), row_meta, view_to_col


def unpack_centers(solution: np.ndarray, view_ids: list[str], anchor_view: str, view_to_col: dict[str, int]) -> dict[str, np.ndarray]:
    """Convert a flattened solution vector into camera centers."""

    centers = {anchor_view: np.zeros(3, dtype=np.float64)}
    for view_id in view_ids:
        if view_id == anchor_view:
            continue
        col = view_to_col[view_id]
        centers[view_id] = np.asarray(solution[col : col + 3], dtype=np.float64)
    return centers


def initial_centers(initial_extrinsics: dict[str, Any], view_ids: list[str]) -> dict[str, np.ndarray]:
    """Return current rough camera centers."""

    return {
        view_id: camera_center_from_view_meta(initial_extrinsics["extrinsics_by_view"][view_id])
        for view_id in view_ids
    }


def normalize_solution_scale(
    centers: dict[str, np.ndarray],
    initial: dict[str, np.ndarray],
    anchor_view: str,
    scale_mode: str,
) -> tuple[dict[str, np.ndarray], float]:
    """Fix global scale and sign gauge for the homogeneous solution."""

    variable_views = [view_id for view_id in centers if view_id != anchor_view]
    dots = [
        float(np.dot(centers[view_id], initial[view_id]))
        for view_id in variable_views
        if np.linalg.norm(centers[view_id]) > 1e-9 and np.linalg.norm(initial[view_id]) > 1e-9
    ]
    sign = -1.0 if dots and float(np.median(dots)) < 0.0 else 1.0
    signed = {view_id: center * sign for view_id, center in centers.items()}

    solution_distances = [
        float(np.linalg.norm(signed[view_id]))
        for view_id in variable_views
        if np.linalg.norm(signed[view_id]) > 1e-9
    ]
    if scale_mode == "unit_median_distance":
        target_median = 1.0
    else:
        initial_distances = [
            float(np.linalg.norm(initial[view_id]))
            for view_id in variable_views
            if np.linalg.norm(initial[view_id]) > 1e-9
        ]
        target_median = float(np.median(initial_distances)) if initial_distances else 1.0

    source_median = float(np.median(solution_distances)) if solution_distances else 1.0
    scale = target_median / max(source_median, 1e-9)
    normalized = {view_id: center * scale for view_id, center in signed.items()}
    return normalized, scale * sign


def solve_scaled_centers(
    relative_pose_results: dict[str, Any],
    initial_extrinsics: dict[str, Any],
    view_ids: list[str],
    anchor_view: str,
    scale_mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Solve global camera centers from pairwise direction constraints."""

    system, row_meta, view_to_col = build_linear_system(
        relative_pose_results=relative_pose_results,
        initial_extrinsics=initial_extrinsics,
        view_ids=view_ids,
        anchor_view=anchor_view,
    )
    _, singular_values, vh = np.linalg.svd(system)
    solution = vh[-1]
    raw_centers = unpack_centers(solution, view_ids=view_ids, anchor_view=anchor_view, view_to_col=view_to_col)
    init_centers = initial_centers(initial_extrinsics, view_ids=view_ids)
    centers, applied_scale = normalize_solution_scale(
        centers=raw_centers,
        initial=init_centers,
        anchor_view=anchor_view,
        scale_mode=scale_mode,
    )
    diagnostics = {
        "num_constraints": len(row_meta),
        "system_shape": list(system.shape),
        "singular_values": singular_values.tolist(),
        "condition_ratio_smallest_over_largest": (
            float(singular_values[-1] / singular_values[0]) if len(singular_values) and singular_values[0] > 1e-12 else None
        ),
        "applied_global_scale": float(applied_scale),
        "scale_mode": scale_mode,
    }
    return centers, diagnostics


def pair_direction_residuals(
    centers: dict[str, np.ndarray],
    relative_pose_results: dict[str, Any],
    rough_extrinsics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Measure angular residuals between solved centers and pairwise directions."""

    residuals = []
    extrinsics = rough_extrinsics["extrinsics_by_view"]
    for pair in relative_pose_results.get("pair_relative_poses", []):
        if pair.get("status") != "ok":
            continue
        view_a = str(pair["view_a"])
        view_b = str(pair["view_b"])
        if view_a not in centers or view_b not in centers:
            continue
        rotation_a = np.asarray(extrinsics[view_a]["rotation"], dtype=np.float64).reshape(3, 3)
        predicted = rotation_a @ (centers[view_b] - centers[view_a])
        predicted_norm = float(np.linalg.norm(predicted))
        observed = pair_direction_in_view_a(pair)
        if predicted_norm < 1e-9:
            angle = None
        else:
            predicted_unit = predicted / predicted_norm
            cosine = float(np.clip(np.dot(predicted_unit, observed), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
        residuals.append(
            {
                "view_a": view_a,
                "view_b": view_b,
                "angle_error_deg": angle,
                "predicted_distance": predicted_norm,
            }
        )
    return residuals


def build_refined_extrinsics(
    rough_extrinsics: dict[str, Any],
    centers: dict[str, np.ndarray],
    view_ids: list[str],
    anchor_view: str,
) -> dict[str, Any]:
    """Build rough_extrinsics-compatible payload with refined camera centers."""

    refined = {
        "anchor_view": anchor_view,
        "translation_scale_note": (
            "Camera centers are refined from pairwise direction constraints. "
            "Global metric scale remains arbitrary."
        ),
        "num_resolved_views": len(view_ids),
        "unresolved_views": [],
        "extrinsics_by_view": {},
    }
    for view_id in view_ids:
        original = rough_extrinsics["extrinsics_by_view"][view_id]
        rotation = np.asarray(original["rotation"], dtype=np.float64).reshape(3, 3)
        center = np.asarray(centers[view_id], dtype=np.float64).reshape(3)
        translation = -rotation @ center
        projection = np.asarray(original["intrinsics"]["camera_matrix"], dtype=np.float64) @ np.hstack(
            [rotation, translation.reshape(3, 1)]
        )
        refined["extrinsics_by_view"][view_id] = {
            **original,
            "is_anchor": view_id == anchor_view,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "projection_matrix": projection.tolist(),
            "camera_center_world": center.tolist(),
            "scale_refinement": {
                "source": "pairwise_direction_linear_solve",
                "distance_from_anchor": float(np.linalg.norm(center)),
            },
        }
    return refined


def main() -> None:
    """Run translation scale refinement."""

    args = parse_args()
    relative_pose_results = load_json(args.relative_pose_json)
    rough_extrinsics = load_json(args.rough_extrinsics_json)
    view_ids = list(args.view_ids)
    anchor_view = args.anchor_view or rough_extrinsics.get("anchor_view") or view_ids[0]
    if anchor_view not in view_ids:
        raise ValueError(f"Anchor view not in view ids: {anchor_view}")

    centers, solve_diagnostics = solve_scaled_centers(
        relative_pose_results=relative_pose_results,
        initial_extrinsics=rough_extrinsics,
        view_ids=view_ids,
        anchor_view=anchor_view,
        scale_mode=str(args.scale_mode),
    )
    refined_extrinsics = build_refined_extrinsics(
        rough_extrinsics=rough_extrinsics,
        centers=centers,
        view_ids=view_ids,
        anchor_view=anchor_view,
    )
    residuals = pair_direction_residuals(
        centers=centers,
        relative_pose_results=relative_pose_results,
        rough_extrinsics=rough_extrinsics,
    )
    finite_angles = [
        float(item["angle_error_deg"])
        for item in residuals
        if item.get("angle_error_deg") is not None
    ]
    summary = {
        "stage": "translation_scale_refinement",
        "anchor_view": anchor_view,
        "solve_diagnostics": solve_diagnostics,
        "camera_centers": {
            view_id: {
                "center": centers[view_id].tolist(),
                "distance_from_anchor": float(np.linalg.norm(centers[view_id])),
            }
            for view_id in view_ids
        },
        "pair_direction_residual_summary": {
            "mean_angle_error_deg": float(np.mean(finite_angles)) if finite_angles else None,
            "median_angle_error_deg": float(np.median(finite_angles)) if finite_angles else None,
            "max_angle_error_deg": float(np.max(finite_angles)) if finite_angles else None,
        },
        "pair_direction_residuals": residuals,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scale_refined_rough_extrinsics.json").open("w", encoding="utf-8") as handle:
        json.dump(refined_extrinsics, handle, ensure_ascii=False, indent=2)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
