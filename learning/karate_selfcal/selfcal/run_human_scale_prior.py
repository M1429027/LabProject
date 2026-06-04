"""Apply a human-height prior to fix the global translation scale gauge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


NOSE = 0
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Scale rough extrinsics by matching triangulated skeleton height to a human-height prior."
    )
    parser.add_argument("--rough-extrinsics-json", required=True, help="Input rough_extrinsics JSON")
    parser.add_argument("--triangulated-json", required=True, help="Triangulated 3D JSON used to estimate current height")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--target-height", type=float, default=1.65, help="Target human height in arbitrary metric units")
    parser.add_argument(
        "--min-height",
        type=float,
        default=0.05,
        help="Reject skeleton height estimates below this value before taking the median",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=5.0,
        help="Reject skeleton height estimates above this value before taking the median",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON payload."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def joints_by_id(identity: dict[str, Any]) -> dict[int, np.ndarray]:
    """Return 3D joints indexed by COCO joint id."""

    out = {}
    for joint in identity.get("joints", []):
        out[int(joint["id"])] = np.array(
            [float(joint["x"]), float(joint["y"]), float(joint["z"])],
            dtype=np.float64,
        )
    return out


def estimate_identity_height(identity: dict[str, Any]) -> float | None:
    """Estimate one skeleton height from nose to mid-ankle."""

    joints = joints_by_id(identity)
    if NOSE not in joints or LEFT_ANKLE not in joints or RIGHT_ANKLE not in joints:
        return None
    mid_ankle = (joints[LEFT_ANKLE] + joints[RIGHT_ANKLE]) * 0.5
    height = float(np.linalg.norm(joints[NOSE] - mid_ankle))
    if not np.isfinite(height):
        return None
    return height


def collect_height_estimates(
    triangulated: dict[str, Any],
    min_height: float,
    max_height: float,
) -> list[dict[str, Any]]:
    """Collect valid per-frame/person skeleton height estimates."""

    estimates = []
    for frame in triangulated.get("frames", []):
        frame_idx = int(frame["frame"])
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            height = estimate_identity_height(identity)
            if height is None:
                continue
            if height < float(min_height) or height > float(max_height):
                continue
            estimates.append(
                {
                    "frame": frame_idx,
                    "identity_id": identity_id,
                    "height": height,
                }
            )
    return estimates


def scale_extrinsics(rough_extrinsics: dict[str, Any], scale: float) -> dict[str, Any]:
    """Scale all camera translations and centers by a global factor."""

    scaled = dict(rough_extrinsics)
    scaled["extrinsics_by_view"] = {}
    for view_id, view_meta in rough_extrinsics.get("extrinsics_by_view", {}).items():
        new_meta = dict(view_meta)
        rotation = np.asarray(view_meta["rotation"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(view_meta["translation"], dtype=np.float64).reshape(3) * float(scale)
        projection = np.asarray(view_meta["intrinsics"]["camera_matrix"], dtype=np.float64) @ np.hstack(
            [rotation, translation.reshape(3, 1)]
        )
        new_meta["translation"] = translation.tolist()
        new_meta["projection_matrix"] = projection.tolist()
        if "camera_center_world" in view_meta:
            new_meta["camera_center_world"] = (
                np.asarray(view_meta["camera_center_world"], dtype=np.float64).reshape(3) * float(scale)
            ).tolist()
        else:
            new_meta["camera_center_world"] = (-rotation.T @ translation).tolist()
        new_meta["human_scale_prior"] = {
            "applied_global_scale": float(scale),
        }
        scaled["extrinsics_by_view"][view_id] = new_meta
    scaled["translation_scale_note"] = (
        "Camera translations are globally scaled by a human-height prior. "
        "Relative camera directions are unchanged."
    )
    scaled["human_scale_prior"] = {
        "applied_global_scale": float(scale),
    }
    return scaled


def main() -> None:
    """Run human-height scale prior."""

    args = parse_args()
    rough_extrinsics = load_json(args.rough_extrinsics_json)
    triangulated = load_json(args.triangulated_json)
    estimates = collect_height_estimates(
        triangulated=triangulated,
        min_height=float(args.min_height),
        max_height=float(args.max_height),
    )
    if not estimates:
        raise ValueError("No valid human-height estimates found.")

    heights = np.array([item["height"] for item in estimates], dtype=np.float64)
    observed_median_height = float(np.median(heights))
    applied_scale = float(args.target_height) / max(observed_median_height, 1e-9)
    scaled_extrinsics = scale_extrinsics(
        rough_extrinsics=rough_extrinsics,
        scale=applied_scale,
    )
    scaled_extrinsics["human_scale_prior"] = {
        "target_height": float(args.target_height),
        "observed_median_height": observed_median_height,
        "observed_mean_height": float(np.mean(heights)),
        "observed_std_height": float(np.std(heights)),
        "num_height_estimates": len(estimates),
        "applied_global_scale": applied_scale,
        "height_method": "COCO nose to midpoint(left_ankle, right_ankle)",
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_extrinsics = output_dir / "human_scaled_rough_extrinsics.json"
    summary = {
        "stage": "human_scale_prior",
        "rough_extrinsics_json": str(Path(args.rough_extrinsics_json).resolve()),
        "triangulated_json": str(Path(args.triangulated_json).resolve()),
        "target_height": float(args.target_height),
        "observed_median_height": observed_median_height,
        "observed_mean_height": float(np.mean(heights)),
        "observed_std_height": float(np.std(heights)),
        "num_height_estimates": len(estimates),
        "applied_global_scale": applied_scale,
        "output_extrinsics_json": str(output_extrinsics),
    }
    write_json(output_extrinsics, scaled_extrinsics)
    write_json(output_dir / "run_summary.json", summary)
    write_json(output_dir / "height_estimates.json", {"estimates": estimates})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
