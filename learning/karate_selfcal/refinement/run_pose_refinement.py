"""Run optimization-based pose-only refinement for triangulated 3D skeletons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .optimize_pose import PoseRefinementConfig, optimize_pose_payload


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Refine triangulated 3D skeletons with pose-level optimization constraints."
    )
    parser.add_argument("--input-json", required=True, help="Input triangulated / filtered 3D JSON")
    parser.add_argument("--output-json", required=True, help="Output refined 3D JSON")
    parser.add_argument("--metrics-json", required=True, help="Output refinement metrics JSON")
    parser.add_argument("--data-weight", type=float, default=1.0, help="Weight for staying near input 3D joints")
    parser.add_argument("--bone-weight", type=float, default=2.0, help="Weight for median bone-length consistency")
    parser.add_argument("--symmetry-weight", type=float, default=0.4, help="Weight for left/right bone symmetry")
    parser.add_argument("--smoothness-weight", type=float, default=0.15, help="Weight for temporal acceleration smoothing")
    parser.add_argument(
        "--reprojection-scale-px",
        type=float,
        default=20.0,
        help="Scale used to downweight joints with high stored reprojection error",
    )
    parser.add_argument("--min-bone-samples", type=int, default=8, help="Minimum samples to estimate a bone prior")
    parser.add_argument("--max-nfev", type=int, default=80, help="Maximum least-squares function evaluations")
    parser.add_argument("--robust-loss", default="soft_l1", help="scipy least_squares robust loss")
    parser.add_argument("--robust-f-scale", type=float, default=0.08, help="Robust loss transition scale")
    parser.add_argument("--window-size", type=int, default=12, help="Number of frames per optimization window")
    parser.add_argument("--window-stride", type=int, default=6, help="Stride between optimization windows")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Adam learning rate")
    parser.add_argument(
        "--bone-outlier-ratio",
        type=float,
        default=1.6,
        help="Downweight data anchors for child joints when bone length exceeds median * ratio",
    )
    parser.add_argument(
        "--bone-outlier-downweight",
        type=float,
        default=0.2,
        help="Multiplier applied to data-anchor weights for bone-outlier child joints",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON payload."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload."""

    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    """Run pose-only refinement."""

    args = parse_args()
    payload = load_json(args.input_json)
    config = PoseRefinementConfig(
        data_weight=float(args.data_weight),
        bone_weight=float(args.bone_weight),
        symmetry_weight=float(args.symmetry_weight),
        smoothness_weight=float(args.smoothness_weight),
        reprojection_scale_px=float(args.reprojection_scale_px),
        min_bone_samples=int(args.min_bone_samples),
        max_nfev=int(args.max_nfev),
        robust_loss=str(args.robust_loss),
        robust_f_scale=float(args.robust_f_scale),
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        learning_rate=float(args.learning_rate),
        bone_outlier_ratio=float(args.bone_outlier_ratio),
        bone_outlier_downweight=float(args.bone_outlier_downweight),
    )
    refined_payload, metrics = optimize_pose_payload(payload, config=config)
    write_json(args.output_json, refined_payload)
    write_json(args.metrics_json, metrics)
    print(
        json.dumps(
            {
                "stage": "optimization_pose_refinement",
                "version": "pose_only_v1",
                "output_json": str(Path(args.output_json).resolve()),
                "metrics_json": str(Path(args.metrics_json).resolve()),
                "identity_diagnostics": metrics.get("identity_diagnostics", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
