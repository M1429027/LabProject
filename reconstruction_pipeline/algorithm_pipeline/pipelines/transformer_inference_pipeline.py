"""Four-camera transformer inference pipeline.

This keeps the learning system isolated under learning/rumpl_fourview and only performs
real-world adapter work here: loading aligned keypoints, building rays, running a checkpoint,
and exporting a skeleton_3d.json compatible with the existing real_world_pipeline schema.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reconstruction_pipeline.algorithm_pipeline.modules.transformer_adapter.postprocess import predictions_to_skeleton_frames
from reconstruction_pipeline.algorithm_pipeline.modules.transformer_adapter.ray_builder import build_sequence_ray_tensors, load_camera_streams
from reconstruction_pipeline.algorithm_pipeline.modules.transformer_adapter.runner import TransformerRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four-view transformer inference on aligned keypoint JSONs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)

    for camera_index in range(1, 5):
        parser.add_argument(f"--keypoints-cam{camera_index}-json", default="")
        parser.add_argument(f"--cam{camera_index}-intrinsics", default="")
        parser.add_argument(f"--cam{camera_index}-extrinsics", default="")

    return parser.parse_args()


def collect_camera_inputs(args: argparse.Namespace):
    cameras = []
    for camera_index in range(1, 5):
        keypoints_json = getattr(args, f"keypoints_cam{camera_index}_json")
        intrinsics = getattr(args, f"cam{camera_index}_intrinsics")
        extrinsics = getattr(args, f"cam{camera_index}_extrinsics")
        if keypoints_json:
            if not intrinsics or not extrinsics:
                raise ValueError(f"cam{camera_index} requires matching intrinsics and extrinsics when keypoints are provided.")
            cameras.append(
                {
                    "name": f"cam{camera_index}",
                    "keypoints_json": keypoints_json,
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                }
            )
    if len(cameras) < 2:
        raise ValueError("Provide at least two cameras for transformer inference.")
    return cameras


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    camera_inputs = collect_camera_inputs(args)
    streams = load_camera_streams(camera_inputs)
    ray_tokens, view_masks, frame_indices, camera_names_per_frame = build_sequence_ray_tensors(
        streams=streams,
        confidence_threshold=args.confidence_threshold,
    )

    runner = TransformerRunner(checkpoint_path=args.checkpoint, device=args.device)
    predictions = runner.predict(ray_tokens=ray_tokens, view_mask=view_masks, batch_size=args.batch_size)
    frames = predictions_to_skeleton_frames(
        predictions=predictions,
        ray_tokens=ray_tokens,
        frame_indices=frame_indices,
        camera_names_per_frame=camera_names_per_frame,
    )

    skeleton_path = out_dir / "skeleton_3d.json"
    with open(skeleton_path, "w", encoding="utf-8") as handle:
        json.dump({"frames": frames}, handle, indent=2)

    summary = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "num_frames": len(frames),
        "cameras": [item["name"] for item in camera_inputs],
        "confidence_threshold": args.confidence_threshold,
    }
    with open(out_dir / "transformer_inference_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved transformer inference outputs to {out_dir}")


if __name__ == "__main__":
    main()
