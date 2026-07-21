"""Render synchronized 3D skeleton JSON payloads with shared camera and axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from learning.karate_selfcal.tools.visualize_triangulation import (
    estimate_axis_limits,
    render_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument(
        "--frame-offsets",
        nargs="+",
        type=int,
        required=True,
        help="Source frame ID corresponding to normalized comparison frame 0.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--axis-percentile", type=float, default=98.0)
    return parser.parse_args()


def load_frames(path: Path, offset: int) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        int(frame.get("frame", 0)) - int(offset): frame
        for frame in payload.get("frames", [])
    }


def main() -> None:
    args = parse_args()
    if not (len(args.inputs) == len(args.labels) == len(args.frame_offsets)):
        raise ValueError("--inputs, --labels, and --frame-offsets must have equal lengths")

    input_paths = [Path(path).resolve() for path in args.inputs]
    frame_maps = [
        load_frames(path, offset)
        for path, offset in zip(input_paths, args.frame_offsets)
    ]
    common_frames = sorted(set.intersection(*(set(mapping) for mapping in frame_maps)))
    if not common_frames:
        raise ValueError("Input payloads have no common normalized frame IDs")

    shared_frames = [mapping[frame_id] for mapping in frame_maps for frame_id in common_frames]
    axis_limits = estimate_axis_limits(
        shared_frames,
        up_axis="z",
        flip_up_axis=False,
        axis_percentile=float(args.axis_percentile),
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (int(args.panel_width) * len(frame_maps), int(args.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video: {output_path}")

    preview_path = output_path.with_name(f"{output_path.stem}_preview.png")
    try:
        for index, frame_id in enumerate(common_frames):
            panels = [
                render_frame(
                    frame_data=mapping[frame_id],
                    axis_limits=axis_limits,
                    canvas_width=int(args.panel_width),
                    canvas_height=int(args.height),
                    title=f"{label} | frame {frame_id}",
                    up_axis="z",
                    flip_up_axis=False,
                    highlight_key_joints=True,
                )
                for label, mapping in zip(args.labels, frame_maps)
            ]
            comparison = np.concatenate(panels, axis=1)
            writer.write(comparison)
            if index == len(common_frames) // 2 and not cv2.imwrite(str(preview_path), comparison):
                raise RuntimeError(f"Failed to write preview image: {preview_path}")
    finally:
        writer.release()

    summary = {
        "stage": "render_triangulation_comparison",
        "inputs": [str(path) for path in input_paths],
        "labels": list(args.labels),
        "frame_offsets": list(args.frame_offsets),
        "common_frame_start": int(common_frames[0]),
        "common_frame_end": int(common_frames[-1]),
        "num_frames": len(common_frames),
        "fps": float(args.fps),
        "size": [int(args.panel_width) * len(frame_maps), int(args.height)],
        "axis_percentile": float(args.axis_percentile),
        "output_video": str(output_path),
        "preview_image": str(preview_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
