"""Render a top/bottom GT-vs-prediction 3D skeleton comparison video."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from . import visualize_triangulation as viz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GT and prediction skeletons into one comparison MP4.")
    parser.add_argument("--sequence", required=True, help="Sequence id, e.g. 11_karate3/008_karate3.")
    parser.add_argument("--zip-path", required=True, help="Harmony4D zip path containing processed_data/poses3d.")
    parser.add_argument("--prediction-json", required=True, help="Prediction JSON in visualize_triangulation format.")
    parser.add_argument("--output-video", required=True, help="Output MP4 path.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--single-height", type=int, default=720)
    parser.add_argument("--axis-percentile", type=float, default=98.0)
    return parser.parse_args()


def export_gt_payload(sequence: str, zip_path: str | Path) -> dict:
    frames = []
    with zipfile.ZipFile(zip_path) as archive:
        frame_ids = sorted(
            int(Path(name).stem)
            for name in archive.namelist()
            if name.startswith(f"{sequence}/processed_data/poses3d/") and name.endswith(".npy")
        )
        for frame_id in frame_ids:
            payload = np.load(
                io.BytesIO(archive.read(f"{sequence}/processed_data/poses3d/{frame_id:05d}.npy")),
                allow_pickle=True,
            ).item()
            identities = []
            for person_id, joints3d in sorted(payload.items()):
                array = np.asarray(joints3d, dtype=float)[:17]
                identity_id = 0 if str(person_id).endswith("01") else 1
                joints = []
                for joint_id, row in enumerate(array):
                    confidence = float(row[3]) if array.shape[1] > 3 else 1.0
                    if confidence <= 0.0:
                        continue
                    joints.append(
                        {
                            "id": int(joint_id),
                            "x": float(row[0]),
                            "y": float(row[1]),
                            "z": float(row[2]),
                            "confidence": confidence,
                        }
                    )
                identities.append(
                    {
                        "identity_id": identity_id,
                        "person_id": str(person_id),
                        "joints": joints,
                    }
                )
            frames.append({"sequence": sequence, "frame": int(frame_id), "identities": identities})
    return {
        "metadata": {
            "stage": "gt_3d_export_for_comparison",
            "sequence": sequence,
            "num_frames": len(frames),
        },
        "frames": frames,
    }


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output_video = Path(args.output_video).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    gt_payload = export_gt_payload(args.sequence, args.zip_path)
    prediction_payload = load_json(args.prediction_json)
    gt_json_path = output_video.with_name(f"{output_video.stem}_gt.json")
    gt_json_path.write_text(json.dumps(gt_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    gt_frames = list(gt_payload.get("frames", []))
    prediction_frames = list(prediction_payload.get("frames", []))
    num_frames = min(len(gt_frames), len(prediction_frames))
    gt_frames = gt_frames[:num_frames]
    prediction_frames = prediction_frames[:num_frames]

    axis_limits = viz.estimate_axis_limits(
        gt_frames + prediction_frames,
        up_axis="z",
        flip_up_axis=False,
        axis_percentile=float(args.axis_percentile),
    )

    width = int(args.width)
    single_height = int(args.single_height)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, single_height * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video: {output_video}")

    try:
        for gt_frame, prediction_frame in zip(gt_frames, prediction_frames):
            gt_image = viz.render_frame(
                frame_data=gt_frame,
                axis_limits=axis_limits,
                canvas_width=width,
                canvas_height=single_height,
                title=f"GT 3D | frame {gt_frame['frame']}",
                up_axis="z",
                flip_up_axis=False,
                highlight_key_joints=True,
            )
            prediction_image = viz.render_frame(
                frame_data=prediction_frame,
                axis_limits=axis_limits,
                canvas_width=width,
                canvas_height=single_height,
                title=f"Composed prediction | frame {prediction_frame['frame']}",
                up_axis="z",
                flip_up_axis=False,
                highlight_key_joints=True,
            )
            writer.write(np.vstack([gt_image, prediction_image]))
    finally:
        writer.release()

    summary = {
        "stage": "render_gt_vs_prediction",
        "sequence": args.sequence,
        "gt_json": str(gt_json_path),
        "prediction_json": str(Path(args.prediction_json).resolve()),
        "output_video": str(output_video),
        "num_frames": num_frames,
        "fps": float(args.fps),
        "size": [width, single_height * 2],
        "layout": "top=GT, bottom=composed prediction",
        "key_joint_colors": "shoulders=cyan, wrists=red, hips=yellow, ankles=magenta",
        "axis_percentile": float(args.axis_percentile),
    }
    output_video.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
