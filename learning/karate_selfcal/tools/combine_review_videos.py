"""Combine synchronized review videos into one labeled comparison video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine synchronized videos horizontally.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")
    captures = [cv2.VideoCapture(str(Path(path).resolve())) for path in args.inputs]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("Failed to open one or more input videos")

    fps_values = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
    fps = min(value for value in fps_values if value > 0)
    width = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * len(captures), height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video: {output_path}")

    frame_count = 0
    try:
        while True:
            frames = []
            for capture, label in zip(captures, args.labels):
                ok, frame = capture.read()
                if not ok:
                    frames = []
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                cv2.rectangle(frame, (12, 12), (width - 12, 58), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    label,
                    (24, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                frames.append(frame)
            if not frames:
                break
            writer.write(np.hstack(frames))
            frame_count += 1
    finally:
        writer.release()
        for capture in captures:
            capture.release()

    print(f"output={output_path}")
    print(f"frames={frame_count}")
    print(f"fps={fps}")
    print(f"size={width * len(captures)}x{height}")


if __name__ == "__main__":
    main()
