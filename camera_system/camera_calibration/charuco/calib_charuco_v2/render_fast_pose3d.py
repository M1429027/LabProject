"""Fast OpenCV renderer for a camera-following 3D skeleton review video."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def _coords(identity: dict[str, Any]) -> dict[int, np.ndarray]:
    return {
        int(joint["id"]): np.asarray([joint["x"], joint["y"], joint["z"]], dtype=np.float64)
        for joint in identity.get("joints", [])
        if np.isfinite([joint["x"], joint["y"], joint["z"]]).all()
    }


def _root(points: dict[int, np.ndarray]) -> np.ndarray | None:
    if 11 in points and 12 in points:
        return 0.5 * (points[11] + points[12])
    if 5 in points and 6 in points:
        return 0.5 * (points[5] + points[6])
    return None


def render_fast(
    input_json: str | Path,
    output_video: str | Path,
    fps: float = 30.0,
    width: int = 960,
    height: int = 720,
) -> dict[str, Any]:
    frames = json.loads(Path(input_json).read_text(encoding="utf-8")).get("frames", [])
    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output}")

    yaw = np.deg2rad(38.0)
    right = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])
    depth = np.asarray([-np.sin(yaw), np.cos(yaw), 0.0])
    scale = min(width / 3.0, height / 2.25)
    origin_px = np.asarray([width * 0.5, height * 0.57])
    colors = [(64, 220, 90), (255, 160, 40), (220, 80, 220)]
    last_root: dict[int, np.ndarray] = {}

    for frame in frames:
        canvas = np.full((height, width, 3), (20, 23, 28), dtype=np.uint8)
        cv2.putText(
            canvas, f"3D pose | frame {int(frame.get('frame', 0))}",
            (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA,
        )
        for identity in frame.get("identities", []):
            identity_id = int(identity.get("identity_id", 0))
            points = _coords(identity)
            root = _root(points)
            if root is not None:
                last_root[identity_id] = root
            center = last_root.get(identity_id, np.zeros(3, dtype=np.float64))

            def project(point: np.ndarray) -> tuple[int, int]:
                rel = point - center
                u = float(np.dot(rel, right))
                v = float(rel[2] + 0.20 * np.dot(rel, depth))
                pixel = origin_px + np.asarray([u * scale, -v * scale])
                return int(round(pixel[0])), int(round(pixel[1]))

            color = colors[identity_id % len(colors)]
            for a, b in SKELETON:
                if a in points and b in points:
                    cv2.line(canvas, project(points[a]), project(points[b]), color, 5, cv2.LINE_AA)
            for joint_id, point in points.items():
                joint_color = (40, 80, 255) if joint_id in (9, 10) else color
                cv2.circle(canvas, project(point), 7, joint_color, -1, cv2.LINE_AA)
            if root is not None:
                cv2.putText(
                    canvas, f"root=({root[0]:+.2f}, {root[1]:+.2f}, {root[2]:+.2f}) m",
                    (24, 70 + identity_id * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, color, 2, cv2.LINE_AA,
                )
        writer.write(canvas)
    writer.release()
    return {
        "stage": "fast_opencv_pose3d_renderer",
        "frames": len(frames),
        "fps": float(fps),
        "size": [width, height],
        "output": str(output.resolve()),
    }
