"""Visualize Stage 3A lifting outputs as 3D skeleton videos."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COCO_SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

H36M_SKELETON_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]

TRACK_COLORS = [
    "#2ECC71",
    "#3498DB",
    "#E67E22",
    "#9B59B6",
    "#E74C3C",
    "#1ABC9C",
]


def infer_skeleton_connections(payload: dict[str, Any]) -> list[tuple[int, int]]:
    """Choose skeleton topology from the payload metadata."""

    metadata = payload.get("metadata", {})
    pose_lift_dataset_name = str(metadata.get("pose_lift_dataset_name", "")).lower()
    keypoint_names = payload.get("keypoint_names", [])

    if pose_lift_dataset_name in {"h36m", "h3wb"}:
        return H36M_SKELETON_CONNECTIONS

    if keypoint_names:
        lowered = [str(name).lower() for name in keypoint_names]
        if "thorax" in lowered and "root" in lowered:
            return H36M_SKELETON_CONNECTIONS

    return COCO_SKELETON_CONNECTIONS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for coarse-lifting visualization."""

    parser = argparse.ArgumentParser(
        description="Render heuristic or learned 3D skeletons from lifted_<view>.json outputs."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to one lifted_<view>.json payload.",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Optional output MP4 path. Defaults next to the input JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=["root_centered", "anchored"],
        default="anchored",
        help="Visualization mode: root-centered pose only or anchored by normalized image root.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS. Defaults to source FPS from metadata.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Optional start frame index within the lifted sequence.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to render.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Output video width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Output video height.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    """Load one lifting JSON payload."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def anchored_coords(
    person: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> dict[int, np.ndarray]:
    """Map pseudo-3D joints into a globally anchored view-space."""

    root = person.get("root", {"x": 0.0, "y": 0.0})
    anchor_x = ((float(root["x"]) / max(frame_width, 1)) - 0.5) * 8.0
    anchor_y = (0.5 - (float(root["y"]) / max(frame_height, 1))) * 4.5
    coords = {}
    for point in person.get("lifted_keypoints", []):
        coords[int(point["id"])] = np.array(
            [
                float(point["x"]) + anchor_x,
                float(point["y"]) + anchor_y,
                float(point["z"]),
            ],
            dtype=np.float32,
        )
    return coords


def root_centered_coords(person: dict[str, Any]) -> dict[int, np.ndarray]:
    """Return root-centered pseudo-3D joints."""

    coords = {}
    for point in person.get("lifted_keypoints", []):
        coords[int(point["id"])] = np.array(
            [float(point["x"]), float(point["y"]), float(point["z"])],
            dtype=np.float32,
        )
    return coords


def person_coords(
    person: dict[str, Any],
    frame_width: int,
    frame_height: int,
    mode: str,
) -> dict[int, np.ndarray]:
    """Return 3D coordinates for a person under the chosen display mode."""

    if mode == "root_centered":
        return root_centered_coords(person)
    return anchored_coords(person, frame_width=frame_width, frame_height=frame_height)


def estimate_axis_limits(
    frames: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    mode: str,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Estimate stable axis limits across the rendered clip."""

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for frame in frames:
        for person in frame.get("people", []):
            coords = person_coords(
                person,
                frame_width=frame_width,
                frame_height=frame_height,
                mode=mode,
            )
            for point in coords.values():
                xs.append(float(point[0]))
                ys.append(float(point[1]))
                zs.append(float(point[2]))

    if not xs:
        return (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)

    def _pad(values: list[float], pad_ratio: float = 0.12, min_half_span: float = 0.75) -> tuple[float, float]:
        low = float(np.min(values))
        high = float(np.max(values))
        center = (low + high) * 0.5
        half_span = max((high - low) * 0.5 * (1.0 + pad_ratio), min_half_span)
        return center - half_span, center + half_span

    return _pad(xs), _pad(ys), _pad(zs)


def render_frame_image(
    frame_data: dict[str, Any],
    frame_width: int,
    frame_height: int,
    mode: str,
    axis_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    canvas_width: int,
    canvas_height: int,
    view_id: str,
    skeleton_connections: list[tuple[int, int]],
) -> np.ndarray:
    """Render one 3D frame into an RGB image."""

    fig = plt.figure(figsize=(canvas_width / 100.0, canvas_height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{view_id} | frame {frame_data['frame']} | {mode}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=18, azim=-62)

    xlim, ylim, zlim = axis_limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.grid(True, alpha=0.3)

    for person in frame_data.get("people", []):
        track_id = int(person.get("track_id", 0))
        color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
        coords = person_coords(
            person,
            frame_width=frame_width,
            frame_height=frame_height,
            mode=mode,
        )
        if not coords:
            continue

        xs = [coords[idx][0] for idx in sorted(coords)]
        ys = [coords[idx][1] for idx in sorted(coords)]
        zs = [coords[idx][2] for idx in sorted(coords)]
        ax.scatter(xs, ys, zs, c=color, s=26, depthshade=False)

        for joint_a, joint_b in skeleton_connections:
            if joint_a not in coords or joint_b not in coords:
                continue
            a = coords[joint_a]
            b = coords[joint_b]
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                [a[2], b[2]],
                color=color,
                linewidth=2.0,
                alpha=0.9,
            )

        root = person.get("root", {})
        label_point = None
        if 0 in coords:
            label_point = coords[0]
        else:
            first_key = sorted(coords)[0]
            label_point = coords[first_key]
        ax.text(
            float(label_point[0]),
            float(label_point[1]),
            float(label_point[2]) + 0.06,
            f"track {track_id}\nfront={float(person.get('frontality', 0.0)):.2f}",
            color=color,
            fontsize=8,
        )

    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    encoded = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode rendered matplotlib frame.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> None:
    """Render a coarse-lifting JSON payload into a debug MP4."""

    args = parse_args()
    input_path = Path(args.input_json).resolve()
    payload = load_payload(input_path)
    metadata = payload.get("metadata", {})
    frames = payload.get("frames", [])
    frame_width = int(metadata.get("width", 1))
    frame_height = int(metadata.get("height", 1))
    view_id = str(metadata.get("view_id", input_path.stem))

    start = max(int(args.start_frame), 0)
    selected_frames = [frame for frame in frames if int(frame["frame"]) >= start]
    if args.max_frames is not None:
        selected_frames = selected_frames[: max(int(args.max_frames), 0)]
    if not selected_frames:
        raise ValueError("No frames selected for visualization.")

    axis_limits = estimate_axis_limits(
        selected_frames,
        frame_width=frame_width,
        frame_height=frame_height,
        mode=args.mode,
    )
    skeleton_connections = infer_skeleton_connections(payload)

    output_path = Path(args.output_video).resolve() if args.output_video else input_path.with_name(
        f"{input_path.stem}_{args.mode}_3d.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(args.fps if args.fps is not None else metadata.get("fps", 20.0))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(args.width), int(args.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output writer: {output_path}")

    try:
        for frame_data in selected_frames:
            image_rgb = render_frame_image(
                frame_data=frame_data,
                frame_width=frame_width,
                frame_height=frame_height,
                mode=args.mode,
                axis_limits=axis_limits,
                canvas_width=int(args.width),
                canvas_height=int(args.height),
                view_id=view_id,
                skeleton_connections=skeleton_connections,
            )
            writer.write(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    summary = {
        "stage": "visualize_coarse_lifting",
        "input_json": str(input_path),
        "output_video": str(output_path),
        "mode": args.mode,
        "num_frames_rendered": len(selected_frames),
        "fps": fps,
        "width": int(args.width),
        "height": int(args.height),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
