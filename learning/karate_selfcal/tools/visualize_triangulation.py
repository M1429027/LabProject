"""Visualize rough triangulation outputs as 3D skeleton videos."""

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

TRACK_COLORS = [
    "#2ECC71",
    "#3498DB",
    "#E67E22",
    "#9B59B6",
    "#E74C3C",
    "#1ABC9C",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Render rough triangulation 3D skeletons into a review video."
    )
    parser.add_argument("--input-json", required=True, help="Path to triangulated_3d.json")
    parser.add_argument("--output-video", default=None, help="Optional output MP4 path")
    parser.add_argument("--fps", type=float, default=10.0, help="Output FPS")
    parser.add_argument("--width", type=int, default=1280, help="Output width")
    parser.add_argument("--height", type=int, default=720, help="Output height")
    parser.add_argument("--start-frame", type=int, default=0, help="Optional start frame index")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum rendered frames")
    parser.add_argument(
        "--per-frame-axis",
        action="store_true",
        help="Auto-scale axes per frame instead of using one global axis range for the whole clip",
    )
    parser.add_argument(
        "--up-axis",
        choices=["x", "y", "z"],
        default="z",
        help=(
            "Source coordinate axis to display as the vertical plot axis. "
            "Default z keeps the original source X/Y/Z display."
        ),
    )
    parser.add_argument(
        "--flip-up-axis",
        action="store_true",
        help="Flip the selected display up axis. This is only for visual inspection.",
    )
    parser.add_argument(
        "--mark-review-imputed",
        action="store_true",
        help="Render review-imputed joints with lighter markers and lines.",
    )
    parser.add_argument(
        "--axis-percentile",
        type=float,
        default=None,
        help="Optional central percentile for global axis limits, e.g. 98 ignores the outer 1%% on each side.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    """Load triangulation JSON."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def remap_point_for_display(point: np.ndarray, up_axis: str, flip_up_axis: bool) -> np.ndarray:
    """Map source coordinates into display coordinates with the requested vertical axis."""

    if up_axis == "x":
        display_point = np.array([point[1], point[2], point[0]], dtype=np.float32)
    elif up_axis == "y":
        display_point = np.array([point[0], point[2], point[1]], dtype=np.float32)
    else:
        display_point = point.astype(np.float32)

    if flip_up_axis:
        display_point[2] *= -1.0
    return display_point


def axis_labels(up_axis: str, flip_up_axis: bool) -> tuple[str, str, str]:
    """Return display labels that preserve the source-axis meaning."""

    sign = "-" if flip_up_axis else ""
    if up_axis == "x":
        return "source Y", "source Z", f"{sign}source X (up)"
    if up_axis == "y":
        return "source X", "source Z", f"{sign}source Y (up)"
    return "source X", "source Y", f"{sign}source Z (up)"


def frame_coords(identity: dict[str, Any], up_axis: str, flip_up_axis: bool) -> dict[int, np.ndarray]:
    """Convert one identity entry into joint-id indexed coordinates."""

    coords = {}
    for joint in identity.get("joints", []):
        source_point = np.array(
            [float(joint["x"]), float(joint["y"]), float(joint["z"])],
            dtype=np.float32,
        )
        coords[int(joint["id"])] = remap_point_for_display(source_point, up_axis, flip_up_axis)
    return coords


def frame_joint_sources(identity: dict[str, Any]) -> dict[int, str]:
    """Return joint-id indexed provenance labels for review-smoothed outputs."""

    sources = {}
    for joint in identity.get("joints", []):
        sources[int(joint["id"])] = str(joint.get("review_source", "original"))
    return sources


def estimate_axis_limits(
    frames: list[dict[str, Any]],
    up_axis: str,
    flip_up_axis: bool,
    axis_percentile: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Estimate stable axis limits across the clip."""

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for frame in frames:
        for identity in frame.get("identities", []):
            for point in frame_coords(identity, up_axis, flip_up_axis).values():
                xs.append(float(point[0]))
                ys.append(float(point[1]))
                zs.append(float(point[2]))

    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)

    def _center_and_half(values: list[float]) -> tuple[float, float]:
        if axis_percentile is not None:
            clipped_percentile = float(np.clip(axis_percentile, 50.0, 100.0))
            tail = (100.0 - clipped_percentile) * 0.5
            low = float(np.percentile(values, tail))
            high = float(np.percentile(values, 100.0 - tail))
        else:
            low = float(np.min(values))
            high = float(np.max(values))
        center = (low + high) * 0.5
        half_span = (high - low) * 0.5
        return center, half_span

    centers_and_halves = [_center_and_half(xs), _center_and_half(ys), _center_and_half(zs)]
    half_span = max([item[1] for item in centers_and_halves] + [0.5]) * 1.15

    def _limits(center: float) -> tuple[float, float]:
        return center - half_span, center + half_span

    return tuple(_limits(center) for center, _ in centers_and_halves)  # type: ignore[return-value]


def render_frame(
    frame_data: dict[str, Any],
    axis_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    canvas_width: int,
    canvas_height: int,
    title: str,
    up_axis: str,
    flip_up_axis: bool,
    mark_review_imputed: bool = False,
) -> np.ndarray:
    """Render one frame of triangulated identities."""

    fig = plt.figure(figsize=(canvas_width / 100.0, canvas_height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    xlabel, ylabel, zlabel = axis_labels(up_axis, flip_up_axis)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=18, azim=-62)

    xlim, ylim, zlim = axis_limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass
    ax.grid(True, alpha=0.3)

    for identity in frame_data.get("identities", []):
        identity_id = int(identity["identity_id"])
        color = TRACK_COLORS[identity_id % len(TRACK_COLORS)]
        coords = frame_coords(identity, up_axis, flip_up_axis)
        sources = frame_joint_sources(identity)
        if not coords:
            continue

        if mark_review_imputed:
            original_points = [
                point
                for joint_id, point in coords.items()
                if sources.get(joint_id, "original") == "original"
            ]
            imputed_points = [
                point
                for joint_id, point in coords.items()
                if sources.get(joint_id, "original") != "original"
            ]
            if original_points:
                arr = np.vstack(original_points)
                ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=28, c=color, depthshade=True, label=f"id {identity_id}")
            if imputed_points:
                arr = np.vstack(imputed_points)
                ax.scatter(
                    arr[:, 0],
                    arr[:, 1],
                    arr[:, 2],
                    s=18,
                    c=color,
                    alpha=0.35,
                    marker="x",
                    depthshade=False,
                    label=f"id {identity_id} imputed",
                )
        else:
            xs = [point[0] for point in coords.values()]
            ys = [point[1] for point in coords.values()]
            zs = [point[2] for point in coords.values()]
            ax.scatter(xs, ys, zs, s=28, c=color, depthshade=True, label=f"id {identity_id}")

        for joint_a, joint_b in COCO_SKELETON_CONNECTIONS:
            if joint_a not in coords or joint_b not in coords:
                continue
            segment = np.vstack([coords[joint_a], coords[joint_b]])
            has_imputed = (
                sources.get(joint_a, "original") != "original"
                or sources.get(joint_b, "original") != "original"
            )
            alpha = 0.35 if mark_review_imputed and has_imputed else 0.95
            linewidth = 1.2 if mark_review_imputed and has_imputed else 2.0
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], c=color, linewidth=linewidth, alpha=alpha)

    if frame_data.get("identities"):
        ax.legend(loc="upper left")

    buffer = BytesIO()
    plt.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    image = cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    return image


def estimate_frame_axis_limits(
    frame_data: dict[str, Any],
    up_axis: str,
    flip_up_axis: bool,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Estimate axis limits for one frame only."""

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for identity in frame_data.get("identities", []):
        for point in frame_coords(identity, up_axis, flip_up_axis).values():
            xs.append(float(point[0]))
            ys.append(float(point[1]))
            zs.append(float(point[2]))

    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)

    def _center_and_half(values: list[float]) -> tuple[float, float]:
        low = float(np.min(values))
        high = float(np.max(values))
        center = (low + high) * 0.5
        half_span = (high - low) * 0.5
        return center, half_span

    centers_and_halves = [_center_and_half(xs), _center_and_half(ys), _center_and_half(zs)]
    half_span = max([item[1] for item in centers_and_halves] + [0.15]) * 1.35

    def _limits(center: float) -> tuple[float, float]:
        return center - half_span, center + half_span

    return tuple(_limits(center) for center, _ in centers_and_halves)  # type: ignore[return-value]


def main() -> None:
    """Render the triangulation payload into an MP4 video."""

    args = parse_args()
    input_path = Path(args.input_json).resolve()
    payload = load_payload(input_path)

    frames = list(payload.get("frames", []))
    frames = [frame for frame in frames if int(frame.get("frame", 0)) >= int(args.start_frame)]
    if args.max_frames is not None:
        frames = frames[: int(args.max_frames)]

    output_path = (
        Path(args.output_video).resolve()
        if args.output_video
        else input_path.with_name(f"{input_path.stem}_review.mp4")
    )

    axis_limits = estimate_axis_limits(
        frames,
        args.up_axis,
        args.flip_up_axis,
        axis_percentile=args.axis_percentile,
    )
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (int(args.width), int(args.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video: {output_path}")

    try:
        for frame_data in frames:
            current_axis_limits = (
                estimate_frame_axis_limits(frame_data, args.up_axis, args.flip_up_axis)
                if args.per_frame_axis
                else axis_limits
            )
            title = (
                f"rough triangulation | frame {frame_data['frame']} | "
                f"anchor {payload.get('metadata', {}).get('anchor_view', 'unknown')} | "
                f"{args.up_axis.upper()} up"
            )
            image = render_frame(
                frame_data=frame_data,
                axis_limits=current_axis_limits,
                canvas_width=int(args.width),
                canvas_height=int(args.height),
                title=title,
                up_axis=args.up_axis,
                flip_up_axis=args.flip_up_axis,
                mark_review_imputed=bool(args.mark_review_imputed),
            )
            writer.write(image)
    finally:
        writer.release()

    summary = {
        "stage": "visualize_triangulation",
        "input_json": str(input_path),
        "output_video": str(output_path),
        "num_frames": len(frames),
        "fps": float(args.fps),
        "size": [int(args.width), int(args.height)],
        "up_axis": args.up_axis,
        "flip_up_axis": bool(args.flip_up_axis),
        "mark_review_imputed": bool(args.mark_review_imputed),
        "axis_percentile": args.axis_percentile,
    }
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
