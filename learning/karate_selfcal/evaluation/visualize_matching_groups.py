"""Visualize Stage 3B multi-view matching groups in a 2x2 review grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .visualize_matches import (
    BACKGROUND_COLOR,
    COCO_SKELETON_CONNECTIONS,
    TEXT_COLOR,
    build_people_index,
    draw_person,
    load_json,
)


GROUP_COLORS = [
    (80, 255, 80),
    (80, 180, 255),
    (255, 160, 80),
    (220, 120, 255),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for group visualization."""

    parser = argparse.ArgumentParser(
        description="Render 2x2 multi-view videos for Stage 3B matching groups."
    )
    parser.add_argument("--matching-graph-json", required=True, help="Path to matching_graph.json")
    parser.add_argument("--track-jsons", nargs="+", required=True, help="Per-view tracking JSON files")
    parser.add_argument("--source-videos", nargs="+", required=True, help="Per-view source videos")
    parser.add_argument("--view-ids", nargs="+", required=True, help="View ids matching the input order")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered group videos")
    parser.add_argument("--frame-step", type=int, default=1, help="Render every Nth frame")
    parser.add_argument("--max-output-width", type=int, default=1920, help="Maximum output video width")
    parser.add_argument("--max-output-height", type=int, default=1080, help="Maximum output video height")
    return parser.parse_args()


def parse_group_nodes(group: dict[str, Any]) -> dict[str, int]:
    """Parse group nodes like view_id:track_id into a view->track map."""

    out = {}
    for node in group.get("nodes", []):
        view_id, track_id = str(node).rsplit(":", 1)
        out[view_id] = int(track_id)
    return out


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    """Read a specific frame from an open capture."""

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def resize_panel(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Resize one view panel to a fixed width while keeping aspect ratio."""

    if frame.shape[1] == target_width:
        return frame
    scale = target_width / frame.shape[1]
    target_size = (target_width, int(frame.shape[0] * scale))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def annotate_view(
    frame: np.ndarray,
    people_by_track: dict[int, dict[str, Any]],
    view_id: str,
    highlight_track: int | None,
    group_id: int,
) -> np.ndarray:
    """Draw all tracks in one view and highlight the selected group track."""

    color = GROUP_COLORS[group_id % len(GROUP_COLORS)]
    for track_id, person in sorted(people_by_track.items()):
        is_highlight = highlight_track is not None and int(track_id) == int(highlight_track)
        frame = draw_person(
            frame,
            person,
            color if is_highlight else BACKGROUND_COLOR,
            f"{view_id} t{track_id}",
            dim=not is_highlight,
        )
    cv2.putText(
        frame,
        f"{view_id} | group {group_id} track {highlight_track if highlight_track is not None else 'missing'}",
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return frame


def compose_grid(
    panels: list[np.ndarray],
    header_lines: list[str],
    max_output_size: tuple[int, int],
) -> np.ndarray:
    """Compose four panels into a 2x2 grid with a header."""

    if not panels:
        raise ValueError("At least one panel is required.")

    target_width = min(panel.shape[1] for panel in panels)
    resized = [resize_panel(panel, target_width) for panel in panels]
    target_height = min(panel.shape[0] for panel in resized)
    resized = [cv2.resize(panel, (target_width, target_height), interpolation=cv2.INTER_AREA) for panel in resized]

    while len(resized) < 4:
        resized.append(np.zeros_like(resized[0]))

    gap = 16
    header_h = 120
    row_w = target_width * 2 + gap
    row_h = target_height
    canvas = np.zeros((header_h + row_h * 2 + gap, row_w, 3), dtype=np.uint8)
    canvas[:header_h, :] = (22, 22, 22)
    canvas[header_h : header_h + row_h, 0:target_width] = resized[0]
    canvas[header_h : header_h + row_h, target_width + gap : row_w] = resized[1]
    canvas[header_h + row_h + gap :, 0:target_width] = resized[2]
    canvas[header_h + row_h + gap :, target_width + gap : row_w] = resized[3]

    y = 34
    for line in header_lines:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_COLOR, 2, cv2.LINE_AA)
        y += 30

    max_width, max_height = max_output_size
    scale = min(max_width / canvas.shape[1], max_height / canvas.shape[0], 1.0)
    if scale < 1.0:
        target_size = (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale))
        canvas = cv2.resize(canvas, target_size, interpolation=cv2.INTER_AREA)
    return canvas


def frame_range_for_group(frame_maps: dict[str, dict[int, dict[int, dict[str, Any]]]], group_tracks: dict[str, int]) -> tuple[int, int]:
    """Return the frame span where at least one selected group track exists."""

    frames = []
    for view_id, track_id in group_tracks.items():
        frame_map = frame_maps.get(view_id, {})
        frames.extend(frame_idx for frame_idx, people in frame_map.items() if track_id in people)
    if not frames:
        return (0, 0)
    return (min(frames), max(frames))


def render_group_video(
    group: dict[str, Any],
    view_ids: list[str],
    frame_maps: dict[str, dict[int, dict[int, dict[str, Any]]]],
    video_paths: dict[str, Path],
    output_path: Path,
    max_output_size: tuple[int, int],
    title_lines: list[str] | None = None,
    frame_step: int = 1,
) -> dict[str, Any]:
    """Render one matching group as a four-view review video."""

    group_id = int(group["group_id"])
    group_tracks = parse_group_nodes(group)
    start_frame, end_frame = frame_range_for_group(frame_maps, group_tracks)

    captures = {view_id: cv2.VideoCapture(str(video_paths[view_id])) for view_id in view_ids}
    if not all(cap.isOpened() for cap in captures.values()):
        raise RuntimeError("Failed to open one or more source videos.")

    frame_step = max(int(frame_step), 1)
    fps = float(next(iter(captures.values())).get(cv2.CAP_PROP_FPS) or 20.0) / frame_step

    sample_panels = []
    for view_id in view_ids:
        frame = read_frame(captures[view_id], start_frame)
        if frame is None:
            raise RuntimeError(f"Failed to read sample frame for {view_id}")
        people = frame_maps[view_id].get(start_frame, {})
        sample_panels.append(
            annotate_view(frame, people, view_id, group_tracks.get(view_id), group_id)
        )
    sample_grid = compose_grid(sample_panels, [f"Group {group_id}", "sample"], max_output_size)
    frame_h, frame_w = sample_grid.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output writer: {output_path}")

    try:
        for frame_idx in range(start_frame, end_frame + 1, frame_step):
            panels = []
            for view_id in view_ids:
                frame = read_frame(captures[view_id], frame_idx)
                if frame is None:
                    continue
                people = frame_maps[view_id].get(frame_idx, {})
                panels.append(
                    annotate_view(frame, people, view_id, group_tracks.get(view_id), group_id)
                )
            if len(panels) != len(view_ids):
                break
            header_lines = title_lines or [f"Stage 3B matching group {group_id}"]
            header_lines = [
                *header_lines,
                " | ".join(f"{view_id}:t{group_tracks.get(view_id, 'missing')}" for view_id in view_ids),
                f"frame {frame_idx} / {start_frame}-{end_frame}",
            ]
            grid = compose_grid(panels, header_lines, max_output_size)
            writer.write(grid)
    finally:
        writer.release()
        for cap in captures.values():
            cap.release()

    return {
        "group_id": group_id,
        "output_video": str(output_path),
        "nodes": group.get("nodes", []),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_step": frame_step,
        "frame_size": [frame_w, frame_h],
    }


def main() -> None:
    """Render all groups from a matching graph."""

    args = parse_args()
    if len(args.track_jsons) != len(args.view_ids) or len(args.source_videos) != len(args.view_ids):
        raise ValueError("track-jsons, source-videos, and view-ids must have the same length.")

    graph = load_json(args.matching_graph_json)
    track_payloads = {
        view_id: load_json(track_json)
        for view_id, track_json in zip(args.view_ids, args.track_jsons)
    }
    frame_maps = {
        view_id: build_people_index(track_payload)
        for view_id, track_payload in track_payloads.items()
    }
    video_paths = {
        view_id: Path(video_path).resolve()
        for view_id, video_path in zip(args.view_ids, args.source_videos)
    }

    output_dir = Path(args.output_dir).resolve()
    renders = []
    for group in graph.get("groups", []):
        group_id = int(group["group_id"])
        output_path = output_dir / f"matching_group_{group_id:02d}.mp4"
        renders.append(
            render_group_video(
                group=group,
                view_ids=list(args.view_ids),
                frame_maps=frame_maps,
                video_paths=video_paths,
                output_path=output_path,
                max_output_size=(int(args.max_output_width), int(args.max_output_height)),
                frame_step=int(args.frame_step),
            )
        )

    summary = {
        "stage": "visualize_matching_groups",
        "num_groups": len(renders),
        "renders": renders,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
