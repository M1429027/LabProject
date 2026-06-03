"""Visualize Stage 3B cross-view candidate matches side-by-side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
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

HIGHLIGHT_COLOR = (80, 255, 80)
SECONDARY_COLOR = (80, 180, 255)
BACKGROUND_COLOR = (90, 90, 90)
TEXT_COLOR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for match visualization."""

    parser = argparse.ArgumentParser(
        description="Render side-by-side cross-view candidate match review videos."
    )
    parser.add_argument("--matches-json", required=True, help="Path to candidate_matches.json")
    parser.add_argument("--track-json-a", required=True, help="Track JSON for the left view")
    parser.add_argument("--track-json-b", required=True, help="Track JSON for the right view")
    parser.add_argument("--video-a", required=True, help="Source video for the left view")
    parser.add_argument("--video-b", required=True, help="Source video for the right view")
    parser.add_argument("--pair-key", default=None, help="Optional pair key to render")
    parser.add_argument("--top-k", type=int, default=3, help="How many ranked candidates to render")
    parser.add_argument(
        "--output-dir",
        default="outputs/karate_selfcal/match_review",
        help="Directory for rendered review videos",
    )
    parser.add_argument(
        "--max-output-width",
        type=int,
        default=1920,
        help="Maximum rendered video width. Large side-by-side frames can be hard to play.",
    )
    parser.add_argument(
        "--max-output-height",
        type=int,
        default=1080,
        help="Maximum rendered video height. The aspect ratio is preserved.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON payload from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_people_index(track_payload: dict[str, Any]) -> dict[int, dict[int, dict[str, Any]]]:
    """Index tracked people by frame and track id."""

    frame_map: dict[int, dict[int, dict[str, Any]]] = {}
    for frame in track_payload.get("frames", []):
        people = {}
        for person in frame.get("people", []):
            people[int(person["track_id"])] = person
        frame_map[int(frame["frame"])] = people
    return frame_map


def draw_person(
    image: np.ndarray,
    person: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
    dim: bool = False,
) -> np.ndarray:
    """Draw one tracked person with bbox and 2D skeleton."""

    annotated = image
    draw_color = tuple(int(value * 0.45) for value in color) if dim else color

    bbox = person.get("bbox")
    if bbox:
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), draw_color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            draw_color,
            2,
            cv2.LINE_AA,
        )

    keypoints_by_id = {}
    for keypoint in person.get("keypoints", []):
        if float(keypoint.get("confidence", 0.0)) <= 0.05:
            continue
        keypoints_by_id[int(keypoint["id"])] = keypoint
        x = int(keypoint["x"])
        y = int(keypoint["y"])
        cv2.circle(annotated, (x, y), 3, draw_color, -1)

    for joint_a, joint_b in COCO_SKELETON_CONNECTIONS:
        point_a = keypoints_by_id.get(joint_a)
        point_b = keypoints_by_id.get(joint_b)
        if point_a is None or point_b is None:
            continue
        ax = int(point_a["x"])
        ay = int(point_a["y"])
        bx = int(point_b["x"])
        by = int(point_b["y"])
        cv2.line(annotated, (ax, ay), (bx, by), draw_color, 2, cv2.LINE_AA)

    return annotated


def compose_panel(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    overlay_lines: list[str],
    max_output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Compose one side-by-side review frame with a text header."""

    if frame_a.shape[0] != frame_b.shape[0]:
        target_height = min(frame_a.shape[0], frame_b.shape[0])
        frame_a = cv2.resize(frame_a, (int(frame_a.shape[1] * target_height / frame_a.shape[0]), target_height))
        frame_b = cv2.resize(frame_b, (int(frame_b.shape[1] * target_height / frame_b.shape[0]), target_height))

    gap = 24
    header_h = 140
    canvas_h = header_h + frame_a.shape[0]
    canvas_w = frame_a.shape[1] + gap + frame_b.shape[1]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:header_h, :] = (22, 22, 22)
    canvas[header_h:, : frame_a.shape[1]] = frame_a
    canvas[header_h:, frame_a.shape[1] + gap :] = frame_b

    y = 28
    for line in overlay_lines:
        cv2.putText(
            canvas,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        y += 28

    if max_output_size is not None:
        max_width, max_height = max_output_size
        scale = min(max_width / canvas.shape[1], max_height / canvas.shape[0], 1.0)
        if scale < 1.0:
            target_size = (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale))
            canvas = cv2.resize(canvas, target_size, interpolation=cv2.INTER_AREA)

    return canvas


def render_candidate_video(
    candidate: dict[str, Any],
    frame_map_a: dict[int, dict[int, dict[str, Any]]],
    frame_map_b: dict[int, dict[int, dict[str, Any]]],
    video_a_path: Path,
    video_b_path: Path,
    output_path: Path,
    max_output_size: tuple[int, int],
) -> dict[str, Any]:
    """Render one candidate pair as a side-by-side debug video."""

    start_frame = int(candidate["overlap_frames"]["start"] or 0)
    end_frame = int(candidate["overlap_frames"]["end"] or start_frame)
    track_a = int(candidate["track_a"])
    track_b = int(candidate["track_b"])

    cap_a = cv2.VideoCapture(str(video_a_path))
    cap_b = cv2.VideoCapture(str(video_b_path))
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise RuntimeError("Failed to open source videos for match visualization.")

    fps = float(cap_a.get(cv2.CAP_PROP_FPS) or 20.0)

    # Build one sample panel before opening the writer so the encoded video size
    # matches the resized review frame exactly.
    cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_b.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok_a, sample_a = cap_a.read()
    ok_b, sample_b = cap_b.read()
    if not ok_a or not ok_b:
        raise RuntimeError("Failed to read the first overlapping frame for match visualization.")
    sample_panel = compose_panel(sample_a, sample_b, ["sample"], max_output_size=max_output_size)
    panel_height, panel_width = sample_panel.shape[:2]
    cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_b.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_width, panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output writer: {output_path}")

    try:
        for frame_idx in range(start_frame, end_frame + 1):
            cap_a.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap_b.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if not ok_a or not ok_b:
                break

            people_a = frame_map_a.get(frame_idx, {})
            people_b = frame_map_b.get(frame_idx, {})

            for other_track_id, person in sorted(people_a.items()):
                frame_a = draw_person(
                    frame_a,
                    person,
                    HIGHLIGHT_COLOR if other_track_id == track_a else BACKGROUND_COLOR,
                    f"A track {other_track_id}",
                    dim=(other_track_id != track_a),
                )
            for other_track_id, person in sorted(people_b.items()):
                frame_b = draw_person(
                    frame_b,
                    person,
                    SECONDARY_COLOR if other_track_id == track_b else BACKGROUND_COLOR,
                    f"B track {other_track_id}",
                    dim=(other_track_id != track_b),
                )

            scores = candidate["scores"]
            overlay_lines = [
                f"{candidate['view_a']} track {track_a}  <->  {candidate['view_b']} track {track_b}",
                f"final={candidate['final_score']:.3f} | temporal={scores['temporal']:.3f} | pose={scores['pose_shape']:.3f} | motion={scores['motion_prior']:.3f} | skeleton={scores['skeleton_consistency']:.3f}",
                f"appearance={scores['appearance']:.3f} | overlap={candidate['overlap_frames']['count']} frames | frame={frame_idx}",
                f"frontality A={candidate['summary']['track_a_frontality']} | frontality B={candidate['summary']['track_b_frontality']}",
            ]
            panel = compose_panel(frame_a, frame_b, overlay_lines, max_output_size=max_output_size)
            writer.write(panel)
    finally:
        cap_a.release()
        cap_b.release()
        writer.release()

    return {
        "output_video": str(output_path),
        "track_a": track_a,
        "track_b": track_b,
        "final_score": float(candidate["final_score"]),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_size": [panel_width, panel_height],
    }


def main() -> None:
    """Render Stage 3B candidate-match review videos."""

    args = parse_args()
    matches_payload = load_json(args.matches_json)
    track_payload_a = load_json(args.track_json_a)
    track_payload_b = load_json(args.track_json_b)
    frame_map_a = build_people_index(track_payload_a)
    frame_map_b = build_people_index(track_payload_b)

    pair_key = args.pair_key
    if pair_key is None:
        pair_key = next(iter(matches_payload))
    pair_payload = matches_payload[pair_key]
    candidates = pair_payload.get("candidates", [])[: max(int(args.top_k), 0)]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    renders = []
    for rank, candidate in enumerate(candidates, start=1):
        output_path = output_dir / f"{pair_key}_rank{rank:02d}_a{candidate['track_a']}_b{candidate['track_b']}.mp4"
        renders.append(
            render_candidate_video(
                candidate=candidate,
                frame_map_a=frame_map_a,
                frame_map_b=frame_map_b,
                video_a_path=Path(args.video_a).resolve(),
                video_b_path=Path(args.video_b).resolve(),
                output_path=output_path,
                max_output_size=(int(args.max_output_width), int(args.max_output_height)),
            )
        )

    summary = {
        "stage": "visualize_cross_view_matches",
        "pair_key": pair_key,
        "num_rendered_candidates": len(renders),
        "renders": renders,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
