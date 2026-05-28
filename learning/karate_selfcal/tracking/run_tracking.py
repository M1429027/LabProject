"""Entry point for generating single-view tracklets from 2D detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .trackers import build_tracker


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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the tracking stage."""

    parser = argparse.ArgumentParser(
        description="Run single-view tracking on per-frame 2D detections."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="learning/karate_selfcal/configs/tracking.yaml",
        help="Path to the tracking config YAML.",
    )
    parser.add_argument(
        "--input-jsons",
        nargs="+",
        default=None,
        help="Per-view detection JSON files. Overrides config when provided.",
    )
    parser.add_argument(
        "--source-videos",
        nargs="+",
        default=None,
        help="Optional source videos for tracked debug renders.",
    )
    parser.add_argument(
        "--view-ids",
        nargs="+",
        default=None,
        help="Optional per-view ids. Must match input JSON count.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for tracking outputs.",
    )
    parser.add_argument(
        "--save-tracked-video",
        action="store_true",
        help="Export tracked per-view debug videos.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def merge_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides onto the tracking config."""

    merged = dict(config)
    merged["tracker"] = dict(config.get("tracker", {}))
    merged["inputs"] = dict(config.get("inputs", {}))
    merged["outputs"] = dict(config.get("outputs", {}))

    if args.input_jsons:
        merged["inputs"]["detection_jsons"] = list(args.input_jsons)
    if args.source_videos:
        merged["inputs"]["source_videos"] = list(args.source_videos)
    if args.view_ids:
        merged["inputs"]["view_ids"] = list(args.view_ids)
    if args.output_dir:
        merged["outputs"]["output_dir"] = args.output_dir
    if args.save_tracked_video:
        merged["outputs"]["save_tracked_video"] = True

    return merged


def resolve_paths(config: dict[str, Any], repo_root: Path) -> tuple[list[Path], list[str], list[Path | None]]:
    """Resolve tracking inputs and optional source videos."""

    inputs_cfg = config.get("inputs", {})
    raw_jsons = inputs_cfg.get("detection_jsons", [])
    if not raw_jsons:
        raise ValueError("No detection JSON inputs provided in config or CLI.")

    json_paths = []
    for raw in raw_jsons:
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        json_paths.append(path.resolve())

    raw_view_ids = inputs_cfg.get("view_ids")
    if raw_view_ids:
        if len(raw_view_ids) != len(json_paths):
            raise ValueError("view_ids count must match detection_jsons count.")
        view_ids = [str(v) for v in raw_view_ids]
    else:
        view_ids = []
        for path in json_paths:
            stem = path.stem
            if stem.startswith("keypoints_"):
                stem = stem[len("keypoints_") :]
            view_ids.append(stem)

    raw_videos = inputs_cfg.get("source_videos", [])
    source_videos: list[Path | None] = []
    if raw_videos:
        if len(raw_videos) != len(json_paths):
            raise ValueError("source_videos count must match detection_jsons count.")
        for raw in raw_videos:
            path = Path(raw)
            if not path.is_absolute():
                path = repo_root / path
            source_videos.append(path.resolve())
    else:
        source_videos = [None for _ in json_paths]

    return json_paths, view_ids, source_videos


def load_detection_payload(path: Path) -> dict[str, Any]:
    """Load a detection JSON payload."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def open_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create a writer for tracked visualization output."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open tracking video writer: {output_path}")
    return writer


def draw_tracking_overlay(frame: Any, people: list[dict[str, Any]]) -> Any:
    """Draw simple tracked overlays for debug videos."""

    annotated = frame.copy()
    colors = [
        (0, 255, 0),
        (0, 180, 255),
        (255, 180, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for person in people:
        track_id = int(person["track_id"])
        color = colors[track_id % len(colors)]
        bbox = person.get("bbox")
        if bbox:
            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])
            score = float(person.get("tracking_score", 0.0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"track {track_id} ({score:.2f})",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        keypoints_by_id: dict[int, dict[str, Any]] = {}
        for keypoint in person.get("keypoints", []):
            if float(keypoint.get("confidence", 0.0)) <= 0.0:
                continue
            keypoints_by_id[int(keypoint["id"])] = keypoint
            x = int(keypoint["x"])
            y = int(keypoint["y"])
            cv2.circle(annotated, (x, y), 3, color, -1)

        for joint_a, joint_b in COCO_SKELETON_CONNECTIONS:
            point_a = keypoints_by_id.get(joint_a)
            point_b = keypoints_by_id.get(joint_b)
            if point_a is None or point_b is None:
                continue
            if min(float(point_a.get("confidence", 0.0)), float(point_b.get("confidence", 0.0))) <= 0.05:
                continue
            ax = int(point_a["x"])
            ay = int(point_a["y"])
            bx = int(point_b["x"])
            by = int(point_b["y"])
            cv2.line(annotated, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)

    return annotated


def build_track_summary(track_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate lightweight per-track statistics."""

    summary: dict[int, dict[str, Any]] = {}
    for frame in track_frames:
        frame_idx = int(frame["frame"])
        for person in frame.get("people", []):
            track_id = int(person["track_id"])
            if track_id not in summary:
                summary[track_id] = {
                    "track_id": track_id,
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    "num_detections": 0,
                }
            stats = summary[track_id]
            stats["start_frame"] = min(stats["start_frame"], frame_idx)
            stats["end_frame"] = max(stats["end_frame"], frame_idx)
            stats["num_detections"] += 1

    return [summary[key] for key in sorted(summary)]


def _people_by_frame_and_track(
    track_frames: list[dict[str, Any]],
) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    """Index tracked people by frame and track id."""

    frame_map: dict[int, dict[int, dict[str, Any]]] = {}
    track_map: dict[int, list[dict[str, Any]]] = {}
    for frame in track_frames:
        frame_idx = int(frame["frame"])
        frame_people: dict[int, dict[str, Any]] = {}
        for person in frame.get("people", []):
            track_id = int(person["track_id"])
            enriched = {"frame": frame_idx, "person": person}
            frame_people[track_id] = person
            track_map.setdefault(track_id, []).append(enriched)
        frame_map[frame_idx] = frame_people
    return frame_map, track_map


def _mean_keypoint_confidence(person: dict[str, Any]) -> float:
    """Return the mean keypoint confidence for one tracked person."""

    keypoints = person.get("keypoints", [])
    if not keypoints:
        return 0.0
    confs = [float(point.get("confidence", 0.0)) for point in keypoints]
    if not confs:
        return 0.0
    return float(np.mean(confs))


def _bbox_center_distance_score(a: dict[str, Any], b: dict[str, Any], width: int, height: int) -> float:
    """Score center proximity between two detections."""

    a_bbox = a.get("bbox")
    b_bbox = b.get("bbox")
    if not a_bbox or not b_bbox:
        return 0.0
    a_center = ((float(a_bbox["x1"]) + float(a_bbox["x2"])) * 0.5, (float(a_bbox["y1"]) + float(a_bbox["y2"])) * 0.5)
    b_center = ((float(b_bbox["x1"]) + float(b_bbox["x2"])) * 0.5, (float(b_bbox["y1"]) + float(b_bbox["y2"])) * 0.5)
    diag = max((width**2 + height**2) ** 0.5, 1.0)
    dist = ((a_center[0] - b_center[0]) ** 2 + (a_center[1] - b_center[1]) ** 2) ** 0.5
    return max(0.0, 1.0 - min(dist / diag, 1.0))


def _bbox_iou_local(a_bbox: dict[str, Any] | None, b_bbox: dict[str, Any] | None) -> float:
    """Compute IoU locally for merge scoring."""

    if not a_bbox or not b_bbox:
        return 0.0
    x1 = max(float(a_bbox["x1"]), float(b_bbox["x1"]))
    y1 = max(float(a_bbox["y1"]), float(b_bbox["y1"]))
    x2 = min(float(a_bbox["x2"]), float(b_bbox["x2"]))
    y2 = min(float(a_bbox["y2"]), float(b_bbox["y2"]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a_bbox["x2"]) - float(a_bbox["x1"])) * max(
        0.0, float(a_bbox["y2"]) - float(a_bbox["y1"])
    )
    area_b = max(0.0, float(b_bbox["x2"]) - float(b_bbox["x1"])) * max(
        0.0, float(b_bbox["y2"]) - float(b_bbox["y1"])
    )
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _pose_similarity_local(a: dict[str, Any], b: dict[str, Any], min_conf: float) -> float:
    """Compute pose similarity between two tracked detections."""

    a_points = {
        int(point["id"]): (float(point["x"]), float(point["y"]))
        for point in a.get("keypoints", [])
        if float(point.get("confidence", 0.0)) >= min_conf
    }
    b_points = {
        int(point["id"]): (float(point["x"]), float(point["y"]))
        for point in b.get("keypoints", [])
        if float(point.get("confidence", 0.0)) >= min_conf
    }
    common = sorted(set(a_points) & set(b_points))
    if not common:
        return 0.0
    distances = []
    a_bbox = a.get("bbox")
    b_bbox = b.get("bbox")
    scale_a = max(_bbox_iou_local(a_bbox, a_bbox), 1.0)  # dummy positive value
    if a_bbox and b_bbox:
        side_a = max(float(a_bbox["x2"]) - float(a_bbox["x1"]), float(a_bbox["y2"]) - float(a_bbox["y1"]), 1.0)
        side_b = max(float(b_bbox["x2"]) - float(b_bbox["x1"]), float(b_bbox["y2"]) - float(b_bbox["y1"]), 1.0)
        scale_a = max((side_a + side_b) * 0.5, 50.0)
    for point_id in common:
        ax, ay = a_points[point_id]
        bx, by = b_points[point_id]
        distances.append(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
    mean_dist = float(np.mean(distances))
    return max(0.0, 1.0 - min(mean_dist / scale_a, 1.0))


def _person_quality(person: dict[str, Any]) -> float:
    """Score one tracked detection for overlap conflict resolution."""

    bbox_conf = float(person.get("bbox", {}).get("confidence", 0.0))
    track_score = float(person.get("tracking_score", 0.0))
    keypoint_conf = _mean_keypoint_confidence(person)
    return track_score * 0.5 + keypoint_conf * 0.35 + bbox_conf * 0.15


def _find_parent(parent: dict[int, int], track_id: int) -> int:
    """Union-find parent lookup."""

    while parent[track_id] != track_id:
        parent[track_id] = parent[parent[track_id]]
        track_id = parent[track_id]
    return track_id


def _union_parent(parent: dict[int, int], a: int, b: int) -> None:
    """Union-find merge."""

    root_a = _find_parent(parent, a)
    root_b = _find_parent(parent, b)
    if root_a == root_b:
        return
    parent[max(root_a, root_b)] = min(root_a, root_b)


def merge_tracklets(
    track_frames: list[dict[str, Any]],
    track_summary: list[dict[str, Any]],
    metadata: dict[str, Any],
    tracker_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Merge fragmented tracklets using overlap and transition similarity."""

    if not track_summary or not bool(tracker_cfg.get("enable_tracklet_merge", True)):
        return track_frames, track_summary, {"merged_pairs": [], "num_groups": len(track_summary)}

    frame_width = int(metadata["width"])
    frame_height = int(metadata["height"])
    min_conf = float(tracker_cfg.get("min_keypoint_confidence", 0.1))
    overlap_threshold = float(tracker_cfg.get("merge_overlap_threshold", 0.78))
    transition_threshold = float(tracker_cfg.get("merge_gap_threshold", 0.72))
    max_gap = int(tracker_cfg.get("merge_max_gap", 20))
    min_overlap_frames = int(tracker_cfg.get("merge_min_overlap_frames", 3))

    _, track_map = _people_by_frame_and_track(track_frames)
    parent = {int(track["track_id"]): int(track["track_id"]) for track in track_summary}
    merged_pairs: list[dict[str, Any]] = []

    def build_frame_person_map(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        return {int(item["frame"]): item["person"] for item in items}

    for i, track_a in enumerate(track_summary):
        for track_b in track_summary[i + 1 :]:
            track_id_a = int(track_a["track_id"])
            track_id_b = int(track_b["track_id"])
            items_a = track_map.get(track_id_a, [])
            items_b = track_map.get(track_id_b, [])
            if not items_a or not items_b:
                continue

            map_a = build_frame_person_map(items_a)
            map_b = build_frame_person_map(items_b)
            overlap_frames = sorted(set(map_a) & set(map_b))

            overlap_score = None
            if len(overlap_frames) >= min_overlap_frames:
                scores = []
                for frame_idx in overlap_frames:
                    person_a = map_a[frame_idx]
                    person_b = map_b[frame_idx]
                    pose = _pose_similarity_local(person_a, person_b, min_conf=min_conf)
                    center = _bbox_center_distance_score(person_a, person_b, frame_width, frame_height)
                    iou = _bbox_iou_local(person_a.get("bbox"), person_b.get("bbox"))
                    scores.append(0.55 * pose + 0.25 * center + 0.20 * iou)
                overlap_score = float(np.mean(scores)) if scores else 0.0
                if overlap_score >= overlap_threshold:
                    _union_parent(parent, track_id_a, track_id_b)
                    merged_pairs.append(
                        {
                            "type": "overlap",
                            "track_a": track_id_a,
                            "track_b": track_id_b,
                            "score": overlap_score,
                            "overlap_frames": len(overlap_frames),
                        }
                    )
                    continue

            gap_ab = int(track_b["start_frame"]) - int(track_a["end_frame"])
            gap_ba = int(track_a["start_frame"]) - int(track_b["end_frame"])
            if 0 <= gap_ab <= max_gap:
                person_a = items_a[-1]["person"]
                person_b = items_b[0]["person"]
            elif 0 <= gap_ba <= max_gap:
                person_a = items_b[-1]["person"]
                person_b = items_a[0]["person"]
            else:
                continue

            pose = _pose_similarity_local(person_a, person_b, min_conf=min_conf)
            center = _bbox_center_distance_score(person_a, person_b, frame_width, frame_height)
            iou = _bbox_iou_local(person_a.get("bbox"), person_b.get("bbox"))
            transition_score = 0.55 * pose + 0.25 * center + 0.20 * iou
            if transition_score >= transition_threshold:
                _union_parent(parent, track_id_a, track_id_b)
                merged_pairs.append(
                    {
                        "type": "transition",
                        "track_a": track_id_a,
                        "track_b": track_id_b,
                        "score": transition_score,
                        "overlap_frames": len(overlap_frames),
                    }
                )

    groups: dict[int, set[int]] = {}
    for track in track_summary:
        root = _find_parent(parent, int(track["track_id"]))
        groups.setdefault(root, set()).add(int(track["track_id"]))

    if all(len(group) == 1 for group in groups.values()):
        return track_frames, track_summary, {"merged_pairs": merged_pairs, "num_groups": len(groups)}

    canonical_track_id = {}
    for root, members in groups.items():
        canonical_track_id[root] = min(members)

    merged_frames = []
    for frame in track_frames:
        frame_idx = int(frame["frame"])
        by_canonical: dict[int, dict[str, Any]] = {}
        for person in frame.get("people", []):
            original_track_id = int(person["track_id"])
            root = _find_parent(parent, original_track_id)
            canonical_id = canonical_track_id[root]
            person_copy = dict(person)
            person_copy["track_id"] = canonical_id
            person_copy["merged_from_track_id"] = original_track_id
            current = by_canonical.get(canonical_id)
            if current is None or _person_quality(person_copy) > _person_quality(current):
                by_canonical[canonical_id] = person_copy
        merged_frames.append(
            {
                "frame": frame_idx,
                "people": [by_canonical[key] for key in sorted(by_canonical)],
            }
        )

    merged_summary = build_track_summary(merged_frames)
    merge_info = {
        "merged_pairs": merged_pairs,
        "num_groups": len(groups),
        "groups": {str(root): sorted(members) for root, members in groups.items()},
    }
    return merged_frames, merged_summary, merge_info


def filter_short_tracks(
    track_frames: list[dict[str, Any]],
    track_summary: list[dict[str, Any]],
    min_track_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop tracks whose lifespan is shorter than the configured threshold."""

    if min_track_length <= 1:
        return track_frames, track_summary

    valid_track_ids = {
        int(track["track_id"])
        for track in track_summary
        if int(track.get("num_detections", 0)) >= min_track_length
    }

    filtered_frames = []
    for frame in track_frames:
        kept_people = [
            person for person in frame.get("people", []) if int(person["track_id"]) in valid_track_ids
        ]
        filtered_frames.append({"frame": int(frame["frame"]), "people": kept_people})

    filtered_summary = [
        track for track in track_summary if int(track["track_id"]) in valid_track_ids
    ]
    return filtered_frames, filtered_summary


def render_tracking_video(
    source_video: Path,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    track_frames: list[dict[str, Any]],
) -> None:
    """Render a tracked debug video after any post-filtering has been applied."""

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video for tracking overlay: {source_video}")

    writer = open_video_writer(output_path, fps, width, height)
    try:
        for frame_entry in track_frames:
            ok, frame = cap.read()
            if not ok:
                break
            annotated = draw_tracking_overlay(frame, frame_entry.get("people", []))
            writer.write(annotated)
    finally:
        cap.release()
        writer.release()


def run_tracking_on_view(
    detection_payload: dict[str, Any],
    view_id: str,
    config: dict[str, Any],
    output_dir: Path,
    source_video: Path | None,
    save_tracked_video: bool,
) -> dict[str, Any]:
    """Track detections within a single view."""

    metadata = detection_payload["metadata"]
    frame_width = int(metadata["width"])
    frame_height = int(metadata["height"])
    fps = float(metadata["fps"])
    frames = detection_payload.get("frames", [])

    tracker = build_tracker(config, frame_width=frame_width, frame_height=frame_height)
    tracked_frames = []
    for frame_entry in frames:
        frame_idx = int(frame_entry["frame"])
        detections = frame_entry.get("people", [])
        tracked_people = tracker.update(frame_idx, detections)
        tracked_frames.append(
            {
                "frame": frame_idx,
                "people": tracked_people,
            }
        )

    raw_track_summary = build_track_summary(tracked_frames)
    merge_info: dict[str, Any] = {"merged_pairs": [], "num_groups": len(raw_track_summary)}
    tracked_frames, merged_track_summary, merge_info = merge_tracklets(
        tracked_frames,
        raw_track_summary,
        metadata={
            "width": frame_width,
            "height": frame_height,
        },
        tracker_cfg=config.get("tracker", {}),
    )
    min_track_length = int(config.get("outputs", {}).get("min_track_length", 1))
    tracked_frames, track_summary = filter_short_tracks(
        tracked_frames,
        merged_track_summary,
        min_track_length=min_track_length,
    )

    if save_tracked_video:
        video_path = source_video
        if video_path is None:
            meta_video = metadata.get("video_path")
            if meta_video:
                video_path = Path(meta_video)
        if video_path is None:
            raise ValueError("save_tracked_video requires source_videos or metadata.video_path.")
        render_tracking_video(
            source_video=video_path,
            output_path=output_dir / f"tracked_{view_id}.mp4",
            fps=fps,
            width=frame_width,
            height=frame_height,
            track_frames=tracked_frames,
        )

    payload = {
        "metadata": {
            "view_id": view_id,
            "fps": fps,
            "width": frame_width,
            "height": frame_height,
            "total_frames": len(tracked_frames),
            "source_detection_backend": metadata.get("detector_backend"),
            "raw_num_tracks": len(raw_track_summary),
            "merged_num_tracks": len(merged_track_summary),
            "filtered_num_tracks": len(track_summary),
            "min_track_length": min_track_length,
        },
        "keypoint_names": detection_payload.get("keypoint_names", []),
        "frames": tracked_frames,
        "tracks": track_summary,
        "merge_info": merge_info,
    }

    output_path = output_dir / f"tracks_{view_id}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return payload


def build_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a compact run summary for all views."""

    views = {}
    total_tracks = 0
    total_frames = 0

    for view_id, payload in results.items():
        meta = payload["metadata"]
        num_tracks = len(payload.get("tracks", []))
        total_tracks += num_tracks
        total_frames += int(meta["total_frames"])
        views[view_id] = {
            "total_frames": meta["total_frames"],
            "num_tracks": num_tracks,
            "source_detection_backend": meta.get("source_detection_backend"),
        }

    return {
        "stage": "tracking",
        "views": views,
        "aggregate": {
            "num_views": len(results),
            "total_frames_processed": total_frames,
            "total_tracks": total_tracks,
        },
    }


def main() -> None:
    """Run the tracking stage."""

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    config = merge_cli_overrides(load_yaml(args.config), args)

    json_paths, view_ids, source_videos = resolve_paths(config, repo_root)
    outputs_cfg = config.get("outputs", {})
    output_dir = outputs_cfg.get("output_dir", "outputs/karate_selfcal/tracking")
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save_tracked_video = bool(outputs_cfg.get("save_tracked_video", False))
    results = {}
    for json_path, view_id, source_video in zip(json_paths, view_ids, source_videos):
        payload = load_detection_payload(json_path)
        results[view_id] = run_tracking_on_view(
            detection_payload=payload,
            view_id=view_id,
            config=config,
            output_dir=output_dir,
            source_video=source_video,
            save_tracked_video=save_tracked_video,
        )

    summary = build_summary(results)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
