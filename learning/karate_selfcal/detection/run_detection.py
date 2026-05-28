"""Entry point for running person detection and 2D pose estimation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .detectors import KEYPOINT_NAMES, build_detector, open_video_writer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the detection stage."""

    parser = argparse.ArgumentParser(
        description="Run multi-view person detection and 2D pose estimation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="learning/karate_selfcal/configs/detection.yaml",
        help="Path to the detection config YAML.",
    )
    parser.add_argument(
        "--input-videos",
        nargs="+",
        default=None,
        help="Input video paths. Overrides config videos when provided.",
    )
    parser.add_argument(
        "--view-ids",
        nargs="+",
        default=None,
        help="Optional per-video view ids. Must match input video count.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for detection outputs.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="YOLO pose model path override.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=None,
        help="Confidence threshold override.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap per video for quick tests.",
    )
    parser.add_argument(
        "--save-annotated-video",
        action="store_true",
        help="Export per-view annotated videos.",
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
    """Apply CLI overrides onto the detection config."""

    merged = dict(config)
    merged["detector"] = dict(config.get("detector", {}))
    merged["inputs"] = dict(config.get("inputs", {}))
    merged["outputs"] = dict(config.get("outputs", {}))

    if args.model_path:
        merged["detector"]["model_path"] = args.model_path
    if args.conf_thresh is not None:
        merged["detector"]["confidence_threshold"] = args.conf_thresh
    if args.max_frames is not None:
        merged["inputs"]["max_frames"] = args.max_frames
    if args.input_videos:
        merged["inputs"]["videos"] = list(args.input_videos)
    if args.view_ids:
        merged["inputs"]["view_ids"] = list(args.view_ids)
    if args.output_dir:
        merged["outputs"]["output_dir"] = args.output_dir
    if args.save_annotated_video:
        merged["outputs"]["save_annotated_video"] = True

    return merged


def resolve_inputs(config: dict[str, Any], repo_root: Path) -> tuple[list[Path], list[str]]:
    """Resolve input videos and view ids from config."""

    inputs_cfg = config.get("inputs", {})
    raw_videos = inputs_cfg.get("videos", [])
    if not raw_videos:
        raise ValueError("No input videos provided in config or CLI.")

    videos = []
    for raw in raw_videos:
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        videos.append(path.resolve())

    raw_view_ids = inputs_cfg.get("view_ids")
    if raw_view_ids:
        if len(raw_view_ids) != len(videos):
            raise ValueError("view_ids count must match input video count.")
        view_ids = [str(v) for v in raw_view_ids]
    else:
        view_ids = [video.stem for video in videos]

    return videos, view_ids


def make_frame_record(frame_index: int, people: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize a frame result for JSON export."""

    return {
        "frame": frame_index,
        "people": people,
    }


def maybe_export_heatmaps(
    view_id: str,
    frame_index: int,
    people: list[dict[str, Any]],
    output_dir: Path,
    save_heatmaps: bool,
) -> list[dict[str, Any]]:
    """Export per-person heatmaps when the backend provides them."""

    normalized_people = []
    heatmap_dir = output_dir / "heatmaps" / view_id

    for person in people:
        person_copy = dict(person)
        heatmaps = person_copy.pop("heatmaps", None)
        if save_heatmaps and heatmaps is not None:
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            person_id = int(person_copy.get("person_id", 0))
            heatmap_path = heatmap_dir / f"frame_{frame_index:06d}_person_{person_id:02d}.npz"
            np.savez_compressed(heatmap_path, heatmaps=heatmaps)
            person_copy["heatmap_path"] = str(heatmap_path)
        normalized_people.append(person_copy)

    return normalized_people


def run_detection_on_video(
    video_path: Path,
    view_id: str,
    detector_config: dict[str, Any],
    output_dir: Path,
    save_annotated_video: bool,
    save_heatmaps: bool,
    max_frames: int,
) -> dict[str, Any]:
    """Run pose detection for a single view and export outputs."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    detector = build_detector(detector_config, draw_annotations=save_annotated_video)
    writer = None
    if save_annotated_video:
        writer = open_video_writer(output_dir / f"{view_id}_annotated.mp4", fps, width, height)

    frames = []
    detected_frames = 0
    total_people = 0
    frame_index = 0

    try:
        while True:
            if max_frames > 0 and frame_index >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            result = detector.run(frame)
            people = maybe_export_heatmaps(
                view_id=view_id,
                frame_index=frame_index,
                people=result.people,
                output_dir=output_dir,
                save_heatmaps=save_heatmaps,
            )
            if people:
                detected_frames += 1
                total_people += len(people)

            frames.append(make_frame_record(frame_index, people))

            if writer is not None and result.annotated_frame is not None:
                writer.write(result.annotated_frame)

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    payload = {
        "metadata": {
            "view_id": view_id,
            "video_path": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": frame_index,
            "source_total_frames": total_frames_hint,
            "detected_frames": detected_frames,
            "total_people": total_people,
            "detector_backend": config_backend_name(detector_config),
            "supports_heatmaps": bool(save_heatmaps),
        },
        "keypoint_names": KEYPOINT_NAMES,
        "frames": frames,
    }

    json_path = output_dir / f"keypoints_{view_id}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return payload


def build_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Create a compact run summary for all views."""

    views = {}
    total_frames = 0
    total_detected_frames = 0
    total_people = 0

    for view_id, payload in results.items():
        meta = payload["metadata"]
        views[view_id] = {
            "video_path": meta["video_path"],
            "total_frames": meta["total_frames"],
            "detected_frames": meta["detected_frames"],
            "total_people": meta["total_people"],
            "fps": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "detector_backend": meta.get("detector_backend"),
            "supports_heatmaps": meta.get("supports_heatmaps", False),
        }
        total_frames += int(meta["total_frames"])
        total_detected_frames += int(meta["detected_frames"])
        total_people += int(meta["total_people"])

    return {
        "stage": "detection",
        "views": views,
        "aggregate": {
            "num_views": len(results),
            "total_frames_processed": total_frames,
            "total_detected_frames": total_detected_frames,
            "total_people_detections": total_people,
        },
    }


def config_backend_name(config: dict[str, Any]) -> str:
    """Return the configured detector backend name for metadata."""

    return str(config.get("detector", {}).get("backend", "yolo_pose"))


def main() -> None:
    """Run the detection stage and export per-view 2D pose outputs."""

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    config = merge_cli_overrides(load_yaml(args.config), args)

    videos, view_ids = resolve_inputs(config, repo_root)
    outputs_cfg = config.get("outputs", {})
    output_dir = outputs_cfg.get("output_dir", "outputs/karate_selfcal/detection")
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save_annotated_video = bool(outputs_cfg.get("save_annotated_video", False))
    save_heatmaps = bool(outputs_cfg.get("save_heatmaps", False))
    max_frames = int(config.get("inputs", {}).get("max_frames", 0) or 0)

    results = {}
    for video_path, view_id in zip(videos, view_ids):
        results[view_id] = run_detection_on_video(
            video_path=video_path,
            view_id=view_id,
            detector_config=config,
            output_dir=output_dir,
            save_annotated_video=save_annotated_video,
            save_heatmaps=save_heatmaps,
            max_frames=max_frames,
        )

    summary = build_summary(results)
    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
