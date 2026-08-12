"""Run YOLO person boxes + HRNet top-down COCO pose on demo videos.

The output schema intentionally matches the JSON produced by
run_demo_pose_sync_triangulation.py so existing sync/triangulation code can
consume it with minimal glue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.karate_selfcal.detection.detectors import KEYPOINT_NAMES, build_detector


def normalize_bbox_for_demo(person: dict[str, Any]) -> dict[str, Any]:
    bbox = person.get("bbox")
    if isinstance(bbox, dict):
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])
        person["bbox"] = [x1, y1, x2, y2]
        person["bbox_score"] = float(bbox.get("confidence", person.get("bbox_score", 1.0)))
        person["bbox_area"] = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
    elif isinstance(bbox, list) and len(bbox) >= 4:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        person["bbox"] = [x1, y1, x2, y2]
        person["bbox_area"] = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
    else:
        xs = [float(k["x"]) for k in person.get("keypoints", []) if float(k.get("confidence", 0.0)) > 0.05]
        ys = [float(k["y"]) for k in person.get("keypoints", []) if float(k.get("confidence", 0.0)) > 0.05]
        if xs and ys:
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            person["bbox"] = [x1, y1, x2, y2]
            person["bbox_area"] = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
    # Keep the demo JSON lightweight/serializable; heatmaps are large ndarrays.
    person.pop("heatmaps", None)
    return person


def draw_light_pose(frame: np.ndarray, people: list[dict[str, Any]], conf: float = 0.2) -> np.ndarray:
    out = frame.copy()
    bones = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    for person in people:
        bbox = person.get("bbox")
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 255), 2)
        pts = {}
        for kp in person.get("keypoints", []):
            if float(kp.get("confidence", 0.0)) >= conf:
                pts[int(kp["id"])] = (int(kp["x"]), int(kp["y"]))
        for a, b in bones:
            if a in pts and b in pts:
                cv2.line(out, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
        for j, p in pts.items():
            color = (0, 255, 255) if j in (9, 10, 15, 16) else (255, 0, 255)
            cv2.circle(out, p, 3, color, -1)
    return out


def process_video(
    video_path: Path,
    detector: Any,
    output_dir: Path,
    prefix: str,
    save_annotated: bool,
    max_frames: int,
    batch_frames: int = 1,
    detection_batch_size: int = 16,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"keypoints_{prefix}.json"
    annotated_path = output_dir / f"hrnet_{prefix}.mp4"
    if json_path.exists():
        print(f"[HRNET] reuse {json_path}")
        return json.loads(json_path.read_text(encoding="utf-8"))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = None
    if save_annotated:
        writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open writer: {annotated_path}")

    frames = []
    frame_id = 0
    print(f"[HRNET] {video_path.name}: {frame_count} frames, {fps:.2f} fps, {width}x{height}")
    effective_batch = max(1, int(batch_frames))
    batch_api = getattr(detector, "run_batch", None)
    while True:
        frame_batch = []
        while len(frame_batch) < effective_batch:
            if max_frames > 0 and frame_id + len(frame_batch) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame_batch.append(frame)
        if not frame_batch:
            break
        if effective_batch > 1 and callable(batch_api):
            results = batch_api(
                frame_batch, detection_batch_size=int(detection_batch_size)
            )
        else:
            results = [detector.run(frame) for frame in frame_batch]
        if len(results) != len(frame_batch):
            raise RuntimeError(
                f"Detector returned {len(results)} results for "
                f"{len(frame_batch)} frames"
            )
        for frame, result in zip(frame_batch, results):
            people = [normalize_bbox_for_demo(dict(p)) for p in result.people]
            frames.append({"frame": int(frame_id), "people": people})
            if writer is not None:
                writer.write(draw_light_pose(frame, people))
            if frame_id % 30 == 0:
                print(f"  frame {frame_id}/{frame_count}", end="\r", flush=True)
            frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()
    payload = {
        "metadata": {
            "source_video": str(video_path.resolve()),
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": frame_id,
            "backend": "yolo_hrnet_topdown",
            "batch_frames": effective_batch,
            "detection_batch_size": int(detection_batch_size),
            "keypoint_layout": "coco17",
        },
        "keypoint_names": KEYPOINT_NAMES,
        "frames": frames,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[HRNET] wrote {json_path}")
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run HRNet top-down pose on camtest line demo videos.")
    ap.add_argument("--line-dir", default="camtest/line")
    ap.add_argument("--output-dir", default="camtest/line/demo_hrnet_pose/pose")
    ap.add_argument("--videos", nargs="+", default=["cam1demo.mkv", "cam2demo.mkv", "cam3demo.mkv", "cam4demo.mkv"])
    ap.add_argument("--prefixes", nargs="+", default=["cam1demo", "cam2demo", "cam3demo", "cam4demo"])
    ap.add_argument("--config", default="learning/karate_selfcal/configs/detection_hrnet_w32.yaml")
    ap.add_argument("--save-annotated", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--batch-frames", type=int, default=1)
    ap.add_argument("--detection-batch-size", type=int, default=16)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.videos) != len(args.prefixes):
        raise ValueError("--videos and --prefixes must have the same length")
    cfg = json.loads(json.dumps(__import__("yaml").safe_load(Path(args.config).read_text(encoding="utf-8"))))
    detector = build_detector(cfg, draw_annotations=False)
    line_dir = Path(args.line_dir)
    output_dir = Path(args.output_dir)
    outputs = {}
    for video_name, prefix in zip(args.videos, args.prefixes):
        payload = process_video(
            line_dir / video_name,
            detector=detector,
            output_dir=output_dir,
            prefix=prefix,
            save_annotated=bool(args.save_annotated),
            max_frames=int(args.max_frames),
            batch_frames=int(args.batch_frames),
            detection_batch_size=int(args.detection_batch_size),
        )
        outputs[prefix] = {
            "json": str((output_dir / f"keypoints_{prefix}.json").resolve()),
            "frames": len(payload["frames"]),
        }
    print(json.dumps({"stage": "demo_hrnet_pose", "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
