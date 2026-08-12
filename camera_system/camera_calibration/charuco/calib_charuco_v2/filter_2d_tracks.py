"""Temporal filtering for single-person 2D COCO keypoint tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter


def bbox_xyxy(person: dict[str, Any] | None) -> np.ndarray:
    if not person:
        return np.full(4, np.nan, dtype=np.float64)
    bbox = person.get("bbox")
    if isinstance(bbox, dict):
        return np.asarray([bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")], dtype=np.float64)
    if isinstance(bbox, list) and len(bbox) >= 4:
        return np.asarray(bbox[:4], dtype=np.float64)
    return np.full(4, np.nan, dtype=np.float64)


def bbox_area(box: np.ndarray) -> float:
    if not np.isfinite(box).all():
        return 0.0
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def select_main_person(frame: dict[str, Any], prev_center: np.ndarray | None = None) -> dict[str, Any] | None:
    people = frame.get("people", [])
    if not people:
        return None

    def score(person: dict[str, Any]) -> float:
        box = bbox_xyxy(person)
        area = bbox_area(box)
        conf_sum = sum(float(k.get("confidence", 0.0)) for k in person.get("keypoints", []))
        score_value = area + 1000.0 * conf_sum
        if prev_center is not None and np.isfinite(box).all():
            center = 0.5 * (box[:2] + box[2:])
            score_value -= 0.55 * float(np.linalg.norm(center - prev_center))
        return score_value

    return max(people, key=score)


def person_to_arrays(person: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    kps = np.full((17, 3), np.nan, dtype=np.float64)
    box = bbox_xyxy(person)
    if person:
        for kp in person.get("keypoints", []):
            idx = int(kp.get("id", -1))
            if 0 <= idx < 17:
                kps[idx] = [float(kp.get("x", np.nan)), float(kp.get("y", np.nan)), float(kp.get("confidence", 0.0))]
    return kps, box


def interpolate_and_smooth(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    # Interpolate only inside the interval where this track was genuinely
    # observed.  np.interp extrapolates its edge values by default, which can
    # otherwise create a person before entry or keep one alive after a camera
    # has lost sight of them.
    out = np.full_like(values, np.nan, dtype=np.float64)
    if valid.sum() == 0:
        return out
    if valid.sum() == 1:
        out[valid] = values[valid]
        return out
    observed = np.flatnonzero(valid)
    lo, hi = int(observed[0]), int(observed[-1])
    segment_x = np.arange(lo, hi + 1)
    out[lo : hi + 1] = np.interp(segment_x, observed, values[valid])
    segment_n = hi - lo + 1
    win = min(int(window), segment_n if segment_n % 2 == 1 else segment_n - 1)
    if win >= 5:
        out[lo : hi + 1] = savgol_filter(out[lo : hi + 1], win, 2)
    return out


def filter_payload(payload: dict[str, Any], min_conf: float, max_joint_jump_px: float, smooth_window: int) -> tuple[dict[str, Any], dict[str, Any]]:
    frames = payload.get("frames", [])
    n = len(frames)
    kps = np.full((n, 17, 3), np.nan, dtype=np.float64)
    boxes = np.full((n, 4), np.nan, dtype=np.float64)
    prev_center = None
    selected_missing = 0
    for i, frame in enumerate(frames):
        person = select_main_person(frame, prev_center)
        if person is None:
            selected_missing += 1
        k, b = person_to_arrays(person)
        kps[i] = k
        boxes[i] = b
        if np.isfinite(b).all():
            prev_center = 0.5 * (b[:2] + b[2:])

    # Reject impossible 2D jumps before smoothing.
    reject_count = 0
    for j in range(17):
        for i in range(1, n):
            if not (np.isfinite(kps[i, j, 0]) and np.isfinite(kps[i - 1, j, 0])):
                continue
            if kps[i, j, 2] < min_conf or kps[i - 1, j, 2] < min_conf:
                continue
            jump = float(np.linalg.norm(kps[i, j, :2] - kps[i - 1, j, :2]))
            if jump > max_joint_jump_px:
                kps[i, j, 2] = min(kps[i, j, 2], 0.05)
                reject_count += 1

    # Smooth bbox.
    filtered_boxes = boxes.copy()
    for c in range(4):
        valid = np.isfinite(boxes[:, c])
        filtered_boxes[:, c] = interpolate_and_smooth(boxes[:, c], valid, smooth_window)

    # Smooth keypoints.
    filtered_kps = kps.copy()
    interpolated = 0
    for j in range(17):
        valid = np.isfinite(kps[:, j, 0]) & (kps[:, j, 2] >= min_conf)
        for c in range(2):
            filtered_kps[:, j, c] = interpolate_and_smooth(kps[:, j, c], valid, smooth_window)
        if valid.sum() >= 2:
            interpolated += int((~valid).sum())
        conf = kps[:, j, 2].copy()
        conf_valid = np.isfinite(conf)
        conf[~conf_valid] = 0.0
        # Keep low-confidence gaps visible as lower confidence after interpolation.
        filtered_kps[:, j, 2] = np.clip(interpolate_and_smooth(conf, conf_valid, smooth_window), 0.0, 1.0)
        filtered_kps[~valid, j, 2] *= 0.55

    out_frames = []
    for i in range(n):
        keypoints = [
            {
                "id": int(j),
                "x": float(filtered_kps[i, j, 0]),
                "y": float(filtered_kps[i, j, 1]),
                "confidence": float(filtered_kps[i, j, 2]),
                "raw_confidence": None if not np.isfinite(kps[i, j, 2]) else float(kps[i, j, 2]),
            }
            for j in range(17)
            if np.isfinite(filtered_kps[i, j, 0]) and np.isfinite(filtered_kps[i, j, 1])
        ]
        b = filtered_boxes[i]
        person = {"person_id": 0, "keypoints": keypoints}
        if np.isfinite(b).all():
            person["bbox"] = [float(v) for v in b]
            person["bbox_area"] = bbox_area(b)
        source_frame = dict(frames[i])
        source_frame["people"] = [person] if keypoints else []
        out_frames.append(source_frame)

    meta = dict(payload.get("metadata", {}))
    meta.update(
        {
            "temporal_filter": {
                "min_confidence_for_anchor": float(min_conf),
                "max_joint_jump_px": float(max_joint_jump_px),
                "smooth_window": int(smooth_window),
                "selected_missing_frames": int(selected_missing),
                "rejected_joint_jumps": int(reject_count),
            }
        }
    )
    out_payload = {
        "metadata": meta,
        "keypoint_names": payload.get("keypoint_names", []),
        "frames": out_frames,
    }
    metrics = {
        "num_frames": int(n),
        "selected_missing_frames": int(selected_missing),
        "rejected_joint_jumps": int(reject_count),
        "interpolated_joint_slots_estimate": int(interpolated),
        "median_raw_bbox_area": float(np.nanmedian([bbox_area(b) for b in boxes])) if n else 0.0,
        "median_filtered_bbox_area": float(np.nanmedian([bbox_area(b) for b in filtered_boxes])) if n else 0.0,
    }
    return out_payload, metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply temporal bbox/keypoint filtering to demo 2D pose JSONs.")
    ap.add_argument("--input-jsons", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-conf", type=float, default=0.25)
    ap.add_argument("--max-joint-jump-px", type=float, default=170.0)
    ap.add_argument("--smooth-window", type=int, default=9)
    ap.add_argument("--suffix", default="_filtered")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for src in args.input_jsons:
        src_path = Path(src)
        payload = json.loads(src_path.read_text(encoding="utf-8"))
        out_payload, metrics = filter_payload(
            payload,
            min_conf=float(args.min_conf),
            max_joint_jump_px=float(args.max_joint_jump_px),
            smooth_window=int(args.smooth_window),
        )
        name = src_path.stem + args.suffix + ".json"
        out_path = out_dir / name
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics_path = out_dir / (src_path.stem + args.suffix + "_metrics.json")
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[src_path.name] = {"output": str(out_path.resolve()), "metrics": metrics}
    print(json.dumps({"stage": "filter_2d_tracks", "outputs": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
