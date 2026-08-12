"""Synchronize the four demo cameras by explicit arm-motion peak events.

For each of the two opening and two closing calibration gestures at the start
and end of the recording, this script detects:

* the positive peak of normalized wrist-separation velocity (arms opening);
* the following negative peak (arms closing).

That gives eight ordered motion events per camera.  A robust affine mapping
``local_frame = a * reference_frame + b`` is then fitted from the event peaks,
so both startup offset and clock drift are measured instead of inferred from a
static arms-open plateau.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera_system.camera_calibration.charuco.calib_charuco_v2.diagnose_cam4_offset import (
    CAM_IDS,
    draw_pose,
    frame_kpts,
    normalized_arm_signal,
)


def build_signal(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(
        [normalized_arm_signal(frame_kpts(payload, i)) for i in range(len(payload["frames"]))],
        dtype=np.float64,
    )
    good = np.isfinite(raw)
    if int(good.sum()) < 20:
        raise RuntimeError("too few valid wrist/shoulder observations for motion sync")
    filled = np.interp(np.arange(len(raw)), np.flatnonzero(good), raw[good])
    window = min(11, len(filled) // 2 * 2 - 1)
    smooth = savgol_filter(filled, window, 2) if window >= 5 else filled
    derivative = np.gradient(smooth)
    return raw, np.asarray(smooth), np.asarray(derivative)


def quadratic_peak(values: np.ndarray, index: int) -> float:
    """Sub-frame location of a local extremum using a three-sample parabola."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    ym, y0, yp = [float(x) for x in values[index - 1:index + 2]]
    denom = ym - 2.0 * y0 + yp
    if abs(denom) < 1e-12:
        return float(index)
    delta = 0.5 * (ym - yp) / denom
    return float(index + np.clip(delta, -0.75, 0.75))


def detect_event_pair(
    derivative: np.ndarray,
    coarse_open: int,
    open_before: int = 25,
    open_after: int = 10,
    close_min_delay: int = 8,
    close_max_delay: int = 34,
) -> tuple[dict[str, Any], dict[str, Any]]:
    open_lo = max(1, int(coarse_open) - open_before)
    open_hi = min(len(derivative) - 1, int(coarse_open) + open_after + 1)
    if open_hi <= open_lo:
        raise RuntimeError(f"empty opening peak window around {coarse_open}")
    open_i = int(open_lo + np.argmax(derivative[open_lo:open_hi]))
    open_sub = quadratic_peak(derivative, open_i)

    close_lo = max(1, int(np.floor(open_sub)) + close_min_delay)
    close_hi = min(len(derivative) - 1, int(np.ceil(open_sub)) + close_max_delay + 1)
    if close_hi <= close_lo:
        raise RuntimeError(f"empty closing peak window after {open_sub:.2f}")
    close_i = int(close_lo + np.argmin(derivative[close_lo:close_hi]))
    close_sub = quadratic_peak(-derivative, close_i)

    local_scale = float(np.median(np.abs(derivative[max(0, open_lo - 20):min(len(derivative), close_hi + 20)])))
    local_scale = max(local_scale, 1e-6)
    opening = {
        "kind": "open_velocity_positive",
        "coarse_anchor_frame": int(coarse_open),
        "frame_integer": open_i,
        "frame_subpixel": float(open_sub),
        "velocity": float(derivative[open_i]),
        "strength_over_local_median": float(abs(derivative[open_i]) / local_scale),
        "search_window": [open_lo, open_hi - 1],
    }
    closing = {
        "kind": "close_velocity_negative",
        "coarse_anchor_frame": int(coarse_open),
        "frame_integer": close_i,
        "frame_subpixel": float(close_sub),
        "velocity": float(derivative[close_i]),
        "strength_over_local_median": float(abs(derivative[close_i]) / local_scale),
        "search_window": [close_lo, close_hi - 1],
    }
    return opening, closing


def detect_eight_events(
    derivative: np.ndarray,
    picked: dict[str, list[int]],
) -> list[dict[str, Any]]:
    events = []
    for segment in ("start", "end"):
        anchors = picked[segment]
        if len(anchors) != 2:
            raise RuntimeError(f"{segment} must contain two coarse gesture anchors")
        for repetition, anchor in enumerate(anchors, start=1):
            opening, closing = detect_event_pair(derivative, int(anchor))
            opening["event_name"] = f"{segment}_{repetition}_open"
            closing["event_name"] = f"{segment}_{repetition}_close"
            events.extend([opening, closing])
    frames = [float(e["frame_subpixel"]) for e in events]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise RuntimeError(f"detected peaks are not strictly ordered: {frames}")
    return events


def fit_affine(reference_peaks: np.ndarray, local_peaks: np.ndarray) -> dict[str, Any]:
    initial = np.polyfit(reference_peaks, local_peaks, 1)

    def residual(params: np.ndarray) -> np.ndarray:
        return params[0] * reference_peaks + params[1] - local_peaks

    result = least_squares(
        residual,
        np.asarray(initial, dtype=np.float64),
        loss="soft_l1",
        f_scale=0.75,
        max_nfev=1000,
    )
    a, b = [float(x) for x in result.x]
    predicted = a * reference_peaks + b
    errors = local_peaks - predicted
    return {
        "a": a,
        "b": b,
        "residual_frames_local_minus_predicted": errors.tolist(),
        "residual_rms_frames": float(np.sqrt(np.mean(errors ** 2))),
        "residual_median_abs_frames": float(np.median(np.abs(errors))),
        "residual_max_abs_frames": float(np.max(np.abs(errors))),
        "equiv_offset_at_first_event": float(a * reference_peaks[0] + b - reference_peaks[0]),
        "equiv_offset_at_last_event": float(a * reference_peaks[-1] + b - reference_peaks[-1]),
        "drift_across_events_frames": float((a - 1.0) * (reference_peaks[-1] - reference_peaks[0])),
        "optimizer_success": bool(result.success),
    }


def make_peak_plot(
    output: Path,
    signals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    events: dict[str, list[dict[str, Any]]],
    affine: dict[str, dict[str, Any]],
    reference_id: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    colors = {"open_velocity_positive": "#18a558", "close_velocity_negative": "#d33f49"}
    reference_peaks = [float(e["frame_subpixel"]) for e in events[reference_id]]
    for ax, cam_id in zip(axes, CAM_IDS):
        _, _, derivative = signals[cam_id]
        a, b = affine[cam_id]["a"], affine[cam_id]["b"]
        ref_axis = (np.arange(len(derivative), dtype=np.float64) - b) / a
        scale = max(float(np.percentile(np.abs(derivative), 97)), 1e-6)
        ax.plot(ref_axis, derivative / scale, color="#3b6ea8", linewidth=1.0, label="normalized wrist-distance velocity")
        for event in events[cam_id]:
            local = float(event["frame_subpixel"])
            ref_frame = (local - b) / a
            idx = int(event["frame_integer"])
            kind = str(event["kind"])
            ax.scatter([ref_frame], [derivative[idx] / scale], s=42, color=colors[kind], zorder=4)
        for ref_frame in reference_peaks:
            ax.axvline(ref_frame, color="#999999", linewidth=0.65, alpha=0.5)
        ax.set_ylabel(cam_id)
        ax.grid(alpha=0.2)
        ax.set_ylim(-1.35, 1.35)
        ax.set_title(
            f"{cam_id}: local = {a:.8f} * reference {b:+.3f}, "
            f"peak RMS={affine[cam_id]['residual_rms_frames']:.2f} frames",
            fontsize=10,
        )
    axes[-1].set_xlabel("cam1 reference frame")
    fig.suptitle("Explicit opening/closing motion peaks after affine alignment", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def read_frame(cap: cv2.VideoCapture, target: int, last_target: int | None) -> tuple[bool, np.ndarray | None]:
    if last_target is None or target != last_target + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
    return cap.read()


def draw_detection_and_pose(frame: np.ndarray, payload: dict[str, Any], frame_index: int) -> None:
    """Draw the selected YOLO person box and filtered HRNet COCO17 pose."""
    if frame_index < 0 or frame_index >= len(payload.get("frames", [])):
        return
    people = payload["frames"][frame_index].get("people", [])
    if people:
        bbox = people[0].get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 3, cv2.LINE_AA)
            cv2.putText(
                frame, "YOLO person + HRNet", (x1, max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 180, 255), 2, cv2.LINE_AA,
            )
    draw_pose(frame, frame_kpts(payload, frame_index))


def write_synced_review(
    output: Path,
    line_dir: Path,
    poses: dict[str, dict[str, Any]],
    affine: dict[str, dict[str, Any]],
    reference_start: int,
    reference_end: int,
) -> None:
    caps = {cam: cv2.VideoCapture(str(line_dir / f"{cam}.mkv")) for cam in CAM_IDS}
    last_targets: dict[str, int | None] = {cam: None for cam in CAM_IDS}
    fps = float(poses["cam1demo"]["metadata"].get("fps", 30.0))
    tile_w, tile_h = 640, 360
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tile_w * 2, tile_h * 2)
    )
    for reference_frame in range(reference_start, reference_end + 1):
        canvas = np.zeros((tile_h * 2, tile_w * 2, 3), dtype=np.uint8)
        for idx, cam_id in enumerate(CAM_IDS):
            a, b = affine[cam_id]["a"], affine[cam_id]["b"]
            target = int(round(a * reference_frame + b))
            ok, frame = read_frame(caps[cam_id], target, last_targets[cam_id])
            last_targets[cam_id] = target
            if not ok or frame is None:
                frame = np.zeros((
                    int(poses[cam_id]["metadata"].get("height", 720)),
                    int(poses[cam_id]["metadata"].get("width", 1280)),
                    3,
                ), dtype=np.uint8)
            draw_detection_and_pose(frame, poses[cam_id], target)
            cv2.putText(
                frame, f"{cam_id} src={target}", (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
            )
            tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            x, y = (idx % 2) * tile_w, (idx // 2) * tile_h
            canvas[y:y + tile_h, x:x + tile_w] = tile
        cv2.putText(
            canvas, f"motion-peak affine sync | reference frame={reference_frame}",
            (24, tile_h * 2 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.82,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()
    for cap in caps.values():
        cap.release()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Synchronize four camera videos using eight explicit arm-motion peaks.")
    ap.add_argument("--line-dir", default="camtest/line")
    ap.add_argument(
        "--pose-dir",
        default="camtest/line/demo_hrnet_enhanced_20260728/pose_filtered",
    )
    ap.add_argument(
        "--coarse-sync-report",
        default="camtest/line/demo_armspread_sync_triangulation/aligned_armspread_v1/sync_report.json",
        help="Only supplies broad gesture anchors; final alignment uses detected velocity peaks.",
    )
    ap.add_argument(
        "--output-dir",
        default="camtest/line/demo_hrnet_enhanced_20260728/motion_peak_sync",
    )
    ap.add_argument("--reference", default="cam1demo")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    line_dir = Path(args.line_dir)
    pose_dir = Path(args.pose_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse = json.loads(Path(args.coarse_sync_report).read_text(encoding="utf-8"))
    poses = {
        cam_id: json.loads((pose_dir / f"keypoints_{cam_id}_filtered.json").read_text(encoding="utf-8"))
        for cam_id in CAM_IDS
    }
    signals = {cam_id: build_signal(poses[cam_id]) for cam_id in CAM_IDS}
    events = {
        cam_id: detect_eight_events(signals[cam_id][2], coarse["picked_events"][cam_id])
        for cam_id in CAM_IDS
    }
    reference_peaks = np.asarray(
        [float(e["frame_subpixel"]) for e in events[args.reference]], dtype=np.float64
    )
    affine = {}
    for cam_id in CAM_IDS:
        local_peaks = np.asarray(
            [float(e["frame_subpixel"]) for e in events[cam_id]], dtype=np.float64
        )
        affine[cam_id] = fit_affine(reference_peaks, local_peaks)
        affine[cam_id]["events_local"] = local_peaks.tolist()
        affine[cam_id]["events_ref"] = reference_peaks.tolist()
        print(
            f"[PEAK-FIT] {cam_id}: a={affine[cam_id]['a']:.9f} "
            f"b={affine[cam_id]['b']:+.3f} "
            f"rms={affine[cam_id]['residual_rms_frames']:.3f} "
            f"max={affine[cam_id]['residual_max_abs_frames']:.3f}",
            flush=True,
        )

    start = int(np.floor(reference_peaks[0]))
    end = int(np.ceil(reference_peaks[-1]))
    report = {
        "mode": "explicit_8_motion_peak_affine_sync",
        "definition": "local_frame = a * reference_frame + b",
        "reference": args.reference,
        "reference_interval_frames": [start, end],
        "coarse_anchor_source": str(Path(args.coarse_sync_report).resolve()),
        "coarse_anchor_usage": "search windows only; affine fit uses detected velocity peaks",
        "event_order": [e["event_name"] for e in events[args.reference]],
        "detected_events": events,
        "affine_local_frame=a_ref_frame_plus_b": affine,
    }
    report_path = output_dir / "motion_peak_sync_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_peak_plot(
        output_dir / "motion_peak_alignment.png",
        signals, events, affine, args.reference,
    )
    write_synced_review(
        output_dir / "synced_2x2_motion_peak_affine.mp4",
        line_dir, poses, affine, max(0, start - 10), end + 10,
    )
    print(json.dumps({
        "report": str(report_path.resolve()),
        "plot": str((output_dir / "motion_peak_alignment.png").resolve()),
        "video": str((output_dir / "synced_2x2_motion_peak_affine.mp4").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
