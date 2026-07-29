"""Refine floor-line extrinsics from tape centerline evidence.

This tool starts from the manually-picked floor-point PnP solution, searches for
bright/edge tape centers near the projected court/grid lines, then optimizes the
world-to-camera pose.  It is intentionally local: detections are only accepted
near the current projected line model, so the hand-picked points remain the
anchor and random floor texture is less likely to take over.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray


def load_npz_camera(intrinsics_path: Path, extrinsics_path: Path) -> CameraModel:
    intr = np.load(str(intrinsics_path), allow_pickle=True)
    ext = np.load(str(extrinsics_path), allow_pickle=True)

    K = None
    for key in ("camera_matrix", "mtx", "K", "intrinsic_matrix"):
        if key in intr:
            K = np.asarray(intr[key], dtype=np.float64)
            break
    if K is None:
        raise KeyError(f"Cannot find camera matrix in {intrinsics_path}")

    dist = None
    for key in ("dist_coeffs", "dist", "distortion_coefficients", "d"):
        if key in intr:
            dist = np.asarray(intr[key], dtype=np.float64).reshape(-1, 1)
            break
    if dist is None:
        dist = np.zeros((5, 1), dtype=np.float64)

    if "R" in ext:
        R = np.asarray(ext["R"], dtype=np.float64)
    elif "rotation_matrix" in ext:
        R = np.asarray(ext["rotation_matrix"], dtype=np.float64)
    elif "rvec" in ext:
        R, _ = cv2.Rodrigues(np.asarray(ext["rvec"], dtype=np.float64).reshape(3))
    else:
        raise KeyError(f"Cannot find rotation in {extrinsics_path}")

    if "tvec" in ext:
        t = np.asarray(ext["tvec"], dtype=np.float64).reshape(3, 1)
    elif "t" in ext:
        t = np.asarray(ext["t"], dtype=np.float64).reshape(3, 1)
    elif "translation_vector" in ext:
        t = np.asarray(ext["translation_vector"], dtype=np.float64).reshape(3, 1)
    else:
        raise KeyError(f"Cannot find translation in {extrinsics_path}")

    return CameraModel(K=K, dist=dist, R=R, t=t)


def load_frame_from_pick(points_path: Path, line_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(points_path.read_text(encoding="utf-8"))
    meta = payload.get("metadata", {})
    video_path = Path(str(meta.get("video_path", "")))
    if not video_path.is_absolute():
        video_path = line_dir / video_path.name if not (line_dir / video_path).exists() else line_dir / video_path
    frame_index = int(meta.get("frame_index", 0))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open pick source video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
    return frame, payload


def project_points(cam: CameraModel, xyz: np.ndarray, rvec: np.ndarray | None = None, tvec: np.ndarray | None = None) -> np.ndarray:
    if rvec is None:
        rvec, _ = cv2.Rodrigues(cam.R)
    if tvec is None:
        tvec = cam.t
    uv, _ = cv2.projectPoints(np.asarray(xyz, dtype=np.float64).reshape(-1, 1, 3), rvec, tvec, cam.K, cam.dist)
    return uv.reshape(-1, 2)


def used_world_points(report_path: Path) -> np.ndarray:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pts = [p["world_xyz_m"] for p in report.get("used_points", [])]
    if len(pts) < 4:
        raise ValueError(f"Cannot find used_points in {report_path}")
    return np.asarray(pts, dtype=np.float64)


def make_grid_line_samples(world_points: np.ndarray, samples_per_line: int = 80) -> tuple[np.ndarray, list[tuple[int, int]]]:
    xs = sorted({round(float(v), 6) for v in world_points[:, 0]})
    ys = sorted({round(float(v), 6) for v in world_points[:, 1]})
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    samples: list[list[float]] = []
    line_ranges: list[tuple[int, int]] = []

    for x in xs:
        start = len(samples)
        for y in np.linspace(min_y, max_y, samples_per_line):
            samples.append([float(x), float(y), 0.0])
        line_ranges.append((start, len(samples)))

    for y in ys:
        start = len(samples)
        for x in np.linspace(min_x, max_x, samples_per_line):
            samples.append([float(x), float(y), 0.0])
        line_ranges.append((start, len(samples)))

    return np.asarray(samples, dtype=np.float64), line_ranges


def response_image(frame: np.ndarray) -> np.ndarray:
    """Return normalized tape-likelihood image.

    White/yellow tape tends to be bright and/or saturated.  We combine local
    brightness, saturation, and Canny-like gradient magnitude.  This is not a
    global detector; it is only sampled near projected line hypotheses.
    """

    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    val = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    # Robust per-frame normalization.
    def norm(x: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(x, [15, 98])
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (0.50 * norm(val) + 0.25 * norm(sat) + 0.25 * norm(grad)).astype(np.float32)


def bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = x - x0
    dy = y - y0
    return (
        img[y0, x0] * (1 - dx) * (1 - dy)
        + img[y0, x1] * dx * (1 - dy)
        + img[y1, x0] * (1 - dx) * dy
        + img[y1, x1] * dx * dy
    )


def find_centerline_observations(
    frame: np.ndarray,
    cam: CameraModel,
    world_samples: np.ndarray,
    line_ranges: list[tuple[int, int]],
    search_radius_px: int,
    min_response: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resp = response_image(frame)
    h, w = resp.shape[:2]
    uv = project_points(cam, world_samples)
    accepted_world: list[np.ndarray] = []
    accepted_image: list[np.ndarray] = []
    accepted_score: list[float] = []

    for start, end in line_ranges:
        line_uv = uv[start:end]
        for local_idx in range(1, len(line_uv) - 1):
            idx = start + local_idx
            p = line_uv[local_idx]
            if not (8 <= p[0] < w - 8 and 8 <= p[1] < h - 8):
                continue
            tangent = line_uv[local_idx + 1] - line_uv[local_idx - 1]
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-6:
                continue
            tangent /= norm
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            offsets = np.arange(-search_radius_px, search_radius_px + 1, dtype=np.float64)
            candidates = p[None, :] + offsets[:, None] * normal[None, :]
            inside = (
                (candidates[:, 0] >= 1)
                & (candidates[:, 0] < w - 2)
                & (candidates[:, 1] >= 1)
                & (candidates[:, 1] < h - 2)
            )
            if not np.any(inside):
                continue
            scores = np.full(len(offsets), -np.inf, dtype=np.float64)
            scores[inside] = bilinear(resp, candidates[inside])
            best = int(np.argmax(scores))
            score = float(scores[best])
            # Require a local peak that is meaningfully better than the ends of the search band.
            edge_score = float(max(scores[0], scores[-1]))
            if score < min_response or score < edge_score + 0.08:
                continue
            accepted_world.append(world_samples[idx])
            accepted_image.append(candidates[best])
            accepted_score.append(score)

    if not accepted_world:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    return (
        np.asarray(accepted_world, dtype=np.float64),
        np.asarray(accepted_image, dtype=np.float64),
        np.asarray(accepted_score, dtype=np.float64),
    )


def reprojection_errors(cam: CameraModel, world: np.ndarray, image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    uv = project_points(cam, world, rvec, tvec.reshape(3, 1))
    return np.linalg.norm(uv - image, axis=1)


def optimize_pose(cam: CameraModel, world: np.ndarray, image: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rvec0, _ = cv2.Rodrigues(cam.R)
    x0 = np.r_[rvec0.reshape(3), cam.t.reshape(3)]
    w = np.sqrt(np.clip(weights, 0.05, 1.0))

    def residual(x: np.ndarray) -> np.ndarray:
        rvec = x[:3].reshape(3, 1)
        tvec = x[3:].reshape(3, 1)
        uv = project_points(cam, world, rvec, tvec)
        return ((uv - image) * w[:, None]).reshape(-1)

    before = reprojection_errors(cam, world, image, rvec0, cam.t)
    res = least_squares(residual, x0, loss="soft_l1", f_scale=8.0, max_nfev=120)
    rvec = res.x[:3].reshape(3, 1)
    tvec = res.x[3:].reshape(3, 1)
    after = reprojection_errors(cam, world, image, rvec, tvec)
    metrics = {
        "line_fit_before_median_px": float(np.median(before)),
        "line_fit_before_p90_px": float(np.percentile(before, 90)),
        "line_fit_after_median_px": float(np.median(after)),
        "line_fit_after_p90_px": float(np.percentile(after, 90)),
        "num_line_observations": int(len(world)),
        "optimizer_cost": float(res.cost),
        "optimizer_success": bool(res.success),
    }
    return rvec, tvec, metrics


def draw_axes(frame: np.ndarray, cam: CameraModel, rvec: np.ndarray, tvec: np.ndarray, length_m: float = 0.75) -> np.ndarray:
    out = frame.copy()
    axes = np.asarray(
        [
            [0, 0, 0],
            [length_m, 0, 0],
            [0, length_m, 0],
            [0, 0, length_m],
        ],
        dtype=np.float64,
    )
    pts = project_points(cam, axes, rvec, tvec).astype(int)
    origin = tuple(pts[0])
    cv2.line(out, origin, tuple(pts[1]), (0, 0, 255), 4, cv2.LINE_AA)
    cv2.line(out, origin, tuple(pts[2]), (0, 255, 0), 4, cv2.LINE_AA)
    cv2.line(out, origin, tuple(pts[3]), (255, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, "X", tuple(pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(out, "Y", tuple(pts[2]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 0), 3, cv2.LINE_AA)
    cv2.putText(out, "Z", tuple(pts[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3, cv2.LINE_AA)
    return out


def draw_line_debug(frame: np.ndarray, cam: CameraModel, world_samples: np.ndarray, detected_world: np.ndarray, detected_image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    out = frame.copy()
    uv0 = project_points(cam, world_samples)
    for p in uv0[::6].astype(int):
        if 0 <= p[0] < out.shape[1] and 0 <= p[1] < out.shape[0]:
            cv2.circle(out, tuple(p), 1, (180, 180, 180), -1)
    uv1 = project_points(cam, detected_world, rvec, tvec)
    for obs, proj in zip(detected_image.astype(int), uv1.astype(int)):
        if 0 <= obs[0] < out.shape[1] and 0 <= obs[1] < out.shape[0]:
            cv2.circle(out, tuple(obs), 2, (0, 255, 255), -1)
        if 0 <= proj[0] < out.shape[1] and 0 <= proj[1] < out.shape[0]:
            cv2.circle(out, tuple(proj), 2, (255, 0, 255), -1)
    return out


def camera_center(R: np.ndarray, t: np.ndarray) -> list[float]:
    return (-R.T @ t.reshape(3, 1)).reshape(3).astype(float).tolist()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refine floor-line extrinsics and draw RGB world axes.")
    ap.add_argument("--line-dir", default="camtest/line")
    ap.add_argument("--camera", required=True, choices=["cam1", "cam2", "cam3", "cam4"])
    ap.add_argument("--points-json", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--extrinsics", required=True)
    ap.add_argument("--report-json", required=True, help="Existing point-PnP report with used_points")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--samples-per-line", type=int, default=90)
    ap.add_argument("--search-radius-px", type=int, default=28)
    ap.add_argument("--min-response", type=float, default=0.48)
    ap.add_argument("--min-observations", type=int, default=80)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    line_dir = Path(args.line_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = load_npz_camera(Path(args.intrinsics), Path(args.extrinsics))
    frame, _ = load_frame_from_pick(Path(args.points_json), line_dir)
    picked_world = used_world_points(Path(args.report_json))
    world_samples, line_ranges = make_grid_line_samples(picked_world, samples_per_line=args.samples_per_line)
    det_world, det_image, scores = find_centerline_observations(
        frame,
        cam,
        world_samples,
        line_ranges,
        search_radius_px=int(args.search_radius_px),
        min_response=float(args.min_response),
    )
    if len(det_world) < int(args.min_observations):
        # Still draw diagnostics using the original pose, but do not silently save a bad refined pose.
        rvec0, _ = cv2.Rodrigues(cam.R)
        debug = draw_line_debug(frame, cam, world_samples, det_world, det_image, rvec0, cam.t)
        cv2.imwrite(str(out_dir / f"{args.camera}_line_detection_debug_failed.jpg"), debug)
        raise RuntimeError(
            f"{args.camera}: not enough tape centerline observations: "
            f"{len(det_world)} < {args.min_observations}. See debug image."
        )

    rvec_refined, tvec_refined, metrics = optimize_pose(cam, det_world, det_image, scores)
    R_refined, _ = cv2.Rodrigues(rvec_refined)
    rvec0, _ = cv2.Rodrigues(cam.R)
    before_axes = draw_axes(frame, cam, rvec0, cam.t)
    after_axes = draw_axes(frame, cam, rvec_refined, tvec_refined)
    line_debug = draw_line_debug(frame, cam, world_samples, det_world, det_image, rvec_refined, tvec_refined)

    cv2.imwrite(str(out_dir / f"{args.camera}_axis_before_line_refine.jpg"), before_axes)
    cv2.imwrite(str(out_dir / f"{args.camera}_axis_after_line_refine.jpg"), after_axes)
    cv2.imwrite(str(out_dir / f"{args.camera}_line_detection_debug.jpg"), line_debug)

    np.savez(
        str(out_dir / f"{args.camera}_line_extrinsics_refined.npz"),
        R=R_refined,
        rotation_matrix=R_refined,
        t=tvec_refined,
        T=tvec_refined,
        tvec=tvec_refined,
        translation_vector=tvec_refined,
        rvec=rvec_refined,
        rotation_vector=rvec_refined,
        source_extrinsics=str(Path(args.extrinsics).resolve()),
        source_points=str(Path(args.points_json).resolve()),
        line_fit_metrics=np.array(metrics, dtype=object),
    )

    report = {
        "camera": args.camera,
        "stage": "line_based_extrinsic_refinement",
        "source_extrinsics": str(Path(args.extrinsics).resolve()),
        "points_json": str(Path(args.points_json).resolve()),
        "metrics": metrics,
        "camera_center_before_world_m": camera_center(cam.R, cam.t),
        "camera_center_after_world_m": camera_center(R_refined, tvec_refined),
        "outputs": {
            "refined_npz": str(out_dir / f"{args.camera}_line_extrinsics_refined.npz"),
            "axis_before": str(out_dir / f"{args.camera}_axis_before_line_refine.jpg"),
            "axis_after": str(out_dir / f"{args.camera}_axis_after_line_refine.jpg"),
            "line_debug": str(out_dir / f"{args.camera}_line_detection_debug.jpg"),
        },
        "notes": "Yellow dots are detected tape centers; magenta dots are refined projected line samples.",
    }
    (out_dir / f"{args.camera}_line_refine_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
