import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_keypoints(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = data["frames"]
    n_frames = len(frames)
    n_joints = len(data.get("keypoint_names", []))
    if n_joints == 0:
        for f in frames:
            people = f.get("people", [])
            if people:
                n_joints = len(people[0].get("keypoints", []))
                break
    arr = np.full((n_frames, n_joints, 3), np.nan, dtype=np.float64)

    for i, frame in enumerate(frames):
        people = frame.get("people", [])
        if not people:
            continue
        kpts = people[0].get("keypoints", [])
        for j in range(min(n_joints, len(kpts))):
            d = kpts[j]
            arr[i, j, 0] = float(d.get("x", np.nan))
            arr[i, j, 1] = float(d.get("y", np.nan))
            arr[i, j, 2] = float(d.get("confidence", 0.0))

    fps = float(data.get("metadata", {}).get("fps", 0.0))
    return arr, fps, n_frames, n_joints


def motion_energy(kpts):
    n = kpts.shape[0]
    e = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        p0 = kpts[t - 1]
        p1 = kpts[t]
        valid = np.isfinite(p0[:, 0]) & np.isfinite(p1[:, 0])
        if not np.any(valid):
            continue
        dxy = p1[valid, :2] - p0[valid, :2]
        w = np.minimum(p0[valid, 2], p1[valid, 2])
        e[t] = np.sum(w * np.linalg.norm(dxy, axis=1))
    std = float(np.std(e))
    if std > 1e-9:
        e = (e - np.mean(e)) / std
    else:
        e = e - np.mean(e)
    return e


def corr_score(e1, e2, shift, min_overlap):
    s1, s2, n = aligned_ranges(len(e1), len(e2), shift)
    if n < min_overlap:
        return None
    a = e1[s1 : s1 + n]
    b = e2[s2 : s2 + n]
    return float(np.dot(a, b) / n)


def load_K(path):
    with np.load(path, allow_pickle=True) as d:
        for k in ["K", "mtx", "camera_matrix", "cameraMatrix", "intrinsic_matrix"]:
            if k in d:
                return np.array(d[k], dtype=np.float64)
    raise ValueError(f"Cannot find K in {path}")


def load_R_t(path):
    with np.load(path, allow_pickle=True) as d:
        R = None
        for k in ["R", "rotation_matrix"]:
            if k in d:
                arr = np.array(d[k], dtype=np.float64)
                if arr.shape == (3, 3):
                    R = arr
                    break
        if R is None:
            for k in ["rvec", "rotation_vector", "rvecs"]:
                if k in d:
                    rv = np.array(d[k], dtype=np.float64).reshape(-1)
                    if rv.size == 3:
                        R, _ = cv2.Rodrigues(rv.reshape(3, 1))
                        break
        if R is None:
            raise ValueError(f"Cannot find R in {path}")

        t = None
        for k in ["t", "T", "tvec", "translation_vector", "translation", "tvecs"]:
            if k in d:
                arr = np.array(d[k], dtype=np.float64).reshape(-1)
                if arr.size >= 3:
                    t = arr[:3].reshape(3, 1)
                    break
        if t is None:
            raise ValueError(f"Cannot find t in {path}")
    return R, t


def load_projection_matrices(args):
    if args.camera_params:
        data = json.loads(Path(args.camera_params).read_text(encoding="utf-8"))
        P1 = np.array(data["cam1"]["P"], dtype=np.float64)
        P2 = np.array(data["cam2"]["P"], dtype=np.float64)
        return P1, P2

    required = [args.cam1_intrinsics, args.cam1_extrinsics, args.cam2_intrinsics, args.cam2_extrinsics]
    if any(not x for x in required):
        return None, None

    K1 = load_K(args.cam1_intrinsics)
    K2 = load_K(args.cam2_intrinsics)
    R1, t1 = load_R_t(args.cam1_extrinsics)
    R2, t2 = load_R_t(args.cam2_extrinsics)
    P1 = K1 @ np.hstack([R1, t1])
    P2 = K2 @ np.hstack([R2, t2])
    return P1, P2


def triangulate_two_view(P1, P2, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    A = np.zeros((4, 4), dtype=np.float64)
    A[0] = x1 * P1[2] - P1[0]
    A[1] = y1 * P1[2] - P1[1]
    A[2] = x2 * P2[2] - P2[0]
    A[3] = y2 * P2[2] - P2[1]
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    if abs(X[3]) < 1e-9:
        return None
    X = X[:3] / X[3]
    if np.any(~np.isfinite(X)):
        return None
    return X


def project(P, X):
    h = P @ np.append(X, 1.0)
    if abs(h[2]) < 1e-9:
        return None
    return h[:2] / h[2]


def aligned_ranges(n1, n2, shift):
    s1 = max(0, shift)
    s2 = max(0, -shift)
    n = min(n1 - s1, n2 - s2)
    return s1, s2, max(0, n)


def fine_score_for_shift(k1, k2, P1, P2, shift, conf_thresh, sample_step, min_points):
    s1, s2, n = aligned_ranges(k1.shape[0], k2.shape[0], shift)
    if n <= 0:
        return None

    errs = []
    for kk in range(0, n, sample_step):
        i1 = s1 + kk
        i2 = s2 + kk
        for j in range(min(k1.shape[1], k2.shape[1])):
            c1 = k1[i1, j, 2]
            c2 = k2[i2, j, 2]
            if not np.isfinite(c1) or not np.isfinite(c2):
                continue
            if c1 < conf_thresh or c2 < conf_thresh:
                continue

            p1 = k1[i1, j, :2]
            p2 = k2[i2, j, :2]
            if np.any(~np.isfinite(p1)) or np.any(~np.isfinite(p2)):
                continue

            X = triangulate_two_view(P1, P2, p1, p2)
            if X is None:
                continue
            u1 = project(P1, X)
            u2 = project(P2, X)
            if u1 is None or u2 is None:
                continue

            e1 = float(np.linalg.norm(u1 - p1))
            e2 = float(np.linalg.norm(u2 - p2))
            e = 0.5 * (e1 + e2)
            if np.isfinite(e):
                errs.append(e)

    if len(errs) < min_points:
        return {
            "shift": int(shift),
            "count": int(len(errs)),
            "median": None,
            "mean": None,
        }

    arr = np.array(errs, dtype=np.float64)
    return {
        "shift": int(shift),
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 1: estimate time offset between cam1/cam2 keypoints.")
    ap.add_argument("--keypoints-cam1", required=True)
    ap.add_argument("--keypoints-cam2", required=True)
    ap.add_argument("--camera-params", default="", help="camera_params.json with P matrices")
    ap.add_argument("--cam1-intrinsics", default="")
    ap.add_argument("--cam1-extrinsics", default="")
    ap.add_argument("--cam2-intrinsics", default="")
    ap.add_argument("--cam2-extrinsics", default="")
    ap.add_argument("--max-shift", type=int, default=120)
    ap.add_argument("--fine-window", type=int, default=12)
    ap.add_argument("--min-overlap", type=int, default=120)
    ap.add_argument("--conf-thresh", type=float, default=0.35)
    ap.add_argument("--sample-step", type=int, default=2)
    ap.add_argument("--min-points", type=int, default=300)
    ap.add_argument("--out-json", default="docs/phase1_time_offset_report.json")
    args = ap.parse_args()

    k1, fps1, n1, j1 = load_keypoints(args.keypoints_cam1)
    k2, fps2, n2, j2 = load_keypoints(args.keypoints_cam2)

    print("== Phase 1 Time Offset ==")
    print(f"cam1: frames={n1}, joints={j1}, fps={fps1:.3f}")
    print(f"cam2: frames={n2}, joints={j2}, fps={fps2:.3f}")

    e1 = motion_energy(k1)
    e2 = motion_energy(k2)

    coarse = []
    for shift in range(-args.max_shift, args.max_shift + 1):
        c = corr_score(e1, e2, shift, args.min_overlap)
        if c is not None:
            coarse.append((shift, c))

    if not coarse:
        raise RuntimeError("No valid coarse candidates. Increase overlap or reduce max_shift.")

    coarse.sort(key=lambda x: x[1], reverse=True)
    coarse_best_shift, coarse_best_corr = coarse[0]
    print(f"Coarse best shift: {coarse_best_shift} (corr={coarse_best_corr:.4f})")

    P1, P2 = load_projection_matrices(args)
    fine_candidates = []
    fine_best = None

    if P1 is not None and P2 is not None:
        low = max(-args.max_shift, coarse_best_shift - args.fine_window)
        high = min(args.max_shift, coarse_best_shift + args.fine_window)
        for s in range(low, high + 1):
            stat = fine_score_for_shift(
                k1,
                k2,
                P1,
                P2,
                s,
                conf_thresh=args.conf_thresh,
                sample_step=max(1, args.sample_step),
                min_points=max(10, args.min_points),
            )
            fine_candidates.append(stat)

        valid = [x for x in fine_candidates if x["median"] is not None]
        if valid:
            valid.sort(key=lambda x: (x["median"], -x["count"]))
            fine_best = valid[0]
            print(
                "Fine best shift: "
                f"{fine_best['shift']} (median={fine_best['median']:.4f}px, count={fine_best['count']})"
            )
        else:
            print("Fine stage: no valid candidates (insufficient points). Keep coarse shift.")
    else:
        print("Fine stage skipped (no camera projection input).")

    best_shift = int(fine_best["shift"] if fine_best is not None else coarse_best_shift)
    s1, s2, n_overlap = aligned_ranges(n1, n2, best_shift)

    result = {
        "inputs": {
            "keypoints_cam1": str(Path(args.keypoints_cam1)),
            "keypoints_cam2": str(Path(args.keypoints_cam2)),
            "camera_params": str(Path(args.camera_params)) if args.camera_params else None,
            "cam1_intrinsics": args.cam1_intrinsics or None,
            "cam1_extrinsics": args.cam1_extrinsics or None,
            "cam2_intrinsics": args.cam2_intrinsics or None,
            "cam2_extrinsics": args.cam2_extrinsics or None,
        },
        "coarse": {
            "best_shift": int(coarse_best_shift),
            "best_corr": float(coarse_best_corr),
            "top10": [{"shift": int(s), "corr": float(c)} for s, c in coarse[:10]],
        },
        "fine": {
            "enabled": bool(P1 is not None and P2 is not None),
            "best": fine_best,
            "candidates": fine_candidates,
        },
        "selected_shift": int(best_shift),
        "convention": "cam1_index = cam2_index + selected_shift",
        "aligned_range": {
            "overlap_frames": int(n_overlap),
            "cam1_start": int(s1),
            "cam1_end": int(s1 + n_overlap - 1) if n_overlap > 0 else None,
            "cam2_start": int(s2),
            "cam2_end": int(s2 + n_overlap - 1) if n_overlap > 0 else None,
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
