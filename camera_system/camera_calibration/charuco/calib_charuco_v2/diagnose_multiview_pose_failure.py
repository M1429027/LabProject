"""Evidence-based diagnostics for the four-camera demo reconstruction.

The report isolates calibration/sync/2D correspondence from the downstream
triangulation and temporal filters.  In particular, epipolar error is measured
before triangulation, and all camera-level COCO left/right swap hypotheses are
tested rather than assuming that a fixed swap discovered for another detector
also applies to HRNet.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera_system.camera_calibration.charuco.calib_charuco_v2.diagnose_cam4_offset import (
    CAM_IDS,
    frame_kpts,
)
from camera_system.camera_calibration.charuco.calib_charuco_v2.run_demo_robust_pose_pipeline import (
    LR_SWAP,
    load_npz_camera,
)
from camera_system.camera_calibration.charuco.calib_charuco_v2.sync_by_motion_peaks import (
    build_signal,
)


PAIRS = list(itertools.combinations(CAM_IDS, 2))
BILATERAL = list(range(5, 17))
TORSO = [5, 6, 11, 12]
DISTAL = [7, 8, 9, 10, 13, 14, 15, 16]


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def fundamental(c1: dict[str, Any], c2: dict[str, Any]) -> np.ndarray:
    r21 = c2["R"] @ c1["R"].T
    t21 = c2["t"].reshape(3) - r21 @ c1["t"].reshape(3)
    essential = skew(t21) @ r21
    return np.linalg.inv(c2["K"]).T @ essential @ np.linalg.inv(c1["K"])


def undistorted(camera: dict[str, Any], point: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64).reshape(1, 1, 2)
    return cv2.undistortPoints(p, camera["K"], camera["dist"], P=camera["K"]).reshape(2)


def symmetric_epipolar_px(fmat: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    x1 = np.asarray([p1[0], p1[1], 1.0], dtype=np.float64)
    x2 = np.asarray([p2[0], p2[1], 1.0], dtype=np.float64)
    l2 = fmat @ x1
    l1 = fmat.T @ x2
    numerator = abs(float(x2 @ fmat @ x1))
    d1 = numerator / max(float(np.hypot(l1[0], l1[1])), 1e-12)
    d2 = numerator / max(float(np.hypot(l2[0], l2[1])), 1e-12)
    return 0.5 * (d1 + d2)


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median_px": percentile(values, 50),
        "p90_px": percentile(values, 90),
        "p95_px": percentile(values, 95),
    }


def load_all(args: argparse.Namespace):
    line = Path(args.line_dir)
    pose_dir = Path(args.pose_dir)
    refined = Path(args.refined_extrinsics_dir)
    poses = {
        cam: json.loads((pose_dir / f"keypoints_{cam}_filtered.json").read_text(encoding="utf-8"))
        for cam in CAM_IDS
    }
    cameras = {
        "cam1demo": load_npz_camera(line / "calib_out_cam1_hvflip/cam1_intrinsics.npz", refined / "cam1_line_extrinsics_refined.npz"),
        "cam2demo": load_npz_camera(line / "calib_out_cam2_hvflip_recalc/cam2_intrinsics.npz", refined / "cam2_line_extrinsics_refined.npz"),
        "cam3demo": load_npz_camera(line / "calib_out_cam3_hvflip_fast/cam3_intrinsics.npz", refined / "cam3_line_extrinsics_refined.npz"),
        "cam4demo": load_npz_camera(line / "calib_out_cam4_hvflip_newintr_20260727/cam4_intrinsics.npz", refined / "cam4_line_extrinsics_refined.npz"),
    }
    sync = json.loads(Path(args.sync_report).read_text(encoding="utf-8"))
    return poses, cameras, sync


def local_frame(sync: dict[str, Any], cam: str, reference_frame: int) -> int:
    item = sync["affine_local_frame=a_ref_frame_plus_b"][cam]
    return int(round(float(item["a"]) * reference_frame + float(item["b"])))


def maybe_swap(array: np.ndarray, enabled: bool) -> np.ndarray:
    return array[np.asarray(LR_SWAP)] if enabled else array


def evaluate_mask(
    poses: dict[str, dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    sync: dict[str, Any],
    swap_mask: dict[str, bool],
    stride: int,
    min_conf: float,
) -> dict[str, Any]:
    start, end = [int(x) for x in sync["reference_interval_frames"]]
    fmats = {(a, b): fundamental(cameras[a], cameras[b]) for a, b in PAIRS}
    pair_groups = {
        f"{a}+{b}": {"all": [], "torso": [], "distal": [], "center": [], "low_motion": [], "high_motion": []}
        for a, b in PAIRS
    }
    joint_values = {str(j): [] for j in BILATERAL}
    _, _, ref_derivative = build_signal(poses["cam1demo"])
    ref_abs_motion = np.abs(ref_derivative)
    sampled_motion = np.asarray([ref_abs_motion[f] for f in range(start, end + 1, stride)])
    low_threshold, high_threshold = np.percentile(sampled_motion, [30, 70])

    for ref_frame in range(start, end + 1, stride):
        arrays = {
            cam: maybe_swap(frame_kpts(poses[cam], local_frame(sync, cam, ref_frame)), swap_mask[cam])
            for cam in CAM_IDS
        }
        motion_group = "low_motion" if ref_abs_motion[ref_frame] <= low_threshold else (
            "high_motion" if ref_abs_motion[ref_frame] >= high_threshold else None
        )
        for a, b in PAIRS:
            key = f"{a}+{b}"
            fmat = fmats[(a, b)]
            ua: dict[int, np.ndarray] = {}
            ub: dict[int, np.ndarray] = {}
            for j in BILATERAL:
                if (
                    np.isfinite(arrays[a][j, :2]).all()
                    and np.isfinite(arrays[b][j, :2]).all()
                    and arrays[a][j, 2] >= min_conf
                    and arrays[b][j, 2] >= min_conf
                ):
                    ua[j] = undistorted(cameras[a], arrays[a][j, :2])
                    ub[j] = undistorted(cameras[b], arrays[b][j, :2])
                    error = symmetric_epipolar_px(fmat, ua[j], ub[j])
                    pair_groups[key]["all"].append(error)
                    pair_groups[key]["torso" if j in TORSO else "distal"].append(error)
                    joint_values[str(j)].append(error)
                    if motion_group:
                        pair_groups[key][motion_group].append(error)
            for group in ((5, 6), (11, 12)):
                if all(j in ua and j in ub for j in group):
                    ca = 0.5 * (ua[group[0]] + ua[group[1]])
                    cb = 0.5 * (ub[group[0]] + ub[group[1]])
                    pair_groups[key]["center"].append(symmetric_epipolar_px(fmat, ca, cb))

    pair_summary = {
        pair: {group: summarize(vals) for group, vals in groups.items()}
        for pair, groups in pair_groups.items()
    }
    all_errors = [
        value
        for groups in pair_groups.values()
        for value in groups["all"]
    ]
    center_errors = [
        value
        for groups in pair_groups.values()
        for value in groups["center"]
    ]
    return {
        "swap_cameras": [cam for cam, enabled in swap_mask.items() if enabled],
        "all_pairs_all_bilateral": summarize(all_errors),
        "all_pairs_lr_invariant_torso_centers": summarize(center_errors),
        "per_pair": pair_summary,
        "per_joint": {joint: summarize(vals) for joint, vals in joint_values.items()},
    }


def pipeline_failure_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    subsets = [frame.get("selected_view_subset", []) for frame in payload["frames"]]
    empty = [i for i, subset in enumerate(subsets) if not subset]
    first = empty[0] if empty else None
    return {
        "num_frames": len(subsets),
        "empty_subset_frames": len(empty),
        "first_empty_subset_frame": first,
        "frames_after_first_empty": len(subsets) - first - 1 if first is not None else 0,
        "viterbi_recovery_possible_in_current_code": False if empty else True,
        "reason": (
            "DP only transitions from the immediately previous frame. An all-empty "
            "candidate row makes every following DP cost infinite; the backtrace then "
            "falls back to arbitrary/per-frame candidates."
        ),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--line-dir", default="camtest/line")
    ap.add_argument("--pose-dir", default="camtest/line/demo_hrnet_enhanced_20260728/pose_filtered")
    ap.add_argument("--refined-extrinsics-dir", default="camtest/line/line_refine_20260728")
    ap.add_argument("--sync-report", default="camtest/line/demo_hrnet_enhanced_20260728/motion_peak_sync/motion_peak_sync_report.json")
    ap.add_argument("--triangulation-json", default="camtest/line/demo_hrnet_enhanced_20260728/robust_3d_motion_peak_sync_line_refined_ext/triangulated_enhanced_robust.json")
    ap.add_argument("--output", default="camtest/line/demo_hrnet_enhanced_20260728/multiview_failure_diagnosis.json")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.30)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    poses, cameras, sync = load_all(args)
    hypotheses = []
    for bits in itertools.product((False, True), repeat=4):
        mask = dict(zip(CAM_IDS, bits))
        result = evaluate_mask(
            poses, cameras, sync, mask, stride=max(1, args.stride), min_conf=args.min_conf
        )
        hypotheses.append(result)
        print(
            f"[LR] {result['swap_cameras']} median="
            f"{result['all_pairs_all_bilateral']['median_px']:.2f}px "
            f"center={result['all_pairs_lr_invariant_torso_centers']['median_px']:.2f}px",
            flush=True,
        )
    hypotheses.sort(key=lambda x: x["all_pairs_all_bilateral"]["median_px"])
    pipeline = next(x for x in hypotheses if set(x["swap_cameras"]) == {"cam3demo", "cam4demo"})
    no_swap = next(x for x in hypotheses if not x["swap_cameras"])
    report = {
        "epipolar_metric": "symmetric point-to-epipolar-line distance after undistortion",
        "best_global_lr_swap_hypothesis": hypotheses[0],
        "current_pipeline_lr_swap_hypothesis": pipeline,
        "no_lr_swap_hypothesis": no_swap,
        "all_lr_hypotheses_ranked": [
            {
                "rank": rank,
                "swap_cameras": item["swap_cameras"],
                "median_px": item["all_pairs_all_bilateral"]["median_px"],
                "p90_px": item["all_pairs_all_bilateral"]["p90_px"],
                "torso_center_median_px": item["all_pairs_lr_invariant_torso_centers"]["median_px"],
            }
            for rank, item in enumerate(hypotheses, start=1)
        ],
        "pipeline_temporal_selection_failure": pipeline_failure_stats(Path(args.triangulation_json)),
        "two_d_filter_metrics": {
            cam: json.loads(
                (Path(args.pose_dir) / f"keypoints_{cam}_filtered_metrics.json").read_text(encoding="utf-8")
            )
            for cam in CAM_IDS
        },
        "sync_peak_fit": {
            cam: {
                key: value
                for key, value in sync["affine_local_frame=a_ref_frame_plus_b"][cam].items()
                if key in {
                    "a", "b", "residual_rms_frames", "residual_max_abs_frames",
                    "equiv_offset_at_first_event", "equiv_offset_at_last_event",
                }
            }
            for cam in CAM_IDS
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "best_lr_swap": hypotheses[0]["swap_cameras"],
        "best_median_px": hypotheses[0]["all_pairs_all_bilateral"]["median_px"],
        "pipeline_median_px": pipeline["all_pairs_all_bilateral"]["median_px"],
        "no_swap_median_px": no_swap["all_pairs_all_bilateral"]["median_px"],
        "pipeline_failure": report["pipeline_temporal_selection_failure"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
