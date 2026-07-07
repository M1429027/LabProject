"""Compare reconstructed 3D joints against dataset GT to diagnose correspondence errors.

This is an evaluation-only tool. Ground truth is used to discover failure
patterns, then the report translates those patterns into non-GT proxy rules that
can be used by the real pipeline later.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_BONES = [
    (5, 6, "shoulder_width"),
    (11, 12, "hip_width"),
    (5, 7, "left_upper_arm"),
    (7, 9, "left_forearm"),
    (6, 8, "right_upper_arm"),
    (8, 10, "right_forearm"),
    (5, 11, "left_torso"),
    (6, 12, "right_torso"),
    (11, 13, "left_thigh"),
    (13, 15, "left_shin"),
    (12, 14, "right_thigh"),
    (14, 16, "right_shin"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use dataset GT 3D points to diagnose joint/view-subset correspondence failures."
    )
    parser.add_argument("--triangulated-json", required=True, help="Path to triangulated_3d-like JSON.")
    parser.add_argument("--dataset-zip", required=True, help="Path to Harmony4D-style karate zip.")
    parser.add_argument("--sequence-prefix", default="09_karate/004_karate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=1,
        help="GT file index = reconstructed frame + frame_offset. Current 004_karate outputs use 0->00001.",
    )
    parser.add_argument(
        "--bad-error-threshold-m",
        type=float,
        default=0.25,
        help="Aligned joint error above this value is considered a bad correspondence sample.",
    )
    parser.add_argument(
        "--low-reprojection-threshold-px",
        type=float,
        default=5.0,
        help="Samples below this reprojection error but above bad-error-threshold are flagged as low-reproj failures.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_gt_poses3d(dataset_zip: str | Path, sequence_prefix: str) -> dict[int, dict[str, np.ndarray]]:
    """Load GT 3D joints as frame_index -> person_key -> (17, 4)."""

    prefix = f"{sequence_prefix.rstrip('/')}/processed_data/poses3d/"
    out: dict[int, dict[str, np.ndarray]] = {}
    with zipfile.ZipFile(dataset_zip) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.startswith(prefix) and name.endswith(".npy")
        )
        for name in names:
            frame_id = int(Path(name).stem)
            payload = np.load(io.BytesIO(archive.read(name)), allow_pickle=True).item()
            out[frame_id] = {
                str(person_id): np.asarray(joints, dtype=np.float64)
                for person_id, joints in payload.items()
            }
    if not out:
        raise FileNotFoundError(f"No GT poses3d files found under {prefix!r} in {dataset_zip}")
    return out


def point_from_joint(joint: dict[str, Any]) -> np.ndarray:
    return np.array([float(joint["x"]), float(joint["y"]), float(joint["z"])], dtype=np.float64)


def collect_points_for_pair(
    frames: list[dict[str, Any]],
    gt: dict[int, dict[str, np.ndarray]],
    pred_identity_id: int,
    gt_person_id: str,
    frame_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_points: list[np.ndarray] = []
    gt_points: list[np.ndarray] = []
    for frame in frames:
        gt_frame_id = int(frame.get("frame", 0)) + int(frame_offset)
        gt_people = gt.get(gt_frame_id)
        if not gt_people or gt_person_id not in gt_people:
            continue
        gt_joints = gt_people[gt_person_id]
        for identity in frame.get("identities", []):
            if int(identity.get("identity_id", -1)) != int(pred_identity_id):
                continue
            for joint in identity.get("joints", []):
                joint_id = int(joint["id"])
                if 0 <= joint_id < min(len(gt_joints), len(COCO_KEYPOINT_NAMES)):
                    pred_points.append(point_from_joint(joint))
                    gt_points.append(gt_joints[joint_id, :3])
    if not pred_points:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.vstack(pred_points), np.vstack(gt_points)


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Estimate similarity transform source -> target using Umeyama alignment."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 4:
        return {"status": "not_enough_points", "scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(3)}

    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    covariance = (tgt_centered.T @ src_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    det = np.linalg.det(u @ vt)
    sign = np.eye(3)
    sign[-1, -1] = 1.0 if det >= 0 else -1.0
    rotation = u @ sign @ vt
    variance = float(np.mean(np.sum(src_centered**2, axis=1)))
    scale = float(np.sum(singular_values * np.diag(sign)) / max(variance, 1e-12))
    translation = tgt_mean - scale * rotation @ src_mean
    return {
        "status": "ok",
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
    }


def apply_similarity(points: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(transform["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(transform["translation"], dtype=np.float64).reshape(3)
    scale = float(transform["scale"])
    return (scale * (rotation @ points.T)).T + translation


def finite_percentile(values: list[float], q: float) -> float | None:
    finite = np.array([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if len(finite) == 0:
        return None
    return float(np.percentile(finite, q))


def finite_mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def infer_identity_mapping(
    frames: list[dict[str, Any]],
    gt: dict[int, dict[str, np.ndarray]],
    frame_offset: int,
) -> tuple[dict[int, str], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    pred_ids = sorted(
        {
            int(identity["identity_id"])
            for frame in frames
            for identity in frame.get("identities", [])
        }
    )
    gt_ids = sorted({person_id for people in gt.values() for person_id in people.keys()})
    pair_cache: dict[tuple[int, str], dict[str, Any]] = {}
    for pred_id in pred_ids:
        for gt_id in gt_ids:
            source, target = collect_points_for_pair(frames, gt, pred_id, gt_id, frame_offset)
            transform = umeyama_similarity(source, target)
            if transform["status"] != "ok":
                mpjpe = float("inf")
                count = len(source)
            else:
                aligned = apply_similarity(source, transform)
                mpjpe = float(np.mean(np.linalg.norm(aligned - target, axis=1))) if len(source) else float("inf")
                count = len(source)
            pair_cache[(pred_id, gt_id)] = {
                "pred_identity_id": pred_id,
                "gt_person_id": gt_id,
                "num_points": int(count),
                "aligned_mpjpe_m": mpjpe,
                "transform": transform,
            }

    best_score = float("inf")
    best_mapping: dict[int, str] = {}
    for gt_perm in permutations(gt_ids, min(len(pred_ids), len(gt_ids))):
        mapping = dict(zip(pred_ids, gt_perm))
        errors = []
        for pred_id, gt_id in mapping.items():
            row = pair_cache[(pred_id, gt_id)]
            if row["num_points"] > 0 and np.isfinite(row["aligned_mpjpe_m"]):
                errors.append(float(row["aligned_mpjpe_m"]))
        score = float(np.mean(errors)) if errors else float("inf")
        if score < best_score:
            best_score = score
            best_mapping = mapping

    transforms = {
        pred_id: pair_cache[(pred_id, gt_id)]["transform"]
        for pred_id, gt_id in best_mapping.items()
    }
    pair_rows = []
    for row in pair_cache.values():
        serializable = {k: v for k, v in row.items() if k != "transform"}
        pair_rows.append(serializable)
    return best_mapping, transforms, sorted(pair_rows, key=lambda row: row["aligned_mpjpe_m"])


def summarize_samples(samples: list[dict[str, Any]], bad_error_threshold_m: float) -> dict[str, Any]:
    errors = [float(sample["gt_error_m"]) for sample in samples if np.isfinite(sample["gt_error_m"])]
    bad = [sample for sample in samples if float(sample["gt_error_m"]) >= bad_error_threshold_m]
    return {
        "count": len(samples),
        "bad_count": len(bad),
        "bad_rate": float(len(bad) / len(samples)) if samples else None,
        "mean_gt_error_m": finite_mean(errors),
        "median_gt_error_m": finite_percentile(errors, 50),
        "p90_gt_error_m": finite_percentile(errors, 90),
        "p95_gt_error_m": finite_percentile(errors, 95),
    }


def make_diagnostic_report(
    payload: dict[str, Any],
    gt: dict[int, dict[str, np.ndarray]],
    frame_offset: int,
    bad_error_threshold_m: float,
    low_reprojection_threshold_px: float,
) -> dict[str, Any]:
    frames = payload.get("frames", [])
    identity_mapping, transforms, pair_identity_errors = infer_identity_mapping(frames, gt, frame_offset)

    samples: list[dict[str, Any]] = []
    bone_samples: list[dict[str, Any]] = []
    for frame in frames:
        frame_idx = int(frame.get("frame", 0))
        gt_frame_id = frame_idx + int(frame_offset)
        gt_people = gt.get(gt_frame_id, {})
        for identity in frame.get("identities", []):
            pred_id = int(identity.get("identity_id", -1))
            gt_id = identity_mapping.get(pred_id)
            if gt_id is None or gt_id not in gt_people or pred_id not in transforms:
                continue
            transform = transforms[pred_id]
            gt_joints = gt_people[gt_id]
            pred_joints = {int(joint["id"]): joint for joint in identity.get("joints", [])}
            for joint_id, joint in pred_joints.items():
                if joint_id >= len(gt_joints) or joint_id >= len(COCO_KEYPOINT_NAMES):
                    continue
                pred_point = apply_similarity(point_from_joint(joint).reshape(1, 3), transform)[0]
                gt_point = gt_joints[joint_id, :3]
                error = float(np.linalg.norm(pred_point - gt_point))
                views = tuple(sorted(str(view) for view in joint.get("views", [])))
                sample = {
                    "frame": frame_idx,
                    "gt_frame": gt_frame_id,
                    "identity_id": pred_id,
                    "gt_person_id": gt_id,
                    "joint_id": joint_id,
                    "joint_name": COCO_KEYPOINT_NAMES[joint_id],
                    "gt_error_m": error,
                    "num_views": safe_int(joint.get("num_views"), 0),
                    "views": list(views),
                    "mean_confidence": joint.get("mean_confidence"),
                    "mean_reprojection_error_px": joint.get("mean_reprojection_error_px"),
                    "candidate_base_score": joint.get("candidate_base_score"),
                    "pose_hypothesis_score": joint.get("pose_hypothesis_score"),
                }
                samples.append(sample)

            for joint_a, joint_b, bone_name in COCO_BONES:
                if joint_a not in pred_joints or joint_b not in pred_joints:
                    continue
                if joint_a >= len(gt_joints) or joint_b >= len(gt_joints):
                    continue
                pred_a = apply_similarity(point_from_joint(pred_joints[joint_a]).reshape(1, 3), transform)[0]
                pred_b = apply_similarity(point_from_joint(pred_joints[joint_b]).reshape(1, 3), transform)[0]
                gt_a = gt_joints[joint_a, :3]
                gt_b = gt_joints[joint_b, :3]
                pred_len = float(np.linalg.norm(pred_b - pred_a))
                gt_len = float(np.linalg.norm(gt_b - gt_a))
                bone_samples.append(
                    {
                        "frame": frame_idx,
                        "identity_id": pred_id,
                        "gt_person_id": gt_id,
                        "bone_name": bone_name,
                        "joint_a": joint_a,
                        "joint_b": joint_b,
                        "pred_length_m": pred_len,
                        "gt_length_m": gt_len,
                        "length_ratio": float(pred_len / gt_len) if gt_len > 1e-9 else None,
                    }
                )

    by_joint: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_view_set: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    by_joint_view_set: dict[tuple[int, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    low_reproj_bad: list[dict[str, Any]] = []
    for sample in samples:
        joint_id = int(sample["joint_id"])
        views = tuple(sample["views"])
        by_joint[joint_id].append(sample)
        by_view_set[views].append(sample)
        by_joint_view_set[(joint_id, views)].append(sample)
        reproj = sample.get("mean_reprojection_error_px")
        if (
            reproj is not None
            and float(reproj) <= float(low_reprojection_threshold_px)
            and float(sample["gt_error_m"]) >= float(bad_error_threshold_m)
        ):
            low_reproj_bad.append(sample)

    joint_rows = []
    for joint_id, rows in sorted(by_joint.items()):
        summary = summarize_samples(rows, bad_error_threshold_m)
        reproj = [float(row["mean_reprojection_error_px"]) for row in rows if row.get("mean_reprojection_error_px") is not None]
        conf = [float(row["mean_confidence"]) for row in rows if row.get("mean_confidence") is not None]
        joint_rows.append(
            {
                "joint_id": joint_id,
                "joint_name": COCO_KEYPOINT_NAMES[joint_id],
                **summary,
                "mean_reprojection_error_px": finite_mean(reproj),
                "mean_confidence": finite_mean(conf),
                "top_view_sets": [
                    {
                        "views": list(view_set),
                        "count": int(count),
                    }
                    for view_set, count in Counter(tuple(row["views"]) for row in rows).most_common(5)
                ],
            }
        )

    view_set_rows = []
    for view_set, rows in by_view_set.items():
        if not view_set:
            continue
        view_set_rows.append(
            {
                "views": list(view_set),
                **summarize_samples(rows, bad_error_threshold_m),
                "low_reprojection_bad_count": sum(
                    1
                    for row in rows
                    if row.get("mean_reprojection_error_px") is not None
                    and float(row["mean_reprojection_error_px"]) <= float(low_reprojection_threshold_px)
                    and float(row["gt_error_m"]) >= float(bad_error_threshold_m)
                ),
            }
        )

    joint_view_rows = []
    for (joint_id, view_set), rows in by_joint_view_set.items():
        if len(rows) < 3:
            continue
        joint_view_rows.append(
            {
                "joint_id": joint_id,
                "joint_name": COCO_KEYPOINT_NAMES[joint_id],
                "views": list(view_set),
                **summarize_samples(rows, bad_error_threshold_m),
            }
        )

    bone_rows = []
    by_bone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bone_samples:
        by_bone[str(row["bone_name"])].append(row)
    for bone_name, rows in sorted(by_bone.items()):
        ratios = [float(row["length_ratio"]) for row in rows if row.get("length_ratio") is not None and np.isfinite(row["length_ratio"])]
        if not ratios:
            continue
        bone_rows.append(
            {
                "bone_name": bone_name,
                "count": len(ratios),
                "median_length_ratio": float(median(ratios)),
                "p90_abs_ratio_error": finite_percentile([abs(r - 1.0) for r in ratios], 90),
                "bad_ratio_count": sum(1 for r in ratios if r < 0.55 or r > 1.8),
            }
        )

    top_bad_samples = sorted(samples, key=lambda row: float(row["gt_error_m"]), reverse=True)[:80]
    risky_joint_views = sorted(
        joint_view_rows,
        key=lambda row: (
            float(row["bad_rate"] or 0.0),
            float(row["p90_gt_error_m"] or 0.0),
            int(row["count"]),
        ),
        reverse=True,
    )[:30]

    proxy_rules = derive_proxy_rules(
        joint_rows=joint_rows,
        view_set_rows=view_set_rows,
        risky_joint_views=risky_joint_views,
        bone_rows=bone_rows,
        low_reproj_bad=low_reproj_bad,
        bad_error_threshold_m=bad_error_threshold_m,
        low_reprojection_threshold_px=low_reprojection_threshold_px,
    )

    return {
        "stage": "gt_correspondence_diagnostic",
        "metadata": {
            "source_triangulated_metadata": payload.get("metadata", {}),
            "frame_offset": frame_offset,
            "bad_error_threshold_m": bad_error_threshold_m,
            "low_reprojection_threshold_px": low_reprojection_threshold_px,
            "note": "GT is used only for diagnosis/evaluation. Proxy rules are designed to be usable without GT.",
        },
        "identity_mapping": {
            str(pred_id): gt_id for pred_id, gt_id in identity_mapping.items()
        },
        "identity_pair_errors": pair_identity_errors,
        "summary": {
            **summarize_samples(samples, bad_error_threshold_m),
            "num_frames": len(frames),
            "num_gt_frames": len(gt),
            "num_low_reprojection_bad_samples": len(low_reproj_bad),
        },
        "joint_error_summary": sorted(joint_rows, key=lambda row: float(row["p90_gt_error_m"] or 0.0), reverse=True),
        "view_set_error_summary": sorted(view_set_rows, key=lambda row: float(row["p90_gt_error_m"] or 0.0), reverse=True),
        "joint_view_risk_summary": risky_joint_views,
        "bone_length_summary": sorted(bone_rows, key=lambda row: float(row["p90_abs_ratio_error"] or 0.0), reverse=True),
        "low_reprojection_bad_samples": sorted(
            low_reproj_bad,
            key=lambda row: float(row["gt_error_m"]),
            reverse=True,
        )[:80],
        "top_bad_samples": top_bad_samples,
        "proxy_rule_recommendations": proxy_rules,
    }


def derive_proxy_rules(
    joint_rows: list[dict[str, Any]],
    view_set_rows: list[dict[str, Any]],
    risky_joint_views: list[dict[str, Any]],
    bone_rows: list[dict[str, Any]],
    low_reproj_bad: list[dict[str, Any]],
    bad_error_threshold_m: float,
    low_reprojection_threshold_px: float,
) -> list[dict[str, Any]]:
    """Translate GT findings into inference-time rules that do not need GT."""

    rules: list[dict[str, Any]] = []

    unstable_joints = [
        row
        for row in joint_rows
        if (row.get("bad_rate") or 0.0) >= 0.35 or (row.get("p90_gt_error_m") or 0.0) >= bad_error_threshold_m
    ]
    if unstable_joints:
        rules.append(
            {
                "rule_id": "joint_specific_risk_weight",
                "gt_finding": "Some joints have consistently higher aligned GT error.",
                "inference_proxy": "Apply a lower prior trust or require stronger multi-view support for these joints.",
                "affected_joints": [
                    {
                        "joint_id": row["joint_id"],
                        "joint_name": row["joint_name"],
                        "bad_rate": row["bad_rate"],
                        "p90_gt_error_m": row["p90_gt_error_m"],
                    }
                    for row in unstable_joints[:8]
                ],
            }
        )

    risky_view_sets = [
        row
        for row in view_set_rows
        if row.get("count", 0) >= 5 and ((row.get("bad_rate") or 0.0) >= 0.35)
    ]
    if risky_view_sets:
        rules.append(
            {
                "rule_id": "view_subset_penalty",
                "gt_finding": "Some view subsets repeatedly produce high 3D error.",
                "inference_proxy": "Penalize these subsets in candidate selection unless they have high confidence, high ray angle, and pose-level support.",
                "risky_view_sets": [
                    {
                        "views": row["views"],
                        "bad_rate": row["bad_rate"],
                        "p90_gt_error_m": row["p90_gt_error_m"],
                        "low_reprojection_bad_count": row["low_reprojection_bad_count"],
                    }
                    for row in risky_view_sets[:8]
                ],
            }
        )

    if low_reproj_bad:
        rules.append(
            {
                "rule_id": "do_not_trust_reprojection_alone",
                "gt_finding": (
                    f"{len(low_reproj_bad)} samples have reprojection <= "
                    f"{low_reprojection_threshold_px:.1f}px but GT error >= {bad_error_threshold_m:.2f}m."
                ),
                "inference_proxy": "Candidate score must combine reprojection, ray angle/depth, bone plausibility, and temporal stability.",
            }
        )

    risky_joint_view_items = [
        row
        for row in risky_joint_views
        if row.get("count", 0) >= 3 and (row.get("bad_rate") or 0.0) >= 0.45
    ]
    if risky_joint_view_items:
        rules.append(
            {
                "rule_id": "joint_view_pair_blacklist_or_penalty",
                "gt_finding": "Certain joint + view-subset combinations are especially unstable.",
                "inference_proxy": "Use as a soft penalty table, not a hard blacklist, to preserve generalization.",
                "risky_joint_view_pairs": [
                    {
                        "joint_id": row["joint_id"],
                        "joint_name": row["joint_name"],
                        "views": row["views"],
                        "bad_rate": row["bad_rate"],
                        "p90_gt_error_m": row["p90_gt_error_m"],
                    }
                    for row in risky_joint_view_items[:10]
                ],
            }
        )

    bad_bones = [
        row
        for row in bone_rows
        if row.get("bad_ratio_count", 0) > 0 or (row.get("p90_abs_ratio_error") or 0.0) >= 0.55
    ]
    if bad_bones:
        rules.append(
            {
                "rule_id": "pose_level_bone_consistency_gate",
                "gt_finding": "Bone-length ratios expose candidates that look valid per joint but break body structure.",
                "inference_proxy": "Reject or downweight full-pose candidates with extreme bone ratios before temporal smoothing.",
                "risky_bones": [
                    {
                        "bone_name": row["bone_name"],
                        "p90_abs_ratio_error": row["p90_abs_ratio_error"],
                        "bad_ratio_count": row["bad_ratio_count"],
                    }
                    for row in bad_bones[:8]
                ],
            }
        )

    return rules


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# GT Correspondence Diagnostic",
        "",
        "This report uses dataset GT only to diagnose failure modes. The final pipeline should use the proxy rules, not GT.",
        "",
        "## Summary",
    ]
    summary = report["summary"]
    lines.extend(
        [
            f"- Frames: {summary['num_frames']}",
            f"- Samples: {summary['count']}",
            f"- Mean aligned joint error: {summary['mean_gt_error_m']:.4f} m" if summary["mean_gt_error_m"] is not None else "- Mean aligned joint error: n/a",
            f"- P90 aligned joint error: {summary['p90_gt_error_m']:.4f} m" if summary["p90_gt_error_m"] is not None else "- P90 aligned joint error: n/a",
            f"- Bad sample rate: {summary['bad_rate']:.2%}" if summary["bad_rate"] is not None else "- Bad sample rate: n/a",
            f"- Low-reprojection but bad samples: {summary['num_low_reprojection_bad_samples']}",
            "",
            "## Identity Mapping",
        ]
    )
    for pred_id, gt_id in report["identity_mapping"].items():
        lines.append(f"- Reconstructed identity {pred_id} -> GT {gt_id}")

    lines.extend(["", "## Worst Joints"])
    for row in report["joint_error_summary"][:8]:
        lines.append(
            f"- {row['joint_name']} ({row['joint_id']}): "
            f"p90={row['p90_gt_error_m']:.4f}m, bad_rate={row['bad_rate']:.2%}, "
            f"mean_reproj={row['mean_reprojection_error_px'] if row['mean_reprojection_error_px'] is not None else 'n/a'}"
        )

    lines.extend(["", "## Risky View Sets"])
    for row in report["view_set_error_summary"][:8]:
        lines.append(
            f"- {','.join(row['views'])}: p90={row['p90_gt_error_m']:.4f}m, "
            f"bad_rate={row['bad_rate']:.2%}, count={row['count']}, "
            f"low_reproj_bad={row['low_reprojection_bad_count']}"
        )

    lines.extend(["", "## Proxy Rule Recommendations"])
    for rule in report["proxy_rule_recommendations"]:
        lines.append(f"- {rule['rule_id']}: {rule['inference_proxy']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(args.triangulated_json)
    gt = load_gt_poses3d(args.dataset_zip, args.sequence_prefix)
    report = make_diagnostic_report(
        payload=payload,
        gt=gt,
        frame_offset=int(args.frame_offset),
        bad_error_threshold_m=float(args.bad_error_threshold_m),
        low_reprojection_threshold_px=float(args.low_reprojection_threshold_px),
    )

    json_path = output_dir / "gt_correspondence_diagnostic_report.json"
    md_path = output_dir / "gt_correspondence_diagnostic_report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item))
    write_markdown(report, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
