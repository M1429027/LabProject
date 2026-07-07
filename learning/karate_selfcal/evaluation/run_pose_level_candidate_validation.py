"""Pose-level validation for triangulated skeleton candidates.

This diagnostic checks whether individually triangulated joints are consistent
with the full body pose. It is meant to guide the next reconstruction step
before adding Transformer refinement.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


COCO_BONES = [
    (5, 6, "shoulder_width"),
    (11, 12, "hip_width"),
    (5, 11, "left_torso"),
    (6, 12, "right_torso"),
    (5, 7, "left_upper_arm"),
    (7, 9, "left_forearm"),
    (6, 8, "right_upper_arm"),
    (8, 10, "right_forearm"),
    (11, 13, "left_thigh"),
    (13, 15, "left_shin"),
    (12, 14, "right_thigh"),
    (14, 16, "right_shin"),
    (5, 0, "left_neck_head"),
    (6, 0, "right_neck_head"),
]

PARENT_BY_JOINT = {
    12: 11,
    5: 11,
    6: 12,
    7: 5,
    9: 7,
    8: 6,
    10: 8,
    13: 11,
    15: 13,
    14: 12,
    16: 14,
    0: 5,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
}

TORSO_JOINTS = {5, 6, 11, 12}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate triangulated joints with pose-level constraints.")
    parser.add_argument("--input-json", required=True, help="Path to triangulated_3d.json")
    parser.add_argument("--output-dir", required=True, help="Directory for validation outputs")
    parser.add_argument("--min-bone-samples", type=int, default=8)
    parser.add_argument("--bone-ratio-min", type=float, default=0.35)
    parser.add_argument("--bone-ratio-max", type=float, default=2.25)
    parser.add_argument("--max-root-distance-ratio", type=float, default=4.0)
    parser.add_argument("--max-temporal-jump-ratio", type=float, default=2.5)
    parser.add_argument("--torso-high-threshold", type=float, default=0.75)
    parser.add_argument("--torso-medium-threshold", type=float, default=0.45)
    parser.add_argument(
        "--drop-invalid-joints",
        action="store_true",
        help="Also write a review JSON where invalid joints are removed.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def point(joint: dict[str, Any]) -> np.ndarray:
    return np.array([float(joint["x"]), float(joint["y"]), float(joint["z"])], dtype=np.float64)


def robust_median(values: list[float], low: float = 10.0, high: float = 90.0) -> float | None:
    finite = np.array([float(v) for v in values if np.isfinite(v) and float(v) > 1e-9], dtype=np.float64)
    if len(finite) == 0:
        return None
    if len(finite) >= 8:
        lo, hi = np.percentile(finite, [low, high])
        finite = finite[(finite >= lo) & (finite <= hi)]
    if len(finite) == 0:
        return None
    return float(np.median(finite))


def estimate_identity_priors(payload: dict[str, Any], min_samples: int) -> dict[int, dict[str, Any]]:
    """Estimate robust bone and torso priors per identity."""

    bone_lengths: dict[int, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    torso_scales: dict[int, list[float]] = defaultdict(list)
    for frame in payload.get("frames", []):
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            joints = {int(joint["id"]): joint for joint in identity.get("joints", [])}
            points = {joint_id: point(joint) for joint_id, joint in joints.items()}
            torso_lengths = []
            for joint_a, joint_b, _ in COCO_BONES:
                if joint_a not in points or joint_b not in points:
                    continue
                length = float(np.linalg.norm(points[joint_b] - points[joint_a]))
                if np.isfinite(length) and length > 1e-9:
                    bone_lengths[identity_id][(joint_a, joint_b)].append(length)
                    if joint_a in TORSO_JOINTS and joint_b in TORSO_JOINTS:
                        torso_lengths.append(length)
            torso_scale = robust_median(torso_lengths, low=0.0, high=100.0)
            if torso_scale is not None:
                torso_scales[identity_id].append(torso_scale)

    priors: dict[int, dict[str, Any]] = {}
    for identity_id, by_bone in bone_lengths.items():
        priors[identity_id] = {"bone_lengths": {}, "torso_scale": None}
        for bone, values in by_bone.items():
            if len(values) < int(min_samples):
                continue
            med = robust_median(values)
            if med is not None:
                priors[identity_id]["bone_lengths"][bone] = med
        torso_med = robust_median(torso_scales.get(identity_id, []))
        if torso_med is not None:
            priors[identity_id]["torso_scale"] = torso_med
    return priors


def torso_center_and_scale(joints: dict[int, dict[str, Any]], fallback_scale: float | None) -> tuple[np.ndarray | None, float | None]:
    points = [point(joints[joint_id]) for joint_id in TORSO_JOINTS if joint_id in joints]
    if not points:
        return None, fallback_scale
    center = np.mean(np.vstack(points), axis=0)
    torso_lengths = []
    for joint_a, joint_b in [(5, 6), (11, 12), (5, 11), (6, 12)]:
        if joint_a in joints and joint_b in joints:
            torso_lengths.append(float(np.linalg.norm(point(joints[joint_b]) - point(joints[joint_a]))))
    scale = robust_median(torso_lengths, low=0.0, high=100.0) or fallback_scale
    return center, scale


def joint_reprojection_score(joint: dict[str, Any], scale_px: float = 25.0) -> float:
    """Convert one joint reprojection error into a soft score."""

    reprojection_error = joint.get("mean_reprojection_error_px")
    if reprojection_error is None:
        return 0.6
    return float(1.0 / (1.0 + max(float(reprojection_error), 0.0) / float(scale_px)))


def joint_view_count(joint: dict[str, Any]) -> int:
    """Return the number of supporting views for one joint."""

    views = joint.get("views") or joint.get("inlier_views") or []
    return len(set(str(view) for view in views))


def expected_shoulder_hip_ratio(priors: dict[str, Any]) -> float | None:
    """Estimate a robust shoulder-width / hip-width ratio prior."""

    shoulder = bone_prior_for(priors, 5, 6)
    hip = bone_prior_for(priors, 11, 12)
    if shoulder is None or hip is None or hip <= 1e-9:
        return None
    return float(shoulder / hip)


def validate_torso_anchor(
    joints: dict[int, dict[str, Any]],
    priors: dict[str, Any],
    fallback_scale: float | None,
    previous_anchor: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, float, dict[str, Any]]:
    """Validate whether torso can be used as a pose anchor for one identity frame."""

    center, scale = torso_center_and_scale(joints, fallback_scale=fallback_scale)
    if scale is None or scale <= 1e-9:
        scale = fallback_scale or 0.4

    available_torso = [joint_id for joint_id in TORSO_JOINTS if joint_id in joints]
    coverage_score = len(available_torso) / float(len(TORSO_JOINTS))

    reprojection_scores = [joint_reprojection_score(joints[joint_id]) for joint_id in available_torso]
    reprojection_score = float(np.mean(reprojection_scores)) if reprojection_scores else 0.0

    view_counts = [min(joint_view_count(joints[joint_id]) / 3.0, 1.0) for joint_id in available_torso]
    view_support_score = float(np.mean(view_counts)) if view_counts else 0.0

    ratio_score = 0.5
    shoulder_hip_ratio = None
    ratio_reference = expected_shoulder_hip_ratio(priors)
    if 5 in joints and 6 in joints and 11 in joints and 12 in joints:
        shoulder_width = float(np.linalg.norm(point(joints[6]) - point(joints[5])))
        hip_width = float(np.linalg.norm(point(joints[12]) - point(joints[11])))
        if hip_width > 1e-9:
            shoulder_hip_ratio = shoulder_width / hip_width
            if ratio_reference is None or ratio_reference <= 1e-9:
                ratio_reference = 1.2
            ratio_error = abs(np.log(max(shoulder_hip_ratio, 1e-6) / max(ratio_reference, 1e-6)))
            ratio_score = float(1.0 / (1.0 + ratio_error * 2.0))

    temporal_score = 0.7
    temporal_center_jump_ratio = None
    if center is not None and previous_anchor is not None and previous_anchor.get("center") is not None:
        previous_center = np.asarray(previous_anchor["center"], dtype=np.float64)
        temporal_center_jump_ratio = float(np.linalg.norm(center - previous_center) / float(scale))
        temporal_score = float(1.0 / (1.0 + temporal_center_jump_ratio))

    anchor_score = (
        0.28 * coverage_score
        + 0.22 * reprojection_score
        + 0.18 * ratio_score
        + 0.17 * temporal_score
        + 0.15 * view_support_score
    )
    if anchor_score >= float(args.torso_high_threshold):
        status = "high"
        selected_center = center
        selected_scale = scale
    elif anchor_score >= float(args.torso_medium_threshold):
        status = "medium"
        selected_center = center
        selected_scale = scale
    elif previous_anchor is not None and previous_anchor.get("status") in {"high", "medium", "fallback_previous"}:
        status = "fallback_previous"
        selected_center = np.asarray(previous_anchor["center"], dtype=np.float64)
        selected_scale = float(previous_anchor["scale"])
    else:
        status = "low"
        selected_center = center
        selected_scale = scale

    diagnostics = {
        "score": float(np.clip(anchor_score, 0.0, 1.0)),
        "status": status,
        "coverage_score": float(coverage_score),
        "reprojection_score": float(reprojection_score),
        "ratio_score": float(ratio_score),
        "temporal_score": float(temporal_score),
        "view_support_score": float(view_support_score),
        "available_torso_joints": available_torso,
        "shoulder_hip_ratio": shoulder_hip_ratio,
        "ratio_reference": ratio_reference,
        "temporal_center_jump_ratio": temporal_center_jump_ratio,
        "scale": float(selected_scale),
        "center": selected_center.tolist() if selected_center is not None else None,
    }
    return selected_center, float(selected_scale), diagnostics


def bone_prior_for(priors: dict[str, Any], parent_id: int, joint_id: int) -> float | None:
    bone_lengths = priors.get("bone_lengths", {})
    return bone_lengths.get((parent_id, joint_id)) or bone_lengths.get((joint_id, parent_id))


def validate_payload(
    payload: dict[str, Any],
    priors_by_identity: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach pose-level validation diagnostics to every joint."""

    output = json.loads(json.dumps(payload))
    previous_points: dict[tuple[int, int], np.ndarray] = {}
    previous_anchors: dict[int, dict[str, Any]] = {}
    invalid_by_reason = defaultdict(int)
    joint_rows = defaultdict(lambda: {"count": 0, "invalid": 0, "score_sum": 0.0})
    pose_scores = []
    anchor_scores = []
    anchor_status_counts = defaultdict(int)

    for frame in output.get("frames", []):
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            priors = priors_by_identity.get(identity_id, {})
            fallback_scale = priors.get("torso_scale")
            joints = {int(joint["id"]): joint for joint in identity.get("joints", [])}
            center, torso_scale, anchor_diag = validate_torso_anchor(
                joints=joints,
                priors=priors,
                fallback_scale=fallback_scale,
                previous_anchor=previous_anchors.get(identity_id),
                args=args,
            )
            anchor_scores.append(float(anchor_diag["score"]))
            anchor_status_counts[str(anchor_diag["status"])] += 1
            if center is not None and anchor_diag["status"] in {"high", "medium", "fallback_previous"}:
                previous_anchors[identity_id] = {
                    "center": center.tolist(),
                    "scale": float(torso_scale),
                    "status": str(anchor_diag["status"]),
                }
            root_distance_hard_check = anchor_diag["status"] in {"high", "medium", "fallback_previous"}
            root_distance_threshold = float(args.max_root_distance_ratio)
            if anchor_diag["status"] == "medium":
                root_distance_threshold *= 1.35
            elif anchor_diag["status"] == "fallback_previous":
                root_distance_threshold *= 1.6

            valid_scores = []
            for joint_id, joint in joints.items():
                joint_point = point(joint)
                reasons = []
                score_terms = []

                parent_id = PARENT_BY_JOINT.get(joint_id)
                bone_ratio = None
                if parent_id is not None and parent_id in joints:
                    prior_length = bone_prior_for(priors, parent_id, joint_id)
                    current_length = float(np.linalg.norm(joint_point - point(joints[parent_id])))
                    if prior_length is not None and prior_length > 1e-9:
                        bone_ratio = current_length / float(prior_length)
                        if bone_ratio < float(args.bone_ratio_min) or bone_ratio > float(args.bone_ratio_max):
                            reasons.append("bone_ratio_outlier")
                        bone_score = 1.0 / (1.0 + abs(bone_ratio - 1.0))
                        score_terms.append(float(np.clip(bone_score, 0.0, 1.0)))

                root_distance_ratio = None
                if center is not None:
                    root_distance_ratio = float(np.linalg.norm(joint_point - center) / float(torso_scale))
                    if root_distance_hard_check and root_distance_ratio > root_distance_threshold:
                        reasons.append("root_distance_outlier")
                    root_score = 1.0 / (1.0 + max(0.0, root_distance_ratio - 1.0))
                    root_score *= max(0.25, float(anchor_diag["score"]))
                    score_terms.append(float(np.clip(root_score, 0.0, 1.0)))

                temporal_jump_ratio = None
                previous = previous_points.get((identity_id, joint_id))
                if previous is not None:
                    temporal_jump_ratio = float(np.linalg.norm(joint_point - previous) / float(torso_scale))
                    if temporal_jump_ratio > float(args.max_temporal_jump_ratio):
                        reasons.append("temporal_jump_outlier")
                    temporal_score = 1.0 / (1.0 + temporal_jump_ratio)
                    score_terms.append(float(np.clip(temporal_score, 0.0, 1.0)))

                reprojection_error = joint.get("mean_reprojection_error_px")
                if reprojection_error is not None:
                    reprojection_score = 1.0 / (1.0 + float(reprojection_error) / 25.0)
                    score_terms.append(float(np.clip(reprojection_score, 0.0, 1.0)))

                pose_score = float(np.mean(score_terms)) if score_terms else 1.0
                is_valid = len(reasons) == 0
                joint["pose_level_validation"] = {
                    "is_valid": bool(is_valid),
                    "score": pose_score,
                    "reasons": reasons,
                    "bone_ratio": bone_ratio,
                    "root_distance_ratio": root_distance_ratio,
                    "root_distance_threshold": root_distance_threshold,
                    "temporal_jump_ratio": temporal_jump_ratio,
                    "torso_scale": float(torso_scale),
                    "torso_anchor_score": float(anchor_diag["score"]),
                    "torso_anchor_status": str(anchor_diag["status"]),
                }
                previous_points[(identity_id, joint_id)] = joint_point
                valid_scores.append(pose_score)
                joint_rows[joint_id]["count"] += 1
                joint_rows[joint_id]["score_sum"] += pose_score
                if not is_valid:
                    joint_rows[joint_id]["invalid"] += 1
                    for reason in reasons:
                        invalid_by_reason[reason] += 1

            identity["pose_level_validation"] = {
                "score": float(np.mean(valid_scores)) if valid_scores else None,
                "num_joints": len(joints),
                "num_invalid_joints": sum(
                    1
                    for joint in joints.values()
                    if not joint.get("pose_level_validation", {}).get("is_valid", True)
                ),
                "torso_anchor": anchor_diag,
            }
            if identity["pose_level_validation"]["score"] is not None:
                pose_scores.append(float(identity["pose_level_validation"]["score"]))

    joint_summary = []
    for joint_id, row in sorted(joint_rows.items()):
        count = int(row["count"])
        joint_summary.append(
            {
                "joint_id": joint_id,
                "count": count,
                "invalid": int(row["invalid"]),
                "invalid_rate": float(row["invalid"] / count) if count else 0.0,
                "mean_pose_score": float(row["score_sum"] / count) if count else None,
            }
        )

    report = {
        "stage": "pose_level_candidate_validation",
        "settings": {
            "bone_ratio_min": float(args.bone_ratio_min),
            "bone_ratio_max": float(args.bone_ratio_max),
            "max_root_distance_ratio": float(args.max_root_distance_ratio),
            "max_temporal_jump_ratio": float(args.max_temporal_jump_ratio),
            "torso_high_threshold": float(args.torso_high_threshold),
            "torso_medium_threshold": float(args.torso_medium_threshold),
        },
        "summary": {
            "num_frames": len(output.get("frames", [])),
            "mean_pose_score": float(np.mean(pose_scores)) if pose_scores else None,
            "mean_torso_anchor_score": float(np.mean(anchor_scores)) if anchor_scores else None,
            "torso_anchor_status_counts": dict(anchor_status_counts),
            "invalid_by_reason": dict(invalid_by_reason),
        },
        "joint_summary": joint_summary,
        "priors": {
            str(identity_id): {
                "torso_scale": priors.get("torso_scale"),
                "num_bone_priors": len(priors.get("bone_lengths", {})),
            }
            for identity_id, priors in priors_by_identity.items()
        },
    }
    return output, report


def write_filtered_review_json(validated: dict[str, Any], output_path: Path) -> None:
    filtered = json.loads(json.dumps(validated))
    for frame in filtered.get("frames", []):
        for identity in frame.get("identities", []):
            identity["joints"] = [
                joint
                for joint in identity.get("joints", [])
                if joint.get("pose_level_validation", {}).get("is_valid", True)
            ]
            identity["num_joints"] = len(identity["joints"])
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    payload = load_json(args.input_json)
    priors = estimate_identity_priors(payload, min_samples=int(args.min_bone_samples))
    validated, report = validate_payload(payload, priors, args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pose_level_validated_3d.json").open("w", encoding="utf-8") as handle:
        json.dump(validated, handle, ensure_ascii=False, indent=2)
    with (output_dir / "pose_level_validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if args.drop_invalid_joints:
        write_filtered_review_json(validated, output_dir / "pose_level_validated_filtered_review.json")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("Top invalid joints:")
    for row in sorted(report["joint_summary"], key=lambda item: item["invalid_rate"], reverse=True)[:8]:
        print(row)


if __name__ == "__main__":
    main()
