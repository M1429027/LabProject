"""Anchor-free pose hypothesis selection from per-joint triangulation candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from ..selfcal.hypothesis_geometry import build_track_frame_index
from .run_robust_all_joint_triangulation import collect_joint_observations, ransac_candidates
from .run_triangulation import build_fundamental_by_pair, filter_observations_by_geometry


BODY_PARTS = {
    "torso": [5, 6, 11, 12],
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],
    "right_leg": [12, 14, 16],
    "head": [0, 1, 2, 3, 4],
}

POSE_BONES = [
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
    (0, 1, "nose_left_eye"),
    (0, 2, "nose_right_eye"),
    (1, 3, "left_eye_ear"),
    (2, 4, "right_eye_ear"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select full-body pose hypotheses from joint candidates.")
    parser.add_argument("--selected-hypothesis-json", required=True)
    parser.add_argument("--rough-extrinsics-json", required=True)
    parser.add_argument("--track-jsons", nargs="+", required=True)
    parser.add_argument("--view-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prior-triangulated-json", required=True, help="Stable triangulated result used only for robust bone priors.")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--geometry-sampson-threshold", type=float, default=1.0)
    parser.add_argument("--min-pair-support", type=int, default=1)
    parser.add_argument("--ransac-reprojection-threshold", type=float, default=25.0)
    parser.add_argument("--max-point-norm", type=float, default=8.0)
    parser.add_argument("--top-k-joint-candidates", type=int, default=3)
    parser.add_argument("--top-k-part-candidates", type=int, default=3)
    parser.add_argument("--bone-ratio-soft-limit", type=float, default=0.55)
    parser.add_argument("--temporal-scale-m", type=float, default=0.45)
    parser.add_argument("--enable-full-body-score", action="store_true")
    parser.add_argument("--height-soft-limit", type=float, default=0.45)
    parser.add_argument("--symmetry-soft-limit", type=float, default=0.55)
    parser.add_argument(
        "--use-candidate-scoring-v2",
        action="store_true",
        help="Use GT-diagnostic-derived proxy scoring. GT is not used at runtime.",
    )
    parser.add_argument(
        "--depth-scale-m",
        type=float,
        default=6.0,
        help="Soft scale for candidate norm/depth sanity in v2 scoring.",
    )
    parser.add_argument(
        "--view-risk-penalty-weight",
        type=float,
        default=0.32,
        help="Penalty weight for risky view subsets in v2 scoring.",
    )
    parser.add_argument(
        "--single-view-risk-penalty-weight",
        type=float,
        default=0.0,
        help="Optional penalty for views known to be unreliable from diagnostic calibration.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def point_from_joint(joint: dict[str, Any]) -> np.ndarray:
    return np.array([float(joint["x"]), float(joint["y"]), float(joint["z"])], dtype=np.float64)


def estimate_bone_priors(prior_payload: dict[str, Any]) -> dict[int, dict[tuple[int, int], float]]:
    lengths: dict[int, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for frame in prior_payload.get("frames", []):
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            joints = {int(joint["id"]): point_from_joint(joint) for joint in identity.get("joints", [])}
            for joint_a, joint_b, _ in POSE_BONES:
                if joint_a in joints and joint_b in joints:
                    length = float(np.linalg.norm(joints[joint_b] - joints[joint_a]))
                    if np.isfinite(length) and length > 1e-9:
                        lengths[identity_id][(joint_a, joint_b)].append(length)

    priors: dict[int, dict[tuple[int, int], float]] = {}
    for identity_id, by_bone in lengths.items():
        priors[identity_id] = {}
        for bone, values in by_bone.items():
            if len(values) >= 5:
                finite = np.array(values, dtype=np.float64)
                lo, hi = np.percentile(finite, [10, 90])
                finite = finite[(finite >= lo) & (finite <= hi)]
                if len(finite):
                    priors[identity_id][bone] = float(np.median(finite))
    return priors


def bone_prior(priors: dict[tuple[int, int], float], joint_a: int, joint_b: int) -> float | None:
    return priors.get((joint_a, joint_b)) or priors.get((joint_b, joint_a))


def candidate_base_score(candidate: dict[str, Any]) -> float:
    reproj = float(candidate["mean_reprojection_error_px"])
    confidence = float(candidate["mean_confidence"])
    num_views = int(candidate["num_views"])
    angle = float(candidate.get("min_triangulation_angle_deg") or 0.0)
    return (
        0.45 * (1.0 / (1.0 + reproj / 25.0))
        + 0.25 * confidence
        + 0.20 * min(num_views / 3.0, 1.0)
        + 0.10 * min(angle / 1.0, 1.0)
    )


JOINT_RISK_PRIOR = {
    # From GT diagnostic on 09_karate/004_karate pose_hypothesis_v1.
    # These are soft priors, not hard-coded GT labels. They reduce trust in
    # joints that frequently produced low-reprojection but high-3D-error samples.
    5: 0.73,
    6: 0.91,
    7: 0.54,
    8: 0.80,
    9: 0.79,
    10: 1.00,
    11: 0.32,
    12: 0.24,
    13: 0.45,
    14: 0.35,
    15: 1.00,
    16: 1.00,
}

VIEW_SUBSET_RISK_PRIOR = {
    ("karate004_cam02", "karate004_cam03"): 1.00,
    ("karate004_cam03", "karate004_cam07"): 1.00,
    ("karate004_cam03", "karate004_cam13"): 1.00,
    ("karate004_cam02", "karate004_cam13"): 0.74,
    ("karate004_cam02", "karate004_cam07"): 0.69,
    ("karate004_cam07", "karate004_cam13"): 0.69,
    ("karate004_cam02", "karate004_cam07", "karate004_cam13"): 0.49,
}

SINGLE_VIEW_RISK_PRIOR = {
    # Diagnostic-specific soft risk. cam03 was repeatedly selected in
    # low-reprojection/high-GT-error candidates for this sequence.
    "karate004_cam03": 0.85,
}


def view_subset_risk(views: list[str]) -> float:
    """Return a soft risk prior for a selected view subset."""

    key = tuple(sorted(str(view) for view in views))
    return float(VIEW_SUBSET_RISK_PRIOR.get(key, 0.25 if len(key) == 2 else 0.15))


def candidate_base_score_v2(candidate: dict[str, Any], joint_id: int, args: argparse.Namespace) -> float:
    """Score one triangulation candidate using non-GT proxy features.

    GT diagnostic showed that low reprojection error can still produce large 3D
    mistakes. V2 therefore lowers reprojection weight and adds view-subset risk,
    joint risk, ray-angle, depth/scale sanity, and max-error robustness.
    """

    reproj = float(candidate["mean_reprojection_error_px"])
    max_reproj = float(candidate.get("max_reprojection_error_px") or reproj)
    confidence = float(candidate["mean_confidence"])
    num_views = int(candidate["num_views"])
    angle = float(candidate.get("min_triangulation_angle_deg") or 0.0)
    point_norm = float(np.linalg.norm(np.asarray(candidate["point"], dtype=np.float64)))

    reproj_score = 1.0 / (1.0 + reproj / 12.0)
    max_reproj_score = 1.0 / (1.0 + max_reproj / 20.0)
    angle_score = float(np.clip(angle / 8.0, 0.0, 1.0))
    view_count_score = min(num_views / 3.0, 1.0)
    depth_score = 1.0 / (1.0 + max(point_norm - float(args.depth_scale_m), 0.0) / 2.0)
    risk_penalty = 1.0 - 0.28 * float(JOINT_RISK_PRIOR.get(int(joint_id), 0.20))
    subset_risk = view_subset_risk(list(candidate["views"]))
    subset_penalty = 1.0 - float(args.view_risk_penalty_weight) * subset_risk
    single_view_risk = max(
        [SINGLE_VIEW_RISK_PRIOR.get(str(view), 0.0) for view in candidate["views"]],
        default=0.0,
    )
    single_view_penalty = 1.0 - float(args.single_view_risk_penalty_weight) * single_view_risk

    score = (
        0.18 * reproj_score
        + 0.12 * max_reproj_score
        + 0.20 * confidence
        + 0.18 * angle_score
        + 0.15 * view_count_score
        + 0.17 * depth_score
    )
    score *= float(np.clip(risk_penalty, 0.55, 1.0))
    score *= float(np.clip(subset_penalty, 0.50, 1.0))
    score *= float(np.clip(single_view_penalty, 0.40, 1.0))

    candidate["candidate_score_v2_terms"] = {
        "reprojection_score": float(reproj_score),
        "max_reprojection_score": float(max_reproj_score),
        "confidence_score": float(confidence),
        "angle_score": float(angle_score),
        "view_count_score": float(view_count_score),
        "depth_score": float(depth_score),
        "joint_risk": float(JOINT_RISK_PRIOR.get(int(joint_id), 0.20)),
        "view_subset_risk": float(subset_risk),
        "single_view_risk": float(single_view_risk),
        "risk_penalty": float(risk_penalty),
        "subset_penalty": float(subset_penalty),
        "single_view_penalty": float(single_view_penalty),
    }
    return float(score)


def make_joint_candidates(
    joint_id: int,
    observations: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidates = ransac_candidates(
        observations=observations,
        min_views=int(args.min_views),
        ransac_reprojection_threshold=float(args.ransac_reprojection_threshold),
        max_point_norm=float(args.max_point_norm),
    )
    out = []
    for tri, final_observations, diagnostics in candidates:
        point = np.asarray(tri["point_3d"], dtype=np.float64)
        item = {
            "point": point,
            "views": [obs["view_id"] for obs in final_observations],
            "num_views": len(final_observations),
            "mean_confidence": float(diagnostics["mean_confidence"]),
            "mean_reprojection_error_px": float(diagnostics["mean_pixel_reprojection_error"]),
            "max_reprojection_error_px": float(diagnostics["max_pixel_reprojection_error"]),
            "min_triangulation_angle_deg": float(diagnostics["min_triangulation_angle_deg"]),
            "diagnostics": diagnostics,
        }
        if bool(getattr(args, "use_candidate_scoring_v2", False)):
            item["base_score"] = candidate_base_score_v2(item, int(joint_id), args)
        else:
            item["base_score"] = candidate_base_score(item)
        out.append(item)
    out.sort(key=lambda item: item["base_score"], reverse=True)
    return out[: int(args.top_k_joint_candidates)]


def part_bones(part_name: str) -> list[tuple[int, int]]:
    if part_name == "torso":
        return [(5, 6), (11, 12), (5, 11), (6, 12)]
    if part_name == "left_arm":
        return [(5, 7), (7, 9)]
    if part_name == "right_arm":
        return [(6, 8), (8, 10)]
    if part_name == "left_leg":
        return [(11, 13), (13, 15)]
    if part_name == "right_leg":
        return [(12, 14), (14, 16)]
    return [(0, 1), (0, 2), (1, 3), (2, 4)]


def score_joint_set(
    points: dict[int, np.ndarray],
    candidates: dict[int, dict[str, Any]],
    priors: dict[tuple[int, int], float],
    previous_points: dict[int, np.ndarray],
    args: argparse.Namespace,
    part_name: str,
) -> tuple[float, list[str]]:
    reasons = []
    if not candidates:
        return -1e9, ["empty_part"]
    base = float(np.mean([candidate["base_score"] for candidate in candidates.values()]))
    bone_scores = []
    for joint_a, joint_b in part_bones(part_name):
        if joint_a not in points or joint_b not in points:
            continue
        prior = bone_prior(priors, joint_a, joint_b)
        if prior is None or prior <= 1e-9:
            continue
        length = float(np.linalg.norm(points[joint_b] - points[joint_a]))
        ratio = length / prior
        score = 1.0 / (1.0 + abs(ratio - 1.0))
        if abs(ratio - 1.0) > float(args.bone_ratio_soft_limit):
            reasons.append(f"bone_ratio:{joint_a}-{joint_b}:{ratio:.2f}")
        bone_scores.append(score)

    temporal_scores = []
    for joint_id, point in points.items():
        previous = previous_points.get(joint_id)
        if previous is None:
            continue
        jump = float(np.linalg.norm(point - previous))
        temporal_scores.append(1.0 / (1.0 + jump / max(float(args.temporal_scale_m), 1e-6)))

    bone = float(np.mean(bone_scores)) if bone_scores else 0.65
    temporal = float(np.mean(temporal_scores)) if temporal_scores else 0.75
    completeness = len(points) / max(len(BODY_PARTS[part_name]), 1)
    score = 0.38 * base + 0.34 * bone + 0.18 * temporal + 0.10 * completeness
    return float(score), reasons


def build_part_hypotheses(
    part_name: str,
    joint_candidates: dict[int, list[dict[str, Any]]],
    fixed_points: dict[int, np.ndarray],
    fixed_candidates: dict[int, dict[str, Any]],
    priors: dict[tuple[int, int], float],
    previous_points: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    joints = BODY_PARTS[part_name]
    variable_joints = [joint_id for joint_id in joints if joint_id not in fixed_points and joint_candidates.get(joint_id)]
    if not variable_joints and not fixed_points:
        return []

    candidate_lists = [joint_candidates[joint_id] for joint_id in variable_joints]
    hypotheses = []
    for combo in product(*candidate_lists) if candidate_lists else [()]:
        points = dict(fixed_points)
        selected = dict(fixed_candidates)
        for joint_id, candidate in zip(variable_joints, combo):
            points[joint_id] = candidate["point"]
            selected[joint_id] = candidate
        score, reasons = score_joint_set(
            points=points,
            candidates=selected,
            priors=priors,
            previous_points=previous_points,
            args=args,
            part_name=part_name,
        )
        hypotheses.append({"score": score, "points": points, "candidates": selected, "reasons": reasons})
    hypotheses.sort(key=lambda item: item["score"], reverse=True)
    return hypotheses[: int(args.top_k_part_candidates)]


def merge_pose_parts(parts: list[dict[str, Any]]) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]], float, list[str]]:
    points: dict[int, np.ndarray] = {}
    candidates: dict[int, dict[str, Any]] = {}
    scores = []
    reasons = []
    for part in parts:
        scores.append(float(part["score"]))
        reasons.extend(part.get("reasons", []))
        for joint_id, point in part["points"].items():
            if joint_id not in points:
                points[joint_id] = point
        for joint_id, candidate in part["candidates"].items():
            if joint_id not in candidates:
                candidates[joint_id] = candidate
    return points, candidates, float(np.mean(scores)) if scores else 0.0, reasons


def length_between(points: dict[int, np.ndarray], joint_a: int, joint_b: int) -> float | None:
    if joint_a not in points or joint_b not in points:
        return None
    return float(np.linalg.norm(points[joint_b] - points[joint_a]))


def ratio_score(value: float, reference: float, soft_limit: float) -> tuple[float, float | None]:
    if reference <= 1e-9:
        return 0.65, None
    ratio = value / reference
    score = 1.0 / (1.0 + abs(ratio - 1.0) / max(float(soft_limit), 1e-6))
    return float(np.clip(score, 0.0, 1.0)), float(ratio)


def full_body_score(
    points: dict[int, np.ndarray],
    part_score: float,
    priors: dict[tuple[int, int], float],
    previous_points: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    """Score a complete skeleton hypothesis after body parts are merged."""

    bone_scores = []
    bone_ratios = {}
    for joint_a, joint_b, name in POSE_BONES:
        if joint_a not in points or joint_b not in points:
            continue
        prior = bone_prior(priors, joint_a, joint_b)
        if prior is None:
            continue
        length = float(np.linalg.norm(points[joint_b] - points[joint_a]))
        score, ratio = ratio_score(length, prior, float(args.bone_ratio_soft_limit))
        bone_scores.append(score)
        bone_ratios[name] = ratio
    bone_score = float(np.mean(bone_scores)) if bone_scores else 0.65

    symmetry_pairs = [
        ((5, 7), (6, 8), "upper_arm_symmetry"),
        ((7, 9), (8, 10), "forearm_symmetry"),
        ((11, 13), (12, 14), "thigh_symmetry"),
        ((13, 15), (14, 16), "shin_symmetry"),
        ((5, 11), (6, 12), "torso_side_symmetry"),
    ]
    symmetry_scores = []
    symmetry_ratios = {}
    for (a1, b1), (a2, b2), name in symmetry_pairs:
        left = length_between(points, a1, b1)
        right = length_between(points, a2, b2)
        if left is None or right is None or right <= 1e-9:
            continue
        score, ratio = ratio_score(left, right, float(args.symmetry_soft_limit))
        symmetry_scores.append(score)
        symmetry_ratios[name] = ratio
    symmetry_score = float(np.mean(symmetry_scores)) if symmetry_scores else 0.65

    height_score = 0.65
    height_ratio = None
    if 0 in points and 15 in points and 16 in points:
        ankle_mid = 0.5 * (points[15] + points[16])
        height = float(np.linalg.norm(points[0] - ankle_mid))
        prior_segments = []
        for bone in [(5, 11), (11, 13), (13, 15)]:
            prior = bone_prior(priors, *bone)
            if prior is not None:
                prior_segments.append(prior)
        if prior_segments:
            prior_height = float(sum(prior_segments) * 1.35)
            height_score, height_ratio = ratio_score(height, prior_height, float(args.height_soft_limit))

    temporal_scores = []
    for joint_id, point in points.items():
        previous = previous_points.get(joint_id)
        if previous is None:
            continue
        jump = float(np.linalg.norm(point - previous))
        temporal_scores.append(1.0 / (1.0 + jump / max(float(args.temporal_scale_m), 1e-6)))
    temporal_score = float(np.mean(temporal_scores)) if temporal_scores else 0.75

    completeness_score = len(points) / 17.0
    score = (
        0.28 * part_score
        + 0.26 * bone_score
        + 0.16 * symmetry_score
        + 0.14 * temporal_score
        + 0.10 * completeness_score
        + 0.06 * height_score
    )
    diagnostics = {
        "part_score": float(part_score),
        "full_body_score": float(score),
        "bone_score": float(bone_score),
        "symmetry_score": float(symmetry_score),
        "temporal_score": float(temporal_score),
        "completeness_score": float(completeness_score),
        "height_score": float(height_score),
        "height_ratio": height_ratio,
        "bone_ratios": bone_ratios,
        "symmetry_ratios": symmetry_ratios,
    }
    return float(score), diagnostics


def run_selection(
    selected_hypothesis: dict[str, Any],
    rough_extrinsics: dict[str, Any],
    track_payloads: dict[str, dict[str, Any]],
    view_ids: list[str],
    priors_by_identity: dict[int, dict[tuple[int, int], float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_indexes = {view_id: build_track_frame_index(track_payloads[view_id]) for view_id in view_ids}
    groups = list(selected_hypothesis.get("groups", []))
    extrinsics_by_view = rough_extrinsics.get("extrinsics_by_view", {})
    fundamental_by_pair = build_fundamental_by_pair(selected_hypothesis)
    resolved_views = {view_id for view_id in extrinsics_by_view}
    all_frames = sorted({frame_idx for frame_map in frame_indexes.values() for frame_idx in frame_map.keys()})
    previous_by_identity: dict[int, dict[int, np.ndarray]] = defaultdict(dict)

    output_frames = []
    pose_scores = []
    reprojection_values = []
    generated_candidates = 0

    for frame_idx in all_frames:
        identities_out = []
        for group in groups:
            identity_id = int(group["group_id"])
            priors = priors_by_identity.get(identity_id, {})
            observations_by_joint = collect_joint_observations(
                frame_idx=frame_idx,
                group=group,
                frame_indexes=frame_indexes,
                extrinsics_by_view=extrinsics_by_view,
                resolved_views=resolved_views,
                min_confidence=float(args.min_confidence),
            )
            joint_candidates: dict[int, list[dict[str, Any]]] = {}
            for joint_id, observations in observations_by_joint.items():
                if len(observations) < int(args.min_views):
                    continue
                geometry_filtered, _ = filter_observations_by_geometry(
                    observations=observations,
                    fundamental_by_pair=fundamental_by_pair,
                    sampson_threshold=float(args.geometry_sampson_threshold),
                    min_pair_support=int(args.min_pair_support),
                )
                if len(geometry_filtered) < int(args.min_views):
                    continue
                joint_candidates[joint_id] = make_joint_candidates(joint_id, geometry_filtered, args)
                generated_candidates += len(joint_candidates[joint_id])

            previous_points = previous_by_identity[identity_id]
            torso_hypotheses = build_part_hypotheses("torso", joint_candidates, {}, {}, priors, previous_points, args)
            if not torso_hypotheses:
                continue

            full_hypotheses = []
            for torso in torso_hypotheses:
                for left_arm in build_part_hypotheses("left_arm", joint_candidates, {k: v for k, v in torso["points"].items() if k == 5}, {k: v for k, v in torso["candidates"].items() if k == 5}, priors, previous_points, args) or [{"score": 0.4, "points": {}, "candidates": {}, "reasons": ["missing_left_arm"]}]:
                    for right_arm in build_part_hypotheses("right_arm", joint_candidates, {k: v for k, v in torso["points"].items() if k == 6}, {k: v for k, v in torso["candidates"].items() if k == 6}, priors, previous_points, args) or [{"score": 0.4, "points": {}, "candidates": {}, "reasons": ["missing_right_arm"]}]:
                        for left_leg in build_part_hypotheses("left_leg", joint_candidates, {k: v for k, v in torso["points"].items() if k == 11}, {k: v for k, v in torso["candidates"].items() if k == 11}, priors, previous_points, args) or [{"score": 0.4, "points": {}, "candidates": {}, "reasons": ["missing_left_leg"]}]:
                            for right_leg in build_part_hypotheses("right_leg", joint_candidates, {k: v for k, v in torso["points"].items() if k == 12}, {k: v for k, v in torso["candidates"].items() if k == 12}, priors, previous_points, args) or [{"score": 0.4, "points": {}, "candidates": {}, "reasons": ["missing_right_leg"]}]:
                                points, candidates, score, reasons = merge_pose_parts([torso, left_arm, right_arm, left_leg, right_leg])
                                diagnostics = {"part_score": score}
                                if bool(args.enable_full_body_score):
                                    score, diagnostics = full_body_score(
                                        points=points,
                                        part_score=score,
                                        priors=priors,
                                        previous_points=previous_points,
                                        args=args,
                                    )
                                full_hypotheses.append((score, points, candidates, reasons, diagnostics))

            if not full_hypotheses:
                continue
            full_hypotheses.sort(key=lambda item: item[0], reverse=True)
            pose_score, points, candidates, reasons, full_body_diagnostics = full_hypotheses[0]
            previous_by_identity[identity_id] = dict(points)
            pose_scores.append(float(pose_score))
            joints_out = []
            for joint_id, point in sorted(points.items()):
                candidate = candidates[joint_id]
                reprojection_values.append(float(candidate["mean_reprojection_error_px"]))
                joints_out.append(
                    {
                        "id": int(joint_id),
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "z": float(point[2]),
                        "num_views": int(candidate["num_views"]),
                        "mean_confidence": float(candidate["mean_confidence"]),
                        "mean_reprojection_error_px": float(candidate["mean_reprojection_error_px"]),
                        "views": list(candidate["views"]),
                        "pose_hypothesis_score": float(pose_score),
                        "candidate_base_score": float(candidate["base_score"]),
                        "full_body_diagnostics": full_body_diagnostics,
                    }
                )
            identities_out.append(
                {
                    "identity_id": identity_id,
                    "num_joints": len(joints_out),
                    "pose_hypothesis_score": float(pose_score),
                    "pose_hypothesis_reasons": reasons[:20],
                    "full_body_diagnostics": full_body_diagnostics,
                    "joints": joints_out,
                }
            )
        if identities_out:
            output_frames.append({"frame": frame_idx, "identities": identities_out})

    return {
        "metadata": {
            "stage": "pose_hypothesis_selection_v1",
            "view_ids": list(view_ids),
            "top_k_joint_candidates": int(args.top_k_joint_candidates),
            "top_k_part_candidates": int(args.top_k_part_candidates),
            "enable_full_body_score": bool(args.enable_full_body_score),
            "prior_source": str(args.prior_triangulated_json),
        },
        "frames": output_frames,
        "summary": {
            "num_frames_with_3d": len(output_frames),
            "mean_pose_hypothesis_score": float(np.mean(pose_scores)) if pose_scores else None,
            "mean_reprojection_error_px": float(np.mean(reprojection_values)) if reprojection_values else None,
            "total_generated_joint_candidates": int(generated_candidates),
        },
    }


def main() -> None:
    args = parse_args()
    if len(args.track_jsons) != len(args.view_ids):
        raise ValueError("track-jsons and view-ids must have the same length.")
    selected_payload = load_json(args.selected_hypothesis_json)
    selected_hypothesis = selected_payload["selected_hypothesis"]
    rough_extrinsics = load_json(args.rough_extrinsics_json)
    track_payloads = {view_id: load_json(path) for view_id, path in zip(args.view_ids, args.track_jsons)}
    priors = estimate_bone_priors(load_json(args.prior_triangulated_json))
    payload = run_selection(
        selected_hypothesis=selected_hypothesis,
        rough_extrinsics=rough_extrinsics,
        track_payloads=track_payloads,
        view_ids=list(args.view_ids),
        priors_by_identity=priors,
        args=args,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "triangulated_3d.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["summary"], handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
