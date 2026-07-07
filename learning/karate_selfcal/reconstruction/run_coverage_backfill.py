"""Backfill missing joints after reliable pose-hypothesis triangulation.

This stage is intended for coverage experiments. It preserves reliable joints
from an existing triangulated JSON and only attempts fallback triangulation for
missing joints. Fallback joints are explicitly marked so accuracy and coverage
can be evaluated separately.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..selfcal.hypothesis_geometry import build_track_frame_index
from .run_robust_all_joint_triangulation import collect_joint_observations, ransac_candidates
from .run_triangulation import build_fundamental_by_pair, filter_observations_by_geometry


COCO_NUM_JOINTS = 17
HEAD_JOINTS = {0, 1, 2, 3, 4}
RISKY_TWO_VIEW_SETS = {
    ("karate004_cam02", "karate004_cam13"),
    ("karate004_cam03", "karate004_cam07"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing triangulated joints with lower-confidence fallback candidates."
    )
    parser.add_argument("--input-json", required=True, help="Reliable triangulated_3d.json")
    parser.add_argument("--selected-hypothesis-json", required=True)
    parser.add_argument("--extrinsics-json", required=True)
    parser.add_argument("--track-jsons", nargs="+", required=True)
    parser.add_argument("--view-ids", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--geometry-sampson-threshold", type=float, default=2.5)
    parser.add_argument("--min-pair-support", type=int, default=1)
    parser.add_argument("--fallback-ransac-threshold-px", type=float, default=35.0)
    parser.add_argument("--fallback-max-mean-reprojection-px", type=float, default=12.0)
    parser.add_argument("--fallback-max-reprojection-px", type=float, default=35.0)
    parser.add_argument("--fallback-min-confidence", type=float, default=0.25)
    parser.add_argument("--fallback-min-angle-deg", type=float, default=0.2)
    parser.add_argument("--head-max-mean-reprojection-px", type=float, default=6.0)
    parser.add_argument("--head-min-views", type=int, default=3)
    parser.add_argument("--risky-two-view-max-mean-reprojection-px", type=float, default=4.0)
    parser.add_argument("--risky-two-view-min-confidence", type=float, default=0.45)
    parser.add_argument("--risky-two-view-min-angle-deg", type=float, default=1.0)
    parser.add_argument("--max-point-norm", type=float, default=8.0)
    parser.add_argument(
        "--allow-raw-if-geometry-empty",
        action="store_true",
        help="Use raw observations when epipolar filtering removes too much support.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def joint_key(identity: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(joint["id"]): dict(joint) for joint in identity.get("joints", [])}


def candidate_score(candidate: dict[str, Any]) -> float:
    reproj = float(candidate["mean_pixel_reprojection_error"])
    max_reproj = float(candidate["max_pixel_reprojection_error"])
    confidence = float(candidate["mean_confidence"])
    num_views = int(candidate["num_inlier_views"])
    angle = float(candidate.get("min_triangulation_angle_deg") or 0.0)
    return (
        0.24 * (1.0 / (1.0 + reproj / 8.0))
        + 0.14 * (1.0 / (1.0 + max_reproj / 20.0))
        + 0.24 * confidence
        + 0.20 * min(num_views / 3.0, 1.0)
        + 0.18 * min(angle / 5.0, 1.0)
    )


def choose_fallback_candidate(
    joint_id: int,
    observations: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    candidates = ransac_candidates(
        observations=observations,
        min_views=int(args.min_views),
        ransac_reprojection_threshold=float(args.fallback_ransac_threshold_px),
        max_point_norm=float(args.max_point_norm),
    )
    scored = []
    for tri, final_observations, diagnostics in candidates:
        mean_reproj = float(diagnostics["mean_pixel_reprojection_error"])
        max_reproj = float(diagnostics["max_pixel_reprojection_error"])
        mean_conf = float(diagnostics["mean_confidence"])
        min_angle = float(diagnostics.get("min_triangulation_angle_deg") or 0.0)
        viewset = tuple(sorted(str(obs["view_id"]) for obs in final_observations))
        is_head = int(joint_id) in HEAD_JOINTS
        is_risky_two_view = viewset in RISKY_TWO_VIEW_SETS
        if mean_reproj > float(args.fallback_max_mean_reprojection_px):
            continue
        if max_reproj > float(args.fallback_max_reprojection_px):
            continue
        if mean_conf < float(args.fallback_min_confidence):
            continue
        if min_angle < float(args.fallback_min_angle_deg):
            continue
        if is_head and len(final_observations) < int(args.head_min_views):
            continue
        if is_head and mean_reproj > float(args.head_max_mean_reprojection_px):
            continue
        if is_risky_two_view:
            if mean_reproj > float(args.risky_two_view_max_mean_reprojection_px):
                continue
            if mean_conf < float(args.risky_two_view_min_confidence):
                continue
            if min_angle < float(args.risky_two_view_min_angle_deg):
                continue
        scored.append((candidate_score(diagnostics), tri, final_observations, diagnostics))

    if not scored:
        return None, [], None
    scored.sort(key=lambda item: item[0], reverse=True)
    _, tri, final_observations, diagnostics = scored[0]
    return tri, final_observations, diagnostics


def summarize_coverage(payload: dict[str, Any], expected_joints: int = COCO_NUM_JOINTS) -> dict[str, Any]:
    frames = payload.get("frames", [])
    identity_ids = sorted(
        {
            int(identity["identity_id"])
            for frame in frames
            for identity in frame.get("identities", [])
        }
    )
    total_expected = len(frames) * max(len(identity_ids), 1) * int(expected_joints)
    source_counts = Counter()
    joint_counts = Counter()
    identity_counts = Counter()
    for frame in frames:
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            for joint in identity.get("joints", []):
                source = str(joint.get("source", joint.get("review_source", "reliable")))
                source_counts[source] += 1
                joint_counts[int(joint["id"])] += 1
                identity_counts[identity_id] += 1
    observed = int(sum(joint_counts.values()))
    return {
        "num_frames": len(frames),
        "identity_ids": identity_ids,
        "expected_joints": int(expected_joints),
        "total_expected_joints": int(total_expected),
        "observed_joints": observed,
        "observed_ratio": float(observed / total_expected) if total_expected else None,
        "source_counts": dict(source_counts),
        "joint_presence": {
            str(joint_id): {
                "count": int(joint_counts[joint_id]),
                "ratio": float(joint_counts[joint_id] / (len(frames) * max(len(identity_ids), 1)))
                if frames and identity_ids
                else None,
            }
            for joint_id in range(expected_joints)
        },
        "identity_presence": {
            str(identity_id): {
                "count": int(identity_counts[identity_id]),
                "ratio": float(identity_counts[identity_id] / (len(frames) * int(expected_joints)))
                if frames
                else None,
            }
            for identity_id in identity_ids
        },
    }


def run_backfill(
    reliable_payload: dict[str, Any],
    selected_hypothesis: dict[str, Any],
    extrinsics: dict[str, Any],
    track_payloads: dict[str, dict[str, Any]],
    view_ids: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame_indexes = {view_id: build_track_frame_index(track_payloads[view_id]) for view_id in view_ids}
    groups = list(selected_hypothesis.get("groups", []))
    groups_by_id = {int(group["group_id"]): group for group in groups}
    extrinsics_by_view = extrinsics.get("extrinsics_by_view", {})
    resolved_views = {view_id for view_id in extrinsics_by_view}
    fundamental_by_pair = build_fundamental_by_pair(selected_hypothesis)

    original_summary = summarize_coverage(reliable_payload)
    output_frames = []
    attempts = 0
    accepted = 0
    reject_reasons = Counter()
    fallback_viewsets = Counter()

    for frame in reliable_payload.get("frames", []):
        frame_idx = int(frame["frame"])
        identities_out = []
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            group = groups_by_id.get(identity_id)
            if group is None:
                identities_out.append(identity)
                continue

            joints = joint_key(identity)
            for joint in joints.values():
                joint.setdefault("source", "reliable")
                joint.setdefault("review_source", "reliable")

            missing_ids = [joint_id for joint_id in range(COCO_NUM_JOINTS) if joint_id not in joints]
            if missing_ids:
                observations_by_joint = collect_joint_observations(
                    frame_idx=frame_idx,
                    group=group,
                    frame_indexes=frame_indexes,
                    extrinsics_by_view=extrinsics_by_view,
                    resolved_views=resolved_views,
                    min_confidence=float(args.min_confidence),
                )
            else:
                observations_by_joint = {}

            for joint_id in missing_ids:
                observations = observations_by_joint.get(joint_id, [])
                if len(observations) < int(args.min_views):
                    reject_reasons["not_enough_observations"] += 1
                    continue
                geometry_filtered, pair_checks = filter_observations_by_geometry(
                    observations=observations,
                    fundamental_by_pair=fundamental_by_pair,
                    sampson_threshold=float(args.geometry_sampson_threshold),
                    min_pair_support=int(args.min_pair_support),
                )
                selected_observations = geometry_filtered
                if len(selected_observations) < int(args.min_views) and bool(args.allow_raw_if_geometry_empty):
                    selected_observations = observations
                if len(selected_observations) < int(args.min_views):
                    reject_reasons["not_enough_geometry_support"] += 1
                    continue

                attempts += 1
                tri, final_observations, diagnostics = choose_fallback_candidate(joint_id, selected_observations, args)
                if tri is None or diagnostics is None:
                    reject_reasons["no_accepted_candidate"] += 1
                    continue
                point = np.asarray(tri["point_3d"], dtype=np.float64)
                views = [str(obs["view_id"]) for obs in final_observations]
                fallback_viewsets[tuple(sorted(views))] += 1
                accepted += 1
                joints[joint_id] = {
                    "id": int(joint_id),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "num_views": int(len(final_observations)),
                    "mean_confidence": float(diagnostics["mean_confidence"]),
                    "mean_reprojection_error_px": float(diagnostics["mean_pixel_reprojection_error"]),
                    "max_reprojection_error_px": float(diagnostics["max_pixel_reprojection_error"]),
                    "views": views,
                    "source": "fallback",
                    "review_source": "fallback",
                    "fallback_diagnostics": {
                        **diagnostics,
                        "geometry_pair_checks": pair_checks,
                    },
                }

            joints_out = [joints[joint_id] for joint_id in sorted(joints)]
            identities_out.append(
                {
                    **{key: value for key, value in identity.items() if key != "joints"},
                    "num_joints": len(joints_out),
                    "joints": joints_out,
                }
            )
        output_frames.append({"frame": frame_idx, "identities": identities_out})

    metadata = dict(reliable_payload.get("metadata", {}))
    metadata["stage"] = "coverage_backfill"
    metadata["coverage_backfill_note"] = (
        "Reliable joints are preserved. Missing joints are backfilled with relaxed triangulation "
        "and marked as source=fallback."
    )
    output_payload = {
        "metadata": metadata,
        "frames": output_frames,
    }
    backfilled_summary = summarize_coverage(output_payload)
    report = {
        "stage": "coverage_backfill",
        "original": original_summary,
        "backfilled": backfilled_summary,
        "attempted_missing_joints": int(attempts),
        "accepted_fallback_joints": int(accepted),
        "fallback_acceptance_rate": float(accepted / attempts) if attempts else None,
        "reject_reasons": dict(reject_reasons),
        "fallback_viewsets": [
            {"views": list(viewset), "count": int(count)}
            for viewset, count in fallback_viewsets.most_common()
        ],
    }
    output_payload["summary"] = report
    return output_payload, report


def main() -> None:
    args = parse_args()
    if len(args.track_jsons) != len(args.view_ids):
        raise ValueError("track-jsons and view-ids must have the same length.")

    reliable_payload = load_json(args.input_json)
    selected_payload = load_json(args.selected_hypothesis_json)
    selected_hypothesis = selected_payload.get("selected_hypothesis", selected_payload)
    extrinsics = load_json(args.extrinsics_json)
    track_payloads = {view_id: load_json(path) for view_id, path in zip(args.view_ids, args.track_jsons)}

    output_payload, report = run_backfill(
        reliable_payload=reliable_payload,
        selected_hypothesis=selected_hypothesis,
        extrinsics=extrinsics,
        track_payloads=track_payloads,
        view_ids=list(args.view_ids),
        args=args,
    )

    output_path = Path(args.output_json).resolve()
    report_path = Path(args.report_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
