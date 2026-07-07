"""Strict torso-only reconstruction diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..reconstruction.run_triangulation import (
    build_fundamental_by_pair,
    build_track_frame_index,
    enrich_observation,
    filter_observations_by_geometry,
    keypoints_by_id,
    parse_group_nodes,
    select_best_observation_subset,
)
from .run_torso_diagnostic import TORSO_BONES, compute_identity_metrics, collect_identity_ids


TORSO_JOINTS = [0, 5, 6, 11, 12]
TORSO_BONE_EDGES = [(5, 6), (11, 12), (5, 11), (6, 12), (5, 12), (6, 11)]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Re-triangulate torso joints with strict observation and torso-feasibility filtering."
    )
    parser.add_argument("--selected-hypothesis-json", required=True)
    parser.add_argument("--rough-extrinsics-json", required=True)
    parser.add_argument("--track-jsons", nargs="+", required=True)
    parser.add_argument("--view-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--geometry-sampson-threshold", type=float, default=0.6)
    parser.add_argument("--min-pair-support", type=int, default=1)
    parser.add_argument("--max-pixel-reprojection-error", type=float, default=20.0)
    parser.add_argument("--min-triangulation-angle-deg", type=float, default=2.0)
    parser.add_argument("--bone-ratio-min", type=float, default=0.55)
    parser.add_argument("--bone-ratio-max", type=float, default=1.65)
    parser.add_argument("--min-prior-samples", type=int, default=8)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON."""

    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def point_from_joint(joint: dict[str, Any]) -> np.ndarray:
    """Return xyz from a joint dict."""

    return np.array([float(joint["x"]), float(joint["y"]), float(joint["z"])], dtype=np.float64)


def estimate_torso_priors(frames: list[dict[str, Any]], min_samples: int) -> dict[int, dict[tuple[int, int], float]]:
    """Estimate per-identity robust torso bone medians from a first-pass reconstruction."""

    samples: dict[int, dict[tuple[int, int], list[float]]] = {}
    for frame in frames:
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            joints = {int(joint["id"]): joint for joint in identity.get("joints", [])}
            samples.setdefault(identity_id, {edge: [] for edge in TORSO_BONE_EDGES})
            for edge in TORSO_BONE_EDGES:
                joint_a, joint_b = edge
                if joint_a not in joints or joint_b not in joints:
                    continue
                length = float(np.linalg.norm(point_from_joint(joints[joint_b]) - point_from_joint(joints[joint_a])))
                if np.isfinite(length) and length > 1e-9:
                    samples[identity_id][edge].append(length)

    priors: dict[int, dict[tuple[int, int], float]] = {}
    for identity_id, by_edge in samples.items():
        priors[identity_id] = {}
        for edge, values in by_edge.items():
            if len(values) < int(min_samples):
                continue
            values_np = np.asarray(values, dtype=np.float64)
            q1, q3 = np.percentile(values_np, [25, 75])
            iqr = q3 - q1
            if iqr > 1e-9:
                values_np = values_np[(values_np >= q1 - 1.5 * iqr) & (values_np <= q3 + 1.5 * iqr)]
            if len(values_np) >= int(min_samples):
                priors[identity_id][edge] = float(np.median(values_np))
    return priors


def triangulate_torso_pass(
    selected_hypothesis: dict[str, Any],
    rough_extrinsics: dict[str, Any],
    track_payloads: dict[str, dict[str, Any]],
    view_ids: list[str],
    args: argparse.Namespace,
    torso_priors: dict[int, dict[tuple[int, int], float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Triangulate torso joints with optional torso bone feasibility filtering."""

    frame_indexes = {
        view_id: build_track_frame_index(track_payloads[view_id])
        for view_id in view_ids
    }
    groups = list(selected_hypothesis.get("groups", []))
    extrinsics_by_view = rough_extrinsics.get("extrinsics_by_view", {})
    fundamental_by_pair = build_fundamental_by_pair(selected_hypothesis)
    resolved_views = {view_id for view_id in extrinsics_by_view}
    all_frames = sorted({
        frame_idx
        for frame_map in frame_indexes.values()
        for frame_idx in frame_map.keys()
    })[:: max(int(args.frame_step), 1)]

    output_frames = []
    rejected_bone_events = []
    total_candidates = 0
    total_geometry_kept = 0
    total_final = 0
    reprojection_values = []

    for frame_idx in all_frames:
        identities_out = []
        for group in groups:
            identity_id = int(group["group_id"])
            group_tracks = parse_group_nodes(group)
            joint_observations: dict[int, list[dict[str, Any]]] = {}
            for view_id, track_id in group_tracks.items():
                if view_id not in resolved_views:
                    continue
                person = frame_indexes.get(view_id, {}).get(frame_idx, {}).get(track_id)
                if person is None:
                    continue
                visible = keypoints_by_id(person, min_confidence=float(args.min_confidence))
                for joint_id in TORSO_JOINTS:
                    point = visible.get(joint_id)
                    if point is None:
                        continue
                    joint_observations.setdefault(joint_id, []).append(
                        enrich_observation(view_id=view_id, point=point, extrinsics_by_view=extrinsics_by_view)
                    )

            joints_out = []
            for joint_id, observations in sorted(joint_observations.items()):
                total_candidates += len(observations)
                if len(observations) < int(args.min_views):
                    continue
                geometry_filtered, geometry_pairs = filter_observations_by_geometry(
                    observations=observations,
                    fundamental_by_pair=fundamental_by_pair,
                    sampson_threshold=float(args.geometry_sampson_threshold),
                    min_pair_support=int(args.min_pair_support),
                )
                total_geometry_kept += len(geometry_filtered)
                if len(geometry_filtered) < int(args.min_views):
                    continue
                tri, final_observations, subset_diagnostics = select_best_observation_subset(
                    observations=geometry_filtered,
                    min_views=int(args.min_views),
                    max_pixel_reprojection_error=float(args.max_pixel_reprojection_error),
                    min_triangulation_angle_deg=float(args.min_triangulation_angle_deg),
                )
                total_final += len(final_observations)
                if tri.get("status") != "ok":
                    continue
                errors = [float(err) for err in tri.get("pixel_reprojection_errors", []) if np.isfinite(err)]
                mean_error = float(np.mean(errors)) if errors else None
                if mean_error is not None:
                    reprojection_values.append(mean_error)
                point_3d = np.asarray(tri["point_3d"], dtype=np.float64)
                joints_out.append(
                    {
                        "id": int(joint_id),
                        "x": float(point_3d[0]),
                        "y": float(point_3d[1]),
                        "z": float(point_3d[2]),
                        "num_views": len(final_observations),
                        "candidate_num_views": len(observations),
                        "geometry_filtered_num_views": len(geometry_filtered),
                        "mean_confidence": float(np.mean([obs["confidence"] for obs in final_observations])),
                        "mean_reprojection_error_px": mean_error,
                        "views": [obs["view_id"] for obs in final_observations],
                        "geometry_pair_checks": geometry_pairs,
                        "triangulation_angles_deg": tri.get("triangulation_angles_deg"),
                        "view_subset_quality": subset_diagnostics,
                        "torso_strict_reconstruction": True,
                    }
                )

            if torso_priors:
                joints_out, events = filter_by_torso_bone_priors(
                    frame_idx=frame_idx,
                    identity_id=identity_id,
                    joints_out=joints_out,
                    priors=torso_priors.get(identity_id, {}),
                    ratio_min=float(args.bone_ratio_min),
                    ratio_max=float(args.bone_ratio_max),
                )
                rejected_bone_events.extend(events)

            if joints_out:
                identities_out.append(
                    {
                        "identity_id": identity_id,
                        "num_joints": len(joints_out),
                        "joints": sorted(joints_out, key=lambda item: int(item["id"])),
                    }
                )
        if identities_out:
            output_frames.append({"frame": int(frame_idx), "identities": identities_out})

    summary = {
        "num_frames_with_torso": len(output_frames),
        "total_candidate_observations": total_candidates,
        "total_geometry_kept_observations": total_geometry_kept,
        "total_final_observations": total_final,
        "geometry_keep_ratio": float(total_geometry_kept / total_candidates) if total_candidates else None,
        "final_keep_ratio": float(total_final / total_candidates) if total_candidates else None,
        "mean_reprojection_error_px": float(np.mean(reprojection_values)) if reprojection_values else None,
        "num_bone_rejection_events": len(rejected_bone_events),
        "bone_rejection_events": rejected_bone_events,
    }
    return output_frames, summary


def filter_by_torso_bone_priors(
    frame_idx: int,
    identity_id: int,
    joints_out: list[dict[str, Any]],
    priors: dict[tuple[int, int], float],
    ratio_min: float,
    ratio_max: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop lower-quality endpoints attached to implausible torso bones."""

    joints = {int(joint["id"]): joint for joint in joints_out}
    events = []
    dropped: set[int] = set()
    for joint_a, joint_b in TORSO_BONE_EDGES:
        target = priors.get((joint_a, joint_b))
        if target is None or target <= 1e-9:
            continue
        if joint_a not in joints or joint_b not in joints:
            continue
        point_a = point_from_joint(joints[joint_a])
        point_b = point_from_joint(joints[joint_b])
        length = float(np.linalg.norm(point_b - point_a))
        ratio = length / float(target)
        if ratio_min <= ratio <= ratio_max:
            continue
        quality_a = joint_quality(joints[joint_a])
        quality_b = joint_quality(joints[joint_b])
        drop_joint = joint_a if quality_a < quality_b else joint_b
        dropped.add(drop_joint)
        events.append(
            {
                "frame": int(frame_idx),
                "identity_id": int(identity_id),
                "joint_a": int(joint_a),
                "joint_b": int(joint_b),
                "dropped_joint": int(drop_joint),
                "length": length,
                "target_length": float(target),
                "ratio": ratio,
            }
        )
    return [joint for joint in joints_out if int(joint["id"]) not in dropped], events


def joint_quality(joint: dict[str, Any]) -> float:
    """Higher is better for deciding which endpoint to keep."""

    confidence = float(joint.get("mean_confidence") or 0.0)
    views = float(joint.get("num_views") or 1.0)
    error = joint.get("mean_reprojection_error_px")
    error_weight = 1.0 / (1.0 + max(float(error or 0.0), 0.0) / 10.0)
    return confidence * min(views / 4.0, 1.0) * error_weight


def main() -> None:
    """Run torso-only strict reconstruction."""

    args = parse_args()
    if len(args.track_jsons) != len(args.view_ids):
        raise ValueError("track-jsons and view-ids must have the same length.")

    selected_payload = load_json(args.selected_hypothesis_json)
    selected_hypothesis = selected_payload["selected_hypothesis"]
    rough_extrinsics = load_json(args.rough_extrinsics_json)
    track_payloads = {
        view_id: load_json(track_json)
        for view_id, track_json in zip(args.view_ids, args.track_jsons)
    }

    first_pass_frames, first_pass_summary = triangulate_torso_pass(
        selected_hypothesis=selected_hypothesis,
        rough_extrinsics=rough_extrinsics,
        track_payloads=track_payloads,
        view_ids=list(args.view_ids),
        args=args,
        torso_priors=None,
    )
    torso_priors = estimate_torso_priors(first_pass_frames, min_samples=int(args.min_prior_samples))
    strict_frames, strict_summary = triangulate_torso_pass(
        selected_hypothesis=selected_hypothesis,
        rough_extrinsics=rough_extrinsics,
        track_payloads=track_payloads,
        view_ids=list(args.view_ids),
        args=args,
        torso_priors=torso_priors,
    )
    output_payload = {
        "metadata": {
            "stage": "torso_only_strict_reconstruction",
            "view_ids": list(args.view_ids),
            "torso_joints": TORSO_JOINTS,
            "min_confidence": float(args.min_confidence),
            "geometry_sampson_threshold": float(args.geometry_sampson_threshold),
            "max_pixel_reprojection_error": float(args.max_pixel_reprojection_error),
            "min_triangulation_angle_deg": float(args.min_triangulation_angle_deg),
            "bone_ratio_min": float(args.bone_ratio_min),
            "bone_ratio_max": float(args.bone_ratio_max),
            "torso_priors": {
                str(identity_id): {f"{a}_{b}": value for (a, b), value in priors.items()}
                for identity_id, priors in torso_priors.items()
            },
        },
        "frames": strict_frames,
        "summary": {
            "first_pass": first_pass_summary,
            "strict_pass": strict_summary,
        },
    }

    identity_ids = collect_identity_ids(output_payload)
    torso_metrics = {
        "stage": "torso_only_strict_reconstruction",
        "identity_metrics": {
            str(identity_id): compute_identity_metrics(strict_frames, identity_id, set(TORSO_JOINTS))
            for identity_id in identity_ids
        },
        "summary": output_payload["summary"],
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "torso_strict_3d.json", output_payload)
    write_json(output_dir / "torso_strict_metrics.json", torso_metrics)
    print(
        json.dumps(
            {
                "stage": "torso_only_strict_reconstruction",
                "output_dir": str(output_dir),
                "torso_strict_json": str(output_dir / "torso_strict_3d.json"),
                "metrics_json": str(output_dir / "torso_strict_metrics.json"),
                "summary": output_payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
