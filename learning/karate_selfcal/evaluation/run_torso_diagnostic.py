"""Run torso-first diagnostics on triangulated 3D skeleton outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TORSO_JOINTS = [0, 5, 6, 11, 12]
TORSO_BONES = {
    "shoulder_width_5_6": (5, 6),
    "hip_width_11_12": (11, 12),
    "left_torso_5_11": (5, 11),
    "right_torso_6_12": (6, 12),
    "diag_5_12": (5, 12),
    "diag_6_11": (6, 11),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Diagnose whether torso/core joints are stable before full-body refinement."
    )
    parser.add_argument("--input-json", required=True, help="Input triangulated/refined 3D JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--torso-joints",
        nargs="+",
        type=int,
        default=DEFAULT_TORSO_JOINTS,
        help="Joint ids used as torso/core diagnostic joints",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON payload."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload."""

    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def joint_map(identity: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return joint-id indexed joint dictionaries."""

    return {int(joint["id"]): joint for joint in identity.get("joints", [])}


def point_from_joint(joint: dict[str, Any]) -> np.ndarray:
    """Return xyz array from one joint dictionary."""

    return np.array([float(joint["x"]), float(joint["y"]), float(joint["z"])], dtype=np.float64)


def build_torso_payload(payload: dict[str, Any], torso_joints: set[int]) -> dict[str, Any]:
    """Keep only torso/core joints for visual inspection."""

    frames_out = []
    for frame in payload.get("frames", []):
        identities_out = []
        for identity in frame.get("identities", []):
            joints_out = [
                {**joint, "torso_diagnostic": True}
                for joint in identity.get("joints", [])
                if int(joint["id"]) in torso_joints
            ]
            if joints_out:
                identities_out.append(
                    {
                        "identity_id": int(identity["identity_id"]),
                        "num_joints": len(joints_out),
                        "joints": joints_out,
                    }
                )
        if identities_out:
            frames_out.append({"frame": int(frame["frame"]), "identities": identities_out})

    return {
        "metadata": {
            **dict(payload.get("metadata", {})),
            "stage": "torso_first_diagnostic",
            "torso_joints": sorted(torso_joints),
            "note": "Torso-only payload for diagnosing core body stability.",
        },
        "frames": frames_out,
    }


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    """Summarize a numeric sequence."""

    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "median": None, "mean": None, "std": None, "cv": None, "p95": None}
    median = float(np.median(finite))
    std = float(np.std(finite))
    return {
        "count": int(finite.size),
        "median": median,
        "mean": float(np.mean(finite)),
        "std": std,
        "cv": float(std / median) if abs(median) > 1e-9 else None,
        "p95": float(np.percentile(finite, 95)),
    }


def compute_temporal_stats(points_by_frame: dict[int, np.ndarray]) -> dict[str, Any]:
    """Compute velocity and acceleration diagnostics for a point track."""

    frames = sorted(points_by_frame)
    velocities = []
    accelerations = []
    for prev_frame, next_frame in zip(frames[:-1], frames[1:]):
        dt = max(next_frame - prev_frame, 1)
        velocities.append(float(np.linalg.norm(points_by_frame[next_frame] - points_by_frame[prev_frame]) / dt))
    for frame_a, frame_b, frame_c in zip(frames[:-2], frames[1:-1], frames[2:]):
        if frame_b - frame_a != frame_c - frame_b:
            continue
        acceleration = points_by_frame[frame_a] - 2.0 * points_by_frame[frame_b] + points_by_frame[frame_c]
        accelerations.append(float(np.linalg.norm(acceleration)))
    return {
        "velocity": summarize_values(velocities),
        "acceleration": summarize_values(accelerations),
    }


def compute_identity_metrics(
    frames: list[dict[str, Any]],
    identity_id: int,
    torso_joints: set[int],
) -> dict[str, Any]:
    """Compute torso stability metrics for one identity."""

    joint_tracks: dict[int, dict[int, np.ndarray]] = {joint_id: {} for joint_id in torso_joints}
    torso_centers: dict[int, np.ndarray] = {}
    bone_lengths: dict[str, list[float]] = {name: [] for name in TORSO_BONES}
    shoulder_vectors = []
    spine_vectors = []
    full_torso_frames = 0
    visible_joint_count = 0
    total_joint_slots = len(frames) * len(torso_joints)

    for frame in frames:
        frame_idx = int(frame["frame"])
        identity = next(
            (item for item in frame.get("identities", []) if int(item["identity_id"]) == int(identity_id)),
            None,
        )
        if identity is None:
            continue
        joints = joint_map(identity)
        present_points = []
        for joint_id in torso_joints:
            if joint_id not in joints:
                continue
            point = point_from_joint(joints[joint_id])
            joint_tracks[joint_id][frame_idx] = point
            present_points.append(point)
            visible_joint_count += 1
        if present_points:
            torso_centers[frame_idx] = np.mean(present_points, axis=0)
        if all(joint_id in joints for joint_id in torso_joints):
            full_torso_frames += 1
        for name, (joint_a, joint_b) in TORSO_BONES.items():
            if joint_a in joints and joint_b in joints:
                bone_lengths[name].append(float(np.linalg.norm(point_from_joint(joints[joint_b]) - point_from_joint(joints[joint_a]))))
        if 5 in joints and 6 in joints:
            shoulder_vectors.append(point_from_joint(joints[6]) - point_from_joint(joints[5]))
        if 5 in joints and 6 in joints and 11 in joints and 12 in joints:
            shoulder_mid = 0.5 * (point_from_joint(joints[5]) + point_from_joint(joints[6]))
            hip_mid = 0.5 * (point_from_joint(joints[11]) + point_from_joint(joints[12]))
            spine_vectors.append(shoulder_mid - hip_mid)

    def _direction_stability(vectors: list[np.ndarray]) -> dict[str, float | int | None]:
        unit_vectors = []
        for vector in vectors:
            norm = float(np.linalg.norm(vector))
            if norm > 1e-9 and np.all(np.isfinite(vector)):
                unit_vectors.append(vector / norm)
        if not unit_vectors:
            return {"count": 0, "mean_cosine_to_average": None}
        average = np.mean(unit_vectors, axis=0)
        average_norm = float(np.linalg.norm(average))
        if average_norm <= 1e-9:
            return {"count": len(unit_vectors), "mean_cosine_to_average": None}
        average = average / average_norm
        cosines = [float(np.dot(vector, average)) for vector in unit_vectors]
        return {
            "count": len(unit_vectors),
            "mean_cosine_to_average": float(np.mean(cosines)),
            "p10_cosine_to_average": float(np.percentile(cosines, 10)),
        }

    return {
        "identity_id": int(identity_id),
        "num_frames": len(frames),
        "torso_joint_coverage": float(visible_joint_count / total_joint_slots) if total_joint_slots else None,
        "full_torso_frame_ratio": float(full_torso_frames / len(frames)) if frames else None,
        "bone_lengths": {name: summarize_values(values) for name, values in bone_lengths.items()},
        "torso_center_temporal": compute_temporal_stats(torso_centers),
        "joint_temporal": {
            str(joint_id): compute_temporal_stats(track)
            for joint_id, track in sorted(joint_tracks.items())
            if track
        },
        "shoulder_direction_stability": _direction_stability(shoulder_vectors),
        "spine_direction_stability": _direction_stability(spine_vectors),
    }


def collect_identity_ids(payload: dict[str, Any]) -> list[int]:
    """Collect all identity ids from a payload."""

    ids = set()
    for frame in payload.get("frames", []):
        for identity in frame.get("identities", []):
            ids.add(int(identity["identity_id"]))
    return sorted(ids)


def main() -> None:
    """Run torso-first diagnostic."""

    args = parse_args()
    input_path = Path(args.input_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    payload = load_json(input_path)
    torso_joints = set(int(joint_id) for joint_id in args.torso_joints)

    torso_payload = build_torso_payload(payload, torso_joints=torso_joints)
    identity_ids = collect_identity_ids(payload)
    metrics = {
        "stage": "torso_first_diagnostic",
        "input_json": str(input_path),
        "torso_joints": sorted(torso_joints),
        "identity_metrics": {
            str(identity_id): compute_identity_metrics(payload.get("frames", []), identity_id, torso_joints)
            for identity_id in identity_ids
        },
    }

    write_json(output_dir / "torso_only_3d.json", torso_payload)
    write_json(output_dir / "torso_metrics.json", metrics)
    print(
        json.dumps(
            {
                "stage": "torso_first_diagnostic",
                "output_dir": str(output_dir),
                "torso_only_json": str(output_dir / "torso_only_3d.json"),
                "metrics_json": str(output_dir / "torso_metrics.json"),
                "identity_ids": identity_ids,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
