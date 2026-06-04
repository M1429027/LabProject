"""Apply a soft bone-length prior to completed triangulated skeletons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


COCO_SKELETON_TREE = [
    (11, 12),
    (11, 5),
    (12, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 0),
    (6, 0),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

ROOT_JOINT = 11


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Softly regularize completed 3D skeletons with median bone-length priors."
    )
    parser.add_argument("--input-json", required=True, help="Input completed triangulated_3d JSON")
    parser.add_argument("--output-json", required=True, help="Output bone-refined JSON")
    parser.add_argument("--metrics-json", required=True, help="Output bone prior metrics JSON")
    parser.add_argument("--alpha", type=float, default=0.35, help="Soft correction strength in [0, 1]")
    parser.add_argument("--min-bone-samples", type=int, default=10, help="Minimum samples needed for a bone prior")
    parser.add_argument("--smooth-window", type=int, default=3, help="Odd temporal smoothing window after bone correction")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON payload."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON payload."""

    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def frame_identity_joint_map(identity: dict[str, Any]) -> dict[int, np.ndarray]:
    """Return one identity skeleton as joint_id -> xyz."""

    joints = {}
    for joint in identity.get("joints", []):
        joints[int(joint["id"])] = np.array(
            [float(joint["x"]), float(joint["y"]), float(joint["z"])],
            dtype=np.float64,
        )
    return joints


def collect_bone_lengths(payload: dict[str, Any]) -> dict[int, dict[tuple[int, int], list[float]]]:
    """Collect observed bone lengths by identity."""

    lengths: dict[int, dict[tuple[int, int], list[float]]] = {}
    for frame in payload.get("frames", []):
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            joints = frame_identity_joint_map(identity)
            lengths.setdefault(identity_id, {edge: [] for edge in COCO_SKELETON_TREE})
            for edge in COCO_SKELETON_TREE:
                joint_a, joint_b = edge
                if joint_a not in joints or joint_b not in joints:
                    continue
                length = float(np.linalg.norm(joints[joint_b] - joints[joint_a]))
                if np.isfinite(length) and length > 1e-9:
                    lengths[identity_id][edge].append(length)
    return lengths


def build_bone_priors(
    payload: dict[str, Any],
    min_bone_samples: int,
) -> dict[int, dict[tuple[int, int], dict[str, float | int | None]]]:
    """Estimate per-identity median bone-length priors."""

    collected = collect_bone_lengths(payload)
    priors: dict[int, dict[tuple[int, int], dict[str, float | int | None]]] = {}
    for identity_id, by_edge in collected.items():
        priors[identity_id] = {}
        for edge, values in by_edge.items():
            if len(values) < int(min_bone_samples):
                priors[identity_id][edge] = {"count": len(values), "median": None, "std": None}
                continue
            priors[identity_id][edge] = {
                "count": len(values),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
            }
    return priors


def children_by_parent() -> dict[int, list[int]]:
    """Build directed children lookup from the skeleton tree."""

    children: dict[int, list[int]] = {}
    for parent, child in COCO_SKELETON_TREE:
        children.setdefault(parent, []).append(child)
    return children


def soft_correct_skeleton(
    joints: dict[int, np.ndarray],
    priors: dict[tuple[int, int], dict[str, float | int | None]],
    alpha: float,
) -> dict[int, np.ndarray]:
    """Apply a tree-ordered soft bone-length correction."""

    if ROOT_JOINT not in joints:
        return {joint_id: point.copy() for joint_id, point in joints.items()}

    corrected = {joint_id: point.copy() for joint_id, point in joints.items()}
    queue = [ROOT_JOINT]
    children = children_by_parent()
    visited = {ROOT_JOINT}
    alpha = float(np.clip(alpha, 0.0, 1.0))

    while queue:
        parent = queue.pop(0)
        for child in children.get(parent, []):
            if child not in corrected or parent not in corrected:
                continue
            edge = (parent, child)
            target_length = priors.get(edge, {}).get("median")
            vector = corrected[child] - corrected[parent]
            current_length = float(np.linalg.norm(vector))
            if target_length is not None and current_length > 1e-9:
                direction = vector / current_length
                desired = corrected[parent] + direction * float(target_length)
                corrected[child] = corrected[child] + alpha * (desired - corrected[child])
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return corrected


def smooth_identity_sequences(
    frames_by_identity: dict[int, dict[int, dict[int, np.ndarray]]],
    smooth_window: int,
) -> dict[int, dict[int, dict[int, np.ndarray]]]:
    """Smooth corrected joint trajectories over time."""

    window = int(smooth_window)
    if window <= 1:
        return frames_by_identity
    if window % 2 == 0:
        window += 1
    half = window // 2

    smoothed: dict[int, dict[int, dict[int, np.ndarray]]] = {}
    for identity_id, by_frame in frames_by_identity.items():
        frames = sorted(by_frame)
        smoothed[identity_id] = {}
        for frame in frames:
            smoothed[identity_id][frame] = {}
            joint_ids = set(by_frame[frame])
            for nearby in frames:
                if abs(nearby - frame) <= half:
                    joint_ids.update(by_frame[nearby])
            for joint_id in joint_ids:
                samples = [
                    by_frame[nearby][joint_id]
                    for nearby in frames
                    if abs(nearby - frame) <= half and joint_id in by_frame[nearby]
                ]
                if not samples:
                    continue
                smoothed[identity_id][frame][joint_id] = np.mean(samples, axis=0)
    return smoothed


def build_corrected_frames(
    payload: dict[str, Any],
    priors: dict[int, dict[tuple[int, int], dict[str, float | int | None]]],
    alpha: float,
    smooth_window: int,
) -> list[dict[str, Any]]:
    """Build corrected output frames."""

    corrected_by_identity: dict[int, dict[int, dict[int, np.ndarray]]] = {}
    original_joint_meta: dict[tuple[int, int, int], dict[str, Any]] = {}
    for frame in payload.get("frames", []):
        frame_idx = int(frame["frame"])
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            joints = frame_identity_joint_map(identity)
            corrected = soft_correct_skeleton(
                joints=joints,
                priors=priors.get(identity_id, {}),
                alpha=alpha,
            )
            corrected_by_identity.setdefault(identity_id, {})[frame_idx] = corrected
            for joint in identity.get("joints", []):
                original_joint_meta[(frame_idx, identity_id, int(joint["id"]))] = {
                    key: value
                    for key, value in joint.items()
                    if key not in {"x", "y", "z"}
                }

    corrected_by_identity = smooth_identity_sequences(corrected_by_identity, smooth_window=smooth_window)

    frames_out = []
    all_frames = sorted({frame_idx for by_frame in corrected_by_identity.values() for frame_idx in by_frame})
    for frame_idx in all_frames:
        identities_out = []
        for identity_id in sorted(corrected_by_identity):
            joints = corrected_by_identity[identity_id].get(frame_idx)
            if not joints:
                continue
            joints_out = []
            for joint_id in sorted(joints):
                point = joints[joint_id]
                meta = original_joint_meta.get((frame_idx, identity_id, joint_id), {})
                joints_out.append(
                    {
                        **meta,
                        "id": int(joint_id),
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "z": float(point[2]),
                        "bone_prior_source": "soft_bone_length_prior",
                    }
                )
            identities_out.append(
                {
                    "identity_id": int(identity_id),
                    "num_joints": len(joints_out),
                    "joints": joints_out,
                }
            )
        if identities_out:
            frames_out.append({"frame": int(frame_idx), "identities": identities_out})
    return frames_out


def summarize_bone_errors(
    payload: dict[str, Any],
    priors: dict[int, dict[tuple[int, int], dict[str, float | int | None]]],
) -> dict[str, Any]:
    """Summarize normalized bone-length errors against priors."""

    errors = []
    per_identity: dict[int, list[float]] = {}
    for frame in payload.get("frames", []):
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            joints = frame_identity_joint_map(identity)
            for edge, prior in priors.get(identity_id, {}).items():
                target = prior.get("median")
                if target is None or float(target) <= 1e-9:
                    continue
                joint_a, joint_b = edge
                if joint_a not in joints or joint_b not in joints:
                    continue
                length = float(np.linalg.norm(joints[joint_b] - joints[joint_a]))
                error = abs(length - float(target)) / float(target)
                errors.append(error)
                per_identity.setdefault(identity_id, []).append(error)
    return {
        "num_errors": len(errors),
        "mean_normalized_error": float(np.mean(errors)) if errors else None,
        "median_normalized_error": float(np.median(errors)) if errors else None,
        "p90_normalized_error": float(np.percentile(errors, 90)) if errors else None,
        "by_identity": {
            str(identity_id): {
                "count": len(values),
                "mean_normalized_error": float(np.mean(values)) if values else None,
                "median_normalized_error": float(np.median(values)) if values else None,
            }
            for identity_id, values in sorted(per_identity.items())
        },
    }


def serialize_priors(
    priors: dict[int, dict[tuple[int, int], dict[str, float | int | None]]],
) -> dict[str, Any]:
    """Convert tuple-keyed priors into JSON-safe dictionaries."""

    return {
        str(identity_id): {
            f"{edge[0]}-{edge[1]}": values
            for edge, values in sorted(by_edge.items())
        }
        for identity_id, by_edge in sorted(priors.items())
    }


def apply_bone_prior(
    payload: dict[str, Any],
    alpha: float,
    min_bone_samples: int,
    smooth_window: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the full bone prior pipeline."""

    priors = build_bone_priors(payload, min_bone_samples=min_bone_samples)
    corrected_frames = build_corrected_frames(
        payload=payload,
        priors=priors,
        alpha=alpha,
        smooth_window=smooth_window,
    )
    corrected_payload = {
        "metadata": {
            **dict(payload.get("metadata", {})),
            "stage": "bone_length_prior_refinement",
            "bone_prior": {
                "alpha": float(alpha),
                "min_bone_samples": int(min_bone_samples),
                "smooth_window": int(smooth_window),
                "root_joint": ROOT_JOINT,
                "note": "Softly adjusts child joints toward per-identity median bone lengths.",
            },
        },
        "frames": corrected_frames,
    }
    before_errors = summarize_bone_errors(payload, priors)
    after_errors = summarize_bone_errors(corrected_payload, priors)
    metrics = {
        "stage": "bone_length_prior_refinement",
        "alpha": float(alpha),
        "min_bone_samples": int(min_bone_samples),
        "smooth_window": int(smooth_window),
        "bone_priors": serialize_priors(priors),
        "before": before_errors,
        "after": after_errors,
    }
    corrected_payload["summary"] = {
        "before_mean_normalized_bone_error": before_errors.get("mean_normalized_error"),
        "after_mean_normalized_bone_error": after_errors.get("mean_normalized_error"),
        "before_median_normalized_bone_error": before_errors.get("median_normalized_error"),
        "after_median_normalized_bone_error": after_errors.get("median_normalized_error"),
    }
    return corrected_payload, metrics


def main() -> None:
    """Run soft bone-length prior refinement."""

    args = parse_args()
    payload = load_json(args.input_json)
    corrected_payload, metrics = apply_bone_prior(
        payload=payload,
        alpha=float(args.alpha),
        min_bone_samples=int(args.min_bone_samples),
        smooth_window=int(args.smooth_window),
    )
    write_json(args.output_json, corrected_payload)
    write_json(args.metrics_json, metrics)
    print(
        json.dumps(
            {
                "stage": "bone_length_prior_refinement",
                "output_json": str(Path(args.output_json).resolve()),
                "metrics_json": str(Path(args.metrics_json).resolve()),
                "before_mean_normalized_bone_error": metrics["before"].get("mean_normalized_error"),
                "after_mean_normalized_bone_error": metrics["after"].get("mean_normalized_error"),
                "before_median_normalized_bone_error": metrics["before"].get("median_normalized_error"),
                "after_median_normalized_bone_error": metrics["after"].get("median_normalized_error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
