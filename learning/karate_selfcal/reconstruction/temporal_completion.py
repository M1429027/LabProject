"""Temporal completion and completeness metrics for triangulated 3D skeletons."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


COCO_NUM_JOINTS = 17

COCO_SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Fill short temporal gaps in triangulated 3D skeletons and report completeness."
    )
    parser.add_argument("--input-json", required=True, help="Path to triangulated_3d.json")
    parser.add_argument("--output-json", required=True, help="Path to completed triangulated JSON")
    parser.add_argument("--metrics-json", required=True, help="Path to completion metrics JSON")
    parser.add_argument("--expected-joints", type=int, default=COCO_NUM_JOINTS, help="Expected joints per skeleton")
    parser.add_argument("--max-gap", type=int, default=5, help="Maximum missing-frame gap to interpolate")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Odd moving-average window after interpolation; use 1 to disable",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON payload."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def joint_map(identity: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Index joints by id for one identity payload."""

    return {int(joint["id"]): joint for joint in identity.get("joints", [])}


def collect_frames(payload: dict[str, Any]) -> list[int]:
    """Collect sorted frame ids."""

    return sorted(int(frame["frame"]) for frame in payload.get("frames", []))


def collect_identity_ids(payload: dict[str, Any]) -> list[int]:
    """Collect sorted identity ids present anywhere in the clip."""

    ids = set()
    for frame in payload.get("frames", []):
        for identity in frame.get("identities", []):
            ids.add(int(identity["identity_id"]))
    return sorted(ids)


def build_dense_arrays(
    payload: dict[str, Any],
    frames: list[int],
    identity_ids: list[int],
    expected_joints: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, dict[tuple[int, int], dict[str, Any]]]]:
    """Convert sparse frame payload into dense arrays.

    Returns:
    - coordinates by identity: shape [frames, joints, xyz]
    - original mask by identity: shape [frames, joints]
    - original joint metadata keyed by (frame_index, joint_id)
    """

    frame_to_index = {frame_id: idx for idx, frame_id in enumerate(frames)}
    coordinates = {
        identity_id: np.full((len(frames), expected_joints, 3), np.nan, dtype=np.float64)
        for identity_id in identity_ids
    }
    original_mask = {
        identity_id: np.zeros((len(frames), expected_joints), dtype=bool)
        for identity_id in identity_ids
    }
    metadata: dict[int, dict[tuple[int, int], dict[str, Any]]] = {identity_id: {} for identity_id in identity_ids}

    for frame in payload.get("frames", []):
        frame_idx = frame_to_index[int(frame["frame"])]
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            if identity_id not in coordinates:
                continue
            for joint in identity.get("joints", []):
                joint_id = int(joint["id"])
                if joint_id < 0 or joint_id >= expected_joints:
                    continue
                coordinates[identity_id][frame_idx, joint_id] = [
                    float(joint["x"]),
                    float(joint["y"]),
                    float(joint["z"]),
                ]
                original_mask[identity_id][frame_idx, joint_id] = True
                metadata[identity_id][(frame_idx, joint_id)] = dict(joint)
    return coordinates, original_mask, metadata


def interpolate_short_gaps(values: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill short gaps between known samples for one [frames, xyz] sequence."""

    filled = values.copy()
    filled_mask = np.zeros(values.shape[0], dtype=bool)
    valid = np.all(np.isfinite(values), axis=1)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 2:
        return filled, filled_mask

    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        gap = int(right - left - 1)
        if gap <= 0 or gap > int(max_gap):
            continue
        for offset in range(1, gap + 1):
            alpha = offset / float(gap + 1)
            frame_idx = left + offset
            filled[frame_idx] = (1.0 - alpha) * values[left] + alpha * values[right]
            filled_mask[frame_idx] = True
    return filled, filled_mask


def smooth_sequence(values: np.ndarray, available_mask: np.ndarray, window: int) -> np.ndarray:
    """Smooth finite samples with a centered moving average."""

    window = int(window)
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1

    smoothed = values.copy()
    half = window // 2
    for idx in range(values.shape[0]):
        if not available_mask[idx]:
            continue
        low = max(0, idx - half)
        high = min(values.shape[0], idx + half + 1)
        local_mask = available_mask[low:high]
        if not np.any(local_mask):
            continue
        smoothed[idx] = np.mean(values[low:high][local_mask], axis=0)
    return smoothed


def complete_coordinates(
    coordinates: dict[int, np.ndarray],
    original_mask: dict[int, np.ndarray],
    max_gap: int,
    smooth_window: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Complete and smooth all identities/joints."""

    completed = {}
    completed_mask = {}
    interpolated_mask = {}
    for identity_id, coords in coordinates.items():
        out = coords.copy()
        interp = np.zeros(coords.shape[:2], dtype=bool)
        for joint_id in range(coords.shape[1]):
            filled, joint_interp = interpolate_short_gaps(coords[:, joint_id, :], max_gap=max_gap)
            available = np.all(np.isfinite(filled), axis=1)
            out[:, joint_id, :] = smooth_sequence(
                values=filled,
                available_mask=available,
                window=smooth_window,
            )
            interp[:, joint_id] = joint_interp
        completed[identity_id] = out
        completed_mask[identity_id] = np.all(np.isfinite(out), axis=2)
        interpolated_mask[identity_id] = interp & ~original_mask[identity_id]
    return completed, completed_mask, interpolated_mask


def summarize_bone_lengths(
    coordinates: dict[int, np.ndarray],
    mask: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Compute median/std bone lengths over available skeleton edges."""

    by_identity = {}
    all_lengths = []
    for identity_id, coords in coordinates.items():
        lengths_by_edge = defaultdict(list)
        for frame_idx in range(coords.shape[0]):
            for joint_a, joint_b in COCO_SKELETON_CONNECTIONS:
                if joint_a >= coords.shape[1] or joint_b >= coords.shape[1]:
                    continue
                if not (mask[identity_id][frame_idx, joint_a] and mask[identity_id][frame_idx, joint_b]):
                    continue
                length = float(np.linalg.norm(coords[frame_idx, joint_a] - coords[frame_idx, joint_b]))
                if np.isfinite(length):
                    lengths_by_edge[f"{joint_a}-{joint_b}"].append(length)
                    all_lengths.append(length)
        by_identity[str(identity_id)] = {
            edge: {
                "count": len(values),
                "median": float(np.median(values)) if values else None,
                "std": float(np.std(values)) if values else None,
            }
            for edge, values in sorted(lengths_by_edge.items())
        }
    return {
        "overall_count": len(all_lengths),
        "overall_median": float(np.median(all_lengths)) if all_lengths else None,
        "overall_std": float(np.std(all_lengths)) if all_lengths else None,
        "by_identity": by_identity,
    }


def completeness_metrics(
    frames: list[int],
    identity_ids: list[int],
    original_mask: dict[int, np.ndarray],
    completed_mask: dict[int, np.ndarray],
    interpolated_mask: dict[int, np.ndarray],
    original_coordinates: dict[int, np.ndarray],
    completed_coordinates: dict[int, np.ndarray],
    expected_joints: int,
) -> dict[str, Any]:
    """Build completeness metrics before and after temporal completion."""

    total_expected = len(frames) * max(len(identity_ids), 1) * int(expected_joints)
    original_count = int(sum(np.sum(mask) for mask in original_mask.values()))
    completed_count = int(sum(np.sum(mask) for mask in completed_mask.values()))
    interpolated_count = int(sum(np.sum(mask) for mask in interpolated_mask.values()))

    joint_presence = {}
    for joint_id in range(expected_joints):
        original_joint_count = int(sum(np.sum(mask[:, joint_id]) for mask in original_mask.values()))
        completed_joint_count = int(sum(np.sum(mask[:, joint_id]) for mask in completed_mask.values()))
        joint_presence[str(joint_id)] = {
            "original_ratio": original_joint_count / float(len(frames) * max(len(identity_ids), 1)) if frames else None,
            "completed_ratio": completed_joint_count / float(len(frames) * max(len(identity_ids), 1)) if frames else None,
        }

    identity_metrics = {}
    for identity_id in identity_ids:
        original_per_frame = np.sum(original_mask[identity_id], axis=1)
        completed_per_frame = np.sum(completed_mask[identity_id], axis=1)
        identity_metrics[str(identity_id)] = {
            "original_mean_joints_per_frame": float(np.mean(original_per_frame)) if len(original_per_frame) else None,
            "completed_mean_joints_per_frame": float(np.mean(completed_per_frame)) if len(completed_per_frame) else None,
            "original_complete_frame_ratio": float(np.mean(original_per_frame >= expected_joints)) if len(original_per_frame) else None,
            "completed_complete_frame_ratio": float(np.mean(completed_per_frame >= expected_joints)) if len(completed_per_frame) else None,
        }

    return {
        "num_frames": len(frames),
        "identity_ids": identity_ids,
        "expected_joints": int(expected_joints),
        "total_expected_joints": total_expected,
        "original_observed_joints": original_count,
        "completed_observed_joints": completed_count,
        "interpolated_joints": interpolated_count,
        "original_observed_ratio": original_count / float(total_expected) if total_expected else None,
        "completed_observed_ratio": completed_count / float(total_expected) if total_expected else None,
        "absolute_ratio_gain": (completed_count - original_count) / float(total_expected) if total_expected else None,
        "relative_observed_gain": (
            (completed_count - original_count) / float(original_count)
            if original_count
            else None
        ),
        "joint_presence": joint_presence,
        "by_identity": identity_metrics,
        "bone_lengths_original": summarize_bone_lengths(original_coordinates, original_mask),
        "bone_lengths_completed": summarize_bone_lengths(completed_coordinates, completed_mask),
    }


def build_completed_payload(
    original_payload: dict[str, Any],
    frames: list[int],
    identity_ids: list[int],
    completed_coordinates: dict[int, np.ndarray],
    completed_mask: dict[int, np.ndarray],
    original_mask: dict[int, np.ndarray],
    interpolated_mask: dict[int, np.ndarray],
    original_metadata: dict[int, dict[tuple[int, int], dict[str, Any]]],
) -> dict[str, Any]:
    """Convert completed dense arrays back to triangulation JSON format."""

    output_frames = []
    for frame_idx, frame_id in enumerate(frames):
        identities_out = []
        for identity_id in identity_ids:
            joints = []
            for joint_id in range(completed_coordinates[identity_id].shape[1]):
                if not completed_mask[identity_id][frame_idx, joint_id]:
                    continue
                coords = completed_coordinates[identity_id][frame_idx, joint_id]
                original_joint = original_metadata[identity_id].get((frame_idx, joint_id), {})
                source = "original" if original_mask[identity_id][frame_idx, joint_id] else "interpolated"
                if interpolated_mask[identity_id][frame_idx, joint_id]:
                    source = "interpolated"
                joints.append(
                    {
                        "id": int(joint_id),
                        "x": float(coords[0]),
                        "y": float(coords[1]),
                        "z": float(coords[2]),
                        "source": source,
                        "num_views": original_joint.get("num_views"),
                        "mean_confidence": original_joint.get("mean_confidence"),
                        "mean_reprojection_error_px": original_joint.get("mean_reprojection_error_px"),
                    }
                )
            if joints:
                identities_out.append(
                    {
                        "identity_id": int(identity_id),
                        "num_joints": len(joints),
                        "joints": joints,
                    }
                )
        if identities_out:
            output_frames.append({"frame": int(frame_id), "identities": identities_out})

    metadata = dict(original_payload.get("metadata", {}))
    metadata["stage"] = "triangulation_temporal_completion"
    metadata["completion_note"] = "Short missing gaps are linearly interpolated and then optionally smoothed."
    return {
        "metadata": metadata,
        "frames": output_frames,
    }


def complete_payload(
    payload: dict[str, Any],
    expected_joints: int,
    max_gap: int,
    smooth_window: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Complete one triangulation payload and return output plus metrics."""

    frames = collect_frames(payload)
    identity_ids = collect_identity_ids(payload)
    coordinates, original_mask, original_metadata = build_dense_arrays(
        payload=payload,
        frames=frames,
        identity_ids=identity_ids,
        expected_joints=expected_joints,
    )
    completed_coordinates, completed_mask, interpolated_mask = complete_coordinates(
        coordinates=coordinates,
        original_mask=original_mask,
        max_gap=max_gap,
        smooth_window=smooth_window,
    )
    metrics = completeness_metrics(
        frames=frames,
        identity_ids=identity_ids,
        original_mask=original_mask,
        completed_mask=completed_mask,
        interpolated_mask=interpolated_mask,
        original_coordinates=coordinates,
        completed_coordinates=completed_coordinates,
        expected_joints=expected_joints,
    )
    metrics["max_gap"] = int(max_gap)
    metrics["smooth_window"] = int(smooth_window)
    completed_payload = build_completed_payload(
        original_payload=payload,
        frames=frames,
        identity_ids=identity_ids,
        completed_coordinates=completed_coordinates,
        completed_mask=completed_mask,
        original_mask=original_mask,
        interpolated_mask=interpolated_mask,
        original_metadata=original_metadata,
    )
    completed_payload["summary"] = metrics
    return completed_payload, metrics


def main() -> None:
    """Run temporal completion from the CLI."""

    args = parse_args()
    payload = load_json(args.input_json)
    completed_payload, metrics = complete_payload(
        payload=payload,
        expected_joints=int(args.expected_joints),
        max_gap=int(args.max_gap),
        smooth_window=int(args.smooth_window),
    )

    output_path = Path(args.output_json).resolve()
    metrics_path = Path(args.metrics_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(completed_payload, handle, ensure_ascii=False, indent=2)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
