"""Create review-only temporally smoothed triangulation outputs.

This tool is intentionally not a reconstruction/refinement step. It fills short
visual gaps so review videos are easier to inspect while preserving provenance
flags on every filled joint.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill short 3D joint gaps for review videos only.")
    parser.add_argument("--input-json", required=True, help="Path to triangulated_3d.json")
    parser.add_argument("--output-json", required=True, help="Path for review-smoothed JSON")
    parser.add_argument(
        "--max-interp-gap",
        type=int,
        default=5,
        help="Maximum missing-frame gap to linearly interpolate when both sides are available.",
    )
    parser.add_argument(
        "--max-hold-gap",
        type=int,
        default=3,
        help="Maximum missing-frame gap to hold the previous position when no next point is available.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def joint_xyz(joint: dict[str, Any]) -> np.ndarray:
    return np.array([float(joint["x"]), float(joint["y"]), float(joint["z"])], dtype=np.float64)


def make_review_joint(
    joint_id: int,
    point: np.ndarray,
    source: str,
    confidence: float,
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    joint = deepcopy(template) if template else {"id": int(joint_id)}
    joint.update(
        {
            "id": int(joint_id),
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
            "review_source": source,
            "review_imputed": source != "original",
            "review_confidence": float(confidence),
        }
    )
    if source != "original":
        joint["mean_reprojection_error_px"] = None
        joint["views"] = []
        joint["num_views"] = 0
    return joint


def frame_identity_map(frames: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    mapping = {}
    for frame_index, frame in enumerate(frames):
        for identity in frame.get("identities", []):
            mapping[(frame_index, int(identity["identity_id"]))] = identity
    return mapping


def ensure_identity(frame: dict[str, Any], identity_id: int) -> dict[str, Any]:
    for identity in frame.setdefault("identities", []):
        if int(identity["identity_id"]) == int(identity_id):
            return identity
    identity = {"identity_id": int(identity_id), "num_joints": 0, "joints": []}
    frame["identities"].append(identity)
    return identity


def smooth_payload(payload: dict[str, Any], max_interp_gap: int, max_hold_gap: int) -> tuple[dict[str, Any], dict[str, Any]]:
    output = deepcopy(payload)
    frames = output.get("frames", [])
    identity_ids = sorted(
        {
            int(identity["identity_id"])
            for frame in frames
            for identity in frame.get("identities", [])
        }
    )
    joint_ids = sorted(
        {
            int(joint["id"])
            for frame in frames
            for identity in frame.get("identities", [])
            for joint in identity.get("joints", [])
        }
    )

    existing = frame_identity_map(frames)
    originals: dict[tuple[int, int, int], dict[str, Any]] = {}
    for (frame_index, identity_id), identity in existing.items():
        for joint in identity.get("joints", []):
            key = (frame_index, identity_id, int(joint["id"]))
            originals[key] = joint
            joint.setdefault("review_source", "original")
            joint.setdefault("review_imputed", False)
            joint.setdefault("review_confidence", float(joint.get("mean_confidence", 1.0)))

    inserted_interp = 0
    inserted_hold = 0
    skipped_long_gaps = 0

    for identity_id in identity_ids:
        for joint_id in joint_ids:
            present = [
                frame_index
                for frame_index in range(len(frames))
                if (frame_index, identity_id, joint_id) in originals
            ]
            if not present:
                continue

            for left, right in zip(present, present[1:]):
                gap = right - left - 1
                if gap <= 0:
                    continue
                if gap > int(max_interp_gap):
                    skipped_long_gaps += gap
                    continue
                left_joint = originals[(left, identity_id, joint_id)]
                right_joint = originals[(right, identity_id, joint_id)]
                left_point = joint_xyz(left_joint)
                right_point = joint_xyz(right_joint)
                template = left_joint
                for offset in range(1, gap + 1):
                    alpha = offset / float(gap + 1)
                    point = (1.0 - alpha) * left_point + alpha * right_point
                    identity = ensure_identity(frames[left + offset], identity_id)
                    identity.setdefault("joints", []).append(
                        make_review_joint(
                            joint_id=joint_id,
                            point=point,
                            source="interpolated",
                            confidence=min(float(left_joint.get("review_confidence", 1.0)), float(right_joint.get("review_confidence", 1.0))) * 0.6,
                            template=template,
                        )
                    )
                    inserted_interp += 1

            last = present[-1]
            last_joint = originals[(last, identity_id, joint_id)]
            last_point = joint_xyz(last_joint)
            for offset in range(1, int(max_hold_gap) + 1):
                frame_index = last + offset
                if frame_index >= len(frames):
                    break
                if (frame_index, identity_id, joint_id) in originals:
                    break
                identity = ensure_identity(frames[frame_index], identity_id)
                identity.setdefault("joints", []).append(
                    make_review_joint(
                        joint_id=joint_id,
                        point=last_point,
                        source="held",
                        confidence=float(last_joint.get("review_confidence", 1.0)) * max(0.1, 0.45 / offset),
                        template=last_joint,
                    )
                )
                inserted_hold += 1

    for frame in frames:
        for identity in frame.get("identities", []):
            dedup: dict[int, dict[str, Any]] = {}
            for joint in identity.get("joints", []):
                joint_id = int(joint["id"])
                if joint_id not in dedup or not bool(joint.get("review_imputed", False)):
                    dedup[joint_id] = joint
            identity["joints"] = sorted(dedup.values(), key=lambda item: int(item["id"]))
            identity["num_joints"] = len(identity["joints"])
        frame["identities"] = sorted(frame.get("identities", []), key=lambda item: int(item["identity_id"]))

    metadata = output.setdefault("metadata", {})
    metadata["review_smoothing"] = {
        "enabled": True,
        "purpose": "visual_review_only",
        "max_interp_gap": int(max_interp_gap),
        "max_hold_gap": int(max_hold_gap),
    }
    summary = {
        "stage": "review_temporal_smoothing",
        "num_frames": len(frames),
        "identity_ids": identity_ids,
        "joint_ids": joint_ids,
        "inserted_interpolated_joints": inserted_interp,
        "inserted_held_joints": inserted_hold,
        "skipped_long_gap_frames": skipped_long_gaps,
    }
    return output, summary


def main() -> None:
    args = parse_args()
    payload = load_json(args.input_json)
    smoothed, summary = smooth_payload(
        payload,
        max_interp_gap=int(args.max_interp_gap),
        max_hold_gap=int(args.max_hold_gap),
    )
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(smoothed, handle, ensure_ascii=False, indent=2)
    with (output_path.parent / "review_smoothing_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
