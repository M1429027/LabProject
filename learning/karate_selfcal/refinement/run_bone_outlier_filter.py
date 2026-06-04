"""Clamp bone-length outliers after bone-prior refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run_bone_length_prior import (
    COCO_SKELETON_TREE,
    ROOT_JOINT,
    build_bone_priors,
    children_by_parent,
    frame_identity_joint_map,
    smooth_identity_sequences,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Detect and clamp over-long 3D bones that dominate skeleton visualization."
    )
    parser.add_argument("--input-json", required=True, help="Input 3D skeleton JSON")
    parser.add_argument("--output-json", required=True, help="Output filtered 3D skeleton JSON")
    parser.add_argument("--metrics-json", required=True, help="Output filtering metrics JSON")
    parser.add_argument("--max-ratio", type=float, default=1.8, help="Clamp when bone length exceeds median * ratio")
    parser.add_argument("--min-bone-samples", type=int, default=10, help="Minimum samples for a median bone prior")
    parser.add_argument("--smooth-window", type=int, default=3, help="Odd temporal smoothing window after clamping")
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


def clamp_outlier_bones(
    joints: dict[int, np.ndarray],
    priors: dict[tuple[int, int], dict[str, float | int | None]],
    max_ratio: float,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    """Clamp child joints when their parent-child bone is much longer than the prior."""

    corrected = {joint_id: point.copy() for joint_id, point in joints.items()}
    if ROOT_JOINT not in corrected:
        return corrected, []

    events = []
    queue = [ROOT_JOINT]
    visited = {ROOT_JOINT}
    children = children_by_parent()
    for parent, child in COCO_SKELETON_TREE:
        children.setdefault(parent, [])

    while queue:
        parent = queue.pop(0)
        for child in children.get(parent, []):
            if parent not in corrected or child not in corrected:
                continue
            prior = priors.get((parent, child), {})
            target = prior.get("median")
            if target is None or float(target) <= 1e-9:
                continue
            vector = corrected[child] - corrected[parent]
            length = float(np.linalg.norm(vector))
            threshold = float(target) * float(max_ratio)
            if length > threshold and length > 1e-9:
                direction = vector / length
                corrected[child] = corrected[parent] + direction * float(target)
                events.append(
                    {
                        "parent_joint": int(parent),
                        "child_joint": int(child),
                        "original_length": length,
                        "target_length": float(target),
                        "threshold": threshold,
                        "ratio": length / float(target),
                    }
                )
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return corrected, events


def build_filtered_frames(
    payload: dict[str, Any],
    priors: dict[int, dict[tuple[int, int], dict[str, float | int | None]]],
    max_ratio: float,
    smooth_window: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build filtered output frames and event log."""

    corrected_by_identity: dict[int, dict[int, dict[int, np.ndarray]]] = {}
    original_joint_meta: dict[tuple[int, int, int], dict[str, Any]] = {}
    all_events = []

    for frame in payload.get("frames", []):
        frame_idx = int(frame["frame"])
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            joints = frame_identity_joint_map(identity)
            corrected, events = clamp_outlier_bones(
                joints=joints,
                priors=priors.get(identity_id, {}),
                max_ratio=max_ratio,
            )
            corrected_by_identity.setdefault(identity_id, {})[frame_idx] = corrected
            for event in events:
                all_events.append(
                    {
                        "frame": frame_idx,
                        "identity_id": identity_id,
                        **event,
                    }
                )
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
                        "bone_outlier_filter": "clamp_overlong_bones",
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
    return frames_out, all_events


def summarize_long_bones(
    payload: dict[str, Any],
    priors: dict[int, dict[tuple[int, int], dict[str, float | int | None]]],
    max_ratio: float,
) -> dict[str, Any]:
    """Count over-long bones in a payload."""

    ratios = []
    overlong = 0
    total = 0
    for frame in payload.get("frames", []):
        for identity in frame.get("identities", []):
            identity_id = int(identity["identity_id"])
            joints = frame_identity_joint_map(identity)
            for parent, child in COCO_SKELETON_TREE:
                target = priors.get(identity_id, {}).get((parent, child), {}).get("median")
                if target is None or float(target) <= 1e-9:
                    continue
                if parent not in joints or child not in joints:
                    continue
                length = float(np.linalg.norm(joints[child] - joints[parent]))
                ratio = length / float(target)
                ratios.append(ratio)
                total += 1
                if ratio > float(max_ratio):
                    overlong += 1
    return {
        "total_bones_checked": total,
        "overlong_bones": overlong,
        "overlong_ratio": float(overlong / total) if total else None,
        "median_length_ratio": float(np.median(ratios)) if ratios else None,
        "p90_length_ratio": float(np.percentile(ratios, 90)) if ratios else None,
        "p95_length_ratio": float(np.percentile(ratios, 95)) if ratios else None,
        "max_length_ratio": float(np.max(ratios)) if ratios else None,
    }


def main() -> None:
    """Run bone outlier filtering."""

    args = parse_args()
    payload = load_json(args.input_json)
    priors = build_bone_priors(payload, min_bone_samples=int(args.min_bone_samples))
    before = summarize_long_bones(payload, priors=priors, max_ratio=float(args.max_ratio))
    frames_out, events = build_filtered_frames(
        payload=payload,
        priors=priors,
        max_ratio=float(args.max_ratio),
        smooth_window=int(args.smooth_window),
    )
    output_payload = {
        "metadata": {
            **dict(payload.get("metadata", {})),
            "stage": "bone_outlier_filter",
            "bone_outlier_filter": {
                "max_ratio": float(args.max_ratio),
                "min_bone_samples": int(args.min_bone_samples),
                "smooth_window": int(args.smooth_window),
                "note": "Over-long child joints are clamped to the median parent-child bone length.",
            },
        },
        "frames": frames_out,
    }
    after = summarize_long_bones(output_payload, priors=priors, max_ratio=float(args.max_ratio))
    metrics = {
        "stage": "bone_outlier_filter",
        "max_ratio": float(args.max_ratio),
        "num_clamp_events": len(events),
        "before": before,
        "after": after,
        "events": events,
    }
    output_payload["summary"] = {
        "num_clamp_events": len(events),
        "before_overlong_ratio": before.get("overlong_ratio"),
        "after_overlong_ratio": after.get("overlong_ratio"),
        "before_max_length_ratio": before.get("max_length_ratio"),
        "after_max_length_ratio": after.get("max_length_ratio"),
    }
    write_json(args.output_json, output_payload)
    write_json(args.metrics_json, metrics)
    print(
        json.dumps(
            {
                "stage": "bone_outlier_filter",
                "output_json": str(Path(args.output_json).resolve()),
                "metrics_json": str(Path(args.metrics_json).resolve()),
                "num_clamp_events": len(events),
                "before": before,
                "after": after,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
